"""P1: on-policy 数据循环 —— 用当前策略的 rollout 微调世界模型。

假设(来自第 6 轮): H=32 的可用性由 `NSE32` 决定; 而当前 WM 只在固定的探索数据上训练,
与策略实际分布有偏差。把 WM 在当前策略的 rollout 上继续训练, 应当降低 on-policy 分布上的
NSE32, 从而延长可用视野。

用法:
  python exp/wm_onpolicy.py --ckpt exp/runs/v3_best.pt \
      --policy exp/runs/p0_clean.pt --steps 20000 --online-frac 0.5
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"exp"))
from smini import motion, spec
from smini.env import EnvCfg, TrackingEnv
from smini.policy import Policy
import wm_train
from wm_train import make_wm, load_shards, build_valid
from collect import to71

XML = str(ROOT.parent/"assets"/"mujoco_menagerie"/"unitree_g1"/"scene_mjx.xml")
MDIR = ROOT/"assets"/"motions"
DATA = ROOT/"exp"/"data_v2"
wm_train.DATA = DATA          # 否则 load_shards 会去读 v1 的 33 维数据
DQ = np.asarray(spec.DEFAULT_Q)
COLLECT_CFG = EnvCfg(obs_noise=True, reset_noise=True, domain_randomization=True, action_noise=0.0)
CLEAN = EnvCfg(obs_noise=False, reset_noise=False, domain_randomization=False, action_noise=0.0)


@torch.no_grad()
def rollout_policy(lib, pol, clips, n_envs_steps, rng, noise=0.1, dev="cuda"):
    """用策略在 clips 上 rollout, 返回 transitions (S71, A, X34, E)。"""
    S, A, X, E = [], [], [], []
    got = 0
    while got < n_envs_steps:
        ci = int(rng.choice(clips)); mo = lib[ci]
        if len(mo) < 60: continue
        f0 = int(rng.integers(0, max(len(mo) - 60, 1)))
        env = TrackingEnv(XML, lib, COLLECT_CFG, seed=int(rng.integers(1_000_000)))
        obs = env.reset(motion_id=ci, frame_idx=f0)
        hist = np.repeat(obs[None], spec.HISTORY, 0)
        for t in range(100):
            fi = min(int(env.frame_idx), len(mo) - 12)
            g = torch.as_tensor(mo.goal_window(fi), dtype=torch.float32).to(dev)[None]
            oh = torch.as_tensor(hist[None].astype(np.float32)).to(dev)
            a = pol(g, oh)[0].cpu().numpy()
            a_exec = np.clip(a + noise * rng.standard_normal(spec.ACTION_DIM), -50, 50)
            s71 = to71(np.stack([env.state()]), DQ)[0]
            xi = np.asarray(env.xi, np.float64)
            obs, r, te, tr, _ = env.step(a_exec)
            hist = np.roll(hist, -1, 0); hist[-1] = obs
            S.append(s71); A.append(a_exec); X.append(xi)
            E.append(1 if (te or tr) else 0)
            got += 1
            if te or tr or got >= n_envs_steps: break
        # 关键: 一个 episode 结束时(即使是步数用完而非终止), 必须标记边界,
        # 否则下一条 episode 的首帧会被当成上一条的后续 -> 产生"瞬移"假窗口
        if E and E[-1] == 0:
            E[-1] = 1
    return (np.stack(S).astype(np.float32), np.stack(A).astype(np.float32),
            np.stack(X).astype(np.float32), np.array(E, np.uint8))


def make_eval_windows(S, A, X, E, k=32, nwin=512, seed=0):
    rng = np.random.default_rng(seed)
    cs = np.concatenate([[0], np.cumsum(E.astype(np.int64))])
    idx = np.arange(0, len(E) - k)
    if len(idx) == 0: raise SystemExit("eval data too short")
    ok = (cs[idx + k] - cs[idx]) == 0
    cand = idx[ok]
    if len(cand) == 0: raise SystemExit("no valid eval windows")
    return rng.choice(cand, size=min(nwin, len(cand)), replace=False)


@torch.no_grad()
def eval_nse(wm, S, A, X, starts, k=32, bs=256):
    wm.eval(); acc = np.zeros(k); cnt = 0
    for i in range(0, len(starts), bs):
        idb = starts[i:i+bs]
        s = torch.as_tensor(S[idb]).cuda(); xi = torch.as_tensor(X[idb]).cuda()
        p = s; row = np.zeros((len(idb), k))
        for t in range(k):
            p = wm.step(p, torch.as_tensor(A[idb+t]).cuda(), xi)
            row[:, t] = wm.spec.nse(p, torch.as_tensor(S[idb+t+1]).cuda()).cpu().numpy()
        acc += row.sum(0); cnt += len(idb)
    wm.train()
    return acc/cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT/"exp/runs/v3_best.pt"))
    ap.add_argument("--policy", default=str(ROOT/"exp/runs/p0_clean.pt"))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--online-frac", type=float, default=0.5)
    ap.add_argument("--online-size", type=int, default=200_000)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--out", default=str(ROOT/"exp/runs/wm_onpolicy.pt"))
    a = ap.parse_args()

    torch.set_num_threads(2); torch.manual_seed(0); rng = np.random.default_rng(0)
    wm = make_wm(34).to("cuda")
    sd = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    wm.load_state_dict(sd["wm"] if "wm" in sd else sd)
    print(f"WM loaded step={sd.get('step')} nse8={sd.get('nse8')}", flush=True)

    pol = Policy(obs_dim=93, goal_dim=730, action_dim=29, hist=10).cuda()
    ck = torch.load(a.policy, map_location="cuda", weights_only=False)
    pol.load_state_dict(ck["policy"]); pol.eval()
    print(f"policy loaded ({a.policy})", flush=True)

    meta = json.load(open(DATA/"clips.json"))
    train_clips = [c for c in meta["train"]]
    lib_all = motion.MotionLibrary([motion.load_motion(str(MDIR/f"{c}.npz")) for c in
                                    sorted(p.stem for p in MDIR.glob("*.npz"))])
    names = [m.name for m in lib_all.motions]
    hold_names = meta["holdout"]
    hold_ids = [names.index(n) for n in hold_names if n in names]
    train_ids = [i for i in range(len(names)) if i not in hold_ids]
    print(f"holdout clips {hold_names} ids={hold_ids}", flush=True)

    # ---------- held-out on-policy 评测集 (用 held-out clip, 策略 rollout) ----------
    print("building held-out on-policy eval set ...", flush=True)
    Sh, Ah, Xh, Eh = rollout_policy(lib_all, pol, hold_ids, 30_000, rng)
    ev = make_eval_windows(Sh, Ah, Xh, Eh, k=32, nwin=512)
    base = eval_nse(wm, Sh, Ah, Xh, ev, k=32)
    print(f"[BASE] held-out on-policy NSE1={base[0]:.4f} NSE8={base[7]:.4f} NSE32={base[31]:.4f}", flush=True)

    # ---------- 固定数据集 (原探索数据) ----------
    Sf, Af, Xf, Ef, allvf = load_shards("shard_")
    validf = build_valid(Ef, allvf, a.k)
    vf = torch.as_tensor(np.nonzero(validf)[0], device="cuda")
    Sf_t = torch.as_tensor(Sf, device="cuda"); Af_t = torch.as_tensor(Af, device="cuda")
    Xf_t = torch.as_tensor(Xf, device="cuda")
    print(f"fixed data {len(Sf):,} (valid {len(vf):,})", flush=True)

    # ---------- 在线数据 (策略 rollout) ----------
    print(f"collecting {a.online_size:,} on-policy transitions ...", flush=True)
    t0 = time.time()
    So, Ao, Xo, Eo = rollout_policy(lib_all, pol, train_ids, a.online_size, rng)
    vo = torch.as_tensor(np.nonzero(build_valid(Eo, [(0, len(Eo), np.concatenate([[0], np.cumsum(Eo.astype(np.int64))]))], a.k))[0], device="cuda")
    So_t = torch.as_tensor(So, device="cuda"); Ao_t = torch.as_tensor(Ao, device="cuda")
    Xo_t = torch.as_tensor(Xo, device="cuda")
    print(f"online data {len(So):,} (valid {len(vo):,}) in {time.time()-t0:.0f}s", flush=True)

    opt = torch.optim.Adam(wm.parameters(), lr=a.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    n_online = int(a.batch * a.online_frac)
    best = base[31]
    log = []
    t0 = time.time()
    for step in range(1, a.steps+1):
        # 混合采样: online + fixed
        io = vo[torch.randint(0, len(vo), (n_online,), device="cuda")]
        ixf = vf[torch.randint(0, len(vf), (a.batch - n_online,), device="cuda")]
        # 按步 gather 成 (B, k+1, 71) 状态窗口与 (B, k, 29) 动作
        So_w = torch.stack([So_t[io + j] for j in range(a.k + 1)], 1)
        Sf_w = torch.stack([Sf_t[ixf + j] for j in range(a.k + 1)], 1)
        Ao_w = torch.stack([Ao_t[io + j] for j in range(a.k)], 1)
        Af_w = torch.stack([Af_t[ixf + j] for j in range(a.k)], 1)
        S_b = torch.cat([So_w, Sf_w], 0)
        A_b = torch.cat([Ao_w, Af_w], 0)
        X_b = torch.cat([Xo_t[io], Xf_t[ixf]], 0)
        loss = 0.0
        with torch.autocast("cuda", dtype=torch.float16):
            p = S_b[:, 0]          # (B,71) 起始状态
            for j in range(a.k):
                p = wm.step(p, A_b[:, j], X_b)
                e = wm.spec.error_vector(p, S_b[:, j+1])
                loss = loss + (e*e).mean()
        loss = loss / a.k
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        if step % 500 == 0:
            print(f"  [{step:5d}] loss {float(loss.detach()):.4f} "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            r = eval_nse(wm, Sh, Ah, Xh, ev, k=32)
            print(f"[EVAL {step}] on-policy NSE1={r[0]:.4f} NSE8={r[7]:.4f} NSE32={r[31]:.4f}", flush=True)
            log.append({"step": step, "nse": r.tolist()})
            if r[31] < best:
                best = r[31]
                torch.save({"wm": wm.state_dict(), "step": step, "nse32": float(r[31])}, a.out)
    print(f"\nDONE best on-policy NSE32 = {best:.4f} (baseline {base[31]:.4f})", flush=True)
    json.dump({"base": base.tolist(), "log": log, "online_size": int(len(So)),
               "fixed_size": int(len(Sf))}, open(str(a.out).replace(".pt", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
