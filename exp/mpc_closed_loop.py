"""阶段三: 闭环滚动规划 (MPC) —— 围绕教师动作做短视野规划, 每步重规划执行首动作。

与预实验的区别: 这里是**闭环**——每一步都用真值仿真器的新状态重新规划,
并与教师的闭环策略在同一整回合评测下比较。

用法:
  python exp/mpc_closed_loop.py --H 8 --bound 0.3 --opt-steps 25 --every 1 --seeds 1
  python exp/mpc_closed_loop.py --mode perturb   # 对照臂: 每步随机扰动
  python exp/mpc_closed_loop.py --mode teacher   # 对照臂: 纯教师
"""
import sys, json, argparse, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts")); sys.path.insert(0, str(ROOT/"exp"))
import evalkit as EK
from eval_pretrained import Pol, goal680, obs930
from smini import motion, spec
from smini.env import TrackingEnv
from wm_train import make_wm
import diff_fk as D
from diff_fk import FKTable
from mpc_test8 import build_ref

XML, CLEAN, DEV = EK.XML, EK.CLEAN, "cuda"
STEPS = 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="exp/runs/D_wm20k.pt")
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--bound", type=float, default=0.3)
    ap.add_argument("--opt-steps", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--every", type=int, default=1, help="每 N 步重规划一次")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--mode", default="plan", choices=["plan", "teacher", "perturb"])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--clips", default="", help="逗号分隔的 clip 名, 空=全部 6 条")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="规划目标里惩罚首动作与上一拍动作的差 (抑制抖动)")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="执行前对动作做指数滑动平均: a = ema*a_plan + (1-ema)*a_prev")
    ap.add_argument("--out", default="exp/runs/mpc_closed.json")
    a = ap.parse_args()
    torch.set_num_threads(4)

    tck = torch.load(ROOT/"runs/pretrained/policy.pt", map_location=DEV, weights_only=False)
    net = Pol(); net.load_state_dict(tck["policy"], strict=True); net.to(DEV).eval()
    wm = make_wm(34).to(DEV)
    sd = torch.load(ROOT/a.wm, map_location=DEV, weights_only=False)
    wm.load_state_dict(sd["wm"] if "wm" in sd else sd); wm.eval()
    fk = FKTable(XML, spec.TRACKED_BODIES, spec.JOINT_NAMES, device=DEV)
    xi = torch.as_tensor(np.concatenate([[1., 1.], np.zeros(29), np.zeros(3)]),
                         dtype=torch.float32).to(DEV)[None]

    lib = EK.load_lib(ROOT/"assets/motions_trim")
    want = [c.strip() for c in a.clips.split(",") if c.strip()] or list(EK.EVAL_CLIPS)
    clips = [i for i, m in enumerate(lib.motions) if m.name in want]
    clips = sorted(clips, key=lambda i: want.index(lib.motions[i].name))
    n = len(clips)
    print(f"[{a.mode}] H={a.H} bound={a.bound} K={a.opt_steps} every={a.every} "
          f"clips={[lib.motions[i].name for i in clips]}", flush=True)

    all_scores, all_surv = [], []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        envs = [TrackingEnv(XML, lib, CLEAN, seed=3) for _ in range(n)]
        obs = [e.reset(motion_id=ci, frame_idx=0) for e, ci in zip(envs, clips)]
        hist = [np.repeat(o[None], spec.HISTORY, 0) for o in obs]
        rets = np.zeros(n); surv = np.zeros(n, dtype=int); alive = np.ones(n, bool)
        rew_hist = np.zeros((a.steps, n), dtype=np.float32)
        raw = np.zeros((n, a.H, 29), dtype=np.float32)
        prev_a = np.zeros((n, 29), dtype=np.float32)
        prev_a0 = np.zeros((n, 29), dtype=np.float32)
        t0 = time.time()
        for step in range(a.steps):
            # --- 教师中心动作 + goal/ref ---
            g680, refs, states = [], [], []
            for i, (e, ci) in enumerate(zip(envs, clips)):
                mo = lib[ci]
                fi = min(int(e.frame_idx), len(mo) - 1)
                g680.append(goal680(mo.qpos, mo.qvel, fi))
                refs.append(build_ref(mo, fi + 1, a.H, device="cpu"))
                q, v = e._data.qpos, e._data.qvel
                states.append(np.concatenate([q[0:3], q[3:7], q[7:36], v[0:3], v[3:6], v[6:35]]))
            with torch.no_grad():
                oh = torch.as_tensor(np.stack([obs930(h) for h in hist]).astype(np.float32)).to(DEV)
                g_t = torch.as_tensor(np.stack(g680).astype(np.float32)).to(DEV)
                a_tc = net(g_t, oh).cpu().numpy()

            if a.mode == "teacher":
                act = a_tc
            elif a.mode == "perturb":
                act = a_tc + rng.uniform(-a.bound, a.bound, a_tc.shape)
            else:
                if step % a.every == 0:
                    S0 = torch.as_tensor(np.stack(states).astype(np.float32)).to(DEV)
                    refb = {k: torch.as_tensor(np.stack([r[k] for r in refs]).astype(np.float32)).to(DEV)
                            for k in refs[0]}
                    # 目标: 未来 H 步的 goal (从当前帧起每步一个 680 窗口)
                    gl = []
                    for i, (e, ci) in enumerate(zip(envs, clips)):
                        mo = lib[ci]; f0 = min(int(e.frame_idx), len(mo) - a.H - 2)
                        gl.append(np.stack([goal680(mo.qpos, mo.qvel, f0 + t) for t in range(a.H)]))
                    gseq = torch.as_tensor(np.stack(gl).astype(np.float32)).to(DEV)
                    # 锚点: 当前教师动作复制 H 份（每步重规划、只执行首动作，后段锚点影响很小）
                    a_tc_t = torch.as_tensor(a_tc).to(DEV).unsqueeze(1).repeat(1, a.H, 1)
                    pv0 = torch.as_tensor(prev_a0).to(DEV)
                    r0 = torch.tensor(raw, device=DEV, requires_grad=True)
                    opt = torch.optim.Adam([r0], lr=a.lr)
                    xi_b = xi.expand(n, -1)
                    for _ in range(a.opt_steps):
                        acts = a_tc_t + a.bound * torch.tanh(r0)
                        s = S0; pv = torch.as_tensor(prev_a).to(DEV); tot = 0.0
                        for t in range(a.H):
                            s = wm.step(s, acts[:, t], xi_b)
                            r = D.reward8(s, acts[:, t], pv, {k: v[:, t] for k, v in refb.items()}, fk)
                            tot = tot + r.sum()
                            pv = acts[:, t]
                        if a.smooth > 0:
                            tot = tot - a.smooth * ((acts[:, 0] - pv0) ** 2).sum()
                        opt.zero_grad(); (-tot).backward(); opt.step()
                    with torch.no_grad():
                        sol = (a_tc_t + a.bound * torch.tanh(r0)).cpu().numpy()
                    # 热启动: 平移一拍
                    k = a.every
                    raw = np.concatenate([sol[:, k:], np.repeat(sol[:, -1:], k, axis=1)],
                                         axis=1)[:, :a.H].astype(np.float32)
                act = raw[:, step % a.every].copy()
            # --- EMA 平滑后再执行 ---
            if a.ema > 0 and a.mode == "plan":
                act = a.ema * act + (1 - a.ema) * prev_a0
            # --- 执行 ---
            new_obs = []
            for i, (e, ci) in enumerate(zip(envs, clips)):
                if not alive[i]:
                    new_obs.append(obs[i]); continue
                o, r, te, tr, _ = e.step(act[i])
                rets[i] += float(r); surv[i] += 1
                rew_hist[step, i] = float(r)
                hist[i] = np.roll(hist[i], -1, 0); hist[i][-1] = o
                new_obs.append(o)
                if te or tr: alive[i] = False
            obs = new_obs
            prev_a = act; prev_a0 = act
            if step % 50 == 0:
                print(f"  s{seed} step {step} alive={alive.sum()}/{n} "
                      f"ret={rets.sum():.2f} ({time.time()-t0:.0f}s)", flush=True)
            if not alive.any(): break
        score = np.array([rets[i] / (0.1 * a.steps) for i in range(n)])
        all_scores.append(score); all_surv.append(surv.copy())
        print(f"  seed {seed}: score={score.mean():.4f} "
              f"存活均值 {surv.mean():.0f}/{a.steps} ({time.time()-t0:.0f}s)", flush=True)

    sc = np.array(all_scores); sv = np.array(all_surv)
    print(f"\n===== [{a.mode}] 整回合结果 ({a.seeds} 种子) =====")
    for i, ci in enumerate(clips):
        print(f"  {lib.motions[ci].name:22s} {sc[:, i].mean():.4f}  存活 {sv[:, i].mean():.0f}")
    print(f"  {'均值':22s} {sc.mean():.4f}  存活 {sv.mean():.0f}")
    json.dump({"mode": a.mode, "H": a.H, "bound": a.bound, "opt_steps": a.opt_steps,
               "every": a.every, "clips": [lib.motions[i].name for i in clips],
               "scores": sc.tolist(), "surv": sv.tolist(),
               "mean": float(sc.mean()), "surv_mean": float(sv.mean()),
               "rew_hist": rew_hist.tolist()},
              open(ROOT/a.out, "w"), indent=1)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
