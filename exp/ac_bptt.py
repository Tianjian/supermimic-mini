"""完整 actor-critic: 在 WM71 上做短视野 BPTT, 奖励用完整 8 项 (可微 FK)。

链路:
  Policy(goal730, obs_hist 10x93) -> a_t
  WM71: s_{t+1} = f(s_t, a_t, xi)
  obs_{t+1} = obs71(s_{t+1})            # 用 FK 算 angvel_b / grav_b
  r_{t+1} = reward8(s_{t+1}, a_t, a_{t-1}, ref_t)   # 与 env 完全一致 (误差 9e-9)
  J = sum gamma^t c_t r + gamma^H c_H V(s_H)
  actor 用 -J 反传; critic 用 imagined GAE 拟合

先 BC 热启动到 PD 跟踪动作 (落在 WM 分布内), 再 BPTT; 评测用真值仿真器的 raw 回报。
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"exp"))
from smini import motion, spec
from smini.env import EnvCfg, TrackingEnv
from smini.policy import Policy, Critic
from wm_train import make_wm
import diff_fk as D
from diff_fk import FKTable
from mpc_test8 import build_ref

XML = str(ROOT.parent/"assets"/"mujoco_menagerie"/"unitree_g1"/"scene_mjx.xml")
MDIR = ROOT/"assets"/"motions"
DATA = ROOT/"exp"/"data_v2"
DQ = torch.as_tensor(np.asarray(spec.DEFAULT_Q), dtype=torch.float32)
D.set_action_scale(spec.ACTION_SCALE)
CLEAN = EnvCfg(obs_noise=False, reset_noise=False, domain_randomization=False, action_noise=0.0)


def obs71(s71, prev_a):
    """71 维状态 -> 93 维观测 (与 spec.pack_obs 同构)。"""
    qj = s71[:, 7:36]; qd = s71[:, 42:71]
    w_body = s71[:, 39:42]
    R = D.quat_to_mat(s71[:, 3:7])
    grav_b = -R[..., 2, :]
    return torch.cat([qj - DQ.to(s71.device), qd, w_body, grav_b, prev_a], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT/"exp/runs/v2_best.pt"))
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--B", type=int, default=64)          # 每轮想象样本数
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--bc-steps", type=int, default=1500)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default=str(ROOT/"exp/runs/ac_bptt.json"))
    ap.add_argument("--init", default=None, help="从 BC 热启动 checkpoint 初始化策略")
    ap.add_argument("--skip-bc", action="store_true", help="有 --init 时跳过 BC")
    ap.add_argument("--kl-target", type=float, default=0.02,
                    help="P2 信赖域: 旧/新策略动作差的平方均值目标 (0=关闭)")
    ap.add_argument("--beta", type=float, default=1.0, help="KL 惩罚初值 (自适应缩放)")
    ap.add_argument("--kl-sigma", type=float, default=1.0, help="高斯近似的 action std")
    a = ap.parse_args()

    torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
    wm = make_wm(34).to("cuda"); sd = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    wm.load_state_dict(sd["wm"] if "wm" in sd else sd); wm.eval()
    for p in wm.parameters(): p.requires_grad_(False)
    print(f"WM step={sd.get('step')} nse8={sd.get('nse8')}", flush=True)

    fk = FKTable(XML, spec.TRACKED_BODIES, spec.JOINT_NAMES, device="cuda")
    meta = json.load(open(DATA/"clips.json"))
    lib = motion.MotionLibrary([motion.load_motion(str(MDIR/f"{c}.npz")) for c in meta["train"]])
    rng = np.random.default_rng(0)

    pol = Policy(obs_dim=93, goal_dim=730, action_dim=29, hist=10).cuda()
    if a.init:
        ck0 = torch.load(a.init, map_location="cuda", weights_only=False)
        pol.load_state_dict(ck0["policy"]); print(f"pol init from {a.init} (score {ck0.get('score')})", flush=True)
    crit = Critic(state_dim=71, goal_dim=730, xi_dim=34).cuda()
    opt_p = torch.optim.Adam(pol.parameters(), lr=a.lr)
    opt_v = torch.optim.Adam(crit.parameters(), lr=a.lr)
    xi = torch.as_tensor(np.concatenate([[1., 1.], np.zeros(29), np.zeros(3)]),
                         dtype=torch.float32).cuda()[None]

    def make_env(mid, f):
        e = TrackingEnv(XML, lib, CLEAN, seed=1)
        e.reset(motion_id=mid, frame_idx=f)
        return e

    # ---------------- BC 热启动 ----------------
    if a.skip_bc and a.init:
        a.bc_steps = 0
    print(f"BC warm start ({a.bc_steps} steps) ...", flush=True)
    t0 = time.time()
    for it in range(a.bc_steps):
        mid = int(rng.integers(0, len(lib))); mo = lib[mid]
        if len(mo) < 30: continue
        f = int(rng.integers(0, len(mo) - 25))
        e = make_env(mid, f)
        hist = []
        for _ in range(10):
            act = e._goal[15:44] / spec.ACTION_SCALE
            hist.append(e._obs.copy())
            e.step(act)
        goal = torch.as_tensor(mo.goal_window(int(e.frame_idx)), dtype=torch.float32).cuda()[None]
        oh = torch.as_tensor(np.stack(hist)[None].astype(np.float32)).cuda()
        tgt = torch.as_tensor((mo.qpos[min(int(e.frame_idx), len(mo)-1)][7:36] - np.asarray(spec.DEFAULT_Q))
                              / np.asarray(spec.ACTION_SCALE), dtype=torch.float32).cuda()[None]
        loss = ((pol(goal, oh) - tgt) ** 2).mean()
        opt_p.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt_p.step()
        if (it+1) % 500 == 0:
            print(f"  BC {it+1} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- 评测函数 (真值仿真器, raw policy) ----------------
    @torch.no_grad()
    def evaluate(n=8, steps=200):
        tot = 0.0; cnt = 0
        for _ in range(n):
            mid = int(rng.integers(0, len(lib))); mo = lib[mid]
            if len(mo) < steps + 15: continue
            f = int(rng.integers(0, len(mo) - steps - 12))
            e = make_env(mid, f)
            hist = [e._obs.copy() for _ in range(10)]
            R = 0.0
            for t in range(steps):
                gi = min(int(e.frame_idx), len(mo) - 12)
                goal = torch.as_tensor(mo.goal_window(gi), dtype=torch.float32).cuda()[None]
                oh = torch.as_tensor(np.stack(hist)[None].astype(np.float32)).cuda()
                act = pol(goal, oh)[0].cpu().numpy()
                _, r, te, tr, _ = e.step(act)
                R += float(r); cnt += 1
                hist = hist[1:] + [e._obs.copy()]
                if te or tr: break
            tot += R
        return tot / max(cnt, 1) / n   # 每步平均

    base = evaluate()
    print(f"BC 后 raw 每步回报: {base:.5f}", flush=True)

    # ---------------- BPTT ----------------
    # 标准化评测: 5 条 clip, frame 0, 500 步, ret/(0.1*steps) — 与 eval_pretrained 同口径
    @torch.no_grad()
    def eval_std(n_clips=5, steps=500):
        env = TrackingEnv(XML, lib, CLEAN, seed=0); rets=[]
        for ci in range(n_clips):
            obs = env.reset(motion_id=ci, frame_idx=0)
            hist = np.repeat(obs[None], 10, 0); ret=0.0; cnt=0
            for t in range(steps):
                fi = min(int(env.frame_idx), len(lib[ci])-1)
                g = torch.as_tensor(lib[ci].goal_window(fi), dtype=torch.float32).cuda()[None]
                oh = torch.as_tensor(hist[None].astype(np.float32)).cuda()
                act = pol(g, oh)[0].cpu().numpy()
                obs, r, te, tr, _ = env.step(act)
                hist = np.roll(hist,-1,0); hist[-1]=obs
                ret += r; cnt += 1
                if te or tr: break
            rets.append(ret/(0.1*max(cnt-1,1)))
        return float(np.mean(rets)), rets
    std0, _ = eval_std()
    print(f"标准化评测(5 clips) 起手: {std0:.4f}", flush=True)

    print(f"BPTT ... (KL/信赖域: target={a.kl_target}, sigma={a.kl_sigma})", flush=True)
    hist_log = []
    best_std = std0

    for it in range(a.iters):
        starts = []
        for _ in range(a.B):
            mid = int(rng.integers(0, len(lib))); mo = lib[mid]
            if len(mo) < a.H + 40: continue
            f = int(rng.integers(0, len(mo) - a.H - 15))
            starts.append((mid, f))
        if not starts: continue
        Nb = len(starts)
        s0, oh0, goals, refs = [], [], [], []
        for mid, f0 in starts:
            mo = lib[mid]
            e = make_env(mid, f0)
            q = e._data.qpos; v = e._data.qvel
            s0.append(np.concatenate([q[0:3], q[3:7], q[7:36], v[0:3], v[3:6], v[6:35]]))
            goals.append(mo.goal_window(f0 + 1))
            r = build_ref(mo, f0 + 1, a.H, device="cpu")
            refs.append({k: vv.numpy() for k, vv in r.items()})
            hist = []
            for _ in range(10):
                hist.append(e._obs.copy())
                e.step(e._goal[15:44] / spec.ACTION_SCALE)
            oh0.append(np.stack(hist))
        s = torch.as_tensor(np.stack(s0).astype(np.float32)).cuda()
        oh = torch.as_tensor(np.stack(oh0).astype(np.float32)).cuda()
        oh_start = oh.clone()          # 信赖域回溯用的初始 obs 历史
        g0 = torch.as_tensor(np.stack(goals).astype(np.float32)).cuda()
        refb = {k: torch.as_tensor(np.stack([r[k] for r in refs]).astype(np.float32)).cuda()
                for k in refs[0]}
        xi_b = xi.expand(Nb, -1).contiguous()

        # 逐目标窗口: step t 用 frames(f0+1+t .. f0+1+t+9)
        # goal_window 返回的是 t+1..t+10, 这里直接用同一窗口随 t 平移
        gl = []
        for mid, f0 in starts:
            mo = lib[mid]
            gl.append(np.stack([mo.goal_window(f0 + 1 + t) for t in range(a.H)]))
        gseq = torch.as_tensor(np.stack(gl).astype(np.float32)).cuda()   # (N,H,730)

        prev = torch.zeros(Nb, 29, device="cuda")
        Rs = []; Vs = []
        for t in range(a.H):
            act = pol(gseq[:, t], oh)
            s = wm.step(s, act, xi_b)
            r = D.reward8(s, act, prev, {k: v[:, t] for k, v in refb.items()}, fk)
            Rs.append(r)
            Vs.append(crit(s.detach(), gseq[:, t], xi_b).squeeze(-1))
            oh = torch.cat([oh[:, 1:], obs71(s, act)[:, None, :]], 1)
            prev = act
        R = torch.stack(Rs, 1)          # (N,H) 可微
        V = torch.stack(Vs, 1)          # (N,H) 可微
        with torch.no_grad():
            v_last = crit(s.detach(), gseq[:, -1], xi_b).squeeze(-1)
        # actor: 可微回报 + 分离的 bootstrap
        disc = 1.0; J = torch.zeros(Nb, device="cuda")
        for t in range(a.H):
            J = J + disc * R[:, t]
            disc *= a.gamma
        J = J + disc * v_last.detach()
        # critic 目标: 同一批回报, 但全部 detach
        with torch.no_grad():
            Rd = R.detach(); retd = torch.zeros_like(Rd); acc = v_last
            for t in reversed(range(a.H)):
                acc = Rd[:, t] + a.gamma * acc
                retd[:, t] = acc
        loss_p = -J.mean()
        loss_v = ((V - retd) ** 2).mean()
        opt_p.zero_grad(); loss_p.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        # P2: 更新前记录参考动作, 更新后做信赖域回溯
        if a.kl_target > 0:
            with torch.no_grad():
                a_ref = pol(g0, oh_start).detach()
            old_par = [q.detach().clone() for q in pol.parameters()]
        opt_p.step()
        dev_val = 0.0
        if a.kl_target > 0:
            with torch.no_grad():
                dev_val = float(((pol(g0, oh_start) - a_ref) ** 2).mean())
            n_bt = 0
            while dev_val > 1.5 * a.kl_target and n_bt < 6:
                alpha = (a.kl_target / max(dev_val, 1e-12)) ** 0.5
                with torch.no_grad():
                    for q, q0 in zip(pol.parameters(), old_par):
                        q.copy_(q0 + alpha * (q - q0))
                    dev_val = float(((pol(g0, oh_start) - a_ref) ** 2).mean())
                n_bt += 1
            beta = 1.0  # 回溯是硬约束, beta 仅记录
        opt_v.zero_grad(); loss_v.backward()
        torch.nn.utils.clip_grad_norm_(crit.parameters(), 1.0); opt_v.step()
        J_val = float(J.mean().detach()); lv = float(loss_v.detach())
        if (it+1) % 10 == 0:
            e_std, e_rets = eval_std()
            if e_std > best_std:
                best_std = e_std
                torch.save({"policy": pol.state_dict(), "it": it+1, "std": e_std},
                           str(a.out).replace(".json", "_best.pt"))
            print(f"  it {it+1}/{a.iters} J={J_val:.4f} Vloss={lv:.4f} "
                  f"std={e_std:.4f} (best {best_std:.4f}, base {std0:.4f}) "
                  f"dev={dev_val:.4f} ({time.time()-t0:.0f}s)", flush=True)
            hist_log.append({"it": it+1, "J": J_val, "vloss": lv, "std": float(e_std),
                             "dev": dev_val})


    final = evaluate(n=10, steps=200)
    print(f"\n最终 raw 每步回报: {final:.5f}  (BC 后 {base:.5f}, 相对 {100*(final-base)/base:+.2f}%)")
    json.dump({"base": float(base), "final": float(final), "log": hist_log,
               "H": a.H, "iters": a.iters, "wm_step": sd.get("step")},
              open(a.out, "w"), indent=1)
    std1, std_rets = eval_std()
    print(f"标准化评测(5 clips) 结束: {std1:.4f}  ({', '.join(f'{r:.3f}' for r in std_rets)})")
    print(f"  teacher=0.469  BC起手={std0:.4f}  BPTT后={std1:.4f}  ({100*(std1-std0)/std0:+.1f}%)")
    d = json.load(open(a.out)); d["std0"]=std0; d["std1"]=std1; d["std_rets"]=std_rets
    json.dump(d, open(a.out,"w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
