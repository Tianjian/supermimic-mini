"""BPTT 可用性判决实验 (Test A: WM 规划的 H 步动作优化)。

科学问题: 对 WM 做 H=32 步 BPTT 得到的动作, 在**真值仿真器**里是否真的更好?
  - 基线 : PD 跟踪动作 (参考动作), 真值回报 R_base
  - BPTT : 同一初始状态, 用 WM 想象的回报对动作序列做梯度上升, 真值回报 R_wm
  - 若 R_wm > R_base 显著 -> WM 的动作梯度可用 -> WM+BPTT 可用
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np, torch, mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"exp"))
from smini.wm_import import WM71
from smini import motion, spec
from smini.env import EnvCfg, TrackingEnv
import collect as C

XML = str(ROOT.parent/"assets"/"mujoco_menagerie"/"unitree_g1"/"scene_mjx.xml")
MDIR = ROOT/"assets"/"motions"
DQ = torch.as_tensor(np.asarray(spec.DEFAULT_Q), dtype=torch.float32)
ASC = torch.as_tensor(np.asarray(spec.ACTION_SCALE), dtype=torch.float32)
JMASK = torch.as_tensor(np.asarray(spec.JOINT_REWARD_MASK), dtype=torch.float32)


def quat_to_mat(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1),
        torch.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
        torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1),
    ], -2)


def rot6d_to_mat(v6):
    a, b = v6[..., 0:3], v6[..., 3:6]
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b = b - a*(a*b).sum(-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    c = torch.cross(a, b, dim=-1)
    return torch.stack([a, b, c], -1)


def reward71(s71, a, prev_a, goal73):
    """71 维状态 + 73 维参考帧 -> 6 项跟踪奖励 (与 imagine.analytic_reward 同构)"""
    p = s71[..., 0:3]; R = quat_to_mat(s71[..., 3:7])
    qj = s71[..., 7:36]; qd = s71[..., 42:71]
    p_ref = goal73[..., 0:3]; R_ref = rot6d_to_mat(goal73[..., 3:9])
    q_ref = goal73[..., 15:44]; qd_ref = goal73[..., 44:73]
    root_pos = ((p - p_ref) ** 2).sum(-1)
    tr = (R_ref.transpose(-1, -2) @ R).diagonal(dim1=-2, dim2=-1).sum(-1)
    root_ori = (3.0 - tr).clamp_min(0.0)
    jm = JMASK.to(s71.device); ac = ASC.to(s71.device)
    denom = jm.sum().clamp_min(1.0)
    jpos = (((qj - q_ref) ** 2) * jm).sum(-1) / denom
    jvel = (((qd - qd_ref) ** 2) * jm).sum(-1) / denom
    smooth = ((ac * (a - prev_a)) ** 2).sum(-1)
    r = (0.5*torch.exp(-root_pos/0.3**2) + 0.5*torch.exp(-root_ori/0.4**2)
         + torch.exp(-jpos/0.5**2) + torch.exp(-jvel/1.0**2) - 0.1*smooth)
    return r * 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT/"exp/runs/fixed_best.pt"))
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--opt-steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--bound", type=float, default=3.0)
    ap.add_argument("--out", default=str(ROOT/"exp/runs/mpc_result.json"))
    a = ap.parse_args()

    torch.set_num_threads(4)
    from wm_train import make_wm
    wm = make_wm(34).to("cuda")
    sd = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    wm.load_state_dict(sd["wm"] if "wm" in sd else sd); wm.eval()
    print(f"WM loaded step={sd.get('step')} nse8={sd.get('nse8')}", flush=True)

    meta = json.load(open(ROOT/"exp/data/clips.json"))
    trclip = meta["train"]
    lib = motion.MotionLibrary([motion.load_motion(str(MDIR/f"{c}.npz")) for c in trclip])
    CLEAN = EnvCfg(obs_noise=False, reset_noise=False, domain_randomization=False, action_noise=0.0)
    rng = np.random.default_rng(0)
    xi_np = np.concatenate([[1.0, 1.0], np.zeros(29), np.zeros(3)]).astype(np.float32)
    xi_t = torch.as_tensor(xi_np)[None].cuda()

    # 取样: 每个 motion 取若干起点
    samples = []
    while len(samples) < a.N:
        mid = rng.integers(0, len(lib))
        mo = lib[mid]
        if len(mo) < a.H + 30: continue
        f = int(rng.integers(0, len(mo) - a.H - 10))
        samples.append((mid, f))

    def make_env():
        e = TrackingEnv(XML, lib, CLEAN, seed=3)
        return e

    base_ret, wm_ret, imag_ret = [], [], []
    base71, wm71 = [], []
    t0 = time.time()
    for si, (mid, f0) in enumerate(samples):
        mo = lib[mid]
        goal_seq = torch.stack([
            torch.as_tensor(mo.goal_frame(f0 + 1 + t), dtype=torch.float32) for t in range(a.H)
        ]).cuda()                                        # (H,73)

        e = make_env(); e.reset(motion_id=mid, frame_idx=f0)
        s0 = torch.as_tensor(C.to71(np.stack([e.state()]), np.asarray(spec.DEFAULT_Q))[0],
                             dtype=torch.float32).cuda()[None]
        prev_a0 = torch.zeros(1, 29, device="cuda")

        # --- 基线: PD 跟踪动作, 在真值仿真器里执行 ---
        e_b = make_env(); e_b.reset(motion_id=mid, frame_idx=f0)
        Rb = 0.0; Rb71 = 0.0; prevb = torch.zeros(1,29,device="cuda")
        for t in range(a.H):
            act = e_b._goal[slice(15,44)] / spec.ACTION_SCALE
            _, r, te, tr, _ = e_b.step(act)
            Rb += float(r)
            st = torch.as_tensor(C.to71(np.stack([e_b.state()]), np.asarray(spec.DEFAULT_Q))[0],
                                 dtype=torch.float32).cuda()[None]
            at = torch.as_tensor(act, dtype=torch.float32).cuda()[None]
            Rb71 += float(reward71(st, at, prevb, goal_seq[t:t+1])); prevb = at
            if te or tr: break
        base_ret.append(Rb); base71.append(Rb71)

        # --- 初始化动作 = PD 参考动作 (a = (q_ref - Q0)/alpha) ---
        pd_acts = torch.stack([
            torch.as_tensor((mo.qpos[f0+1+t][7:36] - np.asarray(spec.DEFAULT_Q)) / np.asarray(spec.ACTION_SCALE),
                            dtype=torch.float32) for t in range(a.H)
        ]).cuda()                                        # (H,29)

        acts = pd_acts.clone().requires_grad_(True)
        opt = torch.optim.Adam([acts], lr=a.lr)
        loss_val = None
        for it in range(a.opt_steps):
            s = s0; prev = prev_a0; tot = 0.0
            for t in range(a.H):
                s = wm.step(s, acts[t:t+1], xi_t)
                r = reward71(s, acts[t:t+1], prev, goal_seq[t:t+1])
                tot = tot + r.sum()
                prev = acts[t:t+1]
            loss = -tot
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                d = acts - pd_acts
                d.clamp_(-a.bound, a.bound)
                acts.copy_(pd_acts + d)
            loss_val = float(loss)
        imag_ret.append(-loss_val / a.H)                 # 每步平均想象回报

        # --- 把 WM 优化的动作序列放进真值仿真器 ---
        acts_np = acts.detach().cpu().numpy()
        e_w = make_env(); e_w.reset(motion_id=mid, frame_idx=f0)
        Rw = 0.0; Rw71 = 0.0; prevw = torch.zeros(1,29,device="cuda")
        for t in range(a.H):
            _, r, te, tr, _ = e_w.step(acts_np[t])
            Rw += float(r)
            st = torch.as_tensor(C.to71(np.stack([e_w.state()]), np.asarray(spec.DEFAULT_Q))[0],
                                 dtype=torch.float32).cuda()[None]
            at = torch.as_tensor(acts_np[t], dtype=torch.float32).cuda()[None]
            Rw71 += float(reward71(st, at, prevw, goal_seq[t:t+1])); prevw = at
            if te or tr: break
        wm_ret.append(Rw); wm71.append(Rw71)
        if (si+1) % 8 == 0:
            print(f"  [{si+1}/{a.N}] base={np.mean(base_ret):.4f} wm={np.mean(wm_ret):.4f} "
                  f"imag/step={np.mean(imag_ret):.4f}  ({time.time()-t0:.0f}s)", flush=True)

    base_ret = np.array(base_ret); wm_ret = np.array(wm_ret)
    d = wm_ret - base_ret
    base71=np.array(base71); wm71=np.array(wm71)
    print("\n=== 同口径 (reward71, 与 WM 优化目标一致) ===")
    print(f"  基线={base71.mean():.5f}  WM规划={wm71.mean():.5f}  "
          f"提升={((wm71-base71).mean()/base71.mean()*100):+.2f}%  改善占比={((wm71-base71)>0).mean()*100:.1f}%")
    from scipy import stats as _s
    _t=_s.ttest_rel(wm71,base71); print(f"  配对 t: t={_t.statistic:.3f} p={_t.pvalue:.4g}")
    print("\n===== 结果 (环境 8 项奖励) =====", flush=True)
    print(f"真值回报 (H={a.H} 步累计): 基线(PD)={base_ret.mean():.5f}  WM规划={wm_ret.mean():.5f}")
    print(f"提升: {d.mean():+.5f}  ({d.mean()/base_ret.mean()*100:+.2f}%)   "
          f"改善样本占比: {(d>0).mean()*100:.1f}%  (n={len(d)})")
    # 配对检验
    from scipy import stats
    tt = stats.ttest_rel(wm_ret, base_ret)
    print(f"配对 t 检验: t={tt.statistic:.3f} p={tt.pvalue:.4g}")
    json.dump({"base": base_ret.tolist(), "wm": wm_ret.tolist(),
               "mean_base": float(base_ret.mean()), "mean_wm": float(wm_ret.mean()),
               "delta_pct": float(d.mean()/base_ret.mean()*100),
               "win_rate": float((d>0).mean()), "p": float(tt.pvalue),
               "mean_base71": float(base71.mean()), "mean_wm71": float(wm71.mean()),
               "H": a.H, "N": a.N}, open(a.out,"w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
