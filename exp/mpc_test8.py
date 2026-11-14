"""关键实验: 用**完整 8 项奖励**(可微 FK) 做 WM 规划的 BPTT, 看收益能否迁移到真值环境奖励。

对比上一轮 (mpc_test.py, 6 项奖励):
  6 项口径 +24.8% (p=1.4e-9), 但环境 8 项奖励 -0.03% (不显著)
本脚本把 WM 的优化目标换成与 env 完全一致的 8 项奖励 (误差 9e-9), 再放回真值仿真器验证。

用法: python exp/mpc_test8.py --ckpt exp/runs/v2_best.pt --H 32 --N 32 --opt-steps 40
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"exp"))
from smini import motion, spec
from smini.env import EnvCfg, TrackingEnv
from wm_train import make_wm
import diff_fk as D
from diff_fk import FKTable

XML = str(ROOT.parent/"assets"/"mujoco_menagerie"/"unitree_g1"/"scene_mjx.xml")
MDIR = ROOT/"assets"/"motions"
DATA = ROOT/"exp"/"data_v2"
D.set_action_scale(spec.ACTION_SCALE)


def build_ref(mo, frame_idx, T, device="cuda"):
    """从 Motion 取 T 帧参考, 打包成 reward8 需要的 dict (每个 (T,·) 再 unsqueeze 成 batch)。"""
    idx = [min(frame_idx, len(mo)-1) + t for t in range(T)]
    bp = np.stack([mo.body_pos[i] for i in idx]).astype(np.float32)
    bq = np.stack([mo.body_quat[i] for i in idx]).astype(np.float32)
    bl = np.stack([mo.body_linvel[i] for i in idx]).astype(np.float32)
    ba = np.stack([mo.body_angvel[i] for i in idx]).astype(np.float32)
    rp = np.stack([mo.qpos[i, 0:3] for i in idx]).astype(np.float32)
    rq = np.stack([mo.qpos[i, 3:7] for i in idx]).astype(np.float32)
    bR = np.stack([[spec.quat_to_mat(q) for q in fb] for fb in bq]).astype(np.float32)
    rR = np.stack([spec.quat_to_mat(q) for q in rq]).astype(np.float32)
    return {
        "body_pos": torch.as_tensor(bp).to(device),
        "body_R": torch.as_tensor(bR).to(device),
        "body_lin": torch.as_tensor(bl).to(device),
        "body_ang": torch.as_tensor(ba).to(device),
        "root_pos": torch.as_tensor(rp).to(device),
        "root_R": torch.as_tensor(rR).to(device),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT/"exp/runs/v2_best.pt"))
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--opt-steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--bound", type=float, default=3.0)
    ap.add_argument("--out", default=str(ROOT/"exp/runs/mpc8_result.json"))
    a = ap.parse_args()

    torch.set_num_threads(4)
    wm = make_wm(34).to("cuda")
    sd = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    wm.load_state_dict(sd["wm"] if "wm" in sd else sd); wm.eval()
    print(f"WM step={sd.get('step')} nse8={sd.get('nse8')}", flush=True)

    fk = FKTable(XML, spec.TRACKED_BODIES, spec.JOINT_NAMES, device="cuda")
    meta = json.load(open(DATA/"clips.json"))
    lib = motion.MotionLibrary([motion.load_motion(str(MDIR/f"{c}.npz")) for c in meta["train"]])
    CLEAN = EnvCfg(obs_noise=False, reset_noise=False, domain_randomization=False, action_noise=0.0)
    rng = np.random.default_rng(0)
    xi = torch.as_tensor(np.concatenate([[1., 1.], np.zeros(29), np.zeros(3)]),
                         dtype=torch.float32).cuda()[None]

    samples = []
    while len(samples) < a.N:
        mid = int(rng.integers(0, len(lib))); mo = lib[mid]
        if len(mo) < a.H + 30: continue
        f = int(rng.integers(0, len(mo) - a.H - 10))
        samples.append((mid, f))

    def new_env(mid, f):
        e = TrackingEnv(XML, lib, CLEAN, seed=3)
        e.reset(motion_id=mid, frame_idx=f)
        return e

    N = len(samples)
    # ---- 一次性把 N 个样本的初始状态与参考打包成 batch ----
    s0_list, pd_list, ref_list = [], [], []
    for mid, f0 in samples:
        mo = lib[mid]
        e0 = new_env(mid, f0)
        q0 = e0._data.qpos.copy(); v0 = e0._data.qvel.copy()
        s0_list.append(np.concatenate([q0[0:3], q0[3:7], q0[7:36], v0[0:3], v0[3:6], v0[6:35]]))
        pd_list.append(np.stack([
            (mo.qpos[f0+1+t][7:36] - np.asarray(spec.DEFAULT_Q)) / np.asarray(spec.ACTION_SCALE)
            for t in range(a.H)]))
        r = build_ref(mo, f0+1, a.H, device="cpu")
        ref_list.append({k: v.numpy() for k, v in r.items()})
    s0 = torch.as_tensor(np.stack(s0_list).astype(np.float32)).cuda()          # (N,71)
    pd_acts = torch.as_tensor(np.stack(pd_list).astype(np.float32)).cuda()     # (N,H,29)
    refb = {k: torch.as_tensor(np.stack([r[k] for r in ref_list]).astype(np.float32)).cuda()
            for k in ref_list[0]}                                              # (N,H,...)
    xi_b = xi.expand(N, -1).contiguous()

    # ---- 基线: PD 动作在真值仿真器里执行 ----
    base = []
    for (mid, f0) in samples:
        e = new_env(mid, f0); Rb = 0.0
        for t in range(a.H):
            act = e._goal[15:44] / spec.ACTION_SCALE
            _, r, te, tr, _ = e.step(act)
            Rb += float(r)
            if te or tr: break
        base.append(Rb)

    # ---- 批量 WM 规划 ----
    raw = torch.zeros_like(pd_acts, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=a.lr)
    t0 = time.time()
    for it in range(a.opt_steps):
        acts = pd_acts + a.bound * torch.tanh(raw)        # (N,H,29)
        s = s0
        pv = torch.zeros(N, 29, device="cuda")
        tot = 0.0
        for t in range(a.H):
            s = wm.step(s, acts[:, t], xi_b)
            r = D.reward8(s, acts[:, t], pv, {k: v[:, t] for k, v in refb.items()}, fk)
            tot = tot + r.sum()
            pv = acts[:, t]
        opt.zero_grad(); (-tot).backward(); opt.step()
        if (it+1) % 10 == 0:
            print(f"  opt {it+1}/{a.opt_steps}  imag_return/step={float(tot)/(N*a.H):.5f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    with torch.no_grad():
        acts_final = (pd_acts + a.bound * torch.tanh(raw)).cpu().numpy()

    # ---- 放回真值仿真器 ----
    plan = []
    for i, (mid, f0) in enumerate(samples):
        e = new_env(mid, f0); Rw = 0.0
        for t in range(a.H):
            _, r, te, tr, _ = e.step(acts_final[i, t])
            Rw += float(r)
            if te or tr: break
        plan.append(Rw)

    base = np.array(base); plan = np.array(plan)
    d = plan - base
    from scipy import stats
    tt = stats.ttest_rel(plan, base)
    print("\n===== 结果 (完整 8 项奖励, 与 env 完全一致) =====")
    print(f"真值回报 (H={a.H}): 基线(PD)={base.mean():.5f}  WM规划={plan.mean():.5f}")
    print(f"提升 {d.mean():+.5f} ({d.mean()/base.mean()*100:+.2f}%)  改善占比 {(d>0).mean()*100:.1f}%")
    print(f"配对 t: t={tt.statistic:.3f} p={tt.pvalue:.4g}")
    json.dump({"base": base.tolist(), "plan": plan.tolist(),
               "mean_base": float(base.mean()), "mean_plan": float(plan.mean()),
               "delta_pct": float(d.mean()/base.mean()*100),
               "win_rate": float((d > 0).mean()), "p": float(tt.pvalue),
               "H": a.H, "N": a.N, "wm_step": sd.get("step")},
              open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
