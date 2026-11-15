"""阶段三预实验: 在**教师状态分布**上，规划能否超过教师本身?

背景: 已有规划结果 (+3.79%~+7.29%) 的基线是 PD 动作、开环执行、只测 H 步短回报。
      而 PD (0.0578) 只有教师 (0.3863) 的 1/6.7 —— "超过教师"从未被测过。

本实验在同一批**教师访问到的状态**上, 比较 4 个条件(全部在真值仿真器里执行):
  PD 开环 / 教师开环 / 从 PD 规划 / 从教师动作规划

一致性保证: 每个条件都用**同一段记录下来的热身动作序列**重放, 确保起点完全相同。

用法: python exp/mpc_teacher_pilot.py --H 8 --N 48
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

XML = EK.XML
CLEAN = EK.CLEAN
DEV = "cuda"


def pd_act(env):
    return env._goal[15:44] / spec.ACTION_SCALE


def tc_act(env, mo, hist, net):
    fi = min(int(env.frame_idx), len(mo) - 1)
    g = torch.as_tensor(goal680(mo.qpos, mo.qvel, fi), dtype=torch.float32).to(DEV)[None]
    oh = torch.as_tensor(obs930(np.stack(hist))[None].astype(np.float32)).to(DEV)
    return net(g, oh)[0].detach().cpu().numpy()


def step_env(e, act):
    obs, r, te, tr, _ = e.step(act)
    return obs, float(r), (te or tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="exp/runs/D_wm20k.pt")
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--opt-steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--bound", type=float, default=0.3)
    ap.add_argument("--out", default="exp/runs/mpc_teacher_pilot.json")
    a = ap.parse_args()
    torch.set_num_threads(4)
    rng = np.random.default_rng(0)
    H, N = a.H, a.N
    WARM = spec.HISTORY

    tck = torch.load(ROOT/"runs/pretrained/policy.pt", map_location=DEV, weights_only=False)
    net = Pol(); net.load_state_dict(tck["policy"], strict=True); net.to(DEV).eval()
    wm = make_wm(34).to(DEV)
    sd = torch.load(ROOT/a.wm, map_location=DEV, weights_only=False)
    wm.load_state_dict(sd["wm"] if "wm" in sd else sd); wm.eval()
    fk = FKTable(XML, spec.TRACKED_BODIES, spec.JOINT_NAMES, device=DEV)
    xi = torch.as_tensor(np.concatenate([[1., 1.], np.zeros(29), np.zeros(3)]),
                         dtype=torch.float32).to(DEV)[None]

    lib = EK.load_lib(ROOT/"assets/motions_trim")
    samples = []
    while len(samples) < N:
        mid = int(rng.integers(0, len(lib))); mo = lib[mid]
        if len(mo) < H + 60: continue
        samples.append((mid, int(rng.integers(0, len(mo) - H - 30))))

    S0, hist0, refs, warm_seqs, tc_seqs, pd_seqs = [], [], [], [], [], []
    for (mid, f0) in samples:
        mo = lib[mid]
        e = TrackingEnv(XML, lib, CLEAN, seed=3)
        obs = e.reset(motion_id=mid, frame_idx=f0)
        hist = [obs] * WARM
        warm = []
        with torch.no_grad():
            for _ in range(WARM):
                act = tc_act(e, mo, np.stack(hist), net)
                warm.append(act)
                obs, _, done = step_env(e, act)
                hist = hist[1:] + [obs]
                if done: break
        hist0.append(np.stack(hist)); warm_seqs.append(np.stack(warm))
        q = e._data.qpos; v = e._data.qvel
        S0.append(np.concatenate([q[0:3], q[3:7], q[7:36], v[0:3], v[3:6], v[6:35]]))
        fs = min(int(e.frame_idx), len(mo) - H - 2)
        refs.append(build_ref(mo, fs + 1, H, device="cpu"))

        # 从同一状态出发, 分别记录教师序列与 PD 序列
        def seq_of(kind):
            e2 = TrackingEnv(XML, lib, CLEAN, seed=3)
            o2 = e2.reset(motion_id=mid, frame_idx=f0)
            h2 = [o2] * WARM
            for wa in warm:
                o2, _, dn = step_env(e2, wa)
                h2 = h2[1:] + [o2]
                if dn: break
            out = []
            for _ in range(H):
                act = tc_act(e2, mo, np.stack(h2), net) if kind == "tc" else pd_act(e2)
                out.append(act)
                o2, _, dn = step_env(e2, act)
                h2 = h2[1:] + [o2]
                if dn: break
            while len(out) < H: out.append(out[-1])
            return np.stack(out)
        tc_seqs.append(seq_of("tc")); pd_seqs.append(seq_of("pd"))
        print(f"  采样 {len(S0)}/{N}", end="\r", flush=True)
    print()

    S0t = torch.as_tensor(np.stack(S0).astype(np.float32)).to(DEV)
    pd_a = torch.as_tensor(np.stack(pd_seqs).astype(np.float32)).to(DEV)
    tc_a = torch.as_tensor(np.stack(tc_seqs).astype(np.float32)).to(DEV)
    refb = {k: torch.as_tensor(np.stack([r[k] for r in refs]).astype(np.float32)).to(DEV)
            for k in refs[0]}
    xi_b = xi.expand(N, -1).contiguous()

    def true_rollout(acts):
        """重放同一段热身, 再开环执行 acts, 返回 H 步累计真值奖励。"""
        out = []
        for i, (mid, f0) in enumerate(samples):
            e = TrackingEnv(XML, lib, CLEAN, seed=3)
            e.reset(motion_id=mid, frame_idx=f0)
            for wa in warm_seqs[i]:
                _, _, dn = step_env(e, wa)
                if dn: break
            R = 0.0
            for t in range(H):
                _, r, dn = step_env(e, acts[i, t])
                R += r
                if dn: break
            out.append(R)
        return np.array(out)

    def plan_from(init_acts, tag):
        raw = torch.zeros_like(init_acts, requires_grad=True)
        opt = torch.optim.Adam([raw], lr=a.lr)
        t0 = time.time()
        for _ in range(a.opt_steps):
            acts = init_acts + a.bound * torch.tanh(raw)
            s = S0t; pv = torch.zeros(N, 29, device=DEV); tot = 0.0
            for t in range(H):
                s = wm.step(s, acts[:, t], xi_b)
                r = D.reward8(s, acts[:, t], pv, {k: v[:, t] for k, v in refb.items()}, fk)
                tot = tot + r.sum(); pv = acts[:, t]
            opt.zero_grad(); (-tot).backward(); opt.step()
        with torch.no_grad():
            A = (init_acts + a.bound * torch.tanh(raw)).cpu().numpy()
        print(f"  [{tag}] 规划完成 {a.opt_steps} 步 ({time.time()-t0:.0f}s)", flush=True)
        return A

    print(f"H={H} N={N}: 构造基线 ...", flush=True)
    base_pd = true_rollout(pd_a.cpu().numpy())
    base_tc = true_rollout(tc_a.cpu().numpy())
    plan_pd = true_rollout(plan_from(pd_a, "from-PD"))
    plan_tc = true_rollout(plan_from(tc_a, "from-teacher"))

    from scipy import stats
    print(f"\n===== 真值回报 (H={H} 步, 同一批教师状态) =====")
    for nm, v in (("PD 开环", base_pd), ("教师开环", base_tc),
                  ("从 PD 规划", plan_pd), ("从教师规划", plan_tc)):
        print(f"  {nm:14s} {v.mean():.5f}")
    print()
    for nm, v, b in (("规划 vs PD", plan_pd, base_pd), ("规划 vs 教师", plan_tc, base_tc)):
        d = v - b
        t = stats.ttest_rel(v, b)
        print(f"  {nm:14s} {d.mean():+.5f} ({d.mean()/b.mean()*100:+.2f}%)  "
              f"胜率 {(d>0).mean()*100:.0f}%  p={t.pvalue:.4g}")
    json.dump({"H": H, "N": N, "pd": base_pd.tolist(), "teacher": base_tc.tolist(),
               "plan_from_pd": plan_pd.tolist(), "plan_from_teacher": plan_tc.tolist()},
              open(ROOT/a.out, "w"), indent=1)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
