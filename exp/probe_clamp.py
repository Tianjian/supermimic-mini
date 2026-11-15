"""诊断: 动作夹具是否把 BPTT 的有效梯度投影掉了?

昨晚结论"夹具把崩塌从 -80% 变成 -1.2%"，但有三种可能:
  (a) 夹具让 BPTT 安全了（真改进）
  (b) 夹具把绝大多数动作夹住 -> 梯度被投影掉 -> 等于关掉了 BPTT
  (c) critic bootstrap 饱和，主导了目标

本脚本量化: 夹取比例 / 策略梯度范数 / J 的奖励项与 bootstrap 项占比 / V 的饱和程度。
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, torch, importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts")); sys.path.insert(0, str(ROOT/"exp"))
import evalkit as EK
from eval_pretrained import Pol, goal680
from smini import motion, spec
from smini.env import TrackingEnv
from smini.policy import Critic
from wm_train import make_wm
from mpc_test8 import build_ref
import diff_fk as D
from diff_fk import FKTable

sp = importlib.util.spec_from_file_location("acb", str(ROOT/"exp"/"ac_bptt.py"))
acb = importlib.util.module_from_spec(sp); sp.loader.exec_module(acb)

XML = EK.XML
CLEAN = EK.CLEAN


def build(wm_ckpt, device="cuda"):
    tck = torch.load(str(ROOT/"runs/pretrained/policy.pt"), map_location=device, weights_only=False)
    net = Pol(); net.load_state_dict(tck["policy"], strict=True)
    pol = acb.TeacherBase(net).to(device).eval()
    for p in pol.parameters(): p.requires_grad_(True)
    wm = make_wm(34).to(device)
    sd = torch.load(str(ROOT/wm_ckpt), map_location=device, weights_only=False)
    wm.load_state_dict(sd["wm"] if "wm" in sd else sd); wm.eval()
    for p in wm.parameters(): p.requires_grad_(False)
    fk = FKTable(XML, spec.TRACKED_BODIES, spec.JOINT_NAMES, device=device)
    crit = Critic(state_dim=71, goal_dim=680, xi_dim=34).to(device)
    return pol, wm, fk, crit


def sample_batch(lib, pol, B, H, rng, device="cuda"):
    starts = []
    while len(starts) < B:
        mid = int(rng.integers(0, len(lib))); mo = lib[mid]
        if len(mo) < H + 40: continue
        starts.append((mid, int(rng.integers(0, len(mo) - H - 15))))
    s0, oh0, goals, refs, used = [], [], [], [], []
    with torch.no_grad():
        for mid, f0 in starts:
            mo = lib[mid]
            e = TrackingEnv(XML, lib, CLEAN, seed=3)
            obs = e.reset(motion_id=mid, frame_idx=f0)
            hist = [obs] * spec.HISTORY
            for _ in range(spec.HISTORY):
                fi = min(int(e.frame_idx), len(mo)-1)
                g = torch.as_tensor(goal680(mo.qpos, mo.qvel, fi), dtype=torch.float32).to(device)[None]
                oh = torch.as_tensor(np.stack(hist)[None].astype(np.float32)).to(device)
                act = pol(g, oh)[0].cpu().numpy()
                obs, r, te, tr, _ = e.step(act)
                hist = hist[1:] + [obs]
                if te or tr: break
            fs = min(int(e.frame_idx), max(len(mo)-H-2, 0))
            q = e._data.qpos; v = e._data.qvel
            s0.append(np.concatenate([q[0:3], q[3:7], q[7:36], v[0:3], v[3:6], v[6:35]]))
            goals.append(goal680(mo.qpos, mo.qvel, fs))
            refs.append(build_ref(mo, fs+1, H, device="cpu"))
            oh0.append(np.stack(hist)); used.append((mid, fs))
    gseq = torch.as_tensor(np.stack([
        np.stack([goal680(lib[m].qpos, lib[m].qvel, f+t) for t in range(H)]) for m, f in used
    ]).astype(np.float32)).to(device)
    return (torch.as_tensor(np.stack(s0).astype(np.float32)).to(device),
            torch.as_tensor(np.stack(oh0).astype(np.float32)).to(device),
            torch.as_tensor(np.stack(goals).astype(np.float32)).to(device),
            {k: torch.as_tensor(np.stack([r[k] for r in refs]).astype(np.float32)).to(device)
             for k in refs[0]}, gseq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="exp/runs/D_wm20k.pt")
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--gamma", type=float, default=0.99)
    a = ap.parse_args()
    torch.set_num_threads(4); rng = np.random.default_rng(0)
    dev = "cuda"
    pol, wm, fk, crit = build(a.wm, dev)
    lib = EK.load_lib(ROOT/"assets/motions_trim")
    xi = torch.as_tensor(np.concatenate([[1.,1.], np.zeros(29), np.zeros(3)]),
                         dtype=torch.float32).to(dev)[None]

    for eps in (0.0, 0.05, 0.2, 0.5):
        torch.manual_seed(0)
        crit2 = Critic(state_dim=71, goal_dim=680, xi_dim=34).to(dev)
        s0, oh0, g0, refb, gseq = sample_batch(lib, pol, a.B, a.H, rng, dev)
        Nb = s0.shape[0]
        with torch.no_grad():
            a_anc_all = [pol(gseq[:, t], oh0) for t in range(1)]  # 仅参考
        s = s0; oh = oh0.clone(); prev = torch.zeros(Nb, 29, device=dev)
        Rs = []; frac_clamped = []; devs = []
        for t in range(a.H):
            act = pol(gseq[:, t], oh)
            with torch.no_grad():
                a_anc = pol(gseq[:, t], oh)
            if eps > 0:
                lo, hi = a_anc - eps, a_anc + eps
                frac_clamped.append(float(((act < lo) | (act > hi)).float().mean()))
                act = torch.clamp(act, lo, hi)
            else:
                frac_clamped.append(0.0)
            s = wm.step(s, act, xi.expand(Nb, -1))
            r = D.reward8(s, act, prev, {k: v[:, t] for k, v in refb.items()}, fk)
            Rs.append(r)
            oh = torch.cat([oh[:, 1:], acb.obs71(s, act)[:, None, :]], 1)
            prev = act
        R = torch.stack(Rs, 1)
        disc = 1.0; Jrew = torch.zeros(Nb, device=dev)
        for t in range(a.H):
            Jrew = Jrew + disc * R[:, t]; disc *= a.gamma
        with torch.no_grad():
            v_last = crit(s.detach(), gseq[:, -1], xi.expand(Nb, -1)).squeeze(-1)
        J = Jrew.mean() + disc * v_last.mean()
        # 策略梯度: 只对奖励项反传(不含 saturated bootstrap)
        pol.zero_grad()
        Jrew.mean().backward(retain_graph=True)
        g_rew = sum(float(p.grad.norm())**2 for p in pol.parameters() if p.grad is not None) ** 0.5
        pol.zero_grad()
        J.backward()
        g_all = sum(float(p.grad.norm())**2 for p in pol.parameters() if p.grad is not None) ** 0.5
        pol.zero_grad()
        print(f"eps={eps:<4}  夹取比例={np.mean(frac_clamped)*100:5.1f}%  "
              f"奖励项={Jrew.mean().item():6.3f}  bootstrap={disc*v_last.mean().item():6.3f}  "
              f"|grad(奖励)|={g_rew:.3e}  |grad(总)|={g_all:.3e}")


if __name__ == "__main__":
    main()
