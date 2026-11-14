"""P0: 强教师热启动 —— 把公开教师 (93 obs + 680 goal) 的行为蒸馏进我们
论文口径的策略 (10x93 obs + 730 goal)。

为什么这样做而不是直接搬权重: 教师用 anchor-relative 的 680 维 goal,
我们的 BPTT 用论文的 730 维 goal; 两者接口不同。行为克隆是接口无关的,
只需要把教师的"动作"当作标签。

流程:
  1) 教师在我们环境里 rollout -> 记录 (hist93 (10,93), goal730, teacher_action)
  2) 训练 Policy(93, 730) 用 MSE 拟合教师动作
  3) 用与 eval_pretrained.py 完全相同的协议评测 (5 clips, frame 0, 500 步)

判据: BC 后归一化 >= 0.42 (教师 0.469 的 ~90%)
"""
import sys, time, argparse
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
from smini import motion, spec
from smini.env import EnvCfg, TrackingEnv
from smini.policy import Policy
from eval_pretrained import Pol, goal680, obs930

XML = str(ROOT.parent/"assets"/"mujoco_menagerie"/"unitree_g1"/"scene_mjx.xml")
CLEAN = EnvCfg(obs_noise=False, reset_noise=False, domain_randomization=False, action_noise=0.0)


def load_teacher(ckpt, dev="cpu"):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    pol = Pol(); pol.load_state_dict(ck["policy"], strict=True); pol.eval().to(dev)
    return pol, ck.get("iteration")


@torch.no_grad()
def teacher_action(teacher, hist, mo, frame_idx):
    g = goal680(mo.qpos, mo.qvel, frame_idx)
    a = teacher(torch.as_tensor(g[None], dtype=torch.float32),
                torch.as_tensor(obs930(hist)[None], dtype=torch.float32))[0].numpy()
    return a, g


def collect(lib, teacher, n_episodes, ep_steps, rng, clips=None):
    """教师 rollout, 记录 (hist, goal730, action)。"""
    H, G, A = [], [], []
    ids = clips if clips is not None else list(range(len(lib)))
    for ep in range(n_episodes):
        ci = int(rng.choice(ids)); mo = lib[ci]
        if len(mo) < ep_steps + 20: continue
        f0 = int(rng.integers(0, len(mo) - ep_steps - 12))
        env = TrackingEnv(XML, lib, CLEAN, seed=int(rng.integers(1_000_000)))
        obs = env.reset(motion_id=ci, frame_idx=f0)
        hist = np.repeat(obs[None], spec.HISTORY, 0)
        for t in range(ep_steps):
            fi = min(int(env.frame_idx), len(mo)-12)
            a, _ = teacher_action(teacher, hist, mo, fi)
            g730 = mo.goal_window(fi)                     # 论文口径 730
            H.append(hist.copy()); G.append(g730); A.append(a)
            obs, r, te, tr, _ = env.step(a)
            hist = np.roll(hist, -1, 0); hist[-1] = obs
            if te or tr: break
    return (np.stack(H).astype(np.float32), np.stack(G).astype(np.float32),
            np.stack(A).astype(np.float32))


@torch.no_grad()
def evaluate_ours(pol, lib, n_clips=5, steps=500, dev="cuda"):
    """与 scripts/eval_pretrained.py 相同协议, 但用我们的 93/730 策略。"""
    pol.eval()
    env = TrackingEnv(XML, lib, CLEAN, seed=0)
    rets = []
    for ci in range(n_clips):
        obs = env.reset(motion_id=ci, frame_idx=0)
        hist = np.repeat(obs[None], spec.HISTORY, 0)
        ret = 0.0; n = 0
        for t in range(steps):
            fi = min(int(env.frame_idx), len(lib[ci])-1)
            g = torch.as_tensor(lib[ci].goal_window(fi), dtype=torch.float32).to(dev)[None]
            oh = torch.as_tensor(hist[None].astype(np.float32)).to(dev)
            a = pol(g, oh)[0].cpu().numpy()
            obs, r, te, tr, _ = env.step(a)
            hist = np.roll(hist, -1, 0); hist[-1] = obs
            ret += r; n += 1
            if te or tr: break
        rets.append(ret / (0.1 * max(n-1, 1)))     # 与 eval_pretrained 同口径
    pol.train()
    return float(np.mean(rets)), rets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT/"runs/pretrained/policy.pt"))
    ap.add_argument("--motion-dir", default=str(ROOT/"assets/motions"))
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--ep-steps", type=int, default=100)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=str(ROOT/"exp/runs/p0_distill.pt"))
    ap.add_argument("--eval-every", type=int, default=4000)
    ap.add_argument("--n-eval-clips", type=int, default=5,
                    help="评测用库序前 N 条; 训练时默认排除它们以避免泄漏")
    ap.add_argument("--allow-leak", action="store_true",
                    help="允许训练集包含评测 clip (默认禁止)")
    a = ap.parse_args()

    torch.set_num_threads(4); torch.manual_seed(0); rng = np.random.default_rng(0)
    teacher, it = load_teacher(a.ckpt)
    print(f"teacher loaded (iter {it})", flush=True)

    lib = motion.MotionLibrary.from_dir(a.motion_dir)
    print(f"motions: {len(lib)}", flush=True)

    eval_ids = list(range(a.n_eval_clips))
    train_ids = list(range(len(lib))) if a.allow_leak else [i for i in range(len(lib)) if i not in eval_ids]
    print(f"训练 clips: {len(train_ids)} 条 (排除评测 {eval_ids})  allow_leak={a.allow_leak}", flush=True)
    t0 = time.time()
    H, G, A = collect(lib, teacher, a.episodes, a.ep_steps, rng, clips=train_ids)
    print(f"collected {len(A):,} teacher transitions in {time.time()-t0:.0f}s", flush=True)
    print(f"  action std={A.std():.3f}  |a| mean={np.abs(A).mean():.3f}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pol = Policy(obs_dim=93, goal_dim=730, action_dim=29, hist=10).to(dev)
    opt = torch.optim.Adam(pol.parameters(), lr=a.lr)
    Ht = torch.as_tensor(H, device=dev); Gt = torch.as_tensor(G, device=dev); At = torch.as_tensor(A, device=dev)
    N = len(At)
    print(f"train samples {N:,} on {dev}", flush=True)

    best = -1.0
    t0 = time.time()
    for step in range(1, a.steps+1):
        idx = torch.randint(0, N, (a.batch,), device=dev)
        pred = pol(Gt[idx], Ht[idx])
        loss = ((pred - At[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
        if step % 500 == 0:
            print(f"  [{step:5d}] mse {loss.item():.4f} rmse {loss.item()**0.5:.4f} "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            sc, rets = evaluate_ours(pol, lib, n_clips=a.n_eval_clips, dev=dev)
            print(f"  [EVAL {step}] raw policy 归一化 {sc:.4f}  "
                  f"({', '.join(f'{r:.3f}' for r in rets)})", flush=True)
            if sc > best:
                best = sc
                torch.save({"policy": pol.state_dict(), "step": step, "score": sc}, a.out)
                print(f"    saved best ({sc:.4f})", flush=True)
    print(f"DONE best raw normalized = {best:.4f}  (teacher 0.469)  in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
