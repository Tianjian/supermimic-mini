#!/usr/bin/env python3
"""E1/E2: world-model accuracy on teacher vs PD distributions.

Collects episode transitions (teacher or PD) on trimmed motions, trains a
residual WM (73 or 283 dims) and reports 1/8/32-step NRMSE normalized by
persistence on held-out episodes.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, torch
from smini import motion, spec, build_model
from smini.env import TrackingEnv, EnvCfg
from smini.replay import ReplayBuffer
from smini.norm import compute_stats
from smini.world_model import WorldModel
from eval_pretrained import Pol, goal680, obs930

XML = str(ROOT.parent / "assets/mujoco_menagerie/unitree_g1/scene_mjx.xml")


def collect(kind, n_steps, motion_dir, seed=0):
    lib = motion.MotionLibrary.from_dir(ROOT / motion_dir); lib.motions = lib.motions[:5]
    env = TrackingEnv(XML, lib, EnvCfg(obs_noise=False, reset_noise=False, domain_randomization=False, action_noise=0), seed=seed)
    teacher = None
    if kind == "teacher":
        ck = torch.load(ROOT / "runs/pretrained/policy.pt", map_location="cpu", weights_only=False)
        teacher = Pol(); teacher.load_state_dict(ck["policy"]); teacher.eval()
    rng = np.random.default_rng(seed)
    eps = []
    total = 0
    while total < n_steps:
        mi = int(rng.integers(0, len(lib.motions)))
        t0 = int(rng.integers(0, max(1, len(lib.motions[mi]) - 600)))
        obs = env.reset(motion_id=mi, frame_idx=t0)
        hist = np.repeat(obs[None], spec.HISTORY, 0)
        S, A, X, R, D, TR, G, F, O = [], [], [], [], [], [], [], [], []
        for t in range(500):
            mo = lib.motions[mi]
            if teacher is not None:
                g = goal680(mo.qpos, mo.qvel, env.frame_idx)
                with torch.no_grad():
                    a0 = teacher(torch.as_tensor(g[None], dtype=torch.float32),
                                 torch.as_tensor(obs930(hist)[None], dtype=torch.float32))[0].numpy()
            else:
                a0 = np.clip((mo.qpos[env.frame_idx, 7:36] - spec.DEFAULT_Q) / spec.ACTION_SCALE, -5, 5)
            a = np.clip(a0 + rng.normal(0, 0.1, 29), -50, 50)
            obs, r, te, tr, info = env.step(a)
            hist = np.roll(hist, -1, 0); hist[-1] = obs
            S.append(info["state"]); O.append(obs); A.append(a); X.append(info["xi"]); R.append(r)
            D.append(float(te)); TR.append(float(tr)); G.append(info["goal"]); F.append(info["frame_idx"])
            if te or tr:
                break
        eps.append({"state": np.asarray(S, np.float32), "action": np.asarray(A, np.float32),
                    "xi": np.asarray(X, np.float32), "reward": np.asarray(R, np.float32),
                    "terminated": np.asarray(D, np.float32), "truncated": np.asarray(TR, np.float32),
                    "goal": np.asarray(G, np.float32), "motion_id": mi,
                    "frame_idx": np.asarray(F, np.int64), "obs": np.asarray(O, np.float32)})
        total += len(S)
    return eps, lib


def make_buf(eps, cap):
    b = ReplayBuffer(cap)
    for e in eps:
        b.add_episode(e)
    return b


def train_wm(buf, dim, iters, k, batch, dev="cuda", seed=0):
    torch.manual_seed(seed)
    stats = compute_stats(buf)
    sm, ss = stats["state"]
    if dim == 73:
        sm, ss = sm[:73], ss[:73]
    wm = WorldModel(state_dim=dim, hidden=(1024, 1024, 1024)).to(dev)
    wm.set_norm(sm, ss, *stats["action"], *stats["xi"], *stats["goal_cur"])
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    wm.train()
    for it in range(1, iters + 1):
        tot = 0.0
        for _ in range(4):
            b = buf.sample_windows(batch, k, device=dev)
            s = wm.norm_state(b["s"][..., :dim]); tgt = wm.norm_state(b["s_next"][..., :dim])
            loss = 0.0
            for j in range(k):
                s = wm.forward_norm(s, b["a"][:, j], b["xi"])
                sl = s[..., :73] if dim == 283 else s
                tl = tgt[:, j, :73] if dim == 283 else tgt[:, j]
                loss = loss + ((sl - tl) ** 2).mean()
            loss = loss / k
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot += float(loss)
        if it % 200 == 0:
            print(f"  wm[{it}] loss {tot/4:.4f}", flush=True)
    wm.eval()
    return wm


@torch.no_grad()
def nrmse(wm, buf, dim, steps=32, batch=256, dev="cuda"):
    b = buf.sample_windows(batch, steps, device=dev)
    s = wm.norm_state(b["s"][..., :dim]); s0 = s.clone()
    preds = []
    for j in range(steps):
        s = wm.forward_norm(s, b["a"][:, j], b["xi"])
        preds.append(wm.denorm_state(s)[..., :73] if dim == 283 else wm.denorm_state(s))
    tgt = b["s_next"][..., :73]
    base = b["s"][..., :73]
    out = {}
    std = wm.s_std[:73].clamp_min(1e-3)
    for j in (1, 8, 32):
        p = preds[j-1]
        err = ((p - tgt[:, j-1]) ** 2).mean().sqrt()
        per = ((base - tgt[:, j-1]) ** 2).mean().sqrt()
        norm = (((p - tgt[:, j-1]) / std) ** 2).mean().sqrt()
        out[j] = (float(err / per), float(norm))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["teacher", "pd"], default="teacher")
    ap.add_argument("--dim", type=int, choices=[73, 283], default=73)
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--motion-dir", default="assets/motions_trim")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda"
    eps, lib = collect(args.kind, args.steps, args.motion_dir, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    idx = rng.permutation(len(eps)); cut = max(1, int(len(eps) * 0.8))
    tr = [eps[i] for i in idx[:cut]]; te = [eps[i] for i in idx[cut:]]
    print(f"{args.kind}: {len(eps)} eps, train {len(tr)} test {len(te)}")
    trb = make_buf(tr, 10**6); teb = make_buf(te, 10**6)
    wm = train_wm(trb, args.dim, args.iters, 8, 512, dev, args.seed)
    res = nrmse(wm, teb, args.dim, 32, 256, dev)
    print(f"E{1 if args.kind=='teacher' else 2} {args.kind} dim={args.dim} NRMSE "
          f"1-step ratio {res[1][0]:.3f}/std {res[1][1]:.3f} 8-step ratio {res[8][0]:.3f}/std {res[8][1]:.3f} 32-step ratio {res[32][0]:.3f}/std {res[32][1]:.3f}")
    torch.save({"wm": wm.state_dict(), "dim": args.dim, "kind": args.kind, "nrmse": res},
               ROOT / f"runs/wm_{args.kind}_{args.dim}.pt")


if __name__ == "__main__":
    main()
