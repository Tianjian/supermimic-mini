#!/usr/bin/env python3
"""One DAgger round: roll out the BC policy, label visited states with the teacher, retrain."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, torch
from smini import motion, spec
from smini.env import TrackingEnv, EnvCfg
from eval_pretrained import Pol, goal680, obs930
from bc_teacher import BC, train, evaluate

XML = str(ROOT.parent / "assets/mujoco_menagerie/unitree_g1/scene_mjx.xml")


def collect_bc(n, pol, motion_dir, seed=0):
    lib = motion.MotionLibrary.from_dir(ROOT / motion_dir)
    env = TrackingEnv(XML, lib, EnvCfg(obs_noise=False, reset_noise=False,
                                       domain_randomization=False, action_noise=0.05), seed=seed)
    rng = np.random.default_rng(seed)
    O, G, A = [], [], []
    mi = int(rng.integers(0, len(lib.motions)))
    t0 = int(rng.integers(0, max(1, len(lib.motions[mi]) - 600)))
    obs = env.reset(motion_id=mi, frame_idx=t0)
    hist = np.repeat(obs[None], spec.HISTORY, 0)
    with torch.no_grad():
        while len(O) < n:
            g = goal680(lib.motions[env.motion_id].qpos, lib.motions[env.motion_id].qvel, env.frame_idx)
            o930 = obs930(hist)
            x = np.concatenate([o930, g])
            a_bc = pol(torch.as_tensor(x[None], dtype=torch.float32, device="cuda"))[0].cpu().numpy()
            O.append(o930); G.append(g); A.append(a_bc)  # placeholder, teacher label below
            obs, r, te, tr, info = env.step(a_bc)
            hist = np.roll(hist, -1, 0); hist[-1] = obs
            if te or tr:
                mi = int(rng.integers(0, len(lib.motions)))
                t0 = int(rng.integers(0, max(1, len(lib.motions[mi]) - 600)))
                obs = env.reset(motion_id=mi, frame_idx=t0)
                hist = np.repeat(obs[None], spec.HISTORY, 0)
    return np.stack(O).astype(np.float32), np.stack(G).astype(np.float32), np.stack(A).astype(np.float32)


def teacher_labels(O, G, ckpt):
    ck = torch.load(ROOT / ckpt, map_location="cpu", weights_only=False)
    t = Pol(); t.load_state_dict(ck["policy"]); t.eval()
    out = np.empty_like(O[:, :29]) if False else None
    acts = []
    with torch.no_grad():
        for i in range(0, len(O), 2048):
            a = t(torch.as_tensor(G[i:i+2048], dtype=torch.float32),
                  torch.as_tensor(O[i:i+2048], dtype=torch.float32)).numpy()
            acts.append(a)
    return np.concatenate(acts, 0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--motion-dir", default="assets/motions_aligned")
    ap.add_argument("--data", default="runs/bc_teacher/data.npz")
    ap.add_argument("--bc", default="runs/bc_teacher/policy.pt")
    ap.add_argument("--out", default="runs/bc_teacher/policy_dagger.pt")
    args = ap.parse_args()
    pol = BC().cuda(); pol.load_state_dict(torch.load(ROOT / args.bc, map_location="cuda")["policy"]); pol.eval()
    O, G, _ = collect_bc(args.steps, pol, args.motion_dir)
    A = teacher_labels(O, G, "runs/pretrained/policy.pt")
    d = np.load(ROOT / args.data)
    O = np.concatenate([d["O"], O]); G = np.concatenate([d["G"], G]); A = np.concatenate([d["A"], A])
    print("dagger data", O.shape)
    pol2 = train(O, G, A, args.iters)
    torch.save({"policy": pol2.state_dict()}, ROOT / args.out)
    print("saved", args.out)
    evaluate(pol2, args.motion_dir)


if __name__ == "__main__":
    main()
