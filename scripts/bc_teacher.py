#!/usr/bin/env python3
"""DAgger/BC from the public pretrained SuperMimic tracker into a local MLP."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, torch, torch.nn as nn
from smini import motion, spec
from smini.env import TrackingEnv, EnvCfg
from eval_pretrained import Pol, goal680, obs930

XML = str(ROOT.parent / "assets/mujoco_menagerie/unitree_g1/scene_mjx.xml")


def mlp(i, h, o):
    layers = []; d = i
    for x in h:
        layers += [nn.Linear(d, x), nn.SiLU()]; d = x
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class BC(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(1610, (1024, 512, 256), 29)
        self.register_buffer("mean", torch.zeros(1610))
        self.register_buffer("std", torch.ones(1610))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def collect(n, motion_dir, ckpt, out, seed=0):
    ck = torch.load(ROOT / ckpt, map_location="cpu", weights_only=False)
    teacher = Pol(); teacher.load_state_dict(ck["policy"]); teacher.eval()
    lib = motion.MotionLibrary.from_dir(ROOT / motion_dir)
    env = TrackingEnv(XML, lib, EnvCfg(obs_noise=False, reset_noise=False,
                                       domain_randomization=False, action_noise=0.1), seed=seed)
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
            a = teacher(torch.as_tensor(g[None], dtype=torch.float32),
                        torch.as_tensor(o930[None], dtype=torch.float32))[0].numpy()
            O.append(o930); G.append(g); A.append(a)
            obs, r, te, tr, info = env.step(a)
            hist = np.roll(hist, -1, 0); hist[-1] = obs
            if te or tr:
                mi = int(rng.integers(0, len(lib.motions)))
                t0 = int(rng.integers(0, max(1, len(lib.motions[mi]) - 600)))
                obs = env.reset(motion_id=mi, frame_idx=t0)
                hist = np.repeat(obs[None], spec.HISTORY, 0)
    O = np.stack(O).astype(np.float32); G = np.stack(G).astype(np.float32); A = np.stack(A).astype(np.float32)
    np.savez_compressed(ROOT / out, O=O, G=G, A=A)
    print("collected", O.shape, "action std", A.std(0).mean())
    return O, G, A


def train(O, G, A, iters=4000, batch=1024):
    X = np.concatenate([O, G], -1)
    mean = X.mean(0); std = np.maximum(X.std(0), 1e-3)
    X = (X - mean) / std
    Xt = torch.as_tensor(X, dtype=torch.float32, device="cuda")
    Yt = torch.as_tensor(A, dtype=torch.float32, device="cuda")
    pol = BC().cuda()
    pol.mean.copy_(torch.as_tensor(mean, dtype=torch.float32)); pol.std.copy_(torch.as_tensor(std, dtype=torch.float32))
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    n = len(Xt); rng = np.random.default_rng(0)
    for it in range(1, iters + 1):
        idx = torch.as_tensor(rng.integers(0, n, batch), device="cuda")
        pred = pol(Xt[idx])
        loss = ((pred - Yt[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if it % 500 == 0:
            print(f"[{it}] mse {float(loss):.4f}", flush=True)
    return pol


def evaluate(pol, motion_dir, clips=5, steps=500):
    lib = motion.MotionLibrary.from_dir(ROOT / motion_dir)
    lib.motions = lib.motions[:clips]
    env = TrackingEnv(XML, lib, EnvCfg(obs_noise=False, reset_noise=False,
                                       domain_randomization=False, action_noise=0), seed=0)
    rets = []
    with torch.no_grad():
        for ci in range(clips):
            obs = env.reset(motion_id=ci, frame_idx=0)
            hist = np.repeat(obs[None], spec.HISTORY, 0)
            ret = 0.0; st = 0
            for t in range(steps):
                g = goal680(lib.motions[ci].qpos, lib.motions[ci].qvel, env.frame_idx)
                x = np.concatenate([obs930(hist), g])
                a = pol(torch.as_tensor(x[None], dtype=torch.float32, device="cuda"))[0].cpu().numpy()
                obs, r, te, tr, info = env.step(a)
                hist = np.roll(hist, -1, 0); hist[-1] = obs
                ret += r; st += 1
                if te or tr:
                    break
            norm = ret / (0.1 * (steps - 1)); rets.append(norm)
            print(f"{lib.motions[ci].name:18s} steps {st:3d} ret {ret:6.2f} norm {norm:.4f}")
    print("mean", float(np.mean(rets)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--motion-dir", default="assets/motions_aligned")
    ap.add_argument("--ckpt", default="runs/pretrained/policy.pt")
    ap.add_argument("--data", default="runs/bc_teacher/data.npz")
    ap.add_argument("--out", default="runs/bc_teacher/policy.pt")
    args = ap.parse_args()
    (ROOT / args.data).parent.mkdir(parents=True, exist_ok=True)
    if (ROOT / args.data).exists():
        d = np.load(ROOT / args.data); O, G, A = d["O"], d["G"], d["A"]; print("loaded", O.shape)
    else:
        O, G, A = collect(args.steps, args.motion_dir, args.ckpt, args.data)
    pol = train(O, G, A, args.iters)
    torch.save({"policy": pol.state_dict()}, ROOT / args.out)
    print("saved", args.out)
    evaluate(pol, args.motion_dir)


if __name__ == "__main__":
    main()
