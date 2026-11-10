#!/usr/bin/env python3
"""Evaluate the public SuperMimic getup checkpoint in the local CPU env.

Checkpoint: 93-d obs (term-major 10x history) + 680-d goal (10 x 68 anchor-
relative features) + 29 actions, 50 Hz.  Our env exposes the same 93-d obs
(per-frame order), so we re-flatten it term-major and build the 680-d goal.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn as nn
from smini import motion, spec
from smini.env import TrackingEnv, EnvCfg

XML = str(ROOT.parent / "assets/mujoco_menagerie/unitree_g1/scene_mjx.xml")


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP(680, (2048, 1024, 512, 256), 32)
        self.register_buffer("scale", torch.ones(680))

    def forward(self, g):
        return torch.tanh(self.mlp(g * self.scale))


class Dec(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP(962, (2048, 1024, 512, 256), 29)
        self.register_buffer("scale", torch.ones(930))

    def forward(self, z, obs):
        return self.mlp(torch.cat([z, obs * self.scale], -1))


class Pol(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleDict({"motion": Enc()})
        self.decoder = Dec()

    def forward(self, goal680, obs930):
        return self.decoder(self.encoders["motion"](goal680), obs930)


def goal680(qpos, qvel, t, k=10):
    """10 x 68 anchor-relative features at frames t+1..t+k (clamped)."""
    out = []
    for j in range(k):
        tt = min(t + 1 + j, len(qpos) - 1)
        q = qpos[tt]; v = qvel[tt]
        R = spec.quat_to_mat(q[3:7])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        Ry = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        out.append(np.concatenate([
            q[7:36] - spec.DEFAULT_Q,          # 29
            v[6:35],                            # 29
            -R[2, :],                           # 3 projected gravity
            R.T @ v[3:6],                       # 3 angvel body
            Ry.T @ v[0:3],                      # 3 linvel heading
            q[2:3],                             # 1 height
        ]))
    return np.concatenate(out)


def obs930(hist93):
    """hist93 (10,93) per-frame -> 930 term-major."""
    terms = [(0, 29), (29, 58), (58, 61), (61, 64), (64, 93)]
    return np.concatenate([hist93[:, a:b].reshape(-1) for a, b in terms])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/pretrained/policy.pt")
    ap.add_argument("--motion-dir", default="assets/motions")
    ap.add_argument("--clips", type=int, default=5)
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()
    ck = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    pol = Pol(); pol.load_state_dict(ck["policy"], strict=True)
    print("loaded strict, iter", ck.get("iteration"))
    pol.eval()

    lib = motion.MotionLibrary.from_dir(ROOT / args.motion_dir)
    lib.motions = lib.motions[: args.clips]
    env = TrackingEnv(XML, lib, EnvCfg(obs_noise=False, reset_noise=False,
                                       domain_randomization=False, action_noise=0), seed=0)
    rets = []
    with torch.no_grad():
        for ci in range(args.clips):
            obs = env.reset(motion_id=ci, frame_idx=0)
            hist = np.repeat(obs[None], spec.HISTORY, 0)
            ret = 0.0; steps = 0
            for t in range(args.steps):
                g = goal680(lib.motions[ci].qpos, lib.motions[ci].qvel, env.frame_idx)
                a = pol(torch.as_tensor(g[None], dtype=torch.float32),
                        torch.as_tensor(obs930(hist)[None], dtype=torch.float32))[0].numpy()
                obs, r, te, tr, info = env.step(a)
                hist = np.roll(hist, -1, 0); hist[-1] = obs
                ret += r; steps += 1
                if te or tr:
                    break
            norm = ret / (0.1 * (args.steps - 1))
            rets.append(norm)
            print(f"{lib.motions[ci].name:18s} steps {steps:3d} ret {ret:6.2f} norm {norm:.4f}")
    print("mean", float(np.mean(rets)))


if __name__ == "__main__":
    main()
