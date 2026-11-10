#!/usr/bin/env python3
"""Behaviour-cloning warm start: PD tracking actions -> policy.

Collects clean PD-tracking rollouts (no action noise), then supervises the
policy to map (goal, obs history) -> (q_ref - q0)/action_scale.  The resulting
checkpoint can initialise the BPTT loop (`init_policy` in the config).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from smini import motion, spec
from smini.env import EnvCfg
from smini.vecenv import VecEnv
from smini.replay import ReplayBuffer
from smini.norm import compute_stats
from smini.policy import Policy
from smini.train import Collector, TrainCfg, evaluate, _episode_arrays

XML = str(ROOT.parent / "assets/mujoco_menagerie/unitree_g1/scene_mjx.xml")


def collect_pd(lib, n_transitions: int, envs: int = 8, seed: int = 0):
    vec = VecEnv(XML, lib, EnvCfg(action_noise=0.0, seed=seed), num_envs=envs)
    buf = ReplayBuffer(200_000)
    col = Collector(vec, lib, buf, TrainCfg(), np.random.default_rng(seed))
    t0 = time.time()
    while len(buf) < n_transitions:
        col.collect(16, col.pd_action)
    print(f"collected {len(buf)} transitions in {time.time()-t0:.1f}s")
    return buf


def bc_train(policy, buf, stats, steps: int, batch: int, lr: float, device: str):
    # flatten valid starts (>= HISTORY observations, and a next target action)
    starts, ep_idx = [], []
    for k, ep in enumerate(buf.episodes):
        T = ep["T"]
        if T >= spec.HISTORY + 1:
            starts.extend(range(spec.HISTORY - 1, T))
            ep_idx.extend([k] * (T - spec.HISTORY + 1))
    starts = np.asarray(starts)
    ep_idx = np.asarray(ep_idx)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(starts))
    n_val = max(1, len(starts) // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    print(f"bc starts: train {len(tr_idx)}, val {len(val_idx)}")
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    a_std = torch.as_tensor(stats["action"][1], device=device).clamp_min(0.1)
    g_mean = torch.as_tensor(stats["goal"][0], device=device)
    g_std = torch.as_tensor(stats["goal"][1], device=device)
    o_mean = torch.as_tensor(stats["obs"][0], device=device)
    o_std = torch.as_tensor(stats["obs"][1], device=device)

    def batch_from(idx):
        obs, goal, act = [], [], []
        for j in idx:
            ep = buf.episodes[ep_idx[j]]
            i = int(starts[j])
            obs.append(ep["obs"][i - spec.HISTORY + 1:i + 1])
            goal.append(ep["goal"][i])
            act.append(ep["action"][i])          # clean PD action (no noise)
        obs = torch.as_tensor(np.stack(obs), dtype=torch.float32, device=device)
        goal = torch.as_tensor(np.stack(goal), dtype=torch.float32, device=device)
        act = torch.as_tensor(np.stack(act), dtype=torch.float32, device=device)
        return obs, goal, act

    def loss_fn(obs, goal, act):
        pred = policy(goal, obs)
        return F.mse_loss((pred - act) / a_std, torch.zeros_like(act))

    for it in range(1, steps + 1):
        sel = rng.choice(tr_idx, size=min(batch, len(tr_idx)), replace=False)
        obs, goal, act = batch_from(sel)
        loss = loss_fn(obs, goal, act)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if it % 200 == 0:
            with torch.no_grad():
                vs = rng.choice(val_idx, size=min(256, len(val_idx)), replace=False)
                vo, vg, va = batch_from(vs)
                vl = loss_fn(vo, vg, va)
            print(f"[{it:4d}] train {loss.item():.4f} val {vl.item():.4f}", flush=True)
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transitions", type=int, default=30000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=str(ROOT / "runs/bc_policy.pt"))
    ap.add_argument("--eval-steps", type=int, default=500)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    lib = motion.MotionLibrary.from_dir(ROOT / "assets/motions")
    buf = collect_pd(lib, args.transitions)
    stats = compute_stats(buf)

    policy = Policy().to(dev)
    a_first = np.stack([
        (m.qpos[0, 7:36] - spec.DEFAULT_Q) / spec.ACTION_SCALE for m in lib.motions
    ])
    policy.set_action_bias(np.clip(a_first.mean(axis=0), -1.5, 1.5))
    policy.set_norm(*stats["goal"], *stats["obs"])
    bc_train(policy, buf, stats, args.steps, args.batch, args.lr, dev)

    cfg = TrainCfg(xml_path=XML, motion_dir=str(ROOT / "assets/motions"),
                   eval_steps=args.eval_steps, eval_clips=5, seed=0)
    ev = evaluate(policy, lib, cfg)
    print(f"[bc eval] return {ev['return']:.4f} fail {ev['fail_rate']:.2f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": policy.state_dict(), "stats": {k: tuple(map(torch.as_tensor, v)) for k, v in stats.items()}},
               args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
