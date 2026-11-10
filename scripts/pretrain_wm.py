#!/usr/bin/env python3
"""M2 check: collect warm-start data and fit only the world model."""
from __future__ import annotations
import argparse, sys, time
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
from smini.world_model import WorldModel
from smini.train import Collector

XML = str(ROOT.parent / "assets/mujoco_menagerie/unitree_g1/scene_mjx.xml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--transitions", type=int, default=20000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--random-frac", type=float, default=0.2)
    args = ap.parse_args()
    dev = "cuda"
    lib = motion.MotionLibrary.from_dir(ROOT / "assets/motions")
    vec = VecEnv(XML, lib, EnvCfg(episode_s=10.0, seed=0), num_envs=args.envs)
    buf = ReplayBuffer(max(100_000, args.transitions))
    from smini.train import TrainCfg, _episode_arrays
    cfg = TrainCfg()
    col = Collector(vec, lib, buf, cfg, np.random.default_rng(0))
    t0 = time.time()
    rng = np.random.default_rng(0)

    def act():
        a = col.pd_action()
        if args.random_frac > 0:
            mask = rng.random((col.n, 1)) < args.random_frac
            a = np.where(mask, rng.uniform(-3, 3, (col.n, 29)), a)
        return a

    while len(buf) < args.transitions:
        col.collect(16, act)
    print(f"collected {len(buf)} transitions in {time.time()-t0:.1f}s")
    stats = compute_stats(buf)
    wm = WorldModel(hidden=(args.hidden,) * 3).to(dev)
    wm.set_norm(*stats["state"], *stats["action"], *stats["xi"], *stats["goal_cur"])
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    wm.train()
    for it in range(1, args.iters + 1):
        tot = 0.0
        for _ in range(args.batches):
            b = buf.sample_windows(args.batch, 8, device=dev)
            s = wm.norm_state(b["s"]); tgt = wm.norm_state(b["s_next"]); loss = 0.0
            for j in range(8):
                s = wm.forward_norm(s, b["a"][:, j], b["xi"])
                loss = loss + F.mse_loss(s[..., :73], tgt[:, j, :73])
                r, dl = wm.heads(wm.denorm_state(s), b["a"][:, j], b["goal_cur"][:, j])
                loss = loss + F.mse_loss(r, b["r"][:, j] * spec.REWARD_SCALE) \
                       + 0.1 * F.binary_cross_entropy_with_logits(dl, b["d"][:, j])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0); opt.step()
            tot += loss.item()
        if it % 25 == 0:
            with torch.no_grad():
                b = buf.sample_windows(256, 8, device=dev)
                s = wm.norm_state(b["s"])
                tgt = wm.norm_state(b["s_next"])
                for j in range(8):
                    s = wm.forward_norm(s, b["a"][:, j], b["xi"])
                err = F.mse_loss(s[..., :73], tgt[:, 7, :73]).item()
                err1 = F.mse_loss(wm.forward_norm(wm.norm_state(b["s"]), b["a"][:, 0], b["xi"])[..., :73], tgt[:, 0, :73]).item()
                persist = F.mse_loss(wm.norm_state(b["s"])[..., :73], tgt[:, 0, :73]).item()
            print(f"[{it:4d}] loss {tot/args.batches:.4f}  nrmse1 {err1:.4f} nrmse8 {err:.4f} "
                  f"persist {persist:.4f} ({time.time()-t0:.0f}s)", flush=True)
    torch.save({"wm": wm.state_dict(), "stats": {k: tuple(map(torch.as_tensor, v)) for k, v in stats.items()}},
               ROOT / "runs/wm_pretrain.pt")


if __name__ == "__main__":
    main()
