#!/usr/bin/env python3
"""Evaluate the JAX actor in the CPU MuJoCo env (8-term reward, same as soup eval)."""
import argparse, pickle, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, jax.numpy as jnp, jax
from smini import motion, spec
from smini.env import TrackingEnv, EnvCfg
from ppo_mjx import build_tables, norm_stats, mlp
XML = str(ROOT.parent / "assets/mujoco_menagerie/unitree_g1/scene_mjx.xml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", required=True)
    ap.add_argument("--motion-dir", default="assets/motions_aligned")
    ap.add_argument("--clips", type=int, default=5)
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()
    with open(args.actor, "rb") as f:
        ps = pickle.load(f)
    if isinstance(ps, tuple) and len(ps) == 3:
        ps = ps[0]
    ps = [(jnp.asarray(W), jnp.asarray(b)) for (W, b) in ps]
    qpos_tab, qvel_tab, goal_tab, lens, lib = build_tables(args.motion_dir)
    om, os_, gm, gs_ = norm_stats(qpos_tab, qvel_tab, goal_tab)
    om, os_, gm, gs_ = map(jnp.asarray, (om, os_, gm, gs_))

    def act(feat):
        return mlp(ps, feat)

    lib2 = motion.MotionLibrary.from_dir(ROOT / args.motion_dir)
    lib2.motions = lib2.motions[:args.clips]
    env = TrackingEnv(XML, lib2, EnvCfg(obs_noise=False, reset_noise=False,
                                        domain_randomization=False, action_noise=0), seed=0)
    act_j = jax.jit(act)
    rets = []
    for ci in range(args.clips):
        obs = env.reset(motion_id=ci, frame_idx=0)
        hist = np.repeat(obs[None], spec.HISTORY, 0)
        ret = 0.0; steps = 0
        for t in range(args.steps):
            g = env.goal()
            feat = np.concatenate([((hist - np.asarray(om)) / np.asarray(os_)).reshape(1, -1),
                                   (g[None] - np.asarray(gm)) / np.asarray(gs_)], -1)
            a = np.asarray(act_j(jnp.asarray(feat, jnp.float32)))[0]
            obs, r, te, tr, info = env.step(a)
            hist = np.roll(hist, -1, 0); hist[-1] = obs
            ret += r; steps += 1
            if te or tr:
                break
        norm = ret / (0.1 * (args.steps - 1))
        rets.append(norm)
        print(f"{lib2.motions[ci].name:18s} steps {steps:3d} ret {ret:6.2f} norm {norm:.4f}")
    print("mean", float(np.mean(rets)))


if __name__ == "__main__":
    main()
