#!/usr/bin/env python3
"""Evaluate teacher + residual (env8, CPU MuJoCo)."""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, jax, jax.numpy as jnp
from smini import motion, spec
from smini.env import TrackingEnv, EnvCfg
from eval_pretrained import goal680, obs930
from teacher_jax import load_params, act
from ppo_teacher_mjx import apply_scales, init_mlp

XML = str(ROOT.parent / "assets/mujoco_menagerie/unitree_g1/scene_mjx.xml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--motion-dir", default="assets/motions_trim")
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()
    tparams = load_params("runs/pretrained/policy.pt")
    if args.ckpt:
        with open(ROOT / args.ckpt, "rb") as f:
            res = [(jnp.asarray(W), jnp.asarray(b)) for W, b in pickle.load(f)]
    else:
        res = init_mlp(jax.random.PRNGKey(0), [1610, 256, 128, spec.ACTION_DIM], zero_last=True)
    jact = jax.jit(lambda g, o, f: act(tparams, g, o) + apply_scales(res, f))
    lib = motion.MotionLibrary.from_dir(ROOT / args.motion_dir); lib.motions = lib.motions[:5]
    env = TrackingEnv(XML, lib, EnvCfg(obs_noise=False, reset_noise=False, domain_randomization=False, action_noise=0), seed=0)
    rets = []
    for ci in range(5):
        obs = env.reset(motion_id=ci, frame_idx=0)
        hist = np.repeat(obs[None], spec.HISTORY, 0)
        ret = 0.0; st = 0
        for t in range(args.steps):
            g = goal680(lib.motions[ci].qpos, lib.motions[ci].qvel, env.frame_idx)
            o = obs930(hist)
            feat = np.concatenate([o, g])
            a = np.asarray(jact(jnp.asarray(g[None]), jnp.asarray(o[None]), jnp.asarray(feat[None])))[0]
            obs, r, te, tr, info = env.step(a)
            hist = np.roll(hist, -1, 0); hist[-1] = obs
            ret += r; st += 1
            if te or tr:
                break
        norm = ret / (0.1*(args.steps-1)); rets.append(norm)
        print(f"{lib.motions[ci].name:18s} steps {st:3d} ret {ret:6.2f} norm {norm:.4f}")
    print("mean", float(np.mean(rets)))


if __name__ == "__main__":
    main()
