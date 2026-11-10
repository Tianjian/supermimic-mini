#!/usr/bin/env python3
"""JAX port of the pretrained SuperMimic teacher (680 goal / 930 obs)."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, jax, jax.numpy as jnp, torch
from smini import motion, spec
from smini.env import TrackingEnv, EnvCfg
from eval_pretrained import goal680, obs930

XML = str(ROOT.parent / "assets/mujoco_manipulation" if False else ROOT.parent / "assets/mujoco_menagerie/unitree_g1/scene_mjx.xml")


def load_params(ckpt):
    ck = torch.load(ROOT / ckpt, map_location="cpu", weights_only=False)
    sd = {k: v.numpy() for k, v in ck["policy"].items()}

    def layers(prefix):
        ps = []
        for i in (0, 2, 4, 6, 8):
            W = sd[f"{prefix}.net.{i}.weight"].T  # (in, out)
            b = sd[f"{prefix}.net.{i}.bias"]
            ps.append((jnp.asarray(W), jnp.asarray(b)))
        return ps

    enc = layers("encoders.motion.mlp")
    dec = layers("decoder.mlp")
    return enc, dec, jnp.asarray(sd["encoders.motion.scale"]), jnp.asarray(sd["decoder.scale"])


def mlp(ps, x):
    for W, b in ps[:-1]:
        x = jax.nn.silu(x @ W + b)
    W, b = ps[-1]
    return x @ W + b


@jax.jit
def act(params, goal, obs):
    enc, dec, es, ds = params
    z = jnp.tanh(mlp(enc, goal * es))
    return mlp(dec, jnp.concatenate([z, obs * ds], -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/pretrained/policy.pt")
    ap.add_argument("--motion-dir", default="assets/motions_trim")
    ap.add_argument("--clips", type=int, default=5)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    params = load_params(args.ckpt)
    print("loaded params", [tuple(p[0].shape) for p in params[0]], [tuple(p[0].shape) for p in params[1]])
    if args.parity:
        from eval_pretrained import Pol
        ck = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
        tp = Pol(); tp.load_state_dict(ck["policy"]); tp.eval()
        rng = np.random.default_rng(0)
        g = rng.normal(0, 1, (4, 680)).astype(np.float32)
        o = rng.normal(0, 1, (4, 930)).astype(np.float32)
        with torch.no_grad():
            ref = tp(torch.as_tensor(g), torch.as_tensor(o)).numpy()
        out = np.asarray(act(params, jnp.asarray(g), jnp.asarray(o)))
        print("parity max|diff|", float(np.abs(out - ref).max()), "mean", float(np.abs(out - ref).mean()))
    if args.eval:
        lib = motion.MotionLibrary.from_dir(ROOT / args.motion_dir)
        lib.motions = lib.motions[: args.clips]
        env = TrackingEnv(XML, lib, EnvCfg(obs_noise=False, reset_noise=False,
                                           domain_randomization=False, action_noise=0), seed=0)
        rets = []
        for ci in range(args.clips):
            obs = env.reset(motion_id=ci, frame_idx=0)
            hist = np.repeat(obs[None], spec.HISTORY, 0)
            ret = 0.0; st = 0
            for t in range(args.steps):
                g = goal680(lib.motions[ci].qpos, lib.motions[ci].qvel, env.frame_idx)
                a = np.asarray(act(params, jnp.asarray(g[None]), jnp.asarray(obs930(hist)[None])))[0]
                obs, r, te, tr, info = env.step(a)
                hist = np.roll(hist, -1, 0); hist[-1] = obs
                ret += r; st += 1
                if te or tr:
                    break
            norm = ret / (0.1 * (args.steps - 1)); rets.append(norm)
            print(f"{lib.motions[ci].name:18s} steps {st:3d} ret {ret:6.2f} norm {norm:.4f}")
        print("mean", float(np.mean(rets)))


if __name__ == "__main__":
    main()
