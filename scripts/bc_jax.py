#!/usr/bin/env python3
"""Distill the torch soup policy into the JAX actor used by ppo_mjx."""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import jax, jax.numpy as jnp, optax
import torch
from smini import motion, spec
from smini.env import TrackingEnv, EnvCfg
from smini.policy import Policy
from ppo_mjx import build_tables, norm_stats, init_mlp, mlp

XML = str(ROOT.parent / "assets/mujoco_menagerie/unitree_g1/scene_mjx.xml")


def collect(n_steps, motion_dir, policy_path, seed=0):
    lib = motion.MotionLibrary.from_dir(ROOT / motion_dir)
    env = TrackingEnv(XML, lib, EnvCfg(obs_noise=False, reset_noise=False,
                                       domain_randomization=False, action_noise=0), seed=seed)
    pol = Policy().cuda()
    pol.load_state_dict(torch.load(ROOT / policy_path, map_location="cuda")["policy"]); pol.eval()
    rng = np.random.default_rng(seed)
    H, G, A = [], [], []
    obs = env.reset(motion_id=0, frame_idx=0)
    hist = np.repeat(obs[None], spec.HISTORY, 0)
    n = 0
    while n < n_steps:
        g = env.goal()
        with torch.no_grad():
            h = torch.as_tensor(hist[None], dtype=torch.float32, device="cuda")
            gt = torch.as_tensor(g[None], dtype=torch.float32, device="cuda")
            a = pol(gt, h)[0].cpu().numpy()
        H.append(hist.copy()); G.append(g); A.append(a); n += 1
        obs, r, te, tr, info = env.step(a)
        hist = np.roll(hist, -1, 0); hist[-1] = obs
        if te or tr:
            mi = int(rng.integers(0, len(lib.motions)))
            t0 = int(rng.integers(0, max(1, len(lib.motions[mi]) - 600)))
            obs = env.reset(motion_id=mi, frame_idx=t0)
            hist = np.repeat(obs[None], spec.HISTORY, 0)
    return (np.stack(H).astype(np.float32), np.stack(G).astype(np.float32),
            np.stack(A).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--train-iters", type=int, default=3000)
    ap.add_argument("--motion-dir", default="assets/motions_aligned")
    ap.add_argument("--policy", default="runs/soup_best.pt")
    ap.add_argument("--out", default="runs/ppo_mjx/actor_bc.pkl")
    args = ap.parse_args()
    data_path = ROOT / "runs/ppo_mjx/bc_data.npz"
    if data_path.exists():
        d = np.load(data_path); H, G, A = d["H"], d["G"], d["A"]
        print("loaded", H.shape)
    else:
        H, G, A = collect(args.steps, args.motion_dir, args.policy)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(data_path, H=H, G=G, A=A)
        print("collected", H.shape)
    qpos_tab, qvel_tab, goal_tab, lens, lib = build_tables(args.motion_dir)
    om, os_, gm, gs_ = norm_stats(qpos_tab, qvel_tab, goal_tab)
    X = np.concatenate([((H - om) / os_).reshape(len(H), -1), (G - gm) / gs_], -1).astype(np.float32)
    print("features", X.shape, "action std", A.std(0).mean())
    key = jax.random.PRNGKey(0)
    ps = init_mlp(key, [1660, 256, 128, 29], last_scale=1.0)
    opt = optax.adam(1e-3); opt_state = opt.init(ps)
    Xj = jnp.asarray(X); Aj = jnp.asarray(A)
    n = len(Xj)

    @jax.jit
    def step(ps, opt_state, x, y):
        def loss_fn(p):
            pred = mlp(p, x)
            return jnp.mean((pred - y) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(ps)
        upd, opt_state = opt.update(grads, opt_state, ps)
        return optax.apply_updates(ps, upd), opt_state, loss

    rng = np.random.default_rng(0)
    for it in range(1, args.train_iters + 1):
        idx = rng.integers(0, n, 1024)
        ps, opt_state, loss = step(ps, opt_state, Xj[idx], Aj[idx])
        if it % 200 == 0:
            print(f"[{it}] mse {float(loss):.4f}", flush=True)
    out = [(np.asarray(W), np.asarray(b)) for W, b in ps]
    with open(ROOT / args.out, "wb") as f:
        pickle.dump(out, f)
    print("saved", args.out)


if __name__ == "__main__":
    main()
