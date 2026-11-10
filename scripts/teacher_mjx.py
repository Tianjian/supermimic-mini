#!/usr/bin/env python3
"""MJX-side parity: run the JAX teacher on 5 trimmed clips and report survival."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, jax, jax.numpy as jnp, mujoco
from mujoco import mjx
from smini import motion, spec
from eval_pretrained import goal680
from teacher_jax import load_params, act

XML = str(ROOT.parent / "assets/mujoco_menagerie/unitree_g1/scene_mjx.xml")
HIST, NQ, NV = spec.HISTORY, 36, 35


def main():
    lib = motion.MotionLibrary.from_dir(ROOT / "assets/motions_trim")
    lib.motions = lib.motions[:5]
    n = len(lib.motions)
    goals = [jnp.asarray(np.stack([goal680(m.qpos, m.qvel, t) for t in range(len(m))])) for m in lib.motions]
    m = mujoco.MjModel.from_xml_path(XML); m.opt.timestep = 0.005
    mx = mjx.put_model(m); d0 = mjx.make_data(mx)
    B = n
    d = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), d0)
    def ctrl(mx, dd):
        return jax.lax.fori_loop(0, 4, lambda _, x: mjx.step(mx, x), dd)
    step = jax.jit(jax.vmap(ctrl, in_axes=(None, 0)))
    q0 = jnp.asarray(spec.DEFAULT_Q, jnp.float32)
    # reset at frame 0
    qp = jnp.asarray(np.stack([mm.qpos[0] for mm in lib.motions]))
    qv = jnp.asarray(np.stack([mm.qvel[0] for mm in lib.motions]))
    d = d.replace(qpos=qp, qvel=qv, ctrl=qp[:, 7:36])
    def obs_of(qpos, qvel, prev):
        qrel = qpos[:, 7:36] - q0
        return jnp.concatenate([qrel, qvel[:, 6:35], qvel[:, 3:6],
                                -jnp.stack([jnp.stack([1.0, 0, 0])[0], 0, 0]) * 0 + (-qmat(qpos[:, 3:7])[..., 2, :]), prev], -1)
    from ppo_mjx import qmat
    def obs_all(qpos, qvel, prev):
        return jnp.concatenate([qpos[:, 7:36] - q0, qvel[:, 6:35], qvel[:, 3:6],
                                -qmat(qpos[:, 3:7])[..., 2, :], prev], -1)
    prev = jnp.zeros((B, spec.ACTION_DIM))
    obs = obs_all(d.qpos, d.qvel, prev)
    hist = jnp.repeat(obs[:, None, :], HIST, 1)
    frame = np.zeros(B, np.int32); term = np.zeros(B, bool)
    params = load_params("runs/pretrained/policy.pt")
    for t in range(500):
        g = jnp.asarray(np.stack([goals[i][min(int(frame[i]), len(goals[i]) - 1)] for i in range(B)]))
        # term-major flatten
        parts = [hist[:, :, 0:29].reshape(B, -1), hist[:, :, 29:58].reshape(B, -1),
                 hist[:, :, 58:61].reshape(B, -1), hist[:, :, 61:64].reshape(B, -1),
                 hist[:, :, 64:93].reshape(B, -1)]
        obs930 = jnp.concatenate(parts, -1)
        a = act(params, g, obs930)
        d = d.replace(ctrl=jnp.clip(q0 + jnp.asarray(spec.ACTION_SCALE, jnp.float32) * a,
                                    jnp.asarray(m.actuator_ctrlrange[:, 0], jnp.float32),
                                    jnp.asarray(m.actuator_ctrlrange[:, 1], jnp.float32)))
        d = step(mx, d)
        frame = np.minimum(frame + 1, np.array([len(mm) - 1 for mm in lib.motions]))
        obs = obs_all(d.qpos, d.qvel, a)
        hist = jnp.concatenate([hist[:, 1:], obs[:, None, :]], 1)
        # termination (height/tilt vs ref)
        for i in range(B):
            if term[i]: continue
            rp = lib.motions[i].qpos[min(frame[i], len(lib.motions[i]) - 1)]
            q = np.asarray(d.qpos[i]); r = rp
            dh = abs(q[2] - r[2])
            gq = spec.projected_gravity(q[3:7]); gr = spec.projected_gravity(r[3:7])
            if dh > 0.25 or abs(gq[2] - gr[2]) > 0.8: term[i] = True
    print("survived steps", [500 if not term[i] else "fallen" for i in range(B)])


if __name__ == "__main__":
    main()
