#!/usr/bin/env python3
"""PPO in MJX warm-started from the frozen SuperMimic teacher.

a = a_teacher(goal, obs930) + residual(features); only the residual, log_std
and critic are trained.  An L2/BC anchor keeps the residual close to the
teacher (no-erosion gate).
"""
from __future__ import annotations
import argparse, os, pickle, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, jax, jax.numpy as jnp, optax, mujoco
from mujoco import mjx
from smini import motion, spec
from eval_pretrained import goal680
from teacher_jax import load_params, act as teacher_act

XML = str(ROOT.parent / "assets/mujoco_menagerie/unitree_g1/scene_mjx.xml")
DT, HIST, NU = 0.02, spec.HISTORY, spec.ACTION_DIM
Q0 = jnp.asarray(spec.DEFAULT_Q, jnp.float32); ASC = jnp.asarray(spec.ACTION_SCALE, jnp.float32)
MASK = jnp.asarray(spec.JOINT_REWARD_MASK, jnp.float32); MDEN = MASK.sum()


def qmat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = w*w + x*x + y*y + z*z; s = 2.0 / jnp.maximum(n, 1e-8)
    wx, wy, wz = s*w*x, s*w*y, s*w*z; xx, xy, xz = s*x*x, s*x*y, s*x*z
    yy, yz, zz = s*y*y, s*y*z, s*z*z
    return jnp.stack([jnp.stack([1-(yy+zz), xy-wz, xz+wy], -1),
                      jnp.stack([xy+wz, 1-(xx+zz), yz-wx], -1),
                      jnp.stack([xz-wy, yz+wx, 1-(xx+yy)], -1)], -2)


def apply_scales(ps, x):
    for W, b in ps[:-1]:
        x = jax.nn.silu(x @ W + b)
    W, b = ps[-1]
    return x @ W + b


def init_mlp(key, sizes, zero_last=False):
    ps = []
    for i in range(len(sizes)-1):
        key, sub = jax.random.split(key)
        W = jax.random.normal(sub, (sizes[i], sizes[i+1])) * jnp.sqrt(2.0/sizes[i])
        if zero_last and i == len(sizes)-2:
            W = jnp.zeros_like(W)
        ps.append((W, jnp.zeros(sizes[i+1])))
    return ps


def obs930(hist):
    B = hist.shape[0]
    return jnp.concatenate([hist[:, :, 0:29].reshape(B, -1), hist[:, :, 29:58].reshape(B, -1),
                            hist[:, :, 58:61].reshape(B, -1), hist[:, :, 61:64].reshape(B, -1),
                            hist[:, :, 64:93].reshape(B, -1)], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--std-init", type=float, default=-1.0)
    ap.add_argument("--bc-coef", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--motion-dir", default="assets/motions_trim")
    ap.add_argument("--out", default="runs/ppo_teacher")
    ap.add_argument("--init-residual", default="")
    ap.add_argument("--res1", type=int, default=256)
    ap.add_argument("--res2", type=int, default=128)
    args = ap.parse_args()
    B = args.envs
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    lib = motion.MotionLibrary.from_dir(ROOT / args.motion_dir)
    lib.motions = lib.motions[:5]
    nclips = len(lib.motions)
    Tmax = max(len(m) for m in lib.motions)
    qpos = np.zeros((nclips, Tmax, 36), np.float32); qvel = np.zeros((nclips, Tmax, 35), np.float32)
    goals = np.zeros((nclips, Tmax, 680), np.float32); lens = np.zeros(nclips, np.int32)
    for k, m in enumerate(lib.motions):
        T = len(m); lens[k] = T
        qpos[k, :T] = m.qpos; qvel[k, :T] = m.qvel
        qpos[k, T:] = m.qpos[-1]; qvel[k, T:] = m.qvel[-1]
        for t in range(T):
            goals[k, t] = goal680(m.qpos, m.qvel, t)
        goals[k, T:] = goals[k, T-1]
    qpos_t = jnp.asarray(qpos); qvel_t = jnp.asarray(qvel); goal_t = jnp.asarray(goals); lens_t = jnp.asarray(lens)
    m = mujoco.MjModel.from_xml_path(XML); m.opt.timestep = 0.005
    mx = mjx.put_model(m); d0 = mjx.make_data(mx)
    d = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (B,)+x.shape), d0)
    cmin = jnp.asarray(m.actuator_ctrlrange[:, 0], np.float32); cmax = jnp.asarray(m.actuator_ctrlrange[:, 1], np.float32)

    def ctrl(mx, dd):
        return jax.lax.fori_loop(0, 4, lambda _, x: mjx.step(mx, x), dd)
    step = jax.jit(jax.vmap(ctrl, in_axes=(None, 0)))
    tparams = load_params("runs/pretrained/policy.pt")
    tvm = jax.jit(jax.vmap(teacher_act, in_axes=(None, 0, 0)))
    key = jax.random.PRNGKey(0); key, k1, k2 = jax.random.split(key, 3)
    res = init_mlp(k1, [1610, args.res1, args.res2, NU], zero_last=True)
    if args.init_residual:
        with open(ROOT / args.init_residual, "rb") as fh:
            res = [(jnp.asarray(W), jnp.asarray(b)) for W, b in pickle.load(fh)]
        print("residual warm start", args.init_residual, flush=True)
    crit = init_mlp(k2, [1610, 512, 256, 1], zero_last=False)
    log_std = jnp.full((NU,), args.std_init)
    opt = optax.adam(args.lr); opt_state = opt.init((res, crit, log_std))

    def sample_reset(key, n):
        k1, k2 = jax.random.split(key)
        clip = jax.random.randint(k1, (n,), 0, nclips)
        t0 = jax.random.randint(k2, (n,), 0, jnp.maximum(1, lens_t[clip]-32))
        return clip, t0

    def reset_data(d, clip, t0, mask):
        qp = qpos_t[clip, t0]; qv = qvel_t[clip, t0]
        d = d.replace(qpos=jnp.where(mask[:, None], qp, d.qpos),
                      qvel=jnp.where(mask[:, None], qv, d.qvel),
                      ctrl=jnp.where(mask[:, None], qp[:, 7:36], d.ctrl))
        return d

    def reward_fn(d, clip, ref, a, prev):
        rp = qpos_t[clip, ref]; rv = qvel_t[clip, ref]
        root_pos = jnp.sum((d.qpos[:, 0:3]-rp[:, 0:3])**2, -1)
        root_ori = (2.0 - jnp.trace(qmat(rp[:, 3:7]).transpose(0, 2, 1) @ qmat(d.qpos[:, 3:7]), axis1=1, axis2=2)).clip(0.0)
        q = d.qpos[:, 7:36]; qd = d.qvel[:, 6:35]
        jpos = (((q-rp[:, 7:36])**2)*MASK).sum(-1)/MDEN
        jvel = (((qd-rv[:, 6:35])**2)*MASK).sum(-1)/MDEN
        smooth = ((ASC*(a-prev))**2).sum(-1)
        r = 0.5*jnp.exp(-root_pos/0.09) + 0.5*jnp.exp(-root_ori/0.16) + jnp.exp(-jpos/0.25) + jnp.exp(-jvel/1.0) - 0.1*smooth
        return r*DT, root_pos, root_ori

    def term_fn(d, clip, ref):
        rp = qpos_t[clip, ref]
        dh = jnp.abs(d.qpos[:, 2]-rp[:, 2])
        tilt = jnp.abs(-qmat(d.qpos[:, 3:7])[:, 2, 2] + qmat(rp[:, 3:7])[:, 2, 2])
        return (dh > 0.25) | (tilt > 0.8)

    def rollout(carry, _):
        d, hist, goal, frame, stepc, clip, prev, key = carry
        key, sub = jax.random.split(key)
        o930 = obs930(hist)
        feat = jnp.concatenate([o930, goal], -1)
        at = tvm(tparams, goal, o930)
        mean = at + apply_scales(res, feat)
        lg = jnp.clip(log_std, -3.0, 0.0); std = jnp.exp(lg)
        eps = jax.random.normal(sub, mean.shape)
        a = jnp.clip(mean + std*eps, -8.0, 8.0)
        logp = (-0.5*((a-mean)/std)**2 - lg - 0.5*jnp.log(2*jnp.pi)).sum(-1)
        v = apply_scales(crit, feat).squeeze(-1)
        d = d.replace(ctrl=jnp.clip(Q0 + ASC*a, cmin, cmax))
        d = step(mx, d)
        ref = jnp.minimum(frame+1, lens_t[clip]-1)
        r, _, _ = reward_fn(d, clip, ref, a, prev)
        term = term_fn(d, clip, ref)
        trunc = (stepc+1 >= 500) | (frame+12 >= lens_t[clip])
        done = term | trunc
        obs = jnp.concatenate([d.qpos[:, 7:36]-Q0, d.qvel[:, 6:35], d.qvel[:, 3:6],
                               -qmat(d.qpos[:, 3:7])[:, 2, :], a], -1)
        nxt = jnp.minimum(frame+1, lens_t[clip]-1)
        goal_next = goal_t[clip, nxt]
        key, sub2 = jax.random.split(key)
        nclip, nt0 = sample_reset(sub2, B)
        d = reset_data(d, nclip, nt0, done)
        obs_new = jnp.concatenate([d.qpos[:, 7:36]-Q0, d.qvel[:, 6:35], d.qvel[:, 3:6],
                                   -qmat(d.qpos[:, 3:7])[:, 2, :], a], -1)
        obs = jnp.where(done[:, None], obs_new, obs)
        hist = jnp.concatenate([hist[:, 1:], obs[:, None, :]], 1)
        hist = jnp.where(done[:, None, None], jnp.repeat(obs[:, None, :], HIST, 1), hist)
        goal = jnp.where(done[:, None], goal_t[nclip, nt0], goal_next)
        frame = jnp.where(done, nt0, nxt); stepc = jnp.where(done, 0, stepc+1)
        clip = jnp.where(done, nclip, clip); prev = jnp.where(done[:, None], 0.0, a)
        return (d, hist, goal, frame, stepc, clip, prev, key), (feat, at, a, logp, v, r, term)
    roll = jax.jit(lambda *a: jax.lax.scan(rollout, a, None, length=args.rollout))

    def loss_fn(params, batch, bc_coef):
        res, crit, log_std = params
        feat, at, a, logp_old, adv, ret = batch
        mean = at + apply_scales(res, feat)
        lg = jnp.clip(log_std, -3.0, 0.0); std = jnp.exp(lg)
        logp = (-0.5*((a-mean)/std)**2 - lg - 0.5*jnp.log(2*jnp.pi)).sum(-1)
        ratio = jnp.exp(logp-logp_old)
        s1 = ratio*adv; s2 = jnp.clip(ratio, 0.8, 1.2)*adv
        pi = -jnp.minimum(s1, s2).mean()
        v = apply_scales(crit, feat).squeeze(-1)
        vl = jnp.mean((v-ret)**2)
        bc = jnp.mean((mean-at)**2)
        return pi + 0.5*vl + bc_coef*bc, (pi, vl, bc)
    grad = jax.jit(jax.value_and_grad(loss_fn, argnums=0, has_aux=True))

    @jax.jit
    def update_step(params, opt_state, batch, bc_coef):
        (loss, aux), g = grad(params, batch, bc_coef)
        upd, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, aux

    # init
    key, sub = jax.random.split(key)
    clip, t0 = sample_reset(sub, B)
    d = reset_data(d, clip, t0, jnp.ones(B, bool))
    prev = jnp.zeros((B, NU)); obs = jnp.concatenate([d.qpos[:, 7:36]-Q0, d.qvel[:, 6:35], d.qvel[:, 3:6],
                                                      -qmat(d.qpos[:, 3:7])[:, 2, :], prev], -1)
    hist = jnp.repeat(obs[:, None, :], HIST, 1)
    goal = goal_t[clip, t0]; frame = t0; stepc = jnp.zeros(B, jnp.int32)
    params = (res, crit, log_std)
    rng = np.random.default_rng(0)
    for it in range(1, args.iters+1):
        t0_ = time.time()
        (d, hist, goal, frame, stepc, clip, prev, key), tr = roll(d, hist, goal, frame, stepc, clip, prev, key)
        feat, at, a, logp, v, r, term = tr
        # last value
        last_feat = jnp.concatenate([obs930(hist), goal], -1)
        last_v = apply_scales(params[1], last_feat).squeeze(-1)
        adv = jnp.zeros(B); advs = []
        for t in reversed(range(args.rollout)):
            nv = last_v if t == args.rollout-1 else v[t+1]
            delta = r[t] + 0.99*nv*(1-term[t]) - v[t]
            adv = delta + 0.99*0.95*(1-term[t])*adv
            advs.append(adv)
        advs = jnp.stack(advs[::-1]); ret = advs + v
        n = args.rollout*B
        f = feat.reshape(n, -1); atf = at.reshape(n, -1); af = a.reshape(n, -1)
        lpf = logp.reshape(n); advf = advs.reshape(n); retf = ret.reshape(n)
        advf = (advf-advf.mean())/(advf.std()+1e-8)
        perm = np.arange(n)
        bc_coef = args.bc_coef * (0.3 + 0.7 * (1.0 - (it - 1) / max(1, args.iters)))
        for ep in range(2):
            rng.shuffle(perm)
            for mb in range(8):
                sl = perm[mb*(n//8):(mb+1)*(n//8)]
                params, opt_state, aux = update_step(params, opt_state,
                    (f[sl], atf[sl], af[sl], lpf[sl], advf[sl], retf[sl]), bc_coef)
        if it % 10 == 0:
            print(f"[{it:3d}] r {float(r.mean()):.4f} pi {float(aux[0]):.3f} v {float(aux[1]):.3f} "
                  f"bc {float(aux[2]):.4f} {time.time()-t0_:.1f}s", flush=True)
        if it % args.eval_every == 0:
            with open(outdir / "ckpt.pkl", "wb") as fh:
                pickle.dump([(np.asarray(W), np.asarray(b)) for W, b in params[0]], fh)
            print("saved", outdir / "ckpt.pkl", flush=True)


if __name__ == "__main__":
    main()
