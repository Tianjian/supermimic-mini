#!/usr/bin/env python3
"""GPU-parallel PPO on MJX (batched MuJoCo) for G1 motion tracking.

Actor input: 10x93 obs history + 730 goal.  Reward: root pose + joint pos/vel
(legs+waist) + action smoothness, matching the BPTT analytic reward.  The
reference arm channels are zero-aligned (assets/motions_aligned).
"""
from __future__ import annotations
import argparse, os, sys, time
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','0.88')
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import jax, jax.numpy as jnp
from jax import lax
import optax
import mujoco
from mujoco import mjx
from smini import motion, spec

XML = str(ROOT.parent / "assets/mujoco_menagerie/unitree_g1/scene_mjx.xml")
HIST, NQ, NV, NU = spec.HISTORY, 36, 35, spec.ACTION_DIM
DT = 0.02
Q0 = jnp.asarray(spec.DEFAULT_Q, jnp.float32)
ASCALE = jnp.asarray(spec.ACTION_SCALE, jnp.float32)
MASK = jnp.asarray(spec.JOINT_REWARD_MASK, jnp.float32)
MDEN = MASK.sum()


def build_tables(motion_dir, max_clips=5):
    lib = motion.MotionLibrary.from_dir(ROOT / motion_dir)
    lib.motions = lib.motions[:max_clips]
    Tm = max(len(m) for m in lib.motions); n = len(lib.motions)
    qpos = np.zeros((n, Tm, NQ), np.float32); qvel = np.zeros((n, Tm, NV), np.float32)
    goal = np.zeros((n, Tm, 730), np.float32); lens = np.zeros(n, np.int32)
    for k, m in enumerate(lib.motions):
        T = len(m); lens[k] = T
        qpos[k, :T] = m.qpos; qvel[k, :T] = m.qvel
        qpos[k, T:] = m.qpos[-1]; qvel[k, T:] = m.qvel[-1]
        for t in range(T):
            for j in range(10):
                tt = min(t + 1 + j, T - 1)
                goal[k, t, j * 73:(j + 1) * 73] = spec.pack_goal_frame(m.qpos[tt], m.qvel[tt])
        goal[k, T:] = goal[k, T - 1]
    return (jnp.asarray(qpos), jnp.asarray(qvel), jnp.asarray(goal),
            jnp.asarray(lens), lib)


def norm_stats(qpos_tab, qvel_tab, goal_tab):
    qp = np.asarray(qpos_tab).reshape(-1, NQ); qv = np.asarray(qvel_tab).reshape(-1, NV)
    obs_all = np.concatenate([qp[:, 7:36] - np.asarray(spec.DEFAULT_Q, np.float32),
                              qv[:, 6:35], qv[:, 3:6],
                              -np.stack([np.asarray(qmat(jnp.asarray(qp[:, 3:7])))[..., 2, i]
                                         for i in range(3)], -1),
                              np.zeros((len(qp), NU), np.float32)], -1)
    obs_mean = obs_all.mean(0); obs_std = np.maximum(obs_all.std(0), 1e-3)
    obs_mean[64:93] = 0.0; obs_std[64:93] = 1.0
    goal_mean = np.asarray(goal_tab).reshape(-1, 730).mean(0)
    goal_std = np.maximum(np.asarray(goal_tab).reshape(-1, 730).std(0), 1e-3)
    return obs_mean, obs_std, goal_mean, goal_std


def qmat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = w * w + x * x + y * y + z * z; s = 2.0 / jnp.maximum(n, 1e-8)
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return jnp.stack([
        jnp.stack([1 - (yy + zz), xy - wz, xz + wy], -1),
        jnp.stack([xy + wz, 1 - (xx + zz), yz - wx], -1),
        jnp.stack([xz - wy, yz + wx, 1 - (xx + yy)], -1)], -2)


def qerr(q1, q2):
    d = jnp.abs(jnp.sum(q1 * q2, -1)).clip(0, 1)
    return 2.0 * jnp.arccos(d)


def grav(q):
    return -qmat(q)[..., 2, :]


def obs_of(qpos, qvel, prev_a, ref_p=None, ref_R6=None):
    out = jnp.concatenate([qpos[:, 7:36] - Q0, qvel[:, 6:35], qvel[:, 3:6],
                           grav(qpos[:, 3:7]), prev_a], -1)
    if ref_p is not None:
        R = qmat(qpos[:, 3:7]); Rr = rot6d_to_mat(ref_R6)
        p_err = ref_p - qpos[:, 0:3]
        Rrel = jnp.swapaxes(Rr, -1, -2) @ R
        six = jnp.swapaxes(Rrel[..., :, :2], -1, -2).reshape(Rrel.shape[0], 6)
        out = jnp.concatenate([out, p_err, six], -1)
    return out


def rot6d_to_mat(v6):
    a = v6[..., 0:3]; b = v6[..., 3:6]
    a = a / jnp.maximum(jnp.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    b = b - a * jnp.sum(a * b, -1, keepdims=True)
    b = b / jnp.maximum(jnp.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
    c = jnp.cross(a, b)
    return jnp.stack([a, b, c], -1)


def init_mlp(key, sizes, last_scale=0.01):
    ps = []
    for i in range(len(sizes) - 1):
        key, sub = jax.random.split(key)
        W = jax.random.normal(sub, (sizes[i], sizes[i + 1])) * jnp.sqrt(2.0 / sizes[i])
        if i == len(sizes) - 2:
            W = W * last_scale
        ps.append((W, jnp.zeros(sizes[i + 1])))
    return ps


def mlp(ps, x):
    for W, b in ps[:-1]:
        x = jax.nn.silu(x @ W + b)
    W, b = ps[-1]
    return x @ W + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="runs/ppo_mjx")
    ap.add_argument("--motion-dir", default="assets/motions_aligned")
    ap.add_argument("--init-actor", default="")
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--std-init", type=float, default=-0.7)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--hidden2", type=int, default=128)
    ap.add_argument("--alive-bonus", action="store_true")
    ap.add_argument("--kp-scale", type=float, default=1.0)
    ap.add_argument("--anchor-obs", action="store_true")
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-update", action="store_true")
    args = ap.parse_args()
    OBS = spec.OBS_DIM + (spec.ANCHOR_DIM if args.anchor_obs else 0)
    ASC = ASCALE / float(args.kp_scale)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    qpos_tab, qvel_tab, goal_tab, lens, lib = build_tables(args.motion_dir)
    print("tables", qpos_tab.shape, goal_tab.shape)
    ref_pos_tab = qpos_tab[..., 0:3]
    _qp = np.asarray(qpos_tab)
    ref_R6_tab = jnp.asarray(np.stack([
        np.stack([spec.quat_to_quat6d(q) for q in _qp[k, :, 3:7]]) for k in range(_qp.shape[0])]))

    def ob(qpos, qvel, prev_a, clip, ref):
        if not args.anchor_obs:
            return obs_of(qpos, qvel, prev_a)
        return obs_of(qpos, qvel, prev_a, ref_pos_tab[clip, ref], ref_R6_tab[clip, ref])
    STATIC = jnp.asarray([i for i, m in enumerate(lib.motions) if m.name.startswith("walk")][0])
    m = mujoco.MjModel.from_xml_path(XML); m.opt.timestep = 0.005
    if args.kp_scale != 1.0:
        m.actuator_gainprm[:, 0] *= args.kp_scale
        m.actuator_biasprm[:, 1] *= args.kp_scale
        m.actuator_biasprm[:, 2] *= args.kp_scale
        print("kp/kd scale", args.kp_scale, flush=True)
    mx = mjx.put_model(m); d0 = mjx.make_data(mx)
    B = args.envs
    d = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), d0)
    cmin = jnp.asarray(m.actuator_ctrlrange[:, 0], jnp.float32)
    cmax = jnp.asarray(m.actuator_ctrlrange[:, 1], jnp.float32)
    jmin = jnp.asarray(m.jnt_range[1:, 0], jnp.float32)
    jmax = jnp.asarray(m.jnt_range[1:, 1], jnp.float32)

    def ctrl_step(mx, dd):
        return lax.fori_loop(0, 4, lambda _, x: mjx.step(mx, x), dd)
    step = jax.jit(jax.vmap(ctrl_step, in_axes=(None, 0)))

    obs_mean, obs_std, goal_mean, goal_std = norm_stats(qpos_tab, qvel_tab, goal_tab)
    if args.anchor_obs:
        obs_mean = np.concatenate([obs_mean, np.zeros(spec.ANCHOR_DIM)])
        obs_std = np.concatenate([obs_std, np.ones(spec.ANCHOR_DIM)])
    print("norm stats done")

    key = jax.random.PRNGKey(0)
    key, k1, k2, k3 = jax.random.split(key, 4)
    a_ps = init_mlp(k1, [spec.HISTORY * OBS + 730, args.hidden, args.hidden2, NU])
    if args.init_actor:
        import pickle
        with open(args.init_actor, "rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, tuple) and len(loaded) == 3:
            loaded = loaded[0]  # (a_ps, v_ps, log_std) checkpoint
        a_ps = [(jnp.asarray(W), jnp.asarray(b)) for (W, b) in loaded]
        print("actor warm start", args.init_actor, flush=True)
    v_ps = init_mlp(k2, [spec.HISTORY * OBS + 730, max(128, args.hidden // 2), 1], last_scale=1.0)
    log_std = jnp.full((NU,), args.std_init)
    opt = optax.adam(args.lr)
    opt_state = opt.init((a_ps, v_ps, log_std))
    om = jnp.asarray(obs_mean, jnp.float32); os_ = jnp.asarray(obs_std, jnp.float32)
    gm = jnp.asarray(goal_mean, jnp.float32); gs_ = jnp.asarray(goal_std, jnp.float32)

    def features(obs_hist, goal):
        return jnp.concatenate([((obs_hist - om) / os_).reshape(obs_hist.shape[0], spec.HISTORY * OBS),
                                (goal - gm) / gs_], -1)

    def actor(ps, x):
        return mlp(ps, x)

    def critic(ps, x):
        return mlp(ps, x).squeeze(-1)

    def sample_reset(key, n):
        if args.static:
            return jnp.full((n,), STATIC, jnp.int32), jnp.zeros((n,), jnp.int32)
        k1, k2 = jax.random.split(key)
        clip = jax.random.randint(k1, (n,), 0, len(lib.motions))
        t0 = jax.random.randint(k2, (n,), 0, jnp.maximum(1, lens[clip] - 12))
        return clip, t0

    def reset_data(d, clip, t0, mask):
        qp = qpos_tab[clip, t0]; qv = qvel_tab[clip, t0]
        d = d.replace(qpos=jnp.where(mask[:, None], qp, d.qpos),
                      qvel=jnp.where(mask[:, None], qv, d.qvel),
                      ctrl=jnp.where(mask[:, None], qp[:, 7:36], d.ctrl))
        return d

    def reward_fn(d, clip, ref_idx, a, prev_a):
        qp = d.qpos; qv = d.qvel
        rp = qpos_tab[clip, ref_idx]; rv = qvel_tab[clip, ref_idx]
        root_pos = jnp.sum((qp[:, 0:3] - rp[:, 0:3]) ** 2, -1)
        root_ori = qerr(qp[:, 3:7], rp[:, 3:7]) ** 2
        q = qp[:, 7:36]; qref = rp[:, 7:36]
        qd = qv[:, 6:35]; qdref = rv[:, 6:35]
        jpos = (((q - qref) ** 2) * MASK).sum(-1) / MDEN
        jvel = (((qd - qdref) ** 2) * MASK).sum(-1) / MDEN
        smooth = ((ASC * (a - prev_a)) ** 2).sum(-1)
        r = (0.5 * jnp.exp(-root_pos / 0.09) + 0.5 * jnp.exp(-root_ori / 0.16)
             + jnp.exp(-jpos / 0.25) + jnp.exp(-jvel / 1.0) - 0.1 * smooth)
        if args.alive_bonus:
            dh = jnp.abs(qp[:, 2] - rp[:, 2])
            tilt = jnp.abs(grav(qp[:, 3:7])[:, 2] - grav(rp[:, 3:7])[:, 2])
            r = r + 0.02 * ((dh <= 0.25) & (tilt <= 0.8))
        return r * DT

    def termination(d, clip, ref_idx):
        rp = qpos_tab[clip, ref_idx]
        dh = jnp.abs(d.qpos[:, 2] - rp[:, 2])
        tilt = jnp.abs(grav(d.qpos[:, 3:7])[:, 2] - grav(rp[:, 3:7])[:, 2])
        return (dh > 0.25) | (tilt > 0.8)

    def make_rollout(a_ps, v_ps, log_std, d, hist, goal, frame, stepc, clip, prev_a, key):
        def body(carry, _):
            d, hist, goal, frame, stepc, clip, prev_a, key = carry
            key, k1 = jax.random.split(key)
            mean = jnp.clip(actor(a_ps, features(hist, goal)), -8.0, 8.0)
            lg = jnp.clip(log_std, -3.0, 0.0)
            std = jnp.exp(lg)
            eps = jax.random.normal(k1, mean.shape)
            a = jnp.clip(mean + std * eps, -8.0, 8.0)
            logp = (-0.5 * ((a - mean) / std) ** 2 - lg - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)
            v = critic(v_ps, features(hist, goal))
            d = d.replace(ctrl=jnp.clip(Q0 + ASC * a, cmin, cmax))
            d = step(mx, d)
            ref = jnp.zeros(B, jnp.int32) if args.static else jnp.minimum(frame + 1, lens[clip] - 1)
            obs = ob(d.qpos, d.qvel, a, clip, ref)
            r = reward_fn(d, clip, ref, a, prev_a)
            term = termination(d, clip, ref)
            trunc = (stepc + 1 >= 500) if args.static else ((stepc + 1 >= 500) | (frame + 12 >= lens[clip]))
            done = term | trunc
            # next goal for non-done
            nxt = jnp.zeros(B, jnp.int32) if args.static else jnp.minimum(frame + 1, lens[clip] - 1)
            goal_next = goal_tab[clip, nxt]
            # reset done
            key, k2 = jax.random.split(key)
            nclip, nt0 = sample_reset(k2, B)
            d = reset_data(d, nclip, nt0, done)
            obs_new = ob(d.qpos, d.qvel, jnp.zeros_like(prev_a), nclip, nt0)
            obs = jnp.where(done[:, None], obs_new, obs)
            hist = jnp.concatenate([hist[:, 1:], obs[:, None, :]], 1)
            rep = jnp.repeat(obs[:, None, :], HIST, 1)
            hist = jnp.where(done[:, None, None], rep, hist)
            goal = jnp.where(done[:, None], goal_tab[nclip, nt0], goal_next)
            frame = jnp.where(done, nt0, nxt)
            stepc = jnp.where(done, 0, stepc + 1)
            clip = jnp.where(done, nclip, clip)
            prev_a = jnp.where(done[:, None], 0.0, a)
            out = (hist, goal, a, logp, v, r, done, term)
            return (d, hist, goal, frame, stepc, clip, prev_a, key), out
        return jax.lax.scan(body, (d, hist, goal, frame, stepc, clip, prev_a, key), None,
                            length=args.rollout)

    rollout = jax.jit(make_rollout)
    # update
    def loss_fn(params, batch):
        a_ps, v_ps, log_std = params
        obs_hist, goal, a, logp_old, adv, ret = batch
        obs_hist = obs_hist.reshape(-1, HIST, OBS)
        goal = goal.reshape(-1, 730)
        mean = jnp.clip(actor(a_ps, features(obs_hist, goal)), -8.0, 8.0)
        lg = jnp.clip(log_std, -3.0, 0.0)
        std = jnp.exp(lg)
        logp = (-0.5 * ((a - mean) / std) ** 2 - lg - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)
        ratio = jnp.exp(logp - logp_old)
        s1 = ratio * adv; s2 = jnp.clip(ratio, 1 - 0.2, 1 + 0.2) * adv
        pi_loss = -jnp.minimum(s1, s2).mean()
        v = critic(v_ps, features(obs_hist, goal))
        v_loss = jnp.mean((v - ret) ** 2)
        ent = (lg + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum()
        return pi_loss + 0.5 * v_loss - 1e-3 * ent, (pi_loss, v_loss)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    @jax.jit
    def update_step(params, opt_state, batch):
        (loss, aux), grads = grad_fn(params, batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, aux


    # init env
    key, sub = jax.random.split(key)
    clip, t0 = sample_reset(sub, B)
    d = reset_data(d, clip, t0, jnp.ones(B, bool))
    prev_a = jnp.zeros((B, NU)); obs = ob(d.qpos, d.qvel, prev_a, clip, t0)
    hist = jnp.repeat(obs[:, None, :], HIST, 1)
    goal = goal_tab[clip, t0]; frame = t0; stepc = jnp.zeros(B, jnp.int32)
    params = (a_ps, v_ps, log_std)
    # ---- JAX eval: 5 clips from frame 0, 500 steps, deterministic ----
    BE = len(lib.motions)
    d_e0 = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (BE,) + x.shape), d0)
    clip_e = jnp.arange(BE); t0_e = jnp.zeros(BE, jnp.int32)

    @jax.jit
    def evaluate_jax(params):
        a_ps, v_ps, lg = params
        d = reset_data(d_e0, clip_e, t0_e, jnp.ones(BE, bool))
        obs = ob(d.qpos, d.qvel, jnp.zeros((BE, NU)), clip_e, t0_e)
        hist = jnp.repeat(obs[:, None, :], HIST, 1)
        goal = goal_tab[clip_e, t0_e]; frame = t0_e; stepc = jnp.zeros(BE, jnp.int32)
        prev_a = jnp.zeros((BE, NU)); done = jnp.zeros(BE, bool); total = jnp.zeros(BE)

        def body(carry, _):
            d, hist, goal, frame, stepc, prev_a, done, total = carry
            a = jnp.clip(actor(a_ps, features(hist, goal)), -8.0, 8.0)
            d = d.replace(ctrl=jnp.clip(Q0 + ASC * a, cmin, cmax))
            d = step(mx, d)
            ref = jnp.zeros(BE, jnp.int32) if args.static else jnp.minimum(frame + 1, lens[clip_e] - 1)
            obs = ob(d.qpos, d.qvel, a, clip_e, ref)
            r = reward_fn(d, clip_e, ref, a, prev_a)
            term = termination(d, clip_e, ref)
            total = total + jnp.where(done, 0.0, r)
            done = done | term
            nxt = jnp.zeros(BE, jnp.int32) if args.static else jnp.minimum(frame + 1, lens[clip_e] - 1)
            goal = goal_tab[clip_e, nxt]; frame = jnp.where(args.static, 0, nxt); stepc = stepc + 1
            prev_a = a
            hist = jnp.concatenate([hist[:, 1:], obs[:, None, :]], 1)
            return (d, hist, goal, frame, stepc, prev_a, done, total), None

        (_, _, _, _, _, _, _, total), _ = jax.lax.scan(
            body, (d, hist, goal, frame, stepc, prev_a, done, total), None, length=500)
        return total.mean() / (0.1 * 499.0)

    rng_np = np.random.default_rng(0)
    best = -1e9
    for it in range(1, args.iters + 1):
        t0_ = time.time()
        params = (params[0], params[1], params[2])
        print(f"before rollout it={it}", flush=True)
        (d, hist, goal, frame, stepc, clip, prev_a, key), tr = rollout(
            params[0], params[1], params[2], d, hist, goal, frame, stepc, clip, prev_a, key)
        print(f"after rollout it={it}", flush=True)
        o_hist, o_goal, o_a, o_logp, o_v, o_r, o_done, o_term = tr
        # last value
        last_v = critic(params[1], features(hist, goal))
        adv = jnp.zeros(B); advs = []
        for t in reversed(range(args.rollout)):
            nv = last_v if t == args.rollout - 1 else o_v[t + 1]
            delta = o_r[t] + 0.99 * nv * (1 - o_term[t]) - o_v[t]
            adv = delta + 0.99 * 0.95 * (1 - o_term[t]) * adv
            advs.append(adv)
        advs = jnp.stack(advs[::-1]); ret = advs + o_v
        if args.no_update:
            print("skip update", flush=True); break
        n = args.rollout * B
        oh = o_hist.reshape(n, HIST, OBS); og = o_goal.reshape(n, 730)
        oa = o_a.reshape(n, NU); olp = o_logp.reshape(n)
        oret = ret.reshape(n); oadv = advs.reshape(n)
        oadv = (oadv - oadv.mean()) / (oadv.std() + 1e-8)
        perm = np.arange(n)
        aux = None
        for ep in range(args.epochs):
            rng_np.shuffle(perm)
            for mb in range(args.minibatches):
                sl = perm[mb * (n // args.minibatches):(mb + 1) * (n // args.minibatches)]
                params, opt_state, aux = update_step(
                    params, opt_state,
                    (oh[sl], og[sl], oa[sl], olp[sl], oadv[sl], oret[sl]))
        if it % 10 == 0:
            print(f"[{it:4d}] r {float(o_r.mean()):.4f} pi {float(aux[0]):.3f} "
                  f"v {float(aux[1]):.3f} {time.time()-t0_:.1f}s", flush=True)
        if it % args.eval_every == 0:
            ev = float(evaluate_jax(params))
            print(f"[eval {it}] return {ev:.4f}", flush=True)
            if ev > best:
                best = ev
                import pickle
                with open(outdir / "best.pkl", "wb") as f:
                    pickle.dump(jax.tree_util.tree_map(lambda x: np.asarray(x), params), f)
                print("  saved best", best, flush=True)
        if it % 50 == 0:
            torch_save = None
            import pickle
            with open(outdir / "ckpt.pkl", "wb") as f:
                pickle.dump(jax.tree_util.tree_map(lambda x: np.asarray(x), params), f)
        if args.smoke and it >= 10:
            break


if __name__ == "__main__":
    main()
