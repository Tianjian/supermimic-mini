#!/usr/bin/env python3
"""Minimal fair BPTT test: 1024 Warp env, paper-style 93/730, H=32, batch 1024.

Collection in Warp; world model / actor / critic trained by the existing
smini.train BPTT code on a 283-dim replay buffer.  raw + EMA eval.
"""
from __future__ import annotations
import argparse, copy, os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, torch, torch.nn as nn
import mujoco, mujoco_warp as mjw, warp as wp
from smini import motion, spec, build_model
from smini.replay import ReplayBuffer
from smini.policy import Policy, Critic
from smini.world_model import WorldModel
from smini.train import TrainCfg, train_wm, train_policy, evaluate
from smini.norm import compute_stats

XML = str(ROOT.parent / "assets/mujoco_menagerie/unitree_g1/scene_mjx.xml")
HIST, NU = spec.HISTORY, spec.ACTION_DIM
Q0 = torch.tensor(spec.DEFAULT_Q, dtype=torch.float32, device="cuda")
ASC = torch.tensor(spec.ACTION_SCALE, dtype=torch.float32, device="cuda")
MASK = torch.tensor(spec.JOINT_REWARD_MASK, dtype=torch.float32, device="cuda")
MDEN = float(MASK.sum())


def qmat(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = w*w + x*x + y*y + z*z; s = 2.0/n.clamp_min(1e-8)
    wx, wy, wz = s*w*x, s*w*y, s*w*z; xx, xy, xz = s*x*x, s*x*y, s*x*z
    yy, yz, zz = s*y*y, s*y*z, s*z*z
    return torch.stack([torch.stack([1-(yy+zz), xy-wz, xz+wy], -1),
                        torch.stack([xy+wz, 1-(xx+zz), yz-wx], -1),
                        torch.stack([xz-wy, yz+wx, 1-(xx+yy)], -1)], -2)


def sixd(R):
    return R[..., :, :2].transpose(-1, -2).reshape(*R.shape[:-2], 6)


def obs_of(qpos, qvel, prev):
    R = qmat(qpos[:, 3:7])
    return torch.cat([qpos[:, 7:36]-Q0, qvel[:, 6:35], qvel[:, 3:6], -R[:, 2, :], prev], -1)


class WarpEnv:
    def __init__(self, N, motion_dir, seed=0, clips=None):
        self.N = N
        m = mujoco.MjModel.from_xml_path(XML); m.opt.timestep = 0.005; m.opt.iterations = 20; m.opt.ls_iterations = 50
        self.m = m; self.mx = mjw.put_model(m)
        try:
            self.mx.opt.warn_overflow = 0
        except Exception:
            pass
        d = mujoco.MjData(m)
        self.dx = mjw.put_data(m, d, nworld=N, nconmax=512, naconmax=512, njmax=1024)
        lib = motion.MotionLibrary.from_dir(ROOT / motion_dir)
        lib.motions = lib.motions[:5] if clips is None else [lib.motions[i] for i in clips]
        self.lib = lib; self.nclips = len(lib.motions)
        Tmax = max(len(x) for x in lib.motions)
        self.qpos = np.zeros((self.nclips, Tmax, 36), np.float32); self.qvel = np.zeros((self.nclips, Tmax, 35), np.float32)
        self.goal = np.zeros((self.nclips, Tmax, 730), np.float32); self.lens = np.zeros(self.nclips, np.int32)
        for k, mo in enumerate(lib.motions):
            T = len(mo); self.lens[k] = T
            self.qpos[k, :T] = mo.qpos; self.qvel[k, :T] = mo.qvel
            self.qpos[k, T:] = mo.qpos[-1]; self.qvel[k, T:] = mo.qvel[-1]
            for t in range(T):
                idx = np.minimum(t + 1 + np.arange(10), T - 1)
                self.goal[k, t] = np.concatenate([spec.pack_goal_frame(mo.qpos[j], mo.qvel[j]) for j in idx])
            self.goal[k, T:] = self.goal[k, T-1]
        self.qpos_t = torch.tensor(self.qpos, device="cuda"); self.qvel_t = torch.tensor(self.qvel, device="cuda")
        self.goal_t = torch.tensor(self.goal, device="cuda"); self.lens_t = torch.tensor(self.lens, device="cuda")
        self.bids = torch.tensor(build_model.body_ids(m), device="cuda", dtype=torch.long)
        self.cmin = torch.tensor(m.actuator_ctrlrange[:, 0], device="cuda", dtype=torch.float32)
        self.cmax = torch.tensor(m.actuator_ctrlrange[:, 1], device="cuda", dtype=torch.float32)
        self.rng = np.random.default_rng(seed)
        self.clip = torch.zeros(N, dtype=torch.long, device="cuda")
        self.frame = torch.zeros(N, dtype=torch.long, device="cuda")
        self.stepc = torch.zeros(N, dtype=torch.long, device="cuda")
        self.prev = torch.zeros(N, NU, device="cuda")
        self.prev_xpos = None
        self.hist = None
        self.reset()

    def _write(self, idx, clip, t0):
        qp = self.qpos_t[clip, t0].contiguous(); qv = self.qvel_t[clip, t0].contiguous()
        if idx is None:
            qfull, vfull, cfull = qp, qv, qp[:, 7:36]
        else:
            qfull = wp.to_torch(self.dx.qpos).clone(); vfull = wp.to_torch(self.dx.qvel).clone(); cfull = wp.to_torch(self.dx.ctrl).clone()
            qfull[idx] = qp; vfull[idx] = qv; cfull[idx] = qp[:, 7:36]
        wp.copy(self.dx.qpos, wp.from_torch(qfull.contiguous(), dtype=wp.float32))
        wp.copy(self.dx.qvel, wp.from_torch(vfull.contiguous(), dtype=wp.float32))
        wp.copy(self.dx.ctrl, wp.from_torch(cfull.contiguous(), dtype=wp.float32)); wp.synchronize()

    def reset(self, idx=None):
        if idx is None:
            idx = torch.arange(self.N, device="cuda")
        clip = torch.randint(0, self.nclips, (len(idx),), device="cuda")
        t0 = (torch.rand(len(idx), device="cuda") * (self.lens_t[clip] - 40).clamp_min(1)).long()
        self.clip[idx] = clip; self.frame[idx] = t0; self.stepc[idx] = 0; self.prev[idx] = 0
        self._write(idx, clip, t0)
        ob = obs_of(wp.to_torch(self.dx.qpos).clone(), wp.to_torch(self.dx.qvel).clone(), self.prev)
        xall = wp.to_torch(self.dx.xpos).clone()[:, self.bids]
        if self.hist is None:
            self.hist = ob[:, None, :].repeat(1, HIST, 1); self.prev_xpos = xall
        else:
            self.hist[idx] = ob[idx][:, None, :].repeat(1, HIST, 1); self.prev_xpos[idx] = xall[idx]

    def current(self):
        qpos = wp.to_torch(self.dx.qpos).clone(); qvel = wp.to_torch(self.dx.qvel).clone()
        xpos = wp.to_torch(self.dx.xpos).clone()[:, self.bids]
        xquat = wp.to_torch(self.dx.xquat).clone()[:, self.bids]
        cvel = wp.to_torch(self.dx.cvel).clone()[:, self.bids]
        body6 = sixd(qmat(xquat.reshape(-1, 4)).reshape(-1, 14, 3, 3))
        lin = (xpos - self.prev_xpos) / 0.02
        ang = cvel[:, :, 0:3]
        bodies = torch.cat([xpos, body6, lin, ang], -1).reshape(self.N, -1)
        state = torch.cat([qpos[:, 0:3], sixd(qmat(qpos[:, 3:7])), qvel[:, 0:3], qvel[:, 3:6],
                           qpos[:, 7:36] - Q0, qvel[:, 6:35], bodies], -1)
        obs = obs_of(qpos, qvel, self.prev)
        goal = self.goal_t[self.clip, self.frame]
        return state, obs, goal, qpos, qvel

    def step(self, a):
        self.prev = a.clone()
        ctrl = torch.clamp(Q0 + ASC * a, self.cmin, self.cmax)
        wp.copy(self.dx.ctrl, wp.from_torch(ctrl.contiguous(), dtype=wp.float32))
        for _ in range(4):
            mjw.step(self.mx, self.dx)
        wp.synchronize()
        qpos = wp.to_torch(self.dx.qpos).clone(); qvel = wp.to_torch(self.dx.qvel).clone()
        xpos = wp.to_torch(self.dx.xpos).clone()[:, self.bids]
        self.frame = torch.minimum(self.frame + 1, self.lens_t[self.clip] - 1)
        self.stepc = self.stepc + 1
        ref = self.frame
        rp = self.qpos_t[self.clip, ref]; rv = self.qvel_t[self.clip, ref]
        root_pos = ((qpos[:, 0:3] - rp[:, 0:3]) ** 2).sum(-1)
        Rr = qmat(rp[:, 3:7]); R = qmat(qpos[:, 3:7])
        root_ori = (2.0 - (Rr.transpose(1, 2) @ R).diagonal(dim1=1, dim2=2).sum(-1)).clamp_min(0)
        jpos = ((((qpos[:, 7:36] - rp[:, 7:36]) ** 2) * MASK).sum(-1)) / MDEN
        jvel = ((((qvel[:, 6:35] - rv[:, 6:35]) ** 2) * MASK).sum(-1)) / MDEN
        smooth = ((ASC * (a - self.prev)) ** 2).sum(-1)  # prev already a -> 0; kept for shape
        r = (0.5 * torch.exp(-root_pos / 0.09) + 0.5 * torch.exp(-root_ori / 0.16)
             + torch.exp(-jpos / 0.25) + torch.exp(-jvel / 1.0)) * 0.02
        grav = -R[:, 2, 2]; gravr = -Rr[:, 2, 2]
        term = ((qpos[:, 2] - rp[:, 2]).abs() > 0.25) | ((grav - gravr).abs() > 0.8)
        trunc = (self.stepc >= 500) | (self.frame + 12 >= self.lens_t[self.clip])
        done = term | trunc
        ob = obs_of(qpos, qvel, a)
        self.hist = torch.cat([self.hist[:, 1:], ob[:, None, :]], 1)
        self.prev_xpos = xpos.clone()
        idx = torch.nonzero(done).squeeze(-1)
        if len(idx):
            self.reset(idx)
        return qpos, qvel, ob, r, term, trunc, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--collect-steps", type=int, default=5)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--wm-updates", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--pi-batch", type=int, default=0)
    ap.add_argument("--wm-batch", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl-coef", type=float, default=0.5)
    ap.add_argument("--ref-ema", type=float, default=0.999)
    ap.add_argument("--wm-refresh", type=int, default=100)
    ap.add_argument("--freeze-wm", action="store_true")
    ap.add_argument("--init-wm", default="runs/wm_teacher_283.pt")
    ap.add_argument("--wm-warmup", type=int, default=0)
    ap.add_argument("--motion-dir", default="assets/motions_trim")
    ap.add_argument("--out", default="runs/bptt_warp")
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    wp.init()
    env = WarpEnv(args.envs, args.motion_dir)
    lib = motion.MotionLibrary.from_dir(ROOT / args.motion_dir); lib.motions = lib.motions[:5]
    pb = args.pi_batch or min(1024, args.envs * 2)
    wb = args.wm_batch or min(512, args.envs * 4)
    cfg = TrainCfg(xml_path=XML, motion_dir=str(ROOT / args.motion_dir), wm_hidden=(1024, 1024, 1024),
                   horizon=args.horizon, pi_batch=pb, wm_batches=4, wm_batch=wb, wm_k=8,
                   buffer_capacity=1_000_000, eval_clips=5, eval_steps=500,
                   eval_every=args.eval_every, seed=0, kl_coef=args.kl_coef,
                   ref_ema=args.ref_ema, wm_refresh_every=args.wm_refresh,
                   freeze_wm=args.freeze_wm)
    wm = WorldModel(spec.STATE_DIM, spec.ACTION_DIM, spec.XI_DIM, spec.GOAL_FRAME_DIM,
                    hidden=cfg.wm_hidden).cuda()
    if args.init_wm:
        ck = torch.load(ROOT / args.init_wm, map_location="cuda")
        wm.load_state_dict(ck["wm"]); print("wm loaded", args.init_wm, flush=True)
    policy = Policy().cuda()
    a_first = np.stack([(m.qpos[0, 7:36] - spec.DEFAULT_Q) / spec.ACTION_SCALE for m in lib.motions]).mean(0)
    policy.set_action_bias(np.clip(a_first, -1.5, 1.5))
    opt_wm = torch.optim.Adam(wm.parameters(), lr=1e-3)
    buf = ReplayBuffer(cfg.buffer_capacity)
    # episode storage per env
    S = [[] for _ in range(args.envs)]; O = [[] for _ in range(args.envs)]
    A = [[] for _ in range(args.envs)]; R = [[] for _ in range(args.envs)]
    D = [[] for _ in range(args.envs)]; TR = [[] for _ in range(args.envs)]
    G = [[] for _ in range(args.envs)]; F = [[] for _ in range(args.envs)]
    counts = [0]

    def flush(i):
        if len(S[i]) < 2:
            S[i].clear(); O[i].clear(); A[i].clear(); R[i].clear(); D[i].clear(); TR[i].clear(); G[i].clear(); F[i].clear(); return
        buf.add_episode({"state": np.asarray(S[i], np.float32), "obs": np.asarray(O[i], np.float32),
                         "action": np.asarray(A[i], np.float32), "reward": np.asarray(R[i], np.float32),
                         "terminated": np.asarray(D[i], np.float32), "truncated": np.asarray(TR[i], np.float32),
                         "xi": np.zeros((len(S[i]), spec.XI_DIM), np.float32),
                         "goal": np.asarray(G[i], np.float32), "motion_id": int(env.clip[i].item()),
                         "frame_idx": np.asarray(F[i], np.int64)})
        counts[0] += len(S[i])
        S[i].clear(); O[i].clear(); A[i].clear(); R[i].clear(); D[i].clear(); TR[i].clear(); G[i].clear(); F[i].clear()

    def collect(steps):
        for _ in range(steps):
            state, obs, goal, _, _ = env.current()
            with torch.no_grad():
                a = policy(goal, env.hist)
            a = (a + 0.1 * torch.randn_like(a)).clamp(-50, 50)
            qpos, qvel, ob, r, term, trunc, done = env.step(a)
            st = state.cpu().numpy(); obn = obs.cpu().numpy(); gn = goal.cpu().numpy()
            an = a.cpu().numpy(); rn = r.cpu().numpy()
            te = term.cpu().numpy(); tr = trunc.cpu().numpy(); fi = env.frame.cpu().numpy()
            for i in range(args.envs):
                S[i].append(st[i]); O[i].append(obn[i]); A[i].append(an[i]); R[i].append(rn[i])
                D[i].append(float(te[i])); TR[i].append(float(tr[i])); G[i].append(gn[i]); F[i].append(int(fi[i]))
            idx = np.nonzero(done.cpu().numpy())[0]
            for i in idx:
                flush(int(i))

    # warmup collection + WM-only
    for _ in range(args.warmup):
        collect(args.collect_steps)
    for i in range(args.envs):
        flush(i)
    print("warmup buffer", len(buf), flush=True)
    stats = compute_stats(buf)
    policy.set_norm(*stats["goal"], *stats["obs"])
    ema = copy.deepcopy(policy).eval(); ref_policy = copy.deepcopy(policy).eval()
    critic = Critic(state_dim=cfg.critic_state_dim).cuda()
    critic.set_norm(stats["state"][0][:cfg.critic_state_dim], stats["state"][1][:cfg.critic_state_dim],
                    *stats["goal"], *stats["xi"])
    opt_pi = torch.optim.Adam(policy.parameters(), lr=args.lr)
    opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
    if args.wm_warmup:
        wm.train()
        for _ in range(args.wm_warmup):
            train_wm(wm, opt_wm, buf, cfg)
        wm.eval(); print("wm warmup done", flush=True)

    def eval_cpu(pol):
        return evaluate(pol, lib, cfg)["return"]

    best = -1.0; init_raw = None; bad = 0
    for it in range(1, args.iters + 1):
        t0 = time.time()
        collect(args.collect_steps)
        if args.freeze_wm:
            wm.eval()
            if args.wm_refresh and it % args.wm_refresh == 0:
                wm.train(); wm_stats = train_wm(wm, opt_wm, buf, cfg); wm.eval()
            else:
                wm_stats = {"loss": 0.0}
        else:
            wm_stats = train_wm(wm, opt_wm, buf, cfg)
        pi_stats = train_policy(policy, critic, ema, wm, opt_pi, opt_v, buf, cfg,
                                ref_policy=ref_policy)
        if cfg.kl_adaptive:
            kl = pi_stats.get("kl", 0.0)
            ratio = kl / max(cfg.kl_target, 1e-6)
            cfg.kl_coef = min(cfg.kl_max, max(cfg.kl_min,
                cfg.kl_coef * (1.0 + cfg.kl_rate * (ratio - 1.0))))
        with torch.no_grad():
            for pr, p in zip(ref_policy.parameters(), policy.parameters()):
                pr.mul_(args.ref_ema).add_(p, alpha=1.0 - args.ref_ema)
            for br, b in zip(ref_policy.buffers(), policy.buffers()):
                br.copy_(b)
        if it % 10 == 0:
            print(f"[{it:4d}] wm {wm_stats['loss']:.3f} pi {pi_stats['loss_pi']:.3f} "
                  f"ret {pi_stats['ret_imag']:.3f} buf {len(buf)} {time.time()-t0:.1f}s", flush=True)
        if it % args.eval_every == 0:
            ev = eval_cpu(policy); ev_ema = eval_cpu(ema)
            print(f"[eval {it}] raw {ev:.4f} ema {ev_ema:.4f} kl {pi_stats.get('kl',0):.3f}", flush=True)
            if init_raw is None:
                init_raw = ev
            if ev < cfg.early_stop * init_raw:
                bad += 1
                if bad >= 2:
                    print("early stop: raw collapsed", flush=True); break
            else:
                bad = 0
            torch.save({"policy": policy.state_dict(), "wm": wm.state_dict(), "critic": critic.state_dict()}, outdir / f"ckpt_{it}.pt")
            if ev > best:
                best = ev; torch.save({"policy": policy.state_dict(), "wm": wm.state_dict(), "critic": critic.state_dict()}, outdir / "best.pt")
    print("done best", best)


if __name__ == "__main__":
    main()
