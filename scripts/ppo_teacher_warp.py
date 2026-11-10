#!/usr/bin/env python3
"""P3.2: torch + MuJoCo Warp PPO, warm-started from the SuperMimic teacher.

Warp runs N world copies; torch holds the frozen teacher (680 goal / 930 obs),
a residual actor (512-256) and the critic.  Same BC/KL anchor as P2.2.
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, torch, torch.nn as nn
import mujoco, mujoco_warp as mjw, warp as wp
from smini import motion, spec, build_model
from eval_pretrained import Pol, goal680

XML = str(ROOT.parent / "assets/mujoco_menagerie/unitree_g1/scene_mjx.xml")
HIST, NU = spec.HISTORY, spec.ACTION_DIM
Q0 = torch.tensor(spec.DEFAULT_Q, dtype=torch.float32, device="cuda")
ASC = torch.tensor(spec.ACTION_SCALE, dtype=torch.float32, device="cuda")
MASK = torch.tensor(spec.JOINT_REWARD_MASK, dtype=torch.float32, device="cuda")
MDEN = float(MASK.sum())


def qmat(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = w*w + x*x + y*y + z*z; s = 2.0 / n.clamp_min(1e-8)
    wx, wy, wz = s*w*x, s*w*y, s*w*z
    xx, xy, xz = s*x*x, s*x*y, s*x*z
    yy, yz, zz = s*y*y, s*y*z, s*z*z
    return torch.stack([torch.stack([1-(yy+zz), xy-wz, xz+wy], -1),
                        torch.stack([xy+wz, 1-(xx+zz), yz-wx], -1),
                        torch.stack([xz-wy, yz+wx, 1-(xx+yy)], -1)], -2)


def obs_of(qpos, qvel, prev):
    R = qmat(qpos[:, 3:7])
    return torch.cat([qpos[:, 7:36] - Q0, qvel[:, 6:35], qvel[:, 3:6], -R[:, 2, :], prev], -1)


def obs930(hist):
    parts = [hist[:, :, 0:29].reshape(hist.shape[0], -1), hist[:, :, 29:58].reshape(hist.shape[0], -1),
             hist[:, :, 58:61].reshape(hist.shape[0], -1), hist[:, :, 61:64].reshape(hist.shape[0], -1),
             hist[:, :, 64:93].reshape(hist.shape[0], -1)]
    return torch.cat(parts, -1)


class Res(nn.Module):
    def __init__(self, h1=512, h2=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1610, h1), nn.SiLU(), nn.Linear(h1, h2), nn.SiLU(), nn.Linear(h2, NU))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1610, 512), nn.SiLU(), nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Env:
    def __init__(self, N, motion_dir, seed=0):
        self.N = N
        m = mujoco.MjModel.from_xml_path(XML); m.opt.timestep = 0.005; m.opt.iterations = 20; m.opt.ls_iterations = 50
        self.m = m; self.mx = mjw.put_model(m)
        try:
            self.mx.opt.warn_overflow = 0
        except Exception:
            pass
        d = mujoco.MjData(m)
        self.dx = mjw.put_data(m, d, nworld=N, nconmax=256, naconmax=256, njmax=512)
        lib = motion.MotionLibrary.from_dir(ROOT / motion_dir); lib.motions = lib.motions[:5]
        self.lib = lib; self.nclips = len(lib.motions)
        Tmax = max(len(x) for x in lib.motions)
        self.bids = torch.tensor(build_model.body_ids(m), device="cuda", dtype=torch.long)
        self.bp = np.zeros((self.nclips, Tmax, 14, 3), np.float32)
        self.bq = np.zeros((self.nclips, Tmax, 14, 4), np.float32)
        self.bl = np.zeros((self.nclips, Tmax, 14, 3), np.float32)
        self.ba = np.zeros((self.nclips, Tmax, 14, 3), np.float32)
        self.qpos = np.zeros((self.nclips, Tmax, 36), np.float32)
        self.qvel = np.zeros((self.nclips, Tmax, 35), np.float32)
        self.goal = np.zeros((self.nclips, Tmax, 680), np.float32)
        self.lens = np.zeros(self.nclips, np.int32)
        for k, mo in enumerate(lib.motions):
            T = len(mo); self.lens[k] = T
            self.qpos[k, :T] = mo.qpos; self.qvel[k, :T] = mo.qvel
            self.bp[k, :T] = mo.body_pos; self.bq[k, :T] = mo.body_quat
            self.bl[k, :T] = mo.body_linvel; self.ba[k, :T] = mo.body_angvel
            self.qpos[k, T:] = mo.qpos[-1]; self.qvel[k, T:] = mo.qvel[-1]
            self.bp[k, T:] = mo.body_pos[-1]; self.bq[k, T:] = mo.body_quat[-1]
            self.bl[k, T:] = mo.body_linvel[-1]; self.ba[k, T:] = mo.body_angvel[-1]
            for t in range(T):
                self.goal[k, t] = goal680(mo.qpos, mo.qvel, t)
            self.goal[k, T:] = self.goal[k, T-1]
        self.qpos_t = torch.tensor(self.qpos, device="cuda")
        self.qvel_t = torch.tensor(self.qvel, device="cuda")
        self.goal_t = torch.tensor(self.goal, device="cuda")
        self.lens_t = torch.tensor(self.lens, device="cuda")
        self.bp_t = torch.tensor(self.bp, device="cuda"); self.bq_t = torch.tensor(self.bq, device="cuda")
        self.bl_t = torch.tensor(self.bl, device="cuda"); self.ba_t = torch.tensor(self.ba, device="cuda")
        self.jmin = torch.tensor(m.jnt_range[1:, 0], device="cuda", dtype=torch.float32)
        self.jmax = torch.tensor(m.jnt_range[1:, 1], device="cuda", dtype=torch.float32)
        self.rng = np.random.default_rng(seed)
        self.cmin = torch.tensor(m.actuator_ctrlrange[:, 0], device="cuda", dtype=torch.float32)
        self.cmax = torch.tensor(m.actuator_ctrlrange[:, 1], device="cuda", dtype=torch.float32)
        self.clip = torch.zeros(N, dtype=torch.long, device="cuda")
        self.frame = torch.zeros(N, dtype=torch.long, device="cuda")
        self.stepc = torch.zeros(N, dtype=torch.long, device="cuda")
        self.prev = torch.zeros(N, NU, device="cuda")
        self.prev_xpos = None
        self.hist = None

    def _write(self, idx, clip, t0):
        qp = self.qpos_t[clip, t0].contiguous(); qv = self.qvel_t[clip, t0].contiguous()
        if idx is None:
            qfull = qp; vfull = qv; cfull = qp[:, 7:36]
        else:
            qfull = wp.to_torch(self.dx.qpos).clone(); vfull = wp.to_torch(self.dx.qvel).clone(); cfull = wp.to_torch(self.dx.ctrl).clone()
            qfull[idx] = qp; vfull[idx] = qv; cfull[idx] = qp[:, 7:36]
        wp.copy(self.dx.qpos, wp.from_torch(qfull.contiguous(), dtype=wp.float32)); wp.copy(self.dx.qvel, wp.from_torch(vfull.contiguous(), dtype=wp.float32))
        wp.copy(self.dx.ctrl, wp.from_torch(cfull.contiguous(), dtype=wp.float32)); wp.synchronize()

    def reset(self, idx=None):
        if idx is None:
            idx = torch.arange(self.N, device="cuda")
        clip = torch.randint(0, self.nclips, (len(idx),), device="cuda")
        t0 = torch.randint(0, 32, (len(idx),), device="cuda")  # placeholder
        t0 = (torch.rand(len(idx), device="cuda") * (self.lens_t[clip] - 40).clamp_min(1)).long()
        self.clip[idx] = clip; self.frame[idx] = t0; self.stepc[idx] = 0; self.prev[idx] = 0
        self._write(idx, clip, t0)
        ob = obs_of(wp.to_torch(self.dx.qpos).clone(), wp.to_torch(self.dx.qvel).clone(), self.prev)
        xall = wp.to_torch(self.dx.xpos).clone()[:, self.bids]
        if self.hist is None:
            self.hist = ob[:, None, :].repeat(1, HIST, 1)
            self.prev_xpos = xall
        else:
            self.hist[idx] = ob[idx][:, None, :].repeat(1, HIST, 1)
            self.prev_xpos[idx] = xall[idx]
        return ob

    def step(self, a):
        ctrl = torch.clamp(Q0 + ASC * a, self.cmin, self.cmax)
        wp.copy(self.dx.ctrl, wp.from_torch(ctrl.contiguous(), dtype=wp.float32))
        for _ in range(4):
            mjw.step(self.mx, self.dx)
        wp.synchronize()
        qpos = wp.to_torch(self.dx.qpos).clone(); qvel = wp.to_torch(self.dx.qvel).clone()
        self.frame = torch.minimum(self.frame + 1, self.lens_t[self.clip] - 1)
        self.stepc = self.stepc + 1
        ref = self.frame
        rp = self.qpos_t[self.clip, ref]; rv = self.qvel_t[self.clip, ref]
        xpos = wp.to_torch(self.dx.xpos).clone()[:, self.bids]
        xquat = wp.to_torch(self.dx.xquat).clone()[:, self.bids]
        cvel = wp.to_torch(self.dx.cvel).clone()[:, self.bids]
        p = xpos; q = xquat; lin = (xpos - self.prev_xpos) / 0.02; ang = cvel[:, :, 0:3]
        self.prev_xpos = xpos.clone()
        bp = self.bp_t[self.clip, ref]; bq = self.bq_t[self.clip, ref]
        bl = self.bl_t[self.clip, ref]; ba = self.ba_t[self.clip, ref]
        pos_err = ((bp - p) ** 2).sum(-1).mean(-1)
        Rdes = qmat(bq.reshape(-1, 4)).reshape(-1, 14, 3, 3)
        R = qmat(q.reshape(-1, 4)).reshape(-1, 14, 3, 3)
        tr = (Rdes.transpose(-1, -2) @ R).diagonal(dim1=-2, dim2=-1).sum(-1)
        ori_err = (3.0 - tr).clamp_min(0).mean(-1)
        lin_err = ((bl - lin) ** 2).sum(-1).mean(-1)
        ang_err = ((ba - ang) ** 2).sum(-1).mean(-1)
        root_pos = ((qpos[:, 0:3] - rp[:, 0:3]) ** 2).sum(-1)
        Rr = qmat(rp[:, 3:7]); Rroot = qmat(qpos[:, 3:7])
        root_ori = (2.0 - (Rr.transpose(1, 2) @ Rroot).diagonal(dim1=1, dim2=2).sum(-1)).clamp_min(0)
        qj = qpos[:, 7:36]
        limit = (torch.clamp(qj - self.jmax, min=0) + torch.clamp(self.jmin - qj, min=0)).sum(-1)
        smooth = ((ASC * (a - self.prev)) ** 2).sum(-1)
        r = (torch.exp(-pos_err / 0.09) + torch.exp(-ori_err / 0.16)
             + torch.exp(-lin_err / 1.0) + torch.exp(-ang_err / 3.14 ** 2)
             + 0.5 * torch.exp(-root_pos / 0.09) + 0.5 * torch.exp(-root_ori / 0.16)
             - 0.1 * smooth - 2.0 * limit) * 0.02
        grav = -Rroot[:, 2, 2]; gravr = -Rr[:, 2, 2]
        term = ((qpos[:, 2] - rp[:, 2]).abs() > 0.25) | ((grav - gravr).abs() > 0.8)
        trunc = (self.stepc >= 500) | (self.frame + 12 >= self.lens_t[self.clip])
        done = term | trunc
        ob = obs_of(qpos, qvel, a)
        self.hist = torch.cat([self.hist[:, 1:], ob[:, None, :]], 1)
        final_hist = self.hist.clone(); final_goal = self.goal_t[self.clip, self.frame].clone()
        idx = torch.nonzero(done).squeeze(-1)
        if len(idx):
            self.reset(idx)
            ob[idx] = self.hist[idx, -1]
        self.prev = a.clone()
        if len(idx):
            self.prev[idx] = 0
        return ob, r, done, term, final_hist, final_goal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bc-coef", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--motion-dir", default="assets/motions_trim")
    ap.add_argument("--out", default="runs/ppo_warp")
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    wp.init()
    env = Env(args.envs, args.motion_dir)
    ck = torch.load(ROOT / "runs/pretrained/policy.pt", map_location="cpu", weights_only=False)
    teacher = Pol(); teacher.load_state_dict(ck["policy"]); teacher.cuda().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    res = Res().cuda(); critic = Critic().cuda()
    log_std = torch.full((NU,), -1.0, device="cuda", requires_grad=True)
    opt = torch.optim.Adam(list(res.parameters()) + list(critic.parameters()) + [log_std], lr=args.lr)
    ob = env.reset()
    print("env ready", args.envs)

    def feats(goal, o930):
        return torch.cat([o930, goal], -1)

    def eval_cpu():
        lib = motion.MotionLibrary.from_dir(ROOT / args.motion_dir); lib.motions = lib.motions[:5]
        from smini.env import TrackingEnv, EnvCfg
        e = TrackingEnv(XML, lib, EnvCfg(obs_noise=False, reset_noise=False, domain_randomization=False, action_noise=0), seed=0)
        rets = []
        with torch.no_grad():
            for ci in range(5):
                o = e.reset(motion_id=ci, frame_idx=0); h = np.repeat(o[None], HIST, 0); ret = 0.0; st = 0
                for t in range(args.eval_steps):
                    g = goal680(lib.motions[ci].qpos, lib.motions[ci].qvel, e.frame_idx)
                    g_t = torch.tensor(g[None], device="cuda", dtype=torch.float32)
                    o930 = obs930(torch.tensor(h[None], device="cuda", dtype=torch.float32))
                    a = (teacher(g_t, o930) + res(feats(g_t, o930)))[0].cpu().numpy()
                    o, r, te, tr, info = e.step(a); h = np.roll(h, -1, 0); h[-1] = o; ret += r; st += 1
                    if te or tr: break
                rets.append(ret / (0.1 * (args.eval_steps - 1)))
                print(f"  eval {lib.motions[ci].name:18s} steps {st:3d} norm {rets[-1]:.4f}")
        print(f"[eval] mean {np.mean(rets):.4f}", flush=True)
        return float(np.mean(rets))

    best = -1.0
    for it in range(1, args.iters + 1):
        t0 = time.time()
        S_F, S_G, S_AT, S_A, S_LP, S_V, S_R, S_T, S_D, S_FH, S_FG = [], [], [], [], [], [], [], [], [], [], []
        for t in range(args.rollout):
            goal = env.goal_t[env.clip, env.frame]
            o930 = obs930(env.hist); F = feats(goal, o930)
            with torch.no_grad():
                at = teacher(goal, o930)
                mean = at + res(F)
                lg = log_std.clamp(-3.0, 0.0); std = lg.exp()
                eps = torch.randn_like(mean); a = (mean + std * eps).clamp(-8, 8)
                logp = (-0.5 * ((a - mean) / std) ** 2 - lg - 0.5 * np.log(2 * np.pi)).sum(-1)
                v = critic(F)
            ob, r, done, term, fh, fg = env.step(a)
            S_F.append(F); S_G.append(goal); S_AT.append(at); S_A.append(a); S_LP.append(logp); S_V.append(v); S_R.append(r); S_T.append(term.float()); S_D.append(done.float()); S_FH.append(fh); S_FG.append(fg)
        F = torch.stack(S_F); G = torch.stack(S_G); AT = torch.stack(S_AT); A = torch.stack(S_A)
        LP = torch.stack(S_LP); V = torch.stack(S_V); R = torch.stack(S_R); T = torch.stack(S_T)
        D = torch.stack(S_D); FH = torch.stack(S_FH); FG = torch.stack(S_FG)
        with torch.no_grad():
            VF = critic(feats(obs930(FH.reshape(-1, HIST, 93)), FG.reshape(-1, 680))).reshape(args.rollout, args.envs)
        # last value
        goal = env.goal_t[env.clip, env.frame]; o930 = obs930(env.hist)
        with torch.no_grad():
            last_v = critic(feats(goal, o930))
        adv = torch.zeros_like(V[0]); advs = []
        for t in reversed(range(args.rollout)):
            nv = last_v if t == args.rollout - 1 else V[t + 1]
            nv = torch.where(T[t] > 0, torch.zeros_like(nv), torch.where(D[t] > 0, VF[t], nv))
            delta = R[t] + 0.99 * nv * (1 - T[t]) - V[t]
            adv = delta + 0.99 * 0.95 * (1 - T[t]) * adv
            advs.append(adv)
        advs = torch.stack(advs[::-1]); ret = advs + V
        n = args.rollout * args.envs
        f = F.reshape(n, -1); at = AT.reshape(n, -1); a = A.reshape(n, -1)
        lp = LP.reshape(n); advf = advs.reshape(n); retf = ret.reshape(n)
        advf = (advf - advf.mean()) / (advf.std() + 1e-8)
        bc_coef = args.bc_coef * (0.3 + 0.7 * (1 - (it - 1) / max(1, args.iters)))
        for ep in range(2):
            perm = torch.randperm(n, device="cuda")
            for mb in range(8):
                sl = perm[mb*(n//8):(mb+1)*(n//8)]
                mean = at[sl] + res(f[sl]); lg = log_std.clamp(-3.0, 0.0); std = lg.exp()
                logp = (-0.5 * ((a[sl] - mean) / std) ** 2 - lg - 0.5 * np.log(2 * np.pi)).sum(-1)
                ratio = (logp - lp[sl]).exp()
                s1 = ratio * advf[sl]; s2 = ratio.clamp(0.8, 1.2) * advf[sl]
                pi = -torch.min(s1, s2).mean()
                v = critic(f[sl]); vl = ((v - retf[sl]) ** 2).mean()
                bc = ((mean - at[sl]) ** 2).mean()
                loss = pi + 0.5 * vl + bc_coef * bc
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(list(res.parameters()) + list(critic.parameters()) + [log_std], 1.0); opt.step()
        if it % 10 == 0:
            print(f"[{it:3d}] r {r.mean():.4f} bc {bc.item():.5f} {time.time()-t0:.1f}s", flush=True)
        if it % args.eval_every == 0:
            ev = eval_cpu()
            torch.save({"res": res.state_dict(), "critic": critic.state_dict()}, outdir / f"ckpt_{it}.pt")
            if ev > best:
                best = ev; torch.save({"res": res.state_dict(), "critic": critic.state_dict()}, outdir / "best.pt")
                print("  best", best, flush=True)


if __name__ == "__main__":
    main()
