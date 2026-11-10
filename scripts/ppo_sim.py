#!/usr/bin/env python3
"""PPO-Sim baseline on the real MuJoCo tracking env (paper Fig.4a baseline)."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from dataclasses import dataclass, asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from smini import motion, spec
from smini.env import EnvCfg
from smini.vecenv import VecEnv
from smini.replay import ReplayBuffer
from smini.norm import compute_stats
from smini.policy import Policy, Critic
from smini.train import Collector, TrainCfg, evaluate

XML = str(ROOT.parent / "assets/mujoco_menagerie/unitree_g1/scene_mjx.xml")


@dataclass
class PPOConfig:
    num_envs: int = 32
    rollout: int = 64
    iters: int = 3000
    epochs: int = 4
    minibatches: int = 8
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 1e-4
    vf_coef: float = 0.5
    ent_coef: float = 1e-3
    max_grad: float = 1.0
    init_policy: str = ""
    out: str = "runs/ppo_sim"
    motion_dir: str = "assets/motions"
    eval_every: int = 25
    eval_clips: int = 5
    eval_steps: int = 500
    seed: int = 0


def ppo_main(cfg: PPOConfig):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    out = Path(cfg.out); out.mkdir(parents=True, exist_ok=True)
    lib = motion.MotionLibrary.from_dir(ROOT / cfg.motion_dir)
    vec = VecEnv(XML, lib, EnvCfg(episode_s=10.0, seed=cfg.seed),
                 num_envs=cfg.num_envs, seed=cfg.seed)
    # stats from a short PD warm-up
    buf = ReplayBuffer(50000)
    col = Collector(vec, lib, buf, TrainCfg(), rng)
    while len(buf) < 10000:
        col.collect(8, col.pd_action)
    stats = compute_stats(buf)
    policy = Policy().to("cuda")
    if cfg.init_policy:
        policy.load_state_dict(torch.load(cfg.init_policy, map_location="cuda")["policy"])
        print("policy init", cfg.init_policy)
    else:
        a_first = np.stack([(m.qpos[0, 7:36] - spec.DEFAULT_Q) / spec.ACTION_SCALE
                            for m in lib.motions]).mean(0)
        policy.set_action_bias(np.clip(a_first, -1.5, 1.5))
    if not cfg.init_policy:
        policy.set_norm(*stats["goal"], *stats["obs"])
    log_std = nn.Parameter(torch.full((spec.ACTION_DIM,), -1.0, device="cuda"))
    critic = Critic().to("cuda"); critic.set_norm(*stats["state"], *stats["goal"], *stats["xi"])
    opt = torch.optim.Adam(list(policy.parameters()) + [log_std] + list(critic.parameters()), lr=cfg.lr)

    @torch.no_grad()
    def value_of(state, goal, xi):
        return critic(torch.as_tensor(state, dtype=torch.float32, device="cuda"),
                      torch.as_tensor(goal, dtype=torch.float32, device="cuda"),
                      torch.as_tensor(xi, dtype=torch.float32, device="cuda")).squeeze(-1)

    obs, infos = vec.reset()
    hist = np.repeat(obs[:, None, :], spec.HISTORY, axis=1)
    best = -1e9
    for it in range(1, cfg.iters + 1):
        t0 = time.time()
        S, G, X, H, A, LP, V, R, D, TR, FV = [], [], [], [], [], [], [], [], [], [], []
        for _ in range(cfg.rollout):
            state = torch.as_tensor(infos["state"], dtype=torch.float32, device="cuda")
            goal = torch.as_tensor(infos["goal"], dtype=torch.float32, device="cuda")
            xi = torch.as_tensor(infos["xi"], dtype=torch.float32, device="cuda")
            h = torch.as_tensor(hist, dtype=torch.float32, device="cuda")
            with torch.no_grad():
                mean = policy(goal, h)
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                a = dist.sample()
                logp = dist.log_prob(a).sum(-1)
                v = critic(state, goal, xi).squeeze(-1)
            obs2, rew, term, trunc, infos2 = vec.step(a.cpu().numpy())
            done = term | trunc
            final = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in infos2.items()}
            # value of the terminal state for bootstrapping (0 if hard termination)
            with torch.no_grad():
                fv = critic(torch.as_tensor(final["state"], dtype=torch.float32, device="cuda"),
                            torch.as_tensor(final["goal"], dtype=torch.float32, device="cuda"),
                            torch.as_tensor(final["xi"], dtype=torch.float32, device="cuda")).squeeze(-1)
                fv = fv * torch.as_tensor(1.0 - term.astype(np.float32), device="cuda")
            S.append(state); G.append(goal); X.append(xi); H.append(h)
            A.append(a); LP.append(logp); V.append(v); R.append(torch.as_tensor(rew, dtype=torch.float32, device="cuda"))
            D.append(torch.as_tensor(done.astype(np.float32), device="cuda")); TR.append(torch.as_tensor(term.astype(np.float32), device="cuda"))
            FV.append(fv)
            # reset done envs for the next step
            for i in np.where(done)[0]:
                o = vec.reset_one(i)
                obs2[i] = o
                infos2["state"][i] = vec.envs[i].state(); infos2["goal"][i] = vec.envs[i].goal()
                infos2["xi"][i] = vec.envs[i].xi; infos2["motion_id"][i] = vec.envs[i].motion_id
                infos2["frame_idx"][i] = vec.envs[i].frame_idx
            hist = np.roll(hist, -1, axis=1); hist[:, -1, :] = obs2
            obs, infos = obs2, infos2
        # GAE
        adv = torch.zeros(cfg.num_envs, device="cuda"); advs = []
        for t in reversed(range(cfg.rollout)):
            next_v = FV[t]
            delta = R[t] + cfg.gamma * next_v - V[t]
            adv = delta + cfg.gamma * cfg.lam * (1.0 - TR[t]) * adv
            advs.append(adv)
        advs = torch.stack(advs[::-1])
        returns = advs + torch.stack(V)
        A = torch.stack(A); LP = torch.stack(LP); H = torch.stack(H); G = torch.stack(G)
        S = torch.stack(S); X = torch.stack(X)
        B = cfg.num_envs * cfg.rollout
        # flatten
        fH = H.reshape(B, spec.HISTORY, spec.OBS_DIM)
        fG, fS, fX = G.reshape(B, -1), S.reshape(B, -1), X.reshape(B, -1)
        fA, fLP = A.reshape(B, -1), LP.reshape(B)
        fAdv, fRet = advs.reshape(B), returns.reshape(B)
        fAdv = (fAdv - fAdv.mean()) / (fAdv.std() + 1e-8)
        inds = np.arange(B)
        for epoch in range(cfg.epochs):
            rng.shuffle(inds)
            for mb in np.array_split(inds, cfg.minibatches):
                mb = torch.as_tensor(mb, device="cuda")
                mean = policy(fG[mb], fH[mb])
                dist = torch.distributions.Normal(mean, log_std.exp())
                logp = dist.log_prob(fA[mb]).sum(-1)
                ratio = (logp - fLP[mb]).exp()
                s1 = ratio * fAdv[mb]; s2 = ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * fAdv[mb]
                loss_pi = -torch.min(s1, s2).mean()
                v = critic(fS[mb], fG[mb], fX[mb]).squeeze(-1)
                loss_v = F.mse_loss(v, fRet[mb])
                ent = dist.entropy().sum(-1).mean()
                loss = loss_pi + cfg.vf_coef * loss_v - cfg.ent_coef * ent
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + [log_std] + list(critic.parameters()), cfg.max_grad)
                opt.step()
        if it % 10 == 0:
            print(f"[{it:4d}] r {float(torch.stack(R).mean()):.3f} v {float(loss_v):.4f} "
                  f"pi {float(loss_pi):.4f} {time.time()-t0:.1f}s", flush=True)
        if it % cfg.eval_every == 0:
            ev = evaluate(policy, lib, TrainCfg(
                xml_path=XML, motion_dir=str(ROOT / cfg.motion_dir),
                eval_clips=cfg.eval_clips, eval_steps=cfg.eval_steps, seed=cfg.seed))
            print(f"[eval {it}] return {ev['return']:.4f} fail {ev['fail_rate']:.2f}", flush=True)
            if ev["return"] > best:
                best = ev["return"]
                torch.save({"policy": policy.state_dict(), "critic": critic.state_dict(),
                            "log_std": log_std.detach().cpu()}, out / "best.pt")
                print("  saved best", best, flush=True)
    torch.save({"policy": policy.state_dict(), "critic": critic.state_dict(),
                "log_std": log_std.detach().cpu()}, out / "last.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--init-policy", default="")
    ap.add_argument("--out", default="runs/ppo_sim")
    ap.add_argument("--motion-dir", default="assets/motions")
    args = ap.parse_args()
    ppo_main(PPOConfig(num_envs=args.envs, rollout=args.rollout, iters=args.iters,
                       init_policy=args.init_policy, out=args.out, motion_dir=args.motion_dir))
