"""SuperMimic-mini training loop (Algorithm A.1, CPU MuJoCo + GPU learner)."""
from __future__ import annotations

import copy
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import build_model, motion, spec
from .env import EnvCfg, TrackingEnv
from .vecenv import VecEnv
from .replay import ReplayBuffer
from .norm import compute_stats
from .world_model import WorldModel
from .policy import Policy, Critic
from .imagine import imagine, gae_targets, policy_loss, critic_loss


@dataclass
class TrainCfg:
    xml_path: str = ""
    motion_dir: str = ""
    out_dir: str = "runs/p0"
    seed: int = 0
    num_envs: int = 8
    episode_s: float = 10.0
    buffer_capacity: int = 100_000
    warmup_transitions: int = 8_000
    iters: int = 500
    transitions_per_iter: int = 512
    eval_every: int = 100
    save_every: int = 250
    eval_clips: int = 2
    eval_steps: int = 500
    anchor_obs: bool = False
    device: str = "cuda"
    # world model
    wm_hidden: tuple = (1024, 1024, 1024)
    wm_batches: int = 2
    wm_batch: int = 256
    wm_k: int = 8
    wm_lr: float = 1e-3
    # policy
    horizon: int = 16
    pi_batch: int = 128
    pi_lr: float = 2e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_global: float = 1.0
    clip_elem: float = 0.1
    critic_epochs: int = 3
    critic_minibatches: int = 4
    ema: float = 0.995
    init_wm: str = ""
    init_policy: str = ""
    freeze_wm: bool = False
    eval_at_start: bool = False
    critic_state_dim: int = 283
    kl_coef: float = 0.5
    kl_adaptive: bool = True
    kl_target: float = 0.01
    kl_rate: float = 0.05
    kl_min: float = 0.01
    kl_max: float = 2.0
    ref_ema: float = 0.999
    wm_refresh_every: int = 100
    early_stop: float = 0.7


def _episode_arrays(ep: dict) -> dict:
    out = {}
    for k, v in ep.items():
        if k in ("motion_id",):
            out[k] = int(v[0])
        else:
            out[k] = np.asarray(v)
    return out


class Collector:
    def __init__(self, vec: VecEnv, lib: motion.MotionLibrary, buffer: ReplayBuffer,
                 cfg: TrainCfg, rng: np.random.Generator):
        self.vec, self.lib, self.buffer, self.cfg, self.rng = vec, lib, buffer, cfg, rng
        self.n = vec.num_envs
        self.obs, self.infos = vec.reset()
        self.hist = np.repeat(self.obs[:, None, :], spec.HISTORY, axis=1)
        self.ep = [self._new_ep() for _ in range(self.n)]

    @staticmethod
    def _new_ep():
        return defaultdict(list)

    def _append(self, i, action, rew, term, trunc, info):
        e = self.ep[i]
        e["state"].append(info["state"])
        e["obs"].append(info["obs"])
        e["action"].append(action)
        e["reward"].append(rew)
        e["terminated"].append(float(term))
        e["truncated"].append(float(trunc))
        e["xi"].append(info["xi"])
        e["goal"].append(info["goal"])
        e["motion_id"].append(int(info["motion_id"]))
        e["frame_idx"].append(int(info["frame_idx"]))

    def _finish(self, i):
        e = _episode_arrays(self.ep[i])
        if len(e["state"]) >= spec.HISTORY + 1:
            self.buffer.add_episode(e)
        self.ep[i] = self._new_ep()
        o = self.vec.reset_one(i)
        env = self.vec.envs[i]
        return o, {
            "state": env.state(), "obs": o, "goal": env.goal(), "xi": env.xi,
            "motion_id": env.motion_id, "frame_idx": env.frame_idx,
        }

    def pd_action(self) -> np.ndarray:
        a = np.zeros((self.n, spec.ACTION_DIM))
        for i in range(self.n):
            m = self.lib[self.infos["motion_id"][i]]
            t = min(int(self.infos["frame_idx"][i]) + 1, len(m) - 1)
            q_ref = m.qpos[t, 7:36]
            a[i] = (q_ref - spec.DEFAULT_Q) / spec.ACTION_SCALE
        return np.clip(a, -50.0, 50.0)

    def policy_action(self, policy: Policy) -> np.ndarray:
        with torch.no_grad():
            goal = torch.as_tensor(self.infos["goal"], dtype=torch.float32, device=self.cfg.device)
            hist = torch.as_tensor(self.hist, dtype=torch.float32, device=self.cfg.device)
            return policy(goal, hist).cpu().numpy()

    def collect(self, n_steps: int, action_fn) -> int:
        added = 0
        for _ in range(n_steps):
            actions = action_fn()
            obs, rew, term, trunc, infos = self.vec.step(actions)
            self.hist = np.roll(self.hist, -1, axis=1)
            self.hist[:, -1, :] = obs
            for i in range(self.n):
                info_i = {k: v[i] for k, v in infos.items()}
                self._append(i, info_i["action_exec"], rew[i], term[i], trunc[i], info_i)
                if term[i] or trunc[i]:
                    o, info = self._finish(i)
                    obs[i] = o
                    for k, v in info.items():
                        infos[k][i] = v
                    self.hist[i] = np.repeat(o[None, :], spec.HISTORY, axis=0)
                    added += 1
            self.obs, self.infos = obs, infos
        return added


def train_wm(wm: WorldModel, opt, buffer: ReplayBuffer, cfg: TrainCfg) -> dict:
    wm.train()
    stats = {"loss": 0.0, "state": 0.0, "reward": 0.0, "term": 0.0}
    for _ in range(cfg.wm_batches):
        b = buffer.sample_windows(cfg.wm_batch, cfg.wm_k, device=cfg.device)
        sn = wm.norm_state(b["s"])
        target = wm.norm_state(b["s_next"])
        # v1 reward uses only the root+joint block (73 dims); the 210 body-link
        # dims are unused, so fit the world model on the 73-dim compact state.
        n_compact = spec.GOAL_FRAME_DIM
        loss = 0.0
        st = rw = te = 0.0
        for j in range(cfg.wm_k):
            sn = wm.forward_norm(sn, b["a"][:, j], b["xi"])
            l_s = F.mse_loss(sn[..., :n_compact], target[:, j, :n_compact])
            raw = wm.denorm_state(sn)
            r_hat, d_logit = wm.heads(raw, b["a"][:, j], b["goal_cur"][:, j])
            l_r = F.mse_loss(r_hat, b["r"][:, j] * spec.REWARD_SCALE)
            l_d = F.binary_cross_entropy_with_logits(d_logit, b["d"][:, j])
            loss = loss + l_s + l_r + 0.1 * l_d
            st += l_s.item(); rw += l_r.item(); te += l_d.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), cfg.clip_global)
        opt.step()
        stats["loss"] += loss.item()
        stats["state"] += st / cfg.wm_k
        stats["reward"] += rw / cfg.wm_k
        stats["term"] += te / cfg.wm_k
    for k in stats:
        stats[k] /= cfg.wm_batches
    return stats


def train_policy(policy: Policy, critic: Critic, ema: Policy, wm: WorldModel,
                 opt_pi, opt_v, buffer: ReplayBuffer, cfg: TrainCfg,
                 ref_policy=None) -> dict:
    for m in (policy, critic, wm):
        m.eval()
    starts = buffer.sample_starts(cfg.pi_batch, cfg.horizon, device=cfg.device)
    traj = imagine(policy, wm, critic, starts, cfg.horizon, cfg.gamma, ref_policy=ref_policy)
    # critic values are a fixed baseline for the actor (stop-gradient);
    # the actor gradient flows through the imagined rewards / WM only.
    returns, adv = gae_targets(traj["r"], traj["d"], traj["v"].detach(),
                               traj["last_v"].detach(), cfg.gamma, cfg.gae_lambda)

    loss_pi, pi_stats = policy_loss(traj, returns, cfg.kl_coef)
    opt_pi.zero_grad(set_to_none=True)
    loss_pi.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.clip_global)
    for p in policy.parameters():
        if p.grad is not None:
            p.grad.data.clamp_(-cfg.clip_elem, cfg.clip_elem)
    opt_pi.step()

    # critic: fixed GAE targets, several epochs (detach: no policy-graph reuse)
    s_c = traj["s"].detach()
    goal_c = traj["goal"].detach()
    xi_c = traj["xi"].detach()
    ret_c = returns.detach()
    n = cfg.pi_batch
    for _ in range(cfg.critic_epochs):
        perm = torch.randperm(n, device=cfg.device)
        for mb in torch.tensor_split(perm, cfg.critic_minibatches):
            # critic is trained on the whole horizon: flatten B,H
            v_all = critic(s_c[mb].reshape(-1, spec.STATE_DIM)[..., :cfg.critic_state_dim],
                           goal_c[mb].reshape(-1, spec.GOAL_DIM),
                           xi_c[mb].repeat_interleave(cfg.horizon, 0))
            loss_v = F.mse_loss(v_all.squeeze(-1), ret_c[mb].reshape(-1).clamp(-5.0, 5.0))
            opt_v.zero_grad(set_to_none=True)
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.clip_global)
            opt_v.step()

    with torch.no_grad():
        for pe, p in zip(ema.parameters(), policy.parameters()):
            pe.mul_(cfg.ema).add_(p, alpha=1.0 - cfg.ema)
        for be, b in zip(ema.buffers(), policy.buffers()):
            be.copy_(b)
    return {"loss_pi": loss_pi.item(), "kl": pi_stats.get("kl", 0.0), "ret_imag": returns.mean().item(),
            "v_imag": traj["v"].mean().item(), **pi_stats}


@torch.no_grad()
def evaluate(policy: Policy, lib: motion.MotionLibrary, cfg: TrainCfg) -> dict:
    env = TrackingEnv(cfg.xml_path, lib, EnvCfg(
        episode_s=cfg.episode_s, action_noise=0.0, obs_noise=False,
        reset_noise=False, domain_randomization=False, seed=cfg.seed + 999,
        anchor_obs=cfg.anchor_obs),
        seed=cfg.seed + 999)
    rets, fails = [], []
    n = min(cfg.eval_clips, len(lib))
    for c in range(n):
        obs = env.reset(motion_id=c, frame_idx=0)
        hist = np.repeat(obs[None, :], spec.HISTORY, axis=0)
        ret, done = 0.0, False
        for t in range(cfg.eval_steps):
            goal = torch.as_tensor(env.goal(), dtype=torch.float32, device=cfg.device)[None]
            h = torch.as_tensor(hist, dtype=torch.float32, device=cfg.device)[None]
            a = policy(goal, h)[0].cpu().numpy()
            obs, r, term, trunc, info = env.step(a)
            hist = np.roll(hist, -1, axis=0); hist[-1] = obs
            ret += r
            if term or trunc:
                done = True
                break
        rets.append(ret / (0.1 * (cfg.eval_steps - 1)))
        fails.append(float(done))
    return {"return": float(np.mean(rets)), "fail_rate": float(np.mean(fails))}


def load_cfg(path: str | None) -> TrainCfg:
    cfg = TrainCfg()
    if path:
        import yaml
        data = yaml.safe_load(Path(path).read_text()) or {}
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, tuple(v) if k == "wm_hidden" else v)
    if not cfg.xml_path:
        cfg.xml_path = str(Path(__file__).resolve().parents[2] / "assets" /
                           "mujoco_menagerie" / "unitree_g1" / "scene_mjx.xml")
    if not cfg.motion_dir:
        cfg.motion_dir = str(Path(__file__).resolve().parents[1] / "assets" / "motions")
    return cfg


def main(cfg: TrainCfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text("\n".join(f"{k}: {v}" for k, v in asdict(cfg).items()))

    lib = motion.MotionLibrary.from_dir(cfg.motion_dir)
    assert len(lib) > 0, f"no motions in {cfg.motion_dir}"
    print(f"[data] {len(lib)} clips, {lib.total_frames} frames @50Hz, "
          f"{lib.total_frames/50/3600:.2f} h")

    env_cfg = EnvCfg(episode_s=cfg.episode_s, seed=cfg.seed, anchor_obs=cfg.anchor_obs)
    vec = VecEnv(cfg.xml_path, lib, env_cfg, num_envs=cfg.num_envs, seed=cfg.seed)
    buffer = ReplayBuffer(cfg.buffer_capacity)
    collector = Collector(vec, lib, buffer, cfg, rng)

    wm = WorldModel(spec.STATE_DIM, spec.ACTION_DIM, spec.XI_DIM,
                    spec.GOAL_FRAME_DIM, hidden=cfg.wm_hidden).to(cfg.device)
    policy = Policy(obs_dim=spec.OBS_DIM_ANCHOR if cfg.anchor_obs else spec.OBS_DIM).to(cfg.device)
    # cold-start prior: mean first-frame action (near standing; safe at reset)
    a_first = np.stack([
        (m.qpos[0, 7:36] - spec.DEFAULT_Q) / spec.ACTION_SCALE for m in lib.motions
    ], axis=0)
    policy.set_action_bias(np.clip(a_first.mean(axis=0), -1.5, 1.5))
    ema = copy.deepcopy(policy).eval()
    ref_policy = copy.deepcopy(policy).eval()
    critic = Critic(state_dim=cfg.critic_state_dim).to(cfg.device)
    opt_wm = torch.optim.Adam(wm.parameters(), lr=cfg.wm_lr)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=cfg.pi_lr)
    opt_v = torch.optim.Adam(critic.parameters(), lr=cfg.pi_lr)
    print(f"[model] wm={sum(p.numel() for p in wm.parameters())/1e6:.2f}M "
          f"pi={sum(p.numel() for p in policy.parameters())/1e6:.2f}M "
          f"v={sum(p.numel() for p in critic.parameters())/1e6:.2f}M")

    # ---- warm start: PD tracking + random exploration
    t0 = time.time()
    warm_iters = 0
    while len(buffer) < cfg.warmup_transitions and warm_iters < 10_000:
        def act():
            a = collector.pd_action()
            rnd = rng.random((collector.n, 1)) < 0.2
            a = np.where(rnd, rng.uniform(-1, 1, a.shape), a)
            return a
        collector.collect(16, act)
        warm_iters += 16
    stats = compute_stats(buffer)
    if cfg.init_wm:
        ck = torch.load(cfg.init_wm, map_location=cfg.device)
        wm.load_state_dict(ck["wm"])
        print(f"[wm] initialised from {cfg.init_wm}")
    else:
        wm.set_norm(*stats["state"], *stats["action"], *stats["xi"], *stats["goal_cur"])
    policy.set_norm(*stats["goal"], *stats["obs"])
    if cfg.init_policy:
        pck = torch.load(cfg.init_policy, map_location=cfg.device)
        policy.load_state_dict(pck["policy"])
        ema.load_state_dict(policy.state_dict())
        print(f"[policy] initialised from {cfg.init_policy}")
    critic.set_norm(stats["state"][0][:cfg.critic_state_dim], stats["state"][1][:cfg.critic_state_dim],
                    *stats["goal"], *stats["xi"])
    print(f"[warmup] {len(buffer)} transitions in {time.time()-t0:.1f}s; "
          f"state std[min/med/max]={stats['state'][1].min():.3f}/"
          f"{np.median(stats['state'][1]):.3f}/{stats['state'][1].max():.3f}")

    if cfg.eval_at_start:
        ev0 = evaluate(policy, lib, cfg)
        print(f"[eval 0] return {ev0['return']:.4f} fail {ev0['fail_rate']:.2f}", flush=True)
    init_raw = None; bad = 0
    best = -1e9
    for it in range(1, cfg.iters + 1):
        t_it = time.time()
        # collect with current policy (exploration via env action noise)
        def act():
            # mixed behaviour for WM coverage: current policy + PD tracking + random
            a = collector.policy_action(policy)
            a = np.where(rng.random((collector.n, 1)) < 0.3, collector.pd_action(), a)
            rnd = rng.random((collector.n, 1)) < 0.2
            return np.where(rnd, rng.uniform(-3, 3, a.shape), a)
        collector.collect(max(1, cfg.transitions_per_iter // cfg.num_envs), act)

        if cfg.freeze_wm:
            wm.eval()
            if cfg.wm_refresh_every and it % cfg.wm_refresh_every == 0:
                wm.train(); wm_stats = train_wm(wm, opt_wm, buffer, cfg); wm.eval()
            else:
                wm_stats = {"loss": 0.0, "state": 0.0, "reward": 0.0, "term": 0.0}
        else:
            wm_stats = train_wm(wm, opt_wm, buffer, cfg)
        pi_stats = train_policy(policy, critic, ema, wm, opt_pi, opt_v, buffer, cfg,
                                ref_policy=ref_policy)
        if cfg.kl_adaptive:
            kl = pi_stats.get("kl", 0.0)
            ratio = kl / max(cfg.kl_target, 1e-6)
            cfg.kl_coef = min(cfg.kl_max, max(cfg.kl_min,
                cfg.kl_coef * (1.0 + cfg.kl_rate * (ratio - 1.0))))
        with torch.no_grad():
            for pr, p in zip(ref_policy.parameters(), policy.parameters()):
                pr.mul_(cfg.ref_ema).add_(p, alpha=1.0 - cfg.ref_ema)
            for br, b in zip(ref_policy.buffers(), policy.buffers()):
                br.copy_(b)

        if it % 10 == 0:
            print(f"[{it:5d}] wm {wm_stats['loss']:.4f} (s {wm_stats['state']:.4f} "
                  f"r {wm_stats['reward']:.4f}) pi {pi_stats['loss_pi']:.3f} "
                  f"ret_imag {pi_stats['ret_imag']:.3f} buf {len(buffer)} "
                  f"{time.time()-t_it:.2f}s/it", flush=True)
        if it % cfg.eval_every == 0:
            ev = evaluate(ema, lib, cfg)
            ev_raw = evaluate(policy, lib, cfg)
            print(f"[eval {it}] ema {ev['return']:.4f} raw {ev_raw['return']:.4f} "
                  f"fail {ev['fail_rate']:.2f} kl {pi_stats.get('kl',0):.3f}", flush=True)
            if init_raw is None:
                init_raw = ev_raw['return']
            if ev_raw['return'] < cfg.early_stop * init_raw:
                bad += 1
                if bad >= 2:
                    print("early stop: raw collapsed", flush=True); break
            else:
                bad = 0
            if ev["return"] > best:
                best = ev["return"]
                torch.save({"policy": ema.state_dict(), "wm": wm.state_dict(),
                            "critic": critic.state_dict(), "cfg": asdict(cfg)},
                           out / "best.pt")
        if it % cfg.save_every == 0:
            torch.save({"policy": ema.state_dict(), "wm": wm.state_dict(),
                        "critic": critic.state_dict(), "cfg": asdict(cfg)},
                       out / f"ckpt_{it}.pt")
    torch.save({"policy": ema.state_dict(), "wm": wm.state_dict(),
                "critic": critic.state_dict(), "cfg": asdict(cfg)}, out / "last.pt")
    print(f"[done] best eval return {best:.4f}; artifacts in {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    main(load_cfg(args.config))
