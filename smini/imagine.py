"""Imagination rollout on the learned world model, GAE targets and BPTT losses.

Paper §3.3 / Eq.2-3::

    a_t        = pi_theta(o_t + eta_t, g_t)                 # deterministic policy
    s_{t+1}    = f_w(s_t, a_t, xi)                          # differentiable WM
    r_hat_t    = r_w(s_{t+1}, a_t, g_t)                     # reward head
    d_hat_t    = sigmoid(d_w(s_{t+1}, a_t, g_t))            # soft termination
    J_hat      = sum_t gamma^t c_t r_hat_t + gamma^H c_H V(s_H, g_H, xi)

The rollout is a single differentiable chain through the policy, the world model
and the critic, so ``loss_pi.backward()`` gives the analytic policy gradient
(BPTT over H steps).  ``d`` is a *probability* in [0, 1] and enters the return
as the soft survival factor ``(1 - d)``; ``truncated`` is deliberately ignored
(time limits are not terminations, the rollout bootstraps through them).

Conventions used here (standard GAE alignment, kept consistent with
``ReplayBuffer``):

* ``traj["s"][:, t]`` is the *pre-action* state s_t, and ``traj["v"][:, t]`` is
  ``V(s_t, g_t, xi)`` -- so the critic target R_t of Eq.3 is directly the return
  of that state.
* ``traj["r"][:, t]`` / ``traj["d"][:, t]`` belong to the transition that starts
  at s_t (reward r_{t+1}, termination probability after a_t).
* ``traj["goal"][:, t]`` is the 730-dim goal fed to the policy / critic at step t.
* ``traj["s"]`` / ``traj["a"]`` are returned detached; ``r`` / ``d`` / ``v`` /
  ``last_v`` keep their graph for BPTT (see the return block below).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import spec

# Paper §3.4 sensor-noise scales, re-injected per imagined step (eta_t):
# q-q0 0.01 | qd 0.5 | base angvel 0.2 | projected gravity 0.05 | prev action 0
OBS_NOISE_STD = np.array(
    [0.01] * spec.N_JOINTS
    + [0.5] * spec.N_JOINTS
    + [0.2] * 3
    + [0.05] * 3
    + [0.0] * spec.ACTION_DIM,
    dtype=np.float32,
)


# --------------------------------------------------------------- obs head
def _rot_from_6d_torch(v6: torch.Tensor) -> torch.Tensor:
    """Reconstruct R from its first two columns (Gram-Schmidt), like ``spec``."""
    a = v6[..., 0:3]
    b = v6[..., 3:6]
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b = b - a * (a * b).sum(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    c = torch.cross(a, b, dim=-1)
    return torch.stack([a, b, c], dim=-1)


def obs_from_state_torch(
    s: torch.Tensor,
    prev_action: torch.Tensor,
    bias: torch.Tensor | None = None,
    ref_pose6: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Torch twin of :func:`smini.spec.pack_obs`.

    (B, 283), (B, 29) [, (B, 29)] -> (B, 93).  ``bias`` is the per-episode
    encoder bias ``xi[:, 2:31]``; ``spec.pack_obs`` adds it to ``q - q0``.
    """
    q_rel = s[..., 15:44]          # joint pos relative to DEFAULT_Q
    qd = s[..., 44:73]             # joint vel
    angvel_w = s[..., 12:15]       # root angular velocity, world
    R = _rot_from_6d_torch(s[..., 3:9])
    angvel_b = torch.einsum("...ji,...j->...i", R, angvel_w)   # R^T @ w
    grav_b = -R[..., 2, :]         # R^T @ [0, 0, -1]
    if bias is not None:
        q_rel = q_rel + bias
    out = torch.cat([q_rel, qd, angvel_b, grav_b, prev_action], dim=-1)
    if ref_pose6 is not None:
        p_ref, R6_ref = ref_pose6
        R_ref = _rot_from_6d_torch(R6_ref)
        p_err = p_ref - s[..., 0:3]
        R_rel = R_ref.transpose(-1, -2) @ R
        six = R_rel[..., :, :2].transpose(-1, -2).reshape(*R_rel.shape[:-2], 6)
        out = torch.cat([out, p_err, six], dim=-1)
    return out


# --------------------------------------------------------- analytic reward
def analytic_reward(s: torch.Tensor, a: torch.Tensor, prev_a: torch.Tensor,
                    goal_cur: torch.Tensor, action_scale: torch.Tensor) -> torch.Tensor:
    """Differentiable 6-term tracking reward from predicted state + reference.

    Uses root pose, joint pos/vel and action smoothness (the 14 body-link terms
    need differentiable FK and are omitted in this v1).  Returns per-sample
    reward already multiplied by dt=0.02.
    """
    p = s[..., 0:3]
    R = _rot_from_6d_torch(s[..., 3:9])
    p_ref = goal_cur[..., 0:3]
    R_ref = _rot_from_6d_torch(goal_cur[..., 3:9])
    q = s[..., 15:44]
    q_ref = goal_cur[..., 15:44]
    qd = s[..., 44:73]
    qd_ref = goal_cur[..., 44:73]

    root_pos = ((p - p_ref) ** 2).sum(-1)
    tr = (R_ref.transpose(-1, -2) @ R).diagonal(dim1=-2, dim2=-1).sum(-1)
    root_ori = (3.0 - tr).clamp_min(0.0)       # 2(1-cos) ~ theta^2
    w = torch.as_tensor(spec.JOINT_REWARD_MASK, dtype=q.dtype, device=q.device)
    denom = w.sum().clamp_min(1.0)
    jpos = (((q - q_ref) ** 2) * w).sum(-1) / denom
    jvel = (((qd - qd_ref) ** 2) * w).sum(-1) / denom
    smooth = ((action_scale * (a - prev_a)) ** 2).sum(-1)

    r = (0.5 * torch.exp(-root_pos / 0.3 ** 2)
         + 0.5 * torch.exp(-root_ori / 0.4 ** 2)
         + torch.exp(-jpos / 0.5 ** 2)
         + torch.exp(-jvel / 1.0 ** 2)
         - 0.1 * smooth)
    return r * 0.02


# -------------------------------------------------------------- imagination
def imagine(
    policy,
    wm,
    critic,
    starts: dict,
    horizon: int = 32,
    gamma: float = 0.99,
    obs_noise: float = 0.0,
    ref_policy=None,
) -> dict:
    """Differentiable H-step rollout on the world model starting from ``starts``.

    ``starts`` comes from :meth:`ReplayBuffer.sample_starts` and must hold
    ``s (B,283)``, ``obs_hist (B,10,93)``, ``goal_seq (B,horizon,730)``,
    ``xi (B,34)`` (``goal_cur`` is not needed here).

    ``obs_noise`` (default 0 = deterministic) scales the paper's per-dimension
    sensor-noise vector ``OBS_NOISE_STD`` before the policy sees each frame.

    Returns a dict of float32 tensors::

        s       (B, H, 283)   pre-action states s_0 .. s_{H-1} (critic inputs)
        a       (B, H, 29)    actions taken
        r       (B, H)        predicted rewards
        d       (B, H)        predicted termination *probability* in [0,1]
        v       (B, H)        V(s_t, g_t, xi)  (pre-action, aligned with ``s``)
        last_v  (B, 1)        V(s_H, g_last, xi) bootstrap value
        goal    (B, H, 730)   goals used at each step (for the critic update)
        xi      (B, 34)       sim context (broadcast to the critic)
        gamma   float         discount, recorded for bookkeeping
    """
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    device = next(policy.parameters()).device
    starts = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in starts.items()
    }
    for key in ("s", "obs_hist", "goal_seq", "xi"):
        if key not in starts:
            raise KeyError(f"starts is missing {key!r} (from ReplayBuffer.sample_starts)")

    s = starts["s"]
    obs_hist = starts["obs_hist"]
    goal_seq = starts["goal_seq"]
    xi = starts["xi"]
    if goal_seq.dim() != 3 or goal_seq.shape[1] < horizon:
        raise ValueError(
            f"goal_seq must be (B, >= {horizon}, 730), got {tuple(goal_seq.shape)}"
        )
    gfd = int(getattr(wm, "goal_frame_dim", spec.GOAL_FRAME_DIM))
    bias = xi[..., 2:31]
    action_scale = torch.as_tensor(spec.ACTION_SCALE, dtype=torch.float32, device=device)
    prev_a = obs_hist[:, -1, 64:93].clone()     # clean previous action from obs

    noise = None
    if obs_noise > 0.0:
        noise = torch.from_numpy(OBS_NOISE_STD).to(device) * float(obs_noise)

    S, A, R_, D, V, Aref = [], [], [], [], [], []
    for t in range(horizon):
        goal_t = goal_seq[:, t]                     # (B, 730)
        obs_in = obs_hist
        if noise is not None:
            # noise only on the newest frame, keep the history clean
            obs_in = torch.cat(
                [obs_hist[:, :-1], obs_hist[:, -1:] + noise], dim=1
            )
        a = policy(goal_t, obs_in)                  # (B, 29)
        # guard against model exploitation / out-of-distribution actions
        a = torch.clamp(a, -25.0, 25.0)
        if ref_policy is not None:
            with torch.no_grad():
                a_ref = ref_policy(goal_t, obs_in)
        else:
            a_ref = torch.zeros_like(a)
        s_prev = s                                  # V(s_t) is evaluated pre-action
        v = critic(s_prev[..., :critic.state_dim], goal_t, xi).squeeze(-1)  # (B,)
        s = wm(s_prev, a, xi)                       # (B, 283)
        # keep the imagined state inside the warm-start data's 5-sigma box
        s = torch.clamp(s, wm.s_mean - 5.0 * wm.s_std, wm.s_mean + 5.0 * wm.s_std)
        r = analytic_reward(s, a, prev_a, goal_t[:, :gfd], action_scale)
        # analytic soft termination (same thresholds as the env):
        # |root_z - ref_z| > 0.25 or |tilt difference| > 0.8, soft-OR.
        gcur = goal_t[:, :gfd]
        pz = s[..., 2]
        pz_ref = gcur[..., 2]
        R = _rot_from_6d_torch(s[..., 3:9])
        R_ref = _rot_from_6d_torch(gcur[..., 3:9])
        grav_z = -R[..., 2, 2]
        grav_z_ref = -R_ref[..., 2, 2]
        d_h = torch.sigmoid(((pz - pz_ref).abs() - 0.25) / 0.02)
        d_t = torch.sigmoid(((grav_z - grav_z_ref).abs() - 0.8) / 0.02)
        d = (d_h + d_t - d_h * d_t).clamp(0.0, 1.0)
        prev_a = a

        ref_pose6 = None
        if getattr(policy, "obs_dim", spec.OBS_DIM) > spec.OBS_DIM:
            ref_pose6 = (gcur[..., 0:3], gcur[..., 3:9])
        obs = obs_from_state_torch(s, a, bias, ref_pose6=ref_pose6)
        obs_hist = torch.cat([obs_hist[:, 1:], obs.unsqueeze(1)], dim=1)

        S.append(s_prev)
        A.append(a)
        Aref.append(a_ref)
        R_.append(r)
        D.append(d)
        V.append(v)

    g_last = goal_seq[:, min(horizon, goal_seq.shape[1] - 1)]
    last_v = critic(s[..., :critic.state_dim], g_last, xi)  # (B, 1)

    # ``s``/``a`` are detached: they exist so the critic can be re-evaluated on
    # the imagined states *after* the policy step (the plan's 5-epoch critic
    # loop).  ``r``/``d``/``v`` keep their graph: ``gae_targets`` feeds them into
    # ``policy_loss`` (BPTT through the world model and the soft termination).
    return {
        "s": torch.stack(S, dim=1).detach(),
        "a": torch.stack(A, dim=1),
        "a_ref": torch.stack(Aref, dim=1).detach(),
        "r": torch.stack(R_, dim=1),
        "d": torch.stack(D, dim=1),
        "v": torch.stack(V, dim=1),
        "last_v": last_v,
        "goal": goal_seq[:, :horizon],
        "xi": xi,
        "gamma": gamma,
    }


def gae_targets(
    r: torch.Tensor,
    d: torch.Tensor,
    v: torch.Tensor,
    last_v: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lambda-returns over the imagined rollout (paper Eq.3).

    ``R_t = r_t + gamma * (1 - d_t) * ((1 - lam) * v_{t+1} + lam * R_{t+1})``
    with ``R_H = last_v``; ``v_{t+1}`` for the last step is ``last_v``.
    Returns ``(returns, advantages)`` of shape ``(B, H)``, ``adv = R - v``.
    """
    if r.dim() != 2:
        raise ValueError(f"r must be (B, H), got {tuple(r.shape)}")
    B, H = r.shape
    if d.shape != (B, H) or v.shape != (B, H):
        raise ValueError(
            f"d/v must be ({B}, {H}), got {tuple(d.shape)} / {tuple(v.shape)}"
        )
    last = last_v.reshape(-1)
    if last.shape[0] != B:
        raise ValueError(
            f"last_v must hold {B} values, got shape {tuple(last_v.shape)}"
        )

    step_returns: list[torch.Tensor] = []
    R = last
    for t in range(H - 1, -1, -1):
        v_next = v[:, t + 1] if t + 1 < H else last
        R = r[:, t] + gamma * (1.0 - d[:, t]) * ((1.0 - lam) * v_next + lam * R)
        R = torch.clamp(R, -5.0, 5.0)
        step_returns.append(R)
    returns = torch.stack(step_returns[::-1], dim=1)
    return returns, returns - v


# ------------------------------------------------------------------- losses
def policy_loss(traj: dict, returns: torch.Tensor, kl_coef: float = 0.0) -> tuple[torch.Tensor, dict]:
    """``-mean(returns)``; the gradient flows through the critic bootstrap too."""
    loss = -returns.mean()
    kl = None
    if kl_coef > 0 and "a_ref" in traj:
        kl = ((traj["a"] - traj["a_ref"]) ** 2).mean()
        loss = loss + kl_coef * kl
    with torch.no_grad():
        d = traj["d"]
        stats = {
            "loss": float(loss),
            "return": float(returns.mean()),
            "reward": float(traj["r"].mean()),
            "terminated": float(d.mean()),
            "survival": float((1.0 - d).prod(dim=1).mean()),
            "value": float(traj["v"].mean()),
            "advantage": float((returns - traj["v"]).mean()),
            "kl": float(kl) if kl is not None else 0.0,
        }
    return loss, stats


def critic_loss(traj: dict, returns: torch.Tensor) -> torch.Tensor:
    """MSE between the critic and the (fixed) lambda-returns."""
    return F.mse_loss(traj["v"], returns.detach())


# ------------------------------------------------------------------ clipping
def clip_gradients(params, global_norm: float = 1.0, elementwise: float = 0.1) -> float:
    """The plan's dual clipping: global-norm 1.0 then element-wise 0.1.

    Returns the pre-clip global gradient norm (float).
    """
    params = [p for p in params if p.grad is not None]
    if not params:
        return 0.0
    total = torch.nn.utils.clip_grad_norm_(params, global_norm)
    for p in params:
        p.grad.clamp_(-elementwise, elementwise)
    return float(total)
