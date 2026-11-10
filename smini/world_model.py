"""Deterministic differentiable world model + reward / termination heads.

Paper §3.2::

    s_hat_{t+1} = f_w(s_hat_t, a_t, xi)     # dynamics, residual MLP
    r_hat_{t+1} = r_w(s_hat_t, a_t, g_t)    # reward head
    d_hat_t     = d_w(s_hat_t, g_t)         # termination head

The dynamics net predicts ``delta = s_{t+1} - s_t`` and is bounded with
``tanh(delta) * delta_limit`` (default +-3) so that a 32-step autoregressive
rollout cannot explode.  The last layer is initialised near zero, i.e. the
model starts as the identity map.

The reward and termination heads share one 512-wide trunk fed by
``[s(283), a(29), goal_cur(73)] = 385``.  ``heads`` returns the raw
termination *logit* (use ``torch.sigmoid`` for the soft survival weight).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import norm, spec


def mlp(sizes, act=nn.SiLU, norm: bool = False) -> nn.Sequential:
    """Plain MLP: ``sizes[0] -> ... -> sizes[-1]``, activations between layers."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if norm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(act())
    return nn.Sequential(*layers)


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


# ---------------------------------------------------------- SE(3) helpers
def _rot6d_to_mat(v6: torch.Tensor) -> torch.Tensor:
    a, b = v6[..., 0:3], v6[..., 3:6]
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b = b - a * (a * b).sum(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    c = torch.cross(a, b, dim=-1)
    return torch.stack([a, b, c], dim=-1)


def _orthonormalize(R: torch.Tensor) -> torch.Tensor:
    a, b = R[..., 0], R[..., 1]
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b = b - a * (a * b).sum(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    c = torch.cross(a, b, dim=-1)
    return torch.stack([a, b, c], dim=-1)


def _mat_to_6d(R: torch.Tensor) -> torch.Tensor:
    return torch.stack([R[..., 0], R[..., 1]], dim=-1).reshape(*R.shape[:-2], 6)


def _skew(w: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(w[..., 0])
    return torch.stack([
        torch.stack([z, -w[..., 2], w[..., 1]], dim=-1),
        torch.stack([w[..., 2], z, -w[..., 0]], dim=-1),
        torch.stack([-w[..., 1], w[..., 0], z], dim=-1),
    ], dim=-2)


ACC_SCALE = torch.tensor([10.0] * 3 + [50.0] * 3 + [200.0] * spec.N_JOINTS)


class _SemiImplicitMixin:
    pass


def _as_dim(stats, dim: int) -> torch.Tensor:
    """Flatten a stats vector and fit it to ``dim`` (goal = K repeats of a frame)."""
    t = torch.as_tensor(stats, dtype=torch.float32).reshape(-1)
    if t.numel() == dim:
        return t
    if t.numel() > dim and t.numel() % dim == 0:
        # per-frame stats for the 10 x 73 goal are the same block repeated
        blocks = t.reshape(-1, dim)
        if torch.allclose(blocks, blocks[:1].expand_as(blocks)):
            return blocks[0].clone()
        return blocks[0].clone()
    raise ValueError(f"normalisation stats of length {t.numel()} cannot be fit to {dim}")


class WorldModel(nn.Module):
    """Residual MLP dynamics + shared-trunk reward / termination heads."""

    def __init__(
        self,
        state_dim: int = spec.STATE_DIM,
        action_dim: int = spec.ACTION_DIM,
        xi_dim: int = spec.XI_DIM,
        goal_frame_dim: int = spec.GOAL_FRAME_DIM,
        hidden: tuple[int, ...] = (1024, 1024, 1024),
        head_hidden: int = 512,
        delta_limit: float = 3.0,
        norm: bool = False,
        semi_implicit: bool = False,
        dt: float = 0.02,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.xi_dim = int(xi_dim)
        self.goal_frame_dim = int(goal_frame_dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.delta_limit = float(delta_limit)
        self.semi_implicit = bool(semi_implicit)
        self.dt = float(dt)
        out_dim = 3 + 3 + spec.N_JOINTS if self.semi_implicit else state_dim

        self.dyn = mlp(
            [state_dim + action_dim + xi_dim, *self.hidden, out_dim], norm=norm
        )
        if self.semi_implicit:
            self.register_buffer("acc_scale", ACC_SCALE.clone())
        head_in = state_dim + action_dim + goal_frame_dim
        trunk: list[nn.Module] = [nn.Linear(head_in, head_hidden)]
        if norm:
            trunk.append(nn.LayerNorm(head_hidden))
        trunk.append(nn.SiLU())
        self.trunk = nn.Sequential(*trunk)
        self.r_head = nn.Linear(head_hidden, 1)
        self.d_head = nn.Linear(head_hidden, 1)

        # normalisation buffers (set from the warm-start buffer stats)
        for name, dim in (("s", state_dim), ("a", action_dim),
                          ("xi", xi_dim), ("g", goal_frame_dim)):
            self.register_buffer(f"{name}_mean", torch.zeros(dim))
            self.register_buffer(f"{name}_std", torch.ones(dim))

        # start as the identity dynamics (delta ~ 0) for a stable WM fit
        last = self.dyn[-1]
        nn.init.zeros_(last.bias)
        with torch.no_grad():
            last.weight.mul_(0.1)

    def set_norm(self, s_mean, s_std, a_mean, a_std, xi_mean, xi_std,
                 g_mean, g_std) -> None:
        """Install per-dim (mean, std); ``goal`` stats may be the full 730-dim
        vector -- the heads only see the current 73-dim frame, which is the
        first block of every goal, so the frame stats are its first 73 entries.
        """
        g_mean = _as_dim(g_mean, self.goal_frame_dim)
        g_std = _as_dim(g_std, self.goal_frame_dim)
        norm.set_buffer(self, "s", s_mean, s_std)
        norm.set_buffer(self, "a", a_mean, a_std)
        norm.set_buffer(self, "xi", xi_mean, xi_std)
        norm.set_buffer(self, "g", g_mean, g_std)

    # -------------------------------------------------------------- forward
    def norm_state(self, s):
        return norm.norm(s, self.s_mean, self.s_std)

    def denorm_state(self, sn):
        return norm.denorm(sn, self.s_mean, self.s_std)

    def forward_norm(self, sn: torch.Tensor, a: torch.Tensor, xi: torch.Tensor):
        """One step in **normalised** state space (used for WM fitting)."""
        an = norm.norm(a, self.a_mean, self.a_std)
        xn = norm.norm(xi, self.xi_mean, self.xi_std)
        out = self.dyn(torch.cat([sn, an, xn], dim=-1))
        if not self.semi_implicit:
            return sn + self.delta_limit * torch.tanh(out)

        # semi-implicit: network predicts accelerations, positions integrate.
        s = self.denorm_state(sn)
        acc = self.delta_limit * torch.tanh(out) * self.acc_scale
        p = s[..., 0:3]; q6 = s[..., 3:9]
        v = s[..., 9:12]; w = s[..., 12:15]
        q = s[..., 15:44]; qd = s[..., 44:73]
        dv, dw, dqd = acc[..., 0:3], acc[..., 3:6], acc[..., 6:35]
        v2 = v + dv * self.dt
        w2 = w + dw * self.dt
        qd2 = qd + dqd * self.dt
        p2 = p + 0.5 * (v + v2) * self.dt
        q2 = q + 0.5 * (qd + qd2) * self.dt
        R = _rot6d_to_mat(q6)
        R2 = R + _skew(0.5 * (w + w2) * self.dt) @ R
        q6_2 = _mat_to_6d(_orthonormalize(R2))
        s2 = torch.cat([p2, q6_2, v2, w2, q2, qd2], dim=-1)
        return self.norm_state(s2)

    def forward(self, s: torch.Tensor, a: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Predict next state ``s_{t+1}`` (raw units) for ``(B, D)`` inputs."""
        return self.denorm_state(self.forward_norm(self.norm_state(s), a, xi))

    def heads(
        self, s: torch.Tensor, a: torch.Tensor, goal_cur: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reward and termination *logit* for ``(B, ...)`` inputs -> ``(B, ...)``."""
        sn = norm.norm(s, self.s_mean, self.s_std)
        an = norm.norm(a, self.a_mean, self.a_std)
        gn = norm.norm(goal_cur, self.g_mean, self.g_std)
        h = self.trunk(torch.cat([sn, an, gn], dim=-1))
        return self.r_head(h).squeeze(-1), self.d_head(h).squeeze(-1)

    # -------------------------------------------------------------- helpers
    def parameters_count(self) -> int:
        return count_parameters(self)

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"xi_dim={self.xi_dim}, goal_frame_dim={self.goal_frame_dim}, "
            f"hidden={self.hidden}, delta_limit={self.delta_limit}"
        )


if __name__ == "__main__":  # tiny self-check
    wm = WorldModel()
    s = torch.randn(2, spec.STATE_DIM)
    a = torch.randn(2, spec.ACTION_DIM)
    xi = torch.randn(2, spec.XI_DIM) * 0.1
    g = torch.randn(2, spec.GOAL_FRAME_DIM)
    print(wm)
    print("params:", wm.parameters_count())
    print("s_next:", tuple(wm(s, a, xi).shape),
          "r/d:", tuple(t.shape for t in wm.heads(s, a, g)))
