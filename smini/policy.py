"""Deterministic actor / critic for SuperMimic-mini.

Policy (paper §3.3): the goal g_t (K=10 future reference frames, 730 dims) is
encoded once into a 32-dim latent z (motion encoder, (1024, 512, 256) -> 32),
then the decoder consumes ``[z, flatten(obs_hist)]`` (32 + 930 = 962 dims) and
outputs a 29-dim action in ``[-1, 1]`` via tanh.  Both are deterministic; action
noise is only used by the collector, never inside the policy.

``obs_hist`` is ``(B, hist, obs_dim)`` ordered oldest -> newest (the same
history-major order produced by stacking ``spec.pack_obs`` frames).

The critic V(s, g, xi) is a (512, 256) MLP over ``state 283 + goal 730 + xi 34``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import norm, spec
from .world_model import count_parameters, mlp


class Policy(nn.Module):
    """Goal-conditioned deterministic actor with history."""

    def __init__(
        self,
        obs_dim: int = spec.OBS_DIM,
        goal_dim: int = spec.GOAL_DIM,
        action_dim: int = spec.ACTION_DIM,
        latent: int = 32,
        hist: int = spec.HISTORY,
        norm: bool = False,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.goal_dim = int(goal_dim)
        self.action_dim = int(action_dim)
        self.latent = int(latent)
        self.hist = int(hist)
        self.norm = bool(norm)

        # motion encoder: 730 -> 1024 -> 512 -> 256 -> 32 (tanh after)
        self.goal_enc = mlp([self.goal_dim, 1024, 512, 256, self.latent], norm=norm)
        # decoder: 962 -> 1024 -> 512 -> 256 -> 29 (tanh after)
        self.dec = mlp(
            [self.latent + self.hist * self.obs_dim, 1024, 512, 256, self.action_dim],
            norm=norm,
        )
        # last layer starts small: bias prior dominates, but all layers still
        # receive gradient (unlike a hard zero init, which blocks the encoder).
        last = self.dec[-1]
        torch.nn.init.zeros_(last.bias)
        last.weight.data.mul_(0.1)
        self.register_buffer("g_mean", torch.zeros(goal_dim))
        self.register_buffer("g_std", torch.ones(goal_dim))
        self.register_buffer("o_mean", torch.zeros(obs_dim))
        self.register_buffer("o_std", torch.ones(obs_dim))

    # -------------------------------------------------------------- forward
    def set_norm(self, g_mean, g_std, o_mean, o_std) -> None:
        norm.set_buffer(self, "g", g_mean, g_std)
        norm.set_buffer(self, "o", o_mean, o_std)

    def encode_goal(self, goal: torch.Tensor) -> torch.Tensor:
        """(B, 730) goal -> (B, 32) latent motion code in ``[-1, 1]``."""
        return torch.tanh(self.goal_enc(norm.norm(goal, self.g_mean, self.g_std)))

    def forward(self, goal: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        """(B, 730), (B, hist, 93) -> (B, 29) unconstrained action setpoint."""
        if obs_hist.dim() != 3 or obs_hist.shape[-1] != self.obs_dim:
            raise ValueError(
                f"obs_hist must be (B, {self.hist}, {self.obs_dim}), "
                f"got {tuple(obs_hist.shape)}"
            )
        if obs_hist.shape[1] != self.hist:
            raise ValueError(
                f"obs_hist must hold {self.hist} frames, got {obs_hist.shape[1]}"
            )
        z = self.encode_goal(goal)
        oh = norm.norm(obs_hist, self.o_mean, self.o_std)
        # NOTE: no output tanh -- action_scale = 0.25*effort/kp is small, so
        # reference actions routinely exceed +-1 (legs ~5, wrists ~20).
        return self.dec(torch.cat([z, oh.reshape(obs_hist.shape[0], -1)], dim=-1))

    def set_action_bias(self, bias) -> None:
        """Initialise the last-layer bias to a prior mean action (speeds up cold start)."""
        import numpy as _np
        b = torch.as_tensor(_np.asarray(bias), dtype=self.dec[-1].bias.dtype,
                            device=self.dec[-1].bias.device)
        with torch.no_grad():
            self.dec[-1].bias.copy_(b)

    # -------------------------------------------------------------- helpers
    def parameters_count(self) -> int:
        return count_parameters(self)

    def extra_repr(self) -> str:
        return (
            f"obs_dim={self.obs_dim}, goal_dim={self.goal_dim}, "
            f"action_dim={self.action_dim}, latent={self.latent}, hist={self.hist}"
        )


class Critic(nn.Module):
    """State-value V(s, g, xi) with the plan's (512, 256) MLP."""

    def __init__(
        self,
        state_dim: int = spec.STATE_DIM,
        goal_dim: int = spec.GOAL_DIM,
        xi_dim: int = spec.XI_DIM,
        hidden: tuple[int, ...] = (512, 256),
        norm: bool = False,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.goal_dim = int(goal_dim)
        self.xi_dim = int(xi_dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.net = mlp([state_dim + goal_dim + xi_dim, *self.hidden, 1], norm=norm)
        # zero-init value head: the bootstrap starts at 0 (returns = reward sum)
        last = self.net[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        self.register_buffer("s_mean", torch.zeros(state_dim))
        self.register_buffer("s_std", torch.ones(state_dim))
        self.register_buffer("g_mean", torch.zeros(goal_dim))
        self.register_buffer("g_std", torch.ones(goal_dim))
        self.register_buffer("x_mean", torch.zeros(xi_dim))
        self.register_buffer("x_std", torch.ones(xi_dim))

    def forward(
        self, s: torch.Tensor, goal: torch.Tensor, xi: torch.Tensor
    ) -> torch.Tensor:
        """(B, 283), (B, 730), (B, 34) -> (B, 1) value estimate."""
        sn = norm.norm(s, self.s_mean, self.s_std)
        gn = norm.norm(goal, self.g_mean, self.g_std)
        xn = norm.norm(xi, self.x_mean, self.x_std)
        # bounded value: prevents the critic-bootstrap feedback loop from diverging
        return 5.0 * torch.tanh(self.net(torch.cat([sn, gn, xn], dim=-1)))

    def set_norm(self, s_mean, s_std, g_mean, g_std, x_mean, x_std) -> None:
        norm.set_buffer(self, "s", s_mean, s_std)
        norm.set_buffer(self, "g", g_mean, g_std)
        norm.set_buffer(self, "x", x_mean, x_std)

    def parameters_count(self) -> int:
        return count_parameters(self)

    def extra_repr(self) -> str:
        return f"state_dim={self.state_dim}, goal_dim={self.goal_dim}, xi_dim={self.xi_dim}"


if __name__ == "__main__":  # tiny self-check
    pol, crit = Policy(), Critic()
    goal = torch.randn(2, spec.GOAL_DIM)
    hist = torch.randn(2, spec.HISTORY, spec.OBS_DIM)
    print(pol)
    print("policy params:", pol.parameters_count(),
          "| z:", tuple(pol.encode_goal(goal).shape),
          "| a:", tuple(pol(goal, hist).shape))
    print("critic params:", crit.parameters_count(),
          "| v:", tuple(crit(torch.randn(2, spec.STATE_DIM), goal,
                             torch.randn(2, spec.XI_DIM)).shape))
