"""Per-dimension normalisation for states / actions / goals / observations."""
from __future__ import annotations

import numpy as np
import torch

# compact 73-dim state groups: root pos / quat6 / linvel / angvel / q / qd
STATE_GROUPS = ((0, 3), (3, 9), (9, 12), (12, 15), (15, 44), (44, 73))
STATE_DIM = 73


def _group_scale(std: np.ndarray, groups=STATE_GROUPS) -> np.ndarray:
    """Replace per-dim std inside each physical group by the group RMS std.

    This keeps a nearly-constant dimension (e.g. root z, quat components) from
    being amplified by a tiny per-dim std, and stops a high-variance group
    (joint velocities) from dominating the loss.
    """
    std = std.copy()
    for a, b in groups:
        if b <= std.shape[0]:
            std[a:b] = np.sqrt(np.mean(std[a:b] ** 2))
    return std


def compute_stats(buffer, keys=("state", "obs", "action", "goal", "xi")) -> dict:
    """Compute mean/std over all transitions currently in a ReplayBuffer."""
    out = {}
    for key in keys:
        parts = [ep[key] for ep in buffer.episodes]
        if not parts:
            raise ValueError("buffer is empty")
        x = np.concatenate(parts, axis=0).astype(np.float64)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.maximum(std, 1e-3)
        if key == "state" and x.shape[1] >= STATE_DIM:
            std = _group_scale(std)
        if key == "goal" and x.shape[1] % STATE_DIM == 0:
            g = std.reshape(-1, STATE_DIM)
            for i in range(g.shape[0]):
                g[i] = _group_scale(g[i])
            std = g.reshape(-1)
        out[key] = (mean.astype(np.float32), std.astype(np.float32))
        if key == "goal":
            gc = x[:, :STATE_DIM]
            gstd = _group_scale(np.maximum(gc.std(axis=0), 1e-3))
            out["goal_cur"] = (gc.mean(axis=0).astype(np.float32),
                               gstd.astype(np.float32))
    return out


def set_buffer(module, name: str, mean, std) -> None:
    """Register/replace a (mean, std) pair as buffers on a module."""
    dev = next(module.parameters()).device
    module.register_buffer(f"{name}_mean", torch.as_tensor(mean, device=dev))
    module.register_buffer(f"{name}_std", torch.as_tensor(std, device=dev))


def norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def denorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std + mean
