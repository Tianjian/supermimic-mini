"""Episode-based replay buffer (CPU) for SuperMimic-mini.

Storage layout
--------------
Each episode is a dict of numpy arrays with a shared length ``T``::

    state      (T, 283)   world state  s_t                  (spec.pack_state)
    obs        (T, 93)    observation  o_t                  (spec.pack_obs)
    action     (T, 29)    *executed* action a_t (noise included)
    reward     (T,)       r_t
    terminated (T,)       hard termination flag (float/bool)
    truncated  (T,)       time-limit / truncation flag
    xi         (T, 34)    per-episode sim context (constant within an episode)
    goal       (T, 730)   K=10 future reference frames starting at t
    motion_id  int
    frame_idx  (T,)       frame index inside the reference clip

Everything is stored as float32 (contiguous) to keep the CPU buffer small:
a 64k-transition buffer is ~0.3 GB.  Episodes are dropped oldest-first once
``sum(len(ep))`` exceeds ``capacity``; a single episode larger than the whole
buffer is kept intact (it is the freshest data) and only a warning is emitted.

All sampling is *within* one episode, so windows, observation histories and
goal sequences are contiguous by construction and never cross an episode
boundary.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch

from . import spec

# keys that must be present in an episode dict
_REQUIRED = ("state", "obs", "action", "reward", "terminated", "truncated", "xi", "goal")
# (key, trailing dim) for the 2-D arrays
_DIMS = (
    ("state", spec.STATE_DIM),
    ("obs", spec.OBS_DIM),
    ("action", spec.ACTION_DIM),
    ("xi", spec.XI_DIM),
    ("goal", spec.GOAL_DIM),
)

_warned_cuda = False


def _resolve_device(device) -> torch.device:
    """Fall back to CPU (once) if a cuda device is requested but unavailable."""
    global _warned_cuda
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        if not _warned_cuda:
            warnings.warn(
                "ReplayBuffer: cuda requested but not available, using CPU",
                RuntimeWarning, stacklevel=3,
            )
            _warned_cuda = True
        dev = torch.device("cpu")
    return dev


def _to_tensor(arr: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(arr).to(_resolve_device(device))


class ReplayBuffer:
    """Fixed-capacity replay buffer of complete episodes (CPU memory)."""

    def __init__(self, capacity: int = 100_000, seed: int | None = None):
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.episodes: list[dict] = []
        self._n = 0
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ io
    def __len__(self) -> int:
        """Total number of transitions currently stored."""
        return self._n

    def clear(self) -> None:
        self.episodes.clear()
        self._n = 0

    def add_episode(self, ep: dict) -> None:
        """Append one episode; drop the oldest episodes to respect ``capacity``."""
        ep = self._prepare(ep)
        self.episodes.append(ep)
        self._n += ep["T"]
        while len(self.episodes) > 1 and self._n > self.capacity:
            old = self.episodes.pop(0)
            self._n -= old["T"]
        if self._n > self.capacity:
            warnings.warn(
                f"episode of {self._n} transitions exceeds capacity {self.capacity}; "
                "keeping it whole (cannot drop the freshest data)",
                RuntimeWarning, stacklevel=2,
            )

    def _prepare(self, ep: dict) -> dict:
        missing = [k for k in _REQUIRED if k not in ep]
        if missing:
            raise ValueError(
                f"episode is missing required keys {missing}; "
                f"need {list(_REQUIRED)}"
            )
        reward = np.asarray(ep["reward"]).reshape(-1)
        T = int(reward.shape[0])
        if T == 0:
            raise ValueError("episode is empty (reward has length 0)")

        out: dict = {"T": T}
        for key, dim in _DIMS:
            arr = np.asarray(ep[key])
            if key == "xi" and arr.shape == (dim,):
                # tolerate a constant xi given as a single 34-vector
                arr = np.broadcast_to(arr, (T, dim))
            if arr.shape != (T, dim):
                raise ValueError(
                    f"episode[{key!r}] must have shape ({T}, {dim}) "
                    f"(T from reward), got {tuple(arr.shape)}"
                )
            out[key] = np.ascontiguousarray(arr, dtype=np.float32)

        for key in ("terminated", "truncated"):
            arr = np.asarray(ep[key]).reshape(-1)
            if arr.shape[0] != T:
                raise ValueError(
                    f"episode[{key!r}] must have shape ({T},), got {tuple(arr.shape)}"
                )
            out[key] = np.ascontiguousarray(arr, dtype=np.float32)

        out["reward"] = np.ascontiguousarray(reward, dtype=np.float32)

        frame_idx = ep.get("frame_idx")
        if frame_idx is None:
            frame_idx = np.arange(T, dtype=np.int64)
        else:
            frame_idx = np.asarray(frame_idx).reshape(-1)
            if frame_idx.shape[0] != T:
                raise ValueError(
                    f"episode['frame_idx'] must have shape ({T},), got {tuple(frame_idx.shape)}"
                )
        out["frame_idx"] = np.ascontiguousarray(frame_idx, dtype=np.int64)
        out["motion_id"] = int(ep.get("motion_id", -1))

        # cheap sanity: no NaN/inf in the float payload
        for key in ("state", "obs", "action", "goal", "xi", "reward"):
            if not np.isfinite(out[key]).all():
                raise ValueError(f"episode[{key!r}] contains NaN/inf")
        return out

    # ------------------------------------------------------------ sampling
    def _sample_episode_starts(
        self, batch: int, counts: np.ndarray, span: int, offset: int = 0
    ):
        """Pick ``batch`` (episode, start) pairs, weighted by #valid starts."""
        keep = counts > 0
        if not keep.any():
            raise ValueError(
                f"no episode holds >= {span} transitions; "
                f"add longer episodes or lower k/horizon"
            )
        idx_all = np.flatnonzero(keep)
        p = counts[idx_all].astype(np.float64)
        p /= p.sum()
        ep_idx = idx_all[self.rng.choice(idx_all.shape[0], size=batch, p=p)]
        # ``offset`` is the first legal start (hist-1 for history sampling)
        starts = int(offset) + self.rng.integers(0, counts[ep_idx])
        return ep_idx, starts

    def sample_windows(self, batch: int, k: int = 8, device="cuda") -> dict:
        """Sample ``batch`` unrolled windows of length ``k`` for world-model fitting.

        Returns float32 tensors on ``device``::

            s        (B, 283)        first state of the window
            a        (B, k, 29)      executed actions
            xi       (B, 34)         sim context (constant inside one episode)
            s_next   (B, k, 283)     states s_{i+1} .. s_{i+k}
            r        (B, k)          rewards
            d        (B, k)          terminated (float, not truncated)
            goal_cur (B, k, 73)      current reference frame of each step,
                                     i.e. goal[i+j][:73] for the reward / term head

        ``s`` and ``s_next[:, 0]`` are consecutive by construction.
        """
        k = int(k)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        batch = int(batch)
        counts = np.array(
            [max(0, ep["T"] - k) for ep in self.episodes], dtype=np.int64
        )
        ep_idx, starts = self._sample_episode_starts(batch, counts, k + 1)

        s = np.empty((batch, spec.STATE_DIM), dtype=np.float32)
        a = np.empty((batch, k, spec.ACTION_DIM), dtype=np.float32)
        xi = np.empty((batch, spec.XI_DIM), dtype=np.float32)
        s_next = np.empty((batch, k, spec.STATE_DIM), dtype=np.float32)
        r = np.empty((batch, k), dtype=np.float32)
        d = np.empty((batch, k), dtype=np.float32)
        goal_cur = np.empty((batch, k, spec.GOAL_FRAME_DIM), dtype=np.float32)

        for b in range(batch):
            ep = self.episodes[ep_idx[b]]
            i = int(starts[b])
            j = i + k
            s[b] = ep["state"][i]
            s_next[b] = ep["state"][i + 1:j + 1]
            a[b] = ep["action"][i:j]
            r[b] = ep["reward"][i:j]
            d[b] = ep["terminated"][i:j]
            xi[b] = ep["xi"][i]
            goal_cur[b] = ep["goal"][i:j, :spec.GOAL_FRAME_DIM]

        return {
            "s": _to_tensor(s, device),
            "a": _to_tensor(a, device),
            "xi": _to_tensor(xi, device),
            "s_next": _to_tensor(s_next, device),
            "r": _to_tensor(r, device),
            "d": _to_tensor(d, device),
            "goal_cur": _to_tensor(goal_cur, device),
        }

    def sample_starts(self, batch: int, horizon: int, device="cuda") -> dict:
        """Sample ``batch`` imagination start points with 10 frames of obs history.

        Returns float32 tensors on ``device``::

            s         (B, 283)          state at the start step
            obs_hist  (B, 10, 93)       obs[t-9 .. t] (oldest first)
            goal_seq  (B, horizon, 730) goals starting at the start step
            xi        (B, 34)           sim context
            goal_cur  (B, 73)           goal_seq[:, 0, :73] (reward/term head frame)

        Only start steps with a full 10-frame history and a full ``horizon``
        of goal frames are used.  Cold start fallback: if *no* episode is long
        enough for a full history (early in training, when the robot falls
        quickly), starts are drawn from ``[0, T - horizon]`` and the missing
        leading frames are filled with the episode's first observation -- the
        same convention the collector uses when it resets
        (``np.repeat(obs, HISTORY)``).
        """
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        batch = int(batch)
        hist = spec.HISTORY
        # start i must satisfy i >= hist-1 and i <= T-horizon
        counts = np.array(
            [max(0, ep["T"] - horizon - hist + 2) for ep in self.episodes],
            dtype=np.int64,
        )
        pad_hist = not counts.any()
        if pad_hist:  # cold start: no episode has a full 10-frame history yet
            counts = np.array(
                [max(0, ep["T"] - horizon + 1) for ep in self.episodes],
                dtype=np.int64,
            )
            if not counts.any():
                raise ValueError(
                    f"no episode holds >= {horizon} transitions; "
                    f"cannot imagine for {horizon} steps"
                )
            offset = 0
            span = horizon
        else:
            offset = hist - 1
            span = horizon + hist - 1
        ep_idx, starts = self._sample_episode_starts(batch, counts, span, offset=offset)

        s = np.empty((batch, spec.STATE_DIM), dtype=np.float32)
        obs_hist = np.empty((batch, hist, spec.OBS_DIM), dtype=np.float32)
        goal_seq = np.empty((batch, horizon, spec.GOAL_DIM), dtype=np.float32)
        xi = np.empty((batch, spec.XI_DIM), dtype=np.float32)

        for b in range(batch):
            ep = self.episodes[ep_idx[b]]
            i = int(starts[b])
            s[b] = ep["state"][i]
            if pad_hist and i < hist - 1:
                n_real = i + 1
                obs_hist[b, hist - n_real:] = ep["obs"][:n_real]
                obs_hist[b, :hist - n_real] = ep["obs"][0]
            else:
                obs_hist[b] = ep["obs"][i - hist + 1:i + 1]
            goal_seq[b] = ep["goal"][i:i + horizon]
            xi[b] = ep["xi"][i]

        goal_cur = np.ascontiguousarray(goal_seq[:, 0, :spec.GOAL_FRAME_DIM])
        return {
            "s": _to_tensor(s, device),
            "obs_hist": _to_tensor(obs_hist, device),
            "goal_seq": _to_tensor(goal_seq, device),
            "xi": _to_tensor(xi, device),
            "goal_cur": _to_tensor(goal_cur, device),
        }
