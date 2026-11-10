"""Single-process vectorized wrapper (v1). Multiprocessing is a later optimization."""
from __future__ import annotations

import numpy as np

from .env import EnvCfg, TrackingEnv


def _stack(infos: list[dict]) -> dict:
    out = {}
    for k in infos[0]:
        vals = [d[k] for d in infos]
        if isinstance(vals[0], np.ndarray):
            out[k] = np.stack(vals)
        else:
            out[k] = np.asarray(vals)
    return out


class VecEnv:
    def __init__(self, xml_path: str, motion_lib, cfg: EnvCfg, num_envs: int = 8,
                 seed: int = 0):
        self.envs = [TrackingEnv(xml_path, motion_lib, cfg, seed=seed + i)
                     for i in range(num_envs)]
        self.num_envs = num_envs

    def reset(self):
        obs, infos = [], []
        for e in self.envs:
            o = e.reset()
            obs.append(o)
            infos.append({
                "state": e.state(), "obs": o, "goal": e.goal(), "xi": e.xi,
                "motion_id": e.motion_id, "frame_idx": e.frame_idx,
            })
        return np.stack(obs), _stack(infos)

    def reset_one(self, i: int, motion_id: int | None = None) -> np.ndarray:
        return self.envs[i].reset(motion_id=motion_id)

    def step(self, actions: np.ndarray):
        assert actions.shape == (self.num_envs, 29), actions.shape
        obs, rew, term, trunc, infos = [], [], [], [], []
        for i, e in enumerate(self.envs):
            o, r, te, tr, info = e.step(actions[i])
            obs.append(o); rew.append(r); term.append(te); trunc.append(tr)
            infos.append(info)
        return (np.stack(obs), np.asarray(rew, dtype=np.float64),
                np.asarray(term, dtype=bool), np.asarray(trunc, dtype=bool),
                _stack(infos))
