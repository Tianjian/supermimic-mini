"""LAFAN1 (G1 retarget) CSV -> 50 Hz motion tensors with MuJoCo FK.

CSV layout (no header, 36 columns, 30 fps):
    root_xyz(3) | root_quat_xyzw(4) | 29 joints (JOINT_NAMES order)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp

from . import build_model, spec


@dataclass
class Motion:
    name: str
    fps: int
    qpos: np.ndarray        # (T, 36)
    qvel: np.ndarray        # (T, 35)
    body_pos: np.ndarray    # (T, 14, 3)
    body_quat: np.ndarray   # (T, 14, 4) wxyz
    body_linvel: np.ndarray  # (T, 14, 3) world
    body_angvel: np.ndarray  # (T, 14, 3) world

    def __len__(self) -> int:
        return self.qpos.shape[0]

    def goal_frame(self, t: int) -> np.ndarray:
        t = min(max(t, 0), len(self) - 1)
        return spec.pack_goal_frame(self.qpos[t], self.qvel[t])

    def goal_window(self, t: int, k: int = spec.K_FUTURE) -> np.ndarray:
        """Frames t+1 .. t+k, clamped at the end; returns (k*73,)."""
        out = np.empty((k, spec.GOAL_FRAME_DIM), dtype=np.float64)
        for i in range(k):
            out[i] = self.goal_frame(t + 1 + i)
        return out.reshape(-1)


def _resample(raw: np.ndarray, fps_in: float, fps_out: float):
    T = raw.shape[0]
    duration = (T - 1) / fps_in
    n_out = int(round(duration * fps_out)) + 1
    t_out = np.linspace(0.0, duration, n_out)
    t_in = np.arange(T) / fps_in
    pos = np.stack([np.interp(t_out, t_in, raw[:, j]) for j in range(3)], axis=1)
    joints = np.stack([np.interp(t_out, t_in, raw[:, j]) for j in range(7, 36)], axis=1)
    slerp = Slerp(t_in, Rotation.from_quat(raw[:, 3:7]))  # scipy uses xyzw
    quat_wxyz = slerp(t_out).as_quat()[:, [3, 0, 1, 2]]
    return pos, quat_wxyz, joints, n_out


def _velocities(pos: np.ndarray, quat_wxyz: np.ndarray,
                joints: np.ndarray, dt: float):
    T = pos.shape[0]
    lin = np.gradient(pos, dt, axis=0, edge_order=1)
    jvel = np.gradient(joints, dt, axis=0, edge_order=1)
    ang_body = np.zeros((T, 3))
    rot = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]])  # xyzw
    for t in range(T):
        t0, t1 = max(t - 1, 0), min(t + 1, T - 1)
        dR = rot[t0].inv() * rot[t1]
        w_world = dR.as_rotvec() / ((t1 - t0) * dt)
        # MuJoCo free joint: angular velocity in body frame
        ang_body[t] = rot[t].inv().apply(w_world)
    return lin, ang_body, jvel


def convert_csv(csv_path: str | Path, model: mujoco.MjModel,
                bids: np.ndarray, fps_in: float = 30.0,
                fps_out: int = 50) -> Motion:
    raw = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 36:
        raise ValueError(f"{csv_path}: expected 36 columns, got {raw.shape}")
    pos, quat, joints, n = _resample(raw, fps_in, fps_out)
    dt = 1.0 / fps_out
    lin, ang_body, jvel = _velocities(pos, quat, joints, dt)

    qpos = np.empty((n, model.nq))
    qpos[:, 0:3] = pos
    qpos[:, 3:7] = quat
    qpos[:, 7:36] = joints
    qvel = np.empty((n, model.nv))
    qvel[:, 0:3] = lin
    qvel[:, 3:6] = ang_body
    qvel[:, 6:35] = jvel

    data = mujoco.MjData(model)
    bp = np.empty((n, spec.N_TRACKED, 3))
    bq = np.empty((n, spec.N_TRACKED, 4))
    bl = np.empty((n, spec.N_TRACKED, 3))
    ba = np.empty((n, spec.N_TRACKED, 3))
    for t in range(n):
        data.qpos[:] = qpos[t]
        data.qvel[:] = qvel[t]
        mujoco.mj_forward(model, data)
        bp[t], bq[t], bl[t], ba[t] = build_model.body_kinematics(model, data, bids)
    return Motion(name=Path(csv_path).stem, fps=fps_out, qpos=qpos, qvel=qvel,
                  body_pos=bp, body_quat=bq, body_linvel=bl, body_angvel=ba)


def save_motion(motion: Motion, path: str | Path) -> None:
    np.savez_compressed(
        path, name=motion.name, fps=motion.fps, qpos=motion.qpos,
        qvel=motion.qvel, body_pos=motion.body_pos, body_quat=motion.body_quat,
        body_linvel=motion.body_linvel, body_angvel=motion.body_angvel,
    )


def load_motion(path: str | Path) -> Motion:
    z = np.load(path, allow_pickle=False)
    return Motion(
        name=str(z["name"]), fps=int(z["fps"]), qpos=z["qpos"], qvel=z["qvel"],
        body_pos=z["body_pos"], body_quat=z["body_quat"],
        body_linvel=z["body_linvel"], body_angvel=z["body_angvel"],
    )


class MotionLibrary:
    def __init__(self, motions: list[Motion]):
        self.motions = motions

    @classmethod
    def from_dir(cls, path: str | Path) -> "MotionLibrary":
        files = sorted(Path(path).glob("*.npz"))
        return cls([load_motion(f) for f in files])

    def __len__(self) -> int:
        return len(self.motions)

    def __getitem__(self, i: int) -> Motion:
        return self.motions[i]

    @property
    def total_frames(self) -> int:
        return sum(len(m) for m in self.motions)
