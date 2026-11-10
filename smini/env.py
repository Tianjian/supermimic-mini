"""Tracking environment for SuperMimic-mini (G1 29-DoF, 50 Hz control).

Semantics follow docs/05-plan.md D3/D5/D7:

* one env owns a private ``MjModel`` copy (per-env friction / torso CoM);
* ``reset`` samples a clip uniformly from the motion library and a start frame
  ``t0`` such that the ``future_frames`` goal window stays inside the clip;
* ``step`` applies a low-PD position action ``q_sp = q0 + alpha * a`` (with
  exploration noise), advances ``substeps`` physics steps (5 ms each by
  default -> 50 Hz), and returns the BeyondMimic 8-term tracking reward,
  mean over the 14 tracked bodies, all terms scaled by ``dt = 0.02 s``;
* ``state`` (283) / ``goal`` (730 = 10 x 73) / ``obs`` (93) follow
  ``smini.spec`` exactly.

Deviation worth knowing about: ``assets/.../scene_mjx.xml`` declares explicit
foot/floor contact pairs with ``friction="1 1"``; MuJoCo contact-pair friction
overrides ``geom_friction``, so ``build_model.apply_xi``'s friction
randomisation would have no effect.  The env therefore additionally mirrors
xi's two foot friction values onto the foot/floor contact pairs (see
``_sync_pair_friction``).  Torso CoM and encoder bias are untouched by that.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import build_model, spec

DT = 1.0 / 50.0  # control period (s)

# observation noise, 05-plan D3 / paper Table A.1 (uniform, half-ranges)
OBS_NOISE_Q = 0.01      # rad
OBS_NOISE_QD = 0.5      # rad/s
OBS_NOISE_ANGVEL = 0.2  # rad/s
OBS_NOISE_GRAV = 0.05   # projected gravity, unitless

# reset noise (conservative)
RESET_NOISE_XY = 0.05
RESET_NOISE_Z = 0.01
RESET_NOISE_RP = 0.1   # rad, roll/pitch
RESET_NOISE_YAW = 0.2  # rad
RESET_NOISE_Q = 0.1    # rad, joints
RESET_NOISE_QD = 0.5   # rad/s

# reward weights / sigmas, docs/05-plan.md D3
REWARD_SIGMA_BODY_POS = 0.3
REWARD_SIGMA_BODY_ORI = 0.4
REWARD_SIGMA_BODY_LINVEL = 1.0
REWARD_SIGMA_BODY_ANGVEL = 3.14
REWARD_SIGMA_ROOT_POS = 0.3
REWARD_SIGMA_ROOT_ORI = 0.4
REWARD_W_ROOT_POS = 0.5
REWARD_W_ROOT_ORI = 0.5
REWARD_W_ACTION_SMOOTH = -0.1
REWARD_W_JOINT_LIMIT = -2.0
# maximum possible one-step reward: (1 + 1 + 1 + 1 + 0.5 + 0.5) * DT
REWARD_MAX = 5.0 * DT


@dataclass
class EnvCfg:
    episode_s: float = 10.0
    substeps: int = 4
    action_noise: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    history: int = 10
    future_frames: int = 10
    root_height_thresh: float = 0.25
    root_tilt_thresh: float = 0.8
    anchor_obs: bool = False
    seed: int = 0


# ------------------------------------------------------------------ helpers
def _quat_to_mat_batch(q: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz -> (N, 3, 3), same convention as ``spec.quat_to_mat``."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = w * w + x * x + y * y + z * z
    s = 2.0 / np.maximum(n, 1e-12)
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    R = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1.0 - (yy + zz)
    R[:, 0, 1] = xy - wz
    R[:, 0, 2] = xz + wy
    R[:, 1, 0] = xy + wz
    R[:, 1, 1] = 1.0 - (xx + zz)
    R[:, 1, 2] = yz - wx
    R[:, 2, 0] = xz - wy
    R[:, 2, 1] = yz + wx
    R[:, 2, 2] = 1.0 - (xx + yy)
    return R


def _rot_from_6d_batch(v6: np.ndarray) -> np.ndarray:
    """(N, 6) -> (N, 3, 3); same Gram-Schmidt as ``spec._rot_from_6d``."""
    a = v6[:, 0:3]
    b = v6[:, 3:6]
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    b = b - a * np.sum(a * b, axis=1, keepdims=True)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    c = np.cross(a, b)
    return np.stack([a, b, c], axis=2)


def _rot_angle_sq(M: np.ndarray) -> np.ndarray:
    """Squared geodesic norm of ``log(M)`` for a batch of relative rotations.

    Uses ``atan2`` on the skew/trace pair instead of ``arccos`` so it stays
    accurate for the near-zero errors that dominate at reset.
    """
    v = np.stack(
        [M[:, 2, 1] - M[:, 1, 2], M[:, 0, 2] - M[:, 2, 0], M[:, 1, 0] - M[:, 0, 1]],
        axis=1,
    )
    sin_t = 0.5 * np.linalg.norm(v, axis=1)
    cos_t = 0.5 * (np.trace(M, axis1=1, axis2=2) - 1.0)
    return np.arctan2(sin_t, cos_t) ** 2


def _euler_small_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Small-angle RPY (world axes) -> wxyz unit quaternion."""
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def _neutral_xi() -> np.ndarray:
    """No-randomisation sim context: friction 1.0, zero bias/CoM offset."""
    xi = np.zeros(spec.XI_DIM, dtype=np.float64)
    xi[0:2] = 1.0
    return xi


# ------------------------------------------------------------------ env
class TrackingEnv:
    """Single CPU MuJoCo tracking env (one env = one model copy)."""

    def __init__(self, xml_path: str, motion_lib, cfg: EnvCfg = EnvCfg(),
                 seed: int = 0):
        self.cfg = cfg
        self._lib = motion_lib

        self._model = build_model.build_model(xml_path)
        # Keep control at 50 Hz no matter how many substeps are used.
        self._model.opt.timestep = DT / max(int(cfg.substeps), 1)
        self._data = mujoco.MjData(self._model)

        self._bids = build_model.body_ids(self._model)
        self._jids = np.array([
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in spec.JOINT_NAMES
        ], dtype=np.int32)
        if (self._jids < 0).any():
            raise ValueError("joint name not found in model")
        self._jmin = self._model.jnt_range[self._jids, 0].copy()
        self._jmax = self._model.jnt_range[self._jids, 1].copy()
        self._ctrl_min = self._model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_max = self._model.actuator_ctrlrange[:, 1].copy()
        if len(self._ctrl_min) != spec.ACTION_DIM:
            raise ValueError(f"expected {spec.ACTION_DIM} actuators, "
                             f"got {len(self._ctrl_min)}")

        torso = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,
                                  "torso_link")
        if torso < 0:
            raise ValueError("torso_link body not found")
        self._base_com = self._model.body_ipos[torso].copy()
        self._left_pairs, self._right_pairs = self._foot_floor_pairs()

        # `cfg.seed` is honoured when the explicit `seed` argument is left at
        # its default; an explicit seed always wins.
        effective_seed = int(seed) if seed != 0 else int(cfg.seed)
        self._rng = np.random.default_rng(effective_seed)

        self._episode_steps = max(int(round(cfg.episode_s * 50)), 1)

        # current episode state (valid after reset(); step() keeps it fresh)
        self._motion_id = 0
        self._motion = motion_lib[0]
        self._frame_idx = 0
        self._step_count = 0
        self._prev_action = np.zeros(spec.ACTION_DIM, dtype=np.float64)
        self._xi = _neutral_xi()
        self._state = np.zeros(spec.STATE_DIM, dtype=np.float64)
        self._body_kin = None
        self._obs_dim = spec.OBS_DIM_ANCHOR if cfg.anchor_obs else spec.OBS_DIM
        self._obs = np.zeros(self._obs_dim, dtype=np.float64)
        self._goal = self._goal_window(0)
        self.last_height_error = 0.0
        self.last_tilt = 0.0

    # -------------------------------------------------------------- props
    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def motion_id(self) -> int:
        return self._motion_id

    @property
    def frame_idx(self) -> int:
        return self._frame_idx

    @property
    def xi(self) -> np.ndarray:
        return self._xi.copy()

    # -------------------------------------------------------------- reset
    def reset(self, motion_id: int | None = None,
              frame_idx: int | None = None) -> np.ndarray:
        """Reset to a (possibly random) reference frame and return obs (93,).

        ``motion_id=None`` picks a clip uniformly; ``frame_idx=None`` picks
        ``t0`` uniformly such that ``t0 + future_frames < T``.  An explicit
        ``frame_idx`` (e.g. 0 for the deterministic eval protocol) is used
        as-is; it only has to be a valid frame of the clip.
        """
        cfg = self.cfg
        rng = self._rng
        if motion_id is None:
            mid = int(rng.integers(len(self._lib)))
        else:
            mid = int(motion_id)
            if not 0 <= mid < len(self._lib):
                raise ValueError(f"motion_id {mid} out of range "
                                 f"[0, {len(self._lib)})")
        motion = self._lib[mid]
        n_frames = len(motion)
        k = int(cfg.future_frames)

        if frame_idx is None:
            max_t0 = n_frames - 1 - k
            if max_t0 < 0:
                raise ValueError("motion shorter than the future window")
            t0 = int(rng.integers(0, max_t0 + 1))
        else:
            t0 = int(frame_idx)
            if not 0 <= t0 < n_frames:
                raise ValueError(f"frame_idx {t0} out of range "
                                 f"[0, {n_frames})")

        self._motion_id = mid
        self._motion = motion
        qpos = motion.qpos[t0]
        qvel = motion.qvel[t0]

        data = self._data
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        if cfg.reset_noise:
            self._perturb_reset(qpos, qvel)

        # Sim context: sampled per episode, fixed for the episode.  Applied
        # even without DR (neutral xi) so an env is always in a well-defined
        # parameter regime.
        xi = spec.sample_xi(rng) if cfg.domain_randomization else _neutral_xi()
        self._apply_xi(xi)
        self._xi = xi

        mujoco.mj_forward(self._model, data)

        self._frame_idx = t0
        self._step_count = 0
        self._prev_action[:] = 0.0
        self._refresh()
        return self._obs.copy()

    # -------------------------------------------------------------- step
    def step(self, action: np.ndarray):
        """Advance one control step (substeps physics steps).

        Returns the paper reward ``r_{t+1}``: it is evaluated on the post-step
        state against the reference frame reached by this step, so a collected
        transition is ``(s_t, a_t, r_{t+1}, s_{t+1})`` with ``s_{t+1}`` being
        the ``state``/``info['state']`` returned here.
        """
        cfg = self.cfg
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (spec.ACTION_DIM,):
            raise ValueError(f"action must have shape ({spec.ACTION_DIM},), "
                             f"got {action.shape}")
        # action_scale is 0.25*effort/kp, so the policy needs |a| >> 1;
        # clip only to a generous safety range (q_sp is clipped to ctrlrange).
        a_clean = np.clip(action, -50.0, 50.0)
        if cfg.action_noise > 0.0:
            # Exploration / data-collection noise (paper: eps ~ N(0, 0.1^2)).
            a_exec = a_clean + self._rng.normal(0.0, cfg.action_noise,
                                                spec.ACTION_DIM)
        else:
            a_exec = a_clean
        prev_action = self._prev_action

        q_sp = spec.DEFAULT_Q + spec.ACTION_SCALE * a_exec
        np.clip(q_sp, self._ctrl_min, self._ctrl_max, out=q_sp)
        self._data.ctrl[:] = q_sp
        for _ in range(int(cfg.substeps)):
            mujoco.mj_step(self._model, self._data)

        self._frame_idx += 1
        self._step_count += 1
        self._prev_action = a_clean
        self._refresh()

        t = self._ref_index()
        reward = self._reward(self._state, t, a_clean, prev_action)
        terminated = self._terminated(t)
        truncated = (
            self._step_count >= self._episode_steps
            or self._frame_idx + int(cfg.future_frames) >= len(self._motion)
        )

        info = {
            "state": self._state.copy(),
            "obs": self._obs.copy(),
            "goal": self._goal.copy(),
            "xi": self._xi.copy(),
            "action_exec": np.array(a_exec, dtype=np.float64),
            "motion_id": self._motion_id,
            "frame_idx": self._frame_idx,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }
        return self._obs.copy(), reward, terminated, truncated, info

    # -------------------------------------------------------------- state / goal
    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def state(self) -> np.ndarray:
        return self._state.copy()

    def goal(self) -> np.ndarray:
        return self._goal.copy()

    def reward_now(self) -> float:
        """Reward of the current state with zero action / zero prev action.

        Uses the current reference frame, so right after a noise-free reset
        (robot == reference pose) it returns the maximum one-step reward
        ``5 * dt = 0.1``.  Handy for eval and reward-alignment tests.
        """
        zeros = np.zeros(spec.ACTION_DIM, dtype=np.float64)
        return self._reward(self._state, self._ref_index(), zeros, zeros)

    # -------------------------------------------------------------- internals
    def _ref_index(self) -> int:
        return min(self._frame_idx, len(self._motion) - 1)

    def _refresh(self) -> None:
        cfg = self.cfg
        self._state = build_model.state_from_data(
            self._model, self._data, self._bids)
        ref_pose6 = None
        if cfg.anchor_obs:
            t = self._ref_index()
            ref_pose6 = (self._motion.qpos[t, 0:3],
                         spec.quat_to_quat6d(self._motion.qpos[t, 3:7]))
        obs = spec.pack_obs(self._state, self._prev_action,
                            bias=self._xi[2:31], ref_pose6=ref_pose6)
        if cfg.obs_noise:
            rng = self._rng
            obs[0:29] += rng.uniform(-OBS_NOISE_Q, OBS_NOISE_Q, 29)
            obs[29:58] += rng.uniform(-OBS_NOISE_QD, OBS_NOISE_QD, 29)
            obs[58:61] += rng.uniform(-OBS_NOISE_ANGVEL, OBS_NOISE_ANGVEL, 3)
            obs[61:64] += rng.uniform(-OBS_NOISE_GRAV, OBS_NOISE_GRAV, 3)
        self._obs = obs
        self._goal = self._goal_window(self._frame_idx)

    def _goal_window(self, t: int) -> np.ndarray:
        """Vectorised ``motion.goal_window(t, k=future_frames)`` (730,)."""
        motion = self._motion
        k = int(self.cfg.future_frames)
        idx = np.minimum(t + 1 + np.arange(k), len(motion) - 1)
        qpos = motion.qpos[idx]
        qvel = motion.qvel[idx]
        out = np.empty((k, spec.GOAL_FRAME_DIM), dtype=np.float64)
        out[:, 0:3] = qpos[:, 0:3]
        R = _quat_to_mat_batch(qpos[:, 3:7])
        # 6D rotation = first two columns, flattened column-major
        out[:, 3:9] = np.stack(
            [R[:, 0, 0], R[:, 1, 0], R[:, 2, 0],
             R[:, 0, 1], R[:, 1, 1], R[:, 2, 1]], axis=1)
        out[:, 9:12] = qvel[:, 0:3]
        out[:, 12:15] = np.matmul(R, qvel[:, 3:6, None])[:, :, 0]
        out[:, 15:44] = qpos[:, 7:36] - spec.DEFAULT_Q
        out[:, 44:73] = qvel[:, 6:35]
        return out.reshape(-1)

    def _perturb_reset(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        rng = self._rng
        data = self._data
        data.qpos[0:2] += rng.uniform(-RESET_NOISE_XY, RESET_NOISE_XY, 2)
        data.qpos[2] += rng.uniform(-RESET_NOISE_Z, RESET_NOISE_Z)
        roll, pitch, yaw = rng.uniform(
            [-RESET_NOISE_RP, -RESET_NOISE_RP, -RESET_NOISE_YAW],
            [RESET_NOISE_RP, RESET_NOISE_RP, RESET_NOISE_YAW])
        q_delta = _euler_small_quat(roll, pitch, yaw)
        q_new = spec.quat_mul(qpos[3:7], q_delta)
        data.qpos[3:7] = q_new / np.linalg.norm(q_new)
        data.qpos[7:] += rng.uniform(-RESET_NOISE_Q, RESET_NOISE_Q, 29)
        data.qvel[:] += rng.uniform(-RESET_NOISE_QD, RESET_NOISE_QD,
                                    data.qvel.shape[0])

    def _apply_xi(self, xi: np.ndarray) -> None:
        build_model.apply_xi(self._model, xi, self._base_com)
        self._sync_pair_friction(xi)

    def _foot_floor_pairs(self):
        """Contact-pair indices linking left/right foot geoms to the floor."""
        model = self._model
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        left, right = set(), set()
        try:
            feet = build_model.foot_geom_ids(model)
        except ValueError:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        for gid in feet:
            (left if model.geom(gid).name.startswith("left") else right).add(
                int(gid))
        left_pairs, right_pairs = [], []
        for i in range(model.npair):
            g1, g2 = int(model.pair_geom1[i]), int(model.pair_geom2[i])
            if floor < 0:
                break
            if (g1 == floor and g2 in left) or (g2 == floor and g1 in left):
                left_pairs.append(i)
            elif (g1 == floor and g2 in right) or (g2 == floor and g1 in right):
                right_pairs.append(i)
        return (np.asarray(left_pairs, dtype=np.int64),
                np.asarray(right_pairs, dtype=np.int64))

    def _sync_pair_friction(self, xi: np.ndarray) -> None:
        """Mirror xi foot friction onto explicit foot/floor contact pairs.

        The scene MJCF sets ``friction="1 1"`` on those pairs, which overrides
        ``geom_friction``; without this the friction part of xi is a no-op.
        """
        mu = xi[0:2]
        pair_fric = self._model.pair_friction
        if len(self._left_pairs):
            pair_fric[self._left_pairs, 0:2] = mu[0]
        if len(self._right_pairs):
            pair_fric[self._right_pairs, 0:2] = mu[1]

    def _reward(self, state: np.ndarray, t: int,
                action: np.ndarray, prev_action: np.ndarray) -> float:
        ref = self._motion
        body = state[73:].reshape(spec.N_TRACKED, 15)
        p, q6 = body[:, 0:3], body[:, 3:9]
        lin, ang = body[:, 9:12], body[:, 12:15]

        pos_err = np.mean(np.sum((ref.body_pos[t] - p) ** 2, axis=1))
        R_des = _quat_to_mat_batch(ref.body_quat[t])
        R = _rot_from_6d_batch(q6)
        ori_err = np.mean(_rot_angle_sq(R_des @ np.swapaxes(R, 1, 2)))
        lin_err = np.mean(np.sum((ref.body_linvel[t] - lin) ** 2, axis=1))
        ang_err = np.mean(np.sum((ref.body_angvel[t] - ang) ** 2, axis=1))

        root_pos_err = np.sum((ref.qpos[t, 0:3] - state[0:3]) ** 2)
        root_R_des = spec.quat_to_mat(ref.qpos[t, 3:7])
        root_R = _rot_from_6d_batch(state[3:9][None])[0]
        root_ori_err = _rot_angle_sq(
            (root_R_des @ root_R.T)[None])[0]

        q = state[15:44] + spec.DEFAULT_Q
        limit = np.sum(np.maximum(q - self._jmax, 0.0)
                       + np.maximum(self._jmin - q, 0.0))
        # penalise joint-setpoint change (alpha*a), not the raw unbounded
        # action: action_scale is small, raw |a| can reach ~20.
        smooth = np.sum((spec.ACTION_SCALE * (action - prev_action)) ** 2)

        r = (
            np.exp(-pos_err / REWARD_SIGMA_BODY_POS ** 2)
            + np.exp(-ori_err / REWARD_SIGMA_BODY_ORI ** 2)
            + np.exp(-lin_err / REWARD_SIGMA_BODY_LINVEL ** 2)
            + np.exp(-ang_err / REWARD_SIGMA_BODY_ANGVEL ** 2)
            + REWARD_W_ROOT_POS * np.exp(-root_pos_err / REWARD_SIGMA_ROOT_POS ** 2)
            + REWARD_W_ROOT_ORI * np.exp(-root_ori_err / REWARD_SIGMA_ROOT_ORI ** 2)
            + REWARD_W_ACTION_SMOOTH * smooth
            + REWARD_W_JOINT_LIMIT * limit
        )
        return float(r * DT)

    def _terminated(self, t: int) -> bool:
        ref = self._motion
        qpos = self._data.qpos
        self.last_height_error = float(abs(qpos[2] - ref.qpos[t, 2]))
        self.last_tilt = float(abs(
            spec.projected_gravity(ref.qpos[t, 3:7])[2]
            - spec.projected_gravity(qpos[3:7])[2]))
        return (self.last_height_error > self.cfg.root_height_thresh
                or self.last_tilt > self.cfg.root_tilt_thresh)
