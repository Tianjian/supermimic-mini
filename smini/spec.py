"""Shared constants, dimensions and geometry helpers for SuperMimic-mini.

Conventions
-----------
* quaternions are wxyz (MuJoCo convention), arrays are float32/float64 numpy.
* state s (283) = root pos(3) + root quat6d(6) + root linvel world(3)
  + root angvel world(3) + q_rel(29) + qd(29)
  + 14 tracked bodies x (pos3 + quat6d6 + linvel3 + angvel3).
* goal frame (73) = root pos3 + quat6d6 + linvel3 + angvel3 + q_rel29 + qd29.
* observation o (93) = q_rel+bias(29) + qd(29) + base angvel body(3)
  + projected gravity body(3) + previous action(29).
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- dimensions
N_JOINTS = 29
ACTION_DIM = 29
OBS_DIM = 93
ANCHOR_DIM = 9                    # root pos error 3 + relative-rotation 6D
OBS_DIM_ANCHOR = OBS_DIM + ANCHOR_DIM  # 102 (BeyondMimic/SuperMimic e3)
HISTORY = 10
K_FUTURE = 10

N_TRACKED = 14
STATE_DIM = 3 + 6 + 3 + 3 + N_JOINTS + N_JOINTS + N_TRACKED * 15  # 283
GOAL_FRAME_DIM = 3 + 6 + 3 + 3 + N_JOINTS + N_JOINTS             # 73
GOAL_DIM = K_FUTURE * GOAL_FRAME_DIM                             # 730
XI_DIM = 2 + N_JOINTS + 3                                        # 34

# Retargeted LAFAN1 G1 arm channels carry a zero-offset (arms abducted ~1.6 rad)
# that the MJCF default does not share.  Averaging all 29 joints makes the joint
# reward collapse to ~exp(-3.8) and the policy never learns legs/balance, so the
# v1 analytic reward weights legs+waist only (arms are still in obs/goal).
JOINT_REWARD_MASK = np.array([1.0] * 12 + [1.0] * 3 + [0.0] * 14)

# reward is O(0.1); train the reward head on scaled targets for a balanced loss
REWARD_SCALE = 10.0

# ---------------------------------------------------------------- names
JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

TRACKED_BODIES = [
    "pelvis",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]

# knees_bent keyframe of menagerie scene_mjx.xml, in JOINT_NAMES order
DEFAULT_Q = np.array([
    -0.3120, 0.0, 0.0, 0.6690, -0.3630, 0.0,
    -0.3120, 0.0, 0.0, 0.6690, -0.3630, 0.0,
    0.0, 0.0, 0.0730,
    0.2000, 0.2200, 0.0, 1.0000, 0.0, 0.0, 0.0,
    0.2000, -0.2200, 0.0, 1.0000, 0.0, 0.0, 0.0,
], dtype=np.float64)

# 0.25 * |forcerange| / kp, read from menagerie g1_mjx.xml (positive)
ACTION_SCALE = np.array([
    0.293333, 0.463333, 0.293333, 0.463333, 0.625, 0.625,
    0.293333, 0.463333, 0.293333, 0.463333, 0.625, 0.625,
    0.293333, 0.166667, 0.166667,
    0.083333, 0.083333, 0.083333, 0.083333, 0.3125, 0.0625, 0.0625,
    0.083333, 0.083333, 0.083333, 0.083333, 0.3125, 0.0625, 0.0625,
], dtype=np.float64)

# ---------------------------------------------------------------- quaternion / rotation
def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    s = 2.0 / max(n, 1e-12)
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1 - (xx + yy)],
    ])


def mat_to_quat6d(R: np.ndarray) -> np.ndarray:
    """First two columns of R, column-major: [R00,R10,R20, R01,R11,R21]."""
    return R[:, :2].reshape(6, order="F")


def quat_to_quat6d(q: np.ndarray) -> np.ndarray:
    return mat_to_quat6d(quat_to_mat(q))


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_to_mat(q) @ v


def projected_gravity(q: np.ndarray) -> np.ndarray:
    """Gravity direction [0,0,-1] expressed in the body frame."""
    R = quat_to_mat(q)
    return -R[2, :]


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return q[[3, 0, 1, 2]]


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return q[[1, 2, 3, 0]]


def world_angvel_from_qvel(qvel: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """MuJoCo free-joint qvel[3:6] is body-frame; convert to world."""
    return quat_to_mat(quat_wxyz) @ qvel[3:6]


# ---------------------------------------------------------------- state / obs packing
def pack_state(qpos: np.ndarray, qvel: np.ndarray,
               body_pos: np.ndarray | None = None,
               body_quat: np.ndarray | None = None,
               body_linvel: np.ndarray | None = None,
               body_angvel: np.ndarray | None = None) -> np.ndarray:
    """Pack the 283-dim state. body arrays are (N_TRACKED, ...)."""
    root_quat = qpos[3:7]
    out = np.empty(STATE_DIM, dtype=np.float64)
    out[0:3] = qpos[0:3]
    out[3:9] = quat_to_quat6d(root_quat)
    out[9:12] = qvel[0:3]
    out[12:15] = world_angvel_from_qvel(qvel, root_quat)
    out[15:44] = qpos[7:36] - DEFAULT_Q
    out[44:73] = qvel[6:35]
    o = 73
    if body_pos is not None:
        for b in range(N_TRACKED):
            out[o:o + 3] = body_pos[b]
            out[o + 3:o + 9] = quat_to_quat6d(body_quat[b])
            out[o + 9:o + 12] = body_linvel[b]
            out[o + 12:o + 15] = body_angvel[b]
            o += 15
    return out


def pack_goal_frame(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """73-dim reference frame; q is relative to DEFAULT_Q."""
    root_quat = qpos[3:7]
    out = np.empty(GOAL_FRAME_DIM, dtype=np.float64)
    out[0:3] = qpos[0:3]
    out[3:9] = quat_to_quat6d(root_quat)
    out[9:12] = qvel[0:3]
    out[12:15] = world_angvel_from_qvel(qvel, root_quat)
    out[15:44] = qpos[7:36] - DEFAULT_Q
    out[44:73] = qvel[6:35]
    return out


def pack_obs(state: np.ndarray, prev_action: np.ndarray,
             bias: np.ndarray | None = None,
             ref_pose6: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """93-dim observation from the 283-dim state.

    With ``ref_pose6=(p_ref, R_ref_6d)`` a 9-dim anchor-relative block is
    appended (root position error + R_ref^T R in 6D), giving 102 dims.
    """
    q_rel = state[15:44]
    qd = state[44:73]
    root_quat6 = state[3:9]
    # recover rotation from 6d (first two columns) via Gram-Schmidt
    R = _rot_from_6d(root_quat6)
    root_angvel_w = state[12:15]
    angvel_b = R.T @ root_angvel_w
    grav_b = -R[2, :]  # R^T @ [0,0,-1]
    if bias is not None:
        q_rel = q_rel + bias
    out = np.concatenate([q_rel, qd, angvel_b, grav_b, prev_action])
    if ref_pose6 is not None:
        p_ref, R6_ref = ref_pose6
        R_ref = _rot_from_6d(np.asarray(R6_ref, dtype=np.float64))
        p_err = np.asarray(p_ref, dtype=np.float64) - state[0:3]
        R_rel = R_ref.T @ R
        out = np.concatenate([out, p_err, mat_to_quat6d(R_rel)])
    return out


def _rot_from_6d(v6: np.ndarray) -> np.ndarray:
    a = v6[0:3]
    b = v6[3:6]
    a = a / max(np.linalg.norm(a), 1e-8)
    b = b - a * np.dot(a, b)
    b = b / max(np.linalg.norm(b), 1e-8)
    c = np.cross(a, b)
    return np.stack([a, b, c], axis=1)


def sample_xi(rng: np.random.Generator) -> np.ndarray:
    """34-dim sim context: 2 foot frictions, 29 encoder biases, 3 torso CoM offsets."""
    mu = rng.uniform(0.3, 1.2, size=2)
    bias = rng.uniform(-0.01, 0.01, size=N_JOINTS)
    com = rng.uniform([-0.025, -0.05, -0.05], [0.025, 0.05, 0.05])
    return np.concatenate([mu, bias, com]).astype(np.float64)


def xi_parts(xi: np.ndarray):
    return xi[0:2], xi[2:31], xi[31:34]
