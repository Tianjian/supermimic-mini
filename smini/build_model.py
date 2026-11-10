"""Build / configure the Unitree G1 29-DoF MuJoCo model for tracking."""
from __future__ import annotations

import re
import numpy as np
import mujoco

from . import spec

DEFAULT_XML = "assets/mujoco_menagerie/unitree_g1/scene_mjx.xml"


def build_model(xml_path: str = DEFAULT_XML, timestep: float = 0.005) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = timestep
    # a bit more solver work for contacts at 5 ms
    model.opt.iterations = 20
    model.opt.ls_iterations = 20
    return model


def body_ids(model: mujoco.MjModel) -> np.ndarray:
    ids = []
    for name in spec.TRACKED_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"tracked body not found: {name}")
        ids.append(bid)
    return np.asarray(ids, dtype=np.int32)


def foot_geom_ids(model: mujoco.MjModel) -> np.ndarray:
    out = []
    for i in range(model.ngeom):
        name = model.geom(i).name
        if re.match(r"^(left|right)_foot.*_collision$", name):
            out.append(i)
    if not out:
        raise ValueError("no foot collision geoms found")
    return np.asarray(out, dtype=np.int32)


def action_scale(model: mujoco.MjModel) -> np.ndarray:
    """0.25 * effort_limit / kp for each actuator (mjlab G1_ACTION_SCALE)."""
    kp = model.actuator_gainprm[:, 0]
    effort = np.abs(model.actuator_forcerange).max(axis=1)
    return 0.25 * effort / kp


def default_qpos(model: mujoco.MjModel) -> np.ndarray:
    """29 joint angles of the `knees_bent` keyframe, in joint order."""
    for i in range(model.nkey):
        if model.key(i).name == "knees_bent":
            return model.key_qpos[i][7:36].copy()
    raise ValueError("knees_bent keyframe not found")


def body_kinematics(model: mujoco.MjModel, data: mujoco.MjData, bids: np.ndarray):
    """Return (pos, quat_wxyz, linvel_world, angvel_world) for tracked bodies."""
    n = len(bids)
    pos = np.empty((n, 3))
    quat = np.empty((n, 4))
    lin = np.empty((n, 3))
    ang = np.empty((n, 3))
    res = np.zeros(6)
    for j, bid in enumerate(bids):
        pos[j] = data.xpos[bid]
        quat[j] = data.xquat[bid]
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, int(bid), res, 0)
        # MuJoCo spatial 6D vectors are [angular; linear] (same as d.cvel)
        ang[j] = res[0:3]
        lin[j] = res[3:6]
    return pos, quat, lin, ang


def state_from_data(model: mujoco.MjModel, data: mujoco.MjData,
                    bids: np.ndarray | None = None) -> np.ndarray:
    """Full 283-dim state: root + joints + 14 tracked-body kinematics."""
    if bids is None:
        bids = body_ids(model)
    pos, quat, lin, ang = body_kinematics(model, data, bids)
    return spec.pack_state(data.qpos, data.qvel, pos, quat, lin, ang)


def reset_to_reference(data: mujoco.MjData, qpos_ref: np.ndarray,
                       qvel_ref: np.ndarray) -> None:
    data.qpos[:] = qpos_ref
    data.qvel[:] = qvel_ref
    mujoco.mj_forward(data.model, data)


def apply_xi(model: mujoco.MjModel, xi: np.ndarray, base_com: np.ndarray) -> None:
    """Apply friction & torso CoM from sim context.

    Encoder bias is observation-level and handled by the env, not the model.
    """
    mu, _bias, com = spec.xi_parts(xi)
    gids = foot_geom_ids(model)
    for gid in gids:
        name = model.geom(gid).name
        model.geom_friction[gid, 0] = mu[0] if name.startswith("left") else mu[1]
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    model.body_ipos[torso] = base_com + com
