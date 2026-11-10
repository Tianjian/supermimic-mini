#!/usr/bin/env python3
"""Remove the LAFAN1-G1 retarget arm zero offset (shoulder roll ~1.4 rad)."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, mujoco
from smini import motion, build_model, spec

XML = str(ROOT.parent / "assets/mujoco_menagerie/unitree_g1/scene_mjx.xml")
OUT = ROOT / "assets/motions_aligned"


def main():
    model = mujoco.MjModel.from_xml_path(XML)
    bids = build_model.body_ids(model)
    home = model.key_qpos[0][7:36].copy()
    lib = motion.MotionLibrary.from_dir(ROOT / "assets/motions")
    OUT.mkdir(parents=True, exist_ok=True)
    for mo in lib.motions:
        off = np.zeros(spec.N_JOINTS)
        off[15:29] = mo.qpos[0, 7 + 15:7 + 29] - home[15:29]   # arms only
        qpos = mo.qpos.copy(); qpos[:, 7:36] -= off
        d = mujoco.MjData(model)
        bp = np.empty((len(mo), spec.N_TRACKED, 3)); bq = np.empty((len(mo), spec.N_TRACKED, 4))
        bl = np.empty((len(mo), spec.N_TRACKED, 3)); ba = np.empty((len(mo), spec.N_TRACKED, 3))
        for t in range(len(mo)):
            d.qpos[:] = qpos[t]; d.qvel[:] = mo.qvel[t]
            mujoco.mj_forward(model, d)
            bp[t], bq[t], bl[t], ba[t] = build_model.body_kinematics(model, d, bids)
        m2 = motion.Motion(name=mo.name, fps=mo.fps, qpos=qpos, qvel=mo.qvel,
                           body_pos=bp, body_quat=bq, body_linvel=bl, body_angvel=ba)
        motion.save_motion(m2, OUT / f"{mo.name}.npz")
        print(f"{mo.name}: shoulder_roll offset L/R {off[16]:+.2f}/{off[23]:+.2f}, elbow {off[18]:+.2f}/{off[25]:+.2f}")


if __name__ == "__main__":
    main()
