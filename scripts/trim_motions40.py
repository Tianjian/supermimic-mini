#!/usr/bin/env python3
"""Trim the T-pose / standing lead-in of each LAFAN1-G1 clip."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from smini import motion

SRC = ROOT / "assets/motions40"
DST = ROOT / "assets/motions40_trim"
LEAD = 25


def main():
    lib = motion.MotionLibrary.from_dir(SRC)
    DST.mkdir(parents=True, exist_ok=True)
    for m in lib.motions:
        v = np.linalg.norm(m.qvel[:, 0:3], axis=1)
        w = np.convolve(v, np.ones(10) / 10, mode="valid")
        idx = np.where(w > 0.5)[0]
        start = max(0, int(idx[0]) - LEAD) if len(idx) else 0
        m2 = motion.Motion(name=m.name, fps=m.fps, qpos=m.qpos[start:], qvel=m.qvel[start:],
                           body_pos=m.body_pos[start:], body_quat=m.body_quat[start:],
                           body_linvel=m.body_linvel[start:], body_angvel=m.body_angvel[start:])
        motion.save_motion(m2, DST / f"{m.name}.npz")
        print(f"{m.name:18s} trim {start:4d}/{len(m)} -> {len(m2)}  first speed {np.linalg.norm(m2.qvel[0,0:3]):.2f}")


if __name__ == "__main__":
    main()
