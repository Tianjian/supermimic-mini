"""Motion pipeline tests (no GPU needed)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smini import build_model, motion, spec  # noqa: E402

XML = ROOT.parent / "assets" / "mujoco_menagerie" / "unitree_g1" / "scene_mjx.xml"


def test_motion_shapes_and_sanity():
    model = build_model.build_model(str(XML))
    bids = build_model.body_ids(model)
    lib = motion.MotionLibrary.from_dir(ROOT / "assets" / "motions")
    assert len(lib) >= 1, "no motions; run scripts/convert_motions.py"

    m = lib[0]
    assert m.fps == 50
    assert m.qpos.shape[1] == model.nq
    assert m.qvel.shape[1] == model.nv
    assert m.body_pos.shape == (len(m), spec.N_TRACKED, 3)
    assert m.body_quat.shape == (len(m), spec.N_TRACKED, 4)
    assert m.body_linvel.shape == (len(m), spec.N_TRACKED, 3)
    assert np.isfinite(m.qpos).all() and np.isfinite(m.qvel).all()
    assert np.isfinite(m.body_pos).all() and np.isfinite(m.body_angvel).all()

    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    q = m.qpos[:, 7:36]
    assert np.all(q >= lo - 1e-5) and np.all(q <= hi + 1e-5)

    g = m.goal_window(100)
    assert g.shape == (spec.GOAL_DIM,)
    assert np.isfinite(g).all()

    d = _data_at(model, m, 100)
    s = build_model.state_from_data(model, d, bids)
    assert s.shape == (spec.STATE_DIM,) == (283,)
    assert np.allclose(s[0:3], m.qpos[100, 0:3], atol=1e-6)
    assert np.allclose(s[9:12], s[73 + 9:73 + 12], atol=1e-6)
    assert np.allclose(s[12:15], s[73 + 12:73 + 15], atol=1e-6)

    obs = spec.pack_obs(s, np.zeros(spec.ACTION_DIM))
    assert obs.shape == (spec.OBS_DIM,)
    assert np.isfinite(obs).all()


def _data_at(model, m, t):
    import mujoco
    d = mujoco.MjData(model)
    d.qpos[:] = m.qpos[t]
    d.qvel[:] = m.qvel[t]
    mujoco.mj_forward(model, d)
    return d


if __name__ == "__main__":
    test_motion_shapes_and_sanity()
    print("PASS test_motion")
