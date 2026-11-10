"""Tests for :class:`smini.env.TrackingEnv`.

Run directly (no pytest needed)::

    python tests/test_env.py

The functions are also pytest-compatible (``test_*``).  The suite covers the
API contract, the reference-alignment reward check, termination / truncation,
observation noise (Table A.1), action semantics, domain randomisation and a
1000-step throughput measurement.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

from smini import build_model, motion, spec  # noqa: E402
from smini.env import (  # noqa: E402
    DT,
    OBS_NOISE_ANGVEL,
    OBS_NOISE_GRAV,
    OBS_NOISE_Q,
    OBS_NOISE_QD,
    REWARD_MAX,
    EnvCfg,
    TrackingEnv,
)

XML_PATH = (ROOT.parent / "assets" / "mujoco_menagerie" / "unitree_g1"
            / "scene_mjx.xml")
MOTION_DIR = ROOT / "assets" / "motions"
NPZ_NAME = "walk1_subject1"

# deterministic / noiseless cfg used by the precise checks
CLEAN = dict(obs_noise=False, reset_noise=False,
             domain_randomization=False, action_noise=0.0)

_LIB: motion.MotionLibrary | None = None


def motion_lib() -> motion.MotionLibrary:
    global _LIB
    if _LIB is None:
        if not MOTION_DIR.is_dir():
            raise FileNotFoundError(f"motion dir not found: {MOTION_DIR}")
        _LIB = motion.MotionLibrary.from_dir(MOTION_DIR)
    return _LIB


def walk_id() -> int:
    for i, m in enumerate(motion_lib().motions):
        if m.name == NPZ_NAME:
            return i
    raise RuntimeError(f"{NPZ_NAME} not found in {MOTION_DIR}")


def make_env(seed: int = 0, **cfg_kw) -> TrackingEnv:
    if not XML_PATH.is_file():
        raise FileNotFoundError(f"G1 MJCF not found: {XML_PATH}")
    return TrackingEnv(str(XML_PATH), motion_lib(), EnvCfg(**cfg_kw), seed=seed)


def check_step_outputs(env: TrackingEnv, obs, reward, terminated,
                       truncated, info) -> None:
    """Common assertions for one ``step`` return value."""
    assert isinstance(reward, float), type(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert obs.shape == (spec.OBS_DIM,)
    expected = {
        "state": spec.STATE_DIM,
        "obs": spec.OBS_DIM,
        "goal": spec.GOAL_DIM,
        "xi": spec.XI_DIM,
        "action_exec": spec.ACTION_DIM,
    }
    for key, dim in expected.items():
        assert key in info, f"missing info[{key!r}]"
        assert info[key].shape == (dim,), (key, info[key].shape)
        assert np.isfinite(info[key]).all(), f"non-finite info[{key!r}]"
    assert info["motion_id"] == env.motion_id
    assert info["frame_idx"] == env.frame_idx
    assert info["reward"] == reward
    assert info["terminated"] == terminated
    assert info["truncated"] == truncated


# ---------------------------------------------------------------- tests
def test_reset_shapes_and_reference() -> None:
    """reset: obs(93) / state(283) / goal(730) / xi(34), goal is exact ref."""
    env = make_env(seed=1, **CLEAN)  # exact reference alignment
    lib = motion_lib()
    mid = walk_id()
    obs = env.reset(motion_id=mid, frame_idx=0)

    assert obs.shape == (spec.OBS_DIM,) == (93,)
    assert env.state().shape == (spec.STATE_DIM,) == (283,)
    assert env.goal().shape == (spec.GOAL_DIM,) == (730,)
    assert env.xi.shape == (spec.XI_DIM,) == (34,)
    assert env.motion_id == mid
    assert env.frame_idx == 0
    for arr in (obs, env.state(), env.goal(), env.xi):
        assert np.isfinite(arr).all()

    # goal must be bit-identical to the reference helper
    assert np.array_equal(env.goal(), lib[mid].goal_window(0))
    # noise-free reset puts the robot exactly at the reference frame
    ref_qpos = lib[mid].qpos[0]
    state = env.state()
    assert np.allclose(state[0:3], ref_qpos[0:3], atol=1e-12)
    assert np.allclose(state[15:44], ref_qpos[7:36] - spec.DEFAULT_Q,
                       atol=1e-12)

    # random resets keep the K-frame goal window inside the clip
    seen = set()
    for _ in range(30):
        env.reset()
        m = lib[env.motion_id]
        t0 = env.frame_idx
        seen.add(env.motion_id)
        assert 0 <= t0
        assert t0 + env.cfg.future_frames < len(m)
        assert env.goal().shape == (spec.GOAL_DIM,)
    assert len(seen) > 1, "reset should sample clips uniformly"


def test_explicit_reset_frame() -> None:
    env = make_env(seed=2, **CLEAN)
    mid = walk_id()
    for frame in (0, 1, 500, len(motion_lib()[mid]) - 1):
        env.reset(motion_id=mid, frame_idx=frame)
        assert env.frame_idx == frame
        assert np.array_equal(env.goal(),
                              motion_lib()[mid].goal_window(frame))
    try:
        env.reset(motion_id=mid, frame_idx=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative frame_idx must raise")


def test_step_info_contract() -> None:
    env = make_env(seed=3, **CLEAN)
    mid = walk_id()
    env.reset(motion_id=mid, frame_idx=100)
    obs, reward, terminated, truncated, info = env.step(np.zeros(29))
    check_step_outputs(env, obs, reward, terminated, truncated, info)
    assert env.frame_idx == 101
    assert info["frame_idx"] == 101
    assert np.array_equal(info["obs"], obs)
    # state/obs/goal returned by the accessors match the info dict
    assert np.array_equal(env.state(), info["state"])
    assert np.array_equal(env.goal(), info["goal"])
    assert np.array_equal(env.goal(),
                          motion_lib()[mid].goal_window(101))


def test_reward_now_is_max_at_reset() -> None:
    """Robot == reference pose, zero action -> maximum reward (5*dt = 0.1)."""
    assert abs(REWARD_MAX - 5.0 * DT) < 1e-12
    env = make_env(seed=4, **CLEAN)
    lib = motion_lib()
    for mid in range(len(lib)):
        for frame in (0, 1, 500, len(lib[mid]) // 2):
            env.reset(motion_id=mid, frame_idx=frame)
            r = env.reward_now()
            assert abs(r - REWARD_MAX) < 1e-9, (
                f"motion {mid} frame {frame}: reward_now={r} != {REWARD_MAX}")
            assert r > 0.9 * REWARD_MAX


def test_random_rollout() -> None:
    """100 random actions (full noise/DR): finite, bounded reward, no NaN."""
    env = make_env(seed=5)  # EnvCfg defaults: noise + DR on
    mid = walk_id()
    env.reset(motion_id=mid, frame_idx=0)
    rng = np.random.default_rng(0)
    prev_frame = env.frame_idx
    rewards = []
    for _ in range(100):
        # reference actions are O(1..20), so use a range well beyond [-1, 1]
        action = rng.uniform(-5.0, 5.0, spec.ACTION_DIM)
        obs, reward, terminated, truncated, info = env.step(action)
        check_step_outputs(env, obs, reward, terminated,
                           truncated, info)
        assert np.isfinite(reward)
        assert -3.0 <= reward <= 7.0, f"reward out of range: {reward}"
        assert np.isfinite(env.state()).all()
        assert np.isfinite(obs).all()
        assert env.frame_idx == prev_frame + 1
        prev_frame = env.frame_idx
        rewards.append(reward)
    assert np.isfinite(rewards).all()
    assert len(set(np.round(rewards, 12))) > 1


def test_obs_noise_table_a1() -> None:
    """Obs noise stays inside the Table A.1 uniform half-ranges."""
    mid = walk_id()
    clean = make_env(seed=6, obs_noise=False, reset_noise=False,
                     domain_randomization=False, action_noise=0.0)
    noisy = make_env(seed=6, obs_noise=True, reset_noise=False,
                     domain_randomization=False, action_noise=0.0)
    obs_clean = clean.reset(motion_id=mid, frame_idx=0)
    obs_noisy = noisy.reset(motion_id=mid, frame_idx=0)
    assert np.array_equal(clean.state(), noisy.state())

    d = obs_noisy - obs_clean
    eps = 1e-12
    assert np.abs(d[0:29]).max() <= OBS_NOISE_Q + eps
    assert np.abs(d[29:58]).max() <= OBS_NOISE_QD + eps
    assert np.abs(d[58:61]).max() <= OBS_NOISE_ANGVEL + eps
    assert np.abs(d[61:64]).max() <= OBS_NOISE_GRAV + eps
    assert np.array_equal(d[64:93], np.zeros(29)), "prev_action must be noiseless"
    assert np.abs(d).max() > 0.0, "obs noise should actually perturb"


def test_action_semantics() -> None:
    """Actions are unbounded policy outputs, clipped only to [-50, 50].

    ``action_scale = 0.25 * effort / kp`` is small, so the reference action
    ``(q_ref - q0) / scale`` has mean ~2.1 and max ~24; a [-1, 1] range could
    not represent it (lead decision, supersedes the original [-1, 1] spec).
    Only executed ``q_sp`` is clipped, to ``actuator_ctrlrange``.
    """
    mid = walk_id()
    # clean env: |a| > 1 passes through, echoed as the observation's prev_action
    env = make_env(seed=7, **CLEAN)
    env.reset(motion_id=mid, frame_idx=0)
    action = np.full(spec.ACTION_DIM, 3.0)
    obs, _, _, _, info = env.step(action)
    assert np.array_equal(obs[64:93], action)
    assert np.array_equal(info["action_exec"], action)
    action = np.full(spec.ACTION_DIM, -3.0)
    obs, _, _, _, info = env.step(action)
    assert np.array_equal(obs[64:93], action)
    assert np.array_equal(info["action_exec"], action)

    # safety clip at +/- 50 (not [-1, 1])
    obs, _, _, _, info = env.step(np.full(spec.ACTION_DIM, 123.0))
    assert np.array_equal(info["action_exec"], np.full(spec.ACTION_DIM, 50.0))
    assert np.array_equal(obs[64:93], np.full(spec.ACTION_DIM, 50.0))
    obs, _, _, _, info = env.step(np.full(spec.ACTION_DIM, -123.0))
    assert np.array_equal(info["action_exec"], np.full(spec.ACTION_DIM, -50.0))
    assert np.array_equal(obs[64:93], np.full(spec.ACTION_DIM, -50.0))

    # noisy env: executed = clean + N(0, 0.1), obs prev_action stays clean
    env = make_env(seed=8, action_noise=0.1, obs_noise=False)
    env.reset(motion_id=mid, frame_idx=0)
    action = np.full(spec.ACTION_DIM, 3.0)
    obs, _, _, _, info = env.step(action)
    assert np.array_equal(obs[64:93], action)
    assert np.abs(info["action_exec"] - action).max() < 0.5  # ~N(0, 0.1)
    assert not np.array_equal(info["action_exec"], action)
    assert np.isfinite(info["action_exec"]).all()

    try:
        env.step(np.zeros(spec.ACTION_DIM + 1))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong action shape must raise")


def test_termination_height() -> None:
    env = make_env(seed=9, **CLEAN)
    mid = walk_id()
    env.reset(motion_id=mid, frame_idx=100)
    env.data.qpos[2] += env.cfg.root_height_thresh + 0.05
    mujoco.mj_forward(env.model, env.data)
    _, _, terminated, truncated, info = env.step(np.zeros(29))
    assert terminated and info["terminated"]
    assert not truncated
    assert env.last_height_error > env.cfg.root_height_thresh


def test_termination_tilt() -> None:
    env = make_env(seed=10, **CLEAN)
    mid = walk_id()
    env.reset(motion_id=mid, frame_idx=100)
    q_old = env.data.qpos[3:7].copy()
    roll = np.pi / 2.0
    q_roll = np.array([np.cos(roll / 2), np.sin(roll / 2), 0.0, 0.0])
    q_new = spec.quat_mul(q_old, q_roll)
    env.data.qpos[3:7] = q_new / np.linalg.norm(q_new)
    mujoco.mj_forward(env.model, env.data)
    _, _, terminated, truncated, info = env.step(np.zeros(29))
    assert terminated and info["terminated"]
    assert not truncated
    assert env.last_tilt > env.cfg.root_tilt_thresh


def test_fall_with_zero_action() -> None:
    """Holding the standing pose while starting mid-walk must topple."""
    env = make_env(seed=11, **CLEAN)
    mid = walk_id()
    env.reset(motion_id=mid, frame_idx=0)
    reward0 = env.reward_now()
    assert reward0 > 0.9 * REWARD_MAX

    rewards = []
    terminated = False
    for _ in range(200):
        _, reward, term, trunc, info = env.step(np.zeros(29))
        rewards.append(reward)
        if term:
            terminated = info["terminated"]
            break
        assert not trunc, "truncated before terminating"
    assert terminated, "robot should terminate within 200 zero-action steps"
    assert rewards[-1] < 0.5 * reward0, "reward should drop as the robot falls"
    assert env.last_height_error > 0.25 or env.last_tilt > 0.8


def test_truncation() -> None:
    mid = walk_id()
    # episode_s=0.1 -> 5 control steps
    env = make_env(seed=12, episode_s=0.1, **CLEAN)
    env.reset(motion_id=mid, frame_idx=0)
    for step_i in range(4):
        _, _, _, trunc, _ = env.step(np.zeros(29))
        assert not trunc, f"truncated at step {step_i + 1}"
    _, _, _, trunc, info = env.step(np.zeros(29))
    assert trunc and info["truncated"]

    # end of clip: frame_idx + future_frames >= T
    env = make_env(seed=13, **CLEAN)
    n_frames = len(motion_lib()[mid])
    env.reset(motion_id=mid, frame_idx=n_frames - 1 - env.cfg.future_frames)
    _, _, _, trunc, info = env.step(np.zeros(29))
    assert trunc and info["truncated"]


def test_domain_randomization() -> None:
    mid = walk_id()
    env = make_env(seed=14)  # DR on
    env.reset(motion_id=mid, frame_idx=0)
    xi = env.xi
    mu, bias, com = spec.xi_parts(xi)
    assert xi.shape == (34,)
    assert 0.3 <= mu[0] <= 1.2 and 0.3 <= mu[1] <= 1.2
    assert np.abs(bias).max() <= 0.01
    assert np.all(com >= [-0.025, -0.05, -0.05])
    assert np.all(com <= [0.025, 0.05, 0.05])

    feet = build_model.foot_geom_ids(env.model)
    sides = np.array([0 if env.model.geom(g).name.startswith("left") else 1
                      for g in feet])
    assert np.allclose(env.model.geom_friction[feet, 0], mu[sides])
    # explicit foot/floor pairs must mirror xi too (scene friction override)
    assert env.model.npair > 0
    pair_fric = env.model.pair_friction
    for side, pairs in enumerate((env._left_pairs, env._right_pairs)):
        assert len(pairs) > 0
        assert np.allclose(pair_fric[pairs, 0], mu[side])

    # torso CoM offset actually applied on top of the captured base CoM
    torso = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY,
                              "torso_link")
    assert np.allclose(env.model.body_ipos[torso], env._base_com + com)

    # DR off -> neutral xi (friction 1.0, no bias / CoM offset)
    env = make_env(seed=15, **CLEAN)
    env.reset(motion_id=mid, frame_idx=0)
    assert np.array_equal(env.xi[0:2], np.ones(2))
    assert np.array_equal(env.xi[2:], np.zeros(32))
    assert np.allclose(env.model.geom_friction[feet, 0], 1.0)


def test_same_seed_is_deterministic() -> None:
    action = np.linspace(-1.0, 1.0, spec.ACTION_DIM)
    env_a, env_b = make_env(seed=21), make_env(seed=21)
    obs_a = env_a.reset()
    obs_b = env_b.reset()
    assert env_a.motion_id == env_b.motion_id
    assert env_a.frame_idx == env_b.frame_idx
    assert np.array_equal(obs_a, obs_b)
    for _ in range(5):
        out_a = env_a.step(action)
        out_b = env_b.step(action)
        assert out_a[1] == out_b[1]
        assert np.array_equal(out_a[0], out_b[0])
        assert np.array_equal(out_a[4]["action_exec"], out_b[4]["action_exec"])


# ---------------------------------------------------------------- runner
def _throughput(n_steps: int = 1000, seed: int = 42) -> tuple[float, float]:
    env = make_env(seed=seed)  # default cfg: noise + DR on
    env.reset(motion_id=walk_id(), frame_idx=0)
    rng = np.random.default_rng(0)
    actions = rng.uniform(-5.0, 5.0, (n_steps, spec.ACTION_DIM))
    t0 = time.perf_counter()
    for action in actions:
        env.step(action)
    elapsed = time.perf_counter() - t0
    return n_steps / elapsed, elapsed


def main() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} tests from {Path(__file__).name}")
    total0 = time.perf_counter()
    for test in tests:
        t0 = time.perf_counter()
        test()
        print(f"  PASS  {test.__name__:<34s} ({time.perf_counter() - t0:5.2f} s)")
    print(f"All {len(tests)} tests passed in {time.perf_counter() - total0:.2f} s")

    steps_per_s, elapsed = _throughput()
    print(f"Throughput: 1000 steps in {elapsed:.3f} s "
          f"-> {steps_per_s:.0f} steps/s "
          f"({steps_per_s / 50.0:.1f}x realtime)")


if __name__ == "__main__":
    main()
