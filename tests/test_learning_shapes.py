"""Shape / continuity / smoke tests for the SuperMimic-mini learning core.

Synthetic data only: no ``env.py``, no MuJoCo, no replay of real episodes.

Run directly::

    /home/toby/Desktop/mujoco/.venv/bin/python tests/test_learning_shapes.py

prints ``PASS`` at the end (also pytest-compatible: every ``test_*`` takes no
arguments and asserts).

Sizes are intentionally small (B=16, T=40, H=8, k=4); the B=512/H=32 VRAM
budget is benchmarked separately, not here.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smini import spec
from smini.imagine import (
    clip_gradients,
    critic_loss,
    gae_targets,
    imagine,
    obs_from_state_torch,
    policy_loss,
)
from smini.policy import Critic, Policy
from smini.replay import ReplayBuffer
from smini.world_model import WorldModel

BATCH, T, H, K = 16, 40, 8, 4
# per-episode base offsets so that any cross-episode leak shows up as a jump
ST_BASE, OBS_BASE, GOAL_BASE, ACT_BASE = 1000.0, 2000.0, 3000.0, 100.0


# ------------------------------------------------------------------ helpers
def _episode(idx: int, terminated_period: int = 7, length: int = T) -> dict:
    """Ramp episode: value at time t is ``idx*BASE + t`` in every dimension."""
    t = np.arange(length, dtype=np.float32)

    def ramp(base: float, dim: int) -> np.ndarray:
        return np.repeat((idx * base + t)[:, None], dim, axis=1).astype(np.float32)

    return {
        "state": ramp(ST_BASE, spec.STATE_DIM),
        "obs": ramp(OBS_BASE, spec.OBS_DIM),
        "action": ramp(ACT_BASE, spec.ACTION_DIM),
        "goal": ramp(GOAL_BASE, spec.GOAL_DIM),
        "reward": (idx + 0.1 * t).astype(np.float32),
        "terminated": (t % terminated_period == 0).astype(np.float32),
        "truncated": np.zeros(length, dtype=np.float32),
        "xi": np.full(spec.XI_DIM, float(idx), dtype=np.float32),
        "motion_id": idx,
        "frame_idx": t.astype(np.int64),
    }


def _buffer() -> ReplayBuffer:
    buf = ReplayBuffer(capacity=10_000, seed=0)
    buf.add_episode(_episode(0))
    buf.add_episode(_episode(1))
    return buf


def _ep_index_from_state(s: torch.Tensor) -> torch.Tensor:
    return torch.div(s[:, 0], ST_BASE, rounding_mode="floor").long()


def _start_from_windows(w: dict) -> torch.Tensor:
    ep = _ep_index_from_state(w["s"])
    return w["s"][:, 0] - ep.float() * ST_BASE


def _small_starts(batch: int = BATCH, horizon: int = H, seed: int = 0) -> dict:
    """Realistic-magnitude imagination starts (not from the ramp buffer)."""
    g = torch.Generator().manual_seed(seed)
    n_goal = spec.GOAL_DIM
    return {
        "s": torch.randn(batch, spec.STATE_DIM, generator=g) * 0.1,
        "obs_hist": torch.randn(batch, spec.HISTORY, spec.OBS_DIM, generator=g) * 0.1,
        "goal_seq": torch.randn(batch, horizon, n_goal, generator=g) * 0.1,
        "xi": torch.randn(batch, spec.XI_DIM, generator=g) * 0.1,
    }


# --------------------------------------------------------------- replay tests
def test_replay_windows():
    buf = _buffer()
    assert len(buf) == 2 * T, f"len={len(buf)}"
    w = buf.sample_windows(BATCH, k=K, device="cpu")
    for key, shape in {
        "s": (BATCH, spec.STATE_DIM),
        "a": (BATCH, K, spec.ACTION_DIM),
        "xi": (BATCH, spec.XI_DIM),
        "s_next": (BATCH, K, spec.STATE_DIM),
        "r": (BATCH, K),
        "d": (BATCH, K),
        "goal_cur": (BATCH, K, spec.GOAL_FRAME_DIM),
    }.items():
        assert tuple(w[key].shape) == shape, f"{key}: {tuple(w[key].shape)} != {shape}"
        assert w[key].dtype == torch.float32, f"{key} dtype {w[key].dtype}"

    # contiguity: within a window every step advances the ramp by exactly 1
    s, s_next, a = w["s"], w["s_next"], w["a"]
    assert torch.allclose(s_next[:, 0] - s, torch.ones(BATCH, spec.STATE_DIM))
    for j in range(1, K):
        assert torch.allclose(
            s_next[:, j] - s_next[:, j - 1], torch.ones(BATCH, spec.STATE_DIM)
        ), f"window step {j} is not contiguous (episode boundary leak?)"
    for j in range(1, K):
        assert torch.allclose(a[:, j, 0] - a[:, j - 1, 0], torch.ones(BATCH))
    assert torch.allclose(
        a[:, 0, 1:], a[:, 0, 1:2].expand(BATCH, spec.ACTION_DIM - 1)
    ), "ramp episodes should be identical in every dimension"

    # r / d / goal_cur must be indexed at the same step as s_next
    ep = _ep_index_from_state(s).float()
    start = _start_from_windows(w)                                # (B,)
    for j in range(K):
        exp_r = ep + 0.1 * (start + j)
        assert torch.allclose(w["r"][:, j], exp_r, atol=1e-5), f"reward index at step {j}"
        exp_d = (((start.long() + j) % 7) == 0).float()
        assert torch.allclose(w["d"][:, j], exp_d), f"terminated index at step {j}"
        exp_goal = ep * GOAL_BASE + start + j
        assert torch.allclose(w["goal_cur"][:, j, 0], exp_goal, atol=1e-4), (
            f"goal_cur index at step {j}"
        )
        assert torch.allclose(
            w["goal_cur"][:, j, 1:],
            w["goal_cur"][:, j, 1:2].expand(BATCH, spec.GOAL_FRAME_DIM - 1),
        ), "goal_cur must be a single 73-dim reference frame per step"

    # every sample stays inside one episode (values within [base, base+T))
    assert (s >= 0).all() and (s < 2 * ST_BASE).all()


def test_replay_starts():
    buf = _buffer()
    st = buf.sample_starts(BATCH, H, device="cpu")
    shapes = {
        "s": (BATCH, spec.STATE_DIM),
        "obs_hist": (BATCH, spec.HISTORY, spec.OBS_DIM),
        "goal_seq": (BATCH, H, spec.GOAL_DIM),
        "xi": (BATCH, spec.XI_DIM),
        "goal_cur": (BATCH, spec.GOAL_FRAME_DIM),
    }
    for key, shape in shapes.items():
        assert tuple(st[key].shape) == shape, f"{key}: {tuple(st[key].shape)}"
        assert st[key].dtype == torch.float32

    ep = _ep_index_from_state(st["s"]).float()
    start = st["obs_hist"][:, -1, 0] - ep * OBS_BASE
    # 10-frame history, oldest -> newest
    assert torch.allclose(
        st["obs_hist"][:, -1, 0] - st["obs_hist"][:, 0, 0], torch.full((BATCH,), 9.0)
    )
    assert torch.allclose(st["obs_hist"][:, 0, 1:], st["obs_hist"][:, 0, 1:2].expand(BATCH, spec.OBS_DIM - 1))
    # full history + full horizon available
    assert int(start.min()) >= spec.HISTORY - 1, f"min start {int(start.min())}"
    assert int(start.max()) <= T - H, f"max start {int(start.max())}"
    # goal_seq starts at the same step and is contiguous
    assert torch.allclose(st["goal_seq"][:, 0, 0] - ep * GOAL_BASE, start, atol=1e-4)
    for j in range(1, H):
        assert torch.allclose(
            st["goal_seq"][:, j, 0] - st["goal_seq"][:, j - 1, 0], torch.ones(BATCH)
        ), f"goal_seq step {j} is not contiguous"
    assert torch.allclose(st["goal_cur"], st["goal_seq"][:, 0, : spec.GOAL_FRAME_DIM])
    # xi is constant inside an episode
    assert torch.allclose(st["xi"], ep.unsqueeze(-1).expand(BATCH, spec.XI_DIM))


def test_replay_eviction_and_validation():
    buf = ReplayBuffer(capacity=60, seed=1)
    buf.add_episode(_episode(0))
    buf.add_episode(_episode(1))
    assert len(buf) == T, f"oldest episode should be dropped, len={len(buf)}"
    w = buf.sample_windows(BATCH, k=K, device="cpu")
    assert (w["s"][:, 0] >= ST_BASE).all(), "evicted episode 0 still sampled"
    assert (w["s"][:, 0] < 2 * ST_BASE).all()

    # a whole-episode add beyond capacity keeps the freshest episode intact
    buf2 = ReplayBuffer(capacity=10)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        buf2.add_episode(_episode(3))
    assert len(buf2) == T
    assert any("exceeds capacity" in str(w.message) for w in caught)

    for bad in (
        {k: v for k, v in _episode(0).items() if k != "goal"},
        {**_episode(0), "action": np.zeros((T, spec.ACTION_DIM + 1), np.float32)},
        {**_episode(0), "reward": np.zeros(T - 1, np.float32)},
    ):
        try:
            buf.add_episode(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"add_episode accepted a malformed episode: {list(bad)}")


# ------------------------------------------------------------ obs head parity
def test_replay_short_episode_fallback():
    """Cold start: no episode long enough for a 10-frame history -> pad it."""
    short = 20  # < HISTORY + horizon - 1 = 25 for horizon=16
    buf = ReplayBuffer(capacity=1_000, seed=2)
    buf.add_episode(_episode(0, length=short))
    st = buf.sample_starts(BATCH, 16, device="cpu")
    assert tuple(st["s"].shape) == (BATCH, spec.STATE_DIM)
    assert tuple(st["goal_seq"].shape) == (BATCH, 16, spec.GOAL_DIM)

    start = st["obs_hist"][:, -1, 0]                       # newest frame = start
    assert int(start.min()) >= 0 and int(start.max()) <= short - 16
    # padded prefix = episode's first obs (obs[0]); the real tail is contiguous
    for b in range(BATCH):
        i = int(start[b])
        n_real = i + 1
        assert n_real <= spec.HISTORY
        assert torch.allclose(
            st["obs_hist"][b, :spec.HISTORY - n_real, 0],
            torch.zeros(spec.HISTORY - n_real),
        ), "padded frames must be the episode's first observation (t=0)"
        real = st["obs_hist"][b, spec.HISTORY - n_real:, 0]
        assert torch.allclose(real, torch.arange(n_real).float())
    print(f"    cold-start fallback: start in [{int(start.min())}, {int(start.max())}], history padded")

    # strictly too short to imagine at all -> clear error, no silent garbage
    tiny = ReplayBuffer(capacity=100, seed=3)
    tiny.add_episode(_episode(0, length=8))
    try:
        tiny.sample_starts(4, 16, device="cpu")
    except ValueError:
        pass
    else:
        raise AssertionError("sample_starts accepted episodes shorter than the horizon")


def test_obs_from_state_matches_spec():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(BATCH, spec.STATE_DIM)).astype(np.float64)
    a = rng.normal(size=(BATCH, spec.ACTION_DIM)).astype(np.float64)
    bias = rng.normal(scale=0.01, size=(BATCH, spec.ACTION_DIM))

    ref = np.stack(
        [spec.pack_obs(s[i], a[i], bias[i]) for i in range(BATCH)]
    )
    got = obs_from_state_torch(
        torch.from_numpy(s.astype(np.float32)),
        torch.from_numpy(a.astype(np.float32)),
        torch.from_numpy(bias.astype(np.float32)),
    )
    assert tuple(got.shape) == (BATCH, spec.OBS_DIM)
    err = float(np.abs(got.numpy() - ref).max())
    assert err < 2e-4, f"obs_from_state_torch deviates from spec.pack_obs by {err:.3e}"

    ref_nobias = np.stack([spec.pack_obs(s[i], a[i], None) for i in range(BATCH)])
    got_nobias = obs_from_state_torch(
        torch.from_numpy(s.astype(np.float32)), torch.from_numpy(a.astype(np.float32))
    )
    assert float(np.abs(got_nobias.numpy() - ref_nobias).max()) < 2e-4

    # gradient path through the whole obs head (needed for BPTT)
    s_t = torch.from_numpy(s.astype(np.float32)).requires_grad_(True)
    obs_from_state_torch(s_t, torch.from_numpy(a.astype(np.float32))).sum().backward()
    assert s_t.grad is not None and torch.isfinite(s_t.grad).all()
    assert float(s_t.grad.abs().sum()) > 0


# ---------------------------------------------------------------- model tests
def test_world_model():
    torch.manual_seed(0)
    wm = WorldModel()
    s = torch.randn(BATCH, spec.STATE_DIM)
    a = torch.randn(BATCH, spec.ACTION_DIM)
    xi = torch.randn(BATCH, spec.XI_DIM) * 0.1
    g = torch.randn(BATCH, spec.GOAL_FRAME_DIM)

    s2 = wm(s, a, xi)
    assert tuple(s2.shape) == (BATCH, spec.STATE_DIM)
    assert s2.dtype == torch.float32 and torch.isfinite(s2).all()
    delta = (s2 - s).abs().max().item()
    assert delta <= 3.0 + 1e-4, f"delta {delta} exceeds tanh limit"

    r, d_logit = wm.heads(s, a, g)
    assert tuple(r.shape) == (BATCH,) and tuple(d_logit.shape) == (BATCH,)
    assert torch.isfinite(r).all() and torch.isfinite(d_logit).all()
    p = torch.sigmoid(d_logit)
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0

    loss = (
        F.mse_loss(s2, s)
        + (r ** 2).mean()
        + F.binary_cross_entropy_with_logits(d_logit, torch.zeros(BATCH))
    )
    assert torch.isfinite(loss), f"WM loss is {loss}"
    loss.backward()
    for name, param in wm.named_parameters():
        assert param.grad is not None, f"no grad for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"
    # with input normalisation the +-delta_limit bound lives in normalised
    # space: raw step <= delta_limit * std
    s_std = torch.rand(spec.STATE_DIM) + 0.5
    wm.set_norm(
        torch.randn(spec.STATE_DIM), s_std,
        torch.randn(spec.ACTION_DIM), torch.rand(spec.ACTION_DIM) + 0.5,
        torch.randn(spec.XI_DIM), torch.rand(spec.XI_DIM) + 0.5,
        torch.randn(spec.GOAL_FRAME_DIM), torch.rand(spec.GOAL_FRAME_DIM) + 0.5,
    )
    s2n = wm(s, a, xi)
    assert torch.isfinite(s2n).all()
    # semi-implicit integration moves positions by dt*velocity even at init,
    # so only require a generous bound and exact norm/denorm inversion.
    bound = 10.0 * float(s_std.max()) + 1e-4
    assert float((s2n - s).abs().max()) <= bound, "one-step change too large"
    rn, dn = wm.heads(s, a, g)
    assert torch.isfinite(rn).all() and torch.isfinite(dn).all()

    n = wm.parameters_count()
    assert n > 1_000_000, f"world model only has {n} parameters"
    print(f"    world model params: {n:,}")


def test_policy_critic():
    torch.manual_seed(0)
    pol, crit = Policy(), Critic()
    goal = torch.randn(BATCH, spec.GOAL_DIM)
    hist = torch.randn(BATCH, spec.HISTORY, spec.OBS_DIM)

    z = pol.encode_goal(goal)
    assert tuple(z.shape) == (BATCH, 32) and z.abs().max() <= 1.0

    act = pol(goal, hist)
    assert tuple(act.shape) == (BATCH, spec.ACTION_DIM)
    # actions are deliberately NOT tanh-bounded: action_scale is
    # 0.25*effort/kp, so tracking a reference needs |a| >> 1; env.py clips
    # only to a generous +-50 safety range.
    assert torch.isfinite(act).all()

    v = crit(torch.randn(BATCH, spec.STATE_DIM), goal, torch.randn(BATCH, spec.XI_DIM))
    assert tuple(v.shape) == (BATCH, 1) and torch.isfinite(v).all()

    # input-normalisation buffers (norm.py) must be settable and leave the nets
    # well-behaved: (mean, std) pairs for goal/obs and state/goal/xi
    g_mean = torch.randn(spec.GOAL_DIM) * 0.1
    g_std = torch.rand(spec.GOAL_DIM) + 0.5
    o_mean = torch.randn(spec.OBS_DIM) * 0.1
    o_std = torch.rand(spec.OBS_DIM) + 0.5
    pol.set_norm(g_mean, g_std, o_mean, o_std)
    crit.set_norm(
        torch.randn(spec.STATE_DIM), torch.rand(spec.STATE_DIM) + 0.5,
        g_mean, g_std, torch.randn(spec.XI_DIM), torch.rand(spec.XI_DIM) + 0.5,
    )
    assert pol.encode_goal(goal).abs().max() <= 1.0, "goal latent is still tanh"
    assert torch.isfinite(pol(goal, hist)).all()
    assert torch.isfinite(crit(torch.randn(BATCH, spec.STATE_DIM), goal,
                               torch.randn(BATCH, spec.XI_DIM))).all()

    try:
        pol(goal, torch.randn(BATCH, spec.HISTORY - 1, spec.OBS_DIM))
    except ValueError:
        pass
    else:
        raise AssertionError("Policy accepted a history of the wrong length")

    print(
        f"    policy params: {pol.parameters_count():,}  "
        f"critic params: {crit.parameters_count():,}"
    )


def test_imagine_gae_losses():
    torch.manual_seed(0)
    starts = _small_starts()
    pol, wm, crit = Policy(), WorldModel(), Critic()

    traj = imagine(pol, wm, crit, starts, horizon=H)
    expected = {
        "s": (BATCH, H, spec.STATE_DIM),
        "a": (BATCH, H, spec.ACTION_DIM),
        "r": (BATCH, H),
        "d": (BATCH, H),
        "v": (BATCH, H),
        "last_v": (BATCH, 1),
        "goal": (BATCH, H, spec.GOAL_DIM),
        "xi": (BATCH, spec.XI_DIM),
    }
    for key, shape in expected.items():
        assert key in traj, f"imagine() is missing traj[{key!r}]"
        assert tuple(traj[key].shape) == shape, f"{key}: {tuple(traj[key].shape)}"
        assert torch.isfinite(traj[key]).all(), f"{key} has non-finite values"
    assert float(traj["d"].min()) >= 0.0 and float(traj["d"].max()) <= 1.0, "d must be a probability"
    assert torch.allclose(traj["goal"], starts["goal_seq"][:, :H])
    assert torch.allclose(traj["xi"], starts["xi"])
    # standard GAE alignment: traj["s"][:, t] is the *pre-action* state s_t and
    # traj["v"][:, t] = V(s_t); r/d belong to the transition starting at s_t.
    assert torch.allclose(traj["s"][:, 0], starts["s"])
    assert torch.allclose(
        traj["v"][:, 0],
        crit(traj["s"][:, 0], traj["goal"][:, 0], traj["xi"]).squeeze(-1),
    )
    # reward is analytic in imagination (v1); just sanity-check its range.
    assert float(traj["r"].min()) >= -0.5 and float(traj["r"].max()) <= 0.2
    # d is analytic soft termination (v1); range/finite is enough here.
    assert float(traj["d"].min()) >= 0.0 and float(traj["d"].max()) <= 1.0

    returns, adv = gae_targets(traj["r"], traj["d"], traj["v"], traj["last_v"])
    assert returns.shape == (BATCH, H) and adv.shape == (BATCH, H)
    assert torch.isfinite(returns).all() and torch.isfinite(adv).all()
    assert torch.allclose(adv, returns - traj["v"])

    loss_pi, stats = policy_loss(traj, returns)
    loss_v = critic_loss(traj, returns)
    assert torch.isfinite(loss_pi) and torch.isfinite(loss_v)
    assert isinstance(stats, dict) and {"return", "reward", "terminated", "survival"} <= set(stats)
    assert loss_pi.requires_grad

    # traj["s"]/["a"] must be graph-free (train.py re-evaluates the critic on
    # them *after* the policy step); r/d/v must keep the BPTT graph.
    assert not traj["s"].requires_grad and not traj["a"].requires_grad
    assert traj["v"].requires_grad and traj["r"].requires_grad

    # one full BPTT + optimizer step; every module must move
    all_params = list(pol.parameters()) + list(wm.parameters()) + list(crit.parameters())
    opt = torch.optim.Adam(all_params, lr=1e-3)
    before = [p.detach().clone() for p in all_params]
    opt.zero_grad(set_to_none=True)
    (loss_pi + loss_v).backward()
    for mod, name in ((pol, "policy"), (wm, "world model"), (crit, "critic")):
        grads = [p.grad for p in mod.parameters() if p.grad is not None]
        assert grads, f"{name} got no gradients from imagine()"
        assert any(float(g.abs().sum()) > 0 for g in grads), f"{name} gradients are all zero"
    gn = clip_gradients(all_params)
    assert np.isfinite(gn)
    opt.step()
    changed = sum(
        1 for b, p in zip(before, all_params) if not torch.equal(b, p.detach())
    )
    assert changed > 0, "optimizer.step() changed no parameter"

    # ---------------------------------------------------------------- train.py
    # update pattern on a fresh rollout (short H so the ramp data stays tame):
    #   1) policy BPTT step first,   2) critic re-evaluated per minibatch from
    #   the stored (detached) traj tensors, 2 epochs -> must not hit a stale
    #   imagination graph ("variable modified by an inplace operation").
    buf = _buffer()
    traj2 = imagine(
        pol, wm, crit, buf.sample_starts(BATCH, 4, device="cpu"), horizon=4
    )
    assert tuple(traj2["s"].shape) == (BATCH, 4, spec.STATE_DIM)
    assert tuple(traj2["goal"].shape) == (BATCH, 4, spec.GOAL_DIM)
    assert torch.isfinite(traj2["s"]).all() and torch.isfinite(traj2["d"]).all()
    ret2, _ = gae_targets(traj2["r"], traj2["d"], traj2["v"], traj2["last_v"])
    l2, _ = policy_loss(traj2, ret2)
    opt_pi = torch.optim.Adam(pol.parameters(), lr=1e-4)
    opt_pi.zero_grad(set_to_none=True)
    l2.backward()
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in wm.parameters()
    ), "BPTT did not reach the world model"
    clip_gradients(pol.parameters())
    opt_pi.step()

    s_c = traj2["s"].detach()
    goal_c = traj2["goal"].detach()
    xi_c = traj2["xi"].detach()
    ret_c = ret2.detach()
    opt_v = torch.optim.Adam(crit.parameters(), lr=1e-4)
    v_before = [p.detach().clone() for p in crit.parameters()]
    n_mb = 2
    for _ in range(2):
        for mb in torch.tensor_split(torch.arange(BATCH), n_mb):
            v_all = crit(
                s_c[mb].reshape(-1, spec.STATE_DIM),
                goal_c[mb].reshape(-1, spec.GOAL_DIM),
                xi_c[mb].repeat_interleave(4, 0),
            )
            lv = F.mse_loss(v_all.squeeze(-1), ret_c[mb].reshape(-1))
            assert torch.isfinite(lv), f"critic minibatch loss is {lv}"
            opt_v.zero_grad(set_to_none=True)
            lv.backward()
            opt_v.step()
    assert any(
        not torch.equal(b, p.detach()) for b, p in zip(v_before, crit.parameters())
    ), "critic minibatch loop changed no parameter"

    print(f"    imagine B={BATCH} H={H}: gn={gn:.3g} changed={changed}/{len(all_params)} params")


def test_gae_math():
    B, n = 2, 3
    r = torch.ones(B, n)
    d = torch.zeros(B, n)
    v = torch.zeros(B, n)
    last = torch.full((B, 1), 2.0)
    ret, _ = gae_targets(r, d, v, last, gamma=0.5, lam=0.95)
    # R_2 = r + gamma*last_v ; R_1 = r + gamma*((1-lam)*v_2 + lam*R_2) ; ...
    exp2 = 1.0 + 0.5 * 2.0
    exp1 = 1.0 + 0.5 * (0.95 * exp2)
    exp0 = 1.0 + 0.5 * (0.95 * exp1)
    assert torch.allclose(ret, torch.tensor([[exp0, exp1, exp2]]).expand(B, n)), ret

    # d = 1 (hard termination) truncates the bootstrap entirely
    ret_hard, _ = gae_targets(r, torch.ones(B, n), v, last, gamma=0.5, lam=0.95)
    assert torch.allclose(ret_hard, torch.ones(B, n)), ret_hard


# ------------------------------------------------------------------ cuda test
def test_cuda_smoke():
    if not torch.cuda.is_available():
        print("    cuda unavailable: skipped")
        return
    dev = "cuda"
    torch.manual_seed(0)
    buf = _buffer()
    w = buf.sample_windows(BATCH, k=K, device=dev)
    st = buf.sample_starts(BATCH, H, device=dev)
    assert w["s"].is_cuda and st["s"].is_cuda and w["goal_cur"].is_cuda

    pol = Policy().to(dev)
    wm = WorldModel().to(dev)
    crit = Critic().to(dev)
    opt = torch.optim.Adam(
        list(pol.parameters()) + list(wm.parameters()) + list(crit.parameters()), lr=1e-3
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    opt.zero_grad(set_to_none=True)
    # one WM window step on GPU
    s_pred = wm(w["s"], w["a"][:, 0], w["xi"])
    r_pred, d_logit = wm.heads(w["s"], w["a"][:, 0], w["goal_cur"][:, 0])
    wm_loss = (
        F.mse_loss(s_pred, w["s_next"][:, 0])
        + F.mse_loss(r_pred, w["r"][:, 0])
        + F.binary_cross_entropy_with_logits(d_logit, w["d"][:, 0])
    )
    # one imagination + BPTT step on GPU
    traj = imagine(pol, wm, crit, st, horizon=H)
    returns, _ = gae_targets(traj["r"], traj["d"], traj["v"], traj["last_v"])
    loss = wm_loss + policy_loss(traj, returns)[0] + critic_loss(traj, returns)
    assert torch.isfinite(loss)
    loss.backward()
    clip_gradients(list(pol.parameters()) + list(wm.parameters()) + list(crit.parameters()))
    opt.step()
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1e3
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    print(f"    cuda smoke B={BATCH} H={H}: {dt_ms:.1f} ms, peak {peak:.0f} MiB")
    assert torch.isfinite(traj["s"]).all()


# ---------------------------------------------------------------------- main
def main() -> None:
    tests = [
        test_replay_windows,
        test_replay_starts,
        test_replay_eviction_and_validation,
        test_replay_short_episode_fallback,
        test_obs_from_state_matches_spec,
        test_world_model,
        test_policy_critic,
        test_imagine_gae_losses,
        test_gae_math,
        test_cuda_smoke,
    ]
    for fn in tests:
        t0 = time.perf_counter()
        fn()
        print(f"[ok] {fn.__name__:<38s} {time.perf_counter() - t0:6.2f} s")
    print("PASS")


if __name__ == "__main__":
    main()
