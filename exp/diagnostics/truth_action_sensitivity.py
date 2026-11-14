"""测 MuJoCo 真值仿真器的 1 步动作敏感度 (每步重同步, 严格 1 步对照)。

做法: e1 走参考轨迹; 每步开始把 e2 的 MjData 同步到 e1 当前状态,
      然后 e1.step(a), e2.step(a+δ), 比较下一步状态差 vs 自然一步变化。
"""
import sys
from pathlib import Path
import numpy as np
import mujoco
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from smini import motion, spec
from smini.env import EnvCfg, TrackingEnv

XML = str(ROOT.parent/"assets"/"mujoco_menagerie"/"unitree_g1"/"scene_mjx.xml")
MDIR = ROOT/"assets"/"motions"; NS = 73; ASC = spec.ACTION_SCALE; J = slice(15, 44)
clips = ["walk1_subject1", "run1_subject2", "dance1_subject1"]
lib = motion.MotionLibrary([motion.load_motion(str(MDIR/f"{c}.npz")) for c in clips])
CLEAN = EnvCfg(obs_noise=False, reset_noise=False, domain_randomization=False, action_noise=0.0)


def sync(dst, src):
    dst._data.qpos[:] = src._data.qpos
    dst._data.qvel[:] = src._data.qvel
    dst._data.ctrl[:] = src._data.ctrl
    mujoco.mj_forward(dst._model, dst._data)


def run(mode, delta, T=300, seed=7):
    e1 = TrackingEnv(XML, lib, CLEAN, seed=seed); e1.reset()
    e2 = TrackingEnv(XML, lib, CLEAN, seed=seed); e2.reset()
    rng = np.random.default_rng(0)
    nat, pert = [], []
    for t in range(T):
        sync(e2, e1)                                  # 严格同步到同一状态
        s0 = e1.state()[:NS].copy()
        a = e1._goal[J]/ASC if mode == "pd" else rng.uniform(-2, 2, spec.ACTION_DIM)
        if mode == "pd" and t % 5 == 0:                # 混一点动作变化
            a = a + rng.standard_normal(spec.ACTION_DIM)*0.3
        _, _, te1, tr1, _ = e1.step(a)
        _, _, te2, tr2, _ = e2.step(np.clip(a + delta, -50, 50))
        sa = e1.state()[:NS].copy(); sb = e2.state()[:NS].copy()
        if te1 or tr1 or te2 or tr2:
            e1.reset(); e2.reset(); continue
        nat.append(np.linalg.norm(sa - s0)); pert.append(np.linalg.norm(sb - sa))
    nat, pert = np.array(nat), np.array(pert)
    print(f"  [{mode:4s}] δ={delta:4.2f}  自然||Δs||={nat.mean():7.4f}  "
          f"扰动||Δs'-Δs||={pert.mean():7.4f}  敏感度={pert.mean()/nat.mean()*100:6.1f}%")
    return pert.mean()/nat.mean()


print("=== 真值仿真器 1 步动作敏感度 (PD 跟踪动作) ===")
for d in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
    run("pd", d)
print("\n=== 真值仿真器 1 步动作敏感度 (随机动作) ===")
for d in (0.1, 0.25, 0.5, 1.0, 2.0):
    run("rand", d)
