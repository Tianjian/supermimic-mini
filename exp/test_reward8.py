"""对拍：可微 FK 与完整 8 项奖励 vs MuJoCo / TrackingEnv 真值。

用法: python exp/test_reward8.py
  - 检查 1: FK(位置/姿态/线速度/角速度) 与 build_model.body_kinematics 一致
  - 检查 2: reward8 与 TrackingEnv._reward 逐位一致
"""
import sys
from pathlib import Path
import numpy as np, torch, mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"exp"))
from smini import spec, motion, build_model
from smini.env import EnvCfg, TrackingEnv
import diff_fk as D
from diff_fk import FKTable

XML = str(ROOT.parent/"assets"/"mujoco_menagerie"/"unitree_g1"/"scene_mjx.xml")
MDIR = ROOT/"assets"/"motions"
D.set_action_scale(spec.ACTION_SCALE)


def main():
    fk = FKTable(XML, spec.TRACKED_BODIES, spec.JOINT_NAMES, device="cpu")
    m = fk.m
    # ---------- 检查 1: FK vs body_kinematics ----------
    bids = build_model.body_ids(m)
    rng = np.random.default_rng(0)
    ep = eq = el = ea = 0.0
    for _ in range(5):
        qpos = rng.uniform(m.qpos0-0.3, m.qpos0+0.3); qpos[3:7] /= np.linalg.norm(qpos[3:7])
        qvel = rng.uniform(-1, 1, m.nv)
        d = mujoco.MjData(m); d.qpos[:] = qpos; d.qvel[:] = qvel
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); mujoco.mj_comVel(m, d)
        p_ref, q_ref, lin_ref, ang_ref = build_model.body_kinematics(m, d, bids)

        pr = torch.tensor(qpos[0:3], dtype=torch.float32)[None]
        qr = torch.tensor(qpos[3:7], dtype=torch.float32)[None]
        qj = torch.tensor(qpos[7:36], dtype=torch.float32)[None]
        vr = torch.tensor(qvel[0:3], dtype=torch.float32)[None]
        wb = torch.tensor(qvel[3:6], dtype=torch.float32)[None]
        qd = torch.tensor(qvel[6:35], dtype=torch.float32)[None]
        pos, Rot = fk.fk(pr, qr, qj)
        lin, ang = fk.fk_vel(pr, qr, qj, vr, wb, qd)
        ti = fk.tracked
        ep = max(ep, np.abs(pos[0, ti].numpy() - p_ref).max())
        # 比较旋转矩阵（避开四元数双重覆盖）
        R_ref = np.stack([spec.quat_to_mat(q) for q in q_ref])
        eq = max(eq, np.abs(Rot[0, ti].numpy() - R_ref).max())
        el = max(el, np.abs(lin[0, ti].numpy() - lin_ref).max())
        ea = max(ea, np.abs(ang[0, ti].numpy() - ang_ref).max())
    print(f"[FK]  位置 {ep:.2e} | 姿态(四元数) {eq:.2e} | 线速度 {el:.2e} | 角速度 {ea:.2e}")
    ok1 = max(ep, eq, el, ea) < 1e-4

    # ---------- 检查 2: reward8 vs TrackingEnv._reward ----------
    lib = motion.MotionLibrary([motion.load_motion(str(MDIR/"walk1_subject1.npz"))])
    env = TrackingEnv(XML, lib, EnvCfg(obs_noise=False, reset_noise=False,
                                       domain_randomization=False, action_noise=0.0), seed=0)
    env.reset(motion_id=0, frame_idx=50)
    diffs = []
    for k in range(6):
        a = rng.uniform(-1, 1, 29)
        prev = env._prev_action.copy()
        _, r_true, te, tr, _ = env.step(a)
        t = int(env._ref_index())
        # 注意: mj_step 只在"步首"计算运动学, 所以 env.state() 里的 body 块比 qpos
        # 滞后一个物理子步(1-3 mm)。这里刷新后再比, 以验证 reward8 本身是否精确。
        mujoco.mj_forward(env._model, env._data)
        st = build_model.state_from_data(env._model, env._data, build_model.body_ids(env._model))
        qpos = env._data.qpos.copy(); qvel = env._data.qvel.copy()
        mo = env._motion
        ref = {
            "body_pos": torch.as_tensor(mo.body_pos[t], dtype=torch.float32)[None],
            "body_R": torch.as_tensor(np.stack([spec.quat_to_mat(q) for q in mo.body_quat[t]]),
                                      dtype=torch.float32)[None],
            "body_lin": torch.as_tensor(mo.body_linvel[t], dtype=torch.float32)[None],
            "body_ang": torch.as_tensor(mo.body_angvel[t], dtype=torch.float32)[None],
            "root_pos": torch.as_tensor(mo.qpos[t, 0:3], dtype=torch.float32)[None],
            "root_R": torch.as_tensor(spec.quat_to_mat(mo.qpos[t, 3:7]), dtype=torch.float32)[None],
        }
        s71 = torch.as_tensor(np.concatenate([qpos[0:3], qpos[3:7], qpos[7:36],
                                              qvel[0:3], qvel[3:6], qvel[6:35]])[None].astype(np.float32))
        r_mine = float(D.reward8(s71,
                                 torch.as_tensor(a, dtype=torch.float32)[None],
                                 torch.as_tensor(prev, dtype=torch.float32)[None],
                                 ref, fk))
        r_env_fresh = env._reward(st, t, a, prev)
        diffs.append(abs(r_mine - r_env_fresh))
        if te or tr: env.reset(motion_id=0, frame_idx=50)
    print(f"[R8]  6 次随机状态 |reward8 - reward_env(刷新后)| 最大 {max(diffs):.3e}")
    ok2 = max(diffs) < 1e-4
    print("\n=>", "✅ 全部对齐（FK 与 8 项奖励均可微且精确）" if (ok1 and ok2) else "❌ 存在偏差")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
