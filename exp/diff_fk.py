"""可微前向运动学 (torch) + 完整 8 项跟踪奖励。

动机：`smini/imagine.py::analytic_reward` 只算 6 项，漏掉 14 个 body link 的
位置/姿态/线速度/角速度误差，导致通过世界模型做 BPTT 规划时，收益无法迁移到
环境完整奖励。本模块用纯 torch 复刻 MuJoCo 的前向运动学，把这 4 项补回去。

约定（与 smini 一致）：
  s71 = [p3, q4(wxyz), qj29(绝对关节角), v3(世界线速度), w3(体坐标系角速度), qd29]
  本模块 FK 全部用世界坐标系。
"""
from __future__ import annotations
import numpy as np
import torch


# ---------------------------------------------------------------- 四元数工具
def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """(...,4) wxyz -> (...,3,3)"""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    return torch.stack([
        torch.stack([ww+xx-yy-zz, 2*(x*y-w*z), 2*(x*z+w*y)], -1),
        torch.stack([2*(x*y+w*z), ww-xx+yy-zz, 2*(y*z-w*x)], -1),
        torch.stack([2*(x*z-w*y), 2*(y*z+w*x), ww-xx-yy+zz], -1),
    ], -2)


def mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """(...,3,3) -> (...,4) wxyz（按最大分量分支，数值稳定）"""
    shp = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)
    m00 = R[:, 0, 0]; m11 = R[:, 1, 1]; m22 = R[:, 2, 2]
    m01 = R[:, 0, 1]; m10 = R[:, 1, 0]
    m02 = R[:, 0, 2]; m20 = R[:, 2, 0]
    m12 = R[:, 1, 2]; m21 = R[:, 2, 1]
    # 四个分支各自成立时取对应公式
    c0 = tr > 0
    c1 = (~c0) & (m00 >= m11) & (m00 >= m22)
    c2 = (~c0) & (~c1) & (m11 >= m22)
    c3 = (~c0) & (~c1) & (~c2)
    if c0.any():
        s = torch.sqrt(tr[c0] + 1.0) * 2
        q[c0, 0] = 0.25 * s; q[c0, 1] = (m21[c0]-m12[c0])/s
        q[c0, 2] = (m02[c0]-m20[c0])/s; q[c0, 3] = (m10[c0]-m01[c0])/s
    if c1.any():
        s = torch.sqrt(1.0 + m00[c1]-m11[c1]-m22[c1]) * 2
        q[c1, 0] = (m21[c1]-m12[c1])/s; q[c1, 1] = 0.25*s
        q[c1, 2] = (m01[c1]+m10[c1])/s; q[c1, 3] = (m02[c1]+m20[c1])/s
    if c2.any():
        s = torch.sqrt(1.0 + m11[c2]-m00[c2]-m22[c2]) * 2
        q[c2, 0] = (m02[c2]-m20[c2])/s; q[c2, 1] = (m01[c2]+m10[c2])/s
        q[c2, 2] = 0.25*s; q[c2, 3] = (m12[c2]+m21[c2])/s
    if c3.any():
        s = torch.sqrt(1.0 + m22[c3]-m00[c3]-m11[c3]) * 2
        q[c3, 0] = (m10[c3]-m01[c3])/s; q[c3, 1] = (m02[c3]+m20[c3])/s
        q[c3, 2] = (m12[c3]+m21[c3])/s; q[c3, 3] = 0.25*s
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return q.reshape(*shp, 4)


def axis_angle_mat(axis: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """axis (B,3) 单位向量, ang (B,) -> (B,3,3)  Rodrigues"""
    a = axis
    ang = ang.reshape(-1)
    c = torch.cos(ang)          # (B,)
    s = torch.sin(ang)          # (B,)
    x, y, z = a[..., 0], a[..., 1], a[..., 2]      # (B,)
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zero, -z, y], -1),
        torch.stack([z, zero, -x], -1),
        torch.stack([-y, x, zero], -1),
    ], -2)                                            # (B,3,3)
    I = torch.eye(3, device=a.device, dtype=a.dtype).expand_as(K)
    outer = a.unsqueeze(-1) * a.unsqueeze(-2)         # (B,3,3)
    c3 = c[..., None, None]
    s3 = s[..., None, None]
    return I * c3 + s3 * K + (1 - c3) * outer


def rot_angle_sq(M: torch.Tensor) -> torch.Tensor:
    """与 smini.env._rot_angle_sq 完全一致：atan2(sin, cos)^2"""
    v = torch.stack([M[..., 2, 1]-M[..., 1, 2],
                     M[..., 0, 2]-M[..., 2, 0],
                     M[..., 1, 0]-M[..., 0, 1]], -1)
    sin_t = 0.5 * v.norm(dim=-1)
    cos_t = 0.5 * (M.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0)
    return torch.atan2(sin_t, cos_t) ** 2


# ---------------------------------------------------------------- FK 表
class FKTable:
    """从 MuJoCo 模型抽取的静态运动学表（常数，不需梯度）。"""

    def __init__(self, xml_path: str, tracked: list[str], joint_names: list[str],
                 device="cuda"):
        import mujoco
        m = mujoco.MjModel.from_xml_path(xml_path)
        self.m = m
        self.nbody = m.nbody
        self.parent = torch.as_tensor(m.body_parentid.copy(), device=device, dtype=torch.long)
        self.body_pos = torch.as_tensor(m.body_pos.copy(), device=device, dtype=torch.float32)
        self.body_quat = torch.as_tensor(m.body_quat.copy(), device=device, dtype=torch.float32)
        # 质心在体坐标系下的位置: repo 的 body_linvel 用的是 mj_objectVelocity(XBODY) =
        # 质心处速度 (与 data.cvel 同一约定), 所以必须做 com 修正
        self.body_ipos = torch.as_tensor(m.body_ipos.copy(), device=device, dtype=torch.float32)
        # 每个 body 的 hinge 轴与对应 dof（无关节则 -1）
        N = m.nbody
        axis = np.zeros((N, 3), np.float64)
        dof = np.full(N, -1, np.int64)
        for b in range(N):
            js, jn = m.body_jntadr[b], m.body_jntnum[b]
            if jn != 1:
                continue
            j = js
            if int(m.jnt_type[j]) == 3:            # hinge
                axis[b] = m.jnt_axis[j]
                dof[b] = m.jnt_dofadr[j] - 6       # 跳过一个自由关节的 6 个 dof
        self.axis = torch.as_tensor(axis, device=device, dtype=torch.float32)
        self.dof = torch.as_tensor(dof, device=device, dtype=torch.long)
        self.tracked = torch.as_tensor(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in tracked],
            device=device, dtype=torch.long)
        # 关节限位（以 dof 顺序，29 个）
        lim = np.zeros((len(joint_names), 2), np.float64)
        for i, jn_ in enumerate(joint_names):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn_)
            lim[i] = m.jnt_range[jid]
        self.jlim = torch.as_tensor(lim, device=device, dtype=torch.float32)
        self.NJ = len(joint_names)

    def fk(self, p_root: torch.Tensor, q_root: torch.Tensor, qj: torch.Tensor):
        """返回 (pos (B,N,3), R (B,N,3,3)) 世界系。用 list 累积, 避免原地切片赋值
        (原地写会破坏 autograd 的版本计数, 在 BPTT 展开时炸掉)。"""
        B = p_root.shape[0]
        dev, dt = p_root.device, p_root.dtype
        R_off_all = quat_to_mat(self.body_quat)           # (N,3,3)
        pos = [None] * self.nbody
        Rot = [None] * self.nbody
        pos[0] = torch.zeros(B, 3, device=dev, dtype=dt)
        Rot[0] = torch.eye(3, device=dev, dtype=dt).expand(B, 3, 3)
        pos[1] = p_root
        Rot[1] = quat_to_mat(q_root)
        for b in range(2, self.nbody):
            pa = int(self.parent[b].item())
            R_new = Rot[pa] @ R_off_all[b]
            d = int(self.dof[b].item())
            if d >= 0:
                R_new = R_new @ axis_angle_mat(self.axis[b].expand(B, 3), qj[:, d])
            Rot[b] = R_new
            pos[b] = pos[pa] + Rot[pa] @ self.body_pos[b]
        return torch.stack(pos, 1), torch.stack(Rot, 1)

    def fk_vel(self, p_root, q_root, qj, v_root, w_body_root, qd):
        """返回 (linvel (B,N,3), angvel (B,N,3)) 世界系。"""
        B = p_root.shape[0]
        pos, Rot = self.fk(p_root, q_root, qj)
        R_off_all = quat_to_mat(self.body_quat)
        lin = [None] * self.nbody
        ang = [None] * self.nbody
        zero = torch.zeros(B, 3, device=p_root.device, dtype=p_root.dtype)
        lin[0] = zero
        ang[0] = zero
        lin[1] = v_root
        ang[1] = (quat_to_mat(q_root) @ w_body_root.unsqueeze(-1)).squeeze(-1)
        for b in range(2, self.nbody):
            pa = int(self.parent[b].item())
            d = int(self.dof[b].item())
            w_b = ang[pa]
            if d >= 0:
                axis_world = (Rot[:, pa] @ R_off_all[b]) @ self.axis[b]
                w_b = w_b + axis_world * qd[:, d:d+1]
            ang[b] = w_b
            lin[b] = lin[pa] + torch.cross(ang[pa], pos[:, b] - pos[:, pa], dim=-1)
        return torch.stack(lin, 1), torch.stack(ang, 1)


# ---------------------------------------------------------------- 8 项奖励
def reward8(s71, a, prev_a, ref, fk: FKTable, dt=0.02):
    """完整 8 项跟踪奖励（与 smini.env.TrackingEnv._reward 同构）。

    s71 (B,71) / a,prev_a (B,29)
    ref: dict
        root_pos (B,3), root_R (B,3,3)
        body_pos (B,14,3), body_R (B,14,3,3),
        body_lin (B,14,3), body_ang (B,14,3)
    """
    p_root = s71[:, 0:3]; q_root = s71[:, 3:7]
    qj = s71[:, 7:36]; v_root = s71[:, 36:39]
    w_body = s71[:, 39:42]; qd = s71[:, 42:71]

    pos, Rot = fk.fk(p_root, q_root, qj)
    lin, ang = fk.fk_vel(p_root, q_root, qj, v_root, w_body, qd)
    ti = fk.tracked

    bp = pos[:, ti]; bR = Rot[:, ti]; bl = lin[:, ti]; ba = ang[:, ti]

    pos_err = ((ref["body_pos"] - bp) ** 2).sum(-1).mean(-1)
    ori_err = rot_angle_sq(ref["body_R"].transpose(-1, -2) @ bR).mean(-1)
    lin_err = ((ref["body_lin"] - bl) ** 2).sum(-1).mean(-1)
    ang_err = ((ref["body_ang"] - ba) ** 2).sum(-1).mean(-1)

    root_pos_err = ((ref["root_pos"] - p_root) ** 2).sum(-1)
    root_ori_err = rot_angle_sq(ref["root_R"].transpose(-1, -2) @ Rot[:, 1])

    q = qj
    lo, hi = fk.jlim[:, 0], fk.jlim[:, 1]
    limit = (torch.clamp(q - hi, min=0) + torch.clamp(lo - q, min=0)).sum(-1)

    asc = torch.as_tensor(_ACTION_SCALE, device=s71.device, dtype=s71.dtype)
    smooth = ((asc * (a - prev_a)) ** 2).sum(-1)

    r = (torch.exp(-pos_err / 0.3**2)
         + torch.exp(-ori_err / 0.4**2)
         + torch.exp(-lin_err / 1.0**2)
         + torch.exp(-ang_err / 3.14**2)
         + 0.5 * torch.exp(-root_pos_err / 0.3**2)
         + 0.5 * torch.exp(-root_ori_err / 0.4**2)
         - 0.1 * smooth
         - 2.0 * limit)
    return r * dt


_ACTION_SCALE = None


def set_action_scale(asc):
    global _ACTION_SCALE
    _ACTION_SCALE = np.asarray(asc, np.float64)
