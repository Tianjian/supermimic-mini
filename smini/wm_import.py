"""Port of the SuperMimic-Reproduced torch world model (71-dim, canonical + explicit integration)."""
from __future__ import annotations
import torch, torch.nn as nn

STATE_DIM, FEAT_DIM, DELTA_DIM, NJ = 71, 71, 70, 29


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1); bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw], -1)


def quat_conj(q):
    return q * torch.tensor([1., -1., -1., -1.], dtype=q.dtype, device=q.device)


def quat_normalize(q):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def quat_apply(q, v):
    q = quat_normalize(q)
    t = 2.0 * torch.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1] * t + torch.cross(q[..., 1:], t, dim=-1)


def quat_apply_inverse(q, v):
    return quat_apply(quat_conj(q), v)


def quat_from_rotvec(rv):
    ang = rv.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    axis = rv / ang
    h = 0.5 * ang
    return torch.cat([torch.cos(h), torch.sin(h) * axis], -1)


def rotvec_from_quat(q):
    q = quat_normalize(q); q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]; s = v.norm(dim=-1, keepdim=True)
    ang = 2.0 * torch.atan2(s, q[..., :1].clamp_min(1e-9))
    return v / s.clamp_min(1e-9) * ang


def quat_to_mat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1),
        torch.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
        torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], -2)


def quat_from_mat(R):
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    tr = m00 + m11 + m22
    q = torch.zeros(*R.shape[:-2], 4, dtype=R.dtype, device=R.device)
    s = torch.sqrt((tr + 1).clamp_min(1e-9)) * 2
    q[..., 0] = 0.25 * s; q[..., 1] = (m21-m12)/s; q[..., 2] = (m02-m20)/s; q[..., 3] = (m10-m01)/s
    return quat_normalize(q)


def rot6d_from_quat(q):
    R = quat_to_mat(quat_normalize(q))
    return R[..., :, :2].reshape(*R.shape[:-2], 6)


def rot6d_to_quat(x):
    # x = first two columns (6,), reconstruct third via cross
    c0, c1 = x[..., 0:3], x[..., 3:6]
    c0 = c0 / c0.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    c1 = c1 - (c0 * c1).sum(-1, keepdim=True) * c0
    c1 = c1 / c1.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    c2 = torch.cross(c0, c1, dim=-1)
    R = torch.stack([c0, c1, c2], -1)
    return quat_from_mat(R)


def heading_quat(q):
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    h = 0.5 * yaw
    return torch.stack([torch.cos(h), torch.zeros_like(h), torch.zeros_like(h), torch.sin(h)], -1)


def quat_canonical(q):
    return torch.where(q[..., :1] < 0, -q, q)


class StateSpec:
    def __init__(self, device, dt=0.02):
        self.dt = dt
        self.err_scale = torch.cat([torch.full((n,), s) for s, n in
            zip((0.05, 0.1, 0.5, 1.0, 0.1, 2.0), (3, 3, 3, 3, NJ, NJ))]).to(device)
        self.out_scale = torch.cat([torch.full((n,), s) for s, n in
            zip((0.3, 0.5, 2.0, 0.01, 0.02, 0.02), (3, 3, NJ, 3, 3, NJ))]).to(device)
        self.in_scale = torch.cat([torch.full((n,), s) for s, n in
            zip((1.0, 1.0, 1.0, 2.0, 1.0, 5.0), (1, 6, 3, 3, NJ, NJ))]).to(device)

    @staticmethod
    def split(s):
        return s[..., 0:3], s[..., 3:7], s[..., 7:36], s[..., 36:39], s[..., 39:42], s[..., 42:71]

    def canonical(self, s):
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q); hq = heading_quat(q)
        q_local = quat_canonical(quat_mul(quat_conj(hq), q))
        v_h = quat_apply_inverse(hq, v)
        feat = torch.cat([p[..., 2:3], rot6d_from_quat(q_local), v_h, w, qj, qd], -1)
        return feat / self.in_scale

    def integrate(self, s, delta):
        d = delta * self.out_scale
        dv, dw, dqd, dp, dth, dq = d[..., 0:3], d[..., 3:6], d[..., 6:35], d[..., 35:38], d[..., 38:41], d[..., 41:70]
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q); hq = heading_quat(q)
        v_new = v + quat_apply(hq, dv); w_new = w + dw; qd_new = qd + dqd
        p_new = p + self.dt * v_new + quat_apply(hq, dp)
        q_new = quat_normalize(quat_mul(q, quat_from_rotvec(self.dt * w_new + dth)))
        qj_new = qj + self.dt * qd_new + dq
        return torch.cat([p_new, q_new, qj_new, v_new, w_new, qd_new], -1)

    def error_vector(self, pred, true):
        p1, q1, qj1, v1, w1, qd1 = self.split(pred); p2, q2, qj2, v2, w2, qd2 = self.split(true)
        rot = rotvec_from_quat(quat_mul(quat_conj(quat_normalize(q2)), quat_normalize(q1)))
        return torch.cat([p1-p2, rot, v1-v2, w1-w2, qj1-qj2, qd1-qd2], -1) / self.err_scale

    def nse(self, pred, true):
        e = self.error_vector(pred, true); return (e*e).mean(-1)


class MLP(nn.Module):
    def __init__(self, i, hidden, o):
        super().__init__(); layers=[]; d=i
        for h in hidden: layers += [nn.Linear(d,h), nn.SiLU()]; d=h
        layers.append(nn.Linear(d,o)); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)


class DynamicsMLP(nn.Module):
    def __init__(self, hidden=(2048,2048,2048)):
        super().__init__(); self.mlp = MLP(71+29+33, hidden, 70)
    def forward(self, feat, a, xi): return self.mlp(torch.cat([feat,a,xi],-1))


class TermHead(nn.Module):
    def __init__(self, hidden=(64,64)):
        super().__init__(); self.mlp = MLP(4, hidden, 1)
        self.register_buffer('scale', torch.tensor([4.0, 1.25]))
    def forward(self, feats):
        x = feats * self.scale; return self.mlp(torch.cat([x, x.abs()], -1)).squeeze(-1)


class WM71(nn.Module):
    def __init__(self, device):
        super().__init__(); self.spec = StateSpec(device); self.dynamics = DynamicsMLP(); self.term_head = TermHead()
        self.register_buffer("xi_offset", torch.cat([torch.tensor([0.75]), torch.zeros(32)]).to(device))
        self.register_buffer("xi_scale", torch.cat([torch.tensor([0.45]), torch.full((29,), 0.01), torch.full((3,), 0.05)]).to(device))
    def step(self, s, a, xi):
        xi_n = (xi - self.xi_offset) / self.xi_scale
        return self.spec.integrate(s, self.dynamics(self.spec.canonical(s), a, xi_n))
