#!/usr/bin/env python3
"""Evaluate the imported SuperMimic-Reproduced WM (71-dim) on local teacher data."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, torch
from smini import spec
from smini.wm_import import WM71, quat_from_mat
from wm_teacher_bench import collect

dev = "cuda"
ck = torch.load(ROOT / "runs/imported/e8_boneseed_ckpt_0100000.pt", map_location=dev, weights_only=False)
wm = WM71(dev).to(dev)
missing, unexpected = wm.load_state_dict(ck["wm"], strict=False)
print("missing", missing, "unexpected", unexpected)
wm.eval()

import argparse
ap=argparse.ArgumentParser(); ap.add_argument("--kind", default="teacher"); ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--perm", action="store_true"); ap.add_argument("--zero", action="store_true"); a=ap.parse_args()
eps, lib = collect(a.kind, a.steps, "assets/motions_trim", 0)
S = np.concatenate([e["state"] for e in eps], 0)  # 283
A = np.concatenate([e["action"] for e in eps], 0)
# convert 283 -> 71
def to71(s):
    s = torch.as_tensor(s, dtype=torch.float32, device=dev)
    pos, q6, lin, ang, qrel, qd = s[:, 0:3], s[:, 3:9], s[:, 9:12], s[:, 12:15], s[:, 15:44], s[:, 44:73]
    c0, c1 = q6[:, 0:3], q6[:, 3:6]
    c0 = c0 / c0.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    c1 = c1 - (c0*c1).sum(-1, keepdim=True)*c0
    c1 = c1 / c1.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    c2 = torch.cross(c0, c1, dim=-1)
    R = torch.stack([c0, c1, c2], -1)
    q = quat_from_mat(R)
    qj = qrel + torch.as_tensor(spec.DEFAULT_Q, dtype=torch.float32, device=dev)
    ang_b = torch.einsum('nji,nj->ni', R, ang)  # world -> body (R^T @ w)
    return torch.cat([pos, q, qj, lin, ang_b, qd], -1)
S71 = to71(S).cpu().numpy(); A = A.astype(np.float32)
# rescale our actions to mjlab G1_ACTION_SCALE (same q_target, different normalized action)
import mujoco
_m = mujoco.MjModel.from_xml_path(str(ROOT.parent / "assets/mujoco_menagerie/unitree_g1/scene_mjx.xml"))
_names = [mujoco.mj_id2name(_m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, _m.njnt)]
def mjlab_scale(n):
    if "hip_pitch" in n or "hip_yaw" in n or n == "waist_yaw_joint": return 0.5486
    if "hip_roll" in n or "knee" in n: return 0.3507
    if "wrist_pitch" in n or "wrist_yaw" in n: return 0.0745
    return 0.4386
_scale_mj = np.array([mjlab_scale(n) for n in _names], np.float32)
if a.perm:
    import re
    pats = ([".*_elbow_joint", ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_wrist_roll_joint"],
            [".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"],
            [".*_hip_roll_joint", ".*_knee_joint"],
            [".*_wrist_pitch_joint", ".*_wrist_yaw_joint"],
            ["waist_pitch_joint", "waist_roll_joint"],
            [".*_ankle_pitch_joint", ".*_ankle_roll_joint"])
    order = []
    for grp in pats:
        for pat in grp:
            for i, n in enumerate(_names):
                if re.fullmatch(pat, n) and i not in order: order.append(i)
    order = np.array(order)
    print("permutation", order[:8], "...")
    A = A[:, order] * (spec.ACTION_SCALE[order].astype(np.float32) / _scale_mj[order])
else:
    A = A * (spec.ACTION_SCALE.astype(np.float32) / _scale_mj)
if a.zero:
    A = np.zeros_like(A)
print("action scale ratio min/max", float((spec.ACTION_SCALE/_scale_mj).min()), float((spec.ACTION_SCALE/_scale_mj).max()))
print("transitions", S71.shape)

def nse(pred, true):
    p = torch.as_tensor(pred, device=dev); t = torch.as_tensor(true, device=dev)
    return wm.spec.nse(p, t).mean().item()

# build windows of 32 within episodes
rng = np.random.default_rng(0); errs = []
for _ in range(64):
    i = rng.integers(0, len(S71) - 33)
    s = torch.as_tensor(S71[i:i+1], device=dev); acts = torch.as_tensor(A[i:i+32][None], device=dev)
    xi = torch.zeros(1, 33, device=dev); xi[:, 0] = 0.75; true = torch.as_tensor(S71[i+1:i+33][None], device=dev)
    preds = []
    for j in range(32):
        s = wm.step(s, acts[:, j], xi); preds.append(s)
    e = [nse(preds[j], true[:, j]) for j in (0, 7, 31)]
    errs.append(e)
errs = np.array(errs).mean(0)
# per-group diagnostics
def group_nse(pred, true):
    p=torch.as_tensor(pred,device=dev); t=torch.as_tensor(true,device=dev)
    e=wm.spec.error_vector(p,t); n=torch.as_tensor([3,3,3,3,29,29],device=dev)
    return [(e[..., a:b]**2).mean().item() for a,b in [(0,3),(3,6),(6,9),(9,12),(12,41),(41,70)]]
g=[group_nse(preds[j], true[:,j]) for j in (0,7,31)]
print("group NSE 1step", [round(x,4) for x in g[0]])
print("group NSE 8step", [round(x,4) for x in g[1]])
print(f"imported WM NSE 1-step {errs[0]:.5f} 8-step {errs[1]:.5f} 32-step {errs[2]:.5f}")
print(f"sqrt (normalized RMSE) 1 {errs[0]**0.5:.4f} 8 {errs[1]**0.5:.4f} 32 {errs[2]**0.5:.4f}")
