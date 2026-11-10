#!/usr/bin/env python3
"""Exact-MJX action sensitivity vs the learned WM."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np, jax, jax.numpy as jnp, mujoco, mujoco.mjx as mjx
from smini import motion, spec
m=mujoco.MjModel.from_xml_path(str(ROOT.parent/"assets/mujoco_menagerie/unitree_g1/scene_mjx.xml")); m.opt.timestep=0.005
mx=mjx.put_model(m); d0=mjx.make_data(m)
lib=motion.MotionLibrary.from_dir(ROOT/'assets/motions40_trim'); mo=lib.motions[0]; t=100
q=np.asarray(mo.qpos[t]); v=np.asarray(mo.qvel[t])
a0=np.clip((mo.qpos[min(t+1,len(mo)-1),7:36]-spec.DEFAULT_Q)/spec.ACTION_SCALE,-5,5)
def step(qp,vp,c):
    d=d0.replace(qpos=jnp.asarray(qp),qvel=jnp.asarray(vp),ctrl=jnp.asarray(c))
    for _ in range(4): d=mjx.step(mx,d)
    return np.asarray(d.qpos), np.asarray(d.qvel)
def rollout(a):
    qp,vp=q.copy(),v.copy(); qs=[]
    for c in a:
        qp,vp=step(qp,vp,c); qs.append(qp.copy())
    return np.asarray(qs)
base=rollout([a0])
print('MJX natural 1-step dq norm', np.linalg.norm(base[0][:3]-q[:3]))
for eps in (0.1,0.5,1.0):
    ds=[]
    for j in range(29):
        aa=a0.copy(); aa[j]+=eps
        out=rollout([aa]); ds.append(np.linalg.norm(out[0]-base[0]))
    print(f'MJX eps {eps}: mean |dq| {np.mean(ds):.4f} max {np.max(ds):.4f}')
