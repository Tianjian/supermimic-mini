import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import jax, jax.numpy as jnp, numpy as np, mujoco, mujoco.mjx as mjx
from smini import spec
m=mujoco.MjModel.from_xml_path(str(ROOT.parent/"assets/mujoco_menagerie/unitree_g1/scene_mjx.xml"))
m.opt.timestep=0.005
mx=mjx.put_model(m); dx=mjx.make_data(m)
q0=np.zeros(m.nq); q0[0:3]=[0,0,0.76]; q0[3:7]=[1,0,0,0]; q0[7:36]=spec.DEFAULT_Q
v0=np.zeros(m.nv)
def rollout(a):
    d=dx.replace(qpos=jnp.asarray(q0), qvel=jnp.asarray(v0), ctrl=jnp.asarray(a))
    for _ in range(2):
        d=mjx.step(mx,d)
    return d.qpos[0:3], d.qpos[3:7], d.qvel[6:9]
a=jnp.zeros(29)
try:
    y=rollout(a); print("forward ok", [np.asarray(x) for x in y])
    g=jax.grad(lambda x: rollout(x)[0][2])(a)
    print("grad norm", float(jnp.linalg.norm(g)), "nan", bool(jnp.isnan(g).any()))
except Exception as e:
    import traceback; traceback.print_exc()
