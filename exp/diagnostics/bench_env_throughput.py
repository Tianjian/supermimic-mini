"""测量 supermimic-mini 的真实 TrackingEnv 吞吐:
   单进程 (作者当前的 VecEnv 路径) vs 多进程 (本机 16 线程可用但未实现)
"""
import sys, time, glob, multiprocessing as mp
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from smini import motion
from smini.env import EnvCfg, TrackingEnv

XML = str(ROOT.parent / "assets" / "mujoco_menagerie" / "unitree_g1" / "scene_mjx.xml")
MDIR = ROOT / "assets" / "motions"

def build_lib(clips):
    return motion.MotionLibrary([motion.load_motion(str(MDIR / f"{c}.npz")) for c in clips])

def run_single(n_envs, steps, clips):
    lib = build_lib(clips)
    cfg = EnvCfg(obs_noise=True, reset_noise=True, domain_randomization=True, action_noise=0.1)
    envs = [TrackingEnv(XML, lib, cfg, seed=i) for i in range(n_envs)]
    rng = np.random.default_rng(0)
    for e in envs: e.reset()
    acts = [rng.standard_normal(29)*0.3 for _ in range(n_envs)]
    for _ in range(20):                       # warmup
        for i,e in enumerate(envs): e.step(acts[i])
    t = time.time()
    for _ in range(steps):
        for i,e in enumerate(envs): e.step(acts[i])
    dt = time.time()-t
    ctrl = (n_envs*steps)/dt
    print(f"  单进程: {n_envs} env x {steps} steps -> {ctrl:,.0f} 控制步/s "
          f"({ctrl*4:,.0f} 物理步/s), {ctrl/n_envs:.1f}x 实时/env")
    return ctrl

def worker(args):
    n_envs, steps, clips, seed, q = args
    lib = build_lib(clips)
    cfg = EnvCfg(obs_noise=True, reset_noise=True, domain_randomization=True, action_noise=0.1)
    envs = [TrackingEnv(XML, lib, cfg, seed=seed*1000+i) for i in range(n_envs)]
    rng = np.random.default_rng(seed)
    for e in envs: e.reset()
    acts = [rng.standard_normal(29)*0.3 for _ in range(n_envs)]
    for _ in range(20):
        for i,e in enumerate(envs): e.step(acts[i])
    t = time.time()
    for _ in range(steps):
        for i,e in enumerate(envs): e.step(acts[i])
    dt = time.time()-t
    q.put((n_envs*steps)/dt)

if __name__ == "__main__":
    clips = sorted(p.stem for p in MDIR.glob("*.npz"))[:8]
    print(f"clips={len(clips)}  xml={Path(XML).name}")
    print("\n=== 1. 单进程 (作者 VecEnv 现状) ===")
    base = run_single(8, 400, clips)

    print("\n=== 2. 多进程聚合 (本机 16 线程) ===")
    for procs, per in ((8, 8), (16, 8), (16, 16)):
        q = mp.Queue()
        jobs = [(per, 200, clips, i, q) for i in range(procs)]
        ps = [mp.Process(target=worker, args=(j,)) for j in jobs]
        t = time.time(); [p.start() for p in ps]; [p.join() for p in ps]
        wall = time.time()-t
        rates = [q.get() for _ in range(procs)]
        agg = sum(rates)
        print(f"  {procs} 进程 x {per} env: {agg:,.0f} 控制步/s "
              f"({agg/base:5.1f}x 单进程, {agg*4:,.0f} 物理步/s), 等效 {agg/50:,.0f} 实时环境")
