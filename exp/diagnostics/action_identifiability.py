"""按状态块拆解动作可解释性: 动作到底影响哪个分量?"""
import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from smini import motion, spec
from smini.env import EnvCfg, TrackingEnv

XML = str(ROOT.parent/"assets"/"mujoco_menagerie"/"unitree_g1"/"scene_mjx.xml")
MDIR = ROOT/"assets"/"motions"; NS = 73; ASC = spec.ACTION_SCALE; J = slice(15, 44)
clips = ["walk1_subject1", "run1_subject2", "dance1_subject1"]
lib = motion.MotionLibrary([motion.load_motion(str(MDIR/f"{c}.npz")) for c in clips])
cfg = EnvCfg(obs_noise=True, reset_noise=True, domain_randomization=True, action_noise=0.0)
envs = [TrackingEnv(XML, lib, cfg, seed=i) for i in range(8)]
rng = np.random.default_rng(0)
BLOCKS = [("pos", 0, 3), ("quat6d", 3, 9), ("linvel", 9, 12), ("angvel", 12, 15),
          ("q", 15, 44), ("qd", 44, 73)]


def r2r(X, Y, lam=1e-2):
    mu, sd = X.mean(0), X.std(0)+1e-8
    Xn = (X-mu)/sd; Xn = np.c_[Xn, np.ones(len(Xn))]
    W = np.linalg.solve(Xn.T@Xn + lam*np.eye(Xn.shape[1]), Xn.T@Y); P = Xn@W
    r = 1-((Y-P)**2).sum(0)/(((Y-Y.mean(0))**2).sum(0)+1e-12)
    return float(np.mean(r)), r


def probe(name, noise, rp, rs, T=4000):
    S, A = [], []; got = 0
    for e in envs: e.reset()
    while got < T:
        for e in envs:
            s = e.state()[:NS].copy(); a = e._goal[J]/ASC
            if rng.random() < rp: a = rng.uniform(-rs, rs, spec.ACTION_DIM)
            a = np.clip(a + noise*rng.standard_normal(spec.ACTION_DIM), -50, 50)
            _, _, te, tr, _ = e.step(a)
            if not (te or tr):
                S.append(s); A.append(a); got += 1
                if got >= T: break
            else:
                e.reset()
    S = np.asarray(S); A = np.asarray(A); dS = S[1:]-S[:-1]
    Xs, Xa = S[:-1], np.c_[S[:-1], A[:-1]]
    print(f"\n--- {name} (n={len(dS)}) ---")
    print(f"  {'block':8s} {'R2(s)':>7s} {'R2(s,a)':>8s} {'gain':>7s}")
    for nm, i, j in BLOCKS:
        m1, _ = r2r(Xs, dS[:, i:j]); m2, _ = r2r(Xa, dS[:, i:j])
        print(f"  {nm:8s} {m1:7.3f} {m2:8.3f} {m2-m1:7.3f}")


probe("作者采集 (n.10 rnd2%)", 0.10, 0.02, 2.0)
probe("纯随机动作", 0.0, 1.0, 2.0)
