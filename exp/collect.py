"""并行采集 WM 训练数据 (CPU MuJoCo, 多进程)。

输出: exp/data/shard_XX.npz
  S[n,71] 状态 [p3, q4, qj29, v3, w3, qd29]
  A[n,29] 动作
  X[n,33] sim context [friction, calib29, com3]
  E[n]    1 = 该 transition 是 episode 末帧(其后状态不连续)
  CLIP[n] 该 transition 所属 clip 名 id
"""
import sys, os, time, json
from pathlib import Path
import numpy as np
import multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
XML = str(ROOT.parent/"assets"/"mujoco_menagerie"/"unitree_g1"/"scene_mjx.xml")
MDIR = ROOT/"assets"/"motions"
OUT = ROOT/"exp"/("data_v2" if __import__("os").environ.get("V2") else "data")
OUT.mkdir(parents=True, exist_ok=True)
NS = 73; ASC = None; J = slice(15, 44)


def to71(s283, DEFAULT_Q, R_from_quat6=None):
    """283 -> 71. 见 scripts/import_wm_bench.py:to71 (逐字复刻)"""
    pos, q6, lin, ang, qrel, qd = s283[:,0:3], s283[:,3:9], s283[:,9:12], s283[:,12:15], s283[:,15:44], s283[:,44:73]
    c0, c1 = q6[:,0:3], q6[:,3:6]
    c0 = c0/np.linalg.norm(c0, axis=-1, keepdims=True).clip(1e-9)
    c1 = c1 - (c0*c1).sum(-1, keepdims=True)*c0
    c1 = c1/np.linalg.norm(c1, axis=-1, keepdims=True).clip(1e-9)
    c2 = np.cross(c0, c1)
    R = np.stack([c0, c1, c2], -1)                      # (n,3,3)
    q = _quat_from_mat(R)
    qj = qrel + DEFAULT_Q
    ang_b = np.einsum('nji,nj->ni', R, ang)
    return np.concatenate([pos, q, qj, lin, ang_b, qd], -1).astype(np.float32)


def _quat_from_mat(R):
    """R (n,3,3) -> quat wxyz"""
    n = R.shape[0]; q = np.zeros((n,4), np.float64)
    m00,m01,m02 = R[:,0,0],R[:,0,1],R[:,0,2]
    m10,m11,m12 = R[:,1,0],R[:,1,1],R[:,1,2]
    m20,m21,m22 = R[:,2,0],R[:,2,1],R[:,2,2]
    tr = m00+m11+m22
    for i in range(n):
        T = tr[i]
        if T > 0:
            S = np.sqrt(T+1.0)*2; q[i] = [0.25*S, (m21[i]-m12[i])/S, (m02[i]-m20[i])/S, (m10[i]-m01[i])/S]
        elif m00[i] > m11[i] and m00[i] > m22[i]:
            S = np.sqrt(1.0+m00[i]-m11[i]-m22[i])*2; q[i] = [(m21[i]-m12[i])/S, 0.25*S, (m01[i]+m10[i])/S, (m02[i]+m20[i])/S]
        elif m11[i] > m22[i]:
            S = np.sqrt(1.0+m11[i]-m00[i]-m22[i])*2; q[i] = [(m02[i]-m20[i])/S, (m01[i]+m10[i])/S, 0.25*S, (m12[i]+m21[i])/S]
        else:
            S = np.sqrt(1.0+m22[i]-m00[i]-m11[i])*2; q[i] = [(m10[i]-m01[i])/S, (m02[i]+m20[i])/S, (m12[i]+m21[i])/S, 0.25*S]
    return q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-9)


def worker(args):
    wid, clips, target, seed, noise, rand_prob, rand_scale = args
    from smini import motion, spec
    from smini.env import EnvCfg, TrackingEnv
    lib = motion.MotionLibrary([motion.load_motion(str(MDIR/f"{c}.npz")) for c in clips])
    cfg = EnvCfg(obs_noise=True, reset_noise=True, domain_randomization=True, action_noise=0.0)
    NE = 8
    envs = [TrackingEnv(XML, lib, cfg, seed=seed*1000+i) for i in range(NE)]
    for e in envs: e.reset()
    rng = np.random.default_rng(seed)
    DQ = np.asarray(spec.DEFAULT_Q, np.float64)
    S, A, X, E = [], [], [], []
    clip_names = [c for c in clips]
    cnt = 0; t0 = time.time()
    while cnt < target:
        s283 = np.stack([e.state() for e in envs])          # (NE,283)
        s71 = to71(s283, DQ)
        acts = np.empty((NE, 29), np.float64)
        for i, e in enumerate(envs):
            a = e._goal[J] / spec.ACTION_SCALE
            if rng.random() < rand_prob:
                a = rng.uniform(-rand_scale, rand_scale, 29)
            acts[i] = np.clip(a + noise*rng.standard_normal(29), -50, 50)
        xi = np.stack([np.asarray(e.xi, np.float64) for e in envs])   # (NE,34)
        term = np.zeros(NE, bool); trunc = np.zeros(NE, bool)
        for i, e in enumerate(envs):
            _, _, te, tr, _ = e.step(acts[i])
            term[i] = te or tr
        S.append(s71); A.append(acts.astype(np.float32)); X.append(xi.astype(np.float32))
        E.append(term)
        for i in range(NE):
            if term[i]:
                envs[i].reset()
        cnt += NE
    S = np.concatenate(S); A = np.concatenate(A); X = np.concatenate(X)
    E = np.concatenate(E).astype(np.uint8)
    np.savez_compressed(OUT/f"shard_{wid:02d}.npz", S=S, A=A, X=X, E=E)
    print(f"[w{wid}] {len(S)} transitions, {time.time()-t0:.0f}s, {len(S)/(time.time()-t0):.0f}/s, "
          f"episode_ends={int(E.sum())}", flush=True)


if __name__ == "__main__":
    TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 6_000_000
    NW = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    allc = sorted(p.stem for p in MDIR.glob("*.npz"))
    # 留出 3 条 clip 做 held-out (dance2_subject1, fallAndGetUp1_subject1, sprint1_subject2)
    hold = [c for c in allc if c in ("dance2_subject1","fallAndGetUp1_subject1","sprint1_subject2")]
    train = [c for c in allc if c not in hold]
    json.dump({"train": train, "holdout": hold}, open(OUT/"clips.json","w"), indent=1)
    print(f"train clips={len(train)}  holdout={hold}", flush=True)
    per = TOTAL // NW
    jobs = []
    for w in range(NW):
        sub = train[w::NW][:6] or train[:6]
        jobs.append((w, sub, per, 100+w, 0.10, 0.02, 2.0))
    t=time.time()
    ps=[mp.Process(target=worker, args=(j,)) for j in jobs]
    [p.start() for p in ps]; [p.join() for p in ps]
    print(f"done in {time.time()-t:.0f}s", flush=True)
