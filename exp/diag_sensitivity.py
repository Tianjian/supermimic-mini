"""动作敏感度诊断: 在 WM 训练用的归一化空间 (error_vector/err_scale) 里比较
   真值仿真器 vs WM 到底捕获了多少动作效应。

指标 (同一状态, 动作扰动 δ):
  nat   = ||e( s_{t+1}(a),        s_t )||            自然一步变化
  d_true= ||e( s_{t+1}(a+δ),      s_{t+1}(a) )||     真值的动作效应
  d_wm  = ||e( ŝ_{t+1}(a+δ),      ŝ_{t+1}(a) )||     WM 的动作效应
  capture = d_wm / d_true                              <-- 关键: 1.0 表示学到, 0.075 表示忽略动作
"""
import sys, json
from pathlib import Path
import numpy as np, torch, mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"exp"))
from smini.wm_import import WM71
from smini import motion, spec
from smini.env import EnvCfg, TrackingEnv
import collect as C

XML = str(ROOT.parent/"assets"/"mujoco_menagerie"/"unitree_g1"/"scene_mjx.xml")
MDIR = ROOT/"assets"/"motions"; J = slice(15,44)
CLEAN = EnvCfg(obs_noise=False, reset_noise=False, domain_randomization=False, action_noise=0.0)


def sync(dst, src):
    dst._data.qpos[:] = src._data.qpos
    dst._data.qvel[:] = src._data.qvel
    dst._data.ctrl[:] = src._data.ctrl
    mujoco.mj_forward(dst._model, dst._data)


def main():
    ck = sys.argv[1] if len(sys.argv) > 1 else str(ROOT/"exp/runs/paper_best.pt")
    from wm_train import make_wm
    wm = make_wm(34).to("cuda")
    sd = torch.load(ck, map_location="cuda", weights_only=False)
    wm.load_state_dict(sd["wm"] if "wm" in sd else sd); wm.eval()
    print("loaded", ck, "step", sd.get("step"), "nse8", sd.get("nse8"), flush=True)

    meta = json.load(open(ROOT/"exp/data/clips.json"))
    lib = motion.MotionLibrary([motion.load_motion(str(MDIR/f"{c}.npz")) for c in meta["train"][:3]])
    DQ = np.asarray(spec.DEFAULT_Q, np.float64)
    rng = np.random.default_rng(0)

    for delta in (0.1, 0.5, 1.0):
        e1 = TrackingEnv(XML, lib, CLEAN, seed=5); e1.reset()
        e2 = TrackingEnv(XML, lib, CLEAN, seed=5); e2.reset()
        xi_np = np.concatenate([[1.0, 1.0], np.zeros(29), np.zeros(3)]).astype(np.float32)
        xi_t = torch.as_tensor(xi_np)[None].cuda().repeat(1,1)
        nat=[]; dtrue=[]; dwm=[]
        for t in range(200):
            sync(e2, e1)
            s283 = np.stack([e1.state()]*2)
            s71 = C.to71(s283, DQ)                     # (2,71)
            a = e1._goal[J]/spec.ACTION_SCALE
            if t % 5 == 0: a = a + rng.standard_normal(29)*0.3
            a2 = np.clip(a+delta, -50, 50)
            _,_,te1,tr1,_ = e1.step(a)
            _,_,te2,tr2,_ = e2.step(a2)
            if te1 or tr1 or te2 or tr2:
                e1.reset(); e2.reset(); continue
            s_next1 = C.to71(np.stack([e1.state()]), DQ)[0]
            s_next2 = C.to71(np.stack([e2.state()]), DQ)[0]
            s0 = torch.as_tensor(s71[0])[None].cuda()
            aa  = torch.as_tensor(a)[None].float().cuda()
            aa2 = torch.as_tensor(a2)[None].float().cuda()
            with torch.no_grad():
                p1 = wm.step(s0, aa, xi_t)
                p2 = wm.step(s0, aa2, xi_t)
            def nrm(pred, true):
                return float(torch.linalg.norm(wm.spec.error_vector(pred, true)))
            st1 = torch.as_tensor(s_next1)[None].cuda(); st2 = torch.as_tensor(s_next2)[None].cuda()
            nat.append(nrm(st1, s0))
            dtrue.append(nrm(st2, st1))
            dwm.append(nrm(p2, p1))
        nat,dtrue,dwm = map(np.mean, (nat,dtrue,dwm))
        print(f"δ={delta:4.2f}  nat={nat:7.3f}  d_true={dtrue:7.3f}  d_wm={dwm:7.3f}  "
              f"| 真值敏感度={dtrue/nat*100:6.1f}%  WM敏感度={dwm/nat*100:6.1f}%  "
              f"**capture={dwm/max(dtrue,1e-9)*100:6.1f}%**", flush=True)

    # 同时报 1 步预测精度(参照系)
    print("\n--- WM 单步归一化误差 (同数据集抽样) ---", flush=True)
    d = np.load(sorted((ROOT/"exp/data_v2").glob("holdout_*.npz"))[0])
    S,A,X = d["S"][:2000], d["A"][:2000], d["X"][:2000]
    s = torch.as_tensor(S).cuda(); x = torch.as_tensor(X).cuda()
    with torch.no_grad():
        p = wm.step(s, torch.as_tensor(A).cuda(), x)
        e = wm.spec.error_vector(p, torch.as_tensor(S).cuda())
    print(f"1 步 NSE = {(e*e).mean().item():.5f}  (sqrt {(e*e).mean().item()**0.5:.4f})", flush=True)


if __name__ == "__main__":
    main()
