"""决定性测试: WM 的动作梯度方向是否与真值一致?

对同一状态、同一组随机动作扰动 δa, 比较:
   Δr_wm  = reward71( ŝ(s,a+δa) )  - reward71( ŝ(s,a) )
   Δr_true= reward71( s'(s,a+δa) ) - reward71( s'(s,a) )
两者用**同一个** reward71, 所以差异只来自状态预测 —— 这正是 BPTT 依赖的梯度信号。

若相关显著为正 -> WM 的 action Jacobian 可用 -> BPTT 有信号
若相关 ~0      -> 无论多少算力, BPTT 都无法工作
"""
import sys, json
from pathlib import Path
import numpy as np, torch, mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"exp"))
from smini import motion, spec
from smini.env import EnvCfg, TrackingEnv
from wm_train import make_wm
from mpc_test import reward71
import collect as C

XML = str(ROOT.parent/"assets"/"mujoco_menagerie"/"unitree_g1"/"scene_mjx.xml")
MDIR = ROOT/"assets"/"motions"
DQ = np.asarray(spec.DEFAULT_Q)
CLEAN = EnvCfg(obs_noise=False, reset_noise=False, domain_randomization=False, action_noise=0.0)
XI = torch.as_tensor(np.concatenate([[1.,1.], np.zeros(29), np.zeros(3)]), dtype=torch.float32)[None].cuda()


def sync(dst, src):
    dst._data.qpos[:] = src._data.qpos; dst._data.qvel[:] = src._data.qvel
    dst._data.ctrl[:] = src._data.ctrl
    mujoco.mj_forward(dst._model, dst._data)


def main():
    ck = sys.argv[1] if len(sys.argv)>1 else str(ROOT/"exp/runs/v2_best.pt")
    wm = make_wm(34).to("cuda"); sd=torch.load(ck,map_location="cuda",weights_only=False)
    wm.load_state_dict(sd["wm"] if "wm" in sd else sd); wm.eval()
    print(f"WM step={sd.get('step')} nse8={sd.get('nse8')}", flush=True)

    meta=json.load(open(ROOT/"exp/data_v2/clips.json"))
    lib=motion.MotionLibrary([motion.load_motion(str(MDIR/f"{c}.npz")) for c in meta["train"]])
    rng=np.random.default_rng(0)
    M=8; NS=40
    all_wm=[]; all_true=[]
    for si in range(NS):
        mid=int(rng.integers(0,len(lib))); mo=lib[mid]
        if len(mo)<40: continue
        f0=int(rng.integers(0,len(mo)-40))
        e1=TrackingEnv(XML,lib,CLEAN,seed=7); e1.reset(motion_id=mid,frame_idx=f0)
        e2=TrackingEnv(XML,lib,CLEAN,seed=7); e2.reset(motion_id=mid,frame_idx=f0)
        goal=torch.as_tensor(mo.goal_frame(int(e1.frame_idx)),dtype=torch.float32).cuda()[None]
        s0n=np.stack([e1.state()]); s0=torch.as_tensor(C.to71(s0n,DQ)[0],dtype=torch.float32).cuda()[None]
        a_pd=(e1._goal[15:44]/spec.ACTION_SCALE).astype(np.float64)
        prev=torch.zeros(1,29,device="cuda")

        # 参考: δa=0 的真值下一步
        sync(e2,e1)
        e1.step(a_pd); s_a=C.to71(np.stack([e1.state()]),DQ)[0]
        s_a_t=torch.as_tensor(s_a,dtype=torch.float32).cuda()[None]
        r0_true=float(reward71(s_a_t,torch.as_tensor(a_pd).float()[None].cuda(),prev,goal))
        with torch.no_grad():
            s_w=wm.step(s0,torch.as_tensor(a_pd).float()[None].cuda(),XI)
            r0_wm=float(reward71(s_w,torch.as_tensor(a_pd).float()[None].cuda(),prev,goal))

        wmv=[]; trv=[]
        for k in range(M):
            d=rng.standard_normal(29)*0.3
            a2=np.clip(a_pd+d,-50,50)
            sync(e2,e1)                       # 回到同一状态
            e2.step(a2); s_b=C.to71(np.stack([e2.state()]),DQ)[0]
            s_b_t=torch.as_tensor(s_b,dtype=torch.float32).cuda()[None]
            at=torch.as_tensor(a2,dtype=torch.float32).cuda()[None]
            r1_true=float(reward71(s_b_t,at,prev,goal))
            with torch.no_grad():
                s_w2=wm.step(s0,at,XI); r1_wm=float(reward71(s_w2,at,prev,goal))
            trv.append(r1_true-r0_true); wmv.append(r1_wm-r0_wm)
        all_true+=trv; all_wm+=wmv

    wt=np.array(all_wm); tt=np.array(all_true)
    from scipy import stats
    rp=stats.pearsonr(wt,tt); rs=stats.spearmanr(wt,tt)
    print(f"\nn={len(wt)}  Δr_wm  std={wt.std():.5f}   Δr_true std={tt.std():.5f}")
    print(f"Pearson  r={rp.statistic:+.4f}  p={rp.pvalue:.3g}")
    print(f"Spearman ρ={rs.statistic:+.4f}  p={rs.pvalue:.3g}")
    # 梯度方向符号一致率
    sign=(np.sign(wt)==np.sign(tt)).mean()
    print(f"符号一致率 (梯度方向对错): {sign*100:.1f}%   (随机=50%)")
    # top-k: WM 认为最好的扰动, 在真值里是否也更好
    print(f"\nΔr_true 均值={tt.mean():+.6f}  Δr_wm 均值={wt.mean():+.6f}")
    json.dump({"pearson":float(rp.statistic),"p_pearson":float(rp.pvalue),
               "spearman":float(rs.statistic),"p_spearman":float(rs.pvalue),
               "sign_agree":float(sign),"n":int(len(wt)),
               "std_wm":float(wt.std()),"std_true":float(tt.std()),
               "wm_step":sd.get("step")},
              open(ROOT/"exp/runs/grad_corr.json","w"), indent=1)
    print("saved exp/runs/grad_corr.json")


if __name__ == "__main__":
    main()
