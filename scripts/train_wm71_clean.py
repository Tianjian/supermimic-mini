#!/usr/bin/env python3
"""Route A (fixed eval): train 4 clips, hold out 1 clip, fixed independent eval."""
from __future__ import annotations
import argparse, copy, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
import numpy as np, torch
from smini.wm_import import WM71
from bptt_warp_train import WarpEnv, Q0, ASC

def collect_episodes(env, steps, det=True, rng=None):
    S=[[] for _ in range(env.N)]; A=[[] for _ in range(env.N)]; eps=[]
    s,_,_,q,v=env.current(); prev=torch.cat([q,v],-1)
    for _ in range(steps):
        ref=env.qpos_t[env.clip,env.frame][:,7:36]
        a=((ref-Q0)/ASC)
        if not det: a=a+0.1*torch.randn_like(a)
        a=a.clamp(-50,50)
        q,v,ob,r,term,trunc,done=env.step(a)
        sn=torch.cat([q,v],-1); dn=done.cpu().numpy()
        sp=prev.cpu().numpy(); an=a.cpu().numpy(); snn=sn.cpu().numpy()
        for i in range(env.N): S[i].append(sp[i]); A[i].append(an[i])
        for i in np.nonzero(dn)[0]:
            S[int(i)].append(snn[i])
            if len(S[int(i)])>34: eps.append((np.asarray(S[int(i)],np.float32),np.asarray(A[int(i)],np.float32)))
            S[int(i)].clear(); A[int(i)].clear()
        s2,_,_,q2,v2=env.current(); prev=torch.cat([q2,v2],-1)
    return eps

@torch.no_grad()
def eval_eps(wm, eps, rng, n=256, steps=32):
    out=[]
    for _ in range(n):
        s,a=eps[rng.integers(0,len(eps))]; j=rng.integers(0,len(a)-steps)
        s=torch.as_tensor(s[None,j:j+steps+1],device="cuda"); act=torch.as_tensor(a[None,j:j+steps],device="cuda")
        xi=torch.zeros(1,33,device="cuda"); xi[:,0]=0.75; pred=s[:,0]; row=[]
        for t in range(steps):
            pred=wm.step(pred,act[:,t],xi)
            row.append(wm.spec.nse(pred,s[:,t+1]).mean().item())
        out.append([row[0],row[7],row[31]])
    e=np.array(out).mean(0); return e

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--envs",type=int,default=1024); ap.add_argument("--iters",type=int,default=600)
    ap.add_argument("--collect",type=int,default=5); ap.add_argument("--warmup",type=int,default=30)
    ap.add_argument("--updates",type=int,default=4); ap.add_argument("--batch",type=int,default=1024)
    ap.add_argument("--k",type=int,default=8); ap.add_argument("--lr",type=float,default=2e-4)
    ap.add_argument("--wd",type=float,default=1e-4); ap.add_argument("--ema",type=float,default=0.999)
    ap.add_argument("--eval-every",type=int,default=100); ap.add_argument("--out",default="runs/wm71_clean")
    ap.add_argument("--train-clips",default="0,1,2,3"); ap.add_argument("--eval-clips",default="4")
    ap.add_argument("--motion-dir",default="assets/motions_trim")
    ap.add_argument("--action-noise",type=float,default=0.1)
    ap.add_argument("--random-prob",type=float,default=0.02)
    ap.add_argument("--random-scale",type=float,default=2.0)
    a=ap.parse_args(); dev="cuda"
    tr=[int(x) for x in a.train_clips.split(",")]; ev=[int(x) for x in a.eval_clips.split(",")]
    env=WarpEnv(a.envs,a.motion_dir,clips=tr)
    # fixed eval set: held-out clip, independent seed, deterministic PD
    eenv=WarpEnv(64,a.motion_dir,seed=12345,clips=ev)
    feps=collect_episodes(eenv,1500,det=True); print("fixed eval episodes",len(feps),flush=True)
    rng=np.random.default_rng(0); evrng=np.random.default_rng(12345)
    # buffer
    cap=2_000_000; eps=[]; nb=0
    def add(S,A):
        nonlocal nb
        if len(A)<3: return
        eps.append((np.asarray(S,np.float32),np.asarray(A,np.float32))); nb+=len(A)
        while len(eps)>1 and nb>cap: s,x=eps.pop(0); nb-=len(x)
    def sample(b,k):
        S=[];A=[]
        while len(S)<b:
            s,x=eps[rng.integers(0,len(eps))]
            if len(x)<k: continue
            j=rng.integers(0,len(x)-k+1); S.append(s[j:j+k+1]); A.append(x[j:j+k])
        return torch.as_tensor(np.stack(S),device=dev), torch.as_tensor(np.stack(A),device=dev)
    wm=WM71(dev).to(dev); ema=copy.deepcopy(wm); ema.eval()
    opt=torch.optim.Adam(wm.parameters(),lr=a.lr,weight_decay=a.wd)
    xi=torch.zeros(a.envs,33,device=dev); xi[:,0]=0.75
    S=[[] for _ in range(a.envs)]; A=[[] for _ in range(a.envs)]
    prev=None
    s,_,_,q,v=env.current(); prev=torch.cat([q,v],-1)
    def flush(i):
        if len(S[i])<3: S[i].clear(); A[i].clear(); return
        add(S[i],A[i]); S[i].clear(); A[i].clear()
    def collect(steps):
        nonlocal prev
        for _ in range(steps):
            sp=prev.clone()
            ref=env.qpos_t[env.clip,env.frame][:,7:36]
            act=((ref-Q0)/ASC)
            # 2% random actions for falls/recovery coverage
            mask=(torch.rand(a.envs,1,device=dev)<a.random_prob)
            act=torch.where(mask, (torch.rand_like(act)*2*a.random_scale-a.random_scale), act)
            act=(act+a.action_noise*torch.randn_like(act)).clamp(-50,50)
            q,v,ob,r,term,trunc,done=env.step(act)
            sn=torch.cat([q,v],-1); dn=done.cpu().numpy()
            spn=sp.cpu().numpy(); an=act.cpu().numpy(); snn=sn.cpu().numpy()
            for i in range(a.envs): S[i].append(spn[i]); A[i].append(an[i])
            for i in np.nonzero(dn)[0]:
                S[int(i)].append(snn[i]); flush(int(i))
            _,_,_,q2,v2=env.current(); prev=torch.cat([q2,v2],-1)
    for _ in range(a.warmup): collect(a.collect)
    print("warmup buffer",nb,"episodes",len(eps),flush=True)
    best=1e9; bad=0
    for it in range(1,a.iters+1):
        t0=time.time(); collect(a.collect)
        lr=a.lr*(0.03+0.97*0.5*(1+np.cos(np.pi*it/a.iters)))
        for g in opt.param_groups: g["lr"]=lr
        for _ in range(a.updates):
            s_,act_=sample(a.batch,a.k); pred=s_[:,0]; loss=0.0
            for j in range(a.k):
                pred=wm.step(pred,act_[:,j],xi)
                e=wm.spec.error_vector(pred,s_[:,j+1]); loss=loss+(e*e).mean()
            loss=loss/a.k; opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(),1.0); opt.step()
            with torch.no_grad():
                for pe,p in zip(ema.parameters(),wm.parameters()): pe.mul_(a.ema).add_(p,alpha=1-a.ema)
                for be,b in zip(ema.buffers(),wm.buffers()): be.copy_(b)
        if it%50==0:
            print(f"[{it:4d}] loss {float(loss):.4f} lr {lr:.1e} buf {nb} eps {len(eps)} {time.time()-t0:.1f}s",flush=True)
        if it%a.eval_every==0:
            e_raw=eval_eps(wm,feps,evrng); e_ema=eval_eps(ema,feps,evrng)
            print(f"[eval {it}] heldout raw NSE8 {e_raw[1]:.5f} (sqrt {e_raw[1]**0.5:.4f}) | ema {e_ema[1]:.5f}",flush=True)
            if e_ema[1]<best:
                best=e_ema[1]; bad=0
                torch.save({"wm":ema.state_dict(),"iter":it,"nse8":e_ema[1],"clips":tr,"eval_clips":ev},Path(a.out)/"best.pt")
            else:
                bad+=1
                if bad>=3: print("early stop",flush=True); break
    print("done best",best)

if __name__=="__main__": main()
