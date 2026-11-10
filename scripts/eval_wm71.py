#!/usr/bin/env python3
"""Fixed held-out evaluation of a trained WM71 (episode-contiguous windows)."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
import numpy as np, torch
from smini.wm_import import WM71
from bptt_warp_train import WarpEnv, Q0, ASC
dev="cuda"; N=64
ck=torch.load(ROOT/"runs/wm71_5m/best.pt",map_location=dev,weights_only=False)
wm=WM71(dev).to(dev); wm.load_state_dict(ck["wm"]); wm.eval()
env=WarpEnv(N,"assets/motions_trim",seed=12345)
S=[[] for _ in range(N)]; A=[[] for _ in range(N)]; eps=[]; rng=np.random.default_rng(999)
s,_,_,q,v=env.current(); prev=torch.cat([q,v],-1)
for step in range(1500):
    ref=env.qpos_t[env.clip,env.frame][:,7:36]
    a=((ref-Q0)/ASC+0.1*torch.randn(N,29,device=dev)).clamp(-50,50)
    q,v,ob,r,term,trunc,done=env.step(a)
    sn=torch.cat([q,v],-1); dn=done.cpu().numpy()
    sp=prev.cpu().numpy(); an=a.cpu().numpy(); snn=sn.cpu().numpy()
    for i in range(N): S[i].append(sp[i]); A[i].append(an[i])
    for i in np.nonzero(dn)[0]:
        S[int(i)].append(snn[i])
        if len(S[int(i)])>34: eps.append((np.asarray(S[int(i)],np.float32),np.asarray(A[int(i)],np.float32)))
        S[int(i)].clear(); A[int(i)].clear()
    s2,_,_,q2,v2=env.current(); prev=torch.cat([q2,v2],-1)
print("held-out episodes",len(eps))
errs=[]
with torch.no_grad():
    for _ in range(256):
        s_,a_=eps[rng.integers(0,len(eps))]; j=rng.integers(0,len(a_)-32)
        s=torch.as_tensor(s_[None,j:j+33],device=dev); act=torch.as_tensor(a_[None,j:j+32],device=dev)
        xi=torch.zeros(1,33,device=dev); xi[:,0]=0.75; pred=s[:,0]; out=[]
        for t in range(32):
            pred=wm.step(pred,act[:,t],xi); out.append(wm.spec.nse(pred,s[:,t+1]).mean().item())
        errs.append([out[0],out[7],out[31]])
e=np.array(errs).mean(0)
print(f"fixed held-out NSE 1 {e[0]:.5f} 8 {e[1]:.5f} 32 {e[2]:.5f} | sqrt 1 {e[0]**0.5:.4f} 8 {e[1]**0.5:.4f} 32 {e[2]**0.5:.4f}")
