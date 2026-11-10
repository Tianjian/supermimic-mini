#!/usr/bin/env python3
"""Route A: train a 71-dim WM (qpos|qvel) on local Warp data with k=8 loss."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, torch
from smini import spec
from smini.wm_import import WM71
from bptt_warp_train import WarpEnv, Q0, ASC

class WBuf:
    def __init__(self, cap=1_000_000):
        self.cap=cap; self.eps=[]; self.n=0; self.rng=np.random.default_rng(0)
    def add(self, S, A):
        if len(A) < 3: return
        self.eps.append((np.asarray(S,np.float32), np.asarray(A,np.float32)))
        self.n += len(A)
        while len(self.eps)>1 and self.n>self.cap:
            s,a=self.eps.pop(0); self.n-=len(a)
    def sample(self, b, k, dev="cuda"):
        S=[]; A=[]
        while len(S)<b:
            i=int(self.rng.integers(0,len(self.eps))); s,a=self.eps[i]
            if len(a)<k: continue
            j=int(self.rng.integers(0,len(a)-k+1))
            S.append(s[j:j+k+1]); A.append(a[j:j+k])
        return torch.as_tensor(np.stack(S),device=dev), torch.as_tensor(np.stack(A),device=dev)
    def __len__(self): return self.n

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--envs",type=int,default=1024)
    ap.add_argument("--iters",type=int,default=300)
    ap.add_argument("--collect",type=int,default=5)
    ap.add_argument("--warmup",type=int,default=20)
    ap.add_argument("--updates",type=int,default=2)
    ap.add_argument("--batch",type=int,default=1024)
    ap.add_argument("--k",type=int,default=8)
    ap.add_argument("--lr",type=float,default=2e-4)
    ap.add_argument("--eval-every",type=int,default=100)
    ap.add_argument("--out",default="runs/wm71")
    a=ap.parse_args(); dev="cuda"
    env=WarpEnv(a.envs,"assets/motions_trim")
    buf=WBuf(); rng=np.random.default_rng(0)
    wm=WM71(dev).to(dev); opt=torch.optim.Adam(wm.parameters(),lr=a.lr)
    xi=torch.zeros(a.envs,33,device=dev); xi[:,0]=0.75
    S=[[] for _ in range(a.envs)]; A=[[] for _ in range(a.envs)]
    evalbuf=WBuf(); ep_count=[0]
    prev=None
    def reset_prev():
        nonlocal prev
        s,_,_,q,v=env.current(); prev=torch.cat([q,v],-1)
    reset_prev()
    def flush(i):
        if len(S[i])<3: S[i].clear(); A[i].clear(); return
        if ep_count[0] % 5 == 0:
            evalbuf.add(S[i],A[i])
        else:
            buf.add(S[i],A[i])
        ep_count[0] += 1
        S[i].clear(); A[i].clear()
    def collect(steps):
        nonlocal prev
        for _ in range(steps):
            sprev=prev.clone()
            ref=env.qpos_t[env.clip,env.frame][:,7:36]
            act=((ref-Q0)/ASC).clamp(-5,5)
            act=(act+0.1*torch.randn_like(act)).clamp(-50,50)
            q,v,ob,r,term,trunc,done=env.step(act)
            snext=torch.cat([q,v],-1)
            sp=sprev.cpu().numpy(); an=act.cpu().numpy(); sn=snext.cpu().numpy()
            dn=done.cpu().numpy()
            for i in range(a.envs):
                S[i].append(sp[i]); A[i].append(an[i])
            # post-step state for done rows (before reset); env.step returned pre-reset q/v
            for i in np.nonzero(dn)[0]:
                S[int(i)].append(sn[i]); flush(int(i))
            # after step: non-done prev = snext; done prev = reset state
            s2,q2,v2=env.current()[0],env.current()[3],env.current()[4]
            prev=torch.cat([q2,v2],-1)
    # warmup
    for _ in range(a.warmup): collect(a.collect)
    print("warmup buffer",len(buf),flush=True)
    # held-out eval buffer
    he=WBuf(); S2=[[] for _ in range(64)]; A2=[[] for _ in range(64)]
    # simpler: reuse collection with 64 env by slicing is complex; eval on recent buffer later
    def eval_nse(wm, steps=32, n=256):
        if len(evalbuf)==0: return None
        s,act=evalbuf.sample(n,steps,dev); x=torch.zeros(n,33,device=dev); x[:,0]=0.75
        pred=s[:,0]; out=[]
        with torch.no_grad():
            for j in range(steps):
                pred=wm.step(pred,act[:,j],x)
                out.append(wm.spec.nse(pred,s[:,j+1]).mean().item())
        return out
    best=1e9
    for it in range(1,a.iters+1):
        t0=time.time(); collect(a.collect)
        for _ in range(a.updates):
            s,act=buf.sample(a.batch,a.k,dev)
            pred=s[:,0]; loss=0.0
            for j in range(a.k):
                pred=wm.step(pred,act[:,j],xi)
                e=wm.spec.error_vector(pred,s[:,j+1]); loss=loss+(e*e).mean()
            loss=loss/a.k
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(),1.0); opt.step()
        if it%50==0:
            print(f"[{it:4d}] loss {float(loss):.5f} buf {len(buf)} {time.time()-t0:.1f}s/it",flush=True)
        if it%a.eval_every==0:
            e=eval_nse(wm,8,256)
            print(f"[eval {it}] nse1 {e[0]:.5f} nse8 {e[7]:.5f} sqrt8 {e[7]**0.5:.4f}",flush=True)
            torch.save({"wm":wm.state_dict(),"iter":it,"nse8":e[7]},Path(a.out)/"best.pt")
    print("done")

if __name__=="__main__":
    main()
