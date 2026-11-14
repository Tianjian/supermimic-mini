"""按论文/参考实现超参训练 71 维世界模型, 评测 held-out 8 步 NSE。

对照:
  作者机 (2060): 1024^3, batch 1024, 1000 iters x4 updates = 4k 梯度步 -> NSE8 0.84-1.28
  参考实现     : 2048^3, ~100k 梯度步 -> NSE8 0.0065-0.0099
  本机 (本脚本): 2048^3, batch 2048, k=8, fp16 AMP, 目标 ~80k 梯度步
"""
import sys, os, json, time, argparse
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smini.wm_import import WM71

DATA = ROOT/"exp"/os.environ.get("DATA_DIR", "data")
OUTDIR = ROOT/"exp"/"runs"; OUTDIR.mkdir(parents=True, exist_ok=True)


def make_wm(xi_dim=34, device="cuda"):
    """WM71 但 xi 维度可配。xi34 = [左脚摩擦, 右脚摩擦, 29 编码器偏置, 3 质心偏移]。"""
    wm = WM71(device)
    if xi_dim == 33:
        return wm
    import torch.nn as nn
    old = wm.dynamics.mlp.net[0]
    wm.dynamics.mlp.net[0] = nn.Linear(71 + 29 + xi_dim, old.out_features).to(device)
    wm.register_buffer("xi_offset", torch.cat(
        [torch.full((2,), 0.75), torch.zeros(29), torch.zeros(3)]).to(device))
    wm.register_buffer("xi_scale", torch.cat(
        [torch.full((2,), 0.45), torch.full((29,), 0.01), torch.full((3,), 0.05)]).to(device))
    return wm


def load_shards(prefix, n_envs=8):
    """读取 shard 并把 tick-major (T*NE) 重排成 env-major, 使每个 env 的时序连续。

    采集器按 [t0e0..t0e7, t1e0..t1e7, ...] 存储, 相邻索引属于不同 env;
    必须 reshape 成 (NE,T) 才能得到真实动力学转移。
    """
    files = sorted(DATA.glob(f"{prefix}*.npz"))
    if not files: raise SystemExit(f"no shards for {prefix}")
    S=[];A=[];X=[];E=[]; allv=[]
    pos=0
    for f in files:
        d = np.load(f)
        s,a,x,e = d["S"], d["A"], d["X"], d["E"]
        n = len(e); assert n % n_envs == 0, (f, n)
        T = n // n_envs
        # (T,NE,·) -> (NE,T,·)
        s = s.reshape(T, n_envs, -1).transpose(1,0,2).reshape(n, -1)
        a = a.reshape(T, n_envs, -1).transpose(1,0,2).reshape(n, -1)
        x = x.reshape(T, n_envs, -1).transpose(1,0,2).reshape(n, -1)
        e = e.reshape(T, n_envs).T.reshape(n)
        # 每个 env 运行段的最后一帧也要断开 (否则窗口跨 env)
        for k in range(n_envs):
            e[(k+1)*T - 1] = 1
        S.append(s); A.append(a); X.append(x); E.append(e)
        cs = np.concatenate([[0], np.cumsum(e.astype(np.int64))])
        allv.append((pos, n, cs)); pos += n
    S=np.concatenate(S); A=np.concatenate(A); X=np.concatenate(X); E=np.concatenate(E)
    return S, A, X, E, allv


def build_valid(E, allv, k):
    n=len(E); valid=np.zeros(n, bool)
    for pos, ln, cs in allv:
        # i in [pos, pos+ln-1-k]
        hi = pos+ln-k
        if hi<=pos: continue
        idx = np.arange(pos, hi)
        win_sum = cs[idx-pos+k] - cs[idx-pos]      # E[i..i+k-1]
        ok = win_sum==0
        valid[idx[ok]] = True
    return valid


def make_eval_windows(S, A, X, E, allv, k=32, nwin=512, seed=0):
    """从 held-out 取 nwin 段连续 k 步窗口(不跨 episode)"""
    rng=np.random.default_rng(seed); out=[]
    for pos, ln, cs in allv:
        idx=np.arange(pos, pos+ln-k)
        if len(idx)==0: continue
        win = cs[idx-pos+k] - cs[idx-pos]
        out.append(idx[win==0])
    cand=np.concatenate(out) if out else np.array([],dtype=int)
    if len(cand)==0: raise SystemExit("no valid eval windows")
    pick=rng.choice(cand, size=min(nwin,len(cand)), replace=False)
    return pick


@torch.no_grad()
def eval_nse(wm, S, A, X, starts, k=32, bs=256):
    """开环 rollout, 返回每步 NSE 均值 (长度 k)"""
    wm.eval(); acc=np.zeros(k); cnt=0
    for i in range(0, len(starts), bs):
        idx = starts[i:i+bs]
        s = torch.as_tensor(S[idx], device="cuda", dtype=torch.float32)
        xi = torch.as_tensor(X[idx], device="cuda", dtype=torch.float32)
        row=np.zeros((len(idx), k))
        pred=s
        for t in range(k):
            a = torch.as_tensor(A[idx+t], device="cuda", dtype=torch.float32)
            pred = wm.step(pred, a, xi)
            true = torch.as_tensor(S[idx+t+1], device="cuda", dtype=torch.float32)
            row[:,t] = wm.spec.nse(pred, true).cpu().numpy()
        acc += row.sum(0); cnt += len(idx)
    wm.train()
    return acc/cnt


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80000, help="梯度步总数")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--tag", default="paper")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--xi-dim", type=int, default=34)
    a=ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(1)
    S,A,X,E,allv = load_shards("shard_")
    print(f"train transitions {len(S):,}", flush=True)
    valid = build_valid(E, allv, a.k)
    print(f"valid window starts {valid.sum():,} / {len(valid):,}", flush=True)

    try:
        Sh,Ah,Xh,Eh,allvh = load_shards("holdout_")
        ev_starts = make_eval_windows(Sh, Ah, Xh, Eh, allvh, k=32, nwin=512)
        print(f"held-out eval windows {len(ev_starts)}", flush=True)
        has_eval=True
    except SystemExit:
        print("!! 无 held-out 数据, 跳过泛化评测", flush=True); has_eval=False

    # 数据常驻 GPU: 省掉每步 9 次 CPU gather + H2D 拷贝 (实测省 57ms/step)
    St = torch.as_tensor(S, device="cuda")
    At = torch.as_tensor(A, device="cuda")
    Xt = torch.as_tensor(X, device="cuda")
    print(f"GPU data S{tuple(St.shape)} {St.numel()*4/1024**3:.2f}GB "
          f"A{tuple(At.shape)} X{tuple(Xt.shape)}", flush=True)
    vstarts = torch.as_tensor(np.nonzero(valid)[0], device="cuda")
    wm = make_wm(a.xi_dim).to("cuda")
    print(f"WM params {sum(p.numel() for p in wm.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.Adam(wm.parameters(), lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(a.amp))
    ema = make_wm(a.xi_dim).to("cuda"); ema.load_state_dict(wm.state_dict())
    for p in ema.parameters(): p.requires_grad_(False)
    best=1e9
    start_step = 1
    rk = OUTDIR/f"{a.tag}_last.pt"
    if a.resume and rk.exists():
        ck = torch.load(rk, map_location="cuda", weights_only=False)
        wm.load_state_dict(ck["wm"])
        if "ema" in ck: ema.load_state_dict(ck["ema"])
        start_step = int(ck.get("step", 0)) + 1
        print(f"RESUME from step {start_step-1}", flush=True)
    log=[]
    t0=time.time()
    for step in range(start_step, a.steps+1):
        lr = a.lr*(0.03+0.97*0.5*(1+np.cos(np.pi*step/a.steps)))
        for g in opt.param_groups: g["lr"]=lr
        idx = vstarts[torch.randint(0, len(vstarts), (a.batch,), device="cuda")]
        s = St[idx]; xi = Xt[idx]
        loss=0.0
        with torch.autocast("cuda", dtype=torch.float16, enabled=bool(a.amp)):
            pred=s
            for j in range(a.k):
                aa = At[idx+j]
                pred = wm.step(pred, aa, xi)
                e = wm.spec.error_vector(pred, St[idx+j+1])
                loss = loss + (e*e).mean()
        loss = loss/a.k
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(wm.parameters(),1.0)
        scaler.step(opt); scaler.update()
        with torch.no_grad():
            for pe,p in zip(ema.parameters(), wm.parameters()):
                pe.mul_(0.999).add_(p, alpha=1e-3)
        if step % 500 == 0:
            print(f"[{step:6d}] loss {float(loss):.4f} lr {lr:.2e} "
                  f"{(time.time()-t0)/step*1000:.0f}ms/step elapsed {(time.time()-t0)/60:.1f}m", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            if has_eval:
                nse = eval_nse(wm, Sh, Ah, Xh, ev_starts, k=32)
                nse_e = eval_nse(ema, Sh, Ah, Xh, ev_starts, k=32)
                msg=(f"[EVAL {step}] heldout NSE1={nse[0]:.5f} NSE8={nse[7]:.5f} NSE32={nse[-1]:.5f} "
                     f"| ema NSE8={nse_e[7]:.5f}")
                print(msg, flush=True); log.append({"step":step,"nse":nse.tolist(),"nse_ema":nse_e.tolist()})
                json.dump(log, open(OUTDIR/f"{a.tag}_log.json","w"), indent=1)
                if nse_e[7] < best:
                    best=nse_e[7]; torch.save({"wm":ema.state_dict(),"step":step,"nse8":best}, OUTDIR/f"{a.tag}_best.pt")
            torch.save({"wm":wm.state_dict(),"ema":ema.state_dict(),"step":step}, OUTDIR/f"{a.tag}_last.pt")
    print(f"DONE {a.steps} steps in {(time.time()-t0)/60:.1f} min; best ema NSE8={best:.5f}", flush=True)


if __name__ == "__main__":
    main()
