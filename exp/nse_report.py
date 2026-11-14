"""NSE 报告 (修正版): 一律报 persistence 比值, 使跨分布可比。

修 CRITICAL-REVIEW B3: NSE 绝对值强依赖数据分布难度; 用 NSE / persistence_NSE
作为分布无关的标定。比值 <1 表示优于"状态不动"的预测。

用法:
  python exp/nse_report.py --ckpt exp/runs/v3_best.pt
  python exp/nse_report.py --ckpt exp/runs/wm_onpolicy.pt --dist onpolicy
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"exp"))
import wm_train
from smini import motion
from smini.policy import Policy
from wm_train import make_wm, load_shards
from wm_onpolicy import rollout_policy, make_eval_windows

torch.set_num_threads(2)


@torch.no_grad()
def nse_rollout(wm, S, A, X, starts, k=32, bs=256, kind="model"):
    acc = np.zeros(k); cnt = 0
    for i in range(0, len(starts), bs):
        idb = starts[i:i+bs]
        s = torch.as_tensor(S[idb]).cuda(); xi = torch.as_tensor(X[idb]).cuda()
        p = s; row = np.zeros((len(idb), k))
        for t in range(k):
            if kind == "model":
                p = wm.step(p, torch.as_tensor(A[idb+t]).cuda(), xi)
            else:
                p = p          # persistence: 状态不动
            row[:, t] = wm.spec.nse(p, torch.as_tensor(S[idb+t+1]).cuda()).cpu().numpy()
        acc += row.sum(0); cnt += len(idb)
    return acc / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT/"exp/runs/v3_best.pt"))
    ap.add_argument("--dist", default="both", choices=["explore", "onpolicy", "both"])
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--nwin", type=int, default=512)
    ap.add_argument("--out", default=str(ROOT/"exp/runs/nse_report.json"))
    a = ap.parse_args()

    wm = make_wm(34).to("cuda")
    sd = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    wm.load_state_dict(sd["wm"] if "wm" in sd else sd)
    print(f"WM: {Path(a.ckpt).name} (step {sd.get('step')})", flush=True)

    sets = {}
    if a.dist in ("explore", "both"):
        wm_train.DATA = ROOT/"exp"/"data_v2"
        S, A, X, E, _ = load_shards("holdout_")
        sets["explore"] = (S, A, X, E)
    if a.dist in ("onpolicy", "both"):
        meta = json.load(open(ROOT/"exp/data_v2/clips.json"))
        lib = motion.MotionLibrary.from_dir(str(ROOT/"assets/motions"))
        names = [m.name for m in lib.motions]
        hold = [names.index(n) for n in meta["holdout"]]
        pol = Policy(obs_dim=93, goal_dim=730, action_dim=29, hist=10).cuda()
        ck = torch.load(str(ROOT/"exp/runs/p0_clean.pt"), map_location="cuda", weights_only=False)
        pol.load_state_dict(ck["policy"]); pol.eval()
        rng = np.random.default_rng(0)
        S, A, X, E = rollout_policy(lib, pol, hold, 30000, rng)
        sets["onpolicy"] = (S, A, X, E)

    out = {"ckpt": str(a.ckpt), "step": sd.get("step"), "dists": {}}
    print(f"\n{'分布':10s} {'NSE1':>8s} {'NSE8':>8s} {'NSE32':>8s} | "
          f"{'pers8':>8s} {'ratio8':>8s} {'ratio32':>8s}", flush=True)
    for name, (S, A, X, E) in sets.items():
        ev = make_eval_windows(S, A, X, E, k=a.k, nwin=a.nwin)
        m = nse_rollout(wm, S, A, X, ev, k=a.k, kind="model")
        p = nse_rollout(wm, S, A, X, ev, k=a.k, kind="persist")
        print(f"{name:10s} {m[0]:8.4f} {m[7]:8.4f} {m[31]:8.4f} | "
              f"{p[7]:8.4f} {m[7]/p[7]:8.4f} {m[31]/p[31]:8.4f}", flush=True)
        out["dists"][name] = {"nse": m.tolist(), "persist": p.tolist(),
                              "ratio8": float(m[7]/p[7]), "ratio32": float(m[31]/p[31])}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nsaved {a.out}")
    print("ratio<1 表示优于 persistence; 只有这个比值才可跨分布比较")


if __name__ == "__main__":
    main()
