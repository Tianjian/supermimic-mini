"""采集 held-out clips 的数据用于 WM 泛化评测。"""
import sys, json
from pathlib import Path
import numpy as np, multiprocessing as mp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collect as C

if __name__ == "__main__":
    meta = json.load(open(C.OUT/"clips.json"))
    hold = meta["holdout"]
    per = 100_000
    jobs = [(90+i, [c], per, 900+i, 0.10, 0.02, 2.0) for i, c in enumerate(hold)]
    ps = [mp.Process(target=C.worker_holdout if hasattr(C,"worker_holdout") else C.worker, args=(j,)) for j in jobs]
    [p.start() for p in ps]; [p.join() for p in ps]
    # 重命名 shard_9X 为 holdout shard
    for i in range(len(hold)):
        for suf in ("", ):
            src = C.OUT/f"shard_{90+i:02d}.npz"
            if src.exists():
                src.rename(C.OUT/f"holdout_{i:02d}.npz")
    print("holdout done", flush=True)
