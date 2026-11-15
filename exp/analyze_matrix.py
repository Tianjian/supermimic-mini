"""按固定迭代点(it20/it40)报数, 不做 best 选择; 并给单样本检验。"""
import json, glob, os, re
import numpy as np
from scipy import stats
BASE = 0.3798
rows = {}
for f in glob.glob("exp/runs/M_*.json") + glob.glob("exp/runs/W*_s*.json"):
    m = re.match(r"(?:M_)?([A-Za-z0-9_]+?)_s(\d+)\.json", os.path.basename(f))
    if not m: continue
    tag, seed = m.group(1), int(m.group(2))
    try: d = json.load(open(f))
    except Exception: continue
    it20 = next((e["std"] for e in d.get("log", []) if e["it"] == 20), None)
    it40 = d.get("std1")
    if it40 is None: continue
    rows.setdefault(tag, {})[seed] = (it20, it40)
print(f"基准(教师起手) = {BASE}\n")
print(f"{'配置':14s} {'n':>2s} {'it20':>10s} {'it40':>18s} {'相对':>8s} {'p':>7s}")
for tag in sorted(rows):
    v20 = [v[0] for v in rows[tag].values() if v[0] is not None]
    v40 = [v[1] for v in rows[tag].values()]
    a20, a40 = np.array(v20), np.array(v40)
    p = stats.ttest_1samp(a40, BASE).pvalue if len(a40) > 1 else float("nan")
    print(f"{tag:14s} {len(a40):2d} {(a20.mean() if len(a20) else float('nan')):10.4f} "
          f"{a40.mean():.4f}±{(a40.std(ddof=1) if len(a40)>1 else 0):.4f} "
          f"{100*(a40.mean()-BASE)/BASE:+7.1f}% {p:7.3f}")
