"""对比教师 vs MPC 的整回合结果, 并按阶段三门禁判定。"""
import json, sys
import numpy as np
from scipy import stats

BASE = 0.3798        # 教师基准
GATE_G2 = BASE * 1.05  # 0.4056

def load(p):
    d = json.load(open(p))
    return d

def main():
    t = load("exp/runs/mpc_teacher500.json")
    print(f"{'clip':24s} {'教师':>8s} {'存活':>6s}")
    for n, s, v in zip(t["clips"], t["scores"][0], t["surv"][0]):
        print(f"  {n:22s} {s:8.4f} {v:6.0f}")
    print(f"  {'均值':22s} {t['mean']:8.4f} {t['surv_mean']:6.0f}")
    for f in sys.argv[1:]:
        try: m = load(f)
        except Exception as e:
            print(f"\n{f}: 无法读取 ({e})"); continue
        ms = np.array(m["scores"]); mv = np.array(m["surv"])
        print(f"\n=== {m['mode']} H={m['H']} bound={m['bound']} K={m['opt_steps']} "
              f"every={m['every']} (n={len(ms)}) ===")
        ts = np.array(t["scores"][0])
        for i, n in enumerate(m["clips"]):
            print(f"  {n:22s} {ms[:, i].mean():8.4f}  存活 {mv[:, i].mean():4.0f}  "
                  f"(教师 {ts[i]:.4f})")
        delta = ms.mean() - t["mean"]
        p = stats.ttest_1samp(ms.mean(1), t["mean"]).pvalue if len(ms) > 1 else float("nan")
        print(f"  {'均值':22s} {ms.mean():8.4f}  存活 {mv.mean():4.0f}  "
              f"相对教师 {100*delta/t['mean']:+.2f}%")
        print(f"\n  门禁 G1 存活>=400 : {'✅' if mv.mean() >= 400 else '❌'} ({mv.mean():.0f})")
        print(f"  门禁 G2 >={GATE_G2:.4f} : {'✅' if ms.mean() >= GATE_G2 else '❌'} ({ms.mean():.4f})")
        reg = [(m['clips'][i], ms[:, i].mean() / ts[i] - 1) for i in range(len(m['clips']))]
        worst = min(reg, key=lambda x: x[1])
        print(f"  门禁 G3 无 clip 回退>10% : {'✅' if worst[1] > -0.10 else '❌'} "
              f"(最差 {worst[0]} {100*worst[1]:+.1f}%)")
        if len(ms) > 1:
            print(f"  p(单样本 vs 教师) = {p:.4f}")

if __name__ == "__main__":
    main()
