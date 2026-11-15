"""生成 README 用的图表 (中文字体: Noto Sans CJK)。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT/"docs"/"figures"; FIG.mkdir(parents=True, exist_ok=True)
rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 140
C = {"teacher": "#4C72B0", "mpc": "#2E8B57", "bad": "#C44E52", "mut": "#8C8C8C",
     "ours": "#DD8452", "paper": "#55A868"}


def save(fig, name):
    p = FIG/name
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", p)


# ---------- F1: 规划收益 vs 视野 ----------
def f1():
    rows = []
    for H in (8, 16, 32):
        f = ROOT/f"exp/runs/mpc_tp_H{H}.json"
        if not f.exists(): continue
        d = json.load(open(f))
        tc, pl = np.array(d["teacher"]), np.array(d["plan_from_teacher"])
        rows.append((H, 100*(pl.mean()-tc.mean())/tc.mean(), d["N"],
                     (pl > tc).mean()*100))
    if not rows: return
    H = [r[0] for r in rows]; g = [r[1] for r in rows]
    wr = [r[3] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    b = ax[0].bar([str(h) for h in H], g, color=[C["mpc"] if x > 2 else C["mut"] for x in g])
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].set_xlabel("规划视野 H"); ax[0].set_ylabel("真值回报提升 (%)")
    ax[0].set_title("规划收益随视野衰减")
    for r, v in zip(b, g):
        ax[0].text(r.get_x()+r.get_width()/2, v, f"{v:+.1f}%", ha="center",
                   va="bottom" if v > 0 else "top", fontsize=9)
    ax[1].bar([str(h) for h in H], wr, color=C["teacher"], alpha=.85)
    ax[1].axhline(50, color=C["bad"], ls="--", lw=1, label="50%（随机）")
    ax[1].set_ylim(0, 105); ax[1].set_xlabel("规划视野 H")
    ax[1].set_ylabel("改善样本占比 (%)"); ax[1].set_title("胜率同步骤下降")
    ax[1].legend(fontsize=8)
    for i, v in enumerate(wr):
        ax[1].text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); save(fig, "f1-planning-vs-horizon.png")


# ---------- F2: 扰动对照 ----------
def f2():
    f = ROOT/"exp/runs/mpc_tp_ctrl_H8.json"
    if not f.exists(): return
    d = json.load(open(f))
    tc = np.array(d["teacher"]); pl = np.array(d["plan_from_teacher"])
    base = tc.mean()
    items = [("教师基线", 0.0, C["teacher"]),
             ("随机扰动 ±0.3", 100*(np.mean(d["random_perturb"])-base)/base, C["bad"]),
             ("常数缩放 ×1.05", 100*(d["scale_1.05"]-base)/base, C["mut"]),
             ("WM 规划", 100*(pl.mean()-base)/base, C["mpc"])]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    b = ax.bar([x[0] for x in items], [x[1] for x in items], color=[x[2] for x in items])
    ax.axhline(0, color="k", lw=.8)
    ax.set_ylabel("相对教师的真值回报 (%)")
    ax.set_title("H=8：只有世界模型引导的优化才有收益")
    for r, it in zip(b, items):
        ax.text(r.get_x()+r.get_width()/2, it[1], f"{it[1]:+.2f}%", ha="center",
                va="bottom" if it[1] >= 0 else "top", fontsize=9)
    plt.xticks(rotation=12, fontsize=9)
    fig.tight_layout(); save(fig, "f2-perturbation-control.png")


# ---------- F3: K 敏感性 ----------
def f3():
    t = {n: s for n, s in zip(
        ["dance", "walk", "run", "sprint", "jumps", "fall"],
        [0.6151, 0.5375, 0.3387, 0.3379, 0.3712, 0.0782])}
    k25 = [0.6641, 0.5656, 0.3469, 0.3879, 0.2783, 0.0888]
    k12 = [0.6313, 0.5487, 0.4368, 0.2778, 0.3149, 0.0871]
    names = list(t.keys()); base = np.array(list(t.values()))
    a = 100*(np.array(k25)/base-1); b = 100*(np.array(k12)/base-1)
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(x-w/2, a, w, label="K=25", color=C["ours"])
    ax.bar(x+w/2, b, w, label="K=12", color=C["mpc"])
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("相对教师 (%)"); ax.set_title("同一规划器、只改每步梯度数 K：逐 clip 效应变号")
    ax.legend(fontsize=9)
    for i in range(len(names)):
        ax.annotate("", xy=(x[i]+w/2, b[i]), xytext=(x[i]-w/2, a[i]),
                    arrowprops=dict(arrowstyle="->", color=C["mut"], lw=1, alpha=.7))
    ax.text(0.99, 0.03, "箭头=同一 clip 在两种配置间的变化", transform=ax.transAxes,
            ha="right", fontsize=8, color=C["mut"])
    fig.tight_layout(); save(fig, "f3-K-sensitivity.png")


# ---------- F4: 逐步奖励曲线 ----------
def f4():
    ft = ROOT/"exp/runs/curve_teacher.json"
    fm = ROOT/"exp/runs/curve_mpc.json"
    if not (ft.exists() and fm.exists()): return
    T = json.load(open(ft)); M = json.load(open(fm))
    if "rew_hist" not in T or "rew_hist" not in M:
        print("缺少 rew_hist, 跳过 F4"); return
    rt = np.array(T["rew_hist"]); rm = np.array(M["rew_hist"])
    n = rt.shape[1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    # 左: 平均单步奖励 (滑动平均)
    def smooth(x, w=25):
        k = np.ones(w)/w
        return np.convolve(x, k, mode="valid")
    mt = rt.mean(1); mm = rm.mean(1)
    ax[0].plot(smooth(mt), color=C["teacher"], lw=1.6, label="教师")
    ax[0].plot(smooth(mm), color=C["mpc"], lw=1.6, label="MPC (K=12)")
    ax[0].set_xlabel("控制步"); ax[0].set_ylabel("平均单步奖励")
    ax[0].set_title("逐步奖励（25 步滑动平均）"); ax[0].legend(fontsize=9)
    ax[0].grid(alpha=.25)
    # 右: 累计归一化分数
    ct = np.cumsum(rt, 0).mean(1)/(0.1*np.arange(1, len(rt)+1))
    cm = np.cumsum(rm, 0).mean(1)/(0.1*np.arange(1, len(rm)+1))
    ax[1].plot(ct, color=C["teacher"], lw=1.6, label="教师")
    ax[1].plot(cm, color=C["mpc"], lw=1.6, label="MPC (K=12)")
    ax[1].set_xlabel("控制步"); ax[1].set_ylabel("累计归一化分数")
    ax[1].set_title(f"累计分数：教师 {ct[-1]:.4f} vs MPC {cm[-1]:.4f}")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.25)
    fig.tight_layout(); save(fig, "f4-per-step-curves.png")



# ---------- F5: 全项目排行榜 ----------
def f5():
    rows = [
        ("零动作（地板）",            0.0496, C["mut"]),
        ("PD 跟踪",                   0.0578, C["mut"]),
        ("BPTT 训练（无约束）",       0.0651, C["bad"]),
        ("BPTT 训练（信赖域）",       0.0670, C["bad"]),
        ("BC 蒸馏",                   0.0912, C["ours"]),
        ("BPTT + 全机制 + 夹具",      0.3811, C["bad"]),
        ("MPC 闭环（K=12）",          0.3828, C["mpc"]),
        ("MPC 闭环（K=25）",          0.3886, C["mpc"]),
        ("仓库教师（外部权重）",      0.3798, C["teacher"]),
    ]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=.62)
    ax.axvline(0.3798, color=C["teacher"], ls="--", lw=1.2, alpha=.9)
    ax.text(0.3798, -0.95, "教师基线 0.3798", color=C["teacher"],
            fontsize=8, ha="center", va="top")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("evalkit 整回合分数（6 clip × 500 步）")
    ax.set_title("所有方法都在教师基线上方集结，但没有任何一个稳定越过它")
    for i, r in enumerate(rows):
        ax.text(r[1]+0.004, i, f"{r[1]:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 0.46); ax.set_ylim(-1.3, len(rows)-0.4)
    fig.tight_layout(); save(fig, "f5-leaderboard.png")

if __name__ == "__main__":
    f1(); f2(); f3(); f5(); f4()
