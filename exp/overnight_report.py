"""汇总 overnight 结果 -> docs/OVERNIGHT-REPORT.md"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"exp"/"runs"
OUT = ROOT/"docs"/"OVERNIGHT-REPORT.md"
BASE_TEACHER = 0.3798      # 教师底座在 evalkit 协议下的起手分


def load(tag):
    p = RUNS/f"{tag}.json"
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def ac_row(tag):
    d = load(tag)
    if not d or "std0" not in d:
        return None
    return {"tag": tag, "base": d["std0"], "final": d.get("std1"),
            "delta": (d.get("std1", 0)-d["std0"]),
            "rel": (d.get("std1", 0)/d["std0"]-1)*100 if d["std0"] else float("nan")}


def main():
    lines = ["# Overnight 执行报告", "",
             f"> 生成时间：{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}",
             "> 协议：`exp/evalkit.py`（裁切数据 / 固定分母 / 6 clip 分层）",
             "> 底座：**教师网络**（`supermimic_getup`，零蒸馏误差）", ""]

    # ---------------- A1 ----------------
    lines += ["## 阶段 A1：教师底座 + BPTT（消掉策略质量混杂）", ""]
    groups = {}
    for H in (16, 8):
        for kl in ("0", "002"):
            rows = []
            for s in range(3):
                r = ac_row(f"A1_H{H}_kl{kl}_s{s}")
                if r: rows.append(r)
            if rows:
                groups[f"H={H} kl={kl}"] = rows
    if groups:
        lines += ["| 配置 | 起手 | 各种子最终 | 均值 ± std | 相对 |", "|---|---|---|---|---|"]
        for k, rows in groups.items():
            f = np.array([r["final"] for r in rows])
            b = rows[0]["base"]
            lines.append(f"| {k} | {b:.4f} | "
                         f"{', '.join(f'{x:.4f}' for x in f)} | "
                         f"{f.mean():.4f} ± {f.std(ddof=1) if len(f)>1 else 0:.4f} | "
                         f"{100*(f.mean()-b)/b:+.1f}% |")
        best = max(groups.items(), key=lambda kv: np.mean([r["final"] for r in kv[1]]))
        bm = np.mean([r["final"] for r in best[1]])
        lines += ["", f"**A1 最好配置**：{best[0]}，均值 **{bm:.4f}**"
                      f"（起手 {best[1][0]['base']:.4f}，相对 {100*(bm-best[1][0]['base'])/best[1][0]['base']:+.1f}%）", ""]
        lines += [("✅ **A1 通过**：BPTT 在教师底座上带来正向收益 → 之前的失败主要是底座问题"
                   if bm > BASE_TEACHER * 1.02 else
                   "❌ **A1 未通过**：BPTT 在教师底座上仍无正向收益 → 瓶颈在 WM，与策略质量无关"), ""]
    else:
        lines += ["（A1 无可用结果）", ""]

    # ---------------- B ----------------
    lines += ["## 阶段 B：on-policy WM 与动作幅度偏置", ""]
    brows = []
    for tag in ("B_base", "B_mag", "B_mag2", "B_mag3"):
        d = load(tag)
        if not d: continue
        log = d.get("log") or []
        if not log: continue
        last = log[-1]
        brows.append({"tag": tag,
                      "nse8": last["nse"][7], "nse32": last["nse"][31],
                      "mag": last.get("mag_ratio"),
                      "base32": (d.get("base") or [None]*32)[31] if d.get("base") else None})
    if brows:
        lines += ["| WM | on-policy NSE8 | on-policy NSE32 | 起手 NSE32 | 大幅/小幅 误差比 |",
                  "|---|---|---|---|---|"]
        for r in brows:
            lines.append(f"| {r['tag']} | {r['nse8']:.4f} | {r['nse32']:.4f} | "
                         f"{(f'{r['base32']:.4f}' if r['base32'] else '-')} | "
                         f"{(f'{r['mag']:.2f}' if r['mag'] else '-')} |")
        ok = [r for r in brows if r["mag"] and r["mag"] <= 1.5 and r["nse32"] <= 0.25]
        lines += ["", ("✅ **B 门禁通过**（存在 mag_ratio ≤1.5 且 NSE32 ≤0.25 的 WM）"
                       if ok else
                       "⚠️ **B 门禁未通过**：幅度偏置或 NSE32 未达标，C 阶段结果需谨慎解读"), ""]
    else:
        lines += ["（B 无可用结果）", ""]

    # ---------------- C ----------------
    lines += ["## 阶段 C：用改进 WM 重试 BPTT", ""]
    cg = {}
    for H in (16, 8):
        for anch in (1, 0):
            rows = []
            for s in range(3):
                r = ac_row(f"C_H{H}_anchor{anch}_s{s}")
                if r: rows.append(r)
            if rows: cg[f"C H={H} anchor={anch}"] = rows
    rows = []
    for s in range(3):
        r = ac_row(f"C_H16_s{s}")
        if r: rows.append(r)
    if rows: cg["C H=16 (改进 WM, 无锚定)"] = rows
    if cg:
        lines += ["| 配置 | 起手 | 各种子最终 | 均值 ± std | 相对 |", "|---|---|---|---|---|"]
        for k, rows in cg.items():
            f = np.array([r["final"] for r in rows]); b = rows[0]["base"]
            lines.append(f"| {k} | {b:.4f} | {', '.join(f'{x:.4f}' for x in f)} | "
                         f"{f.mean():.4f} ± {f.std(ddof=1) if len(f)>1 else 0:.4f} | "
                         f"{100*(f.mean()-b)/b:+.1f}% |")
    else:
        lines += ["（C 无可用结果）", ""]

    lines += ["", "---", "",
              "## 解读提示", "",
              "- 所有分数都是 evalkit 协议（零动作地板约 0.0496、PD 约 0.0578、教师约 0.386）",
              "- `std0` 是本次 run 自己的起手分；教师底座理论值 0.3798，略有波动属正常",
              "- 单次 run 的评测噪声约 ±0.02，**结论要看 3 个种子的方向是否一致**",
              ""]
    OUT.write_text("\n".join(lines))
    print("written", OUT)


if __name__ == "__main__":
    main()
