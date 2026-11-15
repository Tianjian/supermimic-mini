# 文档导航

本目录按**文档的性质**分四类。想快速了解结论，直接读根目录 [README](../README.md)；想深入某个环节，按下表找。

```
docs/
├── findings/     结论 —— 项目的主线产出（读这些就够）
├── technical/    实现细节 —— 世界模型、奖励、BPTT 的具体做法
├── reviews/      评审记录 —— 自我审计与批判性评审
├── archive/      历史归档 —— 早期计划与进度日志，不再维护
└── demo/         演示视频与动图
```

> **一句话结论**：机制成立（动作条件化 81%、梯度符号一致 96.2%、规划 +3.79%~+7.29%），
> 但「用 BPTT 训练策略网络」在两档消费级预算下都是负结果。
> 原因是算力 → 精度 → 机制这条链上有五道检查，只过了两道。

---

## 按问题找文档

| 我想知道… | 读这篇 |
|---|---|
| 项目到底得出了什么结论 | [README](../README.md) |
| **为什么算力不够会导致机制失效** | [findings/compute-precision-attribution](findings/compute-precision-attribution.md) |
| 把论文缺的机制补上会怎样 | [findings/mechanism-ablation-p0-p2](findings/mechanism-ablation-p0-p2.md) |
| 为什么底座够好也救不了 BPTT | [findings/zero-distill-base-collapses](findings/zero-distill-base-collapses.md) |
| 6 GB 阶段做了什么、结论是什么 | [findings/phase1-final-report](findings/phase1-final-report.md) |
| 阶段一的机制验证原始数据 | [findings/phase1-wm-bptt-validation](findings/phase1-wm-bptt-validation.md) |
| 世界模型和 BPTT 具体怎么实现 | [technical/wm-bptt-16gb](technical/wm-bptt-16gb.md) |
| 奖励头为什么是堵点 | [technical/reward-head-and-actor-critic](technical/reward-head-and-actor-critic.md) |
| BPTT 为什么会崩、怎么稳 | [technical/bptt-stability](technical/bptt-stability.md) |
| 我在设计上犯了哪些错 | [reviews/critical-review-18-items](reviews/critical-review-18-items.md) |
| 演示视频 | [demo/README](demo/README.md) |

---

## findings/ —— 结论（主线）

| 文档 | 日期 | 内容 |
|---|---|---|
| [compute-precision-attribution.md](findings/compute-precision-attribution.md) | 11-15 | **核心**。与论文/开源复现逐维对照，把失败归因到「算力 → 精度 → 机制」，含 68 万倍想象样本差距的换算 |
| [mechanism-ablation-p0-p2.md](findings/mechanism-ablation-p0-p2.md) | 11-15 | **核心**。实现可微 bootstrap / survival weight / 在线循环并消融：三者装回去仍无增益，它们是「精度放大器」而非「修复项」 |
| [zero-distill-base-collapses.md](findings/zero-distill-base-collapses.md) | 11-14 | 用零蒸馏误差的教师做底座，BPTT 仍把它打崩 −79.8%（3/3 种子），排除「底座太弱」 |
| [overnight-abcde.md](findings/overnight-abcde.md) | 11-15 | 一夜五组实验（A/B/C/D/E）的完整数据与自我修正记录 |
| [phase1-final-report.md](findings/phase1-final-report.md) | 11-10 | 阶段一（RTX 2060 6 GB）总报告 |
| [phase1-wm-bptt-validation.md](findings/phase1-wm-bptt-validation.md) | 11-09 | 阶段一的 WM/BPTT 机制验证：动作敏感度 7.5% 的诊断 |

## technical/ —— 实现与参考

| 文档 | 日期 | 内容 |
|---|---|---|
| [wm-bptt-16gb.md](technical/wm-bptt-16gb.md) | 11-14 | **核心**。16 GB AMD 卡上的世界模型架构（71 维状态、2048³、k=8）与 BPTT 实现 |
| [reward-head-and-actor-critic.md](technical/reward-head-and-actor-critic.md) | 11-14 | 补 14 个 body link 奖励项 + 接完整 actor-critic 的第四轮实验 |
| [bptt-stability.md](technical/bptt-stability.md) | 11-09 | 信赖域、WM 快照、raw 评测等稳定性改进 |
| [bptt-minimal-fair-test.md](technical/bptt-minimal-fair-test.md) | 11-09 | BPTT 的最小公平测试设计 |
| [wm-upgrade-survey.md](technical/wm-upgrade-survey.md) | 11-10 | 世界模型升级调研：导入 vs 重训 |
| [mjx-feasibility.md](technical/mjx-feasibility.md) | 11-10 | MJX 作为精确可微 WM 的可行性实测（结论：不可用） |
| [paper-gap.md](technical/paper-gap.md) | 11-09 | 与论文的 gap 分析 |

## reviews/ —— 评审记录

| 文档 | 日期 | 内容 |
|---|---|---|
| [critical-review-18-items.md](reviews/critical-review-18-items.md) | 11-14 | 18 条设计与实现的可改进处，含被推翻的 3 个早期论断与修正协议下的重测 |
| [setup-audit.md](reviews/setup-audit.md) | 11-09 | 实验设定审计 |
| [p3-warp-audit.md](reviews/p3-warp-audit.md) | 11-09 | Warp 迁移的设定与实现审计 |
| [phase1-data-review.md](reviews/phase1-data-review.md) | 11-09 | 阶段一数据与方案复盘（按口径列可信数据） |

## archive/ —— 历史归档（不再维护）

早期计划、进度日志与被取代的报告。保留是为了可追溯，**结论以 findings/ 为准**。

| 文档 | 日期 | 内容 |
|---|---|---|
| [review-and-plan.md](archive/review-and-plan.md) | 11-14 | P0–P3 路线与里程碑 M1–M4 |
| [next-plan.md](archive/next-plan.md) | 11-14 | 后续优化的设计与实验规划（已被 findings 取代） |
| [overnight-plan.md](archive/overnight-plan.md) | 11-14 | 8 小时无人值守执行计划 |
| [morning-plan.md](archive/morning-plan.md) | 11-15 | M1–M4 早间实验策略 |
| [roadmap-v2.md](archive/roadmap-v2.md) | 11-09 | 突破路线图 v2 |
| [wm-bptt-plan.md](archive/wm-bptt-plan.md) | 11-09 | 早期的 WM/BPTT 机制验证方案 |
| [bottleneck-survey.md](archive/bottleneck-survey.md) | 11-09 | 突破瓶颈的学术调研 |
| [progress-log.md](archive/progress-log.md) | 11-09 | M0–M4 进度日志 |
| [m3-report.md](archive/m3-report.md) | 11-06 | M0–M3 阶段验收报告 |
| [final-report-legacy.md](archive/final-report-legacy.md) | 11-09 | 早期总报告（已被取代，口径过时） |

---

## 阅读约定

- **findings/ 与 technical/ 中的数字都来自 `exp/evalkit.py` 协议**（裁切数据 + 固定步数分母 + 零动作/PD 基线）。早期文档若出现 0.46 / 0.58 一类分数，属于已废弃的旧协议，已在 reviews 中说明。
- 任何「提升」必须同时报**绝对基线与存活步数**；单 run 评测噪声约 ±0.019。
- `archive/` 里的计划文档反映的是**当时的判断**，其中若干已被后续实验证伪（例如「补齐三个机制就能修好 BPTT」）。
