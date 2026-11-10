# WM / BPTT 机制验证方案（0.466 收口后）

> 日期：2026-11-09
> 前提：策略侧接受 **teacher+residual env8 0.466** 作为本机最佳；不再追求 PPO 超过教师。
> 目标：验证 SuperMimic 的两个核心机制——**可微世界模型**与**短视野 BPTT actor-critic**——在 trimmed 数据上是否按预期工作，并给出可复现的定量证据。

## 0. 非目标

- 不追求超过 0.466 的策略。
- 不跑 100M/4096 env 的论文级训练；只做**机制验证 + 消融**。
- 不混用 6 项/8 项奖励；机制实验统一用 **env8** 评测、**WM 训练用 env8 奖励标签**（或明确标注 6 项）。

## 1. 假设

| ID | 假设 | 判定 |
|---|---|---|
| H1 | WM 在**教师分布**上 8 步开环 NRMSE ≤0.05，且远好于 persistence | 通过/失败 |
| H2 | **73 维状态**（root+joint）至少不差于 283 维（body 辅助不必要） | ΔNRMSE ≤0 |
| H3 | BPTT 能从 BC/教师初始化**提升**策略（env8 +≥0.02），且随训练单调 | 通过/失败 |
| H4 | H=32 比 H=16 的 8 步误差/策略收益更好 | 通过/失败 |
| H5 | 解析 6 项奖励与 env8 奖励训练出的策略在 env8 上差异显著 | 通过/失败 |

## 2. 实验矩阵

| ID | 内容 | 数据 | 预算 |
|---|---|---|---|
| E1 | WM 精度：teacher 分布，73 vs 283 维，1/8/32 步 NRMSE vs persistence | teacher 轨迹 80k（trimmed，噪声 0.1） | 2×1000 iters，~20 min |
| E2 | WM 分布对比：PD-only vs teacher（同一 73 维） | PD 80k vs teacher 80k | ~15 min |
| E3 | BPTT：从 BC 策略（teacher 蒸馏）初始化，H=16，200 iters | trimmed，teacher/BC 数据 | ~30 min |
| E4 | BPTT horizon 消融：H=8/16/32（各 100 iters） | 同上 | ~45 min |
| E5 | 奖励消融：6 项 vs env8（BPTT） | 同上 | ~30 min |

## 3. 协议

### E1/E2（WM 精度）
- 数据：CPU env、trimmed 5 clips、teacher 策略 + action noise 0.1；80k transitions；按 episode 80/20 分 train/test。
- 状态：`s`（283）或前 73；`s_next`；动作 29；xi 34。
- 模型：residual WM，1024³；k=8；batch 512；lr 1e-3；1000 iters；loss 在目标状态的全部维度（73）或前 73（283）。
- 指标：
  - 1/8/32 步 **NRMSE = sqrt(mean((pred-true)^2) / mean((s_t - s0)^2))**（persistence 归一化）。
  - 每步 RMSE 曲线；NaN/发散计数。
- 验收：8 步 NRMSE ≤0.05（H1）；73 ≤ 283 + 0.005（H2）。

### E3/E4/E5（BPTT）
- 策略：`smini.policy.Policy`（93/730）从 BC 初始化（teacher 蒸馏的 BC 权重若可用；否则从零）。
- WM：E1 训练好的 73/283 维；冻结 WM；H 步想象；critic；GAE；BPTT。
- 评测：CPU env8、trimmed、5×500；每 25 iters。
- 验收：env8 从初始值提升 ≥0.02 且至少 2 个连续 eval 点上升；否则记录负结果。
- 消融：H=8/16/32；6 项 vs env8；73 vs 283。

## 4. 交付

- `docs/WM-BPTT-VALIDATION.md`：表格（E1–E5）、曲线、结论、失败原因。
- 脚本：`scripts/wm_teacher_bench.py`、`scripts/bptt_mechanism.py`、配置。
- 数据：`runs/wm_teacher/`、`runs/bptt/`；固定 seed。

## 5. 时间盒

- E1+E2：~40 min；E3：~30 min；E4/E5：~75 min。
- 总 ~2.5 h；每步先跑短程 smoke，再全量。

## 6. 停止条件

- E1 8 步 NRMSE >0.1 或出现 NaN：先修 WM/数据，不进入 BPTT。
- E3 连续 3 个 eval 点不升：停止，记录“本机 BPTT 机制未复现”。
- 任何实验 NaN/发散：回滚该配置，不叠加。
