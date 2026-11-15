# BPTT 最小公平测试设计

> 日期：2026-11-09
> 目的：在“不再从零造轮子”的前提下，把 BPTT 的并行度/样本量提到可与论文做**方向性对照**的最小规模，验证“BPTT 是否能在 raw 策略上产生单调提升”。

## 1. 定位

- 这是“**最小公平测试**”，不是论文复刻：
  - 并行：1024 vs 论文 4096（1/4）；
  - 样本：5M–10M vs 论文 ~100M（1/20–1/10）；
  - WM：1024³ vs 2048³（1/8 容量）；
  - 数据：5 trimmed clips vs 40 clips（后续再扩）。
- 只有在这个规模下 raw 仍不升，才能把“BPTT 负结果”从“欠采样”升级为“方向性负结果”。

## 2. 关键参数

| 项 | 本测试 | 论文 | 说明 |
|---|---|---|---|
| env | **1024**（Warp） | 4096 | Warp 实测 24.7k ctrl-steps/s |
| 每 iter 采集 | 5 steps ×1024 = **5120** transitions | 5×4096 | UTD 0.4 |
| 总样本 | **5M（977 iters）/ 10M（1950）** | ~100M | 1/20–1/10 |
| H | **32** | 32 | |
| policy batch | **1024** | 4096 | 1/4 |
| WM | **1024³**，k=8，4×512/iter | 2048³，10×2048 | 1/8 容量 |
| buffer | **1M** | 4.096M | ring |
| lr | 2e-4（WM 1e-3） | 2e-4 | |
| 观测/目标 | **93 / 730（论文口径）** | 93 / 730 | 不用教师 680/930 |
| 奖励（训练） | 解析 6 项（BPTT） | 同 | 评测 env8 |
| eval | **raw + EMA**，每 100 iters，CPU 5×500 | raw | 避免 EMA 掩盖 |
| 数据 | trimmed 5 clips | 40 clips | 先 5，后续 40 |

## 3. 实现

- 新脚本 `scripts/bptt_warp_train.py`：
  - Warp 1024 env（与 P3 相同，修过 body linvel/GAE）；
  - 采集 **283 维 state**（root+joint+14 body，body lin 用有限差分、ang 用 cvel 角速度）；
  - 每 env 维护 episode 列表，done 时 `ReplayBuffer.add_episode`（容量 1M）；
  - 复用 `smini.train.train_wm` / `train_policy` / `evaluate`（不重写 BPTT）；
  - `Policy(obs93)`、`Critic(283)`、`WorldModel(283, 1024³)`；
  - goal 730 = 10×`spec.pack_goal_frame`。
- 先跑 **20 iters smoke** 验证采集/训练/评测；再跑 5M（977 iters，预计 50–90 min）。

## 4. 验收

- 主指标：**raw policy env8**（trimmed 5×500）。
- 通过：raw 从 init 提升 **≥0.02**，且至少 3 个连续 eval 点上升；
- 失败：raw 不升或下降 → 在 5M/1024 env 规模下 BPTT 方向性负结果；
- 不做：EMA-only 结论；不混用 680/930 教师接口。

## 5. 预算与风险

- 采集 5120/iter，Warp ~0.2 s；WM 4×512 + actor H=32/batch 1024；预计 **2–4 s/iter**，5M ~50–90 min。
- 风险：Warp->torch 拷贝/状态构建开销、buffer 1M 内存、policy batch 1024 显存（6 GB）。
- 回滚：若 1024 env 内存不足，降 512 env，样本不变（iter 数翻倍）。

## 6. 实测结果（2026-11-09）

配置：1024 Warp env、5 steps/iter、H=32、batch 1024、WM 1024³（4×512）、buffer 1M、lr 2e-4；raw+EMA 双评。

| iter | 100 | 200 | 300 | 400 |
|---|---|---|---|---|
| **raw** | 0.0565 | 0.0555 | **0.0629** | **0.0229（崩溃）** |
| EMA | 0.0545 | 0.0562 | 0.0573 | 0.0584 |

- 300 轮（~1.5M transitions）raw 短暂升到 **0.0629**（init ~0.0573）；
- 400 轮（~2M）raw **崩溃到 0.0229**，EMA 仍 0.0584；
- 结论：**规模带来短暂提升，但 BPTT 优化不稳定**；EMA 会掩盖崩溃。
- 原因候选：WM 误差被利用、长视野 credit assignment、缺 trust-region/KL、critic 价值。
- 未完成 5M（967 轮）：400 轮已崩溃，继续无意义；5M 目标未达成。
- 仍不能宣称论文级 BPTT；但已排除“仅 8 env/0.1M 导致”的解释。
