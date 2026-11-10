# BPTT 稳定性改进计划（trust-region / WM 快照 / raw 评测）

> 日期：2026-11-10
> 背景：最小公平测试（1024 env、2M transitions）raw 0.0565→0.0555→0.0629→**0.0229 崩溃**；EMA 掩盖。8 env/0.1M 也退化。规模不是唯一问题，**优化稳定性**才是。

## 1. 诊断（证据）

- raw 在 ~300 轮（1.5M）短暂升到 0.0629，400 轮崩溃到 0.0229；EMA 一直 0.058。
- 无 trust-region：策略可在几步内偏离 replay/WM 的分布；WM 误差被利用。
- WM 在线更新（或冻结 teacher-WM）都崩：模型误差 + 长视野 credit assignment。
- critic 原始 283 维（含未约束 body）；73 维也崩（不是唯一原因）。
- 评测只看 EMA → 崩溃被掩盖。

## 2. 干预（按顺序叠加，每步可回滚）

| ID | 干预 | 参数（初值） | 目的 |
|---|---|---|---|
| S1 | **动作 KL/trust-region**：对慢 EMA 参考策略 π_ref 加 L2 | `kl_coef=0.5`，`ref_ema=0.999` | 限制单步策略漂移 |
| S2 | **WM 快照 + 周期刷新**：冻结 WM，每 100 iter 用新数据刷新一次 | `wm_refresh_every=100` | 防止模型被 exploitation/漂移 |
| S3 | **raw + EMA 双评** + 早停 | raw < 0.7×init 连续 2 次 → 停 | 不让 EMA 掩盖崩溃 |
| S4 | **降低 policy lr** | 2e-4 → **1e-4** | 减小更新步长 |
| S5 | **critic 目标网络/Polyak**（可选，若 S1–S4 仍崩） | `tau=0.99` | 降 value 噪声 |
| S6 | **动作范围限制**：`|a - a_ref| ≤ 3` | 软/硬夹紧 | 防极端动作 |

## 3. 参数自洽性 review

- 1024 env ×5 steps = 5120 transitions/iter；5M ≈ 977 iters；buffer 1M（约 195 iter 数据）足够覆盖。
- H=32、batch 1024；kl_coef 0.5：若 Δa ~1，惩罚 ~0.5，与 return（~0.1–1）同量级，能约束但不冻结。
- ref_ema 0.999：参考策略时间常数 ~1000 次更新；比单次策略步慢，形成 trust region。
- WM 每 100 iter 刷新：~512k 新 transitions，足以更新 1024³ WM；刷新后策略仍受 KL 约束。
- lr 1e-4、1 actor/iter：5M 内约 977 次 actor 更新；比之前 200 次多，但步长减半。
- 评测 raw+EMA、每 50 iter；早停阈值 0.7×init（init≈0.057→0.040），避免把噪声当崩溃。

## 4. 验收

- 主指标：**raw env8**（trimmed 5×500）。
- 通过：5M 内 raw **不崩溃**（≥0.8×init）且至少一次 ≥init+0.02；
- 失败：raw 仍崩溃/不升 → 本机 BPTT 稳定性负结果（比“未复现”更强）。

## 5. 预算

- 1024 env，~1.1 s/iter（当前实测）；5M ≈ 20 min；含 eval ~30 min。
- 回滚：去掉 S1/S2 即回当前基线；不叠加未验证改动。

## 6. 落地顺序

1. 实现 S1/S3/S4 + 参数；短跑 300 iters（1.5M）看 raw 是否还崩；
2. 若不崩，加 S2（WM 刷新）跑 5M；
3. 若仍崩，加 S5/S6；
4. 每次只加一个变量，记录 raw/EMA/动作偏差。

## 7. 实测结果（2026-11-10）

实现：S1 action-KL + S3 raw/EMA + S4 lr 1e-4；S2 先试 frozen teacher-WM，失败后改 online WM（wm-refresh 保留）。

| 设定 | 结果 |
|---|---|
| frozen teacher-WM + KL | `ret_imag ~0.001`，WM 与策略动作分布不匹配；无效 |
| **online WM + KL 0.5 + ref_ema 0.999** | raw 0.0552@100 → 0.0545@200 → 0.0544@300；EMA 0.0546→0.0554→0.0554；**KL 0.018→0.002→0.000** |

- **稳定性目标达成**：300 iters（1.5M）raw 没有崩溃（之前 400 轮崩溃到 0.0229）。
- **但策略几乎不动**：KL 收到 0，说明 ref EMA + kl_coef 0.5 过强；raw 略低于 init 0.0573，无提升。
- 结论：trust-region 能防崩，但当前参数把学习也锁死了；需要**自适应 KL（目标 0.01）**，或 kl_coef 0.05–0.1。

## 8. 下一轮参数

- 自适应 β：`β ← β * (1 + η*(KL - 0.01))`，β∈[0.01, 1.0]；
- 或固定 `kl_coef=0.1`、`ref_ema=0.995`；
- 其余保持：1024 env、H=32、batch 1024、online WM、5M；
- 验收：raw 不崩（≥0.8×init）且 ≥init+0.02；否则负结果。

## 9. 自适应 KL 实测（2026-11-10）

β 自适应（目标 KL 0.01，rate 0.05，β∈[0.01,2]），其余同 §7。

| iter | 100 | 200 | 300 |
|---|---|---|---|
| raw | 0.0549 | 0.0544 | 0.0526 |
| EMA | 0.0546 | 0.0553 | 0.0552 |
| KL | 0.015 | 0.002 | 0.003 |

- **不再崩溃**（对比无 KL：400 轮 0.0229；固定 KL：0.0544）；稳定性目标达成。
- 但 raw **缓慢下滑**（0.0549→0.0526），仍未超 init 0.0573；KL 自适应把 β 放松后，策略仍无正向收益。
- 结论：trust-region 解决了“崩”，但没有解决“学”；在本机 1.5M/1024 env 下，BPTT 仍是稳定负结果。
- 5M 未跑完（300 轮已见下滑，继续意义低）。
- 若继续：需要更强 WM（8 步 std ≤0.05）、critic 目标网络、更大 batch/样本，而不是继续调 KL。

## 10. 最终判定

- 稳定性：✅（无崩溃、可复现）
- 学习：❌（raw 不升，略降）
- 结论：BPTT 在本机“工程可跑、稳定可控，但无策略收益”；与论文级差距仍是 WM 精度 + 样本量。

## 11. 参数/数据复查（2026-11-10，发现实现 bug）

### 已修复

| # | 问题 | 影响 | 修复 |
|---|---|---|---|
| B1 | `ema`/`ref_policy` 在 `policy.set_action_bias` **之前** deepcopy | 参考策略零 bias；KL 把策略往零动作拉；EMA eval 偏低（0.0546 vs 0.0574） | 先 set_action_bias + set_norm，再创建 ema/ref |
| B2 | Warp BPTT **未调用 `policy.set_norm` / `critic.set_norm`** | 策略/critic 直接看原始 goal/obs/state，尺度不匹配 | warmup 后 `compute_stats` 并 set_norm（与 `smini.train` 一致） |

修复后 100 iters（1024 env）：raw **0.0573**、EMA **0.0573/0.0574**、KL **0.000**——数值回正，但 **KL 0.5 把策略完全锁死**（100 轮 raw 恰好等于 init）。

### 仍存在的数据问题（不影响 BPTT 主链路，但需记录）

| # | 问题 | 影响 |
|---|---|---|
| D1 | `env.step` 先 auto-reset，再 `flush(i)`；`motion_id`/`frame_idx` 取自 **reset 后** | 完成 episode 的 metadata 错（state/obs/action/goal 本身正确） |
| D2 | `smooth = (a - self.prev)^2`，但 `self.prev=a` 在 step 前赋值 | buffer reward 的 smooth 项恒 0（BPTT 用 analytic reward，不读 buffer reward） |
| D3 | WM 归一化来自 teacher 数据，未用 buffer stats | 首轮 wm loss ~90；online 更新可适配，但前期误差大 |

### 参数表（本轮实际）

- 1024 env ×5 steps = **5120 transitions/iter**；H=32；batch 1024；WM 1024³，4×512，k=8；buffer 1M；lr 1e-4；KL 0.5（自适应目标 0.01）；ref_ema 0.999；eval 5×500。
- 200–300 iters = 1.0–1.5M transitions；5M 未完成。

### 结论修正

- 之前 `runs/bptt_stab*`（固定 KL/自适应）的 raw/EMA 受 **B1/B2** 污染，**不能作为“KL 稳定但无收益”的干净证据**；需要修复后重跑。
- 修复后当前状态：**策略被 KL 0.5 锁死（KL=0，raw=init）**；下一步应把 `kl_coef` 降到 0.01–0.05 / ref_ema 0.995，或自适应改为“KL 低时增 β 不、减 β”正确方向，才能测试“稳定且能学”。

## 12. 修复后诊断（2026-11-10）

修复 B1（ema/ref bias）与 B2（policy/critic norm）后：

| 设定 | raw | EMA | KL | 结论 |
|---|---|---|---|---|
| kl=0.5，lr=1e-4，100 iters | 0.0573 | 0.0573/0.0574 | 0.000 | **完全冻结** |
| kl=0.02，lr=1e-4，100 iters | 0.0573 | 0.0573 | 0.000 | **仍冻结** |
| **kl=0，lr=1e-3，25 iters** | **0.0030** | 0.0573 | 0.026 | **梯度存在但破坏性，立即崩溃** |

结论：
- 策略梯度**存在**（lr1e-3 时 25 轮 KL 0.026、raw 崩到 0.003），但方向/幅度有害；
- KL 0.02/0.5 只能“冻结”，没有“能学且不崩”的中间区；
- 修复前看到的 raw 变化（0.0546 等）主要来自 **zero-bias ref 的 KL 拉力**，不是有效 BPTT。
- 因此本机 BPTT 是**稳定负结果**：要么冻结，要么崩溃；根因是 WM 误差/critic/credit assignment，而非 KL 系数。

## 13. 参数/数据审计结论

- 参数：1024 env×5=5120/iter；H=32；batch 1024；WM 1024³/4×512/k=8；buffer 1M；lr 1e-4/1e-3；eval 5×500。
- 数据：state 283、obs 93、goal 730、xi=0、reward 6 项；teacher WM 初始化、online 更新。
- 已修：ema/ref 顺序、policy/critic set_norm。
- 仍存：D1 episode metadata（motion_id/frame 取到 reset 后）、D2 smooth 恒 0、D3 WM 归一化用 teacher stats。
- 旧 `runs/bptt_stab*` 数据受 B1/B2 污染，不能作为干净证据。
