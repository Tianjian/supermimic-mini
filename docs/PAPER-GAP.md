# 与 SuperMimic 论文的 Gap 分析

> 日期：2026-11-09
> 论文：SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model（arXiv:2610.28480，CoRL 2026）
> 本机：RTX 2060 6 GB + i7-1165G7 4C/8T + 15 GB RAM；CPU MuJoCo / MJX / MuJoCo Warp。

## 0. 一句话结论

我们做到的是**“教师辅助的机制级复现”**：公开教师 + residual 策略 env8 **0.466**、WM 方向正确、BPTT 链路通但未复现；
论文是**“4096 env + ~100M transitions + 2048³ WM + 40 clip 全量”的系统级结果**（0.82）。
差距不在单点算法，而在**规模、数据、WM 精度、接口口径、评测协议**五条同时差 1–2 个数量级。

## 1. 量化对照

| 维度 | 论文 | 本机 | 差距/影响 |
|---|---|---|---|
| 并行环境 | 4096 | BPTT 8；Warp PPO 1024 | **512×（BPTT）/ 4×（PPO）** |
| 训练时长/样本 | 1×H100 3 h；~100M transitions | BPTT 8 env/0.1M；Warp 1024 env/2M（raw 崩溃）；PPO 0.8–3M | BPTT 仍差 50×（2M vs 100M）且不稳定 |
| WM 网络 | 2048³ | 1024³（教师 2048³） | 容量 ~8× |
| WM 状态 | 71/73 维（root+joint） | 283 维（含 210 body），loss 只 73 | body 维未约束，critic 受污染 |
| 想象视野 H | 32 | 16（BPTT）；PPO 32 | 长视野 credit assignment 弱 |
| WM batch | 10×2048 | 4×512 | **~10×** |
| policy batch | 4096 | 256（BPTT）；32k（Warp PPO） | BPTT 小 16× |
| replay | 4.096M | 0.1–0.5M | 4–40× |
| 参考数据 | LAFAN1 40 clip（2.45 h）或 BONES 129k clip（264 h） | 5 clip（0.32 h），trimmed | **8×（LAFAN1）/ 800×（BONES）** |
| 评测 | 40 clip 整条（Appendix F） | 5 clip × 500 步（10 s），无 held-out | 不可比；无泛化证据 |
| 策略分数 | **0.82**（LAFAN1，论文） | 教师 0.4476；teacher+residual **0.466**；BPTT raw **0.047** | **~0.35** |
| WM 8 步误差 | **0.007–0.011**（论文口径） | ratio 0.257 / std 0.30（口径不同） | 量级差；绝对误差仍大 |
| 观测/目标接口 | 93 obs + 730 goal（10×73） | BPTT：93+730；教师/PPO：**930+680**（anchor-relative） | 两套接口不统一 |
| 动作 setpoint | 不裁到关节限位 | 裁到 ctrlrange | authority 被削弱 |
| 评测策略 | raw policy | 历史 BPTT 用 **EMA**（已修） | 旧 BPTT 结论被 EMA 掩盖 |
| 硬件 | H100 | 2060 6 GB | 吞吐/显存差 ~10–30× |

## 2. 方法层面的 gap

### 2.1 世界模型
- 论文用 **2048³ + 71 维状态 + 速度增量/显式积分**；我们 BPTT 用 **1024³ + 283 维（loss 只 73）**，body 维自由漂移；教师 checkpoint 是 2048³，但只在 73 维上监督。
- 论文 8 步误差 0.007–0.011；我们 8 步 **std 0.30 / persistence ratio 0.257**（口径不同，但绝对误差明显更大）。
- 数据分布：论文 WM 在 4096 env 的策略 rollout 上训练；我们 BPTT 的 teacher-WM 在**教师轨迹**上训练，BPTT 从零策略的 action 分布不在其中——frozen 测试 raw 退化（0.0573→0.0469）；online WM 也崩溃（→0.014），说明模型误差/优化不稳定，不只是分布错配。

### 2.2 策略/观测
- 论文 93 维本体观测 + 730 goal；**没有 anchor 误差**（其复现 e2 因此只有 0.56；加 anchor 9 维到 0.904）。
- 我们 BPTT 用 93+730（论文口径），但只有 5 clip/1M；教师/PPO 用 **930+680**（BONES 复现口径，含 anchor-relative goal），**不是论文接口**。
- 因此 **0.466 是教师（BONES/260k iters）辅助结果，不是论文方法的复现**；BPTT（论文口径）只到 0.047–0.06。

### 2.3 训练协议
- 论文 Algorithm A.1：4096 env、UTD 0.4、H=32、wm 10×2048、policy 4096、buffer 4.096M、cosine lr、500 warmup。
- 我们：8 env、512 transitions/iter、**200 iters = 仅 0.1M transitions**、4×512 WM、policy 256、buffer 100k、lr 恒定；相对论文 **512× env / 1000× 样本 / 16× policy batch / 10× WM batch**。
- 评测用 EMA（现已改为 raw+EMA 双报），旧结论被掩盖。

### 2.4 数据
- 论文 LAFAN1 40 clip / 2.45 h；我们 5 clip / 0.32 h（trimmed 后更短），且没有 held-out。
- LAFAN1 片段含 3–6.5 s T-pose 引入段；我们 trim 后才用于 eval/demo，论文数据管线也会处理类似问题（未逐条核对）。
- 教师训练数据是 BONES（102k clip/198 h），与 LAFAN1 不同分布，解释了教师 0.4476 与论文 0.82 的差距。

### 2.5 评测
- 论文 Appendix F：40 clip 整条、确定性、无噪声/DR；我们 5 clip×500 步、trimmed、无 held-out。
- 我们的 0.466 与论文 0.82 **不可直接比较**；同一 5 clip 上论文方法的分数未知。
- 早期 soup 0.1046 是 5 clip in-sample + 权重在 eval 上挑选，不能作为论文级证据。

## 3. 为什么 BPTT 没复现（证据链）

| 实验 | 设定 | raw env8 | 结论 |
|---|---|---|---|
| E3-a | frozen teacher-WM，283 critic，H=8 | 0.0573→0.0469 | 退化 |
| E3-b | frozen，73 critic | 0.0573→0.0420 | 更差；body 维不是原因 |
| E3-c | online WM，73 critic | 0.0575→0.0140 | 崩溃 |
| 旧 E3 | online WM + **EMA eval** | 0.0573→0.0605（EMA） | 假象，已撤回 |

- WM 绝对误差（8 步 std 0.30）+ 1M transitions + H=16 → 策略梯度被模型误差主导；
- 旧“提升”来自 **online WM 适配 + EMA 滞后**，不是 BPTT；
- 这不是“算法一定错”，而是**本机预算/接口/评测下的负结果**。

## 4. 我们实际复现了什么

- ✅ 数据管线（LAFAN1→G1 npz、FK、T-pose 裁剪）、93/283/730 维度、MuJoCo env、8 项奖励。
- ✅ 可微 WM（residual，283/73）+ 1/8/32 步评测：**8 步优于 persistence 3–4×**；283 > 73。
- ✅ BPTT actor-critic 链路（H=8/16/32）能跑、eval 可复现；但 raw 不提升。
- ✅ 公开教师 0.4476 + residual 0.466；Warp 1024 env 24.7k 控制步/s。
- ❌ 论文级 0.82 / WM 0.007–0.011 / 4096 env / 100M transitions 未达到。

## 5. 若要真正缩小 gap（按优先级）

1. **统一到论文接口**：93 obs + 730 goal（或明确采用 anchor 9 维并声明偏离），不要混用教师 680/930。
2. **数据到 40 LAFAN1 clip**（2.45 h）+ held-out；trim T-pose。
3. **WM 升级**：71 维状态、2048³、H=32、10×2048、body 用可微 FK；目标 8 步 std ≤0.05。
4. **规模**：Warp/mjlab 4096 env、100M transitions（2060 可能需 1024 env + 数小时；内存/显存需压）。
5. **评测**：整条 40 clip、raw policy、held-out；报告 WM 误差与策略分数。
6. **BPTT 稳定化**：先 BC/教师热启动（落在 WM 分布内），再 trust-region；否则从零 BPTT 在本机会持续退化。

## 6. 诚实定位

- 本机成果是**机制验证 + 教师辅助策略（0.466）**，不是论文级复现。
- 论文的核心（2048³ WM + 4096 env + 100M + 40 clip）在本机硬件上不可完整复刻；我们给出的是**可复现的缩小版 + 明确的负结果**。
- 0.466 是当前最佳；BPTT/论文口径为负结果；WM 方向正确但绝对精度不足。
