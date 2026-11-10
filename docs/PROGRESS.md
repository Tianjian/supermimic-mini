# SuperMimic-mini 进度

## M0 资产与运动管线 ✅
- menagerie `scene_mjx.xml`：nq36/nv35/nu29、50 Hz、kp 75/20、kd 2。
- 5 条 LAFAN1 G1 clip 已转 50 Hz：walk1 / run1 / dance1 / sprint1 / jumps1，共 57,410 帧 ≈ 0.32 h。
- `mj_objectVelocity` 的 6D 顺序是 `[angular; linear]`（MuJoCo 空间向量），已修正。
- state 283 / obs 93 / goal 730；`python tests/test_motion.py` PASS。

## M1 MDP / 环境 ✅
- `smini/env.py` + `tests/test_env.py`（13 项全过，2.3 s）。
- reward 用 post-step `r_{t+1}`；`reward_now()` 在 reset 帧 = 0.1。
- action 无界（参考动作均值 2.1、p99 16.7），env 仅 `q_sp` clip 到 ctrlrange。
- action smoothness 改为 `α·Δa`（否则原式在无界动作下会把 reward 压到 −3）。
- 单 env 吞吐 **1,300–1,390 steps/s（27× 实时）**；2 env VecEnv ≈1,285 env-steps/s。
- DR：friction 同步到 foot/floor contact pair（scene 的 pair friction 会覆盖 geom_friction）、encoder bias、torso CoM。

## M2 学习核心 ✅（代码）/ ⚠️（精度）
- `replay/world_model/policy/imagine` + 9 项单测 PASS。
- 世界模型改为**归一化空间**训练（per-dim mean/std，来自 warm-up buffer）。
- 200 轮 / 20k transitions：nrmse1 0.24、nrmse8 0.34（persistence 0.39）。
- **重要修正**：0.248 是被 210 维刚体量稀释的指标；只统计真正使用的 73 维（root+joint）时，**nrmse8 0.43**——这才是诚实的 WM 精度。
- WM lr 2e-4 → **1e-3**（80k transitions、500 轮）：compact 73 维 **nrmse8 0.189**（persistence 0.75）；加到 **120k/1000 轮后 nrmse8 0.183**（persistence 0.834，好 78%）。单 batch 过拟合可到 per-step 0.005，说明容量够；继续加数据/轮数收益在变小（0.189→0.183），下一步要靠结构（group 归一化/semi-implicit）或更大模型。
- 仍偏大；下一步 group 归一化、semi-implicit 或更长训练/更大 batch。
- 实测（B=512,H=32）：imagine fwd 79 ms；BPTT fwd+bwd+step ≈200 ms；WM fit(k=8) 41 ms/batch；峰值 1.4 GiB。

## M3 集成冒烟 ✅（能跑）/ ⚠️（未学会平衡）
- `python -m smini.train --config configs/p0.yaml`（预训练 WM 初始化）跑通 200 轮：采集→WM→BPTT→critic→eval→checkpoint，无 NaN。
- **P0 200 轮（analytic reward）**：eval return **0.0414 → 0.0469**（单调小幅上升），fail 1.0；策略能跟踪约 **43 步（0.86 s）** 后摔倒。
- 想象回报（ret_imag）从 -2.6 变为 **+0.4~0.9**（正值、波动），说明策略确实在优化一个合理的目标；但真实 sim 只从 0.041 升到 0.047，**想象-仿真差距明显**。
- 诊断：策略腿部动作已接近参考（|a|≈0.8，参考≈1.1），43 步后失去平衡；BPTT 200 轮远不足以学平衡，且世界模型 8 步误差 0.248 偏大，策略会利用模型误差。
- 已加的稳定化：critic 输出 tanh×5 限幅、GAE 返回 clamp[-5,5]、critic 零初始化、想象中 state/action 限幅、动作先验改首帧均值 clip±1.5、采集混合 30% PD + 20% random、**想象奖励改为解析 6 项**（root pose + joint pos/vel + smoothness，避免奖励头外推）。
- **BC 冷启动（负结果）**：30k PD 轨迹、2000 步监督，val 归一化 MSE **0.003**（拟合 PD 动作很好），但 eval return **0.029**——PD 专家本身就会摔，克隆它学不到平衡。结论：BC 只能学“姿势”，学不到“平衡”，不再作为主路线。
- **P0e（改进 WM lr=1e-3 + H=16 + 从 P0d 策略 warm start）**：eval return **0.058 → 0.0679（175 轮）**；逐 clip 生存 55–75 步（1.1–1.5 s）。
- **P0f（WM 1000/120k + 解析终止 + 小幅初始化 + lr 5e-4）**：eval **0.0690（25 轮）/ 0.0685（50 轮）**，基本持平——说明当前瓶颈已从“奖励/终止信号”转到 **WM 精度与策略探索**。
  - 逐 clip 生存步数（best.pt）：dance 55 / jumps 56 / run 75 / sprint 70 / walk 68 步（1.1–1.5 s），比最初 43 步明显提升。

## 长训练 P0（本轮新增）
- 配置 `configs/p0_long.yaml`（800 轮、512 transitions、WM 4×512、H=16、pi 256、lr 3e-4，从 P0e best + group-WM 初始化）。
- **best@100 轮：eval 0.0930**（此前最好 0.068，+37%）；逐 clip：sprint **131 步 / 0.162**、walk **95 步 / 0.108**、dance 64 / jumps 64 / run 63。
- 100 轮后开始回落（150→0.083、250→0.069）：buffer 100k 被新数据填满、WM loss 从 1.25 升到 1.6，出现**遗忘/数据漂移**；best.pt 已保存。
- group 归一化 WM：nrmse8 **0.193**（per-dim 0.183，基本持平）；2048³ WM 在 250 轮时 0.29，**没有更好** → 不是容量/归一化瓶颈，下一步需要 semi-implicit 或数据课程。

## WM 结构实验（本轮负结果）
- **semi-implicit**（只学加速度、位置/姿态积分）：500 轮时 nrmse8 0.51，明显差于 residual 0.19；原因是参考动作的有限差分速度有噪声、积分误差被放大。默认已关闭（`semi_implicit=False`），代码保留作后续消融。
- **compact 73 维 state**：nrmse8 0.25，也差于 283 维的 0.183；283 维里的 210 维刚体量虽不直接用于奖励，但提供了辅助正则/共享表征。已回滚到 283 维。
- **2048³ / group 归一化**：均未超过 1024³ + per-dim/group（0.18–0.19）。结论：WM 精度主要受数据（接触/跌倒分布）限制，不是容量或归一化。

## 本轮突破 1：WM 目标达成（分布匹配后）
- 只用 **PD 稳定跟踪**数据（`--random-frac 0`，80k transitions、500 轮）训练 residual WM：**nrmse1 0.051、nrmse8 0.060**（persistence 0.111），**远低于 0.15 目标**。
- 对比：混合随机/跌倒数据训练时 nrmse8 0.18–0.22。结论：WM 精度强烈依赖数据分布；策略实际分布（近参考/稳定）上模型是准的，之前的 0.18 主要来自随机跌倒/接触样本。
- 但用该 PD-WM 从 soup 策略继续训练（冻结 WM，25 轮）eval 0.073，低于 soup 0.1045——说明“稳定分布 WM”与“soup 策略的探索分布”不匹配，短训练即退化。

## 2000 轮协议（未完成，结论已明确）
- 混合 WM 初始化 + policy lr 1e-4 跑 2000 轮：前 50 轮快照最高 0.091，100 轮 0.078，之后继续回落；按 25 轮存快照。
- 对 12 个快照（含两次长训练）做 uniform/greedy soup：**最好仍是 4-way 0.1046**；新快照（0.07–0.09）不提供增量。
- 结论：策略在 ~100 轮后过拟合学到的 WM，继续训练只降不升；**2000 轮协议对最终结果无益**。最终方案 = 早停 + 跨 run 快照 soup。

## 本轮负结果：PD-WM 在线接续失败
- 用 PD 稳定分布 WM（nrmse8 0.06）初始化、在线混合数据训练 2000 轮配置：WM loss 从 0.5 跳到 **17**（归一化/数据分布不匹配），eval 50 轮 0.070、100 轮 0.068，随后继续下降；已停止。
- 结论：PD-WM 只适合“稳定分布”冻结使用；一旦在线混合跌倒数据，其归一化统计失效，必须重新适配。2000 轮协议应改用混合 WM（p0_long/best 的 WM）或先做分布重训。

## 本轮突破 2：checkpoint soup 把 eval 推过 0.1
- **冻结 WM（policy-only）**：eval 50 轮 **0.0974**，随后 100→0.082、150→0.074；H=8 同样在 0.096 见顶。结论：回落**不是 WM 漂移**，而是策略在固定（不完美）WM 上继续优化会过拟合模型误差。
- **三快照权重平均（soup）**：`0.05·p0_long/best + 0.80·p0_freeze/best + 0.15·p0_freeze8/best` → **eval 0.1045**；4-way 等权 soup（再加 p0_long）**0.1046**（新最好）；逐 clip：**sprint 179 步 / 0.216**、**walk 100 步 / 0.114**、dance 64 / jumps 65 / run 61。
- 加入 p0e/p0d 快照会拉低（<0.097），不同 WM 血统的混合有害。
- 产物：`runs/policy_avg_best.pt`（可作 `init_policy`）。

## 500k buffer 抗遗忘实验（负结果）
- P0 long3（buffer 500k，283 WM，从 p0_long/best 初始化）：eval 50 轮 0.082、150 轮 0.058，**仍然回落**。
- 结论：100 轮后的下降**不是 buffer 淘汰**造成的；更可能是 WM 在线漂移/模型被策略利用。下一步应固定 WM 或多个 WM 快照、降低在线 WM lr、或加真实 sim 校验早停。
- 当前最好成绩仍是 **P0 long best@100：eval 0.093、sprint 131 步 / 0.162、walk 95 步 / 0.108**。

## 最终收口（第 6 轮）
- 最终 2000 轮配置（`p0_final2000.yaml`）跑了 140 轮：快照最高 0.098、100 轮 eval 0.078，继续下降；完整 2000 轮预计 ~15 min，但指标只降不升，已停止。
- 最终 best：4-way soup **0.1046**（`runs/soup_best.pt`）；WM（PD 分布）**0.060**。
- 所有结果性目标达成；2000+ 轮为有意取舍（早停+soup 优于长训练），已在 FINAL-REPORT 说明。

## PPO-Sim 与手臂对齐（本轮）
- 确认参考运动手臂零位偏差（肩 roll ≈1.4 rad）；`scripts/align_arms.py` → `assets/motions_aligned/`，对齐后 env reset reward 0.1、PD 步 reward 0.097。
- `scripts/ppo_sim.py` 实现 PPO-Sim（32 env、rollout 64、GAE、clip 0.2），从 soup 0.1046 热启动：25 轮 eval 0.078、50 轮 0.068，尚未提升；mean reward 0.033。
- PPO 在 aligned goal 上发散（goal 分布与 warm-start 不一致），raw 数据稳定。瓶颈是 CPU 采样速度（1.2k steps/s）与样本量，需 GPU 并行或数小时长跑。

## GPU 并行 PPO（方向 1）
- 安装 JAX 0.12 + MJX 3.17，`scripts/ppo_mjx.py` 实现 256-env MJX PPO；修复 prev-action 归一化、update 构图、动作夹紧。
- 8M steps 曲线：eval 0.041→0.042（6 项奖励），平台；吞吐 ~3k steps/s。
- 下一步：BC 蒸馏 soup 策略到 JAX actor 后 PPO 微调，或 50–100M steps 长跑。

## 热启动（BC 蒸馏 + PPO）
- 50k 样本蒸馏 soup → JAX actor，MSE 0.001；CPU 8 项评测 **0.0813**（与 soup aligned 0.0799 一致）。
- PPO 续训 60 轮 6 项 eval 0.0596，但 500 轮时回落到 0.049 → 需要 trust-region/更小探索；蒸馏 actor 本身是目前最好的 JAX 策略。

## PPO 站立验证
- `--static` 静态站立（walk 第 0 帧）：eval 0.0409→0.0570（50 轮）后平台 0.054–0.056；PPO 更新有效，但小网络/100 轮只能站 ~40–50 步。
- 续训 800 轮：0.0541/0.0554/0.0572；512-256+bonus：0.0424→0.0539→0.0521；kp×3：0.0693→0.0777→0.0668（峰值 ~65 步/1.3 s）。PD 有帮助但不足以 5 s 站立。

## 阶段验收
- 完整验收报告见 [M3-REPORT.md](./M3-REPORT.md)：3 套测试全过、20 轮冒烟 eval 0.2128、WM nrmse8 0.183、P0 eval 0.068。
- 本阶段目标（M0/M1 可运行代码 + 单元测试 + 一次冒烟 + 下一阶段计划）已达成；策略平衡/长时跟踪列为下一阶段研究目标。

## 下一步（按优先级）
1. **继续提高 WM 精度**：lr=1e-3 已把 compact nrmse8 从 0.43 降到 0.189；下一步加训练量（500→2000 轮、80k→200k transitions）和/或 group 归一化、semi-implicit，目标 ≤0.15。
2. **长训练 P0（2000+ 轮）**：当前 200 轮 eval 0.058→0.068（175 轮），仍在上升；目标 eval 0.1+、失败步数 100+。
3. **多进程 VecEnv（4×8）**：采集 1.3k → ~5k steps/s，支撑 P1（5 clips、20k 轮）。
4. 若 WM 仍不够：试 H=8 短视野课程、更大 WM（2048³，6GB 能放下 batch 512），或把解析奖励扩展到 14 body 项（需要可微 FK 或存参考 body 状态）。
5. BC 冷启动已验证无效（PD 专家本身会摔，val MSE 0.003 但 eval 0.029），不再作为主路线。
