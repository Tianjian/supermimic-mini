# SuperMimic-mini 实验设定审计

> 日期：2026-11-09
> 范围：`smini/`（env/spec/imagine/train/world_model/policy）、`scripts/ppo_mjx.py`、`scripts/bc_jax.py`、`scripts/eval_jax_cpu.py`、`scripts/soup.py`、`configs/*.yaml`、`docs/M3-REPORT.md`。
> 方法：静态阅读 + 数值核对（模型执行器/关键帧/时间步）+ 与论文/复现对照。所有结论以代码和实测为准。

## 0. 结论先行

1. **代码层面的维度/常量基本自洽**：`ACTION_SCALE` 与模型 `0.25·|forcerange|/kp` 的最大误差 **3.3e-7**；`DEFAULT_Q` 与 `knees_bent` 关键帧完全一致；env 把 XML 的 `timestep=0.004` 覆盖为 `0.005`，控制频率确为 50 Hz。
2. **结果不可比的根因是“训练/评测奖励不一致”**：CPU env 是 **8 项 body 奖励（max 0.1/step）**；BPTT 是 **6 项 joint+root 奖励（max 0.06/step）**；PPO/MJX 又是另一个 **6 项 joint+root（max 0.06/step）**。三套数字被放在同一张表里比较，结论会失真。
3. **算力/数据差 2–3 个数量级**：论文/复现是 **4096 env、~100M transitions、H=32、wm 2048³**；本机是 **8 env、~1M transitions、H=16、wm 1024³**、5 条 clip / 0.32 h。站立/跟踪上不去首先是这个差距，而不是“再调一个 reward”。
4. **若要在本机突破**：优先修审计项（奖励统一、手臂对齐、动作裁剪、kp 混淆、held-out），再把仿真换成 **mjlab/MuJoCo Warp（4096 env）** 或云端；继续在 8 env + MJX 256 env 上加轮数收益很低。

---

## 1. Critical（会直接导致结论不可信）

### C1. 训练/评测奖励不一致（最严重）
- **证据**
  - `smini/env.py::_reward`：body pos/ori/lin/ang（14 bodies）+ root pos/ori + smooth + joint-limit，`×DT`，**max = 5.0×0.02 = 0.1/step**。
  - `smini/imagine.py::analytic_reward`：root pos/ori + **joint pos/vel（只算腿+腰，`JOINT_REWARD_MASK` 把 14 个手臂关节置 0）** + smooth，`×0.02`，**max = (0.5+0.5+1+1)×0.02 = 0.06/step**。
  - `scripts/ppo_mjx.py::reward_fn`：与 BPTT 同构的 6 项（腿+腰 joint），max 0.06。
  - `smini/train.py::evaluate` 用 `ret / (0.1*(T-1))`，只对 **8 项奖励** 是“归一化到 1”；对 6 项奖励，满分只有 **0.6**。
- **影响**
  - 报告里的 `soup eval 0.1046`（8 项）与 `PPO 6 项 0.05–0.08`、`BC 8 项 0.0813` 不是同一把尺子；`0.057`（6 项）被误读为“远低于 0.1046”，实际占各自满分的比例接近（~10%）。
  - BPTT 策略优化的是“腿+腰 joint + root”，评测却按“14 body + root”打分；策略没有优化手臂/腕 body，评测自然低。
- **修复**
  - 统一为一套奖励：要么让 env 也用 6 项（腿+腰 joint）评测/训练，要么在 BPTT 里补可微 body FK（与 env 对齐）。
  - 报告任何数字时标注 `reward=env8 / analytic6` 与分母（`0.1*T` vs `0.06*T`）。
  - 目标阈值（0.1+）只对 env8 有效；6 项应换成“占满分比例”或 0.06 基数。

### C2. ~~手臂 retarget 零位偏差~~（原判断错误，已更正）
- **原判断**：raw 参考手臂外展是恒定零位偏差，`align_arms.py` 用第 0 帧偏移对齐。
- **实测更正**：raw 的肩 roll 随时间自然变化（walk：f0 +1.67 → f150 +0.05；sprint：f0 +1.65 → f150 +0.17），第 0 帧只是 T-pose/起始段，**不是恒定零位**。`align_arms.py` 减常量后，sprint f150 变成 **-1.28/+1.26（反向外展）**，`assets/motions_aligned/` 是**坏数据**，已弃用。
- **影响**：套用 aligned 的 demo/BC/PPO 结果被污染；raw 上的 soup 0.1046、教师 0.4442 是有效口径。之前“0.1046 被手臂 bug 压低”的结论作废。
- **修复**：一律使用 `assets/motions`（raw）；删除/不再生成 aligned；如需手臂权重，按关节/时间自适应处理，而不是减常量。

### C3. soup 权重在评测集上挑选（选择偏差）
- **证据**：`scripts/soup.py` 对 `runs/soup_pool` 的每个 checkpoint 在 **同一 5 条 eval clip** 上评分，再 greedy/加权；0.1046 是“在评测集上挑出来的最好组合”。
- **影响**：0.1046 是乐观估计；单模型 best 是 0.093（`p0_long/best`），soup 的 +12% 至少部分是选择偏差。
- **修复**：留出 held-out clip（如 walk/run 训练、sprint/dance 评测），或 nested split；soup 权重只在 held-out 上选。

### C4. 动作 setpoint 被裁剪到关节 ctrlrange（与论文相反）
- **证据**：`smini/env.py:309` `np.clip(q_sp, ctrl_min, ctrl_max)`；`scripts/ppo_mjx.py` 同样 `clip(Q0+ASCALE*a, cmin, cmax)`。
- **论文**（BeyondMimic III-C）：`q = q̄ + α a`，**“intentionally not clipped by joint kinematic limits”**；setpoint 是生成力矩的中间量，不是位置目标。
- **影响**：在极限姿态/跌倒恢复/高动态动作上，策略的力矩被提前截断，authority 下降；站立/跟踪都会受影响。
- **修复**：去掉 `q_sp` 的 ctrlrange clip（保留 action 的 ±50 安全裁剪）；重跑 eval/训练对比。

### C5. PPO 站立/跟踪数字与 8 项目标的混用
- **证据**：`scripts/ppo_mjx.py` 的 eval 用 `reward_fn`（6 项）却除以 `0.1*499`；`stand` 0.057 实际是 0.057/0.6 ≈ 9.5% 满分，不是 5.7%。
- **影响**：把 6 项 0.057 与 8 项 0.1046 并排，结论“站立远差于跟踪”不成立。
- **修复**：PPO eval 分母改为 `0.06*T`（或输出原始平均奖励 + 满分比例）；报告里同时给“绝对奖励/满分比例”。

### C6. 观测缺少 anchor 相对位姿误差（结构性问题）
- **证据**：本机 `spec.OBS_DIM=93`：`q_rel(29)+qd(29)+base_angvel(3)+projected_gravity(3)+prev_action(29)`；**没有 root 相对参考的位置/朝向误差**。goal 里有未来 10 帧的绝对 root 位姿，但策略看不到自己的绝对 root 位置，无法计算“我现在偏了多远/多歪”。
- **外部对照**：SuperMimic-Reproduced 的 e2（论文 93 维观测）在 LAFAN1 40 clip 上封顶 **0.56**；e3 加 BeyondMimic 式 **anchor-relative 观测（9 维：root 位置误差 3 + 朝向误差 6D）** 后达到 **0.904**（1.66 h / 5090）。BeyondMimic 的观测也明确包含 anchor pose error ξ（9 维）。
- **影响**：全局 root 漂移对策略不可观测，位置/朝向奖励拿不到 → 跟踪上限被结构性压低；这不是“训练不够”，是 MDP 缺信息。
- **修复**：obs 追加 9 维（`p_ref - p` 3 + `R_ref^T R` 前两列 6），OBS_DIM 93→102；训练/评测/BC/PPO 同步；优先做。

## 2. High（显著影响可信度/学习效果）

### H1. critic 用 283 维 state，其中 210 维 body 未被 WM loss 约束
- **证据**：`smini/world_model.py`/`pretrain_wm.py` 的 loss 只在前 73 维；`smini/policy.py::Critic` 输入 `STATE_DIM=283`；`imagine.py` 的 `s` 含 14×15 body 维。
- **影响**：WM 对 body 维只“顺带预测”，误差不受惩罚；critic 读到漂移的 body 维 → 价值噪声 → 策略更新方向变差。
- **修复**：critic 只用 73 维（root+joint）；或给 body 维加辅助 loss（权重 0.1–0.5）；或把 body 维从 state 里去掉（283→73）。

### H2. 没有 held-out clip，eval 全是训练 clip
- **证据**：`TrainCfg.eval_clips=5`，训练库就是这 5 条；`evaluate` 从 frame 0 固定起点。
- **影响**：0.1046/0.0813 都是 in-sample；不能说明泛化；且固定 frame 0 比训练随机起点更容易。
- **修复**：留出 1–2 条 clip（如 sprint/run），训练只用其余；或至少随机 eval 起点多次取平均。

### H3. kp×3 实验被 ACTION_SCALE 混淆
- **证据**：`scripts/ppo_mjx.py --kp-scale` 只改 `actuator_gainprm/biasprm`，**没有把 `ASCALE` 除以 3**；`q_sp = Q0 + ASCALE*a` 的等效偏移变成 1/3。
- **影响**：kp3 的 0.0777 不能解释为“更硬的 PD 更容易平衡”；它同时缩小了动作空间。
- **修复**：kp 改变时同步 `ASCALE ← 0.25·|forcerange|/kp`（或直接按模型重算），再做干净对比。

### H7. WM state 用 283 维，而复现用 71 维
- **证据**：本机 `spec.STATE_DIM=283`（含 14×15 body 维），WM loss 只前 73；SuperMimic-Reproduced 的世界模型是 **71 维状态 + 速度增量 + 显式积分**，8 步开环归一化误差 **0.0065–0.0099**（论文 0.007–0.011）。
- **影响**：210 个 body 维既不被 loss 约束，又进入 critic（`Critic(state_dim=283)`），是价值噪声源；同时浪费模型容量。
- **修复**：WM/critic 改用 71–73 维（root+joint，去掉 body 维）；body 奖励用可微 FK 从 71 维算（复现的做法），不要用未约束的 body 预测。

### H8. WM 精度与复现差 6–9×（口径需注意）
- **证据**：本机 PD 分布 nrmse8 **0.060**；复现 8 步归一化误差 **0.0065–0.0099**。两者定义可能不同（NRMSE vs 归一化 MSE），但量级差距明显。
- **影响**：WM 误差直接进入 BPTT 梯度；策略会在 100 轮后利用模型误差（与观察一致）。
- **修复**：统一误差口径（同一测试集、同一归一化）；用 71 维状态 + H=32 + 2048³ 重训 WM；先把 8 步误差打到 ≤0.02 再谈长训练。

### H4. 与论文/复现的规模差距（不是 bug，但决定结论）
- **证据**（论文/复现 vs 本机）
  - SuperMimic 论文：4096 env、H=32、wm 2048³、wm_batches 10、wm_batch 2048、imagination batch 4096、buffer 4.096M、~100M transitions、单 H100 3 h。
  - SuperMimic_Dyn（E009）：8192 env、40k iters、**327.68M transitions**、2 GPU 4.09 h、95k clips/205 h。
  - supermimic_getup：单 4090、**180k iters**、102k clips/198 h、2000-clip eval 0.921。
  - 本机：8 env、H=16、wm 1024³、wm_batches 4/512、pi_batch 256、buffer 0.1–0.5M、~1M transitions、5 clips/0.32 h。
- **影响**：样本量差 **~100–300×**，数据差 **~600–1000×**；期望本机 100–200 轮达到论文效果不现实。
- **修复**：mjlab/MuJoCo Warp（4096 env）或云端；本机可先 1024 env 做对照，但不要用 8 env 的结论否定方法。

### H5. 采样是均匀随机帧，不是 failure-adaptive
- **证据**：`env.reset`/`sample_starts` 均匀采样 `t0`；论文 III-F 用 1 s bin + 失败率 EMA + 均匀混合。
- **影响**：难段（起跳、快速步态）被欠采样；训练 reward 方差大、收敛慢。
- **修复**：按失败率分 bin 采样；先跑 baseline 记录每 bin 失败率，再按 `p_s` 采样（与均匀混合 λ≈0.2–0.3）。

### H6. 报告混用不同评测协议
- **证据**：`docs/M3-REPORT.md` 写 “eval 0.2128（1 clip/100 步）”，后文 headline 是 0.1046（5 clips/500 步）。
- **影响**：0.2128 与 0.1046 不可比；读者会误以为性能大幅下降。
- **修复**：统一为 5 clips × 500 steps（或 full clip），标注起点/步数/奖励口径；历史数字加协议标签。

### C7. LAFAN1 片段有 3–6.5 s 的 T-pose/站立引入段
- **证据**（根速度 10 帧均值 >0.5 m/s 的首帧）：dance 237、jumps 168、run 326、sprint 195、walk 211；前 150 帧根速 ~0.01、手臂外展，是 T-pose/站立。
- **影响**：eval 从第 0 帧开始、demo 只截 3 s，前 3–6 s 几乎全在 T-pose 里；reference 看起来“怪”，teacher/策略的分数也被引入段稀释。这不是参考动作本身错，而是**数据预处理缺失**。
- **修复**：`scripts/trim_motions.py` 按根速度裁剪引入段（保留 25 帧 lead），生成 `assets/motions_trim/`；eval/demo/BC 用 trimmed。teacher trimmed 0.4476（raw 0.4442）。

## 3. Medium（需记录/清理，不改变核心结论）

- **M1** `ACTION_SCALE` 硬编码但**与模型一致**（max diff 3.3e-7）；`DEFAULT_Q` 与 `knees_bent` 一致（diff 0）。不是 bug，但应在 README 注明“来自 menagerie 当前版本，换模型需重算”。
- **M2** XML `timestep=0.004`，`TrackingEnv` 覆盖为 `0.005`（4×0.005=0.02 s，50 Hz），`ppo_mjx` 也设 0.005；一致。审计直接 `MjModel.from_xml_path` 会看到 0.004，容易误判。
- **M3** 配置里大量绝对路径（`/home/toby/...`），不可移植；仓库有 20+ 个历史 config，最终 recipe 不唯一。
- **M4** `soup.py` 对 pool 里所有 checkpoint 逐个 eval（含训练中途），greedy 选择在 eval 集上；与 C3 同源。
- **M5** WM 的 `nrmse8=0.060` 是 **PD-only 稳定分布**；策略访问的跌倒/恢复状态没有单独测。不能据此说“WM 在策略分布上准”。
- **M6** PPO 的 GAE 已修 `next_v`；但仍无 KL early-stop，探索 std 0.5/0.7 偏大；warm-start 被侵蚀与此一致。
- **M7** `imagine` 动作 clamp ±25，env 是 ±50；BPTT 与 eval 的动作范围不一致（次要）。
- **M8** `eval_steps=500` = 10 s @50 Hz；与论文 full-clip（数十秒到数分钟）不同，长时漂移未被评测。
- **M9** `episode_s=10`、`future_frames=10`、`truncated` 条件一致；无越界 bug。

## 4. 已验证一致/无问题的项

- `spec` 维度：state 283、obs 93、goal 730、xi 34；`tests/test_motion.py`、`test_env.py`（13）、`test_learning_shapes.py`（9）全过。
- `ACTION_SCALE` vs 模型：0.25·|forcerange|/kp，max diff 3.3e-7。
- `DEFAULT_Q` vs `knees_bent`：diff 0。
- 控制频率：`env` 强制 `timestep=0.005`，4 substeps = 50 Hz；与 motion 50 Hz、`DT=0.02` 一致。
- 奖励是 post-step `r_{t+1}`（`env.step` 注释与实现一致）。
- `eval` 的 `0.1*(T-1)` 对 **env8** 的 max 0.1 是正确的归一化。

## 5. 修复优先级（建议按序）

1. **统一奖励口径**（C1）：先决定 eval 用 env8 还是 analytic6；训练/评测/报告一致；所有历史数字加口径标签。
2. **全链路 aligned + 手臂 body 降权**（C2）：`env`/`evaluate`/`PPO`/`BC` 的 motion_dir 统一；env8 对 6 个手臂 body 权重降为 0.2 或 0。
3. **去掉 q_sp 的 ctrlrange clip**（C4），保留 action ±50。
4. **kp 实验同步 ASCALE**（H3），重跑 kp 对比。
5. **critic 用 73 维或 body 辅助 loss**（H1）。
6. **held-out clip + 随机 eval 起点**（H2/C3）；soup 权重只在 held-out 选。
7. **failure-adaptive 采样**（H5）。
8. **mjlab/Warp 4096 env**（H4）或云端；本机 8 env 不再作为“方法结论”的依据。
9. 清理 config/绝对路径，固定一个 `configs/final.yaml` + 一条 eval 命令（M3）。

## 6. 对已有结论的影响（诚实重述）

- **“eval 0.1046”**：是 **env8 + raw 参考 + 5 训练 clip + soup 在 eval 集上选权重** 的乐观数字；公平的单模型是 **0.093**；held-out 未测。
- **“WM nrmse8 0.060”**：是 **PD-only 分布**；策略分布未测，不能外推。
- **“站立 1.3 s / PPO 0.057”**：是 **analytic6** 口径；占满分 ~9.5%；与 0.1046 不可直接比较。
- **“2000 轮跑完/早停”**：实际 140 轮停止；配置在，但结论是“长训练退化”，不是“2000 轮验证通过”。
- **“soup 0.1046 > 单模型 0.093”**：部分来自选择偏差；需 held-out 复核。
