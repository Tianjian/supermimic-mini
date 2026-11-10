# 突破路线图 v2（基于 2026-11-09 证据重排）

> 修订原因：v1 把“从零训练 + anchor obs”放在最前，但实测 **BC 0.0408**、**anchor 在 256 env 无增益**；同时**预训练教师 0.4476（trimmed）**已可用。v2 改为**教师中心**：先移植教师架构、用 PPO 热启动 + KL 锚定，再谈规模和 WM。

## 0. 当前证据（决定优先级）

| 项 | 结果 | 含义 |
|---|---|---|
| 预训练教师（trimmed，5×500） | **0.4476**，5/5 clip 跑满不摔 | 唯一“能干”的策略；可做教师/初始化 |
| 教师架构 | goal 680（10×68 anchor-relative）+ obs 930（term-major）+ 29 动作；encoder/decoder 2048-1024-512-256 | **本机应沿用这个接口**，不要再从 93/730 从零学 |
| B8 从零 PPO（256 env，8M steps） | 0.041–0.043 | 样本量差 100×，不可作为方法结论 |
| BC flat MLP（trimmed 50k） | **0.0408**（MSE 0.0106） | 平面 MLP 容量/分布偏移失败；**停止 naive BC** |
| Anchor obs（256 env，8M） | 0.0411 vs 0.0421 | 结构对，但小样本无增益；**降级为规模后的条件项** |
| Soup（raw，in-sample） | 0.1046（单模型 0.093） | 仅作历史对照，不再作为主线 |
| WM（PD 分布） | nrmse8 0.060 | 远低于复现 0.007–0.01；WM 主线后置 |
| 数据 | T-pose 引入段 3–6.5 s；`motions_aligned` 坏数据 | 已修：`motions_trim/`，raw 有效 |

## 1. 优化后的原则

1. **教师优先**：任何新策略先以教师 0.4476 为基线；先“保住教师”，再谈超过。
2. **接口统一**：obs 930（term-major）+ goal 680 + 29 动作；env 增加适配层，不再用 93/730 训练新策略。
3. **防侵蚀**：PPO 必须带 **KL/BC 锚定**（对教师或行为策略），否则会重复 warm3 的 0.059→0.047 侵蚀。
4. **规模门槛**：<1024 env 只做正确性/回归；性能结论必须 ≥1024 env 或 mjlab/Warp。
5. **一次一个变量**：每步有 gate，失败回滚，不叠加未验证改动。
6. **评测固定**：主指标 **env8 + trimmed + 5 clips×500 + 第 0 帧**；held-out 另报；6 项仅内部监测。

## 2. 阶段与 Gate

### P1 数据/接口收口（0.5–1 天）
- [x] trimmed 数据（`assets/motions_trim/`）
- [x] 教师 adapter（680 goal / 930 obs），trimmed 0.4476
- [x] 审计（AUDIT v2：C2 更正、C7 T-pose）
- [ ] 去 `q_sp` 的 ctrlrange clip（论文明确不裁）；加 `EnvCfg.clip_qsp` 开关
- [ ] held-out：walk+sprint 留出，训练/评测分离
- **Gate G1**：教师 trimmed 复现 0.4476 ±0.01；测试全过；held-out 有数。

### P2 教师移植 + PPO 热启动（2–4 天，最高优先级）
#### 2.1 教师 → JAX（1–2 天）
- 把 680 goal encoder（2048-1024-512-256→32）和 930 obs decoder（2048-1024-512-256→29）移植到 JAX；加载 checkpoint 权重。
- 用 MJX 跑 5×500，**必须复现 0.4476 ±0.01**（CPU 与 MJX 各一次）。
- **Gate G2.1**：JAX 教师 = 0.4476±0.01；否则查 obs 顺序/scale/termination。

#### 2.2 PPO 热启动 + KL 锚定（1–2 天）
- 初始化 actor = 教师；critic 新训；actor 加 **KL(π‖π_teacher) 惩罚**（系数 0.1–1.0，逐步降）或 BC 正则。
- 256–1024 env、rollout 32–64、H=32、5–20M steps；先只验证“不降”。
- **Gate G2.2**：5M steps 后 env8 ≥0.42（不侵蚀）；10–20M 后 ≥0.45。
- 失败回滚：提高 KL/冻结 actor 只训 critic，或退回教师直接部署。

#### 2.3 教师蒸馏（可选，1 天）
- 用教师架构（goal encoder+decoder）在教师访问状态上做 DAgger；**不要用 flat MLP**。
- **Gate**：DAgger ≥0.35；否则放弃蒸馏，直接用教师/PPO。

### P3 规模（3–5 天）
- 3.1 装 mjlab/MuJoCo Warp，benchmark 512/1024/2048 env（2060 6GB）；**Gate**：≥20k steps/s @1024，显存 <6GB。
- 3.2 PPO 从教师热启动 + KL，20–50M steps；**Gate**：env8 ≥0.55，held-out ≥0.45。
- 3.3 若 2060 只能 512 env：改为 512 env + 更长训练，或云端。

### P4 WM / 论文版（3–5 天，条件触发）
- 4.1 71 维状态、H=32、wm 2048³、wm_batches 10/2048；body 奖励用可微 FK；**Gate**：WM 8 步 ≤0.02。
- 4.2 BPTT 从教师/PPO 策略热启动（或从零 100M）；**Gate**：env8 ≥0.5 且单调上升。
- 若 G4 不过，**不跑长 BPTT**，把 WM 结论限定为“机制验证”。

### P5 收口（1 天）
- 40 LAFAN1 或 held-out 5 条；full-clip（不截 500 步）；teacher/PPO/BC 对比表。
- demo：策略 vs trimmed 参考；标注“教师/本机训练”。
- 报告：AUDIT/RESEARCH/ROADMAP v2 + 复现命令。

## 3. 建议的最小可行胜利（MVW）

**1 周内**：JAX 教师复现 0.4476 → PPO+KL 5–10M 保住 ≥0.44 → 用**本机 PPO 策略**录一版 demo（不是直接教师）。
**2 周内**：mjlab 1024 env、20M steps，冲到 ≥0.55。
**3–4 周**：WM 71/H=32 + BPTT 对照（条件）。

## 4. Do-not 列表（血泪教训）

- 不再在 **8 env / 1M transitions** 上从零训练并据此下结论。
- 不再用 **flat MLP BC**（0.0408）；要用教师架构 + DAgger。
- 不再用 **`motions_aligned`**（坏数据）；只用 raw/trimmed。
- 不再无 KL 的 PPO warm-start（warm3 已侵蚀）。
- 不再把 **6 项 / 8 项** 数字并列；报告必须标口径。
- 不再用 **1 clip/100 步** 的 0.2128 与 5×500 的 0.1046 比较。

## 5. 固定评测/命令

```bash
PY=../.venv/bin/python
$PY tests/test_motion.py && $PY tests/test_env.py && $PY tests/test_learning_shapes.py
# 教师（trimmed）
$PY scripts/eval_pretrained.py --motion-dir assets/motions_trim
# 训练数据准备
$PY scripts/trim_motions.py
# P2.2（待实现）
$PY scripts/ppo_mjx_teacher.py --envs 1024 --steps 10000000 --kl-teacher 0.5
```

## 6. 当前进度

- [x] P1 数据/教师/审计
- [x] P2.1 教师 JAX 移植：**完成**。权重对拍 max diff **2.6e-6**；CPU env8 **0.4410**（torch 0.4476，差 0.0066，Gate ±0.01 内）；MJX 5/5 clip **500 步不摔**。`scripts/teacher_jax.py`、`scripts/teacher_mjx.py`。
- [~] P2.2 PPO 热启动 + KL：已实现 `scripts/ppo_teacher_mjx.py`（冻结教师 + residual + BC 锚）。100 iters/0.8M steps：首次 CPU env8 **0.4584**（walk 0.648），但从该 ckpt 继续 100 iters 回落到 **0.4424**（≈教师 0.4421）——**无侵蚀，但 +0.016 不稳定，按噪声处理**。residual 512-256 + BC 0.1→0.03 退火；100 iters（0.8M）→ **0.4614**；200 iters（1.6M）→ **0.4668**；300 iters（2.4M）→ **0.4662**（sprint 在第 281 步摔，0.2195）（教师 0.4421，**+4.4%**；dance 0.700 / walk 0.570 / jumps 0.391 / run 0.353 / sprint 0.292），无侵蚀。5M（600 iters）run 进行中，600 轮后复评。
- [x] P3.1 Warp 基准：**达成**。独立 `.venv-warp`（mujoco 3.16 + mujoco-warp 3.16 + warp 1.17）在 2060 上实测：256/512/1024 env = 6.1k/11.9k/**24.7k** 控制步/s（98.9k 物理步/s）；≥20k Gate 通过。`scripts/bench_warp.py`。
- [x] P4 WM 升级（负结果）：动作敏感度 ~7.5%（WM 忽略动作）；高探索 held-out 1.28 更差；MJX 29 方向 vmap/编译过慢（>10min），H=4 对照不实用。结论：WM/BPTT 在本机不达标，接受 0.466 收口。
- [x] P4 WM/BPTT 稳定性（负结果）：修复 ema/ref 顺序 + set_norm 两个 bug；修复后 KL 0.02/0.5 冻结（raw=0.0573），kl=0/lr1e-3 在 25 iters 崩到 0.0030；无“能学且不崩”区间。
- [x] P4 WM/BPTT 机制验证（修正）：E1/E2 WM 8 步 ratio 0.257（283）/0.303（73）；E3 三种设定（frozen 283 / frozen 73 / online WM）raw 均退化或崩溃（0.0469 / 0.0420 / 0.0140），EMA 掩盖；BPTT **未复现** → H3 拒绝；E4/E5 无效/未做。详见 `docs/WM-BPTT-VALIDATION.md`。
- [ ] P5 收口
