# SuperMimic-mini 最终报告

> 日期：2026-11-10
> 目标：在 RTX 2060 6 GB + i7-1165G7 + 15 GB RAM 上，尽可能复现 SuperMimic（arXiv:2610.28480）的“可微世界模型 + BPTT”思路，并给出可信的结论与边界。
> 结论先行：**策略侧做到 teacher+residual env8 0.466；WM/BPTT 在本机未复现（WM 动作条件化失败、BPTT 冻结或崩溃）；论文级 0.82/4096 env/100M 不可达。** 本报告把“研究思路、正结果、负结果、根因、gap”分开写清楚。

---

## 1. 研究思路

SuperMimic 的核心不是“又一个跟踪策略”，而是：
1. 用 **一个可微世界模型**替代刚体仿真器；
2. 在 WM 里做 **短视野 BPTT**，把策略梯度直接回传；
3. 以 **4096 env / ~100M transitions / 2048³ WM / H=32** 的规模，把 LAFAN1 40 条整条跟踪做到 0.82。

本机无法复刻规模，于是把问题拆成四条可验证的线：

| 线 | 问题 | 判据 |
|---|---|---|
| A 策略侧 | 公开教师 + 残差能否超过教师？ | env8 ≥0.45 |
| B 规模 | Warp 1024 env 能否支撑 PPO/数据采集？ | ≥10k ctrl-steps/s |
| C WM | 71/283 维 WM 的 8 步开环误差能到多少？ | held-out NSE8 ≤0.05 |
| D BPTT | 在 WM 里做 BPTT 能否稳定提升 raw 策略？ | raw 不崩且 ≥init+0.02 |

关键方法论：**先修评测，再谈算法**。本轮多次因为评测口径（EMA、同族 held-out、流式 eval）得出过乐观结论，最终都以“独立 held-out + 留出 clip”推翻。

---

## 2. 系统与数据

- 数据：LAFAN1 G1 全量 **40 clips / 2.45 h**（HF `lvhaidong/LAFAN1_Retargeting_Dataset/g1`），转 50 Hz npz；去 T-pose 后 **2.35 h**（`assets/motions40_trim`）。类别含 dance 8、walk 12、run 4、sprint 2、jumps 3、fight 3、fightAndSports 2、**fallAndGetUp 6**。
- 观测/目标：论文口径 **93 obs + 730 goal（10×73）**；教师接口为 **930 obs + 680 goal**（BONES/anchor-relative 口径，非论文口径）。
- 状态：283 维（root+joint+14 body）与 71 维（`[qpos36|qvel35]`）两套。
- 仿真：CPU MuJoCo（50 Hz，4×5 ms 子步）、MJX、MuJoCo Warp 1.17（`.venv-warp`）。
- 教师：公开 `supermimic_getup` 260k iters（34 MB），本机 trimmed env8 **0.4476**。

---

## 3. 正结果

### 3.1 策略侧（本机最佳）

| 策略 | env8（trimmed 5×500） | 说明 |
|---|---|---|
| 教师 | 0.4476 | torch/JAX 对拍 2.6e-6 |
| **teacher + residual（P2.2）** | **0.4662 / 0.4668** | 本机最佳；+5.4%，无侵蚀 |
| PPO Warp 1024 env（frozen teacher + residual + BC） | 0.4460 | 强 BC 保守；未超过 P2.2 |
| soup（历史） | 0.1046 | 仅在 5 clips in-sample 选权重，**有选择偏差，不作数** |
| BC flat（历史） | 0.0408 | 负结果 |

- 0.466 的组成：公开教师（BONES 260k）+ 残差 MLP（1610→512→256→29），BC/PPO 在 8 env、~0.8M transitions 上微调。
- 这是**教师辅助结果**，不是论文方法从零复现；与论文 0.82 不可直接比较。

### 3.2 工程吞吐

- Warp 1024 env：**24.7k 控制步/s**（98.9k 物理步/s），比 MJX 256 env ~8×。
- 100M transitions ≈67 min（仅物理采集；WM/BPTT 另计）。

---

## 4. 负结果与根因（核心）

### 4.1 WM 精度

| WM | 数据 | 8 步误差 | 结论 |
|---|---|---|---|
| 283 维/1024³ | teacher/PD 80k | std≈0.30；persistence 比 0.257 | 方向对，绝对误差大 |
| 73 维/1024³ | PD 80k | std≈0.35；比 0.303 | 更差 |
| **71 维/2048³（重训）** | PD 4M | **held-out NSE8 0.84–1.28（sqrt 0.92–1.13）** | 失败 |
| 71 维/40 clips | 39 train / walk4 held-out | 0.84–1.28 | 数据扩展仅 ~10% |
| 导入 SuperMimic-Reproduced ckpt | — | 1 步 NSE 0.37–1.25 | 跨栈不可 drop-in |

**根因（关键诊断）**：WM 动作敏感度只有 **~7.5%**。
- 固定状态，动作扰动 1.0 → 下一状态变化 **0.041**；自然一步变化 **0.543**。
- 数据里 PD 动作几乎是状态/参考的确定函数（+0.1 噪声），动作独立变化太少；WM 学到“状态自演化”，策略改动作时 WM 几乎不动 → BPTT 梯度无意义。
- 加探索（noise 0.5 + 10% random）后 held-out 反而 **1.28**：动作覆盖提高但状态预测更难，不是解。

### 4.2 BPTT 稳定性

| 设定 | raw env8 | 结论 |
|---|---|---|
| 8 env / 0.1M，无 KL | 0.0573→0.0469 | 退化 |
| 1024 env / 2M，无 KL | 0.0629@300 → **0.0229@400** | 崩溃 |
| 固定 KL 0.5 | 0.0544（KL→0） | 冻结 |
| 自适应 KL | 0.0549→0.0526 | 仍不升 |

- 中间发现并修复了两个**评测/实现 bug**：EMA-only 评测掩盖 raw 退化；`ref_policy` 在 `set_action_bias` 之前复制导致零 bias。修复后“假上升”消失，真实行为是**冻结或崩溃**。
- 结论：**本机 BPTT 未复现**；瓶颈在 WM 动作条件化，而不是单纯 KL/样本量。

### 4.3 MJX 对照

- MJX forward 可用；**reverse-mode 被 solver 的 `lax.while_loop` 挡住**。
- forward-mode `jvp`/`jacfwd` 可用，但 29 方向 vmap/jit 在本机 **>10 min 未编译完**；H=4/batch32 的 BPTT 对照不实用。
- 因此“WM vs credit assignment”的最终分离实验未能在本机完成；但动作敏感度诊断已足够解释 BPTT 失败。

---

## 5. 与论文的 gap

| 维度 | 论文 | 本机 | 差距 |
|---|---|---|---|
| env | 4096 | BPTT 8；Warp 1024 | 4–512× |
| transitions | ~100M | BPTT 0.1–2M | 50–1000× |
| WM | 2048³、71 维、8 步 0.007–0.011 | 1024³/2048³，held-out 0.84–1.28 | 量级差 |
| 数据 | 40 clips/2.45 h（论文）/129k BONES | 40 clips（已备） | 同量级（LAFAN1）；BONES 不可得 |
| 策略 | 0.82（LAFAN1 整条） | **0.466**（5 clips trimmed，教师辅助） | 0.35 |
| 评测 | 40 条整条 | 5 条×500 步；held-out clip | 不可直接比 |
| 硬件 | 1×H100 3 h | 2060 6 GB | ~10–30× |

详见 `docs/PAPER-GAP.md`。

---

## 6. 方法论教训（可复用）

1. **评测必须独立**：同族 1/5 episode split 会把 0.85 报成 0.17；必须独立 seed + 留出 clip + 固定 eval set。
2. **raw + EMA 双报**：EMA 会掩盖 raw 崩溃（旧 BPTT 结论因此作废）。
3. **数据 T-pose 必须裁**：LAFAN1 clip 有 3–6.5 s 站立引入段；不裁会看到“T-pose 跟踪”。
4. **`motions_aligned` 是坏数据**：用 frame0 常量偏移修手臂，导致动作漂移；已弃用。
5. **教师接口 ≠ 论文接口**：930/680（anchor-relative）与 93/730 不可混用；0.466 不能标成“论文复现”。
6. **跨栈导入 WM 不可 drop-in**：state 可逆、动作 scale/顺序/xi 任一不匹配，1 步误差就 1e0。
7. **动作条件化是 WM 的生命线**：只采 PD 数据，WM 会忽略动作；BPTT 必然失败。

---

## 7. 交付物

### 代码
- `smini/`：env、spec、motion、policy、WM、imagine、train、replay。
- `scripts/`：数据转换/裁剪、教师 JAX/MJX、PPO Warp、WM 训练、BPTT、MJX/JAX 诊断、渲染。
- 数据：`assets/motions40/`、`assets/motions40_trim/`。
- 产物：`runs/pretrained/`、`runs/ppo_teacher5m/`（0.466）、`runs/wm71_40*/`、`runs/imported/`。

### Demo
- **0.466 长版**：`docs/demo/supermimic_ppo_demo_long.mp4`（5 clips × 400 步，~40 s）。
- 0.466 标准版：`docs/demo/supermimic_ppo_demo.mp4`（19.6 s）。
- 教师版：`docs/demo/supermimic_pretrained_demo.mp4`。
- 说明：`docs/demo/README.md`。

### 文档
- 本报告；`PAPER-GAP.md`；`AUDIT.md`；`REVIEW.md`；`WM-UPGRADE.md`；`WM-BPTT-PLAN/VALIDATION.md`；`BPTT-STABILITY.md`；`BPTT-MINIMAL-FAIR.md`；`MJX-FEASIBILITY.md`。

---

## 8. 后续（若要继续）

1. **动作条件化 WM**：policy/teacher rollout + 大动作扰动（noise 0.5–1.0、random 10–20%）、term head、action encoder/FiLM、multi-step curriculum；目标 held-out 8 步 ≤0.05。
2. **规模**：4096 env / 100M / 2048³ / 40 clips；需云端或更强 GPU。
3. **MJX**：在支持 reverse-mode 的 solver/更强 GPU 上做 H=4–8 对照；本机不可行。
4. **策略侧**：0.466 已可交付；若要 0.5+，优先在 40 clips 上做 PPO/教师+residual，而不是先修 WM。

---

## 9. 一句话结论

**本机复现了 SuperMimic 的数据/环境/教师/可视化与“WM+BPTT”工程链路，但没有复现其算法收益：策略侧靠公开教师到 0.466，WM 因动作条件化失败（敏感度 7.5%）而无法支撑 BPTT；论文级 0.82 需要 4096 env/100M/2048³，本机不可达。**
