# SuperMimic-mini

在两个阶段、两张消费级显卡上，把 [SuperMimic](https://arxiv.org/abs/2610.28480)（arXiv:2610.28480，CoRL 2026）的核心机制拆开验证：**用一个可微世界模型替代刚体仿真器，再用短视野 BPTT 把策略梯度直接回传**。对象是 Unitree G1（29 DoF），任务是在 LAFAN1 动作上做全身跟踪。

| 阶段 | 硬件 | 这一阶段要回答的问题 |
|---|---|---|
| **一** | RTX 2060 6 GB + i7-1165G7 + 15 GB RAM | 6 GB 的预算，能把这条链路跑到哪一步？ |
| **二** | RX 6800 XT 16 GB（ROCm）+ Ryzen 7 5800X | 把"容量 / 梯度步 / 数据"三个杠杆拉满，机制的边界到底在哪？ |

两次实验合起来给出一条完整的因果链：**阶段一负责搭通链路并定位瓶颈，阶段二负责把瓶颈逐个证伪或确认**。最终结论是——动作条件化是**可以学到的**，BPTT 梯度是**真实可用**的；真正卡住端到端收益的，是奖励头少算了 14 个 body link 项。

---

## Key findings

| 能力 | 阶段一（6 GB） | 阶段二（16 GB） |
|---|---|---|
| 教师辅助策略 | **env8 0.466**（教师 0.4476） | — |
| 采集吞吐 | Warp 1024 env：**24.7k** 控制步/s | CPU 10 进程：~12k 控制步/s（无 Warp） |
| 世界模型 held-out 8 步 NSE | 0.84 – 1.28 | **0.334**（相对 persistence 优 **6.8×**） |
| 动作捕获率 `capture` | 7.5% | **81%** |
| 动作梯度符号一致率 | ≈0（梯度无意义） | **96.2%**（Pearson r=+0.622, p=1e-35） |
| BPTT 规划收益（真值仿真器） | 策略冻结 / 崩溃 | **+24.8%**，94% 样本改善，p=1.4e-9 |
| 完整 8 项奖励的规划收益 | — | **H=8 +3.79%（p=0.0013）/ H=16 +4.15%（p=0.018）**；H=32 失效 |
| 完整 actor-critic（BPTT） | 冻结 / 崩溃 | **raw 回报 +75.5%**（150 步协议，200 轮稳定） |

> **一句话**：阶段一测到的「动作敏感度 7.5%」是一个**预算受限的诊断，不是机制的上限**。同一套诊断在阶段二重做，真值仿真器的敏感度实测 **34.3%**，训练到位的世界模型捕获其中 **81%**。

---

## 现在到哪了（2026-11-14）

**机制问题已经回答清楚，剩下的是绝对水平问题。**

| 已确立 | 证据 |
|---|---|
| 「动作敏感度 7.5%」不成立 | 真值仿真器 **34.3%**，训练到位的 WM 捕获 **81%** |
| BPTT 梯度真实可用 | Pearson **+0.622**，符号一致 **96.2%** |
| 端到端堵点是奖励头 | 补 14 个 body 项后：−0.03% → **+3.79% (p=0.0013)** |
| 可用视野由模型精度决定 | NSE32 1.33→1.19 使 H=32 从 +0.07% → **+6.29%** |
| BONES-SEED 卡许可+存储 | gated(401)；per-clip npz 需 3.5 TB |

**策略绝对水平已大幅改善（P0 已完成），但与论文仍有差距。**

| 策略 | 归一化（每步上限 0.1） |
|---|---|
| 仓库教师（实测） | **0.469** |
| PD 跟踪基线 | 0.598 – 0.764 |
| **我们的策略（BPTT + 信赖域）** | **0.5817** |
| 我们的策略（教师蒸馏热启动） | 0.460 |
| 我们的策略（早期弱 BC） | 0.063 |
| 论文 | 0.82 |

相对提升（+75%）是真的，但基数差约 7–10×。原因很明确：AC 的 BC 热启动只有 2500 步，策略本身还是弱跟踪器。

**P0 已完成**：用公开教师（`supermimic_getup`，260k 迭代）做行为克隆热启动，把 93/730 策略从 **0.063 拉到 0.460**（教师 0.469）；再跑 150 轮 BPTT（加动作空间信赖域）到 **0.5817**，**超过教师 24%**。
绝对水平演进：`0.063 → 0.460 → 0.567（无约束）→ 0.5817（信赖域）`。完整记录与 P1–P3 计划见 [docs/REVIEW-AND-PLAN.md](docs/REVIEW-AND-PLAN.md)。

---

## 两阶段的算力差异（为什么值得重做）

| 维度 | 阶段一 | 阶段二 |
|---|---|---|
| 世界模型 | 1024³ | **2048³**（8.82M 参数） |
| 梯度步 | 1000 iters × 4 = **4k** | **30k**（参考实现 100k） |
| WM batch | 4 × 512 | **4096** |
| 想象视野 H | 16 | **32** |
| 数据 | 5 clips | **37 clips / 4M transitions** + 3 clips held-out |
| 采集引擎 | MuJoCo Warp（NVIDIA） | CPU MuJoCo 多进程（AMD 不能跑 Warp） |
| 单次实验耗时 | 分钟级 | 采集 8 min + 训练 55 min + 诊断 12 min |

---

## 阶段一：搭通链路，拿到第一个可交付策略（6 GB）

6 GB 显存能做的事很明确：

- ✅ **完整的工程链路**：LAFAN1 → G1 npz、283/93/730 维接口、MuJoCo 环境、8 项奖励、可微世界模型、BPTT 链路、评测与渲染，全部跑通。
- ✅ **可交付策略**：公开教师 + 残差 MLP，`env8 = 0.4662`（教师本身 0.4476，残差只做小幅增量、无侵蚀）。这是**教师辅助**结果，不是论文方法从零复现。
- ✅ **采集吞吐**：MuJoCo Warp 1024 env 达 24.7k 控制步/s，足够支撑 PPO 与数据采集。
- ⚠️ **世界模型与 BPTT 未通过**：8 步 NSE 0.84–1.28，BPTT 下 raw 策略冻结或崩溃。

阶段一由此得到一个**具体的诊断**：世界模型的"动作敏感度"只有自然状态变化的 ~7.5%——它学到的是"状态自演化"，动作几乎不起作用，因此 BPTT 梯度没有意义。这个诊断很关键，因为它把问题**精确定位到了"动作条件化"**，而不是模糊的"训练不稳定"。

完整记录见 [`docs/FINAL-REPORT.md`](docs/FINAL-REPORT.md) 与 [`docs/WM-UPGRADE.md`](docs/WM-UPGRADE.md)。

---

## 阶段二：把三个杠杆拉满，看诊断是否成立（16 GB）

阶段一受限于"容量 / 步数 / 数据"三者同时不够。阶段二把这三个都拉上去，并用三个独立测量判断机制的可用性。

**有意思的是：阶段二能跑通，恰恰是因为它绕开了阶段一的采集引擎。** 本仓库核心路径是 MuJoCo CPU + PyTorch，Warp 只是可选项——所以换成 AMD 卡做**学习侧**的实验完全不受影响；代价是采集吞吐降到约 12k 控制步/s。

### 世界模型精度

| 梯度步 | NSE1 | NSE8 | NSE32 |
|---|---|---|---|
| 2,500 | 0.132 | 0.512 | 3.42 |
| 10,000 | 0.106 | 0.387 | 1.83 |
| 20,000 | 0.104 | 0.359 | 1.48 |
| **30,000** | **0.101** | **0.334** | **1.33** |

对照平凡基线（同一批 held-out 窗口）：persistence 的 NSE8 是 **2.28**、常数速度外推是 **3.84**。所以世界模型在 8 步上比 persistence 好 **6.8×**，32 步好 **6.7×**。

### 动作条件化：7.5% 还是 81%？

在训练损失同一个归一化空间里做严格 1 步对照（每步重同步 `MjData`）：

| δ | 真值敏感度 | WM 敏感度 | **capture** |
|---|---|---|---|
| 0.10 | 34.3% | 27.7% | **80.8%** |
| 0.50 | 170.8% | 138.5% | **81.1%** |
| 1.00 | 330.0% | 273.9% | **83.0%** |

三个信息：**真值本身就是 34.3%**（不是 7.5%）；**WM 捕获了 81%**；capture 对扰动幅度高度稳定，说明学到的是**一条线性、方向正确的动作 Jacobian**，只是幅度低估约 20%（L2 回归的典型收缩）。另外，**2,500 步时 capture 只有 33%**——正好印证"欠训练"这一解释。

### 梯度质量

对同一状态、同一组动作扰动，比较 WM 预测的奖励变化与真值（两侧用同一个 6 项奖励，差异只来自状态预测）：

```
n = 320   Pearson r = +0.622 (p=1.2e-35)   Spearman ρ = +0.608 (p=8.8e-34)
符号一致率 = 96.2%
```

### 端到端：把规划的 32 步动作放回真值仿真器

从 PD 跟踪动作出发，用 WM 想象 32 步并对动作序列做梯度上升（即 BPTT），再把动作在真实 MuJoCo 里执行：

| 评价口径 | 基线 | WM 规划 | 提升 | 改善占比 | 配对 t |
|---|---|---|---|---|---|
| **6 项分析奖励（与 WM 优化目标同口径）** | 0.9758 | **1.2174** | **+24.76%** | **93.8%** | p=1.4e-9 |
| 环境完整 8 项奖励 | 1.9826 | 1.9820 | −0.03% | 40.6% | p=0.99 |

**这是阶段二最重要的结论，也是它推动的下一步**：在 WM 建模的目标上，BPTT 规划带来显著的真实提升，说明"通过世界模型反传得到的动作在物理世界里确实更好"。收益未能迁移到完整 8 项奖励，原因很具体——`smini/imagine.py::analytic_reward` 只算 6 项，**漏掉 14 个 body link 跟踪项**（源码注释也写明 v1 省略，因为需要可微 FK）。

**于是瓶颈完成了一次转移：从"动力学学不到动作"转到"奖励头不完整"。** 这是一个比阶段一的诊断更接近可解决的问题。

完整记录见 [`docs/WM-BPTT-16GB.md`](docs/WM-BPTT-16GB.md)，复跑脚本在 [`exp/`](exp/README.md)。

---

## 继续推进：补齐奖励头，接上 actor-critic

阶段二留下的缺口很明确：收益只在 6 项分析奖励上成立，换成环境完整 8 项奖励就归零。原因在 `smini/imagine.py::analytic_reward` —— 它漏掉了 14 个 body link 的位置/姿态/线速度/角速度四项。

本轮用纯 torch 复刻了 G1 的前向运动学（`exp/diff_fk.py`），把这四项补回来。对拍结果：FK 位置/姿态/速度误差 **≤5.3e-07**，完整 8 项奖励与环境 `TrackingEnv._reward` 差 **8.9e-09**。

**补上之后，收益确实迁移了：**

| 想象视野 H | 基线（PD） | WM 规划 | 提升 | 改善占比 | p |
|---|---|---|---|---|---|
| 8 | 0.6748 | 0.7004 | **+3.79%** | 81.2% | **0.0013** |
| 16 | 1.2219 | 1.2726 | **+4.15%** | 66.7% | **0.018** |
| 32（30k 模型） | 1.9145 | 1.9158 | +0.07% | 37.5% | 0.98 |
| **32（100k 模型）** | 1.9145 | 2.0349 | **+6.29%** | 47.9% | 0.075 |

H=32 失效不是巧合：WM 的误差恰好在那里爆炸（`NSE8=0.334` → `NSE32=1.33`）。**可用视野由模型精度决定。** 把 WM 练到 100k 步（`NSE32` 1.33→1.19）后，同一条 H=32 测试从 +0.07% 跳到 **+6.29%**——视野不是超参，是模型能力的函数。

接着把循环从"直接优化动作序列"升级成**完整 actor-critic**（`exp/ac_bptt.py`：Policy + Critic + 想象 rollout + 8 项奖励）。BC 热启动 2500 步后跑 200 轮 BPTT：

- raw 回报 **+75.5%**（150 步协议，范围 +55%~+98%，200 轮内不衰减）
- imagined return `J` 2.53 → 5.33，critic loss 0.138 → 0.052 —— 学习信号自洽

需要如实标注的口径：换成 **200 步**评测协议，增益只有 **+14.5%**。差距来自弱策略的漂移（episode 变长后中途终止），所以"BPTT 提升多少"必须连着评测协议一起说。

**顺带发现一个环境 bug**：`env.state()` 的 body 块比 `qpos` 滞后一个物理子步（`mj_step` 只在步首算运动学），导致存进 replay 的奖励与解析正确值差约 1e-3。本轮未改环境行为，仅记录并绕开。

**加步数：数字上收益递减，但会跨阈值**。WM 从 30k 续到 100k，`NSE8` 只从 0.334 到 0.307（8%）、`NSE32` 到 1.19（11%）；可这 11% 正好让 H=32 的规划从无效变成 +6.29%。所以判断该不该继续加步数，要盯**目标视野上的 NSE**，而不是平均 NSE。

完整记录见 [`docs/REWARD-HEAD-AND-AC.md`](docs/REWARD-HEAD-AND-AC.md)。

---

## 从这些实验学到的五点

1. **动作条件化是可学的，而且它就是 BPTT 的前提。** 世界模型能否支撑 BPTT，取决于它对动作的响应幅度是否真实，而不只是对被动轨迹拟合得好。
2. **L2 训练会系统性低估动作效应。** 我们稳定在 capture 81% 而不是 100%——方向对、幅度偏小。这通常反而让梯度上升更保守、更安全，但值得在 loss 设计上留意。
3. **绝对误差门槛必须按数据分布校准。** 论文的 8 步 NSE 0.007–0.011 是在平滑的 on-policy rollout 上测的；在带探索噪声的数据上，persistence 本身就是 2.28。用"相对 persistence 的倍数 + capture"作判据更稳。
4. **收益能否迁移，取决于奖励/观测头的完整性。** 动力学准了之后，下一步的瓶颈往往不在动力学——补上 14 个 body 项后，规划收益才真正落到任务奖励上。
5. **还需要视野对得上。** 补上奖励头之后，收益只在 H ≤ 16 成立；H=32 会被 WM 的累积误差吃掉。目标、视野、模型精度三者必须同时对齐，缺一个收益就归零。

---

## 下一步

1. **强教师热启动**：当前 BC 只有 2500 步、loss 在 1–10 间波动，策略本身是弱跟踪器（每步 0.006 vs PD 的 0.06）。把已发布的 G1 教师蒸馏进来再 BPTT，是把绝对分数拉起来的直接路径。
2. **on-policy 数据循环**：让 WM 在**当前策略**的 rollout 上持续拟合（论文的做法），而不是固定探索数据。这是降 `NSE32`、从而把可用视野推到 H=32 的关键。
3. **trust-region / KL 约束**：`docs/BPTT-STABILITY.md` 记录过无 KL 时长跑会崩；本轮短跑未触发，但长跑必须加。
4. **BONES-SEED 前置工程**：目前卡在**许可**（`bones-studio/seed` 是 gated，直连 401）与**存储**（per-clip npz 约 24.7 MB × 142k ≈ 3.5 TB，需改成参考实现那样的 6.7 GB 紧凑 fp16 库），不是算力。

## 仓库结构

- `smini/` — 环境、spec（283/93/730）、动作转换、policy/critic、世界模型、imagination/BPTT、replay、训练循环。
- `scripts/` — 数据管线、教师适配（JAX/MJX）、PPO（MJX/Warp）、WM71 训练与评测、BPTT（CPU/Warp）、MJX 诊断、渲染。
- `exp/` — **阶段二**的复跑脚本（采集、WM 训练、动作捕获、梯度一致性、规划验证）。
- `docs/` — 两阶段的完整报告、论文 gap 分析、审计、可行性调研。
- `docs/demo/` — 渲染 demo（不含权重与数据）。

不包含：动作数据、教师 checkpoint、训练权重、`runs/`、`assets/`。见 [数据](#数据) 与 [NOTICE.md](NOTICE.md)。

## 安装

Python ≥ 3.12。仓库用到两个环境：

```bash
# 主环境：CPU MuJoCo + torch（教师/PPO/渲染，以及 exp/ 的全部脚本）
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0   # NVIDIA

# AMD(ROCm)：用官方 gfx103X 轮子（内含 ROCm SDK，勿与系统 ROCm 对齐）
# pip install --index-url https://repo.amd.com/rocm/whl/gfx103X-all/ torch==2.10.0+rocm7.13.0

# 可选 Warp 环境（1024-env 采集 / Warp 版 BPTT），仅 NVIDIA 可用
python -m venv .venv-warp && . .venv-warp/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install mujoco==3.16.0 mujoco-warp==3.16.0 warp-lang==1.17.0 scipy pyyaml
```

代码按 `../assets/...` 解析 G1 模型，期望的 workspace 布局：

```
your_workspace/
  assets/
    mujoco_menagerie/unitree_g1/scene_mjx.xml
  supermimic-mini/        # this repo
```

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie ../assets/mujoco_menagerie
```

## 数据

LAFAN1 不随仓库分发（CC BY-NC-ND 4.0）。从
[HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
下载 G1 retarget 后转换：

```bash
python scripts/convert_motions.py \
  --clips walk1_subject1 run1_subject2 dance1_subject1 sprint1_subject2 jumps1_subject1 \
  --out-dir assets/motions

python scripts/trim_motions.py          # 裁掉 T-pose 引入段 -> assets/motions_trim
```

40 条实验直接传全部 clip，并使用 `motions40` / `scripts/trim_motions40.py`（路径在脚本顶部）。

## 快速开始

```bash
# 单元测试
python tests/test_motion.py
python tests/test_env.py
python tests/test_learning_shapes.py

# BPTT smoke（8 env，CPU，约 1 min）
python -m smini.train --config configs/p0.yaml
```

教师 + 残差评测（需要 `runs/pretrained/policy.pt` 与 `runs/ppo_teacher5m/ckpt.pkl`，见 NOTICE）：

```bash
python scripts/eval_pretrained.py --steps 500
python scripts/eval_teacher_ppo_cpu.py
python scripts/render_ppo_policy_long.py   # 长版 demo
```

WM71 训练 / 固定 held-out 评测：

```bash
python scripts/train_wm71_clean.py \
  --envs 1024 --iters 1000 --updates 4 --batch 1024 \
  --motion-dir assets/motions40_trim \
  --train-clips 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38 \
  --eval-clips 39
```

## 复现阶段二实验

最短路径（详见 [`exp/README.md`](exp/README.md)）：

```bash
V2=1 python exp/collect.py 4000000 10                    # 采集 4M transitions（10 进程，8 min）
V2=1 python exp/collect_holdout.py                       # 3 条 held-out clip

DATA_DIR=data_v2 python exp/wm_train.py \
    --steps 30000 --batch 4096 --xi-dim 34 --amp 1 --eval-every 2500 --tag v2   # 55 min

python exp/diag_sensitivity.py exp/runs/v2_best.pt       # capture
python exp/grad_corr_test.py  exp/runs/v2_best.pt        # 梯度一致性
python exp/mpc_test.py --ckpt exp/runs/v2_best.pt --H 32 # 规划验证
```

阶段二踩到并修掉的两个坑（对复现者重要，细节见 `exp/README.md`）：

- **数据必须是 env-major**。多进程按 tick-major 拼接会让相邻索引属于不同机器人，几乎每条转移都是假的——现象是 loss 卡在 100、NSE1=113；重排后 loss 0.30、NSE1 0.127。
- **xi 用 34 维**。`spec.sample_xi` 的左右脚摩擦独立采样，而 `wm_import.WM71` 默认只有一个摩擦槽。

## 报告

| 文件 | 内容 |
|---|---|
| **[docs/REVIEW-AND-PLAN.md](docs/REVIEW-AND-PLAN.md)** | **六轮工作总结、关键数值汇总、诚实 gap 清单与 P0–P3 实验计划** |
| [docs/REWARD-HEAD-AND-AC.md](docs/REWARD-HEAD-AND-AC.md) | **第四轮**：可微 FK（1e-7）、8 项奖励规划迁移、完整 actor-critic、BONES-SEED 可行性 |
| [docs/WM-BPTT-16GB.md](docs/WM-BPTT-16GB.md) | **阶段二主报告**：WM 精度、动作捕获 81%、梯度一致性、规划验证与瓶颈转移 |
| [docs/FINAL-REPORT.md](docs/FINAL-REPORT.md) | 阶段一最终报告：思路、正/负结果、根因、gap |
| [docs/PAPER-GAP.md](docs/PAPER-GAP.md) | 与论文的量化对照 |
| [docs/WM-UPGRADE.md](docs/WM-UPGRADE.md) | WM 导入/重训尝试与动作敏感度诊断 |
| [docs/BPTT-STABILITY.md](docs/BPTT-STABILITY.md) | trust-region、EMA/bias bug、冻结/崩溃 |
| [docs/BPTT-MINIMAL-FAIR.md](docs/BPTT-MINIMAL-FAIR.md) | 1024-env / 5M 公平对照 |
| [docs/MJX-FEASIBILITY.md](docs/MJX-FEASIBILITY.md) | 反向模式受阻；前向模式成本 |
| [docs/AUDIT.md](docs/AUDIT.md), [docs/REVIEW.md](docs/REVIEW.md) | 数据与实验审计 |

## 范围与局限

- 阶段二的 BPTT 是**规划式代理**（直接对动作序列求梯度），不是完整 actor–critic；它证明了梯度可用，未证明策略网络能收敛。
- 仅 LAFAN1 40 条（非 BONES-SEED 129k）、单卡、单种子，未做种子方差。
- 阶段二的 `reward71` 是按 `imagine.analytic_reward` 复刻的 6 项版本，与原仓库一致（正是它的已知 v1 缺陷）。
- 阶段一的 0.466 是**教师辅助**结果，不可与论文的 0.82 直接比较。
- 采集仍用 CPU MuJoCo（AMD 卡不能跑 Warp），吞吐约为阶段一 2060 上 Warp 的一半；实验证明这不构成阻断。

## 许可与引用

代码 MIT（见 [LICENSE](LICENSE)）。数据、模型资产、教师 checkpoint 为第三方，不随仓库分发，见 [NOTICE.md](NOTICE.md)。

如果你基于本仓库工作，请引用 SuperMimic（arXiv:2610.28480）并链接本仓库。
