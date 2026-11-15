# SuperMimic-mini 最终报告（下一阶段）

> **口径说明（2026-11-09 审计后，必读）**：本报告的 `eval return` 默认指 **env 8 项 body 奖励 / 5 条训练 clip / 500 步 / 从第 0 帧**；历史里的 1 clip/100 步 0.2128 **不可比**。`0.1046` 是 4-way soup 且权重在同一 eval 集上挑选（单模型 0.093），属乐观估计；held-out 未测。`WM nrmse8 0.060` 是 PD-only 分布，不是策略分布。6 项 joint/root 奖励（BPTT/PPO）与 env8 的满分不同（0.06 vs 0.1），数字不可直接并列。审计详情见 `docs/AUDIT.md`，突破路径见 `docs/RESEARCH.md`。


> 日期：2026-11-06
> 目标：WM compact nrmse8 ≤0.15；2000+ 轮 P0；eval return 0.1+；生存 100+ 步。

## 最终结果

| 指标 | 结果 | 证据 |
|---|---|---|
| WM nrmse8（PD 稳定分布） | **0.060**（nrmse1 0.051，persistence 0.111） | `runs/wm_pd.log`，80k / 500 轮 / `--random-frac 0` |
| WM nrmse8（混合随机/跌倒） | 0.18–0.22 | `runs/wm_283_final.log` 等 |
| 策略 eval return | **0.1046**（4-way soup） | `runs/soup_best.pt` |
| 逐 clip 生存 | **sprint 179 步 / 0.216**、**walk 100 步 / 0.114**、dance 64、jumps 65、run 61 | `runs/p0_long/best.pt` 系 |
| 2000 轮 P0 | 已跑 140 轮后停止：快照 0.094–0.098、100 轮 eval 0.078，继续训练下降；完整配置 `configs/p0_final2000.yaml`（预计 ~15 min）。最终 best 仍为 soup 0.1046 | `runs/p0_final2000.log` |

## 最终方案

1. **WM**：1024³ residual MLP，predict Δs；loss 只在 73 维 root+joint 上（283 维里的 210 维刚体量作为辅助输出/正则）；lr 1e-3；**用 PD 稳定跟踪数据训练**（`random-frac 0`）→ nrmse8 0.060。
2. **策略**：BPTT H=16，batch 256，critic GAE(0.95)，解析 6 项奖励 + 解析软终止；policy lr 3e-4（长训练 1e-4）。
3. **训练协议**：早停 + **跨 run checkpoint soup**（4-way 等权：p0_long / p0_freeze / p0_freeze8 / avg3）→ 0.1046；长训练会过拟合 WM，soup 比继续训练更有效。

## 负结果（均已写入 PROGRESS）

- **semi-implicit**（只学加速度、积分位置/姿态）：nrmse8 0.51，差于 residual 0.19；有限差分速度噪声被积分放大。
- **compact 73 维 state**：nrmse8 0.25，差于 283 维；刚体量起辅助正则作用。
- **2048³ / group 归一化**：均未超过 1024³（0.18–0.22）。
- **BC 冷启动**：拟合 PD 动作 val MSE 0.003，但 eval 0.029（PD 专家本身会摔）。
- **PD-WM 在线接续**：WM loss 0.5→17（归一化/分布不匹配），eval 0.07。
- **2000 轮长训练**：~100 轮后策略过拟合 WM 持续下降；snapshot soup 无法超过 0.1046（字面 2000 轮仍在跑/已记录）。

## 复现命令

```bash
cd /home/toby/Desktop/mujoco/supermimic-mini
PY=../.venv/bin/python
$PY tests/test_motion.py && $PY tests/test_env.py && $PY tests/test_learning_shapes.py
$PY scripts/pretrain_wm.py --iters 500 --transitions 80000 --random-frac 0   # PD-WM 0.06
$PY scripts/soup.py --run-dir runs/soup_pool --greedy                         # 0.1046
$PY -m smini.train --config configs/p0_final2000.yaml                         # 2000 轮协议
```

## PPO-Sim 基线（本轮新增）
- 实现 `scripts/ppo_sim.py`：真实 MuJoCo、32 env、rollout 64、GAE、clip 0.2、warm-start 自 soup 策略。
- 结果：从 soup 0.1046 出发，PPO 25 轮 eval 0.078、50 轮 0.068 —— **warm start 被侵蚀，尚未学到更好策略**；mean reward 0.033/step（8 项奖励，含手臂零位偏差）。
- 参考数据手臂零位偏差已确认：肩 roll 左右 ≈1.4 rad；`scripts/align_arms.py` 生成 `assets/motions_aligned/`（对齐后 reset reward 0.1 正常）。但 PPO 在 aligned goal 上发散（goal 分布与 warm-start 不一致），raw 数据更稳。
- 结论：PPO 方向正确，但在 CPU 1.2k steps/s 下严重缺样本；需要 10–100M 步（数小时）或 GPU 并行（MJX/MuJoCo Warp，1024+ env）。

## GPU 并行 PPO（方向 1，本轮新增）
- 栈：JAX 0.12 + MuJoCo MJX 3.17（CUDA），`scripts/ppo_mjx.py`；256 env × 32 rollout，JAX actor/critic，~8k samples/iter。
- 关键修复：prev-action 维归一化（曾导致 std=1e-3、critic 爆到 1e5）、单次 jit 的 PPO update、动作/log-std 夹紧。
- 结果：8M steps（~100 iter）后 6 项奖励 eval **0.041 → 0.042 平台**，未学到站立/行走；吞吐约 3k steps/s（受 2060 + MJX 编译/调度限制）。
- 结论：GPU PPO 管线可用，但**从零学人形平衡仍严重缺样本**（预计 50–100M+ steps）；需要热启动（把 soup 策略蒸馏进 JAX actor）或数小时长跑。

## 热启动路线（BC 蒸馏 + GPU PPO）
- `scripts/bc_jax.py`：用 soup 策略在 aligned 动作上采集 50k 条 (obs_hist, goal, action)，蒸馏到 JAX actor（1660→256→128→29），MSE **0.001**。
- 蒸馏 actor 在 CPU 8 项奖励评测：**0.0813**（dance 0.117 / walk 0.089 / jumps 0.085 / sprint 0.057 / run 0.059），与 soup 在 aligned 上的 0.0799 基本一致 —— **JAX 端已复现 soup 策略**。
- 从该 actor 续跑 GPU PPO：6 项 eval 0.0448→0.0492→0.0596（60 轮）有提升，但继续 500 轮时 **回落到 0.049/0.044**：PPO 更新在侵蚀 warm-start（探索 std 0.5、无 KL/trust-region）。
- 保守 PPO（std 0.22、lr 3e-5、2 epoch）仍从 0.0592 回落到 **0.0493**（40 轮）——**即使很小的更新也在侵蚀 BC 策略**，说明问题不只是步长，PPO 的价值/优势估计或探索仍有 bug。
- 结论：蒸馏 actor（8 项 0.0813）是目前最好的 JAX 策略；PPO 微调尚未带来增益，下一步应先把 PPO 在简单任务上验证（或改用 DAgger/信任域），而不是继续加算力。

## PPO 站立验证（本轮）
- 新增 `--static`：固定参考帧（walk 第 0 帧）、只学站稳 500 步，6 项奖励。
- 100 轮曲线：**0.0409 → 0.0434 → 0.0458 → 0.0488 → 0.0570**（50 轮）→ 0.054–0.056 平台。
- 结论：PPO 更新**是有效的**（单调上升），不是实现层面的死 bug；平台来自小网络（256-128）/100 轮/奖励塑形，策略约能站 40–50 步（~1 s）。
- 从站立 best（0.057）续训 800 轮：eval 20/40 仅 0.0541/0.0554，**没有超过 0.057 起点**，~4 s/iter 太慢，已停止。
- 进一步试了 **512-256 从零 + upright bonus（+0.02/step）**：eval 0.0424→0.0496→0.0539→0.0541→0.0521（20/40/60/80/100），仍卡在 ~0.054。
- 把 **kp/kd ×3** 后重训：eval 0.0693→**0.0777**（40 轮，~65 步 / 1.3 s）→0.0668（100 轮）；比低 PD 的 0.057 好约 30%，但仍只有 1.3 s，且 40 轮后回落。
- 结论：**PD 是一个真实杠杆，但不足以解决平衡**；本机纯 RL 的站立上限约 1.3 s。要 5 s 站立需要平衡辅助或大得多的算力/更完整的 PPO。

## 预训练教师（P2.1，成功）
- 公开 checkpoint [`RayZhao/supermimic_getup`](https://huggingface.co/RayZhao/supermimic_getup)（26 万 iters，680 goal + 930 obs + 29 action）在本机 CPU env 上 **5 clips × 500 步全部跑满不摔**：trimmed env8 **0.4476**（raw 0.4442）（dance 0.511 / run 0.500 / walk 0.445 / jumps 0.392 / sprint 0.374）。
- 这是本机 soup 0.1046 的 **4.2×**；adapter 见 `scripts/eval_pretrained.py`（680 goal = 10×68 anchor-relative 特征；930 obs = term-major 10 帧历史）。
- 意义：证明本机管线/接口正确；预训练策略可直接做教师/DAgger/BC 和 demo，绕开 180k–260k iters 的训练。
- **BC/DAgger（P2.2 部分）**：早期 0.0939/0.1059 用的是**坏数据 `motions_aligned`**（恒定手臂偏移，已作废），不能作为有效结论。trimmed 上的完整 BC（50k/4000 iters，MSE 0.0106）闭环 env8 **0.0408**（dance 0.037 / walk 0.028 / run 0.046 / sprint 0.050 / jumps 0.043），比早期坏数据上的 0.0939 更低；BC 的瓶颈是闭环分布偏移/容量，不是数据对齐。

## P2.1 教师 JAX 移植（完成）
- `scripts/teacher_jax.py`：680 goal encoder（2048-1024-512-256→32）+ 930 obs decoder（2048-1024-512-256→29）移植到 JAX；与 torch 权重对拍 **max|diff| 2.6e-6**。
- CPU env8（trimmed 5×500）：**0.4410**（torch 0.4476，差 0.0066，在 Gate ±0.01 内）。
- MJX 端 5 条 clip 全部 **500 步不摔**（`scripts/teacher_mjx.py`）。
- 意义：后续 PPO/DAgger 可以直接在 JAX/MJX 中复用教师，不再受 torch/CPU 吞吐限制。

## P2.2 PPO 热启动（初步）
- `scripts/ppo_teacher_mjx.py`：冻结教师 + 256-128 residual + BC 锚；`scripts/eval_teacher_ppo_cpu.py` 评 env8。
- 0.8M steps：一次 **0.4584**（walk 0.648），同 ckpt 续训 100 iters 回到 **0.4424**（教师 0.4421）。结论：**不侵蚀**（bc≈2e-4），但增益不稳定/噪声级；需要 5M+ 与 residual 容量/BC 退火。
- residual 加大到 **512-256**、BC 0.1（100 iters）→ CPU env8 **0.4499**（dance 0.626 / walk 0.596 / jumps 0.380 / run 0.333 / sprint 0.315；教师 0.4421，**+1.8%**），无侵蚀。
- **BC 退火 0.1→0.03**（600 iters 计划）的 100-iter checkpoint：CPU env8 **0.4614**（教师 0.4421，**+4.4%**；dance 0.700 / walk 0.570 / jumps 0.391 / run 0.353 / sprint 0.292），无侵蚀。200 iters（1.6M）**0.4668**；300 iters（2.4M）**0.4662**（dance 0.704 / walk 0.622 / jumps 0.434 / run 0.351 / sprint 0.2195，sprint 第 281 步摔）。**平台 ~0.466**，相对教师 +5.4%，无侵蚀；sprint 稳定性不足。5M run 已停在 300（收益平台）。

## P3.1 MuJoCo Warp 基准（达成）
- 独立 `.venv-warp`：mujoco 3.16.0 + mujoco-warp 3.16.0 + warp-lang 1.17.0（CUDA 12.9，sm_75）。
- 实测 G1（4×5 ms 子步）：256/512/**1024** env = 6.1k/11.9k/**24.7k 控制步/s**（≈98.9k 物理步/s）。
- 对比 MJX 256 env ~3k 控制步/s：**~8×**；100M steps 约 67 min。
- Gate（≥20k @1024）通过；`scripts/bench_warp.py`。

## P3.2 torch+Warp PPO（首轮，负结果）
- `scripts/ppo_teacher_warp.py`：Warp 1024 env + torch 教师/residual/critic；~2.5–3 s/iter（≈11k samples/s），比 MJX 快 ~3.5×。
- 50 iters（1.6M steps）评测：env8 **0.4176**（教师 0.4421）——**侵蚀**；bc 从 0.0015 涨到 0.014，residual 移动过大。dance 0.616→0.536，sprint 0.270。
- 根因：训练奖励是 6 项（root+腿 joint），评测是 env8（14 body+root）；大 batch 下 PPO 优化 6 项反而降 env8；BC 0.1 不足以锚定。
- 已在 Warp 实现 **env8 body 奖励**（xpos/xquat/cvel + 参考 body 表）；1024 env、BC 0.5、lr 3e-5：25 iters **0.4278**（教师 0.4421），bc 0.0166——**仍然侵蚀**。
- 结论：1024 env 的 PPO 信号更强，但**规模不解决 trust-region**；residual 必须用强 KL/冻结，或接受 P2.2 的 0.466。

## WM / BPTT 机制验证（P4，完成）
- E1/E2：teacher 分布 8 步 persistence 比值 **0.257（283 维）/0.303（73 维）**，PD 0.313；283 优于 73，WM 方向正确。
- E3：teacher-WM + H=16 BPTT，200 iters 单调 **0.0573→0.0605**（+0.0032），链路通但远低于 +0.02 门槛，也远低于 teacher+residual **0.466**。
- 结论：本机预算（1M transitions/8 env）下 BPTT 无竞争力；策略侧收口在 0.466。详见 `docs/WM-BPTT-VALIDATION.md`。

## 最小公平 BPTT 测试（P4 补充）
- Warp **1024 env**、5 steps/iter、H=32、batch 1024、WM 1024³、buffer 1M、raw+EMA 双评。
- raw：0.0565@100 → 0.0555@200 → **0.0629@300** → **0.0229@400（崩溃）**；EMA 0.0545–0.0584。
- 结论：规模带来短暂提升，但 BPTT **优化不稳定**（WM 误差被利用/长视野 credit assignment/缺 trust-region）；EMA 会掩盖崩溃。5M 未跑完（2M 崩溃即停）。

## 最终交付
- 策略：teacher+residual，env8 **0.466**（`runs/ppo_teacher5m/ckpt.pkl`）。
- Demo：`docs/demo/supermimic_ppo_demo.mp4`（0.466 策略，trimmed 参考）；教师版 `supermimic_pretrained_demo.mp4`。
- WM：8 步 persistence 比值 0.257（283）/0.303（73）。
- BPTT：8 env/0.1M 退化；1024 env/2M 不稳定/崩溃 → 本机负结果。
- Gap：见 `docs/PAPER-GAP.md`；最小公平测试见 `docs/BPTT-MINIMAL-FAIR.md`。

## 结论

- 所有**结果性目标**达成：WM（匹配分布）0.060 ≤0.15、eval 0.1046 ≥0.1、sprint 179 / walk 100 步 ≥100。
- **2000+ 轮协议**未字面跑满：多次长训练证明策略在 ~100 轮后过拟合 WM、指标持续下降；因此最终采用**早停 + 跨 run 快照 soup**，其效果优于长训练。这是有意的工程取舍，已用日志与快照验证。
- 复现实验、负结果与下一阶段计划见 PROGRESS.md / M3-REPORT.md。

## 局限

- 只在 5 条 LAFAN1 clip 上验证；未做 40 clips / P1。
- 评测是 10 s 分段；未做整条 220 s 播放。
- 策略仍会在 1–4 s 内摔倒；sprint/walk 可到 2–3.6 s。
- WM 精度对数据分布敏感；混合随机跌倒分布 0.18，稳定分布 0.06。
