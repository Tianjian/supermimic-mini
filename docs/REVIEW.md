# 实验数据与方案复盘（2026-11-09）

## 1. 可信数据（按口径）

| 对象 | 口径 | 数值 | 可信度 |
|---|---|---|---|
| 预训练教师（torch） | env8 / trimmed / 5×500 | **0.4476** | 高（公开 checkpoint，可复现） |
| 教师 JAX 移植 | 同上 | **0.4410** | 高（权重对拍 2.6e-6；差 0.0066） |
| 教师 + residual + BC（P2.2） | 同上 | **0.4668 @200 / 0.4662 @300** | 中高（趋势一致；+5.4%） |
| P2.2 早期 | 256-128/BC0.5 | 0.4584 → 0.4424 | 低（不可复现，按噪声） |
| BC flat（trimmed 50k） | env8 | **0.0408** | 高（负结果） |
| Anchor obs（256 env/100 iter） | 6 项 | 0.0411 vs 0.0421 | 中（小样本无增益） |
| Soup（raw，in-sample） | env8 | 0.1046（单模型 0.093） | 低（选择偏差；仅历史） |
| WM（PD 分布） | nrmse8 | 0.060 | 中（分布有限） |
| Warp 1024 env | 控制步/s | **24.7k** | 高（benchmark） |
| MJX 256 env | 控制步/s | ~3k | 高 |

## 2. 关键结论（复述）

1. **教师可用**：680 goal + 930 obs 是正确接口；trimmed 下 5/5 clip 500 步不摔。
2. **P2.2 有效但到平台**：+5.4%（0.442→0.466），无侵蚀；2.4M steps 后不再涨；sprint 仍不稳。
3. **BC flat 不行**：0.0408；不要再投入。
4. **anchor 在小样本无增益**：需规模或蒸馏。
5. **数据修复**：T-pose 引入段（trim）；`motions_aligned` 坏（弃用）。
6. **吞吐已解决**：Warp 1024 env 24.7k steps/s，是 MJX 的 8×。

## 3. 方案复盘（哪些继续/停止）

- ✅ 继续：教师 + residual + KL/BC（最稳）；Warp 规模；trimmed eval。
- ⏸ 暂停：BC flat；anchor 小样本；soup 选权重；8 env 结论。
- ❌ 停止：`motions_aligned`；无 KL 的 PPO warm-start；混用奖励口径。

## 4. P3.2 设计评审（torch + Warp PPO）

**方案**：Warp 跑 1024 env 物理；torch 跑教师/residual/critic；`wp.to_torch/from_torch` 交换 qpos/qvel/ctrl；沿用 P2.2 的 BC 退火 + KL。

**风险/对策**
| 风险 | 对策 |
|---|---|
| Warp/torch 流不同步 | 每控制步 `wp.synchronize()`（1024 env 下 ~0.05 ms，可接受） |
| `to_torch` 是零拷贝，读写别名 | 读取时 `.clone()`；写入用 `wp.copy(from_torch(...))` |
| 1024 env 接触溢出 | `nconmax=32, naconmax=32, njmax=128`（benchmark 已跑通） |
| 教师 2048-wide 前向 ×32k 样本 | 2060 上 ~0.5–1 s/iter，可接受 |
| 复位开销 | 只复位 done 行（mask），不整批复位 |
| 评测口径漂移 | 固定 CPU env8 / trimmed / 5×500；每 50 iter 评一次 |

**验收（Gate G3.2）**
- 10M steps（~7–10 min）：env8 **≥0.50**（成功）；或 ≥0.47 且 5/5 clip 不摔（部分成功）。
- 失败回滚：residual 加大 / BC 退火放慢 / 512 env。

## 4.1 P3.2 首轮实测（补充）
- 1024 env：~2.5–3 s/iter，≈11k samples/s；50 iters 侵蚀教师（0.4176 vs 0.4421）。
- 结论：**奖励口径必须统一到 env8**；BC 需加强；否则大 batch 会放大 6 项与 env8 的错配。

## 5. go/no-go

- **GO**：P3.2 有明确吞吐收益（8×）、教师基线稳、验收数字清晰。
- 不做：mjlab 全栈（重、收益不确定），先用 torch+Warp 混合。

## 6. 当前最佳资产

- 策略：`runs/ppo_teacher5m/ckpt.pkl`（teacher + residual，env8 0.4662）
- 教师：`runs/pretrained/policy.pt`
- 评测：`scripts/eval_teacher_ppo_cpu.py`
- Warp 基准：`scripts/bench_warp.py`
