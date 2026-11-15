# P3（Warp 迁移）设定与实现审计

> 日期：2026-11-09
> 对象：`scripts/bench_warp.py`、`scripts/ppo_teacher_warp.py`、`.venv-warp`。

## 1. 结论

1. **吞吐设定正确**：Warp 1024 env = **24.7k 控制步/s**（4×5 ms 子步，50 Hz），显存 ~0.6 GB；Gate 通过。
2. **P3.2 首轮 0.4176/0.4278/0.4309 的“侵蚀”不是纯 PPO 问题**，而是实现里有 3 个实质 bug：
   - **body 线速度错配**（Warp `cvel[3:6]` 是 com-based，参考是 body 原点有限差分）；
   - **GAE 截断处理错误**（time-limit 截断用了 reset 后状态的 V）；
   - **log_std/clamp 不一致**、`mx` typo、`njmax=8192` OOM。
3. 修复后（BC 2.0、25 iters）：**env8 0.4460**（教师 0.4421，+0.9%），bc 0.0008，**无侵蚀**。
4. 当前最佳仍是 P2.2（256 env、teacher+residual）：**0.4662/0.4668**；P3.2 修复版 0.4460 尚未超过它。

## 2. 已修复的 bug

| ID | 位置 | 问题 | 修复 | 验证 |
|---|---|---|---|---|
| B1 | `Env.step` reward | `lin = cvel[:,:,3:6]` 是 **com 线速度**；参考 `body_linvel` 是 body 原点 `np.gradient(pos)` | `lin = (xpos - prev_xpos)/0.02` | CPU 对数：cvel 角速度与参考一致；线速度改为有限差分 |
| B2 | `Env.step` / GAE | 只 mask `term`；`done` 含 time-limit 截断，截断时用 reset 后 V[t+1] bootstrap | 返回 reset 前 `final_hist/final_goal`；`term→0`、`done(trunc)→VF`、否则 `V[t+1]` | 25 iters 0.4460，无侵蚀 |
| B3 | rollout/update | `std=exp(log_std).clamp` 但 logp 用未 clamp 的 `log_std` | 统一 `lg=log_std.clamp(-3,0)`，logp 用 `lg` | 训练稳定 |
| B4 | `Env.__init__` | `mx.opt` typo（NameError 被吞） | `self.mx.opt` | — |
| B5 | `put_data` | `njmax=8192` × 1024 world → **OOM** | `nconmax/naconmax=256, njmax=512` | 1024 env 正常、无 broadphase 溢出 |

## 3. 仍存在的设定问题（未改，记录）

| ID | 问题 | 影响 | 建议 |
|---|---|---|---|
| S1 | `q_sp` 仍被裁到 `ctrlrange` | 与论文“不裁 setpoint”相反，削弱极限/恢复 authority | 加 `--no-qsp-clip` 开关，做 A/B |
| S2 | critic 输入 1610 维未归一化 | 目标/观测尺度差异大，value 噪声 | 固定 mean/std 或 LayerNorm |
| S3 | 动作 clip ±8、log_std [-3,0] | 保守；限制探索 | 与教师动作分布对齐后再放宽 |
| S4 | Warp 与 CPU env 的接触求解器不同 | 训练/评测动力学有差异 | 评测仍用 CPU；报告注明 |
| S5 | reset 未清 `qacc_warmstart`/`time` | 少量 warmstart 残留 | 复位时清零 |
| S6 | `reset` 里 `t0` 先 randint 再覆盖 | 无功能影响 | 清理 |
| S7 | BC 系数 2.0 时 residual 几乎不动（bc 0.0008） | 无侵蚀但也无提升 | 需要 KL 自适应/课程式放开 |

## 4. 数值对拍（关键）

- `cvel` 前 3 维 = 角速度，与参考 `body_angvel` **一致**（t=0/50/100 完全对齐）。
- `cvel` 后 3 维 = com 线速度，与参考 `body_linvel`（原点有限差分）**不一致**（例：t=50 cvel [0.295,-0.039,0.024] vs ref [1.262,-0.115,0.246]）——B1 修复前 body linvel 奖励项基本是错的。
- 修复后 25 iters 0.4460，说明该项影响显著。

## 5. 结论与建议

- P3.2 的实现**在修复 B1–B5 后才算可信**；之前的 0.4176/0.4278/0.4309 不应作为“PPO 侵蚀”的方法结论。
- 修复版“强 BC”能保住教师（0.446），但提升有限；P2.2 的 0.466 仍是最好。
- 下一步二选一：
  1. **KL 自适应**（β 调 KL≈0.01）+ BC 退火，在 1024 env 上尝试超过 0.466；
  2. 接受 0.466，收口 demo/报告，P4 WM 仅作机制验证。
