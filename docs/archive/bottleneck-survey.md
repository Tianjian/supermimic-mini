# 突破瓶颈的学术调研与可行路径

> 日期：2026-11-09
> 问题：为什么本机 SuperMimic-mini 的策略只能站/走 1–3.6 s，如何突破？
> 方法：检索 SuperMimic 论文/复现、BeyondMimic、mjlab、PPO 实现细节与 G1 跟踪复现；对照本机配置做量化差距分析。

## 0. 结论先行

1. **主要瓶颈是“规模”**：论文/复现是 **4096–8192 env、H=32、~100–327M transitions、40–95k 条参考**；本机是 **8 env（CPU）/256 env（MJX）、H=16、~1M transitions、5 条参考**。样本量差 **100–300×**，数据差 **600–1000×**。这解释了为什么 100–200 轮只能到 0.1 量级。
2. **有一个“免费”的结构性修复**：本机 obs 只有 93 维，**缺 root 相对参考的 9 维位姿误差**；复现实测 93 维观测在 LAFAN1 封顶 **0.56**，加 anchor-relative 9 维后到 **0.904**。这是当前最值得先做的算法改动。
3. **WM 本身也不够准**：本机 PD 分布 nrmse8 **0.060**；复现 8 步开环误差 **0.0065–0.0099**（口径需统一），且复现用 **71 维状态 + 速度增量 + 显式积分**，不是 283 维。先把 WM 做到 ≤0.02，再谈长训练。
4. **最快拿到“能站/走”的路径**：用公开的 **预训练 G1 tracker**（`RayZhao/supermimic_getup`，35.6 MB）做教师/直接评测；接口是 93 obs + 730 goal + 29 action，和本机高度相似。其次是 **mjlab/MuJoCo Warp 4096 env** 重训。

---

## 1. 关键文献与复现（带数字）

| 来源 | 方法/栈 | 规模 | 结果 | 对本机的启示 |
|---|---|---|---|---|
| [SuperMimic 论文](https://arxiv.org/abs/2610.28480)（CoRL 2026） | 可微 WM + BPTT actor-critic | 4096 env、H=32、wm 2048³、~100M transitions、1×H100 3 h | LAFAN1 40 clip **0.82**（1.2 h） | 目标规模；本机差 ~100× |
| [SuperMimic-Reproduced](https://github.com/Mr-Zqr/SuperMimic-Reproduced)（mjlab/Warp） | 同论文，71 维状态、解析奖励/观测头、torch.compile | 4096 env、H=32、40 LAFAN1、1×5090 | 论文 93 维 obs：**0.56**（1.8 h）；**加 anchor 9 维：0.904**（1.66 h）；WM 8 步误差 **0.0065–0.0099**；4096 env ≈ **2×10⁵ env-steps/s** | ① anchor obs 是最大算法杠杆；② 71 维状态更合理；③ Warp 吞吐是 MJX 的数十倍 |
| [SuperMimic_Dyn / E009](https://github.com/Baojch/SuperMimic_Dyn) | mjlab/Warp，8192 env，2 GPU | 40k iters、**327.68M transitions**、95k clips/205 h、4.09 h | 3966 validation **99.32%** 完成 | 数据/环境规模；40k iters 是“从零到可用”的量级 |
| [supermimic_getup](https://huggingface.co/RayZhao/supermimic_getup) | 同复现，180k iters 单 4090，BONES 102k clips/198 h | 93 obs + 10 帧历史 + 730 goal；29 action；50 Hz | 2000-clip **0.921**；起身 100%；提供 `policy.pt`（35.6 MB） | **可直接当教师/预训练策略**；接口与本机几乎一致 |
| [BeyondMimic](https://arxiv.org/abs/2508.08241)（arXiv 2508.08241） | PPO-Sim，body tracking + 3 penalties | 4096 env；低 PD（10 Hz，ζ=2） | 真机跑跳/冲刺/侧手翻 | 奖励/观测/动作/终止的“标准答案”；**q_sp 不裁剪**；adaptive sampling |
| [mjlab](https://github.com/mujocolab/mjlab) | Isaac Lab API + MuJoCo Warp | 4096 env 一条命令 | G1 velocity/motion imitation | 本机替换 MJX 的最现实路径 |
| [PPO 37 details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) | PPO 工程细节 | – | – | 修我们 PPO 侵蚀 warm-start 的 checklist |

### 1.1 与本机的量化差距

| 项 | 论文/复现 | 本机 | 倍数 |
|---|---|---|---|
| 并行 env | 4096–8192 | 8（CPU）/256（MJX） | **16–1024×** |
| 想象 horizon | 32 | 16 | 2× |
| WM | 2048³ / 71 维 | 1024³ / 283 维 | 容量/结构都偏 |
| wm_batch × batches | 2048×10 | 512×4 | 10× |
| policy batch | 4096 | 256–512 | 8–16× |
| transitions | 100–327M | ~1M | **100–300×** |
| 参考数据 | 40 clips / 95k clips | 5 clips / 0.32 h | **8–600×** |
| 评测 | full clip / 2000 条 | 5 clips×500 步 | 样本外未测 |

> 这张表说明：**继续在 8 env / 1M transitions 上调 reward，期望达到 0.8 是不现实的**；先把结构修对，再上规模。

---

## 2. 为什么当前策略学不会（结合审计）

1. **观测缺 anchor 误差**（audit C6）：策略看不到 root 漂移，位置/朝向奖励不可优化。复现 0.56→0.904 的差异直接证明。
2. **奖励口径不一致**（audit C1）：BPTT 优化 6 项 joint/root，评测用 8 项 body；策略没优化手臂/腕 body，评测量不出它真正学的东西。
3. **WM 状态/精度**（audit H7/H8）：283 维里 210 维 body 不被 loss 约束却进入 critic；WM 8 步误差 0.06 vs 复现 0.007–0.01，BPTT 梯度被模型误差主导。
4. **规模**（audit H4）：1M transitions vs 100M；PPO/BPTT 都严重欠采样。
5. **PPO 实现细节**：无 value clipping、无 lr 退火、无 KL early-stop、fixed obs norm、std 偏大 → warm-start 被侵蚀（warm3 0.059→0.047）。

---

## 3. 突破选项（按本机可行性排序）

### A. 预训练教师 + DAgger/BC（最快，推荐先做）
- 下载 `RayZhao/supermimic_getup` 的 `policy.pt`（35.6 MB），按其 config 加载；obs 93 + goal 730 + 29 action + 50 Hz 与本机一致，只需核对 obs 顺序/归一化。
- 在本机 env 评测：如果直接能到 **0.3–0.5+**，就有了** competent teacher**；用 DAgger 在教师访问状态上标注，BC 到小策略，再做 PPO/BPTT 微调。
- 优点：绕开 180k iters 的训练；可立即产出像样的 demo。
- 风险：接口/归一化不完全一致；模型基于 BONES 训练，在 LAFAN1 上可能降分；许可证（代码 MIT，数据 BONES 许可）需遵守。

### B. 加 anchor-relative 观测（9 维，最高性价比）
- obs：`[q_rel, qd, base_angvel, gravity, prev_action]` + `[p_ref-p, rot6d(R_ref^T R)]` → OBS_DIM 102。
- goal 不变（730）；训练/评测/BC/PPO 同步；预计跟踪奖励显著上升（复现 0.56→0.904）。
- 成本：改 spec/env/policy 输入维度 + 重训 100–200 轮；本机可做。

### C. 换 mjlab / MuJoCo Warp，4096 env
- 复现 4096 env 在 5090 约 **2×10⁵ env-steps/s**；2060 可能 2–5×10⁴；100M transitions ≈ 35–80 min。
- 6 GB 显存可能只能 1024–2048 env；先 benchmark，再决定。
- 这是把“样本量”补上的唯一现实路径。

### D. 改成 71 维状态 + H=32 + 2048³ 的“论文版” WM
- 状态：root pos+quat6D+lin/ang vel + q_rel + qd ≈ 71；body 奖励用可微 FK（复现做法）。
- WM 8 步误差目标 ≤0.02；H=32、wm_batches 10、wm_batch 2048。
- 成本：改代码 + 重训 WM；比 C 便宜，但需要正确性对拍。

### E. 修 PPO（如果继续用 PPO/BC 路线）
- value clipping、advantage norm、lr 退火、KL early-stop、obs running norm、`log_std=-1.5`、reward scaling；先在“站立 500 步”上验证单调提升，再回 tracking。

### F. 平衡辅助（仅 demo）
- 根姿态 PD 叠加到踝/髋 setpoint；能快速做出“站/走”视频，但非纯 RL，需标注。

---

## 4. 建议的执行顺序（下一步）

1. **当天可做**：加 anchor obs 9 维（B），同时去掉 q_sp 的 ctrlrange clip（audit C4），统一 aligned 数据；用现有 CPU/MJX 重训 100–200 轮，看 eval 是否从 0.1 抬到 0.2+。
2. **1–2 天**：下载预训练 teacher（A），核对接口并在本机 eval；若可用，DAgger/BC + demo。这是最可能快速看到“能站/走”的路径。
3. **3–5 天**：把 env 迁到 mjlab/Warp（C），4096（或 1024–2048）env 跑 PPO-Sim/BPTT；目标 10M–50M transitions，观察 0.3–0.6。
4. **并行**：71 维 WM + H=32（D），把 8 步误差压到 ≤0.02；否则 BPTT 仍会被模型误差主导。

## 5. 参考链接

- SuperMimic: https://arxiv.org/abs/2610.28480
- SuperMimic-Reproduced: https://github.com/Mr-Zqr/SuperMimic-Reproduced
- SuperMimic_Dyn (E009): https://github.com/Baojch/SuperMimic_Dyn
- supermimic_getup: https://huggingface.co/RayZhao/supermimic_getup
- BeyondMimic: https://arxiv.org/abs/2508.08241
- mjlab: https://github.com/mujocolab/mjlab
- PPO 37 details: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

> 注：外部数字来自各自 README/论文；本机数字来自本仓库日志（`runs/`、`docs/FINAL-REPORT.md`），口径差异已在 AUDIT.md 标注。
