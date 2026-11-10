# SuperMimic-mini 演示动画

- `supermimic_mini_demo.mp4`：50 fps，约 16 s，640×510。
  - 5 条 LAFAN1 G1 动作（Sprint / Walk / Dance / Jumps / Run），左侧为 MuJoCo 闭环策略，右侧为参考运动。
  - 每段显示 clip、step、累计 return、归一化 return；终止时标注 `terminated @ step`。
  - 片尾展示最终指标：eval 0.1046、sprint 179 步 / walk 100 步、WM nrmse8 0.060。
- `poster.png`：Walk 第 78 步的封面帧。
- 复现：`python scripts/render_demo.py`（需要 EGL 与系统 ffmpeg）。

说明：参考运动的 G1 retarget 数据在手臂上存在零位偏置（手臂外展），因此右侧参考的手臂姿态与左侧策略差异较大；策略主要跟踪 root/腿部，这也是 return 偏低的原因之一。

## 预训练策略演示（2026-11-09 新增）

- `supermimic_pretrained_demo.mp4`：公开 SuperMimic getup checkpoint（26 万 iters）在本机 CPU MuJoCo 上的 5 条 LAFAN1 clip，每条 150 步（3 s，已裁剪 T-pose 引入段），全部不摔。
- 指标：5 clips × 500 步全跑满，trimmed env8 **0.4476**（raw 0.4442）（dance 0.511 / run 0.500 / walk 0.445 / jumps 0.392 / sprint 0.374）。
- Adapter：`scripts/eval_pretrained.py --motion-dir assets/motions_trim`（680 goal = 10×68 anchor-relative；930 obs = term-major 10 帧历史）；渲染：`scripts/render_pretrained.py`。
- 这是“教师”口径，不是本机从零训练的策略；下一步 P2.2 用 DAgger/BC 蒸馏到本机策略。

数据说明（2026-11-09 修正）：
- 早期 `assets/motions_aligned/` 用第 0 帧常量减手臂偏移，是**坏数据**（f150 会反向外展），已弃用。
- LAFAN1 片段开头有 3–6.5 s 的 T-pose/站立引入段；demo/eval 从第 0 帧开始会显示 T-pose。`scripts/trim_motions.py` 生成 `assets/motions_trim/`，本 demo 用 trimmed 参考。

## 0.466 策略演示（2026-11-09）

- `supermimic_ppo_demo.mp4`：teacher（260k iters）+ residual（P2.2，env8 **0.466**）在本机 CPU MuJoCo 的 5 条 trimmed LAFAN1 clip，每条 150 步。
- 渲染：`scripts/render_ppo_policy.py`；权重：`runs/ppo_teacher5m/ckpt.pkl`。

## 长版演示（2026-11-10）

- `supermimic_ppo_demo_long.mp4`：teacher + residual（env8 **0.466**）在 5 条 trimmed LAFAN1 clip 上各 400 步，约 40 s；渲染脚本 `scripts/render_ppo_policy_long.py`。
- 标准版 `supermimic_ppo_demo.mp4`（150 步/条，19.6 s）保留用于快速查看。
