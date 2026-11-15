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

## 阶段三：MPC vs 教师（2026-11-15 新增）

三格同步对比（参考 / 教师 / MPC），150 控制步：

| 视频 | clip | 教师 | MPC | 相对 |
|---|---|---|---|---|
| `mpc/mpc_vs_teacher_run1_subject2.mp4` | run1 | 0.5232 | **0.5874** | **+12.3%** |
| `mpc/mpc_vs_teacher_sprint1_subject2.mp4` | sprint1 | 0.6065 | 0.5934 | −2.2% |

说明：150 步内 sprint 只差 −2.2%，但回退随回合**累积**——500 步时到 **−17.8%**。
这正是"规划器的效应随超参与回合长度变化"的直接可视化，与 §阶段三的结论一致。

渲染：`MUJOCO_GL=glfw python exp/render_mpc.py --clip run1_subject2 --steps 150 --opt-steps 10`

## README 内嵌动图（2026-11-15）

- `phase1_wmbptt.gif`（677 KB）：阶段一（CPU MuJoCo + RTX 2060）从零训练的 WM+BPTT 策略跟随 Sprint，右侧为参考动作。
- `phase3_mpc.gif`（505 KB）：阶段三 MPC 与教师在 run1 上的三格同步对比（前 3 秒）。
- `phase2_compare.gif`（762 KB）：阶段二五格对比——参考 / 教师 / BC / BPTT+信赖域 / PD 同 clip 同步，左上角实时分数与摔倒步数。

均由对应 mp4 压制：`ffmpeg -i <in>.mp4 -vf "fps=10,scale=420:-1,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse" <out>.gif`。

## 修正协议下的策略对比视频（2026-11-14 新增）

`compare/compare_walk1_subject1.mp4` 与 `compare/compare_dance1_subject1.mp4`
（1500×300，25 fps，各 300 控制步 = 6 s，`:300` 五格并排）。

五格从左到右：**REF**（参考动作的运动学播放）、**TEACHER**（公开权重）、
**BC**（教师蒸馏）、**BPTT+TR**（BPTT + 信赖域）、**PD**（按参考位姿直接给 setpoint）。
每格左上角显示当前累计归一化分（固定分母 `0.1×已走步数`）与被终止时的 `FALL@N`。

单 clip 实测（同起点、同 clip）：

| clip | TEACHER | BC | BPTT+TR | PD |
|---|---|---|---|---|
| walk1_subject1 | 0.516（跑满 300） | 0.179（94 步） | 0.115（48 步） | 0.085（42 步） |
| dance1_subject1 | 0.698（跑满 300） | 0.136（61 步） | **0.166（82 步）** | 0.112（50 步） |

**看图能直接得到三个结论**（和 6-clip 平均的数字一致，但更直白）：

1. **教师是唯一能跑完 300 步的**。其余三个都在 1–2 秒内摔倒——这正是修正协议揭示的"旧协议里靠站着不动拿分"的真相。
2. **BC 蒸馏学到的动作幅度不够**：它比 PD 活得久（94 vs 42 步），但远达不到教师。
3. **BPTT 的效果逐 clip 不一致**：dance 上它比 BC 活得更久（82 vs 61 步、0.166 vs 0.136），walk 上却更差（48 vs 94 步）。与 6-clip 平均 −26.6% 并不矛盾——**平均负、个别正**，说明 BPTT 在当前 WM 精度下主要是在拟合模型误差，而不是稳定地改进控制。

渲染：`MUJOCO_GL=glfw python exp/render_compare.py --clip <name> --steps 300`。
（本机 EGL/OSMesa 不可用，需 `glfw`；视频编码用 `imageio-ffmpeg`。）
