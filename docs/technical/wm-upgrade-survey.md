# WM 升级调研：导入 vs 重训

> 日期：2026-11-10
> 现状：本机 WM（283 维、1024³、80k transitions）teacher/PD 分布 8 步 **std≈0.30 / persistence 比值 0.257–0.303**；论文口径 0.007–0.011。BPTT 在 WM 误差下要么冻结要么崩溃，WM 是首要瓶颈。

## 1. 外部调研结论

### 1.1 有靠谱的可导入 WM
- **SuperMimic-Reproduced**（MIT，独立复现，torch）：[GitHub](https://github.com/CuboiLeo/SuperMimic-Reproduced)。
  - WM：`state=71`（qpos36+qvel35），heading-canonical 特征，MLP `2048³`，**explicit integration**，输出 70 维增量；k=8 自回归损失；学习式终止头；经 FK 的解析 reward/obs 头。
  - 报告 8 步开环归一化误差 **0.0065–0.0099**（论文 0.007–0.011）。
  - **223 MB release checkpoint 同时包含 world model / policy / critic / optimizer**（`e8_boneseed_ckpt_0100000.pt`，BONES-SEED 全量 10 万 iter）。
  - 栈：Python 3.12、torch 2.14.1+cu130、mjlab、warp 1.18、RTX 5090。
- 其他：`zita-ch/supermimic_vibed`（Isaac Lab，1.5 GPU·h）、`awesome-supermimic`（mjlab/LocoMuJoCo/ONNX 变体）；官方未放代码。
- 通用 WM（TD-MPC2/DreamerV3）没有 G1 人形先验，不适合直接替换。

### 1.2 直接导入实测（本机）
已下载 checkpoint，移植 `StateSpec71 + DynamicsMLP + TermHead`（`smini/wm_import.py`），在本地 teacher/PD 数据上测 8 步误差：

| 步骤 | 1-step NSE | 8-step NSE | 结论 |
|---|---|---|---|
| 直接加载（我们的 state/action） | 1.03 | 9.98 | 不可用 |
| 修正 root ang 到 body 系 | 1.03 | 9.98 | 无改善 |
| 动作换算到 mjlab `G1_ACTION_SCALE` | **0.367** | **7.47** | 改善 3×，仍差 30× |

- 误差集中在 **root ang vel（2.35σ）** 和 **joint vel（0.98σ）**，位置/姿态误差很小 → 主要是**动作/速度约定**，不是动力学完全不对。
- 动作尺度差异巨大：我们的 `ACTION_SCALE` vs mjlab：hip 0.293 vs 0.549、肩/肘 0.083 vs 0.439（5×）、踝 0.625 vs 0.439、腕 0.0625 vs 0.0745。
- 还可能存在 **actuator 顺序 ≠ joint 顺序**（mjlab `G1_ACTUATION` 按 5020/7520_14/7520_22/4010/waist/ankle 分组），仅按名字缩放不足以对齐。

**结论**：checkpoint 的 WM 在其 mjlab 栈里是可靠的；**跨到我们的 MuJoCo/menagerie 栈不是 drop-in**，需要完整复刻 mjlab 的 state/action 顺序、scale、默认姿态、xi，或直接跑它的 mjlab 管线。本机（torch 2.6/warp 1.17/2060）不适合直接跑其栈。

## 2. 推荐路线（按性价比）

### 路线 A（推荐）：移植**架构**，在本机数据上重训 71 维 WM
- 状态：`[root_pos3, quat4, qpos_j29, lin3, ang_body3, qd29]`（71）；动作 29（我们的 scale/顺序，无跨栈问题）。
- 网络：canonical 特征 + MLP 2048³ + explicit integration；输出 70 增量；k=8 自回归；err_scale 用其 `(0.05,0.1,0.5,1.0,0.1,2.0)`。
- 数据：teacher/PD + 噪声 0.1，**5–10M transitions**（当前 80k 的 60–120×）；包含跌倒/恢复；40 clips 更佳。
- 训练：batch 2048、4–10 batches/update、lr 2e-4、cosine；2060 显存可跑 2048³（权重 ~100 MB）。
- 目标：8 步 **NSE ≤0.05**（sqrt ≤0.22），先达到论文同量级；再接入 BPTT。
- 预期：架构（canonical/显式积分/2048³/k=8）比我们 1024³/283 维更稳；数据量是主要变量。

### 路线 B：完整复刻 mjlab 管线
- 用其 `src/supermimic/` 全栈（Python 3.12、torch 2.14、warp 1.18、mjlab）跑 LAFAN1/自有数据；可直接用 checkpoint 的 WM。
- 代价：需 CUDA 13/warp 1.18/5090 级环境；本机 2060 6GB 可能不足（其训练 4096 env/2048³）。

### 路线 C：用 MJX 作为“精确可微 WM”做机制对照
- `mujoco-mjx` 可微，H=8–16、batch 64–256；无模型误差，但接触非光滑、慢。
- 用途：判断“BPTT 失败是 WM 误差还是 credit assignment”；不是长期 WM。

### 路线 D：通用 WM（TD-MPC2/Dreamer）
- 需大量适配、无 G1 先验；不推荐在本机走。

## 3. 落地计划（路线 A）

1. **数据**：Warp 1024 env 采集 teacher/PD + 0.1 噪声，5M transitions（~20 min）；存 71 维 state、action、done。
2. **WM71 训练**：移植 state spec/dynamics（可直接复用 `smini/wm_import.py` 的 StateSpec）；2048³；k=8；batch 2048；2e-4；1–2 h。
3. **评测**：teacher/PD held-out，1/8/32 NSE + std；目标 8 步 ≤0.05。
4. **BPTT**：WM71 + policy 93/730 + critic71；1024 env、H=32、batch 1024、trust-region；raw/EMA。
5. **判定**：WM 达标后 BPTT 仍不升 → credit assignment/critic 问题；WM 不达标 → 数据/架构继续。

## 4. 结论

- **有靠谱 WM 可导入**（SuperMimic-Reproduced checkpoint，8 步 0.0065–0.0099），但**跨栈不可直接用**（动作 scale/顺序差异，实测 1 步 1.03，换算后 0.367）。
- **最现实的是移植其架构 + 本机重训 71 维 WM**，而不是硬导权重；目标 8 步 NSE ≤0.05。
- 若只做机制验证，**MJX 精确模型**是更快的对照。

## 5. 复查（2026-11-10，第二轮）

### 5.1 已验证正确的部分
- **283→71 状态转换可逆**：用 `build_model.state_from_data` 生成 283，转 71 与直接 `[qpos|qvel]` 对比，最大误差 **8.5e-8**（pos/quat/qj/lin/ang_body/qd 全部对齐）。
- 关节顺序与 mjlab `g1.xml` **完全一致**；默认姿态 `KNEES_BENT` 完全一致；`G1_ACTION_SCALE` 公式一致。
- `xi` 归一化已补齐（默认 raw：摩擦 0.75、encoder 0、com 0 → 归一化 0）。

### 5.2 导入 WM 的误差（teacher 数据）

| 处理 | 1-step NSE | 8-step NSE |
|---|---|---|
| 直接加载（原动作/顺序） | 1.03 | 9.98 |
| root ang 修正到 body | 1.03 | 9.98 |
| 动作按 mjlab scale 换算 | **0.367** | 7.47 |
| + 候选 actuator 顺序置换 | 4.63 | 18.6 |
| **零动作（只测状态）** | **1.25** | 9.61 |

- 误差集中在 **root ang vel（~2–3σ）/ joint vel（~1–3σ）**，位置/姿态很小。
- **零动作也差** → 不只是动作 scale/顺序；更可能是 **BONES-SEED vs 本机 LAFAN1 的数据分布/xi/特征约定**，或 checkpoint 与我们的 task 不完全同分布。
- 结论：**该 checkpoint 不能直接导入本机栈**；继续抠转换的边际收益低。

### 5.3 对“导入”路线的最终判断
- 导入权重不可行（跨 harness + 数据分布），除非完整复刻其 mjlab 环境（Python 3.12/torch2.14/warp1.18/5090）。
- 值得保留的是**架构**（71 维 canonical + explicit integration + 2048³ + k=8）和其**训练配方**。

## 6. 细化落地计划（路线 A，可执行）

### 阶段 A1：数据（~20–30 min）
- Warp 1024 env，trimmed 5 clips（后续 40），teacher/PD + 动作噪声 0.1，**5M transitions**。
- 每条 transition 存：
  - `state71 = [qpos36 | qvel35]`（root pos3, quat4, qj29, lin3, ang_body3, qd29）；
  - `action29`（本机 scale，不换算）；`xi33 = [0.75, 0×32]`（本机默认）；`done`；`goal730`（给 BPTT 用）。
- 数据分 train/held-out（按 clip 或 episode），held-out 至少 10%。

### 阶段 A2：WM71 训练（~1–2 h）
- 模型：`DynamicsMLP(71+29+33 → 2048³ → 70)`，zero-last；`StateSpec71`（canonical + explicit integration；err/out/in scale 用其默认）。
- 损失：k=8 自回归 NSE + 终止 BCE（可选，权重 1.0）；batch **2048**；每 iter 4–10 个 batch；lr **2e-4**；cosine；grad clip 1.0；**50k–100k updates**（5M 数据约 10–20 epoch）。
- 显存：2048³ 权重 ~100 MB，batch 2048 激活可控（2060 6 GB；若 OOM 降 batch 1024 / hidden 1024）。
- 评测：held-out 1/8/32 步 **NSE** 与 `sqrt(NSE)`；目标 **8 步 NSE ≤0.05**（sqrt ≤0.22），先复现“比 persistence 好 10×”。

### 阶段 A3：门禁
- 通过：8 步 NSE ≤0.05 且 32 步不发散 → 进入 BPTT。
- 失败：先加数据到 10M / 40 clips；仍失败则用 **MJX 精确模型**做 BPTT 对照，判断是 WM 还是 credit assignment。

### 阶段 A4：BPTT（WM71）
- policy 93/730（论文口径）、critic 71、H=32、batch 1024；1024 Warp env；trust-region/KL；raw+EMA；5M transitions。
- 门禁：raw 不崩（≥0.8×init）且 ≥init+0.02；否则记录“WM 达标但 BPTT 仍不升”。

### 关键参数自洽性
- 5M / batch 2048 / 50k updates ≈ 100M sample-passes（合理）；
- 71 维 + 2048³ 与论文同量级；本机 2060 显存是唯一约束（可降 hidden 1024 做消融）；
- 数据 xi 固定默认（不做 DR），避免再引入分布差；
- 评测 held-out，避免 in-sample 虚高。

### 预算
- A1 0.5 h；A2 1–2 h；A3 0.2 h；A4 0.5 h。总计 ~3–4 h（不含 40 clips 扩展）。

## 7. 路线 A 实测（2026-11-10）

### A1 数据
- Warp 1024 env，PD + 0.1 噪声，`state71=[qpos|qvel]`、action（本机 scale）、xi 默认（raw 摩擦 0.75）。
- 5M 目标：实际跑到 **800×5×1024 ≈ 4.1M transitions**（977 未跑完）。
- 每 5 个 episode 留 1 个 held-out；但 held-out 缓冲随训练**流式增长**（评测集漂移，见下）。

### A2 WM71
- `71+29+33 → 2048³ → 70`，canonical + explicit integration，k=8 自回归 NSE；batch 1024，4 updates/iter，lr 2e-4。
- 训练 loss：0.40 → **0.12**（800 iters）。

| iter（data） | NSE1 | NSE8 | sqrt(NSE8) |
|---|---|---|---|
| 200（1.0M） | 0.0897 | 0.4333 | 0.658 |
| 400（2.0M） | 0.0808 | 0.3392 | 0.582 |
| **600（3.1M）** | **0.0622** | **0.1662** | **0.408** |
| 800（4.1M） | 0.2923 | 0.2866 | 0.535 |

- **最好 NSE8=0.166（sqrt 0.41）**，远未达 0.05；800 轮反而变差（loss 仍降）→ **过拟合/评测漂移**。
- 评测集是流式 held-out（每个 eval 用当时累积的 20% episodes），**不同 iter 的 NSE 不可严格比较**；这是评测设计缺陷，下一轮必须固定 held-out。
- 结论：71 维架构 + 4M 数据在本机**未达到目标精度**；需要固定评测、更多/更广数据（40 clips、10M）、更强正则/更小模型或更长训练。

### 与 283 旧 WM 对比
- 旧 283/1024³/80k：8 步 std≈0.30（sqrt≈0.30），本 71/2048³/4M 最好 sqrt 0.41 → **并未更好**（口径/指标需统一）。
- 说明“架构 + 数据量”不是简单线性；需要固定评测后重比。

### 下一步（修正后）
1. **固定 held-out**（先采 20 个 episode，训练只用其余）；
2. 数据扩到 **40 clips / 10M**，加 falls/恢复；
3. 训练 **更多 updates（10/iter）**，加 term head、weight decay/EMA；
4. 若仍 >0.1：改用 **MJX 精确模型**做 BPTT 对照，或回到 283 旧 WM 但先统一评测口径。

## 8. 固定 held-out 复查（2026-11-10）

为避免“流式 held-out”漂移，用**独立 seed=999、64 env、1500 步**重新采集 held-out（episode 内取样，1424 episodes）：

| 指标 | 训练期 1/5 split（流式） | **独立固定 held-out** |
|---|---|---|
| NSE1 | 0.062–0.090 | **0.327** |
| NSE8 | **0.166**（最好） | **0.884** |
| sqrt(NSE8) | 0.408 | **0.940** |

结论：
- 训练期 eval 明显**偏乐观**；独立 held-out 的 8 步 sqrt 0.94，远未达 0.05。
- WM 在 4M PD 数据上**过拟合**：训练 loss 0.11，但新轨迹 1 步 NSE 0.33。
- 训练期“1/5 episode split”仍与训练同分布/同轨迹族，不能代表泛化。
- **不进入 BPTT**；路线 A 第一轮失败。

修正方向（下一轮）：
1. 数据：**40 clips / 10M**，加入 falls/恢复/随机动作，覆盖更广状态；
2. 正则：weight decay、EMA、dropout/较小 hidden（1024³）、early stop；
3. 评测：**固定 held-out（独立采集）**，训练期只做监控；
4. 若仍过拟合：用 **MJX 精确模型**做 BPTT 对照；或接受 WM 只能做短视野（H≤4）。

## 9. 本轮（路线 A 首轮）设计与结果 review

### 9.1 结果（两个独立 held-out seed）

| 指标 | 训练期 1/5 split（流式） | 固定 held-out seed 999 | 固定 held-out seed 12345 |
|---|---|---|---|
| NSE1 | 0.062–0.090 | 0.327 | 0.360 |
| NSE8 | **0.166** | **0.884** | **0.849** |
| sqrt(NSE8) | 0.408 | 0.940 | 0.921 |

- 两个独立 seed 一致（0.849–0.884），说明**训练期 0.166 不是泛化估计**，而是同族 held-out 的乐观值。
- 8 步 sqrt ~0.92，距离目标 0.05 差 ~18×；**本轮 WM 不可用于 BPTT**。

### 9.2 设计问题（按影响排序）

1. **评测 split 不独立**：训练期 evalbuf 只是“每 5 个 episode 留 1 个”，与训练同 clips/同 PD 分布/同 env seed 族；不能代表新轨迹。固定 seed 一换，NSE 从 0.166 跳到 0.85。
2. **无 clip 级泛化**：5 个 clip 全参与训练；没有“训练 4 条、测试 1 条”的协议。
3. **数据窄**：只有 PD + 0.1 噪声；没有 falls/恢复/teacher/随机动作；4M 数据但 buffer 1M，训练只见到最近 1M。
4. **无正则/调度**：无 weight decay、EMA、dropout、lr decay、early stop；2048³ 在 4M 上过拟合（train loss 0.11，held-out NSE1 0.36）。
5. **指标口径**：训练 loss 是 k=8 平均 NSE；训练期 eval 也只 8 步；固定 eval 32 步但只报前 8；虽可比，但训练期 eval 的 episode 族偏差更大。
6. **固定 eval 的窗口仍来自 PD 轨迹**，不是“任意状态”；若 BPTT 策略偏离 PD，WM 误差会更大。

### 9.3 结论修正

- **路线 A 首轮失败**：唯一可信的 8 步 NSE ≈0.85（sqrt 0.92），不是 0.166。
- 失败原因不是“71 维架构”本身，而是**评测不独立 + 数据窄 + 无正则**的综合；但当前结果不足以证明架构优劣。
- **不进入 BPTT**；继续前必须先修评测与数据。

### 9.4 下一轮硬性要求

1. **评测**：固定 held-out（独立 seed + **留出 1 个 clip**），训练期只监控；保存每次 eval 的 checkpoint。
2. **数据**：40 clips / 10M，含 falls、恢复、teacher/随机动作；buffer ≥2M。
3. **正则**：weight decay 1e-4、EMA、dropout 0.1、lr cosine、early stop；hidden 1024/2048 消融。
4. **门禁**：held-out 8 步 NSE ≤0.05 才进 BPTT；否则先 MJX 对照。

## 10. 修正评测与数据后的结果（2026-11-10）

### 修正内容
- **评测**：留出 **walk1（clip 4）**，训练仅用 dance/jumps/run/sprint（0–3）；固定独立 seed 12345、64 env、1500 步、确定性 PD；1672 个 held-out episodes。
- **数据**：2% 随机动作（覆盖 falls/恢复）+ 0.1 噪声；buffer **2M**；训练期不再用同族 1/5 split 作为结论。
- **正则/调度**：weight decay 1e-4、EMA 0.999、cosine lr（到 0.03×）、early stop、每 100 iter 固定评测。

### 结果（held-out walk clip）

| iter | loss | raw NSE8 | sqrt | EMA NSE8 |
|---|---|---|---|---|
| 100 | 0.514 | 1.291 | 1.136 | 0.975 |
| 200 | 0.459 | 0.940 | 0.970 | **0.863** |

- 训练 loss 下降（0.51→0.46），但 **held-out NSE8 仍 0.86–0.94**，距 0.05 约 17×。
- 留出 clip 的泛化失败；此前“0.166”确认是评测不独立造成的假象。
- 结论：**修正评测后，路线 A 仍然失败**；71 维 WM + 4 clips/2M 不足以泛化到新 clip。

### 判断
- 不是“训练不够”这一条能解释：loss 在降、held-out 不动 → **数据覆盖/架构容量/正则**的组合问题。
- 下一步要么 **40 clips/10M + teacher 数据**，要么 **MJX 精确模型**做 BPTT；在本机 5 clips 上继续加训练轮数收益很低。

## 11. 扩展数据探索（2026-11-10）

### 数据
- 下载 LAFAN1 G1 全量 **40 clips**（HF `lvhaidong/LAFAN1_Retargeting_Dataset/g1`），转 50 Hz npz：`assets/motions40`（2.45 h），trim T-pose 后 `assets/motions40_trim`（**2.35 h**）。
- 类别：dance 8、walk 12、run 4、sprint 2、jumps 3、fight 3、fightAndSports 2、**fallAndGetUp 6**；时长 102 s–273 s。
- 留出 **walk4_subject1（index 39，0.039 h）**，训练其余 39 条；固定独立 seed 评测（1595 episodes）。

### 训练（39 clips，held-out walk4）

| iter | loss | raw NSE8 | sqrt | EMA NSE8 |
|---|---|---|---|---|
| 100 | 0.500 | 1.018 | 1.009 | 1.370 |
| 200 | 0.371 | **0.841** | **0.917** | 1.152 |

- 相比 4-clip/5-clip 的 0.94（200 轮），扩到 39 clips **略有改善（0.94→0.84）**，但仍距 0.05 约 17×。
- EMA 在 200 轮仍落后（1.15），不能作为早期结论。
- 训练 loss 0.66→0.37，held-out 下降慢 → **数据覆盖不是唯一瓶颈**；还需要更长训练、架构/正则或 teacher 数据。

### 判断
- 扩展数据**方向正确但不足以达标**；本机 5→40 clips 的收益约 10%，远小于 17× 的精度缺口。
- 下一步优先级：
  1. 同一 39/1 split 上跑 **10M + 1000+ iters**，看 held-out 是否继续下降；
  2. 加 **teacher 动作/状态** 或更大 hidden/term head；
  3. 仍不行 → **MJX 精确模型**做 BPTT 对照，停止 WM 训练。

## 12. 扩展数据长跑（2026-11-10）

- 同 split（39 train / walk4 held-out），`iters=1000`（目标 5M），其余同 §11。
- 200 轮：held-out **raw NSE8 0.877（sqrt 0.937）**，EMA 1.102；与 600-iter 版本 0.841 基本同级。
- 训练 loss 0.69→0.48（100）→0.33（200），但 held-out 下降缓慢；到 1000 轮预计仍 >0.4。
- 结论与 §11 一致：**数据从 5→40 clips 的收益约 10%，不足以接近 0.05**；EMA 早期不可用。
- 不再在本机继续加 WM 训练轮数；转 **MJX 精确模型**做 BPTT 对照，或引入 teacher 数据/改架构。

## 13. 动作敏感度 + 高探索数据（2026-11-10）

### 动作敏感度（39-clip WM）
- 固定状态，扰动动作 0.1/0.5/1.0，下一状态变化均值 **0.0041 / 0.0203 / 0.0406**；
- 自然一步状态变化 **0.543**；
- 即动作 1.0 只造成 ~7.5% 的自然变化 → **WM 基本忽略动作**。这直接解释 BPTT 失效：策略改动作，WM 几乎不动，梯度无意义。
- 原因：PD 数据中 action 是 state/reference 的确定函数（+0.1 噪声），动作独立变化太少。

### 高探索数据
- PD + 0.5 噪声 + 10% random±3，39 clips，300 iters：
  - loss 2.08→1.29；100 轮 held-out **NSE8 1.283（sqrt 1.133）**，比 PD 数据 1.02 更差。
- 探索提高动作覆盖，但状态预测更难；100 轮内没有收益。
- 结论：**单纯加探索不是解**；需要“动作条件化”的结构/数据比例/更长训练，或换路线。

### 下一步判断
- 本机 WM 路线（71 维 MLP + PD/探索数据）已连续失败：held-out 0.84–1.28，动作敏感度低。
- 停止 WM 训练；转 **MJX 对照**（无模型误差，判断 credit assignment）或接受 **0.466** 收口。
