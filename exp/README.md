# exp/ —— 16 GB 卡上的 WM+BPTT 复现实验

结论与完整分析见 [`docs/WM-BPTT-16GB.md`](../docs/WM-BPTT-16GB.md)。这里只放可复跑的脚本。

## 硬件与依赖

本实验在 **AMD RX 6800 XT 16 GB（ROCm）** 上跑通，因此**不依赖 Warp/CUDA**：

```bash
python -m venv .venv && . .venv/bin/activate
pip install mujoco==3.17.0 "numpy>=2.0" "scipy>=1.11" "PyYAML>=6.0"
# AMD: 用官方 gfx103X 轮子（内含 ROCm SDK，勿与系统 ROCm 对齐）
pip install --index-url https://repo.amd.com/rocm/whl/gfx103X-all/ torch==2.10.0+rocm7.13.0
# NVIDIA: pip install torch==2.6.0 即可
```

G1 资产与动作数据见仓库根 README 的 *Install* / *Data* 两节。本实验把 LAFAN1 G1 全 40 条放在 `assets/motions/`。

## 脚本

| 文件 | 作用 |
|---|---|
| `collect.py` | 多进程 CPU MuJoCo 采集，输出 `S[71] / A[29] / X[34] / E` 分片 |
| `collect_holdout.py` | 采集 held-out clip（3 条）用于泛化评测 |
| `wm_train.py` | WM 训练 + held-out NSE 评测；支持 `--resume` / `--xi-dim` / `DATA_DIR` |
| `diag_sensitivity.py` | 动作捕获率 `capture = d_wm / d_true`（直接对比原报告 7.5%） |
| `grad_corr_test.py` | 动作梯度一致性（Pearson / Spearman / 符号一致率） |
| `mpc_test.py` | 用 WM 规划 H 步动作（**6 项**分析奖励），放回真值仿真器验证 |
| `diff_fk.py` | **可微前向运动学 + 完整 8 项奖励**（补齐 14 个 body link 项） |
| `test_reward8.py` | 对拍：FK vs MuJoCo、reward8 vs `TrackingEnv._reward`（期望 ~1e-7 / ~1e-8） |
| `mpc_test8.py` | 用 **8 项**奖励做规划验证（H=8/16 显著，H=32 失效） |
| `ac_bptt.py` | **完整 actor-critic**：Policy+Critic + WM71 + reward8 的 BPTT 循环 |
| `diagnostics/bench_env_throughput.py` | 单进程 vs 多进程采集吞吐 |
| `diagnostics/truth_action_sensitivity.py` | 真值仿真器的严格 1 步动作敏感度 |
| `diagnostics/action_identifiability.py` | 动作在数据中的线性可识别性（按状态块） |

## 最短复跑路径

```bash
# 1) 采集 4M transitions（10 进程，约 8 min）
V2=1 python exp/collect.py 4000000 10
V2=1 python exp/collect_holdout.py

# 2) 训练 WM：2048^3, batch 4096, fp16, 30k 步（约 55 min）
DATA_DIR=data_v2 python exp/wm_train.py \
    --steps 30000 --batch 4096 --xi-dim 34 --amp 1 --eval-every 2500 --tag v2

# 3) 三项判决测试
python exp/diag_sensitivity.py exp/runs/v2_best.pt
python exp/grad_corr_test.py  exp/runs/v2_best.pt
python exp/mpc_test.py --ckpt exp/runs/v2_best.pt --H 32 --N 32 --opt-steps 40
```

## 产物（不入库，见 .gitignore）

- `exp/data/`、`exp/data_v2/` —— 采集分片（4M + 300k，各约 1.6–2.3 GB）
- `exp/runs/` —— checkpoint（`*_best.pt` 约 35 MB、`*_last.pt` 约 70 MB）、训练日志、结果 json

## 关键实验（第四轮）

```bash
# 先验证 FK 与奖励头是否精确（不通过就不要往下跑）
python exp/test_reward8.py

# 8 项奖励的规划验证
python exp/mpc_test8.py --ckpt exp/runs/v2_best.pt --H 8  --N 48 --opt-steps 40
python exp/mpc_test8.py --ckpt exp/runs/v2_best.pt --H 16 --N 48 --opt-steps 40

# 完整 actor-critic（BC 热启动 + BPTT）
python exp/ac_bptt.py --bc-steps 2500 --iters 200 --B 64 --H 16
```

结果与解读见 [`docs/REWARD-HEAD-AND-AC.md`](../docs/REWARD-HEAD-AND-AC.md)。

## 两个已修的坑

1. **数据必须是 env-major**。多进程若按 tick-major `[t0e0..t0e7, t1e0..]` 拼接，相邻索引属于不同 env，几乎每条转移都是假的 → loss 卡 100、NSE1=113。`wm_train.py::load_shards` 做 `(T,NE,·)→(NE,T,·)` 重排并在每段末帧打边界。
2. **xi 用 34 维**。`spec.sample_xi` 的左右脚摩擦独立采样，`wm_import.WM71` 默认只有 1 个摩擦槽；`make_wm(34)` 把 xi 扩到 34 并重建 dynamics 首层。

## 关键参数备忘

| 项 | 值 |
|---|---|
| 状态 | 71 维 `[p3, q4, qj29, v3, w3, qd29]`，经 `canonical` + `explicit integration` |
| xi | 34 维 `[μL, μR, bias29, com3]`（offset 0.75/0.75，scale 0.45/0.45/0.01×29/0.05×3） |
| 损失 | k=8 自回归 `error_vector` 平方均值（err_scale 0.05/0.1/0.5/1.0/0.1/2.0） |
| 训练 | Adam 2e-4 cosine→0.03×，grad clip 1.0，EMA 0.999，fp16 AMP |
| 本机吞吐 | 采集 ~12k 控制步/s（10 进程）；WM 训练 110 ms/step（batch 4096） |
