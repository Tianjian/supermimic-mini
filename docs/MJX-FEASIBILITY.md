# MJX 作为“精确可微 WM”的可行性（2026-11-10）

## 测试

- 模型：`scene_mjx.xml`（G1），`mjx.put_model` / `mjx.step`，2 步 rollout。
- **forward**：正常（qpos/qvel 变化合理）。
- **reverse-mode** `jax.grad`：失败
  `ValueError: Reverse-mode differentiation does not work for lax.while_loop ...`
  （solver 内部动态 while_loop）。
- **forward-mode**：
  - `jax.jvp`：成功（方向导数非零）；
  - `jax.jacfwd`：成功，输出 `(3,29)`，梯度非零。

## 含义

- MJX **不能直接做 reverse-mode BPTT**；可做 **forward-mode**（`jvp`/`jacfwd`），代价 ~29×（对 29 维动作逐个方向）。
- 替代：
  1. **有限差分**（29 次扰动/步）——与 forward-mode 同量级；
  2. 短视野 H=4、小 batch（16–32）的 `jacfwd` 诊断；
  3. 用 **MJX 的解析/有限差分雅可比**做 policy gradient（非 BPTT）。

## 建议的最小对照

- H=4、batch 32、29 动作、`jacfwd` 标量 loss（或有限差分）；
- 策略：`smini.Policy` 93/730；从零/BC 初始化；
- 评测：CPU env8；
- 目的：在**无模型误差**下看策略能否提升；若仍不升 → credit assignment/critic，而非 WM。
- 预计：H=4/batch 32 的 jacfwd 每次 ~29 次前向，2060 上可行但慢；先跑 20–50 次更新 smoke。

## 实测补充（2026-11-10）

- 单样本 2 步 forward、`jax.jvp`、`jax.jacfwd` 均可跑（jvp/jacfwd 成功，梯度非零）。
- 尝试 **29 方向 vmap 的动作敏感度/梯度**：`jax.jit + vmap` 在 RTX 2060 上编译/执行 >10 min 未完成（JAX CUDA，但 MJX solver + 29 方向展开过重）。
- H=4/batch32 的 forward-mode BPTT 会在此基础上再放大，**本机不实用**。
- 结论：MJX 只能做**单轨迹短步**诊断，不足以承担 H=4 以上的 policy gradient 对照；不再继续。
- 已有替代证据：WM 动作敏感度仅 **7.5%** 自然步进变化（§13），足以解释 BPTT 失效；无需 MJX 才能下结论。
