# Third-party notices

This repository contains original code and documentation. It does **not**
redistribute motion data or model checkpoints.

- **SuperMimic** method: Q. Liao, Y. Shao, J. Long, K. Zakka, R. Horowitz,
  P. Abbeel, K. Sreenath, *SuperMimic: Scaling Humanoid Whole-Body Control with
  One Differentiable World Model*, CoRL 2026, arXiv:2610.28480.
- **SuperMimic-Reproduced** (MIT): https://github.com/CuboiLeo/SuperMimic-Reproduced
  — the 71-dim world-model architecture/port in `smini/wm_import.py` follows its
  design; its released checkpoint is **not** included.
- **LAFAN1** motion data (Ubisoft La Forge) is licensed **CC BY-NC-ND 4.0**;
  the G1 retargets are from
  https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset.
  Data is not included; download and convert it yourself.
- **Unitree G1** MJCF/assets from `mujoco_menagerie` (BSD-3-Clause); not included.
- The public teacher checkpoint used for the reported 0.466 result
  (`RayZhao/supermimic_getup`) is third-party and is not included.

If you use these results, cite the SuperMimic paper and this repository.
