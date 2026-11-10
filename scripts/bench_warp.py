#!/usr/bin/env python3
"""Benchmark MuJoCo Warp: batched G1 control steps (4x5ms substeps)."""
import time, numpy as np, mujoco, mujoco_warp as mjw, warp as wp
wp.init()
XML = "../assets/mujoco_menagerie/unitree_g1/scene_mjx.xml"
m = mujoco.MjModel.from_xml_path(XML); m.opt.timestep = 0.005
mx = mjw.put_model(m); d = mujoco.MjData(m)
for N in (256, 512, 1024):
    dx = mjw.put_data(m, d, nworld=N, nconmax=32, naconmax=32, njmax=128)
    dx.ctrl.assign(np.zeros((N, m.nu), np.float32))
    for _ in range(4):
        mjw.step(mx, dx)
    wp.synchronize()
    t0 = time.time(); steps = 20
    for _ in range(steps):
        for _ in range(4):
            mjw.step(mx, dx)
    wp.synchronize()
    dt = time.time() - t0
    print(f"N={N:4d}: {steps} ctrl steps in {dt:.3f}s -> {N*steps/dt:.0f} ctrl-steps/s, {N*steps*4/dt:.0f} phys-steps/s")
