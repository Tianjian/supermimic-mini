#!/usr/bin/env python3
"""Render a policy-vs-reference demo video (MuJoCo EGL, 50 fps)."""
from __future__ import annotations
import os, shutil, subprocess, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import jax, jax.numpy as jnp
import mujoco
import torch
from PIL import Image, ImageDraw, ImageFont
from smini import motion, spec
from smini.env import TrackingEnv, EnvCfg
sys.path.insert(0, str(ROOT / "scripts"))
from eval_pretrained import Pol, goal680, obs930
from ppo_teacher_mjx import apply_scales, init_mlp
import pickle

XML = str(ROOT.parent / "assets/mujoco_menagerie/unitree_g1/scene_mjx.xml")
OUT = ROOT / "docs/demo"
FR = OUT / "frames"
W, H = 640, 510
PW, PH = 320, 400
HEADER = 56
FOOT = H - HEADER - PH
FPS = 50
CLIPS = [("sprint1_subject2", "Sprint", 400), ("walk1_subject1", "Walk", 400),
         ("dance1_subject1", "Dance", 400), ("jumps1_subject1", "Jumps", 400),
         ("run1_subject2", "Run", 400)]


def font(size, bold=False):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else \
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(p, size)


F_TITLE, F_SUB, F_LABEL, F_BODY = font(20, True), font(12), font(13, True), font(13)
F_BIG, F_BIG2 = font(34, True), font(16)


def tint_reference(model):
    m = mujoco.MjModel.from_xml_path(XML)
    for g in range(m.ngeom):
        name = m.geom(g).name
        if name == "floor":
            m.geom_rgba[g] = [0.16, 0.18, 0.22, 1.0]
        else:
            m.geom_rgba[g] = [0.25, 0.45, 0.95, 1.0]
    return m


def card(lines, path, frames=30):
    im = Image.new("RGB", (W, H), (16, 18, 24))
    d = ImageDraw.Draw(im)
    y = H // 2 - 60
    for text, f, col in lines:
        w = d.textlength(text, font=f)
        d.text(((W - w) / 2, y), text, font=f, fill=col)
        y += 42
    for _ in range(frames):
        im.save(path())


def main():
    shutil.rmtree(FR, ignore_errors=True); FR.mkdir(parents=True)
    lib = motion.MotionLibrary.from_dir(ROOT / "assets/motions_trim")
    env = TrackingEnv(XML, lib, EnvCfg(obs_noise=False, reset_noise=False,
                                       domain_randomization=False, action_noise=0), seed=0)
    ref_model = tint_reference(XML)
    ref_data = mujoco.MjData(ref_model)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 135.0, -12.0, 2.4
    rend_p = mujoco.Renderer(env.model, PH, PW)
    rend_r = mujoco.Renderer(ref_model, PH, PW)

    ck = torch.load(ROOT / "runs/pretrained/policy.pt", map_location="cpu", weights_only=False)
    pol = Pol(); pol.load_state_dict(ck["policy"]); pol.eval()
    with open(ROOT / "runs/ppo_teacher5m/ckpt.pkl", "rb") as f:
        res_params = [(np.asarray(W), np.asarray(b)) for W, b in pickle.load(f)]
    res = init_mlp(jax.random.PRNGKey(0), [1610, 512, 256, 29], zero_last=True)
    res = [(jnp.asarray(W), jnp.asarray(b)) for W, b in res_params]
    print("teacher + residual", len(res_params))
    idx = [0]

    def frame_path():
        p = FR / f"frame_{idx[0]:05d}.png"; idx[0] += 1; return p

    def compose(pol_img, ref_img, name, step, ret, norm, done, term_step=None):
        im = Image.new("RGB", (W, H), (16, 18, 24))
        d = ImageDraw.Draw(im)
        d.text((14, 10), "SuperMimic-mini", font=F_TITLE, fill=(235, 238, 245))
        d.text((14, 36), "G1 29-DoF · LAFAN1 · teacher + residual (env8 0.466)",
               font=F_SUB, fill=(140, 150, 168))
        d.text((W - 150, 14), "policy vs reference", font=F_LABEL, fill=(120, 170, 255))
        im.paste(Image.fromarray(pol_img), (0, HEADER))
        im.paste(Image.fromarray(ref_img), (PW, HEADER))
        d.rectangle([0, HEADER, W, HEADER + 2], fill=(50, 55, 70))
        d.rectangle([PW - 1, HEADER, PW + 1, HEADER + PH], fill=(50, 55, 70))
        d.text((8, HEADER + 6), "teacher+residual (MuJoCo sim)", font=F_LABEL, fill=(200, 220, 255))
        d.text((PW + 8, HEADER + 6), "reference", font=F_LABEL, fill=(120, 170, 255))
        y = HEADER + PH + 6
        d.text((12, y), f"{name}", font=F_LABEL, fill=(235, 238, 245))
        d.text((120, y), f"step {step:3d}   return {ret:5.2f}   norm {norm:.3f}",
               font=F_BODY, fill=(190, 198, 215))
        if done:
            d.text((W - 170, y), f"terminated @ {term_step}", font=F_LABEL, fill=(255, 120, 120))
        im.save(frame_path())

    for mid, title, max_steps in CLIPS:
        card([(title, F_BIG, (240, 242, 248)),
              ("one general motion tracker · LAFAN1 G1", F_BIG2, (150, 160, 180))],
             frame_path, frames=30)
        clip_i = [i for i, m in enumerate(lib.motions) if m.name == mid][0]
        obs = env.reset(motion_id=clip_i, frame_idx=0)
        hist = np.repeat(obs[None], spec.HISTORY, axis=0)
        ret = 0.0; done = False; term = None
        for t in range(max_steps):
            with torch.no_grad():
                g = goal680(lib.motions[clip_i].qpos, lib.motions[clip_i].qvel, env.frame_idx)
                h = obs930(hist)
                feat = np.concatenate([h, g])
                with torch.no_grad():
                    a_t = pol(torch.as_tensor(g[None], dtype=torch.float32),
                              torch.as_tensor(h[None], dtype=torch.float32))[0].numpy()
                a = a_t + np.asarray(apply_scales(res, jnp.asarray(feat[None])))[0]
            obs, r, te, tr, info = env.step(a)
            hist = np.roll(hist, -1, axis=0); hist[-1] = obs
            ret += r
            fi = min(env.frame_idx, len(lib[clip_i]) - 1)
            ref_data.qpos[:] = lib[clip_i].qpos[fi]
            ref_data.qvel[:] = lib[clip_i].qvel[fi]
            # align root x/y so both panels share the same framing; keep z/joints
            ref_data.qpos[0:2] = env.data.qpos[0:2]
            mujoco.mj_forward(ref_model, ref_data)
            cam.lookat[:] = env.data.qpos[0:3] + np.array([0.0, 0.0, 0.08])
            rend_p.update_scene(env.data, camera=cam)
            rend_r.update_scene(ref_data, camera=cam)
            norm = ret / (0.1 * 499)
            done = bool(te or tr); term = t + 1 if done else None
            compose(rend_p.render(), rend_r.render(), title, t + 1, ret, norm, done, term)
            if done:
                for _ in range(20):
                    compose(rend_p.render(), rend_r.render(), title, t + 1, ret, norm, done, term)
                break

    card([("env8 tracking 0.466", F_BIG, (120, 230, 160)),
          ("5/5 clips · 400-step excerpts (no falls)", F_BIG2, (210, 220, 235)),
          ("teacher 0.4476 + residual (P2.2)", F_BIG2, (150, 160, 180)),
          ("680 goal + 930 obs adapter · local MuJoCo", F_BIG2, (150, 160, 180))],
         frame_path, frames=80)

    print(f"{idx[0]} frames rendered")
    ff = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                         "-i", str(FR / "frame_%05d.png"), "-c:v", "libx264",
                         "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                         str(OUT / "supermimic_ppo_demo_long.mp4")], capture_output=True, text=True)
    print("ffmpeg rc", ff.returncode, ff.stderr[-300:])
    # poster
    shutil.copy(FR / "frame_00109.png", OUT / "poster_ppo_long.png")
    print("video", (OUT / "supermimic_ppo_demo_long.mp4").stat().st_size / 1e6, "MB")


if __name__ == "__main__":
    main()
