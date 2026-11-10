#!/usr/bin/env python3
"""Render a policy-vs-reference demo video (MuJoCo EGL, 50 fps)."""
from __future__ import annotations
import os, shutil, subprocess, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import mujoco
import torch
from PIL import Image, ImageDraw, ImageFont
from smini import motion, spec
from smini.env import TrackingEnv, EnvCfg
from smini.policy import Policy

XML = str(ROOT.parent / "assets/mujoco_menagerie/unitree_g1/scene_mjx.xml")
OUT = ROOT / "docs/demo"
FR = OUT / "frames"
W, H = 640, 510
PW, PH = 320, 400
HEADER = 56
FOOT = H - HEADER - PH
FPS = 50
CLIPS = [("sprint1_subject2", "Sprint", 200), ("walk1_subject1", "Walk", 200),
         ("dance1_subject1", "Dance", 200), ("jumps1_subject1", "Jumps", 200),
         ("run1_subject2", "Run", 200)]


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
    lib = motion.MotionLibrary.from_dir(ROOT / "assets/motions")
    env = TrackingEnv(XML, lib, EnvCfg(obs_noise=False, reset_noise=False,
                                       domain_randomization=False, action_noise=0), seed=0)
    ref_model = tint_reference(XML)
    ref_data = mujoco.MjData(ref_model)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 135.0, -12.0, 2.4
    rend_p = mujoco.Renderer(env.model, PH, PW)
    rend_r = mujoco.Renderer(ref_model, PH, PW)

    pol = Policy().cuda(); ck = torch.load(ROOT / "runs/soup_best.pt", map_location="cuda")
    pol.load_state_dict(ck["policy"]); pol.eval()
    idx = [0]

    def frame_path():
        p = FR / f"frame_{idx[0]:05d}.png"; idx[0] += 1; return p

    def compose(pol_img, ref_img, name, step, ret, norm, done, term_step=None):
        im = Image.new("RGB", (W, H), (16, 18, 24))
        d = ImageDraw.Draw(im)
        d.text((14, 10), "SuperMimic-mini", font=F_TITLE, fill=(235, 238, 245))
        d.text((14, 36), "G1 29-DoF · LAFAN1 · CPU MuJoCo + RTX 2060 · WM+BPTT",
               font=F_SUB, fill=(140, 150, 168))
        d.text((W - 150, 14), "policy vs reference", font=F_LABEL, fill=(120, 170, 255))
        im.paste(Image.fromarray(pol_img), (0, HEADER))
        im.paste(Image.fromarray(ref_img), (PW, HEADER))
        d.rectangle([0, HEADER, W, HEADER + 2], fill=(50, 55, 70))
        d.rectangle([PW - 1, HEADER, PW + 1, HEADER + PH], fill=(50, 55, 70))
        d.text((8, HEADER + 6), "policy (MuJoCo sim)", font=F_LABEL, fill=(200, 220, 255))
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
                g = torch.as_tensor(env.goal(), dtype=torch.float32, device="cuda")[None]
                h = torch.as_tensor(hist, dtype=torch.float32, device="cuda")[None]
                a = pol(g, h)[0].cpu().numpy()
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

    card([("eval return 0.1046", F_BIG, (120, 230, 160)),
          ("sprint 179 steps · walk 100 steps", F_BIG2, (210, 220, 235)),
          ("WM nrmse8 0.060 (PD distribution)", F_BIG2, (150, 160, 180)),
          ("4-run checkpoint soup · BPTT H=16 · critic", F_BIG2, (150, 160, 180))],
         frame_path, frames=80)

    print(f"{idx[0]} frames rendered")
    ff = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                         "-i", str(FR / "frame_%05d.png"), "-c:v", "libx264",
                         "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                         str(OUT / "supermimic_mini_demo.mp4")], capture_output=True, text=True)
    print("ffmpeg rc", ff.returncode, ff.stderr[-300:])
    # poster
    shutil.copy(FR / "frame_00109.png", OUT / "poster.png")
    print("video", (OUT / "supermimic_mini_demo.mp4").stat().st_size / 1e6, "MB")


if __name__ == "__main__":
    main()
