"""渲染对比视频: 参考动作 vs 各策略, 同 clip 同步并排, 供人工检视。

用法:
  MUJOCO_GL=glfw python exp/render_compare.py --clip walk1_subject1 --steps 300
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "glfw")
from pathlib import Path
import numpy as np, torch, mujoco
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts")); sys.path.insert(0, str(ROOT/"exp"))
from smini import motion, spec
from smini.env import EnvCfg, TrackingEnv
from smini.policy import Policy
from eval_pretrained import Pol, goal680, obs930
import evalkit as EK

XML = EK.XML
MAXR = 0.1
W = H = 300


def cam_for(data, cam):
    cam.lookat[:] = [float(data.qpos[0]), float(data.qpos[1]), 0.75]
    cam.distance = 2.7
    cam.azimuth = 135.0
    cam.elevation = -9.0


def label(img, text, color=(255, 255, 255)):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.load_default(size=13)
    except TypeError:
        f = ImageFont.load_default()
    d.rectangle([0, 0, im.width, 22], fill=(0, 0, 0))
    d.text((5, 3), text, fill=color, font=f)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="walk1_subject1")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--every", type=int, default=2, help="每 N 个控制步渲染一帧")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--mdir", default=str(ROOT/"assets/motions_trim"))
    ap.add_argument("--out", default=str(ROOT/"docs/demo/compare"))
    a = ap.parse_args()

    lib_full = motion.MotionLibrary.from_dir(a.mdir)
    by = {m.name: m for m in lib_full.motions}
    if a.clip not in by: raise SystemExit(f"clip {a.clip} not found")
    lib = motion.MotionLibrary([by[a.clip]])
    mo = lib[0]
    print(f"clip={a.clip} frames={len(mo)} render {a.steps} steps", flush=True)

    # --- 策略 ---
    tck = torch.load(ROOT/"runs/pretrained/policy.pt", map_location="cpu", weights_only=False)
    teacher = Pol(); teacher.load_state_dict(tck["policy"], strict=True); teacher.eval()

    def load_ours(p):
        pol = Policy(obs_dim=93, goal_dim=730, action_dim=29, hist=10).cuda()
        ck = torch.load(p, map_location="cuda", weights_only=False)
        pol.load_state_dict(ck["policy"]); pol.eval(); return pol

    bc = load_ours(ROOT/"exp/runs/p0_clean.pt")
    bptt = load_ours(ROOT/"exp/runs/ac_probe_final.pt")

    panels = []   # (name, kind)
    panels.append(("REFERENCE", "ref"))
    panels.append(("TEACHER", "teacher"))
    panels.append(("BC distilled", "bc"))
    panels.append(("BPTT+trust-region", "bptt"))
    panels.append(("PD tracking", "pd"))

    # --- 环境 (每个策略一个) ---
    envs = {}
    for nm, kind in panels:
        if kind == "ref": continue
        e = TrackingEnv(XML, lib, EK.CLEAN, seed=0)
        e.reset(motion_id=0, frame_idx=0)
        envs[kind] = e

    # 参考播放用独立 data
    mref = mujoco.MjModel.from_xml_path(XML)
    dref = mujoco.MjData(mref)

    renderers = {}
    for nm, kind in panels:
        mdl = mref if kind == "ref" else envs[kind]._model
        renderers[kind] = mujoco.Renderer(mdl, H, W)

    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)

    score = {k: 0.0 for k in envs}
    dead = {k: False for k in envs}
    nstep = {k: 0 for k in envs}
    hist = {k: np.repeat(envs[k]._obs[None], spec.HISTORY, 0) for k in envs}
    prev_a = {k: np.zeros(29) for k in envs}

    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    w = imageio_writer = None
    import imageio
    writer = imageio.get_writer(str(outdir/f"compare_{a.clip}.mp4"), fps=a.fps, quality=8,
                                macro_block_size=1)
    t_ref = 0
    for step in range(a.steps):
        frames = []
        # 参考
        t_ref = min(int(step + 1), len(mo) - 1)
        dref.qpos[:] = mo.qpos[t_ref]; dref.qvel[:] = mo.qvel[t_ref]
        mujoco.mj_forward(mref, dref)
        cam_for(dref, cam)
        renderers["ref"].update_scene(dref, cam)
        frames.append(label(renderers["ref"].render(),
                            f"REF t={t_ref/mo.fps:.1f}s", (120, 220, 255)))
        # 各策略
        for nm, kind in panels:
            if kind == "ref": continue
            e = envs[kind]
            if not dead[kind]:
                mo_ = lib[0]; fi = min(int(e.frame_idx), len(mo_)-1)
                if kind == "teacher":
                    g = goal680(mo_.qpos, mo_.qvel, fi)
                    act = teacher(torch.as_tensor(g[None], dtype=torch.float32),
                                  torch.as_tensor(obs930(hist[kind])[None], dtype=torch.float32))[0].detach().numpy()
                elif kind == "pd":
                    act = e._goal[15:44] / spec.ACTION_SCALE
                else:
                    pol = bc if kind == "bc" else bptt
                    with torch.no_grad():
                        g = torch.as_tensor(mo_.goal_window(fi), dtype=torch.float32).cuda()[None]
                        oh = torch.as_tensor(hist[kind][None].astype(np.float32)).cuda()
                        act = pol(g, oh)[0].cpu().numpy()
                obs, r, te, tr, _ = e.step(act)
                hist[kind] = np.roll(hist[kind], -1, 0); hist[kind][-1] = obs
                score[kind] += float(r); nstep[kind] += 1
                prev_a[kind] = act
                if te or tr: dead[kind] = True
            e2 = envs[kind]._data
            cam_for(e2, cam)
            renderers[kind].update_scene(e2, cam)
            sc = score[kind] / (MAXR * (step + 1))
            short = {"TEACHER": "TEACHER", "BC distilled": "BC",
                     "BPTT+trust-region": "BPTT+TR", "PD tracking": "PD"}[nm]
            tag = f"{short} {sc:.3f}"
            if dead[kind]: tag += f" FALL@{nstep[kind]}"
            col = (255, 120, 120) if dead[kind] else (255, 255, 255)
            frames.append(label(renderers[kind].render(), tag, col))
        grid = np.concatenate(frames, axis=1)
        if step % a.every == 0:
            writer.append_data(grid)
    writer.close()
    print(f"saved {outdir}/compare_{a.clip}.mp4")
    for nm, kind in panels:
        if kind == "ref": continue
        print(f"  {nm:20s} final score={score[kind]/(MAXR*a.steps):.4f}  survived={nstep[kind]}")
    for r in renderers.values():
        r.close()


if __name__ == "__main__":
    main()
