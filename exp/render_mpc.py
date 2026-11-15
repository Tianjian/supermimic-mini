"""录制 MPC vs 教师 的对比视频: 三格并排 (参考 / 教师 / MPC), 同一 clip 同步。

用法:
  MUJOCO_GL=glfw python exp/render_mpc.py --clip run1_subject2 --steps 150
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "glfw")
from pathlib import Path
import numpy as np, torch, mujoco
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts")); sys.path.insert(0, str(ROOT/"exp"))
import evalkit as EK
from eval_pretrained import Pol, goal680, obs930
from smini import motion, spec
from smini.env import TrackingEnv
from wm_train import make_wm
import diff_fk as D
from diff_fk import FKTable
from mpc_test8 import build_ref

XML, CLEAN, DEV = EK.XML, EK.CLEAN, "cuda"
MAXR = 0.1
W = H = 300


def cam_for(d, cam):
    cam.lookat[:] = [float(d.qpos[0]), float(d.qpos[1]), 0.75]
    cam.distance = 2.7; cam.azimuth = 135.0; cam.elevation = -9.0


def label(img, text, color=(255, 255, 255)):
    im = Image.fromarray(img); dr = ImageDraw.Draw(im)
    try: f = ImageFont.load_default(size=13)
    except TypeError: f = ImageFont.load_default()
    dr.rectangle([0, 0, im.width, 22], fill=(0, 0, 0))
    dr.text((5, 3), text, fill=color, font=f)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="run1_subject2")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--bound", type=float, default=0.3)
    ap.add_argument("--opt-steps", type=int, default=10)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", default=str(ROOT/"docs/demo/mpc"))
    a = ap.parse_args()
    torch.set_num_threads(2)

    tck = torch.load(ROOT/"runs/pretrained/policy.pt", map_location=DEV, weights_only=False)
    net = Pol(); net.load_state_dict(tck["policy"], strict=True); net.to(DEV).eval()
    wm = make_wm(34).to(DEV)
    sd = torch.load(ROOT/"exp/runs/D_wm20k.pt", map_location=DEV, weights_only=False)
    wm.load_state_dict(sd["wm"] if "wm" in sd else sd); wm.eval()
    fk = FKTable(XML, spec.TRACKED_BODIES, spec.JOINT_NAMES, device=DEV)
    xi = torch.as_tensor(np.concatenate([[1., 1.], np.zeros(29), np.zeros(3)]),
                         dtype=torch.float32).to(DEV)[None]

    full = motion.MotionLibrary.from_dir(ROOT/"assets/motions_trim")
    idx = [i for i, m in enumerate(full.motions) if m.name == a.clip]
    if not idx: raise SystemExit(f"clip {a.clip} not found")
    lib = motion.MotionLibrary([full.motions[idx[0]]])
    mo = lib[0]
    print(f"clip={a.clip} frames={len(mo)} steps={a.steps}", flush=True)

    # 两个独立环境: 教师 / MPC
    e_t = TrackingEnv(XML, lib, CLEAN, seed=3); e_t.reset(motion_id=0, frame_idx=0)
    e_p = TrackingEnv(XML, lib, CLEAN, seed=3); e_p.reset(motion_id=0, frame_idx=0)
    hist_t = [ np.repeat(e_t._obs[None], spec.HISTORY, 0) ]
    hist_p = [ np.repeat(e_p._obs[None], spec.HISTORY, 0) ]
    hist_t = hist_t[0]; hist_p = hist_p[0]
    raw = np.zeros((1, a.H, 29), dtype=np.float32)
    prev_a0 = np.zeros((1, 29), dtype=np.float32)
    ret_t = ret_p = 0.0
    surv_t = surv_p = 0
    dead_t = dead_p = False

    mref = mujoco.MjModel.from_xml_path(XML); dref = mujoco.MjData(mref)
    ren = {"ref": mujoco.Renderer(mref, H, W),
           "t": mujoco.Renderer(e_t._model, H, W),
           "p": mujoco.Renderer(e_p._model, H, W)}
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    import imageio
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(outdir/f"mpc_vs_teacher_{a.clip}.mp4"),
                                fps=a.fps, quality=8, macro_block_size=1)

    for step in range(a.steps):
        # 参考面板
        ti = min(step + 1, len(mo) - 1)
        dref.qpos[:] = mo.qpos[ti]; dref.qvel[:] = mo.qvel[ti]
        mujoco.mj_forward(mref, dref); cam_for(dref, cam)
        ren["ref"].update_scene(dref, cam)
        frames = [label(ren["ref"].render(), f"REF t={ti/mo.fps:.1f}s", (120, 220, 255))]

        # 教师
        if not dead_t:
            fi = min(int(e_t.frame_idx), len(mo) - 1)
            with torch.no_grad():
                g = torch.as_tensor(goal680(mo.qpos, mo.qvel, fi), dtype=torch.float32).to(DEV)[None]
                oh = torch.as_tensor(obs930(hist_t)[None].astype(np.float32)).to(DEV)
                at = net(g, oh)[0].cpu().numpy()
            o, r, te, tr, _ = e_t.step(at)
            ret_t += float(r); surv_t += 1
            hist_t = np.roll(hist_t, -1, 0); hist_t[-1] = o
            if te or tr: dead_t = True
        cam_for(e_t._data, cam); ren["t"].update_scene(e_t._data, cam)
        sc = ret_t / (MAXR * (step + 1))
        frames.append(label(ren["t"].render(),
                            f"TEACHER {sc:.3f}" + (f" FALL@{surv_t}" if dead_t else ""),
                            (255, 120, 120) if dead_t else (255, 255, 255)))

        # MPC
        if not dead_p:
            fi = min(int(e_p.frame_idx), len(mo) - 1)
            f0 = min(fi, len(mo) - a.H - 2)
            with torch.no_grad():
                g = torch.as_tensor(goal680(mo.qpos, mo.qvel, fi), dtype=torch.float32).to(DEV)[None]
                oh = torch.as_tensor(obs930(hist_p)[None].astype(np.float32)).to(DEV)
                a_tc = net(g, oh).cpu().numpy()
            q, v = e_p._data.qpos, e_p._data.qvel
            S0 = torch.as_tensor(np.concatenate(
                [q[0:3], q[3:7], q[7:36], v[0:3], v[3:6], v[6:35]])[None].astype(np.float32)).to(DEV)
            ref = build_ref(mo, fi + 1, a.H, device="cpu")
            refb = {k: torch.as_tensor(vv[None]).to(DEV) for k, vv in ref.items()}
            gseq = torch.as_tensor(np.stack(
                [goal680(mo.qpos, mo.qvel, f0 + t) for t in range(a.H)])[None].astype(np.float32)).to(DEV)
            a_tc_t = torch.as_tensor(a_tc).to(DEV).unsqueeze(1).repeat(1, a.H, 1)
            r0 = torch.tensor(raw, device=DEV, requires_grad=True)
            opt = torch.optim.Adam([r0], lr=0.08)
            for _ in range(a.opt_steps):
                acts = a_tc_t + a.bound * torch.tanh(r0)
                s = S0; pv = torch.as_tensor(prev_a0).to(DEV); tot = 0.0
                for t in range(a.H):
                    s = wm.step(s, acts[:, t], xi)
                    r = D.reward8(s, acts[:, t], pv, {k: vv[:, t] for k, vv in refb.items()}, fk)
                    tot = tot + r.sum(); pv = acts[:, t]
                opt.zero_grad(); (-tot).backward(); opt.step()
            with torch.no_grad():
                sol = (a_tc_t + a.bound * torch.tanh(r0)).cpu().numpy()
            raw = np.concatenate([sol[:, 1:], sol[:, -1:]], axis=1).astype(np.float32)
            act = sol[:, 0]
            o, r, te, tr, _ = e_p.step(act[0])
            ret_p += float(r); surv_p += 1
            hist_p = np.roll(hist_p, -1, 0); hist_p[-1] = o
            prev_a0 = act
            if te or tr: dead_p = True
        cam_for(e_p._data, cam); ren["p"].update_scene(e_p._data, cam)
        sc = ret_p / (MAXR * (step + 1))
        frames.append(label(ren["p"].render(),
                            f"MPC (K={a.opt_steps}) {sc:.3f}" + (f" FALL@{surv_p}" if dead_p else ""),
                            (120, 255, 160) if not dead_p else (255, 120, 120)))
        writer.append_data(np.concatenate(frames, axis=1))
        if step % 25 == 0:
            print(f"  step {step}: teacher {ret_t/(MAXR*(step+1)):.3f} "
                  f"mpc {ret_p/(MAXR*(step+1)):.3f}", flush=True)
        if dead_t and dead_p: break
    writer.close()
    for r in ren.values(): r.close()
    print(f"saved {outdir}/mpc_vs_teacher_{a.clip}.mp4")
    print(f"  teacher 最终 {ret_t/(MAXR*a.steps):.4f} 存活 {surv_t}")
    print(f"  mpc     最终 {ret_p/(MAXR*a.steps):.4f} 存活 {surv_p}")


if __name__ == "__main__":
    main()
