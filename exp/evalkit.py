"""统一评测模块 (修正版协议)。

修复 CRITICAL-REVIEW 的 A1–A3:
  A1 度量地板: 同时报 零动作 与 PD 基线, 并给相对分 rel=(s-zero)/(pd-zero)
  A2 T-pose 前导: 默认用 assets/motions_trim
  A3 早停改分母: 固定步数归一化 score = sum(r) / (0.1 * STEPS),
     提前终止时剩余步按 0 记, 并额外报 survival

用法:
  python exp/evalkit.py                      # 评测 baselines + 所有可用 checkpoint
  python exp/evalkit.py --policy exp/runs/xxx.pt
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
from smini import motion, spec
from smini.env import EnvCfg, TrackingEnv
from smini.policy import Policy
from eval_pretrained import Pol, goal680, obs930

XML = str(ROOT.parent/"assets"/"mujoco_menagerie"/"unitree_g1"/"scene_mjx.xml")
CLEAN = EnvCfg(obs_noise=False, reset_noise=False, domain_randomization=False, action_noise=0.0)
MAXR = 0.1          # 每步奖励上限 (5 项 exp × dt=0.02)

# 分层评测集: 覆盖 dance/walk/run/sprint/jumps/fall
EVAL_CLIPS = ["dance1_subject1", "walk1_subject1", "run1_subject2",
              "sprint1_subject2", "jumps1_subject1", "fallAndGetUp1_subject1"]


def load_lib(mdir, clips=EVAL_CLIPS):
    lib = motion.MotionLibrary.from_dir(str(mdir))
    by = {m.name: m for m in lib.motions}
    return motion.MotionLibrary([by[c] for c in clips if c in by])


def make_policy_fn(kind, ckpt=None, teacher=None, noise=0.0, seed=0):
    """返回 fn(lib, ci, env, hist) -> action (29,)"""
    rng = np.random.default_rng(seed)
    if kind == "zero":
        return lambda lib, ci, env, hist: np.zeros(29)
    if kind == "pd":
        return lambda lib, ci, env, hist: env._goal[15:44] / spec.ACTION_SCALE
    if kind == "teacher":
        def f(lib, ci, env, hist):
            mo = lib[ci]; fi = min(int(env.frame_idx), len(mo)-1)
            g = goal680(mo.qpos, mo.qvel, fi)
            a = teacher(torch.as_tensor(g[None], dtype=torch.float32),
                        torch.as_tensor(obs930(hist)[None], dtype=torch.float32))[0].detach().numpy()
            return np.clip(a + noise*rng.standard_normal(29), -50, 50)
        return f
    if kind == "ours":
        pol = Policy(obs_dim=93, goal_dim=730, action_dim=29, hist=10).cuda()
        ck = torch.load(ckpt, map_location="cuda", weights_only=False)
        pol.load_state_dict(ck["policy"]); pol.eval()
        @torch.no_grad()
        def f(lib, ci, env, hist):
            mo = lib[ci]; fi = min(int(env.frame_idx), len(mo)-1)
            g = torch.as_tensor(mo.goal_window(fi), dtype=torch.float32).cuda()[None]
            oh = torch.as_tensor(hist[None].astype(np.float32)).cuda()
            a = pol(g, oh)[0].cpu().numpy()
            return np.clip(a + noise*rng.standard_normal(29), -50, 50)
        return f
    raise ValueError(kind)


def evaluate(fn, lib, steps=500, seed=0):
    """固定步数归一化评测。返回 dict(score, survival, per_clip)。"""
    env = TrackingEnv(XML, lib, CLEAN, seed=seed)
    scores, surv, clips = [], [], []
    for ci in range(len(lib)):
        obs = env.reset(motion_id=ci, frame_idx=0)
        hist = np.repeat(obs[None], spec.HISTORY, 0)
        ret = 0.0; n = 0
        for t in range(steps):
            a = fn(lib, ci, env, hist)
            obs, r, te, tr, _ = env.step(a)
            hist = np.roll(hist, -1, 0); hist[-1] = obs
            ret += r; n += 1
            if te or tr: break
        # 固定分母: 提前终止的剩余步按 0 计
        scores.append(ret / (MAXR * steps))
        surv.append(n); clips.append(lib[ci].name)
    return {"score": float(np.mean(scores)), "survival": float(np.mean(surv)),
            "per_clip": dict(zip(clips, [float(s) for s in scores]))}


def relative(score, zero, pd):
    d = pd - zero
    return (score - zero) / d if abs(d) > 1e-9 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdir", default=str(ROOT/"assets/motions_trim"))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--ckpt", default=str(ROOT/"runs/pretrained/policy.pt"))
    ap.add_argument("--policy", nargs="*", default=None,
                    help="额外评测的本地 checkpoint 列表")
    ap.add_argument("--out", default=str(ROOT/"exp/runs/evalkit.json"))
    a = ap.parse_args()

    torch.set_num_threads(4)
    lib = load_lib(a.mdir)
    print(f"评测集: {len(lib)} 条 ({', '.join(m.name for m in lib.motions)})")
    print(f"数据: {a.mdir}   steps={a.steps}  seeds={a.seeds}\n", flush=True)

    tck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    teacher = Pol(); teacher.load_state_dict(tck["policy"], strict=True); teacher.eval()

    def run(kind, ckpt=None, name=""):
        rows = []
        for sd in a.seeds:
            fn = make_policy_fn(kind, ckpt=ckpt, teacher=teacher, seed=sd)
            r = evaluate(fn, lib, steps=a.steps, seed=sd if kind in ("zero", "pd") else 0)
            rows.append(r)
        sc = float(np.mean([r["score"] for r in rows]))
        sv = float(np.mean([r["survival"] for r in rows]))
        return sc, sv, rows

    z_sc, z_sv, _ = run("zero", name="zero")
    p_sc, p_sv, _ = run("pd", name="pd")
    print(f"{'策略':26s} {'score':>8s} {'surv':>7s} {'相对(PD=1)':>12s}")
    print(f"{'零动作(地板)':26s} {z_sc:8.4f} {z_sv:7.0f} {0.0:12.3f}")
    print(f"{'PD 跟踪(上限参考)':26s} {p_sc:8.4f} {p_sv:7.0f} {1.0:12.3f}")
    res = {"clips": [m.name for m in lib.motions], "steps": a.steps,
           "zero": z_sc, "pd": p_sc, "zero_surv": z_sv, "pd_surv": p_sv, "policies": {}}

    t_sc, t_sv, t_rows = run("teacher", name="teacher")
    print(f"{'仓库教师':26s} {t_sc:8.4f} {t_sv:7.0f} {relative(t_sc,z_sc,p_sc):12.3f}")
    res["policies"]["teacher"] = {"score": t_sc, "surv": t_sv, "rel": relative(t_sc, z_sc, p_sc)}

    targets = a.policy if a.policy else [
        str(ROOT/"exp/runs/p0_clean.pt"),
        str(ROOT/"exp/runs/ac_warm_best.pt"),
        str(ROOT/"exp/runs/ac_p2_best.pt"),
    ]
    for ck in targets:
        if not Path(ck).exists(): continue
        sc, sv, rows = run("ours", ckpt=ck, name=Path(ck).stem)
        print(f"{Path(ck).stem:26s} {sc:8.4f} {sv:7.0f} {relative(sc,z_sc,p_sc):12.3f}")
        res["policies"][Path(ck).stem] = {"score": sc, "surv": sv,
                                          "rel": relative(sc, z_sc, p_sc),
                                          "per_clip": rows[0]["per_clip"]}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
