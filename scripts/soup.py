#!/usr/bin/env python3
"""Greedy / uniform checkpoint soup for the policy.

Usage:
  python scripts/soup.py --run-dir runs/p0_2000 --top-k 5 10 20 --greedy
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from smini import motion
from smini.policy import Policy
from smini.train import TrainCfg, evaluate

XML = str(ROOT.parent / "assets/mujoco_menagerie/unitree_g1/scene_mjx.xml")


def load(path):
    ck = torch.load(path, map_location="cpu")
    return ck["policy"]


def mix(sds, weights=None):
    if weights is None:
        weights = [1.0] * len(sds)
    s = sum(weights)
    return {k: sum(w / s * sd[k].float() for sd, w in zip(sds, weights))
            for k in sds[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--top-k", type=int, nargs="*", default=[3, 5, 10, 20])
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--clips", type=int, default=5)
    args = ap.parse_args()
    lib = motion.MotionLibrary.from_dir(ROOT / "assets/motions")
    cfg = TrainCfg(xml_path=XML, motion_dir=str(ROOT / "assets/motions"),
                   eval_clips=args.clips, eval_steps=500, seed=0)
    files = sorted(glob.glob(str(Path(args.run_dir) / "*.pt")))
    print(f"{len(files)} checkpoints")
    pol = Policy().cuda(); pol.eval()
    scored = []
    for f in files:
        try:
            sd = load(f)
            pol.load_state_dict(sd)
            ev = evaluate(pol, lib, cfg)["return"]
            scored.append((ev, f, sd))
            print(f"{Path(f).name:20s} {ev:.4f}")
        except Exception as e:
            print("skip", f, e)
    scored.sort(key=lambda t: t[0], reverse=True)
    best_ev, best_f, best_sd = scored[0]
    print(f"best single {best_f} {best_ev:.4f}")
    torch.save({"policy": best_sd}, ROOT / "runs/soup_best_single.pt")
    for k in args.top_k:
        sds = [sd for _, _, sd in scored[:k]]
        sd = mix(sds)
        pol.load_state_dict(sd)
        ev = evaluate(pol, lib, cfg)["return"]
        print(f"uniform top-{k}: {ev:.4f}")
        if ev > best_ev:
            best_ev, best_sd = ev, sd
            torch.save({"policy": sd}, ROOT / "runs/soup_best_uniform.pt")
    if args.greedy:
        sel = [(best_ev, best_sd)]
        pol.load_state_dict(best_sd)
        cur = best_ev
        for ev, f, sd in scored[1:]:
            cand = mix([s for _, s in sel] + [sd])
            pol.load_state_dict(cand)
            cev = evaluate(pol, lib, cfg)["return"]
            if cev > cur + 1e-4:
                sel.append((ev, sd)); cur = cev
                print(f"greedy +{Path(f).name} -> {cev:.4f} (n={len(sel)})")
                best_ev, best_sd = cev, cand
        print(f"greedy final {best_ev:.4f} n={len(sel)}")
        torch.save({"policy": best_sd}, ROOT / "runs/soup_best_greedy.pt")
    torch.save({"policy": best_sd}, ROOT / "runs/soup_best.pt")
    print(f"saved runs/soup_best.pt  eval {best_ev:.4f}")


if __name__ == "__main__":
    main()
