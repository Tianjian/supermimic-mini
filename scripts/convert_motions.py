#!/usr/bin/env python3
"""Download (if needed) and convert LAFAN1 G1 CSVs to 50 Hz npz.

Usage:  python scripts/convert_motions.py --clips walk1_subject1 run1_subject2 ...
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smini import build_model, motion  # noqa: E402

HF = "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/g1/{name}.csv"
DEFAULT_CLIPS = ["walk1_subject1", "run1_subject2", "dance1_subject1",
                 "sprint1_subject2", "jumps1_subject1"]


def download(name: str, csv_dir: Path) -> Path:
    csv_dir.mkdir(parents=True, exist_ok=True)
    dst = csv_dir / f"{name}.csv"
    if dst.exists():
        return dst
    url = HF.format(name=name)
    print(f"download {url}")
    urllib.request.urlretrieve(url, dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=DEFAULT_CLIPS)
    ap.add_argument("--csv-dir", default=str(ROOT / "assets" / "lafan1_csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "assets" / "motions"))
    ap.add_argument("--xml", default=str(ROOT.parent / "assets" / "mujoco_menagerie" / "unitree_g1" / "scene_mjx.xml"))
    args = ap.parse_args()

    model = build_model.build_model(args.xml)
    bids = build_model.body_ids(model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.clips:
        csv = download(name, Path(args.csv_dir))
        m = motion.convert_csv(csv, model, bids)
        out = out_dir / f"{name}.npz"
        motion.save_motion(m, out)
        print(f"{name}: {len(m)} frames @ {m.fps} Hz -> {out} "
              f"({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
