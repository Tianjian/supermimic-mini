"""D 轮: 测两个尚未验证的关键杠杆 — 动作硬夹具 / 极短视野。

动机:
  A1 (-79.8%) 与 C (-68.7%) 都崩塌, WM 精度提升 3.5x 也只把伤害从 -80% 降到 -69%。
  剩下两个可能:
    (1) 优化器跑到了模型不准的动作区域 -> 用动作硬夹具 (--act-clamp) 从源头禁止
    (2) 视野 H=16 太长, 误差已主导 -> 试 H=4

用法: setsid nohup python -u exp/run_D.py > exp/overnight/runD.log 2>&1 &
"""
import subprocess, sys, time, json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"exp"/"overnight"
RUNS = ROOT/"exp"/"runs"
MAX_PAR = 3


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def batch(jobs):
    procs, res, q = [], {}, list(jobs)
    while q or procs:
        while q and len(procs) < MAX_PAR:
            cmd, name = q.pop(0)
            f = open(OUT/f"{name}.log", "w")
            procs.append((subprocess.Popen(cmd, cwd=str(ROOT), stdout=f,
                                           stderr=subprocess.STDOUT), name, f, time.time()))
            log(f"launch {name}")
        time.sleep(5)
        keep = []
        for p, name, f, t0 in procs:
            if p.poll() is None:
                keep.append((p, name, f, t0))
            else:
                f.close(); res[name] = p.returncode == 0
                log(f"{'DONE' if p.returncode==0 else 'FAIL'} {name} ({time.time()-t0:.0f}s)")
        procs = keep
    return res


def ac_extra(wm, extra, tag):
    return ([sys.executable, "-u", "exp/ac_bptt.py", "--ckpt", str(wm),
             "--teacher-base", "--skip-bc", "--evalkit", "--B", "64", "--iters", "60",
             "--out", str(RUNS/f"{tag}.json")] + extra, tag)


def main():
    wm = RUNS/"B_base.pt"
    log(f"D 轮开始, WM = {wm.name}")
    jobs = []
    # (1) 动作硬夹具: 把想象动作夹在锚定策略 ±ε
    for eps in ("0.05", "0.2"):
        for s in range(3):
            jobs.append(ac_extra(wm, ["--H", "16", "--kl-target", "0.02", "--seed", str(s),
                                      "--reset-adam-on-bt", "--act-clamp", eps],
                                 f"D_clamp{eps.replace('.','')}_s{s}"))
    # (2) 极短视野, 无夹具
    for s in range(3):
        jobs.append(ac_extra(wm, ["--H", "4", "--kl-target", "0.02", "--seed", str(s),
                                  "--reset-adam-on-bt"], f"D_H4_s{s}"))
    # (3) WM 长训(教师分布, 20k 步, 无课程), 与上面并行
    jobs.append(([sys.executable, "-u", "exp/wm_onpolicy.py",
                  "--ckpt", str(RUNS/"v3_best.pt"), "--teacher-policy",
                  "--steps", "20000", "--online-size", "150000",
                  "--eval-every", "4000", "--out", str(RUNS/"D_wm20k.pt")], "D_wm20k"))
    res = batch(jobs)
    json.dump(res, open(OUT/"runD_manifest.json", "w"), indent=1)
    log("D 轮结束")


if __name__ == "__main__":
    main()
