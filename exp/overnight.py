"""Overnight 编排器: 无人值守跑完阶段 A / B / C, 自动汇总报告。

设计原则(为无人反馈优化):
  * 每一步失败只记录、不中断(结果缺失会在报告里体现)
  * 最多 3 个并发 GPU 任务(实测 16 GB 卡 6 并发也只占 4.9 GB, 3 个足够保守)
  * 全局时间上限(默认 7 h), 超时后跳过剩余步骤直接写报告
  * 所有判断都基于 evalkit 协议, 不产生旧协议数字

用法:
  setsid nohup python -u exp/overnight.py > exp/overnight/run.log 2>&1 &
"""
import subprocess, sys, time, json, os
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"exp"/"overnight"; OUT.mkdir(parents=True, exist_ok=True)
RUNS = ROOT/"exp"/"runs"
BUDGET_H = float(os.environ.get("OVERNIGHT_BUDGET_H", "7.0"))
MAX_PAR = int(os.environ.get("OVERNIGHT_MAX_PAR", "3"))
DEADLINE = datetime.now() + timedelta(hours=BUDGET_H)
START = datetime.now()


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def out_of_time():
    return datetime.now() >= DEADLINE


def run(cmd, logname, timeout=5400):
    """跑一条命令, 返回 (ok, elapsed)."""
    lg = OUT/f"{logname}.log"
    t0 = time.time()
    log(f"START {logname}")
    try:
        with open(lg, "w") as f:
            p = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                               timeout=timeout)
        ok = p.returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
        log(f"TIMEOUT {logname}")
    log(f"{'DONE ' if ok else 'FAIL '} {logname} ({time.time()-t0:.0f}s)")
    return ok, time.time()-t0


def run_batch(jobs, tag):
    """最多 MAX_PAR 个并发; jobs = [(cmd, logname), ...]"""
    procs = []
    results = {}
    queue = list(jobs)
    while queue or procs:
        while queue and len(procs) < MAX_PAR and not out_of_time():
            cmd, name = queue.pop(0)
            lg = OUT/f"{name}.log"
            f = open(lg, "w")
            p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
            procs.append((p, name, f, time.time()))
            log(f"launch {name}")
        time.sleep(5)
        still = []
        for p, name, f, t0 in procs:
            if p.poll() is None:
                still.append((p, name, f, t0))
            else:
                f.close()
                results[name] = (p.returncode == 0, time.time()-t0)
                log(f"{'DONE ' if p.returncode == 0 else 'FAIL '} {name} ({time.time()-t0:.0f}s)")
        procs = still
        if out_of_time() and queue:
            log(f"时间用尽, 丢弃 {len(queue)} 个未启动任务")
            queue = []
    return results


def ac(ckpt, extra, tag):
    cmd = [sys.executable, "-u", "exp/ac_bptt.py",
           "--ckpt", str(ckpt), "--teacher-base", "--skip-bc", "--evalkit",
           "--B", "64", "--iters", "60", "--out", str(RUNS/f"{tag}.json")] + extra
    return cmd, tag


def main():
    log(f"overnight 开始, 预算 {BUDGET_H}h, 并发 {MAX_PAR}")
    manifest = {"start": START.isoformat(), "budget_h": BUDGET_H, "batches": {}}

    # ---------------- A1: 教师底座 + BPTT ----------------
    jobs = []
    cfg = [(16, "0.02")]      # A1 已显示崩塌; 只留 3 种子做干净记录
    for H, kl in cfg:
        for s in range(3):
            tag = f"A1_H{H}_kl{kl.replace('.','')}_s{s}"
            cmd, _ = ac(RUNS/"v3_best.pt",
                        ["--H", str(H), "--kl-target", kl, "--seed", str(s),
                         "--reset-adam-on-bt"], tag)
            jobs.append((cmd, tag))
    log(f"A1: {len(jobs)} 个 run")
    manifest["batches"]["A1"] = run_batch(jobs, "A1")

    # ---------------- B: on-policy WM (基线 vs 幅度课程) ----------------
    b_jobs = []
    for tag, extra in (
        ("B_base",  ["--online-frac", "0.5", "--steps", "6000", "--online-size", "100000"]),
        ("B_mag",   ["--online-frac", "0.5", "--steps", "6000", "--online-size", "100000",
                     "--mag-prob", "0.2", "--mag-scale", "3.0"]),
        ("B_mag2",  ["--online-frac", "0.5", "--steps", "10000", "--online-size", "150000",
                     "--mag-prob", "0.3", "--mag-scale", "4.0"]),
        ("B_mag3",  ["--online-frac", "0.6", "--steps", "14000", "--online-size", "200000",
                     "--mag-prob", "0.4", "--mag-scale", "5.0"]),
    ):
        cmd = [sys.executable, "-u", "exp/wm_onpolicy.py",
               "--ckpt", str(RUNS/"v3_best.pt"), "--policy", str(RUNS/"p0_clean.pt"),
               "--teacher-policy",
               "--eval-every", "2000", "--out", str(RUNS/f"{tag}.pt")] + extra
        b_jobs.append((cmd, tag))
    log(f"B: {len(b_jobs)} 个 run")
    manifest["batches"]["B"] = run_batch(b_jobs, "B")

    # ---------------- C: 用改进后的 WM 重试 BPTT ----------------
    # 选 B 里 on-policy NSE32 最好的 checkpoint
    best_wm, best_nse = RUNS/"v3_best.pt", 1e9
    for tag in ("B_mag2", "B_mag", "B_base"):
        js = RUNS/f"{tag}.json"
        ck = RUNS/f"{tag}.pt"
        if not (js.exists() and ck.exists()):
            continue
        d = json.load(open(js))
        base = d.get("base") or []
        log_ = d.get("log") or []
        nse = min([e["nse"][31] for e in log_], default=1e9)
        if nse < best_nse:
            best_wm, best_nse = ck, nse
    log(f"C: 选用的 WM = {best_wm.name} (best on-policy NSE32={best_nse:.3f})")
    manifest["best_wm"] = str(best_wm)

    c_jobs = []
    for H in (16,):
        for s in range(3):
            for anch in (True, False):
                tag = f"C_H{H}_anchor{int(anch)}_s{s}"
                extra = ["--H", str(H), "--kl-target", "0.02", "--seed", str(s),
                         "--reset-adam-on-bt"] + (["--anchor-init"] if anch else [])
                cmd, _ = ac(best_wm, extra, tag)
                c_jobs.append((cmd, tag))
    manifest["batches"]["C"] = run_batch(c_jobs, "C")

    # ---------------- 视频 (失败不影响报告) ----------------
    if not out_of_time():
        for clip in ("walk1_subject1",):
            run([sys.executable, "-u", "exp/render_compare.py", "--clip", clip,
                 "--steps", "300", "--out", str(ROOT/"docs/demo/compare")],
                f"video_{clip}", timeout=1800)

    if not out_of_time():
        run([sys.executable, "-u", "exp/overnight_report.py"], "report", timeout=900)

    manifest["end"] = datetime.now().isoformat()
    manifest["elapsed_h"] = (datetime.now()-START).total_seconds()/3600
    json.dump(manifest, open(OUT/"manifest.json", "w"), indent=1)
    log(f"全部结束, 用时 {manifest['elapsed_h']:.2f}h")


if __name__ == "__main__":
    main()
