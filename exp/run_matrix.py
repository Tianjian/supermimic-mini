"""P0/P1/P2 消融矩阵 (教师底座, H=16, 40 轮, 3 种子)。"""
import subprocess, sys, time, json
from pathlib import Path
from datetime import datetime
ROOT = Path(__file__).resolve().parents[1]; RUNS = ROOT/"exp"/"runs"; OUT = ROOT/"exp"/"overnight"
CFGS = {
 "A_old":      ["--bootstrap-grad","0","--survival","0","--critic-epochs","1"],
 "B_p0":       ["--bootstrap-grad","1","--survival","0","--critic-epochs","1"],
 "C_p0p1":     ["--bootstrap-grad","1","--survival","1","--critic-epochs","5"],
 "D_p0p1_cl":  ["--bootstrap-grad","1","--survival","1","--critic-epochs","5","--act-clamp","0.2"],
 "E_p0p1_bc":  ["--bootstrap-grad","1","--survival","1","--critic-epochs","5","--bc-anchor","5.0"],
 "F_full":     ["--bootstrap-grad","1","--survival","1","--critic-epochs","5","--act-clamp","0.2",
                "--online-every","10","--online-size","5000","--wm-steps","300"],
}
def main():
    jobs=[]
    for tag, extra in CFGS.items():
        for s in range(3):
            jobs.append(([sys.executable,"-u","exp/ac_bptt.py","--ckpt",str(RUNS/"D_wm20k.pt"),
                          "--teacher-base","--skip-bc","--evalkit","--B","32","--iters","40",
                          "--H","16","--kl-target","0.02","--seed",str(s),"--reset-adam-on-bt",
                          "--out",str(RUNS/f"M_{tag}_s{s}.json")]+extra, f"M_{tag}_s{s}"))
    print(f"共 {len(jobs)} 个 run", flush=True)
    procs=[]; q=list(jobs)
    while q or procs:
        while q and len(procs)<3:
            cmd,name=q.pop(0); f=open(OUT/f"{name}.log","w")
            procs.append((subprocess.Popen(cmd,cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT),name,f,time.time()))
            print(f"[{datetime.now():%H:%M:%S}] launch {name}", flush=True)
        time.sleep(5); keep=[]
        for p,name,f,t0 in procs:
            if p.poll() is None: keep.append((p,name,f,t0))
            else:
                f.close(); print(f"[{datetime.now():%H:%M:%S}] {'DONE' if p.returncode==0 else 'FAIL'} {name} ({time.time()-t0:.0f}s)", flush=True)
        procs=keep
if __name__=="__main__": main()
