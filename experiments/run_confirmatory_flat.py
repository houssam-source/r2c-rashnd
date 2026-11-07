"""Confirmatory flat-E4 runner: per-trial rows for paired analysis.

Writes seed,mode,controller,recovered,REC,thr,mean_q per trial."""
import argparse, csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_e4_variants import trial as e4_trial
from r2c.config import R2CParams
from r2c.rf import RFParams


def one(task):
    mode, c, seed = task
    p = R2CParams(**{**R2CParams().__dict__, "seed": seed})
    rf_p = RFParams(jammer_mode=mode, seed=seed)
    if c == "R2C":
        return e4_trial(p, rf_p, 0.5, "r2c", "")
    return e4_trial(p, rf_p, 0.5, "hybrid", c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--modes", nargs="+", default=["constant", "sweep", "reactive"])
    ap.add_argument("--controllers", nargs="+", default=["R2C", "v2", "v3b"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/confirmatory_flat.csv")
    args = ap.parse_args()
    tasks = [(m, c, i) for m in args.modes for c in args.controllers
             for i in range(args.trials)]
    res = {}
    if args.workers > 1:
        from experiments._parallel import map_trials
        tmp = {}
        map_trials(one, tasks, args.workers, lambda i, r: tmp.__setitem__(i, r))
        res = tmp
    else:
        for idx, t in enumerate(tasks):
            res[idx] = one(t)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "mode", "controller", "recovered", "REC",
                    "thr_mbps", "mean_q"])
        for idx, (m, c, s) in enumerate(tasks):
            r = res[idx]
            w.writerow([s, m, c, f"{r['recovered']:.0f}", f"{r['REC']:.4f}",
                        f"{r['thr_mbps']:.3f}", f"{r['mean_q']:.4f}"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
