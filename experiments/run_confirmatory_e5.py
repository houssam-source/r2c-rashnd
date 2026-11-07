"""Confirmatory C4: E5 converged quality, per-trial paired rows.

Reuses run_e5.run_e5_trial; converged quality = mean of mean_quality trace
over the last 100 steps."""
import argparse, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_e5 import run_e5_trial
from r2c.config import R2CParams
from r2c.rf import RFParams


def one(task):
    mode, controller, seed = task
    p = R2CParams(seed=seed)
    rf_p = RFParams(jammer_mode=mode, seed=seed)
    trace = run_e5_trial(p, rf_p, 0.5, controller)
    last = [q for t, q, _thr in trace if t >= p.T - 100]
    return float(np.mean(last)) if last else float("nan")


import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--modes", nargs="+",
                    default=["constant", "sweep", "reactive"])
    ap.add_argument("--controllers", nargs="+", default=["r2c", "hybrid"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/confirmatory_e5.csv")
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
        w.writerow(["seed", "mode", "controller", "conv_q"])
        for idx, (m, c, s) in enumerate(tasks):
            w.writerow([s, m, c, f"{res[idx]:.6f}"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
