"""E1/E3: R2C (+ hybrid) vs the 8 baselines under the paper protocol.

Runs every controller through the same failure protocol and reports
REC / COV / AND / lambda2 (E1 comparison) plus a CDD snapshot (E3).
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2c.baselines import BASELINES
from r2c.config import R2CParams
from r2c.controller import R2CController
from r2c.metrics import run_trial


def _one_trial(args):
    """Worker: run a single trial; must be top-level for multiprocessing."""
    d, failure, deployment, name, seed_offset, i = args
    pi = R2CParams(**{**d, "seed": d["seed"] + seed_offset + i})
    ctrl = R2CController(pi) if name == "R2C" else BASELINES[name](pi)
    return run_trial(pi, failure, deployment, ctrl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--failure", type=float, default=0.5)
    ap.add_argument("--deployment", default="random",
                    choices=["random", "grid"])
    ap.add_argument("--controllers", nargs="+",
                    default=["R2C"] + list(BASELINES))
    ap.add_argument("--out", default="results/baselines.csv")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="offset added to the base seed for each trial index")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel trial processes (1 = serial; results are "
                         "identical either way since every trial is seeded)")
    args = ap.parse_args()

    from experiments._parallel import map_trials

    p = R2CParams()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["controller", "failure_pct", "metric", "mean", "std",
                        "trials"])
        for name in args.controllers:
            base = dict(p.__dict__)
            tasks = [(base, args.failure, args.deployment, name,
                      args.seed_offset, i) for i in range(args.trials)]
            rows = [None] * args.trials
            map_trials(_one_trial, tasks, args.workers,
                       lambda i, r: rows.__setitem__(i, r))
            print(f"[{name} | {args.deployment} | failure {args.failure:.0%}]")
            for k in ["REC", "COV", "AND", "lambda2_final", "recovered"]:
                a = np.array([r[k] for r in rows])
                print(f"  {k:14s} = {a.mean():.4f} +- {a.std():.4f}")
                w.writerow([name, args.failure * 100, k, f"{a.mean():.6f}",
                            f"{a.std():.6f}", args.trials])
            cdds = [r["CDD"] for r in rows]
            m = max(len(x) for x in cdds)
            cdd = np.mean([np.pad(x, (0, m - len(x)), constant_values=1.0)
                           for x in cdds], axis=0)
            w.writerow([name, args.failure * 100, "CDD", "|".join(
                f"{x:.4f}" for x in cdd), "", args.trials])
            f.flush()


if __name__ == "__main__":
    main()
