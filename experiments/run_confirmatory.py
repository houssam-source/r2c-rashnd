"""Confirmatory runner: v4 vs blind (v3b), occluded post-disaster, paired
seeds. Writes PER-TRIAL rows (seed-pairable) to results/confirmatory_v4.csv.
Analysis lives in build/analyze_confirmatory.py."""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "gazebo", "uav_swarm_r2c"))

from experiments.run_terrain_study import make_ruins, trial  # noqa: E402
from r2c.visibility import los_cell_matrix  # noqa: E402
from experiments.run_terrain_study import AREA, ALT  # noqa: E402


def _one(task):
    controller, seed, ruins, V = task
    return trial(controller, seed, ruins, V)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--controllers", nargs="+", default=["blind", "v4"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/confirmatory_v4.csv")
    args = ap.parse_args()

    ruins = make_ruins()
    V = los_cell_matrix(ruins, AREA, ALT)
    tasks = [(c, i, ruins, V) for c in args.controllers
             for i in range(args.trials)]
    results = {}
    if args.workers > 1:
        from experiments._parallel import map_trials
        tmp = {}
        map_trials(_one, tasks, args.workers, lambda i, r: tmp.__setitem__(i, r))
        results = tmp
    else:
        for idx, task in enumerate(tasks):
            results[idx] = _one(task)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "controller", "recovered", "REC", "thr_mbps",
                    "mean_q"])
        for idx, (c, seed, _r, _v) in enumerate(tasks):
            r = results[idx]
            w.writerow([seed, c, f"{r['recovered']:.0f}", f"{r['REC']:.4f}",
                        f"{r['thr_mbps']:.3f}", f"{r['mean_q']:.4f}"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
