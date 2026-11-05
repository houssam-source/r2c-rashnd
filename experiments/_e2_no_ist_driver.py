"""E2 no_IST ablation driver (agent-3 resilience wrapper).

Executes EXACTLY the protocol of experiments/run_experiment.py
(--trials 50 --ablation no_ist --failures 20 40 50 60 80, both deployments)
but persists per-trial results to a checkpoint after every trial and
rewrites results/e2_no_ist.csv (same schema as run_experiment.py) from the
checkpoint on every restart. Fleet management on this machine periodically
kills python processes and deletes results CSVs; this driver survives that
by resuming from results/.e2_no_ist_ckpt.json.

Trial seeding, controller construction, and aggregation are copied
line-for-line from r2c/metrics.py::run_metrics so results are identical
to the stock runner.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from r2c.config import R2CParams
from r2c.controller import R2CController
from r2c.metrics import run_trial

CKPT = "results/.e2_no_ist_ckpt.json"
OUT = "results/e2_no_ist.csv"
METRICS = ["REC", "COV", "AND", "lambda2_final", "recovered"]
TRIAL_KEYS = ["REC", "COV", "COV_final", "AND", "lambda2_final", "recovered"]


def cell_key(dep, fr):
    return f"{dep}|{fr}"


def run_cell(p, dep, fr, trials, ckpt, use_ist, use_awf, use_ise):
    k = cell_key(dep, fr)
    cell = ckpt.setdefault(k, {})
    done = len(cell)
    for i in range(done, trials):
        pi = R2CParams(**{**p.__dict__, "seed": p.seed + i})
        ctrl = R2CController(pi, use_ist=use_ist, use_awf=use_awf)
        r = run_trial(pi, fr / 100.0, dep, ctrl, use_ise=use_ise)
        cell[str(i)] = {key: float(r[key]) for key in TRIAL_KEYS}
        save_ckpt(ckpt)
        print(f"[{k}] trial {i + 1}/{trials} REC={r['REC']:.3f} "
              f"recovered={r['recovered']:.0f}", flush=True)


def save_ckpt(ckpt):
    tmp = CKPT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ckpt, f)
    os.replace(tmp, CKPT)


def write_csv(ckpt, failures, deployments, trials, ablation):
    rows = []
    for dep in deployments:
        for fr in failures:
            cell = ckpt.get(cell_key(dep, fr), {})
            if len(cell) < trials:
                continue
            stats = {}
            for key in TRIAL_KEYS:
                a = np.array([cell[str(i)][key] for i in range(trials)])
                stats[key] = (float(a.mean()), float(a.std()))
            print(f"[{dep} | failure {fr}% | ablation={ablation}] (from ckpt)")
            for key in METRICS:
                m, s = stats[key]
                print(f"  {key:14s} = {m:.4f} +- {s:.4f}")
                rows.append([dep, fr, ablation, key, f"{m:.6f}",
                             f"{s:.6f}", trials])
    tmp = OUT + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["deployment", "failure_pct", "ablation", "metric",
                    "mean", "std", "trials"])
        w.writerows(rows)
    os.replace(tmp, OUT)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--failures", type=float, nargs="+",
                    default=[20, 40, 50, 60, 80])
    ap.add_argument("--deployments", nargs="+", default=["random", "grid"])
    ap.add_argument("--ablation", default="no_ist")
    args = ap.parse_args()

    p = R2CParams()
    os.makedirs("results", exist_ok=True)
    ckpt = {}
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            ckpt = json.load(f)

    use_ist = args.ablation != "no_ist"
    use_awf = args.ablation != "no_awf"
    use_ise = args.ablation != "no_ise"

    for dep in args.deployments:
        for fr in args.failures:
            run_cell(p, dep, fr, args.trials, ckpt, use_ist, use_awf, use_ise)
            n = write_csv(ckpt, args.failures, args.deployments,
                          args.trials, args.ablation)
            print(f"csv synced: {n} data rows", flush=True)

    n = write_csv(ckpt, args.failures, args.deployments,
                  args.trials, args.ablation)
    print(f"DONE: {n} data rows in {OUT}")


if __name__ == "__main__":
    main()
