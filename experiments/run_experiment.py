"""E1: R2C reproduction protocol (guide §4/§6).

Both deployments x failure ratios {20,40,50,60,80}% x N trials; reports
mean +- std of REC/COV/AND/lambda2. CSV is appended to results/e1.csv.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2c.config import R2CParams
from r2c.metrics import run_metrics


def checkpointed_run(p, fr, dep, trials, ablation, ckpt_path):
    """run_metrics equivalent that saves per-trial results to a JSON file
    after every trial so a killed run can resume exactly. Numerically
    identical to run_metrics(p, fr, dep, trials, ablation)."""
    keys = ["REC", "COV", "COV_final", "AND", "lambda2_final", "recovered"]
    ckpt = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
    for i in range(trials):
        if str(i) in ckpt:
            continue
        pi = R2CParams(**{**p.__dict__, "seed": p.seed + i})
        r = run_metrics(pi, fr, dep, trials=1, ablation=ablation)
        ckpt[str(i)] = {k: r[k][0] for k in keys}
        tmp = ckpt_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(ckpt, f)
        os.replace(tmp, ckpt_path)
    out = {}
    for k in keys:
        a = np.array([ckpt[str(i)][k] for i in range(trials)])
        out[k] = (float(a.mean()), float(a.std()))
    out["CDD_example"] = np.float64(0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--failures", type=float, nargs="+",
                    default=[20, 40, 50, 60, 80])
    ap.add_argument("--deployments", nargs="+",
                    default=["random", "grid"], choices=["random", "grid"])
    ap.add_argument("--ablation", default=None,
                    choices=[None, "no_ist", "no_ise", "no_awf"])
    ap.add_argument("--out", default="results/e1.csv")
    ap.add_argument("--checkpoint", default=None,
                    help="optional JSON path for per-trial resume; identical "
                         "results to the default non-checkpointed run")
    args = ap.parse_args()

    p = R2CParams()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["deployment", "failure_pct", "ablation", "metric",
                        "mean", "std", "trials"])
        for dep in args.deployments:
            for fr in args.failures:
                if args.checkpoint:
                    ckpt_path = args.checkpoint.replace(
                        "{fr}", str(int(fr))).replace("{dep}", dep)
                    r = checkpointed_run(p, fr / 100.0, dep, args.trials,
                                         args.ablation, ckpt_path)
                else:
                    r = run_metrics(p, fr / 100.0, dep, trials=args.trials,
                                    ablation=args.ablation)
                print(f"[{dep} | failure {fr}% | ablation={args.ablation}]")
                for k in ["REC", "COV", "AND", "lambda2_final", "recovered"]:
                    m, s = r[k]
                    print(f"  {k:14s} = {m:.4f} +- {s:.4f}")
                    w.writerow([dep, fr, args.ablation, k, f"{m:.6f}",
                                f"{s:.6f}", args.trials])
                f.flush()


if __name__ == "__main__":
    main()
