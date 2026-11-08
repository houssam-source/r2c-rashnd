"""E4 variant study: v2 / v3a / v3b / v3 (+ R2C baseline) under jamming.

Replicates experiments/run_jamming.py's trial loop with per-variant hybrid
construction and adds controller counters (sensing events, re-plans) plus the
quality-gated recovery metric. Writes results/e4_variants.csv.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scipy.sparse.csgraph as cg

from r2c.config import R2CParams
from r2c.controller import R2CController
from r2c.environment import SwarmEnvironment
from r2c.hybrid import HybridController
from r2c.metrics import algebraic_connectivity
from r2c.rf import RFLayer, RFParams


def trial(p, rf_p, failure_ratio, controller, variant):
    env = SwarmEnvironment(p, p.n_random, "random")
    rf = RFLayer(p, rf_p)
    if controller == "field" or variant == "field":
        from r2c.field import FieldController
        ctrl = FieldController(p, rf_p, seed=p.seed)
    elif controller == "hybrid":
        ctrl = HybridController(p, rf_p=rf_p, variant=variant)
    else:
        ctrl = R2CController(p)
    t_rec, recovered = None, False
    thr_sum, q_sum, steps = 0.0, 0.0, 0
    for t in range(p.T):
        if t == p.t1:
            env.inject_failures(failure_ratio)
        env.step(ctrl.velocity(env, t))
        if t > p.t1:
            thr_sum += rf.throughput(env.pos, env.active, t)
            q_sum += rf.mean_quality(env.pos, env.active, t)
            steps += 1
            if not recovered and rf.gated_adjacency(
                    env.pos, env.active, t).size and env.connected() and \
                    rf.mean_quality(env.pos, env.active, t) > 0.5:
                t_rec = t - p.t1
                recovered = True
    gadj = rf.gated_adjacency(env.pos, env.active, p.T - 1)
    out = {
        "REC": (t_rec / p.T) if recovered else 1.0,
        "recovered": float(recovered),
        "thr_mbps": thr_sum / max(steps, 1),
        "mean_q": q_sum / max(steps, 1),
        "lambda2_gated": algebraic_connectivity(gadj),
    }
    if hasattr(ctrl, "counters"):
        out["sensing_events"] = float(ctrl.counters["sensing_events"])
        out["replans"] = float(ctrl.counters["replans"])
    else:
        out["sensing_events"] = np.nan
        out["replans"] = np.nan
    return out


def _one(task):
    """Top-level worker for multiprocessing: (controller, mode, i, fails)."""
    c, mode, i, fails = task
    pi = R2CParams(**{**R2CParams().__dict__, "seed": i})
    ri = RFParams(jammer_mode=mode, seed=i)
    if c == "R2C":
        return trial(pi, ri, fails, "r2c", "")
    return trial(pi, ri, fails, "hybrid", c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--modes", nargs="+",
                    default=["constant", "sweep", "reactive"])
    ap.add_argument("--controllers", nargs="+",
                    default=["R2C", "v2", "v3a", "v3b", "v3"])
    ap.add_argument("--failures", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/e4_variants.csv")
    args = ap.parse_args()

    tasks = [(c, m, i, args.failures) for m in args.modes
             for c in args.controllers for i in range(args.trials)]
    results = {}
    if args.workers > 1:
        from experiments._parallel import map_trials

        def on_res(idx, res):
            tmp[idx] = res
        tmp = {}
        map_trials(_one, tasks, args.workers, on_res)
        results = tmp
    else:
        for idx, task in enumerate(tasks):
            results[idx] = _one(task)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["jammer", "controller", "metric", "mean", "std",
                    "trials"])
        # regroup by (mode, controller)
        groups = {}
        for idx, (c, mode, _i, _f) in enumerate(tasks):
            groups.setdefault((mode, c), []).append(results[idx])
        for (mode, c), rows in sorted(groups.items()):
            for k in rows[0]:
                a = np.array([r[k] for r in rows], dtype=float)
                w.writerow([mode, c, k, f"{np.nanmean(a):.6f}",
                            f"{np.nanstd(a):.6f}", len(rows)])
    for (mode, c), rows in sorted(groups.items()):
        rec = np.nanmean([r["recovered"] for r in rows])
        thr = np.nanmean([r["thr_mbps"] for r in rows])
        q = np.nanmean([r["mean_q"] for r in rows])
        sen = np.nanmean([r["sensing_events"] for r in rows])
        rp = np.nanmean([r["replans"] for r in rows])
        print(f"[{mode:8s} | {c:4s}] rec={rec:.2f} thr={thr:6.0f} "
              f"q={q:.3f} sensing={sen:6.1f} replans={rp:4.1f}", flush=True)


if __name__ == "__main__":
    main()
