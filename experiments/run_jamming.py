"""E4: R2C alone vs R2C+RASHND hybrid under jamming (guide §6, headline exp).

For each jammer mode (constant / sweep / reactive): both controllers run
the failure+jamming protocol; report connectivity recovery AND achieved
throughput. R2C alone should "recover" geometrically unusable networks;
the hybrid should recover working (quality-gated) ones.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2c.config import R2CParams
from r2c.controller import R2CController
from r2c.environment import SwarmEnvironment
from r2c.hybrid import HybridController
from r2c.metrics import algebraic_connectivity
from r2c.rf import RFLayer, RFParams


def run_jamming_trial(p: R2CParams, rf_p: RFParams, failure_ratio: float,
                      deployment: str, hybrid: bool) -> dict:
    env = SwarmEnvironment(p, p.n_random if deployment == "random" else p.n_grid,
                           deployment)
    rf = RFLayer(p, rf_p)
    ctrl = HybridController(p, rf) if hybrid else R2CController(p)
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
    return {
        "REC": (t_rec / p.T) if recovered else 1.0,
        "recovered": float(recovered),
        "thr_mbps": thr_sum / max(steps, 1),
        "mean_q": q_sum / max(steps, 1),
        "lambda2_gated": algebraic_connectivity(gadj),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="offset added to the base seed per trial "
                         "(allows resumable chunks appending to one CSV)")
    ap.add_argument("--failures", type=float, default=0.5)
    ap.add_argument("--modes", nargs="+",
                    default=["constant", "sweep", "reactive"])
    ap.add_argument("--out", default="results/e4.csv")
    args = ap.parse_args()

    p = R2CParams()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["jammer", "controller", "metric", "mean", "std", "trials"])
        for mode in args.modes:
            for hybrid in [False, True]:
                name = "R2C+RASHND" if hybrid else "R2C"
                rows = []
                for i in range(args.trials):
                    pi = R2CParams(**{**p.__dict__,
                                      "seed": p.seed + args.seed_offset + i})
                    ri = RFParams(jammer_mode=mode,
                                  seed=p.seed + args.seed_offset + i)
                    rows.append(run_jamming_trial(pi, ri, args.failures,
                                                  "random", hybrid))
                print(f"[jammer={mode} | {name}]")
                for k in rows[0]:
                    a = np.array([r[k] for r in rows])
                    print(f"  {k:14s} = {a.mean():.4f} +- {a.std():.4f}")
                    w.writerow([mode, name, k, f"{a.mean():.6f}",
                                f"{a.std():.6f}", args.trials])
                f.flush()


if __name__ == "__main__":
    main()
