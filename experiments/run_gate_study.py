"""Gate-design study: how much of the hybrid recovery deficit is metric design?

Same E4 trials (constant jammer, 50% failures), but every step we record
geo_ncomp, gated_ncomp, mean_q. Recovery is then scored under three criteria:
  A) original E4 gate: geometrically connected AND q > 0.5
  B) quality-only:    quality-gated graph connected AND q > 0.5
  C) geometry-only:   geometrically connected (no quality requirement)
Also records t_rec under A and B to expose horizon effects.
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
from r2c.rf import RFLayer, RFParams

T1 = 100


def trial(controller, seed):
    p = R2CParams(seed=seed)
    rf_p = RFParams(jammer_mode="constant", seed=seed)
    env = SwarmEnvironment(p, p.n_random, "random")
    rf = RFLayer(p, rf_p)
    if controller == "R2C":
        ctrl = R2CController(p)
    else:
        ctrl = HybridController(p, rf_p=rf_p, variant=controller)
    rec = {"A": None, "B": None, "C": None}
    t_rec = {"A": None, "B": None}
    q_at_geo1 = []
    for t in range(p.T):
        if t == T1:
            env.inject_failures(0.5)
        env.step(ctrl.velocity(env, t))
        if t > T1:
            g = rf.gated_adjacency(env.pos, env.active, t)
            ids = np.where(env.active)[0]
            sub = np.linalg.norm(env.pos[ids][:, None] - env.pos[ids][None],
                                 axis=-1)
            geo1 = cg.connected_components((sub <= p.r_c).astype(float),
                                           directed=False)[0] == 1
            gsize = int(g[np.ix_(ids, ids)].sum())
            gated1 = gsize > 0 and cg.connected_components(
                g[np.ix_(ids, ids)], directed=False)[0] == 1
            q = rf.mean_quality(env.pos, env.active, t) if gsize else 0.0
            condA = geo1 and gsize > 0 and q > 0.5
            condB = gated1 and q > 0.5
            if geo1:
                q_at_geo1.append(q)
            for key, cond in (("A", condA), ("B", condB), ("C", geo1)):
                if rec[key] is None and cond:
                    rec[key] = True
                    if key in t_rec:
                        t_rec[key] = t - T1
    out = {f"rec{k}": float(rec[k] or False) for k in rec}
    out.update({f"trec{k}": float(t_rec[k]) if t_rec[k] is not None else np.nan
                for k in t_rec})
    out["q_when_geo1"] = float(np.mean(q_at_geo1)) if q_at_geo1 else np.nan
    return out


def _one(task):
    controller, seed = task
    return trial(controller, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--controllers", nargs="+", default=["R2C", "v2", "v3b"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/gate_study.csv")
    args = ap.parse_args()

    tasks = [(c, i) for c in args.controllers for i in range(args.trials)]
    results = {}
    if args.workers > 1:
        from experiments._parallel import map_trials
        tmp = {}
        map_trials(_one, tasks, args.workers, lambda i, r: tmp.__setitem__(i, r))
        results = tmp
    else:
        for idx, task in enumerate(tasks):
            results[idx] = _one(task)
    groups = {}
    for idx, (c, _i) in enumerate(tasks):
        groups.setdefault(c, []).append(results[idx])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["controller", "metric", "mean", "std", "trials"])
        for c, rows in sorted(groups.items()):
            for k in rows[0]:
                a = np.array([r[k] for r in rows], dtype=float)
                w.writerow([c, k, f"{np.nanmean(a):.6f}",
                            f"{np.nanstd(a):.6f}", len(rows)])
    for c, rows in sorted(groups.items()):
        line = f"[{c:4s}]"
        for k in ("recA", "recB", "recC", "trecA", "trecB", "q_when_geo1"):
            a = np.array([r[k] for r in rows], dtype=float)
            line += f" {k}={np.nanmean(a):.3f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
