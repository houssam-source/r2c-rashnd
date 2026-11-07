"""Confirmatory S2: v8 vs v2 on coverage rebuild. Per-trial paired rows."""
import argparse, csv, os, sys
import numpy as np
import scipy.sparse.csgraph as cg
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from r2c.config import R2CParams
from r2c.controller import R2CController
from r2c.environment import SwarmEnvironment
from r2c.hybrid import HybridController
from r2c.metrics import algebraic_connectivity
from r2c.rf import RFLayer, RFParams

GRID = 40


def coverage(pos, active, r_c, area):
    ax = np.linspace(0, area, GRID)
    gx, gy = np.meshgrid(ax, ax)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    pts = pos[active]
    if len(pts) == 0:
        return 0.0
    d = np.linalg.norm(grid[:, None] - pts[None], axis=-1)
    return float((d <= r_c).any(axis=1).mean())


def one(task):
    c, seed = task
    p = R2CParams(seed=seed)
    rf_p = RFParams(jammer_mode="constant", seed=seed)
    env = SwarmEnvironment(p, p.n_random, "random")
    rf = RFLayer(p, rf_p)
    ctrl = (R2CController(p) if c == "R2C"
            else HybridController(p, rf_p=rf_p, variant=c))
    t_rec, recovered = None, False
    thr_sum = q_sum = steps = 0.0
    cov_end = cov_post_sum = cov_post_n = 0.0
    for t in range(p.T):
        if t == p.t1:
            env.inject_failures(0.5)
        env.step(ctrl.velocity(env, t))
        if t > p.t1:
            thr_sum += rf.throughput(env.pos, env.active, t)
            q_sum += rf.mean_quality(env.pos, env.active, t)
            steps += 1
            c_t = coverage(env.pos, env.active, p.r_c, p.area)
            cov_post_sum += c_t
            cov_post_n += 1
            if not recovered and rf.gated_adjacency(
                    env.pos, env.active, t).size and env.connected() and                     rf.mean_quality(env.pos, env.active, t) > 0.5:
                t_rec = t - p.t1
                recovered = True
    cov_end = coverage(env.pos, env.active, p.r_c, p.area)
    return {"recovered": float(recovered),
            "REC": (t_rec / p.T) if recovered else 1.0,
            "thr_mbps": thr_sum / max(steps, 1),
            "mean_q": q_sum / max(steps, 1),
            "cov_end": cov_end,
            "cov_post_mean": cov_post_sum / max(cov_post_n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--controllers", nargs="+", default=["v2", "v8"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/confirmatory_v8.csv")
    args = ap.parse_args()
    tasks = [(c, i) for c in args.controllers for i in range(args.trials)]
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
                    "thr_mbps", "mean_q", "cov_end", "cov_post_mean"])
        for idx, (c, s) in enumerate(tasks):
            r = res[idx]
            w.writerow([s, "constant", c, f"{r['recovered']:.0f}",
                        f"{r['REC']:.4f}", f"{r['thr_mbps']:.3f}",
                        f"{r['mean_q']:.4f}", f"{r['cov_end']:.4f}",
                        f"{r['cov_post_mean']:.4f}"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
