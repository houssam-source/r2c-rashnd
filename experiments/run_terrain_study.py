"""Terrain study: camera-construct (visibility) knowledge routing, v4 vs v3b.

Post-disaster scenario (400 m ruins, 10 drones, 3 crashes at t1=60, damaged
tower = constant jammer at center). The terrain TRUTH blocks LOS links in the
RF layer for all controllers; only v4 sees the shared camera-construct and
clusters/bridges on the visibility graph.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "gazebo", "uav_swarm_r2c"))

import scipy.sparse.csgraph as cg

from r2c.config import R2CParams
from r2c.hybrid import HybridController
from r2c.metrics import algebraic_connectivity
from r2c.rf import RFLayer, RFParams
from r2c.visibility import vis_mask, los_cell_matrix
from uav_swarm_r2c.bridge import SwarmBridge, _BareEnv

AREA, ALT = 400.0, 30.0
T, T1, N = 300, 60, 10


def make_ruins(cells=20, seed=5):
    rng = np.random.default_rng(seed)
    h = np.round(rng.uniform(1.5, 8.0, size=(cells, cells))) * 4.0
    damaged = rng.random((cells, cells)) < 0.22
    h[damaged] = np.round(
        h[damaged] * rng.uniform(0.15, 0.45, damaged.sum()) / 4.0) * 4.0
    h[::5, :] = 0.0
    h[:, ::5] = 0.0
    return h


def trial(controller, seed, ruins, V):
    p = R2CParams(seed=seed, area=AREA, n_random=N, T=T, t1=T1)
    rng = np.random.default_rng(seed)
    if controller == "R2C":
        b = SwarmBridge(N, p, controller="r2c", jammer_mode="constant",
                        seed=seed)
    else:
        b = SwarmBridge(N, p, controller="hybrid", jammer_mode="constant",
                        seed=seed,
                        variant="v3b" if controller == "blind" else "v4")
    if controller != "R2C":
        b.ctrl.rf.attach_terrain(ruins, AREA, ALT)
    tower = np.array([AREA / 2, AREA / 2])
    b.set_jammer_pose(*tower)
    # terrain truth on the metrics RF layer (and the controller's rf above)
    rf = RFLayer(p, RFParams(jammer_mode="constant", seed=seed))
    rf.attach_terrain(ruins, AREA, ALT)

    pos = 40 + rng.uniform(0, 120, size=(N, 2))
    b.update_poses(pos)
    b.inject_failures(0.0)

    t_rec, recovered = None, False
    thr_sum, q_sum, steps = 0.0, 0.0, 0
    for t in range(T):
        if t == T1:
            b.inject_failures(3 / N, rng)
        v = b.compute()
        a = b.env.active
        pos[a] += v[a]
        np.clip(pos, 0, AREA, out=pos)
        b.update_poses(pos)
        if t > T1:
            thr_sum += rf.throughput(pos, a, t)
            q_sum += rf.mean_quality(pos, a, t)
            steps += 1
            ids = np.where(a)[0]
            if len(ids) > 1:
                sub = np.linalg.norm(pos[ids][:, None] - pos[ids][None],
                                     axis=-1)
                vis = vis_mask(pos[ids], V, AREA)
                geo1 = (cg.connected_components(((sub <= p.r_c) & vis
                                                 ).astype(float),
                                                directed=False)[0] == 1)
                g = rf.gated_adjacency(pos, a, t)[np.ix_(ids, ids)]
                q = rf.mean_quality(pos, a, t)
                if (not recovered and geo1 and g.sum() > 0 and q > 0.5):
                    t_rec = t - T1
                    recovered = True
    return {
        "recovered": float(recovered),
        "REC": (t_rec / T) if recovered else 1.0,
        "thr_mbps": thr_sum / max(steps, 1),
        "mean_q": q_sum / max(steps, 1),
    }


def _one(task):
    controller, seed, ruins, V = task
    return trial(controller, seed, ruins, V)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--controllers", nargs="+", default=["blind", "v4"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/terrain_study.csv")
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
    groups = {}
    for idx, (c, _i, _r, _v) in enumerate(tasks):
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
        rec = np.nanmean([r["recovered"] for r in rows])
        thr = np.nanmean([r["thr_mbps"] for r in rows])
        q = np.nanmean([r["mean_q"] for r in rows])
        tr = np.nanmean([r["REC"] for r in rows])
        print(f"[{c:5s}] rec={rec:.2f} REC={tr:.3f} thr={thr:6.0f} "
              f"q={q:.3f}", flush=True)


if __name__ == "__main__":
    main()
