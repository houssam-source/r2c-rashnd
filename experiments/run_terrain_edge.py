"""Terrain stress battery: 10 edge cases (x baseline) in the occluded ruins.

blind (v3b, terrain truth hurts it but it can't plan with the map) vs
v4 (camera-construct visibility routing). Same hooks as the flat-terrain
stress battery; recovery requires visibility-connected + q > 0.5.
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
from r2c.rf import RFLayer, RFParams
from r2c.visibility import vis_mask, los_cell_matrix
from uav_swarm_r2c.bridge import SwarmBridge

from experiments.run_terrain_study import make_ruins, AREA, ALT, T, T1, N
from experiments.run_edge_cases import (SCENARIOS, hook_cascading,
                                        hook_quadrant, hook_power_surge,
                                        hook_jammer_move, hook_fast_sweep,
                                        hook_range_loss, hook_rejoin,
                                        hook_extreme)

HOOKS = {
    "baseline":      {"inject": 3 / N, "hook": lambda *a: None},
    "cascading":     {"inject": 3 / N, "hook": hook_cascading},
    "quadrant_kill": {"inject": 3 / N, "hook": hook_quadrant,
                      "skip_inject": True},
    "power_surge":   {"inject": 3 / N, "hook": hook_power_surge},
    "jammer_move":   {"inject": 3 / N, "hook": hook_jammer_move},
    "fast_sweep":    {"inject": 3 / N, "hook": hook_fast_sweep,
                      "sweep_period": 60},
    "noisy_gps":     {"inject": 3 / N, "hook": hook_fast_sweep,
                      "gps_sigma": 5.0},
    "range_loss":    {"inject": 3 / N, "hook": hook_range_loss},
    "rejoin":        {"inject": 3 / N, "hook": hook_rejoin},
    "extreme80":     {"inject": 0.8, "hook": hook_extreme},
    "edge_mission":  {"inject": 3 / N, "hook": hook_fast_sweep,
                      "deploy_offset": (60.0, 60.0)},
}


def _shift(hook, dt=40):
    """Flat-battery hooks are written for t1=100; terrain t1=60 -> shift +40."""
    def wrapped(env, rf, ctrl, t, p, rng):
        hook(env, rf, ctrl, t + dt, p, rng)
    return wrapped


def trial(scenario, controller, seed, ruins, V):
    spec = HOOKS[scenario]
    p = R2CParams(seed=seed, area=AREA, n_random=N, T=T, t1=T1)
    rng = np.random.default_rng(seed)
    if "sweep_period" in spec:
        rf_p = RFParams(jammer_mode="sweep", seed=seed,
                        sweep_period=spec["sweep_period"])
    else:
        rf_p = RFParams(jammer_mode="constant", seed=seed)
    b = SwarmBridge(N, p, controller="hybrid", jammer_mode=rf_p.jammer_mode,
                    seed=seed,
                    variant="v3b" if controller == "blind" else "v4")
    if "sweep_period" in spec:
        b.ctrl.rf.rf.sweep_period = spec["sweep_period"]
    b.set_jammer_pose(AREA / 2, AREA / 2)
    b.ctrl.rf.attach_terrain(ruins, AREA, ALT)
    rf = RFLayer(p, rf_p)
    rf.attach_terrain(ruins, AREA, ALT)

    pos = 40 + rng.uniform(0, 120, size=(N, 2))
    if "deploy_offset" in spec:
        pos = pos * 0.3 + np.array(spec["deploy_offset"])
    b.update_poses(pos)
    b.inject_failures(0.0)

    sigma = spec.get("gps_sigma", 0.0)
    t_rec, recovered = None, False
    thr_sum, q_sum, steps = 0.0, 0.0, 0
    for t in range(T):
        if t == T1 and not spec.get("skip_inject"):
            b.inject_failures(spec["inject"], rng)
        _shift(spec["hook"])(b.env, rf, b.ctrl, t, p, rng)
        if sigma > 0:
            true = b.env.pos.copy()
            b.env.pos = true + rng.normal(0, sigma, true.shape)
            b.env._dist_cache = None
            b.env._adj_cache = None
            v = b.compute()
            b.env.pos = true
        else:
            v = b.compute()
        a = b.env.active
        pos = pos.copy()
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
                geo1 = cg.connected_components(((sub <= p.r_c) & vis
                                                  ).astype(float),
                                                 directed=False)[0] == 1
                g = rf.gated_adjacency(pos, a, t)[np.ix_(ids, ids)]
                q = rf.mean_quality(pos, a, t)
                if not recovered and geo1 and g.sum() > 0 and q > 0.5:
                    t_rec = t - T1
                    recovered = True
    return {
        "recovered": float(recovered),
        "REC": (t_rec / T) if recovered else 1.0,
        "thr_mbps": thr_sum / max(steps, 1),
        "mean_q": q_sum / max(steps, 1),
    }


def _one(task):
    scenario, controller, seed, ruins, V = task
    return trial(scenario, controller, seed, ruins, V)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--scenarios", nargs="+", default=list(HOOKS))
    ap.add_argument("--controllers", nargs="+", default=["blind", "v4"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/terrain_edge.csv")
    args = ap.parse_args()

    ruins = make_ruins()
    V = los_cell_matrix(ruins, AREA, ALT)
    tasks = [(s, c, i, ruins, V) for s in args.scenarios
             for c in args.controllers for i in range(args.trials)]
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
    for idx, (s, c, _i, _r, _v) in enumerate(tasks):
        groups.setdefault((s, c), []).append(results[idx])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "controller", "metric", "mean", "std",
                    "trials"])
        for (s, c), rows in sorted(groups.items()):
            for k in rows[0]:
                a = np.array([r[k] for r in rows], dtype=float)
                w.writerow([s, c, k, f"{np.nanmean(a):.6f}",
                            f"{np.nanstd(a):.6f}", len(rows)])
    for (s, c), rows in sorted(groups.items()):
        rec = np.nanmean([r["recovered"] for r in rows])
        thr = np.nanmean([r["thr_mbps"] for r in rows])
        q = np.nanmean([r["mean_q"] for r in rows])
        print(f"[{s:14s} | {c:5s}] rec={rec:.2f} thr={thr:6.0f} q={q:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
