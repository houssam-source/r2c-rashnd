"""Stress battery: 10 frequent edge cases x {R2C, v2, v3b} under jamming.

Each scenario modifies the E4 trial loop with one realistic adversity. All
mutations hit BOTH the metrics RFLayer and the controller's own RF layer.
Reports gated recovery, throughput, quality, final components.
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

T1 = 100


# --------------------------------------------------------------------------
# scenario hooks: mutate(env, rf, ctrl, t, p, rng) -> extra_failed_count (int)
# --------------------------------------------------------------------------
def apply_inject(env, ratio, rng, mode="uniform"):
    before = int((~env.active).sum())
    if mode == "uniform":
        env.inject_failures(ratio, rng)
    elif mode == "quadrant":
        # spatially correlated: kill 70% of drones in the SW quadrant
        idx = np.where(env.active & (env.pos[:, 0] < 500)
                       & (env.pos[:, 1] < 500))[0]
        k = int(round(0.7 * len(idx)))
        if k:
            env.active[rng.choice(idx, size=k, replace=False)] = False
        rest = int(round(ratio * 200)) - (int((~env.active).sum()) - before)
        if rest > 0:
            env.inject_failures(rest / max(env.active.sum(), 1))
    return int((~env.active).sum()) - before


def hook_cascading(env, rf, ctrl, t, p, rng):
    if t >= T1 and (t - T1) % 40 == 0:
        env.inject_failures(0.05 / max(env.active.sum() / 200.0, 0.2))


def hook_quadrant(env, rf, ctrl, t, p, rng):
    if t == T1:
        apply_inject(env, 0.5, rng, mode="quadrant")


def hook_power_surge(env, rf, ctrl, t, p, rng):
    if t == 200:
        for r in (rf, getattr(ctrl, "rf", None)):
            if r is not None:
                r.rf.jammer_power_dbm = 40.0


def hook_jammer_move(env, rf, ctrl, t, p, rng):
    if t == 200:
        for r in (rf, getattr(ctrl, "rf", None)):
            if r is not None:
                r.jammer_pos = np.array([250.0, 750.0])


def hook_fast_sweep(env, rf, ctrl, t, p, rng):
    pass  # configured pre-run (sweep_period=60)


def hook_noisy_gps(env, rf, ctrl, t, p, rng):
    return 0  # handled in trial loop via sigma


def hook_range_loss(env, rf, ctrl, t, p, rng):
    if t == 200:
        p.r_c = 80.0
        p.d_lo, p.d_hi = 0.8 * 80.0, 0.9 * 80.0
        env.link.r_link = p.r_c   # degraded link budget: 80 m range
        env._dist_cache = None
        env._adj_cache = None


def hook_rejoin(env, rf, ctrl, t, p, rng):
    if t == 250:
        dead = np.where(~env.active)[0]
        back = dead[: len(dead) // 2]
        env.active[back] = True
        env._dist_cache = None
        env._adj_cache = None


def hook_extreme(env, rf, ctrl, t, p, rng):
    if t == T1:
        env.inject_failures(0.8)


def hook_edge_mission(env, rf, ctrl, t, p, rng):
    pass  # configured pre-run (deployment offset)


SCENARIOS = {
    "cascading":     {"inject": 0.5, "hook": hook_cascading},
    "quadrant_kill": {"inject": 0.5, "hook": hook_quadrant},
    "power_surge":   {"inject": 0.5, "hook": hook_power_surge},
    "jammer_move":   {"inject": 0.5, "hook": hook_jammer_move},
    "fast_sweep":    {"inject": 0.5, "hook": hook_fast_sweep,
                      "sweep_period": 60},
    "noisy_gps":     {"inject": 0.5, "hook": hook_noisy_gps, "gps_sigma": 5.0},
    "range_loss":    {"inject": 0.5, "hook": hook_range_loss},
    "rejoin":        {"inject": 0.5, "hook": hook_rejoin},
    "extreme80":     {"inject": 0.8, "hook": hook_extreme},
    "edge_mission":  {"inject": 0.5, "hook": hook_edge_mission,
                      "deploy_offset": (60.0, 60.0)},
}


def trial(scenario, controller, seed):
    spec = SCENARIOS[scenario]
    p = R2CParams(seed=seed)
    if "sweep_period" in spec:
        rf_p = RFParams(jammer_mode="sweep", seed=seed,
                        sweep_period=spec["sweep_period"])
    else:
        rf_p = RFParams(jammer_mode="constant", seed=seed)
    env = SwarmEnvironment(p, p.n_random, "random")
    if "deploy_offset" in spec:
        env.pos = env.pos * 0.3 + np.array(spec["deploy_offset"])
    rf = RFLayer(p, rf_p)
    if controller == "R2C":
        ctrl = R2CController(p)
    else:
        ctrl = HybridController(p, rf_p=rf_p, variant=controller)
    rng = np.random.default_rng(seed)
    sigma = spec.get("gps_sigma", 0.0)

    t_rec, recovered = None, False
    thr_sum, q_sum, steps = 0.0, 0.0, 0
    for t in range(p.T):
        if t == T1:
            env.inject_failures(spec["inject"])
        spec["hook"](env, rf, ctrl, t, p, rng)
        if sigma > 0:
            true = env.pos.copy()
            env.pos = true + rng.normal(0, sigma, true.shape)
            env._dist_cache = None
            env._adj_cache = None
            v = ctrl.velocity(env, t)
            env.pos = true
        else:
            v = ctrl.velocity(env, t)
        env.step(v)
        if t > T1:
            thr_sum += rf.throughput(env.pos, env.active, t)
            q_sum += rf.mean_quality(env.pos, env.active, t)
            steps += 1
            if not recovered and rf.gated_adjacency(
                    env.pos, env.active, t).size and env.connected() and \
                    rf.mean_quality(env.pos, env.active, t) > 0.5:
                t_rec = t - T1
                recovered = True
    a = env.active
    ncomp = cg.connected_components(env.adjacency(), directed=False)[0]
    return {
        "REC": (t_rec / p.T) if recovered else 1.0,
        "recovered": float(recovered),
        "thr_mbps": thr_sum / max(steps, 1),
        "mean_q": q_sum / max(steps, 1),
        "final_components": float(ncomp),
    }


def _one(task):
    scenario, controller, seed = task
    return trial(scenario, controller, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--controllers", nargs="+", default=["R2C", "v2", "v3b"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/edge_cases.csv")
    args = ap.parse_args()

    tasks = [(s, c, i) for s in args.scenarios for c in args.controllers
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
    for idx, (s, c, _i) in enumerate(tasks):
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
        nc = np.nanmean([r["final_components"] for r in rows])
        print(f"[{s:14s} | {c:4s}] rec={rec:.2f} thr={thr:6.0f} "
              f"q={q:.3f} final_comp={nc:5.1f}", flush=True)


if __name__ == "__main__":
    main()
