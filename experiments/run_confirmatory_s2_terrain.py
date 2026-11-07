"""S2 terrain: v8 vs v2 coverage rebuild in the occluded post-disaster
scenario (10 drones) - the regime with a REAL persistent coverage deficit."""
import argparse, csv, os, sys
import numpy as np
import scipy.sparse.csgraph as cg
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "gazebo", "uav_swarm_r2c"))
from experiments.run_terrain_study import make_ruins, AREA, ALT, T, T1, N
from r2c.visibility import los_cell_matrix, vis_mask
from r2c.rf import RFLayer, RFParams
from uav_swarm_r2c.bridge import SwarmBridge

GRID = 20


def coverage(pos, active, r_c, area, V=None):
    ax = np.linspace(0, area, GRID)
    gx, gy = np.meshgrid(ax, ax)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    pts = pos[active]
    if len(pts) == 0:
        return 0.0
    d = np.linalg.norm(grid[:, None] - pts[None], axis=-1)
    return float((d <= r_c).any(axis=1).mean())


def one(task):
    c, seed, ruins, V = task
    p = __import__("r2c.config", fromlist=["R2CParams"]).R2CParams(
        seed=seed, area=AREA, n_random=N, T=T, t1=T1)
    rng = np.random.default_rng(seed)
    b = SwarmBridge(N, p, controller="hybrid", jammer_mode="constant",
                    seed=seed, variant=c)
    b.set_jammer_pose(AREA / 2, AREA / 2)
    b.ctrl.rf.attach_terrain(ruins, AREA, ALT)
    rf = RFLayer(p, RFParams(jammer_mode="constant", seed=seed))
    rf.attach_terrain(ruins, AREA, ALT)
    pos = 40 + rng.uniform(0, 120, size=(N, 2))
    b.update_poses(pos)
    b.inject_failures(0.0)
    t_rec, recovered = None, False
    thr_sum = q_sum = steps = 0.0
    cov_post_sum = cov_post_n = 0.0
    for t in range(T):
        if t == T1:
            b.inject_failures(3 / N, rng)
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
            cov_post_sum += coverage(pos, a, p.r_c, AREA)
            cov_post_n += 1
            ids = np.where(a)[0]
            if len(ids) > 1 and not recovered:
                sub = np.linalg.norm(pos[ids][:, None] - pos[ids][None],
                                     axis=-1)
                vis = vis_mask(pos[ids], V, AREA)
                geo1 = cg.connected_components(((sub <= p.r_c) & vis
                                                  ).astype(float),
                                                 directed=False)[0] == 1
                g = rf.gated_adjacency(pos, a, t)[np.ix_(ids, ids)]
                q = rf.mean_quality(pos, a, t)
                if geo1 and g.sum() > 0 and q > 0.5:
                    t_rec = t - T1
                    recovered = True
    return {"recovered": float(recovered),
            "REC": (t_rec / T) if recovered else 1.0,
            "thr_mbps": thr_sum / max(steps, 1),
            "mean_q": q_sum / max(steps, 1),
            "cov_end": coverage(pos, b.env.active, p.r_c, AREA),
            "cov_post_mean": cov_post_sum / max(cov_post_n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--controllers", nargs="+", default=["v2", "v8"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/confirmatory_v8_terrain.csv")
    args = ap.parse_args()
    ruins = make_ruins()
    V = los_cell_matrix(ruins, AREA, ALT)
    tasks = [(c, i, ruins, V) for c in args.controllers
             for i in range(args.trials)]
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
        for idx, (c, s, _r, _v) in enumerate(tasks):
            r = res[idx]
            w.writerow([s, "terrain", c, f"{r['recovered']:.0f}",
                        f"{r['REC']:.4f}", f"{r['thr_mbps']:.3f}",
                        f"{r['mean_q']:.4f}", f"{r['cov_end']:.4f}",
                        f"{r['cov_post_mean']:.4f}"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
