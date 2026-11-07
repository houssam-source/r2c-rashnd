"""Confirmatory C3: terrain-stress scenarios at n=100, per-trial paired rows."""
import argparse, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "gazebo", "uav_swarm_r2c"))
from experiments.run_terrain_edge import trial
from experiments.run_terrain_study import make_ruins
from r2c.visibility import los_cell_matrix
from experiments.run_terrain_study import AREA, ALT


def one(task):
    scenario, c, seed, ruins, V = task
    return trial(scenario, c, seed, ruins, V)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--scenarios", nargs="+",
                    default=["quadrant_kill", "rejoin", "fast_sweep"])
    ap.add_argument("--controllers", nargs="+", default=["blind", "v4"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/confirmatory_terrain.csv")
    args = ap.parse_args()
    ruins = make_ruins()
    V = los_cell_matrix(ruins, AREA, ALT)
    tasks = [(s, c, i, ruins, V) for s in args.scenarios
             for c in args.controllers for i in range(args.trials)]
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
                    "thr_mbps", "mean_q"])
        for idx, (s, c, sd, _r, _v) in enumerate(tasks):
            r = res[idx]
            w.writerow([sd, s, c, f"{r['recovered']:.0f}", f"{r['REC']:.4f}",
                        f"{r['thr_mbps']:.3f}", f"{r['mean_q']:.4f}"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
