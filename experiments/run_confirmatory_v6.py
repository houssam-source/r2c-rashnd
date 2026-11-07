"""Confirmatory S1: v6 vs v2 on power_surge, per-trial paired rows."""
import argparse, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_edge_cases import trial as edge_trial, SCENARIOS


def one(task):
    c, seed = task
    return edge_trial("power_surge", c, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--controllers", nargs="+", default=["v2", "v6"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/confirmatory_v6.csv")
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
                    "thr_mbps", "mean_q"])
        for idx, (c, s) in enumerate(tasks):
            r = res[idx]
            w.writerow([s, "power_surge", c, f"{r['recovered']:.0f}",
                        f"{r['REC']:.4f}", f"{r['thr_mbps']:.3f}",
                        f"{r['mean_q']:.4f}"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
