"""Stage A: field-controller configuration sweep (E4 constant, n=30)."""
import argparse, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_e4_variants import trial as e4_trial
from r2c.config import R2CParams
from r2c.rf import RFParams

CONFIGS = [
    "field",
    "field(beta=0.20)", "field(beta=0.55)", "field(eta=0.30)",
    "field(hops=3)", "field(w_flee=0.25)",
    "field(w_cover=0.85,cover_margin=0.30)", "field(eta=0.30,hops=3)",
]


def one(task):
    cfg, seed = task
    p = R2CParams(seed=seed)
    rf_p = RFParams(jammer_mode="constant", seed=seed)
    return e4_trial(p, rf_p, 0.5, "field", cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/field_sweep.csv")
    args = ap.parse_args()
    tasks = [(c, i) for c in CONFIGS for i in range(args.trials)]
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
        w.writerow(["seed", "config", "recovered", "REC", "thr_mbps", "mean_q"])
        for idx, (c, s) in enumerate(tasks):
            r = res[idx]
            w.writerow([s, c, f"{r['recovered']:.0f}", f"{r['REC']:.4f}",
                        f"{r['thr_mbps']:.3f}", f"{r['mean_q']:.4f}"])
    import numpy as np
    print("config, recovery, thr, q")
    for c in CONFIGS:
        rows = [res[i] for i, (cc, _s) in enumerate(tasks) if cc == c]
        rec = np.mean([r["recovered"] for r in rows])
        thr = np.mean([r["thr_mbps"] for r in rows])
        q = np.mean([r["mean_q"] for r in rows])
        print(f"{c:45s} {rec:.2f} {thr:6.0f} {q:.3f}", flush=True)


if __name__ == "__main__":
    main()
