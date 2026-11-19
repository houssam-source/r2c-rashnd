"""Focused local sweep around the HPO baseline (deliverable 3).

Grid: lr x wd, hidden=512, layers=8, dropout=0.0, max_epochs=5, val_frac=0.0.
Selection: among configs whose mean recovery on seeds 0-4 is within 0.02 of
the baseline mean on the SAME seeds, pick the lowest mean val loss.
Held-out: chosen config vs BASELINE on seeds 5-9 (report only, no re-select).
"""
import itertools
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "build"))

import numpy as np
import torch

from hpo_gcn import episode, BASELINE

OUT = os.path.join(REPO, "hpo_out", "agent_study")
TOL = 0.02
SEEDS_FIT = list(range(5))
SEEDS_HOLDOUT = list(range(5, 10))


def _worker(args):
    seed, cfg = args
    torch.set_num_threads(1)
    return episode(seed, cfg)


def run(cfg, seeds, pool):
    res = pool.map(_worker, [(s, dict(cfg)) for s in seeds])
    return {
        "val_loss_mean": float(np.mean([r["val_loss"] for r in res])),
        "val_loss_median": float(np.median([r["val_loss"] for r in res])),
        "recovery_mean": float(np.mean([r["recovered"] for r in res])),
        "rec_mean": float(np.mean([r["REC"] for r in res])),
        "thr_mbps_mean": float(np.mean([r["thr_mbps"] for r in res])),
        "per_seed": {str(s): res[i] for i, s in enumerate(seeds)},
    }


def main():
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(5)

    grid = [dict(lr=lr, hidden=512, layers=8, wd=wd, dropout=0.0,
                 max_epochs=5, val_frac=0.0)
            for lr, wd in itertools.product(
                [5e-5, 1e-4, 2e-4, 5e-4], [0.0, 1e-5, 1e-4])]

    print("baseline on seeds 0-4 ...", flush=True)
    base_fit = run(BASELINE, SEEDS_FIT, pool)
    print(f"baseline fit: rec={base_fit['recovery_mean']:.3f} "
          f"vl={base_fit['val_loss_mean']:.4f}", flush=True)

    rows = []
    for cfg in grid:
        r = run(cfg, SEEDS_FIT, pool)
        row = {**cfg, **{f"fit_{k}": v for k, v in r.items()
                        if k != "per_seed"}, "per_seed_fit": r["per_seed"]}
        rows.append(row)
        print(f"lr={cfg['lr']:g} wd={cfg['wd']:g}: "
              f"rec={r['recovery_mean']:.3f} vl={r['val_loss_mean']:.4f}",
              flush=True)

    ok = [r for r in rows
          if abs(r["fit_recovery_mean"] - base_fit["recovery_mean"]) <= TOL]
    if not ok:
        chosen, note = min(rows, key=lambda r: abs(
            r["fit_recovery_mean"] - base_fit["recovery_mean"])), \
            "no config within tolerance; closest recovery chosen"
    else:
        chosen = min(ok, key=lambda r: r["fit_val_loss_mean"])
        note = "selected by rule: within 0.02 recovery of baseline on " \
               "seeds 0-4, lowest mean val loss"
    print("CHOSEN:", {k: chosen[k] for k in
                      ("lr", "wd", "hidden", "layers")}, note, flush=True)

    cfg_chosen = {k: chosen[k] for k in
                  ("lr", "hidden", "layers", "wd", "dropout",
                   "max_epochs", "val_frac")}
    print("held-out seeds 5-9 ...", flush=True)
    ch_hold = run(cfg_chosen, SEEDS_HOLDOUT, pool)
    base_hold = run(BASELINE, SEEDS_HOLDOUT, pool)

    out = {
        "selection_rule": "among configs with mean recovery on seeds 0-4 "
                          "within 0.02 of baseline mean on the same seeds, "
                          "pick lowest mean val loss",
        "note": note,
        "baseline_fit_seeds_0_4": {k: v for k, v in base_fit.items()},
        "grid": rows,
        "chosen_cfg": cfg_chosen,
        "chosen_fit_seeds_0_4": {k: v for k, v in chosen.items()
                                 if k != "per_seed_fit"},
        "heldout_seeds_5_9": {
            "chosen": {k: v for k, v in ch_hold.items()},
            "baseline": {k: v for k, v in base_hold.items()},
        },
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "refined_cfg.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE", os.path.join(OUT, "refined_cfg.json"), flush=True)
    print(f"HELDOUT chosen: rec={ch_hold['recovery_mean']:.3f} "
          f"vl={ch_hold['val_loss_mean']:.4f} | baseline: "
          f"rec={base_hold['recovery_mean']:.3f} "
          f"vl={base_hold['val_loss_mean']:.4f}", flush=True)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
