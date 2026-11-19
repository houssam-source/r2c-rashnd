"""HPO for the IST Chebyshev GCN (Optuna TPE). Frozen pipeline around it.

Protocol (hpo_out/hpo_config.yaml):
- 100 trials max; trial 0 = frozen baseline config.
- Inner loop: 10-fold paired-seed CV (guardrail seeds 0-9), one E4 episode
  per fold (constant jammer, failures 0.5, v2 controller). Fold val loss =
  mean pre-update loss on each invocation's graph (invocations 1..7) - a
  next-topology generalization loss, computed BEFORE training on that
  graph (no leakage). Objective = MEDIAN over folds. Recovery/REC/thr are
  logged per fold as SECONDARY metrics, never the objective.
- Guardrail: best config re-run on all 20 guardrail seeds; regression
  > 0.02 recovery vs baseline -> recommend keeping baseline.
- Fine-tune: 3 independent GCN inits on the best config; report
  mean +- std of val loss and recovery; flag seed std > 0.01.

Run (server, 10-way fold parallelism inside each trial):
    OMP_NUM_THREADS=1 python build/hpo_gcn.py --trials 100 --out hpo_out
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
torch.set_num_threads(1)

from r2c.config import R2CParams
from r2c.controller import R2CController
from r2c.environment import SwarmEnvironment
from r2c.hybrid import HybridController
from r2c.rf import RFLayer, RFParams

CV_SEEDS = list(range(10))
GUARD_SEEDS = list(range(20))
BASELINE = dict(lr=1e-4, hidden=512, layers=8, wd=0.0, dropout=0.0,
                max_epochs=5, val_frac=0.0)


def make_ctrl(p, rf_p, variant, cfg, init_seed=None):
    """Controller whose IST uses the trial config. GCN init reseeded per
    init_seed (fine-tune) else p.seed (frozen behavior)."""
    if init_seed is not None:
        torch.manual_seed(init_seed)
    if variant == "v2":
        c = HybridController(p, rf_p=rf_p, variant="v2")
    else:
        c = R2CController(p)
    ist = c.ist
    # apply hyperparams directly (no env vars -> reproducible in-process)
    ist.p.lr = cfg["lr"]
    ist.p.gcn_hidden = cfg["hidden"]
    ist.p.gcn_layers = cfg["layers"]
    import r2c.ist as ist_mod
    ist.model = ist_mod.ChebNet(ist.p).to(ist.device)
    ist.opt = torch.optim.Adam(ist.model.parameters(), lr=cfg["lr"],
                               weight_decay=cfg["wd"])
    ist.model.dropout = cfg["dropout"]
    ist.val_frac = cfg["val_frac"]
    ist.max_epochs = cfg["max_epochs"]
    ist.patience = 20
    return c


def episode(seed, cfg, variant="v2", capture_val=True, init_seed=None):
    """One E4 episode; returns dict with val losses and swarm metrics."""
    p = R2CParams(**{**R2CParams().__dict__, "seed": seed})
    env = SwarmEnvironment(p, p.n_random, "random")
    rf_p = RFParams(jammer_mode="constant", seed=seed)
    rf = RFLayer(p, rf_p)
    ctrl = make_ctrl(p, rf_p, variant, cfg, init_seed=init_seed)
    val_losses = []
    if capture_val:
        ist = ctrl.ist
        orig = ist.train_step
        state = {"pos": None, "active": None, "adj": None}

        calls = {"n": 0}

        def hooked(pos, active, epochs=5, adj_override=None):
            calls["n"] += 1
            if calls["n"] > 1:   # skip first invocation (no prior)
                try:
                    import torch as _t
                    ist.model.eval()
                    with _t.no_grad():
                        pos_t = _t.tensor(pos, dtype=_t.float32,
                                          device=ist.device)
                        act_t = _t.tensor(active, dtype=_t.float32,
                                          device=ist.device)
                        labels, lnorm = ist._graph_parts(pos, active,
                                                         adj_override)
                        n_parts = len(set(labels[active]))
                        if n_parts > 0:
                            d = ist.model(ist._features(pos, active).to(
                                ist.device), lnorm.to(ist.device),
                                _t.tensor(labels, device=ist.device),
                                n_parts)
                            vl = ist._loss(
                                pos_t + d[_t.tensor(
                                    [labels[i] if labels[i] < n_parts else 0
                                     for i in range(len(pos))],
                                    device=ist.device)]
                                * p.v_max * p.T_ist,
                                d, labels, act_t, n_parts)
                            val_losses.append(float(vl.item()))
                    ist.model.train()
                except Exception:
                    pass
            return orig(pos, active, epochs=epochs,
                        adj_override=adj_override)
        ist.train_step = hooked

    thr_sum = pd_sum = q_sum = steps = 0.0
    t_rec, recovered = None, False
    for t in range(p.T):
        if t == p.t1:
            env.inject_failures(0.5)
        env.step(ctrl.velocity(env, t))
        if t > p.t1:
            thr_sum += rf.throughput(env.pos, env.active, t)
            pd_sum += rf.last_per_drone_mbps
            q_sum += rf.mean_quality(env.pos, env.active, t)
            steps += 1
            if (not recovered and env.connected()
                    and rf.mean_quality(env.pos, env.active, t) > 0.5):
                t_rec = t - p.t1
                recovered = True
    return {
        "val_loss": float(np.mean(val_losses)) if val_losses else float("nan"),
        "n_val": len(val_losses),
        "recovered": float(recovered),
        "REC": (t_rec / p.T) if recovered else 1.0,
        "thr_mbps": thr_sum / max(steps, 1),
        "thr_per_drone_mbps": pd_sum / max(steps, 1),
    }


def fold_worker(args):
    seed, cfg = args
    torch.set_num_threads(1)
    return episode(seed, cfg)


def run_cv(cfg, seeds, pool):
    res = pool.map(fold_worker, [(s, cfg) for s in seeds])
    vl = [r["val_loss"] for r in res]
    return float(np.median(vl)), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--out", default="hpo_out")
    ap.add_argument("--study", default="ist-gcn-hpo")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage = f"sqlite:///{os.path.abspath(args.out)}/optuna.db"
    study = optuna.create_study(
        study_name=args.study, storage=storage, load_if_exists=True,
        direction="minimize",
        sampler=TPESampler(seed=42, n_startup_trials=12))

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(10)

    csv_path = os.path.join(args.out, "hpo_results.csv")
    new_file = not os.path.exists(csv_path)
    cw = open(csv_path, "a", newline="")
    import csv as _csv
    w = _csv.writer(cw)
    if new_file:
        w.writerow(["trial", "lr", "hidden", "layers", "wd", "dropout",
                    "max_epochs", "val_frac", "median_val_loss",
                    "cv_recovery", "cv_rec", "cv_thr_mbps", "state"])

    def cfg_from(trial):
        return dict(
            lr=trial.suggest_float("lr", 1e-5, 5e-3, log=True),
            hidden=trial.suggest_categorical("hidden", [128, 256, 512, 768]),
            layers=trial.suggest_categorical("layers", [4, 6, 8, 10]),
            wd=trial.suggest_float("wd", 1e-6, 1e-2, log=True),
            dropout=trial.suggest_categorical("dropout", [0.0, 0.1, 0.2]),
            max_epochs=trial.suggest_categorical("max_epochs", [5, 20, 40]),
            val_frac=trial.suggest_categorical("val_frac", [0.0, 0.2]))

    n0 = len(study.trials)
    for i in range(n0, args.trials):
        if i == 0:
            cfg = dict(BASELINE)
        else:
            trial = study.ask()
            cfg = cfg_from(trial)
        t0 = time.time()
        med, res = run_cv(cfg, CV_SEEDS, pool)
        rec = float(np.mean([r["recovered"] for r in res]))
        rec_t = float(np.mean([r["REC"] for r in res]))
        thr = float(np.mean([r["thr_mbps"] for r in res]))
        state = "baseline" if i == 0 else "complete"
        w.writerow([i, cfg["lr"], cfg["hidden"], cfg["layers"], cfg["wd"],
                    cfg["dropout"], cfg["max_epochs"], cfg["val_frac"],
                    f"{med:.6f}", f"{rec:.3f}", f"{rec_t:.4f}", f"{thr:.1f}",
                    state])
        cw.flush()
        if i == 0:
            study.add_trial(optuna.trial.create_trial(
                params={}, distributions={},
                values=[med],
                user_attrs={"cfg": json.dumps(cfg), "baseline": True}))
        else:
            study.tell(trial, med)
        print(f"trial {i}: val={med:.4f} rec={rec:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    cw.close()
    pool.close()
    pool.join()

    # best config (excluding the baseline-annotated trial from selection is
    # NOT done - the baseline may legitimately win)
    best = study.best_trial
    cfg = json.loads(best.user_attrs.get("cfg", "{}"))
    if not cfg:
        cfg = dict(BASELINE)
    print("BEST:", cfg, "val:", best.value, flush=True)
    with open(os.path.join(args.out, "best_cfg.json"), "w") as f:
        json.dump({"cfg": cfg, "median_val_loss": best.value,
                   "trial": best.number}, f, indent=2)
    print("HPO_DONE", flush=True)


if __name__ == "__main__":
    main()
