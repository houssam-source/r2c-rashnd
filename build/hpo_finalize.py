"""Stage 2: guardrail, 3-init fine-tune, artifacts, REPORT.md for the GCN HPO.

Run AFTER build/hpo_gcn.py completes (--out hpo_out with best_cfg.json).
Writes hpo_out/best_model.pt, best_manifest.json, REPORT.md, and
seed_variance.csv.
"""
import json
import os
import sys

import numpy as np
import torch

torch.set_num_threads(1)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "build"))

from hpo_gcn import episode, BASELINE, GUARD_SEEDS  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "hpo_out"
INIT_SEEDS = [1000, 1001, 1002]
TOL = 0.02
STD_FLAG = 0.01


def main():
    best = json.load(open(os.path.join(OUT, "best_cfg.json")))
    cfg = best["cfg"]
    print("best cfg:", cfg, "cv val:", best["median_val_loss"], flush=True)

    # ---- guardrail: baseline and best on all 20 paired seeds ----------
    rows = []
    for tag, c in (("baseline", dict(BASELINE)), ("best", dict(cfg))):
        for s in GUARD_SEEDS:
            r = episode(s, c)
            rows.append({"arm": tag, "seed": s, **r})
            print(f"{tag} seed {s}: rec={r['recovered']:.0f} "
                  f"val={r['val_loss']:.3f}", flush=True)
    base_rec = np.mean([r["recovered"] for r in rows if r["arm"] == "baseline"])
    best_rec = np.mean([r["recovered"] for r in rows if r["arm"] == "best"])
    regress = base_rec - best_rec
    print(f"guardrail: baseline rec={base_rec:.3f} best rec={best_rec:.3f} "
          f"regression={regress:+.3f} (tol {TOL})", flush=True)

    # ---- 3-init fine-tune of the best config ---------------------------
    ft = []
    for i, iseed in enumerate(INIT_SEEDS):
        recs, vls = [], []
        for s in GUARD_SEEDS:
            r = episode(s, dict(cfg), init_seed=iseed)
            recs.append(r["recovered"])
            vls.append(r["val_loss"])
        ft.append({"init": iseed,
                   "recovery": float(np.mean(recs)),
                   "val_loss": float(np.mean(vls))})
        print(f"init {iseed}: rec={np.mean(recs):.3f} +- "
              f"{np.std(recs):.3f} val={np.mean(vls):.3f}", flush=True)
    rec_std = float(np.std([f["recovery"] for f in ft]))
    seed_flag = rec_std > STD_FLAG
    print(f"init-seed recovery std across inits: {rec_std:.4f} "
          f"(flag > {STD_FLAG}: {seed_flag})", flush=True)

    with open(os.path.join(OUT, "seed_variance.csv"), "w") as f:
        f.write("init_seed,recovery_mean_20seeds,val_loss_mean_20seeds\n")
        for r in ft:
            f.write(f"{r['init']},{r['recovery']:.4f},{r['val_loss']:.4f}\n")

    # ---- artifacts ------------------------------------------------------
    # best_model.pt: a trained GCN snapshot under the best config
    from r2c.config import R2CParams
    from r2c.ist import IST
    p = R2CParams(seed=0)
    ist = IST(p)
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
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 1000, (200, 2))
    act = np.ones(200, bool)
    for _ in range(3):
        ist.train_step(pos, act, epochs=5)
    torch.save(ist.model.state_dict(), os.path.join(OUT, "best_model.pt"))

    manifest = {
        "cfg": cfg,
        "cv_median_val_loss": best["median_val_loss"],
        "trial": best["trial"],
        "guardrail": {
            "seeds": GUARD_SEEDS, "baseline_recovery": base_rec,
            "best_recovery": best_rec, "regression": regress,
            "tolerance": TOL, "pass": regress <= TOL},
        "fine_tune": ft,
        "init_recovery_std": rec_std,
        "seed_std_flag": seed_flag,
        "versions": {"torch": torch.__version__,
                     "numpy": np.__version__,
                     "python": sys.version.split()[0]},
        "git_commit": subprocess_git(),
        "baseline_cfg": dict(BASELINE),
        "notes": ("val loss = mean pre-update loss on invocation graphs "
                  "1..7 (next-topology generalization); recovery logged "
                  "secondary, never the objective. best_model.pt is a "
                  "3-invocation warmup snapshot for reproduction; the "
                  "pipeline retrains online per episode."),
    }
    with open(os.path.join(OUT, "best_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ---- REPORT.md -------------------------------------------------------
    with open(os.path.join(OUT, "REPORT.md"), "w") as f:
        w = f.write
        w("# HPO Report — IST Chebyshev GCN\n\n")
        w("**Study:** ist-gcn-hpo · TPE (seed 42), 100 trials max, 10-fold "
          "paired-seed CV (seeds 0–9), median fold val loss objective. "
          "Recovery/REC/throughput logged as secondary metrics only. "
          "Patience 20 fixed.\n\n")
        w("## Best config vs baseline\n\n")
        w("| knob | baseline | best |\n|---|---|---|\n")
        for k in ("lr", "hidden", "layers", "wd", "dropout",
                  "max_epochs", "val_frac"):
            w(f"| {k} | {BASELINE[k]} | {cfg[k]} |\n")
        w(f"\nCV median val loss: baseline vs best = see "
          f"hpo_results.csv (trial {best['trial']}).\n\n")
        w("## Guardrail (20 paired seeds, failures 0.5, constant jammer)\n\n")
        w(f"- baseline recovery = {base_rec:.3f}\n")
        w(f"- best-config recovery = {best_rec:.3f}\n")
        w(f"- regression = {regress:+.3f} (tolerance {TOL}) → "
          f"{'PASS' if regress <= TOL else 'FAIL'}\n\n")
        w("## Seed variance (3 inits × 20 seeds)\n\n")
        w("| init seed | recovery | val loss |\n|---|---|---|\n")
        for r in ft:
            w(f"| {r['init']} | {r['recovery']:.3f} | "
              f"{r['val_loss']:.3f} |\n")
        w(f"\ninit-seed recovery std = {rec_std:.4f} → "
          f"{'FLAGGED (> ' + str(STD_FLAG) + ')' if seed_flag else 'ok'}\n\n")
        w("## Recommendation\n\n")
        w(recommendation(regress, seed_flag, base_rec, best_rec, cfg))
        w("\n\n## Re-run the 100-seed confirmatory with the new config\n\n")
        w("```bash\n")
        w(f"R2C_IST_LR={cfg['lr']} R2C_IST_HIDDEN={cfg['hidden']} "
          f"R2C_IST_LAYERS={cfg['layers']} R2C_IST_WD={cfg['wd']} "
          f"R2C_IST_VAL_FRAC={cfg['val_frac']} "
          f"R2C_IST_MAX_EPOCHS={cfg['max_epochs']} "
          f"R2C_IST_INIT_WEIGHTS={os.path.abspath(OUT)}/best_model.pt \\\n")
        w("  python experiments/run_confirmatory_flat.py --trials 100 "
          "--controllers R2C v2 v3b\n")
        w("```\n")
        w("(dropout is baked into the saved state_dict head path only when "
          "> 0; re-run uses the config above)\n")
    print("FINALIZE_DONE", flush=True)


def subprocess_git():
    import subprocess
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=REPO).stdout.strip()
    except Exception:
        return "unknown"


def recommendation(regress, seed_flag, base_rec, best_rec, cfg):
    if regress > TOL:
        return (f"**Keep the baseline.** The best HPO config degrades "
                f"guardrail recovery by {regress:+.3f} (> {TOL}). "
                f"Optimizing the surrogate val loss at the expense of the "
                f"downstream system is a known failure mode and has "
                f"occurred here: the config that minimizes pre-update "
                f"topology loss ({cfg}) moves the swarm worse. Report both "
                f"numbers, adopt nothing.")
    if seed_flag:
        return (f"**Need more seeds / keep baseline provisionally.** "
                f"Recovery is within tolerance ({regress:+.3f}) but the "
                f"3-init std exceeds {STD_FLAG} — init-seed variance is a "
                f"real finding, not noise to average away. Investigate "
                f"(larger study or init stabilization) before adoption.")
    return (f"**Adopt with the 100-seed confirmatory as the arbiter.** "
            f"Recovery within tolerance ({regress:+.3f}) and init-seed "
            f"std acceptable; the confirmatory campaign decides.")


if __name__ == "__main__":
    main()
