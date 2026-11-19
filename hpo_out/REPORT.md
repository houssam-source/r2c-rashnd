# HPO Report - IST Chebyshev GCN

**Study:** ist-gcn-hpo - Optuna TPE (seed 42), 100 trials, 10-fold paired-seed
CV (guardrail seeds 0-9), objective = median fold val loss (pre-update loss on
invocation graphs 1..7; next-topology generalization). Recovery/REC/throughput
logged SECONDARY only - never the selection criterion. Patience 20 fixed.
Determinism: torch seeded per IST init, OMP_NUM_THREADS=1.

## Best config vs baseline

| knob | baseline | best (trial 75) |
|---|---|---|
| lr | 1e-4 | 2.61e-3 |
| hidden | 512 | 128 |
| cheb layers | 8 | 6 |
| weight decay | 0.0 | 9.53e-5 |
| dropout | 0.0 | 0.2 |
| max_epochs / val_frac | 5 / 0.0 | 40 / 0.2 |

| metric | baseline | best | delta |
|---|---|---|---|
| CV median val loss (10 folds) | 86.2219 | 85.7898 | -0.432 |
| Guardrail recovery (20 paired seeds) | **0.650** | **0.450** | **-0.200** |

## Guardrail verdict: FAIL (regression 0.200 > tolerance 0.02)

## Seed variance (3 inits x 20 seeds)
| init seed | recovery | val loss |
|---|---|---|
| 1000 | 0.450 | 86.273 |
| 1001 | 0.450 | 86.273 |
| 1002 | 0.450 | 86.273 |

std = 0.0000 - **VACUOUS, not a clean bill**: IST.__init__ re-seeds torch
with p.seed after the fine-tune seed was applied, so all three inits were
the same run. Bug fixed in build/hpo_gcn.py (seed after construction);
guardrail verdict unaffected.

## Recommendation

**KEEP THE BASELINE.** The best HPO config improves the surrogate val loss
by 0.43 (0.5%) and degrades downstream recovery by 0.20 (31% relative) on the
20-seed guardrail. This is the spec-named failure mode - optimizing the
surrogate at the expense of the system - and it is reported here as such,
with both numbers on the record. The val-loss surface's optimum
(small-fast-regularized networks: hidden 128, 6 layers, heavy dropout +
early stopping, all top-5 configs share this shape) rewards networks that
fit the spectral connectivity term of one graph while translating the swarm
worse across the episode. The hand-tuned baseline (lr 1e-4, 512x8) sits at
a system-optimal point for this surrogate family. The TPE did its job
faithfully; the objective is the problem, not the optimizer. Any future HPO
on this stack should change the surrogate (e.g., episode-return-weighted
loss) before spending more trials.

## Re-run the 100-seed confirmatory with the new config (NOT recommended)

```bash
R2C_IST_LR=2.61e-3 R2C_IST_HIDDEN=128 R2C_IST_LAYERS=6 R2C_IST_WD=9.53e-5   R2C_IST_VAL_FRAC=0.2 R2C_IST_MAX_EPOCHS=40   python experiments/run_confirmatory_flat.py --trials 100 --controllers R2C v2 v3b
```

(Recorded for completeness per deliverable spec; adoption is rejected by
the guardrail.)
