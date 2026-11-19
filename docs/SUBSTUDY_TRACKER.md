# Sub-Study Execution Tracker — Search-Guided Belief Dissemination

**Master document:** `substudy-architecture-and-simulation-plan.md` (2026-11-19)
**Branch:** `astar-shared-belief` · **Server battery launched:** 2026-11-19 03:36 UTC
**Code state at launch:** `6550eff` (instrumentation pre-launch per plan §10.2)

## Step 1 — Recompute pass / guardrail: DONE before launch

- Corrected-accounting recompute of the v3-family suites ran 2026-11-11
  (commits `3ce791e` fix + server battery; see `results/archive_pre_thrfix/`).
- **Guardrail result: PASS.** Recovery/REC/quality/mean_q columns of
  `confirmatory_flat.csv` (R2C/v2/v3b × 3 modes, n=100) and `e4_variants.csv`
  (R2C/v2/v3a/v3b/v3 × 3 modes, n=50) are **bit-identical** between the
  pre-fix archive and the corrected recompute on all paired seeds
  (`build/verify_thr_fix.py`, 0 mismatches). The corrected accounting is
  confirmed trajectory-neutral for the v3 family; recovery columns of the
  pre-fix exploratory runs carry forward per the plan.
- New throughput columns (thr_mbps corrected + thr_per_drone_mbps) collected
  in the same pass.

## Step 2 — n=100 paired confirmatory (failure ratios 20/40/60/80%)

- Launched as four suites (`results/confirm_v3_fr{02,04,06,08}.csv`) with
  per-trial side dumps (`*_trials.csv`: seed/jammer/controller/failures +
  REC/recovered/thr/thr_per_drone/mean_q/sensing_events/replans/k_max).
- `k_max` = max post-failure cluster count per episode (budget-vs-K curve,
  H2 / Fig. 3 data). Instrumentation added BEFORE launch.
- Sensing budget pooled from `sensing_events` (v2 = divided-sensing counter,
  v3* = frontier-only counter, both already in the runner outputs).

## Step 3 — Edge battery (supported cells)

- `results/edge_v3family.csv`: existing scenario set (cascading,
  quadrant_kill, power_surge, jammer_move, fast_sweep, noisy_gps, extreme80,
  rejoin, edge_mission, range_loss) × {R2C, v2, v3a, v3b, v3}, n=50.
  Mapping to EC numbers: EC1=extreme80; EC2 fast-reactive / EC3 dual-jammer /
  EC4 cluster-head death / EC5 zero-contrast / EC6 corridor-collapse hooks
  are NOT yet implemented as scenario configs — flagged as branch work
  before their cells can run.
- `results/ec10_gamma.csv`: v3b gamma ∈ {0.25, 0.5, 1.0}, n=50 × 3 modes.

## Pending (branch work, before analysis week)

- EC2–EC7 scenario hooks in `run_edge_cases.py` (spec'd in the master doc).
- Adopt/valve rate instrumentation for EC2 data (belief-store counters
  per trial).
- Deviations-from-preregistration table drafted BEFORE pooled analysis
  (plan §10.4), into this file.
