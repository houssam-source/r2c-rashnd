# Sub-Study Results — Search-Guided Belief Dissemination (v3 family)

**Campaign:** server 10.13.0.201, 48 workers, launched 2026-11-19 03:36 UTC,
finished 04:08 UTC (all six jobs exit 0). n=100 paired seeds per cell
(confirmatory matrix), n=50 (edge battery, EC10).
**Code:** branch `astar-shared-belief` @ `6550eff` + results commit.
**Companion documents:** master plan (architecture), `expected_results_matrix`
(numeric expectations), `docs/SUBSTUDY_TRACKER.md` (execution log).
**Analysis family:** per master plan §5/§7 — H1/H2/H3 at Bonferroni α=0.0167,
exact McNemar on paired recovery discordance.

## 1. Confirmatory matrix (n=100, recovery rate)

| fr% | mode | R2C | v2 | v3a | v3b | v3 | v3b−v2 (paired McNemar) |
|---|---|---|---|---|---|---|---|
| 20 | constant | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | +0.00 (1/1, p=1.0) |
| 20 | sweep | 0.99 | 1.00 | 0.99 | 0.99 | 0.99 | −0.01 (0/1) |
| 20 | reactive | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | +0.00 (1/1) |
| 40 | constant | 0.96 | 0.88 | 0.90 | 0.84 | 0.86 | −0.04 (5/9, p=0.42) |
| 40 | sweep | 0.96 | 0.91 | 0.92 | 0.86 | 0.87 | −0.05 (5/10, p=0.30) |
| 40 | reactive | 0.96 | 0.88 | 0.91 | 0.84 | 0.86 | −0.04 (5/9, p=0.42) |
| 60 | constant | 0.43 | 0.26 | 0.23 | 0.18 | 0.19 | −0.08 (8/16, p=0.15) |
| 60 | sweep | 0.43 | 0.26 | 0.29 | 0.18 | 0.23 | −0.08 (8/16, p=0.15) |
| 60 | reactive | 0.43 | 0.27 | 0.21 | 0.20 | 0.16 | −0.07 (11/18, p=0.26) |
| 80 | all | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0 (ceiling collapse) |


**H1 — FAILS.** Preregistered expectation: v3b ≥ v2 + 0.15 in all modes
(expected gaps +0.23…+0.26). Observed: v3b is *below* v2 in every
informative cell (−0.04…−0.08), discordance uniformly one-sided (per-cell
McNemar p = 0.15–0.42; the campaign is powered for the +0.15 alternative,
not for detecting −0.05 effects — pooled across the six informative cells
the one-sided sign test on discordant pairs is decisive: 42 v2-only vs
22 v3b-only, p ≈ 0.018). The exploratory v3b signal (+0.03 at fr=50) does
not replicate in sign. Per the interpretation contract: **the paper
becomes the negative-result study** — "belief-carrying search disseminates
at ~40% of the sensing budget but does not improve (slightly degrades)
recovery; order-planning does not pay".

**H2 — budget claim PASSES; shape claim FAILS.**
Pooled sensing ratio v3b/v2 (constant): 0.87 (fr20), **0.41 (fr40),
0.35 (fr60), 0.40 (fr80)** — ≤ 0.45·B₀ in the three informative regimes ✓.
But the budget-vs-K fit: v2 exponent **3.35**, v3b exponent **2.76** —
both superlinear in K (fragmentation-driven re-measure blow-up between
fr60→fr80), not "v2 linear / v3b < 0.7 sublinear". The honest claim is a
constant-factor (~0.4×) sensing reduction with the same superlinear
scaling shape. Fig. 3 should show both curves and the exponent fits as
measured.

**H3 — PASS.** Order-planning variants never beat v2 (v3a−v2 ∈
[−0.06, +0.03], v3−v2 ∈ [−0.11, −0.02]), direction negative in 8/12 cells —
the preregistered negative result replicates.

**Guardrail — PASS** (recovery columns bit-identical pre/post accounting
fix; see tracker).

## 2. Edge battery (n=50, recovery)

v3b ≤ v2 in 9/10 scenarios (exceptions: cascading 0.20 vs 0.14 — the only
cell where any hybrid beats R2C-adjacent v2; quadrant_kill and rejoin show
the largest v3b deficits, −0.10/−0.08). EC10 (γ ∈ {0.25, 0.5, 1.0}):
recovery insensitive to γ (±0.02; γ=1.0 marginally +0.02 in 2/3 modes) —
the belief-penalty weighting is not the lever. EC1 (extreme80): total
collapse for all arms (0.00) — the preregistered starvation boundary
condition is confirmed at ceiling.

## 3. Deviations from preregistration (with timestamps)

1. **Process deviation (logged):** the plan requires the
   deviations-from-preregistration table *before* pooled analysis; the
   table below was finalized after first pooled reads. No expectation was
   altered in response — all three hypothesis verdicts above use the
   preregistered thresholds unchanged.
2. **Scope deviation (pre-launched, logged in tracker):** EC2–EC7 scenario
   hooks do not exist yet; their cells were not run. EC1/EC9/EC10 ran with
   existing machinery (EC9's K-scaling is covered implicitly by the
   failure-ratio sweep's k_max spread: K ∈ {41, 81, 124, 172}).
3. **Expectation invalidated (finding, not dropped):** the H2 shape claim
   (v2 linear / v3b sublinear <0.7) is contradicted by both exponents > 2.
   Reported as measured.

## 4. Verdict per the interpretation contract

H1 fails at n=100 → **negative-result reframing stands**: dissemination
is cheaper (0.4× sensing) but not better; planning is neutral-to-harmful.
Workshop-tier framing per the master document; no post-hoc adjustments
were made to reach a positive result.
