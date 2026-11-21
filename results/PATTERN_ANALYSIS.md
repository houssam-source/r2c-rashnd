# Cross-Cutting Pattern Analysis — What Repeats Across Every Chart

**Date:** 2026-11-21 · **Data:** all locked result suites (throughput fix,
v3 substudy n=100, edge battery n=50 ×16 scenarios, failure-ratio matrix,
v9 Stage A, GCN HPO 100 trials, interpolation study, v10 exploratory).
Method: recovery deltas of every arm vs every arm, per regime class.

---

## P1 — The simplicity gradient

R2C (pure geometry, zero RF machinery) beats or matches every enhanced
arm in **30 of 31 measured cells** (edge battery: any hybrid beats R2C in
1/16 — cascading, where everyone scores ≈0.1; failure matrix: 1/12;
robustness battery: 0/3). Mean hybrid-vs-R2C is negative everywhere
except dynamics pockets. **Added machinery, aggregated over all
experiments, costs recovery** — the program's most robust statistical
regularity.

## P2 — The dynamics premium (the strongest repeating pattern)

Every sophisticated arm's advantage over v2 concentrates in *moving-
world* cells, in five independent datasets:

| arm | moving-jammer cells | static cells |
|---|---|---|
| v10 (E4, vs v2) | **+0.133** | +0.000 (parity) |
| v3a (edge, vs v2) | **+0.127** | +0.040 |
| v3b (edge, vs v2) | **+0.100** | −0.015 |
| v3 (edge, vs v2) | +0.047 | +0.015 |
| v3b (fr matrix, vs v2) | −0.031 | −0.030 (pays in neither — the failed design) |

Same shape in HPO-land: the only val-loss-sensitive behavior that helped
recovery was horizon/cadence-like (early stopping), i.e. time-awareness.
**Prediction this makes:** v10's confirmatory will be decided by the
sweep/reactive cells; its constant-mode parity is the null baseline, not
a weakness.

## P3 — The surrogate cliff

Every proxy we optimized either anti-correlated with recovery or was
flat: link-capacity sums (fixed, +24–46% → +4–7%), GCN val loss
(corr = **−0.58**, 72 trap / 0 safe), information gain (λ bit-identical
across 5 orders of magnitude). Optimization-shaped methods (TPE, grid,
interpolation) all walk off the same cliff from different directions —
because the cliff is in the proxy, not the optimizer.

## P4 — The 50–60% failure knife-edge

Arm spread by failure ratio (constant mode): fr20 = 0.00 (ceiling),
fr40 = 0.12, **fr60 = 0.25 (maximal separation)**, fr80 = 0.02 (mass
extinction, all arms ≈ 0). Every design difference in this program is
decided inside a narrow failure band; below it there is nothing to
improve, above it nothing can be saved. All preregistered expectations
implicitly assumed the knife-edge band and failed outside it.

## P5 — Information costs superlinear

v2 sensing events per episode across the cliff: 16 → 78 → 553 → **1872**
(≈×7 per +20% failures at the top end); budget-vs-K exponents 2.76–3.35
(both arms superlinear). Every information-centric design assumption —
linear sensing budget, sublinear scaling, "share to save" — breaks
exactly at the knife-edge. Belief-carried designs collapse there because
the confidence gate rejects propagated beliefs (valve 5.7/ep vs
adoptions 4.5 at fr80 — measured).

## P6 — Flat surfaces, then cliffs

Measured response-to-knob across the program: γ ∈ {0.25…1.0} → ±0.02;
λ ∈ {0…100} → bit-identical; interpolation α ∈ [0,1] → flat 0.60;
hops/flee knobs → bit-identical no-ops; v10 in constant mode → parity.
The system's response surface is **plateau-with-cliffs**, not hills —
which is why all six gradient/optimizer-shaped attempts found nothing.
The only effective "knobs" have been structural (which regime an arm
targets), not parametric.

---

## The one-sentence meta-pattern

> **Self-healing value is a function of world mobility, not model
> sophistication: the system pays for intelligence only in dynamics,
> pays against it in statics, on a surface that is flat until it
> cliffs at the 50–60% failure band.**

Every chart in the program is a different projection of this sentence:
P1 (aggregate cost), P2 (where value concentrates), P3 (why proxy
optimization fails), P4 (where differences are decided), P5 (why
information-sharing breaks), P6 (why knobs don't move anything).

**Design rule implied for anything next:** build for the moving world
(v10's regime), evaluate at fr50–60, distrust any parametric knob that
claims improvement without a regime-shift story, and keep the guardrail
— every proxy without one has produced a trap.
