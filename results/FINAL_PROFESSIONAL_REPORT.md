# Communication-Budgeted Self-Healing UAV Swarms — Final Professional Report

**Program:** R2C (Peng et al., AAAI-26) reproduction & extension
**Repository:** `github.com/houssam-source/r2c-rashnd`, branch
`astar-shared-belief`, final commit `907009c` (pushed)
**Period:** 2026-11-07 → 2026-11-21 · **Compute:** server 10.13.0.201
(48–144 workers) + local orchestration
**Methodological frame:** preregistration-first; paired seeds; physical
guardrails; every claim tied to a bit-identity or statistical check.

---

## Executive summary

This program set out to improve the R2C self-healing swarm controller
with RF-awareness, shared beliefs, information gain, learned-path
optimization, and simulation-based planning. Executed at confirmatory
scale with rigorous guardrails, the results form a coherent and — we
argue — publishable negative-and-positive arc:

1. **A metric foundation fix**: the field's throughput accounting
   overstates aggregate rate by 4–10×; corrected with a per-radio
   physical cap, verified trajectory-neutral across 16 result suites.
2. **A significant negative result**: A*-guided belief dissemination —
   the natural "smart" upgrade — *significantly degrades* recovery
   (pooled McNemar p = 0.0013) while genuinely saving 0.4× of the
   sensing budget; the mechanism (confidence-gate rejection of
   propagated beliefs under fragmentation) is measured, not speculated.
3. **A null result for information-gain control** at the tuning stage,
   with the precise structural reason (action effects never alter
   motion).
4. **A keep-the-baseline verdict for GCN hyperparameter optimization**,
   with the surrogate-optimization trap quantified in weight space
   (principal-vs-trap cosine distance 0.198; 72 trap / 0 safe trials).
5. **A null for weight-space interpolation** at guardrail scale.
6. **A positive candidate**: Monte-Carlo path-probability planning (v10)
   — the only layer to clear its exploratory guardrails — winning
   exactly in the moving-jammer regime (+0.23 recovery over the v2
   hybrid) and holding the robustness bound under model mismatch.
7. **A cross-cutting law**: self-healing value is a function of *world
   mobility, not model sophistication* — repeated identically across
   every experiment (six patterns, §6).

## 1. System and method

The stack is layered (master document, Part I): Layer 0 RF/PHY
accounting (corrected); Layer 1 R2C core (ISE spacing, IST GCN bridging,
AWF weighting — frozen throughout); Layer 2 v2 cluster-belief hybrid;
Layer 3 belief-carrying A* search (the sub-study's contribution seam);
evaluation harness with paired seeds and preregistered statistics
(family α = 0.0167, exact McNemar, paired-bootstrap CIs).

Discipline carried across all work: preregistration files locked before
runs (`results/preregistration_*.md`); amendments only pre-confirmatory
and logged; exploratory results labeled; no metric substitution; the GNU
Radio-measured 12.5 dB packet gate used wherever a delivery decision is
made; determinism verified (serial bit-identity; parallel identity on
Linux forkserver; a Windows-spawn drift on secondary columns is flagged
as a finding).

## 2. Workstream results

### 2.1 Throughput accounting correction
`RFLayer.throughput` summed Shannon capacity over all ordered in-range
pairs — every link simultaneously saturated, each physical link counted
twice, no per-radio constraint — reporting 2.5–6 Gbps for 200 drones
whose ceiling is n × 30 Mbps (QPSK, 20 MHz). Corrected to a per-drone
carried-rate cap with an assert guard (never a clamp). Recompute of 16
suites: **all non-throughput columns bit-identical** (verified twice,
0/180 + 0/180 mismatches). Scientific consequences: the v2 hybrid's
throughput gain is +4–7% (not "+24–46%"); a few ≤1%-apart arms swap
ordering; recovery science unchanged.

### 2.2 Sub-study — search-guided belief dissemination (negative, significant)
n=100 paired × failure ratios {20,40,60,80}% × 3 jammer modes ×
{R2C, v2, v3a, v3b, v3} + 16-scenario edge battery + EC10.

- **H1 fails with significance**: v3b (the method) vs v2, pooled over 600
  pairs: 42 vs 78 discordant, **p = 0.0013**; v3a (order only) exactly
  neutral (p = 1.0); v3 (combined) significantly worse (p = 0.0006).
- **H2 partial**: v3b sensing = 0.35–0.41× v2 (≤ 0.45·B₀ as
  preregistered), but budget-vs-K is superlinear for *both* arms
  (exponents 2.76 vs 3.35) — a constant-factor saving, not a scaling
  class. The preregistered shape expectation is rejected as a finding.
- **Mechanism (measured)**: adopt/valve counters show propagated beliefs
  mostly fail the confidence gate at high fragmentation (valve 5.7/ep vs
  adoptions 4.5 at fr80): the swarm pays dissemination *and* local
  re-sensing. The method rejects its own product where it is most needed.
- **Expectations**: every arm's preregistered band holds or is benignly
  exceeded except the method arm's own (0.66–0.78 expected vs 0.58–0.63
  observed; rejected in all three modes; provenance traced to
  pre-ablation exploratory claims).
- **Edge matrix**: v3b ≤ v2 in 12/16 cells; R2C leads 13/16. Boundary
  condition **EC5 zero-contrast collapses all belief-driven hybrids**
  (0.12–0.22 vs R2C 0.70) — belongs in any abstract's scope sentence.
  EC7 poisoned origin is absorbed by the confidence valve (resilience
  works as designed). EC10 (γ sweep = the cost-interpolation knob):
  ±0.02.
- **Verdict per the interpretation contract**: the negative-result study
  — "dissemination is 0.4× cheaper but significantly degrades recovery;
  order-planning is neutral-to-harmful."

### 2.3 v9 — information-gain control on a probability map (null at Stage A)
Spatio-temporal log-odds belief source (3×200×200, range tomography,
per-mode forgetting, ECE calibration), preregistered (rev 2 + logged
amendments), 6/6 unit tests, nine smoke iterations (each amendment
logged: tomography introduction, far-detection skip, per-mode interior
weights, multiplicative forgetting, divided sensing, stale ground-truth
diagnostic). Stationary-jammer MAP error ~10 m (cell-quantization
floor); ECE gate fails on construction (cell-marginal vs point-reference;
revision proposed, not applied).

Server Stage A (the training): **λ has zero effect on recovery or
throughput in any scenario — bit-identical across λ ∈ {0…100}** — while
the IG accounting is λ-sensitive exactly as designed (ig_net 0.072 → 0).
Mechanism: the preregistered action effects never alter velocity; the IG
layer prices and meters information but cannot move the swarm. λ = 1.0
retained per the locked rule. H-A untestable as mechanized; decision
deferred (H-B-only / amend action effects / shelve).

### 2.4 GCN hyperparameter optimization (keep the baseline)
Optuna TPE, 100 trials, 10-fold paired-seed CV, median **val loss**
objective (next-topology generalization), recovery secondary-only,
20-seed guardrail (tolerance 0.02).

- Best surrogate config (hidden 128, 6 layers, dropout 0.2, early
  stop): val loss −0.5%, recovery 0.45 vs 0.65 → **+0.200 regression,
  guardrail FAIL** — the spec-named failure mode, reported with both
  numbers.
- Full-matrix study: **72 trap / 28 neutral / 0 safe**; corr(surrogate
  gain, recovery) = −0.58. A constrained lr×wd re-fine-tune around the
  baseline converged back onto the baseline — the safe basin is a single
  point.
- **Weight-space vectorization**: canonical 57-dim vectors; principal
  (baseline) vs trap-feed cosine distance **0.198** — the surrogate
  gradient points ~20% of a unit sphere away from the system optimum;
  surrogate-level HPO cannot walk back; the guardrail must remain the
  adoption gate.
- Bug found en route (vacuous 3-init seeding; IST reseeds at
  construction) — fixed, documented; verdict unaffected.

### 2.5 Weight-matrix interpolation (null)
First scan (n=10) suggested a midpoint gain (0.80 vs 0.70); the
guardrail-scale replication (3 init pairs × 4 α × 20 paired seeds) shows
a **completely flat curve** (0.60 everywhere; every paired McNemar 1/1,
p = 1.0). Interpolation cannot create information the endpoints lack;
recovery is invariant along the entire blend path. The apparent gain was
one episode of small-sample noise — flagged before replication, killed
by it.

### 2.6 v10 — Monte-Carlo path-probability planner (positive candidate)
The user's architecture: replace myopic belief forces with decision-time
imagination — MCL posterior over the jammer from received powers only
(range-circle particles, power likelihood, mode-aware resampling);
M = 64 particle rollouts × 30-step swarm futures × candidate corridors;
bridge the corridor maximizing P(connected at horizon ∧ 12.5 dB gate) −
0.01·length; v2 fallback. The planning objective *is* the system
objective (no surrogate gap).

Exploratory battery (locked): **H-A′ PASS** — constant parity with v2
(honest fallback under posterior certainty), **sweep 0.70 vs v2 0.47 and
R2C 0.67** (+0.23), reactive +0.03; **H-B′ PASS at the −0.10 bound** —
noisy_gps −0.08 (weakest cell), range_loss +0.04, ec5 +0.04 (both
belief-agnostic arms collapse there as before); **H-C′ PASS by 30×** —
0.17 s/plan at n=200 (bound 5.0 s). Determinism: serial bit-identical;
recovery columns stable across paths; Windows-spawn parallel drift on
secondary columns flagged. **Next step: confirmatory n=100 (serial or
server-side).**

## 3. The cross-cutting pattern (from `results/PATTERN_ANALYSIS.md`)

Six patterns repeat across every chart; they compose into one law:

- **P1 simplicity gradient** — R2C beats/matches enhanced arms in 30/31
  cells; aggregated, machinery costs recovery.
- **P2 dynamics premium** — every sophisticated arm's edge concentrates
  in moving-jammer cells (v10 +0.133 moving vs +0.000 static; v3a
  +0.127 vs +0.040; v3b +0.100 vs −0.015) — in five independent
  datasets.
- **P3 surrogate cliff** — every optimized proxy anti-correlates with
  recovery or is flat.
- **P4 the 50–60% failure knife-edge** — arm spread 0.00 → 0.12 →
  **0.25 (maximal)** → 0.02 (extinction); all design differences are
  decided in that band.
- **P5 information costs superlinear** — sensing 16 → 78 → 553 → 1872
  events across the cliff; budget exponents 2.76–3.35.
- **P6 flat surfaces, then cliffs** — γ, λ, α, hops/flee: plateau
  responses; only structural (regime-targeting) changes move the system.

> **Meta-pattern:** *self-healing value is a function of world mobility,
> not model sophistication — the system pays for intelligence only in
> dynamics, pays against it in statics, on a surface that is flat until
> it cliffs at the 50–60% failure band.*

## 4. Engineering quality (final code state)

- **Optimization refactor (commit `907009c`, bit-identical):**
  profile-guided; the dominant waste was per-step recomputation of the
  gated adjacency / SINR / quality map across metric calls (measured
  2.9 s of an 8.9 s trial). A one-pass `RFLayer.metrics()` bundle now
  serves the four hottest runners. **8.9 → 5.6 s/trial (−37%)**, with
  bit-identity verified against HEAD on two runners, all columns.
- **Determinism hygiene:** per-trial global np/torch seeding in the
  runner (parallel worker assignment made bit-identical to serial by
  construction); serial reruns verified bit-identical across the
  program.
- **Structure:** `r2c/` (core, frozen semantics), `experiments/`
  (runners), `tests/` (v9, v10 suites green), `hpo_out/` (HPO artifacts
  + agent study), `results/` (all data + preregistrations + reports),
  `docs/` (sub-study tracker/results, work plans).

## 5. Limitations

Rate accounting is not contention modeling (scoped, one sentence, per
plan). Zero-contrast environments defeat every sensing-based layer —
environmental, not architectural. The v10 confirmatory is pending; its
verdict may not replicate at n=100. The sub-study's write-up week
(figures, reproducibility package) remains. The R2C author query is
prepared but only the user can send it. Server SSH was intermittently
unavailable (two multi-day outages); all compute fell back to local or
queued successfully on recovery.

## 6. Conclusions

The program's contribution is threefold. **Methodologically**: a working
discipline for surrogate-heavy swarm research — preregister, guardrail,
verify bit-identity, report nulls with mechanisms. **Scientifically**: a
significant negative result on budgeted belief dissemination with a
measured mechanism; a quantified surrogate trap in learned topology
repair; a cross-cutting law (mobility, not sophistication) that
organizes the field's design space. **Positively**: simulation-based
path planning (v10) as the first improvement layer to clear every
guardrail, winning precisely where the law predicts — in the moving
world.

---
*Reproducibility:* every number in this report traces to a committed CSV
in `results/` or `hpo_out/`, generated by committed runners under locked
preregistrations; the 100 confirmatory seeds were never used for
training or tuning.
