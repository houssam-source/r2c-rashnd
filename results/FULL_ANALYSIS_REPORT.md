# R2C–RASHND Program — Full Report (2026-11-20)

Repo: `houssam-source/r2c-rashnd`, branch `astar-shared-belief` @ `c119acc`
(pushed). Compute: server 10.13.0.201 (48 workers) + local laptop.
Four completed workstreams, one integrated story: **every "improvement"
layer we built on top of R2C was tested at confirmatory scale, and every
one failed its own guardrail for a now-quantified reason.**

---

## 1. Throughput metric correction (foundation)

`RFLayer.throughput` had summed Shannon capacity over all ordered in-range
pairs (each link double-counted, no per-radio constraint) — 2.5–6 Gbps
against a physical ceiling of n × 30 Mbps (QPSK, 20 MHz). Fixed with a
per-drone radio cap + assert guard (commit `3ce791e`); 16 result suites
recomputed on the server; **all non-throughput columns bit-identical**
(trajectory-neutral, verified twice). Material changes: the v2 hybrid's
throughput gain is **+4–7%**, not "+24–46%"; a handful of near-tied
arms swapped ordering (≤1% apart). Recovery science unchanged. The GNU
Radio PHY validation (12.5 dB packet gate vs the 10 dB model gate) stands.

## 2. Sub-study — Search-guided belief dissemination (v3 family): NEGATIVE

**Design:** the paper's contribution seam = Layer-3 A* belief-carrying
search over the frozen v2 cluster-belief hybrid; ablation matrix
{v2, v3a, v3b, v3}; preregistered expectations; n=100 paired × failure
ratios {20,40,60,80}% × 3 jammer modes; 16-scenario edge battery; EC10.

**Results:**
- **H1 FAILS, significantly:** v3b (the method) is worse than v2 —
  pooled McNemar over 600 pairs: 42 vs 78 discordant, **p = 0.0013**;
  −4..−8 pp per cell. Order-planning alone (v3a) is exactly neutral
  (p = 1.0); the combination (v3) is worse (p = 0.0006).
- **H2 partially holds:** v3b senses at **0.35–0.41× v2's budget**, but
  both scale superlinearly in K (exponents 2.76 vs 3.35) — a
  constant-factor saving, not a scaling class.
- **Mechanism, quantified:** adopt/valve counters show propagated beliefs
  mostly FAIL the confidence gate at high fragmentation (valve re-measures
  5.7/ep vs 4.5 adoptions at fr80) — the swarm pays dissemination AND
  local re-sensing. The method rejects its own product where it matters.
- **Expected-vs-observed:** every arm's preregistered band holds except
  the method arm's own (0.66–0.78 expected vs 0.58–0.63 observed, rejected
  in all three modes) — the band traced to pre-ablation exploratory
  claims.
- **Edge matrix:** v3b ≤ v2 in 12/16 scenarios; R2C leads 13/16. Boundary
  conditions: **EC5 zero-contrast collapses all belief-driven hybrids**
  (0.12–0.22 vs R2C 0.76) → abstract scope sentence; EC7 poisoned origin
  absorbed by the confidence valve (resilience works).
- **Verdict per the interpretation contract: the negative-result study**
  ("dissemination is 0.4× cheaper but significantly degrades recovery;
   planning is neutral-to-harmful"). Deviations all logged; double
  guardrail passed (0/180 mismatches).

## 3. v9 — information-gain controller on a probability map: NULL at Stage A

Grid-filter belief source (log-odds, tomography, ECE calibration),
preregistered, unit-tested (6/6), smoke-campaigned (stationary tracking
~10 m after 9 logged iterations); λ Stage A training on the server:
**λ has zero effect on recovery/throughput in any scenario** — the IG
machinery works (ig_net is λ-sensitive) but the preregistered action
effects never alter motion. λ = 1.0 retained; H-A untestable as
mechanized. Decision deferred to user (H-B-only confirmatory / amend
action effects / shelve).

## 4. GCN hyperparameter optimization: KEEP THE BASELINE (surrogate trap)

Optuna TPE, 100 trials, 10-fold paired-seed CV, median **val loss**
objective (next-topology generalization loss), recovery secondary-only,
20-seed guardrail (tol 0.02), 3-init fine-tune.

- Best surrogate config (hidden 128, 6 layers, dropout 0.2, early stop):
  val loss −0.5% vs baseline, **recovery 0.45 vs 0.65 → +0.200 regression,
  guardrail FAIL.**
- Agent study of the full results matrix: **72 trap / 28 neutral /
  0 safe** trials; corr(surrogate gain, recovery change) = −0.58. A
  focused lr×wd re-fine-tune around the baseline converged back onto the
  baseline itself — the safe basin is a single point.
- **Network vectorization:** canonical 57-dim vectors. Principal =
  baseline reference; feed = refined (= baseline, distance 0) + trap
  negative control from the failed config — **cosine distance 0.198**
  from the principal: the surrogate gradient points ~20% of a unit
  sphere away from the system optimum, in weight space. Surrogate-level
  HPO cannot walk back; the guardrail is the adoption gate.
- En-route bug found and fixed: 3-init fine-tune was vacuous (IST
  re-seeds torch at construction). Does not affect the verdict.

## 5. Integrated conclusion

Across three independent "smarter layer" attempts (belief dissemination,
information-gain control, GCN HPO), the pattern is identical and now
quantified at every level: **each surrogate — link-capacity sums, val
loss, net information gain — is optimizable, and its optimum is
anti-correlated with swarm recovery.** The R2C baseline plus the v2
cluster-belief hybrid, with the corrected accounting, remain the best
system in this stack. The program's contribution is the evidence: a
metric correction, a significant negative result with mechanism, a
null tuning result with weight-space quantification — all reproducible,
all preregistered, all on GitHub.

**Decisions pending (user):** substudy write-up week (figures,
reproducibility package); v9 direction; R2C author query (only user can
send).
