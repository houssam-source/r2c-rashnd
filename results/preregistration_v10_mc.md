# Pre-Registration — v10: Monte-Carlo Path-Probability Planner

**Status:** locked 2026-11-21 before any v10 run. **Variant slot:** v10.
**Tuning knobs:** exactly ONE — `M` (rollout count), default 64.

## Design (what replaces what)

Self-healing decisions stop depending on myopic belief forces and become
**decision-time Monte-Carlo planning**:

1. **Posterior (from physical information — anti-cheat preserved).**
   Monte-Carlo localization over the jammer position: particles are
   range-circle intersections implied by received powers (free-space
   inversion, same measurement model as v9's tomography); weights =
   Gaussian likelihood of all drones' observed jammer powers; resample.
   Motion model per mode posterior (constant/sweep/reactive, estimated
   from displacement+chase features exactly as preregistered in v9).
   Jammer truth is NEVER read.
2. **Rollout.** At each IST invocation (T_I = 50): for each candidate
   corridor (existing IST convex-hull boundary pairs within R̂_com) and
   each of M particles: simulate the swarm T_h = 30 steps forward under
   a simplified point-mass model (jammer-repulsion, harm-gated, +
   corridor-endpoint attraction), with the particle's jammer trajectory.
3. **Score.** score(corridor) = P(geometrically connected at horizon ∧
   horizon believed SINR ≥ 12.5 dB PHY gate) − 0.01 × length. Bridge the
   argmax corridor via the existing planned-adjacency machinery; if no
   corridor scores above 0, fall back to standard v2 bridging.
4. **Objective integrity.** The planning objective IS the system
   objective (connectivity probability under the physics) — there is no
   surrogate gap to exploit (the failure mode of the GCN HPO).

## Hypotheses (numeric, before any run)

- **H-A′ (unperturbed):** v10 recovery ≥ v2 recovery − 0.05 in all three
  jammer modes (n=100 paired, McNemar non-inferiority α=0.0167 margin
  0.05). EXPLORATORY first pass at n=30.
- **H-B′ (robustness — the real claim):** under the model-mismatch
  battery (noisy_gps σ=5, range_loss r_c→80, ec5_zero_contrast), v10
  recovery ≥ v2 − 0.10 (n=50 paired). Rationale: sim-in-sim circularity
  is broken only by winning under perturbed physics.
- **H-C′ (latency):** planner wall time ≤ 5.0 s per IST invocation at
  n=200, M=64, on the server CPU (median over 20 invocations).
- **Guardrail:** v10 recovery columns on unperturbed E4 must be
  bit-identical across reruns (seeded determinism check).

## Success / failure

- H-A′+H-B′+H-C′ all pass → v10 claim = "simulation-based self-healing
  is robust where belief-based self-healing failed (EC5)".
- Any failure → reported as null/negative; no post-hoc adjustment.
- Exploratory smoke (n=5/mode) must show: posterior concentrates near
  true jammer in constant mode (mean error < 60 m), determinism (same
  seed → same adjacency), no assertion fires, latency logged per
  invocation.

## Constraints carried

v1–v8 controllers, recovery metrics, seeds, statistics code untouched.
12.5 dB PHY gate only. Corrected throughput accounting. One knob (M).
No ground-truth reads except diagnostics. Determinism: seeded
np.random.Generator per trial.

---

# Exploratory results (locked 2026-11-21, after runs; n=30/n=50, local)

**H-A′ (unperturbed, n=30 paired): PASS.**
constant: v10 0.50 = v2 0.50 (parity; v10 falls back to v2 plan under
posterior certainty); **sweep: v10 0.70 vs v2 0.47, R2C 0.67** (+0.23 over
v2, +0.03 over R2C — the moving-jammer regime the rollouts target);
reactive: v10 0.50 vs v2 0.47. Throughput parity everywhere (±3%).

**H-B′ (robustness, n=50): PASS at the −0.10 bound.**
noisy_gps: v10 0.50 vs v2 0.58 (−0.08, inside bound, weakest cell);
range_loss: v10 0.40 vs v2 0.36 (+0.04); ec5_zero_contrast: v10 0.14 vs
v2 0.10 (+0.04; both collapse vs R2C 0.70 — belief/planner-agnostic
degradation confirmed again).

**H-C′ (latency): PASS by 30x** — 0.17 s median per plan at n=200, M=64
(laptop CPU; bound 5.0 s).

**Determinism guardrail: PARTIAL — flagged.**
Serial reruns: bit-identical (verified twice, all columns). Recovery
columns: bit-identical across serial/parallel/Windows-spawn paths.
Secondary columns (thr/cov_end/mean_q): numeric drift (3rd decimal) on
the Windows spawn parallel path; serial is the reference path. The
program-level "parallel == serial" claim was established on the Linux
forkserver; Windows spawn does not preserve it. Confirmatory n=100 must
run serially or on the Linux server. Recorded as a finding, not fixed
post-hoc.

**Reading.** v10 is the first improvement layer to clear its guardrails:
it wins exactly where designed (moving jammers: forward simulation beats
reactive plans) and holds the robustness bound everywhere tested. The
zero-contrast collapse remains environmental, not architectural. The
confirmatory n=100 (serial/server) is the next step before any claim.
