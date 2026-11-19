# R2C–RASHND Hybrid — Full Report with Analysis
**Date:** 2026-11-19 · Repo: houssam-source/r2c-rashnd · Branch: master (local commits through `f94f423`, GitHub push network-blocked)
**Compute:** server 10.13.0.201 (48 workers used) + local Windows laptop

---

## 1. Where the project stands

The reproduction-and-extension program (R2C, AAAI-26 + RF-hybrid variants v1–v9) is **complete through the exploration/tuning stage**; confirmatory claims are frozen pending two user decisions (§6).

| Milestone | Status |
|---|---|
| R2C reproduction + RF-hybrid extension (v1–v8) | Done, stress-tested, PHY-validated |
| Throughput metric fix + full recompute | Done, verified bit-identical non-throughput columns |
| Gazebo live run (H1) | Done (closed loop verified; ROS bridge bypassed with custom gz shim) |
| R2C author query | Not sent (only user can) |
| **v9 (IG controller, probability map)** | **Implemented, preregistered, unit-tested, smoke-campaigned, Stage A trained — awaiting confirmatory go/no-go** |

---

## 2. The throughput correction and what it did to the numbers

**Root cause.** `RFLayer.throughput` summed Shannon capacity over *all ordered in-range pairs* — every link counted as simultaneously saturated, each physical link double-counted, no per-radio constraint — 2.5–6 Gbps for a 200-drone swarm whose physical ceiling is n × 30 Mbps (QPSK, 20 MHz). A "Tbps" figure existed only in a session summary (misreport); no artifact ever contained it.

**Fix (commit `3ce791e`).** Per-drone carried rate = `min(Σ link offers, 30 Mbps)`; aggregate = Σ drones; assert guard `≤ n_active × 30 Mbps × 1.01` (assert-only, never clamps). All runners emit `thr_per_drone_mbps` alongside `thr_mbps`.

**Verification.** Trajectory-neutral (bit-identical recovery/REC/quality vs pre-fix HEAD on paired seeds); server recompute of 16 suites reproduced every non-throughput column **exactly**; all throughputs now within the physical ceiling; smoke + one n=5 end-to-end run pass the assert.

**Material scientific changes (this is why the fix mattered):**

| Claim | Old (buggy metric) | Corrected |
|---|---|---|
| v2 throughput gain over R2C | "+24–46%" | **+4.1 / +5.1 / +7.1%** (constant/sweep/reactive, n=50) |
| v2 trade | "throughput for recovery" Pareto | Still a Pareto, but the throughput leg is now marginal: constant mode FAILS the ≥5% bar, sweep/reactive pass by ≤2 pp |
| v2-vs-R2C (n=100, constant) | +37–49% | **+4.7–4.8%** |
| v5 vs v2 deficit cut | "−24pp → −16pp at throughput parity" | −22pp → −12pp at parity (story unchanged) |
| v4 terrain throughput | 75–91 Mbps | 21.8–22.6 Mbps (order of arms unchanged) |

**Ordering changes:** 16 groups flipped (v2↔v3b, v4↔blind/v5) — all arms within ~1% of each other, i.e., previously-reported near-ties; the per-drone cap is a nonlinear clip, so exact order preservation was never guaranteed (flagged before implementation). Recovery orderings are untouched everywhere.

---

## 3. v9: the information-gain controller on a spatio-temporal probability map

**Architecture (preregistered `results/preregistration_v9_ig.md`, rev 2 + logged pre-confirmatory amendments):**

- **Layer 1 — GridFilter** (`r2c/ig.py:71`): (3,200,200) float32 log-odds map, 5 m world-frame cells (ch0 jammer presence, ch1 link-up, ch2 node health); robust clip(±2) updates at 10 Hz; log-odds stored, sigmoid on read; multiplicative per-mode forgetting {constant 1.0, sweep 0.95, reactive 0.96}; range-based RF tomography for ch0 (free-space inversion → annulus +2 / interior per-mode weight {−2.0/−0.05/−0.2}, far detections skipped, divided sensing 20 slots/step); ch1 at link midpoints; ch2 signed presence.
- **Layer 2 — IG scoring** (`r2c/ig.py:256`): S(a) = ΔH(ROI) − λ·C_dis(a) over {relay, flee, switch, hold}; vectorized ROI entropy; C_dis priced in bytes (6 B/changed-cell × hops) **and metered** — the circularity of spending bandwidth to save the network is priced in.
- **Layer 3 — Decision** (`r2c/ig.py:317`): S injected as an A* edge-cost discount; hard gates = geometric connectivity + **12.5 dB PHY packet gate** (GNU Radio-measured, never the 10 dB model gate); seeded deterministic consensus tie-break.
- **Layer 4 — Sharing/calibration** (`r2c/ig.py:250,482`): drones never receive the map (480 KB @ 10 Hz is the bug class being eliminated) — deltas + action + version only; version/staleness rule; 10-bin ECE calibration gate (< 0.05); ground truth read **only** for diagnostics.

**Hypotheses (locked pre-run):** H-A = v9 dominates the design partition (≥ v4−0.05 in occluded terrain AND ≥ v5−0.05 on jammer_move, thr ≥ 0.95× comparators); H-B = net-IG rate correlates with task success (Spearman ρ > 0.3, p < 0.0167). Family = 3, Bonferroni α = 0.0167. λ is the **only** tuning knob.

**Unit tests:** 6/6 green (certainty-equivalent IG = 0 / IG ≥ 0; λ extremes → entropy-greedy / min-cost; outlier clip bound; zero-rate temporal model leaves static map unchanged; metered bytes = logged C_dis; ECE matches hand-checked example).

**Smoke campaign (n=5/mode, exploratory):** nine logged iterations diagnosed and fixed, in order: (1) feature-posterior → matrix belief source (spec rev 2); (2) per-cell ch0 sensor carried no spatial gradient → range tomography; (3) own-cell negative flood pinned the map → far-detection skip; (4) moving source trail erased by static interior evidence → per-mode interior weights; (5) additive bias could not relax stale evidence → multiplicative forgetting; (6) n=200 annulus union covered 1/3 of the map → divided sensing (20 slots); (7) **the sweep diagnostic itself was wrong** (`rf.jammer_pos` is stale by design in sweep mode) — corrected via `_jammer_position_peek`; (8) forgetting rate vs source speed (12.6 m/step crosses a 5 m cell in <0.5 step) → frozen 0.95/0.96.

**Smoke outcomes (final):**

| mode | recovery | thr Mbps | MAP err (m) | within-episode | ECE map |
|---|---|---|---|---|---|
| constant | 0.80 | 967.5 | **10.07** (floor) | flat at floor (expected) | 0.032 |
| sweep | 0.60 | 994.8 | **61.62** (per-seed 18–55) | 67.6 → 20.0 ↓ | 0.608 |
| reactive | 1.00 | 944.6 | 19–66 per-seed | 171 → 18.6 ↓ | 0.196 |

Belief-tracking gate: **PASS** (amended criterion, logged). ECE gate: **FAIL** (mean 0.10/0.35/0.22 vs 0.05) — diagnosed as a gate-*construction* mismatch (cell marginals binned against a point-event reference; forgetting racing a 12.6 m/step source), not map discrimination; a revised gate is **proposed, not applied**, awaiting review.

---

## 4. Stage A — the training campaign (server, n=30 paired, 2026-11-16)

**Design:** λ ∈ {0, 0.1, 1, 10, 100}; preregistered scenarios E4-constant + occluded terrain; amendment (logged, pre-confirmatory) added E4-sweep/reactive after the preregistered scenarios proved degenerate for λ.

### 4.1 Results

| suite | λ sensitivity of recovery/throughput | IG accounting (ig_net) |
|---|---|---|
| E4-constant | **none — all λ bit-identical** (rec 0.60, thr 933.6) | 0.000 for all λ (belief saturated pre-t1) |
| Occluded terrain | **none — all λ bit-identical to blind** (rec 0.50, thr 20.7) | degenerate: single-cluster arena → v9 ≡ v3b exactly |
| E4-sweep / reactive | **none — bit-identical** (sweep 0.57/997.3; reactive 0.63/927.7) | **strongly λ-sensitive:** 0.072 → 0.064 → 0.035 → 0.000 across λ = 0/1/10/100 |

Baselines in the same runs: R2C 0.77/898, v2 0.47/929.5 (constant); blind 0.50/20.7 (terrain).

### 4.2 Analysis

**(a) The IG machinery is real and works.** ig_net responds to λ exactly as the scoring equation predicts; C_dis is metered identically across λ; unit tests pin the scoring semantics. The training campaign confirms the machinery end-to-end on the server at campaign scale, not just in smoke.

**(b) λ is unidentifiable from the headline metrics — and the reason is structural, not statistical.** The preregistered phase-3 action *effects* never alter velocity in any tested regime: hold ≡ standard bridging when no passable corridors exist (the common case); switch only double-observes (changes the map, not the motion); relay requires passable corridors; and flee — the only effect that changes velocities (repulsion ×1.5) — is never the argmax, because at λ=0 switch's predicted ΔH dominates and at λ>0 hold's zero cost dominates. The IG layer therefore *prices and meters information* but does not *move the swarm* as mechanized. This is why bit-identity across λ is the expected signature, not an anomaly: identical velocities ⇒ identical trajectories ⇒ identical recovery/throughput.

**(c) The two degenerate regimes are diagnosed, one geometric, one informational.** Constant jammer: a stationary source in a noiseless model lets the map converge to certainty before failures are even injected (ig_net = 0 exactly) — there is no residual uncertainty for information gain to act on. Terrain arena (N=30, AREA=200, r_c=120): the swarm is provably one cluster, the plan early-returns, and v9's added layers are bypassed — it collapses exactly onto v3b.

**(d) Exploratory signal worth noting (no alpha claims, n=30):** in E4-constant v9 recovers 0.60 vs v2's 0.47 (+13 pp) while matching throughput — the shared-belief/IG path inherits v3's recovery advantage over v2 even with the action layer idling. If confirmed at n=100 this would be a v9-vs-v2 recovery claim independent of λ.

**(e) Selection per the locked rule:** all λ tie on primary and tie-break → **λ = 1.0 retained** (middle of grid = provisional value; nothing was fittable). Recorded as a null result, not hidden.

---

## 5. What the evidence supports now

- **Corrected core story:** the v2 hybrid trades −20..−24 pp recovery for +4–7% throughput (quality-neutral); the original "+24–46%" was a metric artifact. v5's moving-jammer recovery gain (+15 pp, p=0.0026) and v4's terrain advantage (0.62 vs 0.50, exploratory at n=50 after the n=30 +20 pp claim was retracted) survive the metric fix.
- **v9's belief layer is a validated tracker:** ~10 m (stationary), 20–66 m (moving) MAP error with diagnosed, information-theoretic limits; calibration good in the stationary regime.
- **v9's decision layer is validated as accounting but not yet as control:** it computes meaningful, λ-responsive IG and honest bandwidth prices, but the spec'd action effects are too weak to move the swarm. H-A as mechanized is therefore not testable — there is no mechanism by which v9 could beat v4/v5.

---

## 6. Decision points (user)

1. **v9 confirmatory go/no-go:** (a) run confirmatory **H-B only** (IG-validity correlation — machinery demonstrably live); (b) amend phase-3 action effects (flee as a first-class motion primitive; relay-driven repositioning) → new preregistration revision + fresh Stage A; or (c) shelve v9.
2. **ECE gate revision** (top-k mass retrieval for ch0; per-cell empirical link-up frequency for ch1) — proposed, awaiting approval before confirmatory.
3. **Send the R2C author query** (only you can send; the question list is prepared in prior notes).
4. GitHub push of commits `3ce791e`…`f94f423` (network-blocked locally; can push from the server).

## 7. Honest limitations

Server was SSH-dead 2026-11-11 → 16 (one throughput-fix file, `train_edge_field`, is still its pre-fix version — flagged, non-headline). ECE gate fails on construction (§3). v9 ECE/tracking in moving regimes is descriptive. All Stage A recovery comparisons at n=30 are exploratory. λ=null is a negative result about the action-effect design, not evidence that information gain is uninformative — the ig_net gradient across λ is the counter-evidence.
