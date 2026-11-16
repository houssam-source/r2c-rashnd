# Pre-Registration — v9: Information-Gain Controller, Spatio-Temporal Probability Map

**Status:** locked 2026-11-11 (rev 2) before any v9-map run. Supersedes the
2026-11-11 rev-1 feature-posterior draft (same file; rev 1 archived in git
history of this session's notes) — the belief source is now a matrix, not
feature posteriors.
**Variant slot:** v9. **New tuning knobs:** exactly ONE — `lambda` (C_dis price).
**Implements:** spec "v9 — IG controller with spatio-temporal probability map"
(2026-11-11), Layers 1–4. Pure NumPy. No learned components.

## Layer 1 — GridFilter: the belief source

Belief stack `B`, shape `(3, 200, 200)` float32, world frame, 5 m cells over
the 1000×1000 m arena:

- channel 0: `P(jammer present at cell)`
- channel 1: `P(link up | cell)`
- channel 2: `node health` (P(node present at cell))

Stored internally as LOG-ODDS `self.L` (init 0 ⇒ p = 0.5); probability is
computed only on read: `p = 1/(1+exp(-L))`. Robust update: only observed
cells, `L[cells] += clip(sensor_logodds, -2, 2)` — one corrupted reading
cannot flip a cell. Per-mode temporal process noise (PREREGISTERED design
constants, not tuned post hoc): multiplicative log-odds forgetting
`decay(): L *= f` with `f = 1.0` (constant — a static world accumulates
evidence exactly), `0.95` (sweep), `0.96` (reactive). AMENDED
2026-11-14 (pre-confirmatory, smoke-diagnosed): the original additive
bias could not relax stale negative trail evidence — a moving jammer's
erased path never recovered and the head cell could not stand out. Values
frozen at the smoke campaign: a source crossing 5 m cells at up to
12.6 m/step needs stale evidence to decay within ~30 steps or the
previous pass's hot trail outvotes the fresh pocket.

Sensor models (deterministic, anti-cheat — received powers/beacons only).
REVISED at smoke (2026-11-11, pre-confirmatory; original per-cell scheme
saturated or carried no spatial gradient — amendments logged):
- ch0: range-based RF tomography. Observed jammer power at drone i implies
  the jammer lies on a circle of radius d_i (free-space inversion);
  annulus cells (|dist − d_i| ≤ 1 cell) get `+2` (ANNULUS_GAIN, within the
  clip bound); cells strictly inside the circle get the per-mode
  NEAR_WEIGHT (they would imply a *higher* received power than observed):
  `−2.0` constant (stationary source), `−0.05` sweep, `−0.2` reactive —
  a moving source only satisfies the static assumption instantaneously, so
  interior negatives are weighted down by mobility (smoke-diagnosed trail
  erasure; preregistered as design constants 2026-11-14, pre-confirmatory).
  Detections with d_i > 1.1×arena are uninformative (jammer out of
  informative range) and are SKIPPED — no evidence flooding.
- ch1: each observed pair (i,j) updates its LINK-MIDPOINT cell:
  quality-gated pairs `+1`, in-range gated-out pairs `−1` (channel
  semantics: P(link up | cell) lives at the link midpoint).
- ch2: signed presence evidence per deployed node at its current cell:
  active `+0.5`, inactive `−2.0` (failures saturate to p≈0 within ~30
  steps; node moves leave the marginal intact).
- All updates at the full 10 Hz control cadence; clip(±2) robust bound
  per application unchanged.

## Layer 2 — Vectorized net-IG scoring

For each action `a ∈ {construct relay, flee/reposition, switch channel,
hold & observe}`:

```
S(a) = [ H(B_roi) − H(B_pred_roi) ] − lambda * C_dis(a)
H = shannon(p).sum() over the action's ROI slice only
shannon(p) = −(p·log p + (1−p)·log(1−p)), p clipped to [1e-9, 1−1e-9]
```

Vectorized over cells and channels (numpy on `(3, h, w)` slices); only a
scalar leaves the matrix. ROI slices (world frame):
- hold & observe: bounding box of active drones' cells.
- flee/reposition: that box shifted one v_max step along the flee
  (repulsion) direction, plus a 2-cell margin.
- switch channel: observed box with predicted increments doubled
  (clean-channel re-sense).
- construct relay (per corridor k→m): bounding box of the two clusters'
  boundary cells.

`C_dis(a)` = metered bytes of log-odds deltas this action's update would
disseminate: 6 bytes per changed cell (int16 cell index + float32 delta)
× changed cells × A* dissemination path hops. The circularity (spending
bandwidth to save the network) is PRICED IN, never ignored. Predicted C_dis
is scored before acting; actual metered bytes are logged; both reported.

`lambda` is the ONLY tuning knob. Stage A explores
`lambda ∈ {0, 0.1, 1, 10, 100}` on n=30 paired seeds (E4-constant +
occluded terrain; selection rule: recovery primary, throughput tie-break —
same procedure as preregistration_field_tune.md Stage A). Stage B is
confirmatory at the selected lambda, n=100. One knob, one selection.

## Layer 3 — Decision & execution

- `S(a)` feeds the existing A* belief-dissemination search as edge cost:
  relay-corridor edges use
  `g'(k,m) = (length/v_max)·(1+γ·norm_power(corr))·(1 − s̄(k,m))`,
  `s̄ = clip(S_relay(k,m)/S_ref, 0, 1)`, `S_ref = 2.0` (constant, not data
  normalization).
- Hard gates: (i) geometric connectivity (existing IST machinery); (ii) the
  corridor midpoint's believed SINR must clear the **12.5 dB packet gate**
  (GNU Radio-measured, PHY_VALIDATION.md) — never the legacy 10 dB gate.
- Deterministic seeded neighbor-consensus tie-break on equal scores
  (np.random.default_rng(seed), majority over clusters, seeded draw on
  exact ties).
- Swarm-level action effects per IST invocation: hold → skip bridging,
  re-observe; flee → standard plan, repulsion gain ×1.5 this invocation;
  switch → force full re-observe (double increments this step) then
  standard plan; relay → standard plan with S-discounted edges.

## Layer 4 — Sharing, staleness, calibration

- Drones never receive the full matrix (480 KB at 10 Hz is the bug class
  being eliminated). Shared payload per invocation: chosen action (1 B
  code), tie-break result (1 B), belief VERSION number (uint32 version +
  uint32 step timestamp = 8 B) — plus log-odds DELTAS of changed cells
  only (6 B/cell, metered).
- **Staleness rule (preregistered):** an action scored on belief version
  < current−2 is re-scored once against the current map; if the re-scored
  action still relies on cells older than version−2, it is DISCARDED
  (falls back to hold & observe). No silent staleness.
- **IG is internal only.** Headline metrics remain exactly the v1–v8
  pipeline: recovery rate and corrected throughput (Mbps,
  per-drone-radio-capped, physical-ceiling assert — commit 3ce791e). IG
  never appears in results tables.
- **Calibration gate (ECE):** each episode, 10-bin expected calibration
  error of each channel against simulator ground truth (ch0: jammer cell
  neighborhood; ch1: sampled gated-adjacency cells; ch2: active-node
  cells). Preregistered gate: **mean ECE across channels < 0.05**, else
  the map is declared miscalibrated and the episode is FLAGGED (reported
  alongside; H-B sensitivity reported with and without flagged episodes).
  ECE logged every episode.

## Hypotheses (numerical, stated before any run)

- **H-A (dominates the design partition):** v9 recovery ≥ v4 recovery −
  0.05 in **occluded terrain** (v4 battery, n=100 paired seeds, McNemar
  non-inferiority, margin 0.05) **AND** v9 recovery ≥ v5 recovery − 0.05 on
  **jammer_move** (confirmatory_v5 battery, n=100). Throughput legs: v9
  thr_mbps ≥ 0.95 × comparator mean on both.
- **H-B (IG validity):** per-episode net-IG rate (mean S(a*) per step)
  correlates with task success (Spearman ρ over the n=100 v9 episodes;
  success = recovered; REC secondary), **ρ > 0.3** and **p < 0.05/3**.

**Confirmatory family = 3 tests** (H-A-terrain, H-A-move, H-B);
**Bonferroni α = 0.0167**. Exact McNemar two-sided for recovery legs;
paired-bootstrap 95% CI (10k, seed 12345) alongside. ECE gate is a
calibration flag, not a hypothesis test.

## Success / failure criteria (numeric, before runs)

- **H-A PASS** iff both non-inferiority legs reject inferiority at
  α=0.0167 (margin 0.05) AND both throughput legs ≥ 0.95× comparator.
  Otherwise H-A FAIL.
- **H-B PASS** iff ρ > 0.3 with p < 0.0167 on recovery (REC exploratory,
  labeled if it disagrees).
- **v9 claim ("one controller, both regimes") only if H-A AND H-B pass.**
  Any other outcome reported as null/negative.
- **ECE gate:** mean 3-channel ECE < 0.05 per episode; episodes failing
  the gate are flagged, counted, and reported; if >20% of episodes are
  flagged, the variant is declared miscalibrated and H-B is not
  interpretable (reported as failed gate, no claim).
- **Smoke (n=5/mode, exploratory, no alpha claims):** 10 Hz loop holds
  (400 steps, T_I=50 invocations); belief error (MAP-cell distance to true
  jammer) decreases within-episode for the stationary jammer and stays
  bounded (±10%) for moving jammers; ECE gate passes on all smoke episodes;
  zero assertion failures (incl. the throughput ceiling assert); CSV
  schema = v1–v8 columns + {ig_net, c_dis_bits, c_dis_pred_bits,
  belief_version, ece_map, ece_link, ece_health, belief_err, lambda}.

## Constraints carried

- v1–v8 controllers, recovery metrics, seeds, statistics code, and other
  variants' preregistrations untouched. No learned/neural map extractor.
  No min-max/z-score normalization. No clamping at the physical ceiling
  (assert only). Determinism: seeded Generator per run; parallel
  --workers output CSV-identical to serial (per-trial seeding).
- Packet gate 12.5 dB used for v9 planning only; the v1–v8 metric pipeline
  is unchanged.
- Confirmatory per-seed writers gain the diagnostic columns at
  confirmatory time (schema additive only).


---

# Stage A results (locked 2026-11-16, after runs)

**Campaign:** server (48 workers), n=30 paired seeds per cell.
Files: `results/stageA_e4_v9lam.csv` (E4-constant, λ grid + R2C/v2),
`results/stageA_terrain_v9lam.csv` (occluded terrain, λ grid + blind),
`results/stageA_move_v9lam.csv` (AMENDMENT, pre-confirmatory: E4-sweep +
E4-reactive λ grid — the preregistered scenarios turned out to be
degenerate for λ; amendment logged below).

**Outcome 1 (preregistered scenarios):** λ has NO effect on recovery or
throughput — all five λ values are BIT-IDENTICAL in E4-constant and in
occluded terrain. Diagnosis: E4-constant saturates the belief pre-t1
(dH ≈ 0, ig_net = 0.000); the terrain arena (N=30, AREA=200) keeps the
swarm in one cluster, where the plan early-returns and v9 degenerates to
the v3b path exactly.

**Outcome 2 (amended moving-jammer scenarios):** ig_net is λ-sensitive
(0.072 → 0.064 → 0.035 → 0.000 across λ=0/1/10/100; c_dis metered
identically) — the scoring and pricing machinery works — yet recovery and
throughput remain BIT-IDENTICAL for all λ. Diagnosis: the preregistered
action EFFECTS never alter velocity: hold ≡ standard bridging when no
corridors exist, switch only double-observes, relay needs passable
corridors, and flee (the only motion effect, repulsion ×1.5) is never the
argmax because switch's predicted dH dominates at λ=0 and hold's zero cost
dominates at λ>0.

**Selection per the preregistered rule:** all λ tie on the primary and the
tie-break → **λ = 1.0 retained** (middle of the explored grid, equal to
the provisional value; no post-hoc fitting occurred because there is
nothing to fit).

**Consequence for H-A (flagged for user review before any confirmatory
run):** as mechanized, the IG layer prices and meters information but does
not change motion in the tested regimes, so a v9-vs-v4/v5 recovery
advantage cannot come from λ. Options: (a) proceed to confirmatory with
H-B only; (b) amend the phase-3 action effects (e.g., flee as a first-
class motion primitive, relay-driven repositioning) — this is a design
change requiring a preregistration revision and a fresh Stage A. No
confirmatory n=100 has been run.
