# A* for the Shared Network: Optimizing the Shared Belief Across Drones

**Branch:** `astar-shared-belief` · **Status:** planning (no code yet)
**Motivation:** v9 Stage A (locked 2026-11-16, `results/preregistration_v9_ig.md`)
found that the IG machinery works (ig_net is λ-sensitive) but the action
effects never alter motion — so the IG layer cannot yet improve the
network. The A* belief-dissemination search is the natural backbone to fix
that: it already decides *which corridors beliefs cross and in what
order*; today its costs are static (time × belief-penalty). Making the
search itself belief-optimal is the missing mechanism.

## Problems to attack (in priority order)

1. **Belief-optimal corridor selection.** Today A* minimizes
   time-to-bridge with a fixed γ·P_belief penalty. Target: edge cost =
   expected *value of information delivered* − cost, so the search routes
   belief deltas where they reduce swarm-level uncertainty the most
   (map-entropy reduction per byte, using the v9 GridFilter's ROI
   entropy), not merely where bridging is fastest.
2. **Shared-network constraint.** The search must account for the
   dissemination budget as a shared resource: corridors compete for the
   same relay airtime (metered C_dis). Target: a budget-feasible plan
   (knapsack/Lagrangian layer over the A* result, λ as the shadow price —
   connecting to v9's single tuning knob).
3. **Delay-free consensus.** Belief versions and tie-breaks already
   propagate with the deltas; target: prove/verify that re-scoring on
   version < current−2 keeps every drone's plan within one IST invocation
   of the swarm plan (no silent divergence), and make the tie-break
   a genuine 1-hop gossip vote rather than a seeded global draw.
4. **Action effects that move drones.** From Stage A (d): flee must
   become a first-class plan primitive inside the search (reposition =
   an edge in the corridor graph whose cost is the IG of the new
   vantage), not a post-hoc gain multiplier.

## Working rules (carried from prior review)

- Preregistration FIRST for any experiment-facing change; one tuning knob
  per variant; exploratory results labeled; no metric substitution; the
  corrected throughput metric and the 12.5 dB PHY gate are frozen.
- `master` stays the reproducible snapshot of everything confirmed;
  all A*/belief work lands here, rebased/merged only via review.
- Reuse `r2c/search.py` (astar, CorridorGraph, BeliefStore) and
  `r2c/ig.py` (GridFilter, ROI entropy) — extend, don't fork.

## First milestones

- M1: edge-cost interface refactor — g_cost becomes a pluggable
  (time, belief-value, budget) triple; v3/v5 behavior reproducible as the
  λ→time-only special case (bit-identical regression test).
- M2: IG-valued edges — corridor cost includes ΔH_roi of the belief the
  crossing would deliver (per byte), wired to the GridFilter.
- M3: budget-feasible planning over the A* output; unit tests for
  optimality and determinism (parallel == serial CSV-identical).
- M4: preregister the confirmatory campaign only after M1–M3 unit tests
  and a labeled smoke pass.
