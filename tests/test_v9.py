"""Unit tests for v9 (rev 2): grid-map information-gain controller.

Spec-required tests:
 1. IG of a certainty-equivalent action is 0; gross IG (dH) >= 0.
 2. lambda=0 reduces to entropy-greedy; lambda->inf reduces to min-cost.
 3. One outlier observation cannot move any cell by more than the clip.
 4. decay() with bias 0 (constant mode) leaves a static map unchanged.
 5. Sum of metered sharing bytes equals logged C_dis totals.
 6. ECE computation matches a hand-checked 10-bin example.

Run: python tests/test_v9.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2c.config import R2CParams                    # noqa: E402
from r2c.environment import SwarmEnvironment        # noqa: E402
from r2c.ig import (IGController, GridFilter, ece_score, DELTA_BYTES)  # noqa: E402
from r2c.rf import RFLayer, RFParams                # noqa: E402


def _make(n=24, mode="constant", seed=0):
    p = R2CParams(**{**R2CParams().__dict__, "seed": seed})
    env = SwarmEnvironment(p, n, "random")
    rf_p = RFParams(jammer_mode=mode, seed=seed)
    return p, env, rf_p


def _graph_of(c, env):
    from r2c.search import CorridorGraph
    import scipy.sparse.csgraph as cg
    pos, active, p = env.pos, env.active, c.p
    mdist = c._mission_dist(env)
    adj = ((mdist <= p.r_c) & active[:, None] & active[None, :]).astype(float)
    np.fill_diagonal(adj, 0.0)
    _, labels = cg.connected_components(adj, directed=False)
    return CorridorGraph(c.ist, pos, active, labels, dist=mdist)


def _roi_cells_of(c, env, t=0):
    pw = c.rf.sensed_jammer_power(env.pos, env.active, t)
    cells = c.grid.cell_of(env.pos[env.active])
    return cells, np.where(pw[env.active] >= 1.0, 2.0, -2.0)


def test_certainty_equivalent_ig_zero_and_ig_nonnegative():
    p, env, rf_p = _make()
    c = IGController(p, rf_p=rf_p, lam=1.0, seed=0)
    c.grid.L[:] = 700.0                     # saturated: p = 1 everywhere
    graph = _graph_of(c, env)
    cells, incs = _roi_cells_of(c, env)
    scores, relay = c._score_actions(env.pos, env.active, graph,
                                     (cells, incs))
    for name, (S, dH, _c) in scores.items():
        assert dH == 0.0, f"{name}: certainty-equivalent dH = {dH}"
    assert all(dH == 0.0 for _S, dH, _c in relay.values())
    # gross IG non-negativity on a random moderate map
    rng = np.random.default_rng(1)
    c.grid.L = rng.normal(0, 2, size=c.grid.L.shape).astype(np.float32)
    scores, relay = c._score_actions(env.pos, env.active, graph,
                                     (cells, incs))
    for name, (S, dH, _c) in scores.items():
        assert dH >= -1e-9, f"{name}: negative gross IG {dH}"
    assert all(dH >= -1e-9 for _S, dH, _c in relay.values())


def test_lambda_extremes():
    p, env, rf_p = _make()
    graph = None
    rng = np.random.default_rng(2)
    base = IGController(p, rf_p=rf_p, lam=0.0, seed=0)
    cheap = IGController(p, rf_p=rf_p, lam=1e9, seed=0)
    graph = _graph_of(base, env)
    for c in (base, cheap):
        c.grid.L = rng.normal(0, 1.5, size=c.grid.L.shape).astype(np.float32)
    cells, incs = _roi_cells_of(base, env)
    s0, _ = base._score_actions(env.pos, env.active, graph, (cells, incs))
    sinf, _ = cheap._score_actions(env.pos, env.active, graph, (cells, incs))
    a0 = max(s0, key=lambda k: s0[k][0])
    ainf = max(sinf, key=lambda k: sinf[k][0])
    dHs = {k: v[1] for k, v in s0.items()}
    cs = {k: v[2] for k, v in sinf.items()}
    assert dHs[a0] == max(dHs.values()), "lambda=0 must be entropy-greedy"
    assert cs[ainf] == min(cs.values()), "lambda->inf must be min-cost"
    assert a0 != ainf or len(set(dHs.values())) == 1, \
        "extreme lambdas should not coincide on distinct scores"


def test_outlier_cannot_move_cell_beyond_clip():
    gf = GridFilter()
    before = gf.L.copy()
    gf.apply(0, np.array([[10, 10]]), [100.0])     # outlier reading
    gf.apply(1, np.array([[20, 20]]), [-100.0])
    gf.apply(2, np.array([[30, 30]]), [2.0])
    d = np.abs(gf.L - before)
    assert d.max() <= 2.0 + 1e-6, d.max()
    assert d[0, 10, 10] == 2.0 and d[1, 20, 20] == 2.0
    assert d[2, 30, 30] == 2.0


def test_zero_bias_decay_leaves_static_map():
    gf = GridFilter()
    rng = np.random.default_rng(3)
    gf.L = rng.normal(0, 1, size=gf.L.shape).astype(np.float32)
    snap = gf.L.copy()
    for _ in range(50):
        gf.decay("constant")            # preregistered bias 0.0
    assert np.array_equal(gf.L, snap), "bias-0 decay changed the map"
    gf2 = GridFilter()
    gf2.L = snap.copy()
    gf2.decay("reactive")               # forgetting factor 0.96 (frozen)
    assert np.allclose(gf2.L, snap * 0.96, atol=1e-6)


def test_metered_bytes_equal_logged_cdis():
    p, env, rf_p = _make(mode="reactive")
    c = IGController(p, rf_p=rf_p, lam=1.0, seed=0)
    for t in range(120):
        env.step(c.velocity(env, t))
    measured = c.counters["dis_bits"]
    logged = float(sum(c._dis_log))
    assert measured > 0, "no dissemination in 120 steps"
    assert measured == logged, (measured, logged)
    # logged events: belief deltas (multiple of DELTA_BYTES x hops)
    # or the 10-byte action/version payload
    for bits in c._dis_log:
        assert bits % DELTA_BYTES == 0 or bits == 10.0, bits


def test_ece_hand_checked():
    # 4 predictions, each in its own bin: ECE = mean |acc - conf|
    pred = [0.05, 0.15, 0.55, 0.95]
    truth = [0, 0, 1, 1]
    got = ece_score(np.array(pred), np.array(truth), n_bins=10)
    want = (abs(0 - 0.05) + abs(0 - 0.15) + abs(1 - 0.55) + abs(1 - 0.95)) / 4
    assert abs(got - want) < 1e-12, (got, want)
    # perfectly calibrated: ECE = 0
    assert ece_score(np.array([0.0, 1.0]), np.array([0, 1])) == 0.0
    # empty bins are skipped, weights by bin count
    pred2 = [0.1, 0.1, 0.9]
    truth2 = [1, 1, 0]
    got2 = ece_score(np.array(pred2), np.array(truth2), n_bins=10)
    want2 = (2 / 3) * abs(1 - 0.1) + (1 / 3) * abs(0 - 0.9)
    assert abs(got2 - want2) < 1e-12, (got2, want2)


if __name__ == "__main__":
    test_certainty_equivalent_ig_zero_and_ig_nonnegative()
    test_lambda_extremes()
    test_outlier_cannot_move_cell_beyond_clip()
    test_zero_bias_decay_leaves_static_map()
    test_metered_bytes_equal_logged_cdis()
    test_ece_hand_checked()
    print("ALL 6 V9 UNIT TESTS PASSED")
