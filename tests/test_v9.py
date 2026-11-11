"""Unit tests for the v9 information-gain controller (preregistration_v9_ig).

Run: python -m pytest tests/test_v9.py -q   (or: python tests/test_v9.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2c.config import R2CParams          # noqa: E402
from r2c.environment import SwarmEnvironment  # noqa: E402
from r2c.ig import IGController, PACKET_GATE_DB  # noqa: E402
from r2c.rf import RFParams               # noqa: E402


def _make(n=24, mode="constant", seed=0):
    p = R2CParams(**{**R2CParams().__dict__, "seed": seed})
    env = SwarmEnvironment(p, n, "random")
    rf_p = RFParams(jammer_mode=mode, seed=seed)
    return p, env, rf_p


def test_certainty_equivalent_ig_is_zero_and_ig_nonnegative():
    """With a certain belief (conf=1, one-hot mode posterior, healthy nodes)
    every action's S must be 0; in arbitrary states S must be >= 0 for the
    always-feasible actions at lambda=0 (entropy reduction is non-negative
    by construction)."""
    p, env, rf_p = _make()
    c = IGController(p, rf_p=rf_p, lam=0.0, seed=0)
    # force certainty (mode posterior exactly one-hot to machine precision)
    c.mode_alpha = np.array([1e15, 1.0, 1.0])
    c._misses = np.zeros(len(env.pos))
    c._prev_jhat = np.zeros(2)
    from r2c.search import CorridorGraph
    import scipy.sparse.csgraph as cg
    pos, active = env.pos, env.active
    adj = ((env.distances() <= p.r_c) & active[:, None]
           & active[None, :]).astype(float)
    np.fill_diagonal(adj, 0.0)
    _, labels = cg.connected_components(adj, directed=False)
    graph = CorridorGraph(c.ist, pos, active, labels)
    pw = c.rf.sensed_jammer_power(pos, active, 0)
    for k in graph.parts:
        key = c.beliefs.key(graph.parts[k])
        c.beliefs.store[key] = dict(vec=pos[graph.parts[k]].mean(axis=0),
                                    conf=1.0, age=0, source="local")
    # topology also certain: no believed-passable corridors -> P_conn = 0
    scores, relay = c._score_actions(pos, active, graph, pw, [])
    for name, (S, dH, _) in scores.items():
        assert abs(S) < 1e-9, f"{name} IG not zero under certainty: {S}"
    assert all(abs(S) < 1e-9 for S, _, _ in relay.values())


def test_cdis_accounting_matches_logged_bandwidth():
    """Every dissemination event must be counted exactly once: the measured
    counter equals the sum of logged per-event bits."""
    p, env, rf_p = _make(mode="reactive")
    c = IGController(p, rf_p=rf_p, lam=1.0, seed=0)
    for t in range(120):                    # spans several IST invocations
        env.step(c.velocity(env, t))
    measured = c.counters["dis_bits"]
    logged = float(sum(c._dis_log))
    assert measured > 0, "no dissemination happened in 120 steps"
    assert abs(measured - logged) < 1e-9, (measured, logged)
    d = c.diagnostics()
    assert d["c_dis_bits"] == measured


def test_lambda_extremes_select_expected_actions():
    """lambda=0 -> entropy-greedy (switch channel, max dH);
    lambda->inf -> min-cost (hold & observe, c=0)."""
    p, env, rf_p = _make()
    base = IGController(p, rf_p=rf_p, lam=0.0, seed=0)
    cheap = IGController(p, rf_p=rf_p, lam=1e9, seed=0)
    for c in (base, cheap):
        c._misses = np.zeros(len(env.pos))
        c.mode_alpha = np.array([2.0, 2.0, 2.0])   # uncertain -> dH > 0
    from r2c.search import CorridorGraph
    import scipy.sparse.csgraph as cg
    pos, active = env.pos, env.active
    adj = ((env.distances() <= p.r_c) & active[:, None]
           & active[None, :]).astype(float)
    np.fill_diagonal(adj, 0.0)
    _, labels = cg.connected_components(adj, directed=False)
    graph = CorridorGraph(base.ist, pos, active, labels)
    pw = base.rf.sensed_jammer_power(pos, active, 0)
    for k in graph.parts:
        for c in (base, cheap):
            c.beliefs.store[c.beliefs.key(graph.parts[k])] = dict(
                vec=pos[graph.parts[k]].mean(axis=0), conf=0.2, age=0,
                source="local")
    passable = [(a, b) for a in graph.edges for b, _ in graph.edges[a]
                if a < b]
    s0, _ = base._score_actions(pos, active, graph, pw, passable)
    sinf, _ = cheap._score_actions(pos, active, graph, pw, passable)
    a0 = max(s0, key=lambda k: s0[k][0])
    ainf = max(sinf, key=lambda k: sinf[k][0])
    assert a0 == "switch channel", f"lambda=0 should be entropy-greedy, got {a0}"
    assert ainf == "hold & observe", f"lambda->inf should be min-cost, got {ainf}"
    # and the other two actions sit strictly between
    assert s0["hold & observe"][0] < s0["switch channel"][0]
    assert sinf["switch channel"][0] < sinf["hold & observe"][0]


def test_physics_guard_on_throughput():
    """Closed loop must never exceed the per-drone radio cap (assert guard
    fires inside RFLayer.throughput otherwise)."""
    p, env, rf_p = _make(mode="sweep", seed=3)
    c = IGController(p, rf_p=rf_p, lam=1.0, seed=3)
    n = len(env.pos)
    for t in range(150):
        env.step(c.velocity(env, t))
        thr = c.rf.throughput(env.pos, env.active, t)
        assert 0.0 <= thr <= n * 30.0 * 1.01 + 1e-9
    d = c.diagnostics()
    for key in ("ig_net", "c_dis_bits", "belief_err", "mode_err", "lambda"):
        assert key in d


if __name__ == "__main__":
    test_certainty_equivalent_ig_is_zero_and_ig_nonnegative()
    test_cdis_accounting_matches_logged_bandwidth()
    test_lambda_extremes_select_expected_actions()
    test_physics_guard_on_throughput()
    print("ALL V9 UNIT TESTS PASSED")
