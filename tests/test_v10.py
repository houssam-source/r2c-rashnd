"""v10 unit tests (preregistration_v10_mc.md smoke criteria)."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2c.config import R2CParams          # noqa: E402
from r2c.environment import SwarmEnvironment  # noqa: E402
from r2c.v10 import MCPlannerController   # noqa: E402
from r2c.rf import RFLayer, RFParams      # noqa: E402


def test_posterior_concentrates_constant():
    p = R2CParams(**{**R2CParams().__dict__, "seed": 0})
    env = SwarmEnvironment(p, 60, "random")
    c = MCPlannerController(p, rf_p=RFParams(jammer_mode="constant", seed=0),
                            seed=0, M=64)
    errs = []
    for t in range(0, 200, 50):
        env.step(c.velocity(env, t))
        pw = c.rf.sensed_jammer_power(env.pos, env.active, t)
        pts, w = c._localize(env.pos, env.active, pw)
        if pts is not None:
            est = np.average(pts, axis=0, weights=w)
            errs.append(np.linalg.norm(est - c.rf.jammer_pos))
    assert errs, "posterior never materialized"
    assert np.mean(errs) < 60.0, f"posterior error {np.mean(errs):.1f} m >= 60"


def test_determinism_same_seed_same_motion():
    def run(seed_ctl):
        p = R2CParams(**{**R2CParams().__dict__, "seed": 3})
        env = SwarmEnvironment(p, 40, "random")
        c = MCPlannerController(p, rf_p=RFParams(jammer_mode="constant", seed=3),
                                seed=seed_ctl, M=32)
        traj = []
        for t in range(120):
            env.step(c.velocity(env, t))
            if t % 25 == 0:
                traj.append(env.pos.copy().round(6).tobytes())
        return traj
    a, b = run(7), run(7)
    assert a == b, "same seed -> different motion"
    # rng wiring: same inputs, different controller seeds -> different
    # particle draws (at small n the plan may not differ, the sampler must)
    p = R2CParams(**{**R2CParams().__dict__, "seed": 3})
    env = SwarmEnvironment(p, 60, "random")
    pwA = None
    draws = []
    for sd in (7, 8):
        c = MCPlannerController(p, rf_p=RFParams(jammer_mode="constant", seed=3),
                                seed=sd, M=64)
        pw = c.rf.sensed_jammer_power(env.pos, env.active, 0)
        pts, w = c._localize(env.pos, env.active, pw)
        draws.append(pts.copy())
    assert not np.allclose(draws[0], draws[1]), "rng not wired to localize"


def test_latency_budget():
    p = R2CParams(**{**R2CParams().__dict__, "seed": 0})
    env = SwarmEnvironment(p, p.n_random, "random")
    c = MCPlannerController(p, rf_p=RFParams(jammer_mode="constant", seed=0),
                            seed=0, M=64)
    for t in range(200):
        if t == p.t1:
            env.inject_failures(0.5)
        env.step(c.velocity(env, t))
    lat = [x for x in c.plan_time if x > 0]
    med = float(np.median(lat)) if lat else 0.0
    print(f"planner latency median {med:.2f}s over {len(lat)} calls")
    assert med <= 10.0, f"latency {med:.2f}s (soft smoke bound)"


def test_physics_guard_and_no_crash_all_modes():
    for mode in ("constant", "sweep", "reactive"):
        p = R2CParams(**{**R2CParams().__dict__, "seed": 1})
        env = SwarmEnvironment(p, p.n_random, "random")
        c = MCPlannerController(p, rf_p=RFParams(jammer_mode=mode, seed=1),
                                seed=1, M=32)
        rf = RFLayer(p, RFParams(jammer_mode=mode, seed=1))
        for t in range(150):
            if t == p.t1:
                env.inject_failures(0.5)
            env.step(c.velocity(env, t))
            thr = rf.throughput(env.pos, env.active, t)
            assert 0 <= thr <= 200 * 30 * 1.01 + 1e-9
    print("all modes: no crash, throughput guard holds")


if __name__ == "__main__":
    test_posterior_concentrates_constant()
    test_determinism_same_seed_same_motion()
    test_latency_budget()
    test_physics_guard_and_no_crash_all_modes()
    print("ALL V10 UNIT TESTS PASSED")
