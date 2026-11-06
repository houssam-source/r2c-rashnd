"""Smoke test for SwarmBridge (no ROS, no Gazebo)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))

import numpy as np

from r2c.config import R2CParams
from uav_swarm_r2c.bridge import SwarmBridge


def test_bridge_runs_and_stays_in_arena():
    p = R2CParams(seed=3)
    rng = np.random.default_rng(3)
    for controller in ("r2c", "hybrid"):
        b = SwarmBridge(15, p, controller=controller,
                        jammer_mode="constant", seed=3)
        pos = rng.uniform(0, p.area, size=(15, 2))
        b.update_poses(pos)
        b.inject_failures(0.4, rng)
        for t in range(120):  # past t1; includes several IST invocations
            v = b.compute()
            assert v.shape == (15, 2)
            assert np.all(np.linalg.norm(v, axis=1) <= p.v_max + 1e-9)
            act = b.env.active
            assert np.all(v[~act] == 0.0)
            pos[act] += v[act]
            np.clip(pos, 0, p.area, out=pos)
            b.update_poses(pos)
        assert np.all(pos >= 0) and np.all(pos <= p.area)
    print("bridge smoke test OK (r2c + hybrid)")


if __name__ == "__main__":
    test_bridge_runs_and_stays_in_arena()
