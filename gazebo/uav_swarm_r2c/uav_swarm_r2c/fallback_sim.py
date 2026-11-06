"""Kinematic fallback simulator: mocks Gazebo by integrating the bridge's
velocity setpoints, runs the paper protocol end-to-end with no ROS/Gazebo.

This validates the exact code path the Gazebo/real-drone deployment uses:
    bridge.update_poses -> bridge.compute -> world integrates

Run:  PYTHONPATH=. python gazebo/uav_swarm_r2c/uav_swarm_r2c/fallback_sim.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))

import numpy as np

from r2c.config import R2CParams
from r2c.rf import RFLayer, RFParams
from uav_swarm_r2c.bridge import SwarmBridge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drones", type=int, default=20)
    ap.add_argument("--controller", default="hybrid",
                    choices=["r2c", "hybrid"])
    ap.add_argument("--jammer", default="constant",
                    choices=["constant", "sweep", "reactive"])
    ap.add_argument("--failures", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    p = R2CParams(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    bridge = SwarmBridge(args.drones, p, controller=args.controller,
                         jammer_mode=args.jammer, seed=args.seed)
    rf = RFLayer(p, RFParams(jammer_mode=args.jammer, seed=args.seed))
    bridge.set_jammer_pose(*rf.jammer_pos)

    # initial deployment (random in the arena)
    pos = rng.uniform(0, p.area, size=(args.drones, 2))
    bridge.update_poses(pos)
    bridge.inject_failures(0.0)  # all active

    dt = 1.0
    t_rec, recovered = None, False
    thr_sum, q_sum, steps = 0.0, 0.0, 0
    for t in range(p.T):
        if t == p.t1:
            bridge.inject_failures(args.failures, rng)
            down = int((~bridge.env.active).sum())
            print(f"t={t}: {down}/{args.drones} drones failed")
        # reactive jammer chases the swarm centroid (world truth)
        if args.jammer == "reactive" and t > 0:
            act = bridge.env.active
            c = pos[act].mean(axis=0)
            d = c - rf.jammer_pos
            n = np.linalg.norm(d)
            if n > 1e-9:
                rf.jammer_pos += d / n * min(rf.rf.jammer_speed, n)
            bridge.set_jammer_pose(*rf.jammer_pos)

        v = bridge.compute()                      # controller setpoints
        act = bridge.env.active
        pos[act] += v[act] * dt                   # 'Gazebo' integrates
        np.clip(pos, 0.0, p.area, out=pos)
        bridge.update_poses(pos)

        if t > p.t1:
            thr_sum += rf.throughput(pos, bridge.env.active, t)
            q_sum += rf.mean_quality(pos, bridge.env.active, t)
            steps += 1
            if not recovered and bridge.env.active.sum() > 1:
                # mirror the E4 quality-gated recovery check
                g = rf.gated_adjacency(pos, bridge.env.active, t)
                n_comp = __import__("scipy.sparse.csgraph", fromlist=["x"])
                adj = ((np.linalg.norm(pos[:, None] - pos[None], axis=-1)
                        <= p.r_c) & bridge.env.active[:, None]
                       & bridge.env.active[None, :]).astype(float)
                k = n_comp.connected_components(adj, directed=False)[0]
                if k == 1 and rf.mean_quality(pos, bridge.env.active, t) > 0.5:
                    t_rec = t - p.t1
                    recovered = True
        if t % 100 == 0:
            print(f"t={t:3d} active={int(bridge.env.active.sum()):3d} "
                  f"|v| mean={np.linalg.norm(v[act], axis=1).mean():.3f}")
    print(f"\n[{args.controller}/{args.jammer}] {args.drones} drones, "
          f"failures={args.failures:.0%}")
    print(f"  mean throughput = {thr_sum/steps:.0f} Mbps")
    print(f"  mean quality    = {q_sum/steps:.4f}")
    print(f"  recovered       = {recovered} (t_rec={t_rec})")


if __name__ == "__main__":
    main()
