"""Gazebo <-> R2C/hybrid controller bridge (core logic, ROS-free).

This module is the single source of truth for flying the R2C / hybrid
controllers on real or simulated drones:

- Simulation (this repo): r2c.sim integrates the velocities itself.
- Gazebo:   bridge_node.py feeds measured poses in and sends the velocity
            setpoints from `compute()` out (VelocityControl plugin or MAVROS).
- Real UAVs: identical interface — poses from GPS/mocap/UWB in, velocity
             setpoints to MAVROS `setpoint_velocity/cmd_vel` (OFFBOARD).

The controller never integrates motion and never invents state: positions
come from the outside world each cycle, exactly like r2c.sim's env.step but
with Gazebo/reality as the physics engine.
"""
import numpy as np

from r2c.config import R2CParams
from r2c.controller import R2CController
from r2c.hybrid import HybridController
from r2c.rf import RFLayer, RFParams


class SwarmBridge:
    """Pose-in, velocity-out bridge around R2CController/HybridController.

    Usage per control cycle (e.g. 10 Hz):
        bridge.update_poses(xy)          # measured Nx2 positions [m]
        bridge.set_active(mask)          # failed/crashed drones -> False
        bridge.set_jammer_pose(x, y)     # RF-shadow truth (world model)
        setpoints = bridge.compute()     # Nx2 velocities [m/s], |v| <= v_max
    """

    def __init__(self, n: int, params: R2CParams | None = None,
                 controller: str = "r2c", jammer_mode: str = "constant",
                 seed: int = 0, variant: str = "v2"):
        self.p = params or R2CParams()
        self.n = n
        self.t = 0
        self.controller_kind = controller
        self.jammer_mode = jammer_mode
        self.env = _BareEnv(self.p, n)
        if controller == "hybrid":
            self.ctrl = HybridController(
                self.p, rf_p=RFParams(jammer_mode=jammer_mode, seed=seed),
                variant=variant)
        else:
            self.ctrl = R2CController(self.p)

    # ---- world interface ---------------------------------------------------
    def update_poses(self, xy: np.ndarray):
        xy = np.asarray(xy, dtype=float)
        assert xy.shape == (self.n, 2), f"poses must be ({self.n}, 2)"
        self.env.pos = xy.copy()

    def set_active(self, mask):
        self.env.active = np.asarray(mask, dtype=bool).copy()

    def set_jammer_pose(self, x: float, y: float):
        """Set the RF-shadow jammer position (world truth for the quality
        map). The controller's belief stays estimated from sensing."""
        rf = getattr(self.ctrl, "rf", None)
        if rf is not None:
            rf.jammer_pos = np.array([x, y], dtype=float)

    def inject_failures(self, ratio: float, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        idx = np.where(self.env.active)[0]
        k = int(round(ratio * len(idx)))
        if k:
            victims = rng.choice(idx, size=k, replace=False)
            self.env.active[victims] = False

    # ---- control -----------------------------------------------------------
    def compute(self) -> np.ndarray:
        """Advance one control step; return velocity setpoints (n, 2) [m/s]."""
        v = self.ctrl.velocity(self.env, self.t)
        v = np.asarray(v, dtype=float)
        v[~self.env.active] = 0.0
        self.t += 1
        return v


class _BareEnv:
    """Minimal SwarmEnvironment stand-in: positions/active owned externally,
    everything else (distances, adjacency, links) computed on demand."""

    def __init__(self, p: R2CParams, n: int):
        self.p = p
        from r2c.environment import LinkModel
        self.link = LinkModel(p)
        self.n = n
        self.pos = np.zeros((n, 2))
        self.active = np.ones(n, dtype=bool)
        self.t = 0

    def distances(self):
        d = self.pos[:, None, :] - self.pos[None, :, :]
        return np.linalg.norm(d, axis=-1)

    def adjacency(self):
        adj = self.link.adjacency(self.distances())
        adj[~self.active, :] = 0.0
        adj[:, ~self.active] = 0.0
        np.fill_diagonal(adj, 0.0)
        return adj
