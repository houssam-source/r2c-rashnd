"""ROS 2 node wrapping SwarmBridge for Gazebo or real UAVs.

Topics (per drone i, zero-padded names):
  in:  /swarm/pose/{i}        geometry_msgs/PoseStamped   (any source: gz
                              pose bridge, mocap, GPS, UWB)
  out: /swarm/cmd_vel/{i}     geometry_msgs/Twist         (linear.x/y used)

For PX4/ArduPilot SITL or real drones, remap cmd_vel to
  /uav{i}/mavros/setpoint_velocity/cmd_vel   (OFFBOARD mode)
or run with use_mavros:=true to publish directly there.

Parameters:
  n_drones      (int)   number of drones
  controller    (str)   r2c | hybrid
  jammer_mode   (str)   constant | sweep | reactive
  failure_time  (int)   step to inject failures (-1 = disable)
  failure_ratio (float) fraction to fail at failure_time
  rate_hz       (double) control rate (default 10)

Run:
  ros2 run uav_swarm_r2c bridge_node --ros-args -p n_drones:=20 \
      -p controller:=hybrid -p jammer_mode:=reactive
"""
import os
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node

# locate the r2c package (repo root is 5 levels up from this file)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, _REPO)

from r2c.config import R2CParams          # noqa: E402
from uav_swarm_r2c.bridge import SwarmBridge  # noqa: E402


class BridgeNode(Node):
    def __init__(self):
        super().__init__("uav_swarm_r2c_bridge")
        self.declare_parameter("n_drones", 20)
        self.declare_parameter("controller", "hybrid")
        self.declare_parameter("jammer_mode", "constant")
        self.declare_parameter("failure_time", 100)
        self.declare_parameter("failure_ratio", 0.5)
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("use_mavros", False)

        n = int(self.get_parameter("n_drones").value)
        p = R2CParams()
        self.bridge = SwarmBridge(
            n, p,
            controller=str(self.get_parameter("controller").value),
            jammer_mode=str(self.get_parameter("jammer_mode").value))
        self.p = p
        self.n = n
        self.poses = np.full((n, 2), np.nan)
        self.have_pose = np.zeros(n, dtype=bool)
        self.fail_t = int(self.get_parameter("failure_time").value)
        self.fail_r = float(self.get_parameter("failure_ratio").value)
        self.fail_done = self.fail_t < 0
        self.use_mavros = bool(self.get_parameter("use_mavros").value)

        self.subs = []
        for i in range(n):
            self.subs.append(self.create_subscription(
                PoseStamped, f"/swarm/pose/drone_{i}", self._mk_pose_cb(i), 10))
        self.pubs = []
        for i in range(n):
            topic = (f"/uav{i}/mavros/setpoint_velocity/cmd_vel"
                     if self.use_mavros else f"/swarm/cmd_vel/drone_{i}")
            self.pubs.append(self.create_publisher(Twist, topic, 10))

        rate = float(self.get_parameter("rate_hz").value)
        self.timer = self.create_timer(1.0 / rate, self.step)
        self.get_logger().info(
            f"bridge up: n={n} controller={self.bridge.controller_kind} "
            f"jammer={self.bridge.jammer_mode}")

    def _mk_pose_cb(self, i):
        def cb(msg):
            self.poses[i] = [msg.pose.position.x, msg.pose.position.y]
            self.have_pose[i] = True
        return cb

    def step(self):
        if not self.have_pose.all():
            return  # wait for all poses
        t = self.bridge.t
        if not self.fail_done and t == self.fail_t:
            self.bridge.inject_failures(self.fail_r)
            self.fail_done = True
            self.get_logger().warn(
                f"t={t}: injected {self.fail_r:.0%} failures")
        self.bridge.update_poses(self.poses)
        v = self.bridge.compute()
        for i in range(self.n):
            msg = Twist()
            msg.linear.x = float(v[i, 0])
            msg.linear.y = float(v[i, 1])
            self.pubs[i].publish(msg)


def main():
    rclpy.init()
    node = BridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
