"""Relay gz TF/pose info to per-drone PoseStamped on /swarm/pose/{i}.

Subscribes /world/swarm_arena/pose (gz pose vector via ros_gz_bridge) and
republishes each drone's planar position as PoseStamped so SwarmBridge's
ROS node can consume it. Ignores the ground/jammer_marker models.
"""
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from tf2_msgs.msg import TFMessage


class PoseRelay(Node):
    def __init__(self):
        super().__init__("swarm_pose_relay")
        self.declare_parameter("n_drones", 20)
        self.n = int(self.get_parameter("n_drones").value)
        self.sub = self.create_subscription(TFMessage, "/world/swarm_arena/pose",
                                            self.cb, 10)
        self.pubs = {i: self.create_publisher(PoseStamped, f"/swarm/pose/{i}", 10)
                     for i in range(self.n)}

    def cb(self, msg):
        for tr in msg.transforms:
            name = tr.child_frame_id
            if not name.startswith("drone_"):
                continue
            try:
                i = int(name.split("_", 1)[1])
            except ValueError:
                continue
            if i not in self.pubs:
                continue
            out = PoseStamped()
            out.header = tr.header
            out.header.frame_id = "swarm_arena"
            out.pose.position.x = tr.transform.translation.x
            out.pose.position.y = tr.transform.translation.y
            out.pose.orientation.w = 1.0
            self.pubs[i].publish(out)


def main():
    rclpy.init()
    node = PoseRelay()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
