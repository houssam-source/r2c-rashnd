"""ROS 2 launch: gz sim + pose/cmd bridges + swarm bridge node."""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    n = LaunchConfiguration("n_drones")
    controller = LaunchConfiguration("controller")
    jammer = LaunchConfiguration("jammer_mode")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    return LaunchDescription([
        DeclareLaunchArgument("n_drones", default_value="20"),
        DeclareLaunchArgument("controller", default_value="hybrid"),
        DeclareLaunchArgument("jammer_mode", default_value="constant"),

        # Gazebo Sim (headless-friendly; add -g for GUI)
        ExecuteProcess(
            cmd=["gz", "sim", "-r", os.path.join(here, "worlds",
                                                 "swarm_arena.sdf")],
            output="screen"),

        # gz <-> ROS pose bridge (publishes /world/swarm_arena/pose info;
        # a small relay node can repack per-drone PoseStamped on /swarm/pose/i)
        ExecuteProcess(
            cmd=["ros2", "run", "ros_gz_bridge", "parameter_bridge",
                 "/world/swarm_arena/pose@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V"],
            output="screen"),

        # the swarm controller bridge
        Node(
            package="uav_swarm_r2c",
            executable="bridge_node",
            parameters=[{
                "n_drones": n,
                "controller": controller,
                "jammer_mode": jammer,
                "failure_time": 100,
                "failure_ratio": 0.5,
                "rate_hz": 10.0,
            }],
            output="screen"),
    ])
