"""Native UR5e/MoveIt, wrist RGB camera and stationary identified target."""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    if os.environ.get("ROS_DOMAIN_ID") != "44" or not os.environ.get("GZ_PARTITION", "").startswith(
        "opencv_hand_eye_"
    ):
        raise RuntimeError("Launch through the isolated example runner")
    overlay = Path(LaunchConfiguration("overlay").perform(context)).resolve(strict=True)
    gui = LaunchConfiguration("gui").perform(context)
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(overlay / "launch/ur_sim_control_windows.launch.py")),
            launch_arguments=dict(
                ur_type="ur5e",
                world_file=str(overlay / "world.sdf"),
                gazebo_gui=gui,
                launch_rviz="false",
                controllers_file=str(overlay / "config/ur5e_gripper_controllers.yaml"),
                initial_joint_controller="joint_trajectory_controller",
            ).items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(overlay / "launch/ur_moveit_windows.launch.py")),
            launch_arguments=dict(
                ur_type="ur5e",
                use_sim_time="true",
                launch_rviz=gui,
                warehouse_sqlite_path=str(overlay / "warehouse.sqlite"),
            ).items(),
        ),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="hand_eye_camera_bridge",
            parameters=[dict(config_file=str(overlay / "bridge.yaml"), use_sim_time=True)],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("overlay"),
            DeclareLaunchArgument("gui", default_value="false"),
            OpaqueFunction(function=setup),
        ]
    )
