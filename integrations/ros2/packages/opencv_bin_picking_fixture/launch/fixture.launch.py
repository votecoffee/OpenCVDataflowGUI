"""Launch only an allocated, privately prepared native simulation overlay."""
from pathlib import Path
import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PACKAGE_SOURCE = Path(__file__).resolve().parents[1]
if (PACKAGE_SOURCE / "opencv_bin_picking_fixture").is_dir():
    sys.path.insert(0, str(PACKAGE_SOURCE))


def setup(context):
    from opencv_bin_picking_fixture.node import require_isolation
    from opencv_bin_picking_fixture.scene import load
    require_isolation()
    overlay = Path(LaunchConfiguration("overlay").perform(context)).resolve(strict=True)
    if not (overlay / "preparation.json").is_file(): raise ValueError("Prepare a new fixture overlay first")
    scene = load(overlay / "scene.json")
    gui = LaunchConfiguration("gui").perform(context)
    return [
        SetEnvironmentVariable("ROS_LOG_DIR", str(overlay / "logs")),
        SetEnvironmentVariable("PYTHONPATH", str(PACKAGE_SOURCE) + os.pathsep + os.environ.get("PYTHONPATH", "")),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(overlay / "launch/ur_sim_control_windows.launch.py")),
            launch_arguments={"ur_type": "ur5e", "world_file": str(overlay / "world.sdf"), "gazebo_gui": gui,
                "launch_rviz": "false", "controllers_file": str(overlay / "config/ur5e_gripper_controllers.yaml"),
                "initial_joint_controller": "joint_trajectory_controller"}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(overlay / "launch/ur_moveit_windows.launch.py")),
            launch_arguments={"ur_type": "ur5e", "use_sim_time": "true", "launch_rviz": gui,
                "warehouse_sqlite_path": str(overlay / "warehouse.sqlite")}.items()),
        Node(package="ros_gz_bridge", executable="parameter_bridge", name="opencv_fixture_bridge",
            arguments=[f"/world/{scene['world']}/control@ros_gz_interfaces/srv/ControlWorld"],
            parameters=[{"config_file": str(overlay / "bridge.yaml"), "use_sim_time": True}]),
        ExecuteProcess(cmd=[sys.executable, "-m", "opencv_bin_picking_fixture.node", "--overlay", str(overlay),
            "--simulation-only", "--ros-args", "-p", "use_sim_time:=true"], output="screen"),
    ]


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument("overlay"), DeclareLaunchArgument("gui", default_value="false"),
                              OpaqueFunction(function=setup)])
