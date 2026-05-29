import os
import subprocess

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import OpaqueFunction, TimerAction
from launch_ros.actions import Node
from launch import LaunchContext

def get_param_from_move_group(param_name: str) -> str:
    result = subprocess.run(
        ["ros2", "param", "get", "/move_group", param_name],
        capture_output=True, text=True
    )
    value = result.stdout.strip()
    if "String value is:" in value:
        value = value.split("String value is:")[1].strip()
    return value


def run_pick_and_place(context: LaunchContext):
    urdf = get_param_from_move_group("robot_description")
    srdf = get_param_from_move_group("robot_description_semantic")

    moveit_pkg = get_package_share_directory("openarm_bimanual_moveit_config")
    kinematics_path = os.path.join(
        moveit_pkg, "config", "openarm_v2.0", "kinematics.yaml")
    with open(kinematics_path, "r") as f:
        kinematics = yaml.safe_load(f)

    return [
        Node(
            package="arm_commander",
            executable="pick_and_place",
            name="openarm_commander",
            output="screen",
            parameters=[
                {
                    "robot_description": urdf,
                    "robot_description_semantic": srdf,
                    "robot_description_kinematics": kinematics,
                }
            ],
        )
    ]


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=run_pick_and_place)
    ])
