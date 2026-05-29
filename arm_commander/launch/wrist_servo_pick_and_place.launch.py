#!/usr/bin/env python3
# PERCEPTION PIPELINE — wrist visual servoing
# Part of arm_commander package
# Created for precise pick-and-place with 3-camera system

"""
wrist_servo_pick_and_place.launch.py

Attach-only launch file — assumes the full MuJoCo + MoveIt stack is already
running (started by mujoco_moveit.launch.py from openarm_bringup).

Run order:
  Terminal 1:
    ros2 launch openarm_bringup mujoco_moveit.launch.py

  Terminal 2:
    ros2 launch arm_commander wrist_servo_pick_and_place.launch.py

Nodes started by THIS file only:
    t=0s   aruco_pose_visualizer      — RViz markers for chest-cam detection
    t=5s   wrist_servo_pick_and_place — C++ orchestrator (3-camera sequence)

Everything else (MuJoCo physics, move_group, robot_state_publisher,
controller spawners, camera_publisher, aruco_detector, wrist servo nodes, RViz)
is already running from mujoco_moveit.launch.py — this file does NOT restart them.
"""

import os
import subprocess

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import OpaqueFunction, TimerAction
from launch_ros.actions import Node


def get_param_from_move_group(param_name: str) -> str:
    """
    WHAT: Reads a string parameter from the already-running /move_group node.
    WHY:  The URDF and SRDF are large strings already loaded by move_group;
          reading them here avoids duplicating the xacro processing and ensures
          all nodes share exactly the same robot description.
    INPUT: param_name — ROS 2 parameter name on /move_group.
    OUTPUT: Parameter value as a plain string.
    """
    result = subprocess.run(
        ["ros2", "param", "get", "/move_group", param_name],
        capture_output=True, text=True
    )
    value = result.stdout.strip()
    if "String value is:" in value:
        value = value.split("String value is:")[1].strip()
    return value


def launch_wrist_servo_nodes(context: LaunchContext):
    """
    WHAT: Reads MoveIt params from the running /move_group and spawns only the
          new wrist servo + orchestrator nodes.
    WHY:  Using OpaqueFunction lets us call subprocess at launch time (after
          move_group is already up) rather than at import time.
    INPUT: LaunchContext (unused — params come from live /move_group).
    OUTPUT: List of Node and TimerAction items to add to the launch.
    """
    # ── Read MoveIt params from the already-running move_group ───────────────
    urdf = get_param_from_move_group("robot_description")
    srdf = get_param_from_move_group("robot_description_semantic")

    moveit_pkg = get_package_share_directory("openarm_bimanual_moveit_config")
    kinematics_path = os.path.join(
        moveit_pkg, "config", "openarm_v2.0", "kinematics.yaml")
    with open(kinematics_path, "r") as f:
        kinematics = yaml.safe_load(f)

    # Shared MoveIt parameter dict passed to the C++ commander node
    moveit_params = {
        "robot_description":            urdf,
        "robot_description_semantic":   srdf,
        "robot_description_kinematics": kinematics,
    }

    # ── aruco_pose_visualizer ─────────────────────────────────────────────────
    # Publishes RViz MarkerArray showing the chest-camera cube detection result.
    # Already provided by mujoco_moveit.launch.py in some configurations;
    # safe to run again as it only publishes markers.
    aruco_visualizer = Node(
        package="arm_commander",
        executable="aruco_pose_visualizer.py",
        name="aruco_pose_visualizer",
        output="screen",
    )

    # ── wrist_servo_pick_and_place (orchestrator) ─────────────────────────────
    # C++ node that runs the full 10-phase 3-camera pick-and-place sequence.
    # Delayed 5 s to give the wrist servo nodes time to start their services
    # before the orchestrator tries to call /right_wrist/align.
    wrist_servo_commander = TimerAction(
        period=5.0,
        actions=[
            Node(
                package="arm_commander",
                executable="wrist_servo_pick_and_place",
                name="wrist_servo_pick_and_place",
                output="screen",
                parameters=[moveit_params],
            )
        ],
    )

    return [
        aruco_visualizer,
        wrist_servo_commander,
    ]


def generate_launch_description():
    """
    WHAT: Top-level entry-point called by ros2 launch.
    WHY:  Single OpaqueFunction that attaches the new nodes to the
          already-running mujoco_moveit stack.
    INPUT: None — no launch arguments needed; everything is read from the
           live /move_group node.
    OUTPUT: LaunchDescription with only the new wrist servo + orchestrator nodes.
    """
    return LaunchDescription([
        OpaqueFunction(function=launch_wrist_servo_nodes)
    ])
