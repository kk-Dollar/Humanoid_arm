#!/usr/bin/env python3
# PERCEPTION PIPELINE — wrist visual servoing
# Part of arm_commander package
# Created for precise pick-and-place with 3-camera system

# from __future__ must be the first statement — keep it here before all else
from __future__ import annotations

import os
import sys

# Ensure wrist_servo_base.py (installed alongside this script) is importable.
# ROS 2 runs scripts from the libexec directory but does not add it to sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

"""
left_wrist_servo.py — Left wrist camera visual servoing node.

All logic is in wrist_servo_base.py (WristServoBase / LeftWristServoNode).
This file is the ROS 2 entry-point script installed by CMakeLists.txt.

Topics (left arm):
  SUB  /left_wrist_cam/image_raw    (sensor_msgs/Image)     camera frames
  SUB  /left_wrist_cam/camera_info  (sensor_msgs/CameraInfo) intrinsics
  PUB  /left_wrist/servo_correction (geometry_msgs/Twist)   XY corrections (m)
  PUB  /left_wrist/cube_width_m     (std_msgs/Float64)      measured cube width
  PUB  /left_wrist/aligned          (std_msgs/Bool)         convergence flag
  PUB  /left_wrist/debug_image      (sensor_msgs/Image)     annotated debug frame
  SRV  /left_wrist/align            (std_srvs/Trigger)      blocking align call
"""

import rclpy
from rclpy.executors import MultiThreadedExecutor

# All logic resides in the shared base module
from wrist_servo_base import LeftWristServoNode


def main(args: list | None = None) -> None:
    """
    WHAT: Entry point for the left wrist servo process.
    WHY:  Starts the ROS node and runs the multi-threaded executor so the
          blocking align_callback service and the image_callback timer can
          run concurrently without deadlocking.
    INPUT: Optional CLI arguments (forwarded to rclpy.init).
    OUTPUT: Long-running ROS 2 node until SIGINT / shutdown.
    """
    rclpy.init(args=args)
    node = LeftWristServoNode()

    # MultiThreadedExecutor is required because align_callback blocks in a
    # while-loop while image_callback must continue running to update state.
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
