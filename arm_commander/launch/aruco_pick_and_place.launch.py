# NEW FILE — created for ArUco perception pipeline
# Does NOT replace or modify any existing file
#
# Install Python dependencies:
#   pip install opencv-contrib-python scipy --break-system-packages
#
# Build:
#   cd ~/humanoid_arm
#   colcon build --packages-select arm_commander
#   source install/setup.bash
#
# Run (MuJoCo must already be running with your scene XML):
#   ros2 launch arm_commander aruco_pick_and_place.launch.py \
#     mujoco_model:=/path/to/your/openarm_bimanual_aruco.xml
#
# Verify cameras are publishing:
#   ros2 topic list | grep cam
#
# Verify cube is detected:
#   ros2 topic echo /cube_detected
#
# Verify cube pose:
#   ros2 topic echo /cube_pose
#
# View debug image in RViz or:
#   ros2 run rqt_image_view rqt_image_view /aruco_debug_image

import os

import xacro
from ament_index_python.packages import get_package_share_directory
import yaml
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context: LaunchContext, arm_type, mujoco_model):
    arm_type_str = context.perform_substitution(arm_type)
    mujoco_model_str = context.perform_substitution(mujoco_model)

    description_pkg_path = get_package_share_directory("openarm_description")
    robot_xacro_path = os.path.join(
        description_pkg_path,
        "assets",
        "robot",
        "openarm_v2.0",
        "urdf",
        "openarm_v20.bimanual.mujoco.urdf.xacro",
    )

    robot_description_xml = xacro.process_file(
        robot_xacro_path,
        mappings={
            "arm_type": arm_type_str,
            "bimanual": "true",
            "use_fake_hardware": "true",
            "ros2_control": "true",
            "mujoco_model_path": mujoco_model_str,
        },
    ).toxml()
    robot_description_param = {"robot_description": robot_description_xml}

    moveit_pkg_path = get_package_share_directory("openarm_bimanual_moveit_config")
    config_dir = os.path.join(moveit_pkg_path, "config", arm_type_str)

    with open(os.path.join(config_dir, "openarm_bimanual.srdf"), "r", encoding="utf-8") as f:
        robot_semantic_xml = f.read()
    with open(os.path.join(config_dir, "kinematics.yaml"), "r", encoding="utf-8") as f:
        robot_kinematics = yaml.safe_load(f)
    with open(os.path.join(config_dir, "joint_limits.yaml"), "r", encoding="utf-8") as f:
        joint_limits = yaml.safe_load(f)
    with open(os.path.join(config_dir, "moveit_controllers.yaml"), "r", encoding="utf-8") as f:
        moveit_controllers = yaml.safe_load(f)

    moveit_params = {
        "robot_description_semantic": robot_semantic_xml,
        "robot_description_kinematics": robot_kinematics,
        "robot_description_planning": joint_limits.get("robot_description_planning", {}),
        "moveit_controller_manager": moveit_controllers.get("moveit_controller_manager"),
        "moveit_simple_controller_manager": moveit_controllers.get("moveit_simple_controller_manager", {}),
        "moveit_manage_controllers": moveit_controllers.get("moveit_manage_controllers", True),
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
    }

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[robot_description_param],
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description_param,
            moveit_params,
            {
                "publish_monitored_planning_scene": True,
                "publish_planning_scene": True,
                "publish_geometry_updates": True,
                "publish_state_updates": True,
            },
        ],
    )

    camera_publisher = Node(
        package="arm_commander",
        executable="camera_publisher.py",
        name="mujoco_camera_publisher",
        output="screen",
        parameters=[
            {
                "mujoco_model_path": mujoco_model_str,
                "publish_rate_hz": 30.0,
            }
        ],
    )

    aruco_detector = Node(
        package="arm_commander",
        executable="aruco_detector.py",
        name="aruco_detector",
        output="screen",
    )

    aruco_visualizer = Node(
        package="arm_commander",
        executable="aruco_pose_visualizer.py",
        name="aruco_pose_visualizer",
        output="screen",
    )

    aruco_pick_and_place = Node(
        package="arm_commander",
        executable="aruco_pick_and_place",
        name="aruco_pick_and_place",
        output="screen",
        parameters=[
            robot_description_param,
            {
                "robot_description_semantic": robot_semantic_xml,
                "robot_description_kinematics": robot_kinematics,
            },
        ],
    )

    return [
        robot_state_publisher,
        move_group,
        TimerAction(period=2.0, actions=[camera_publisher]),
        TimerAction(period=4.0, actions=[aruco_detector]),
        TimerAction(period=4.0, actions=[aruco_visualizer]),
        TimerAction(period=8.0, actions=[aruco_pick_and_place]),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("arm_type", default_value="openarm_v2.0"),
            DeclareLaunchArgument(
                "mujoco_model",
                default_value="/home/krishna/humanoid_arm/src/openarm_mujoco/v2/pedestal_pose1_cube.xml",
            ),
            OpaqueFunction(
                function=launch_setup,
                args=[LaunchConfiguration("arm_type"), LaunchConfiguration("mujoco_model")],
            ),
        ]
    )
