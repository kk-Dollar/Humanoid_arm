# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def generate_robot_description(
    context: LaunchContext, description_package, arm_type, use_fake_hardware, mujoco_model_path
):
    description_package_str = context.perform_substitution(description_package)
    arm_type_str = context.perform_substitution(arm_type)
    use_fake_hardware_str = context.perform_substitution(use_fake_hardware)
    mujoco_model_path_str = context.perform_substitution(mujoco_model_path)

    if arm_type_str not in {"openarm_v2.0", "v2.0", "v20", "v2_0", "openarm_v20", "openarm_v2_0"}:
        raise ValueError(
            f"This launch is intended for OpenArm v2.0 bimanual. Got arm_type='{arm_type_str}'."
        )

    xacro_path = os.path.join(
        get_package_share_directory(description_package_str),
        "assets",
        "robot",
        "openarm_v2.0",
        "urdf",
        "openarm_v20.bimanual.mujoco.urdf.xacro",
    )

    return xacro.process_file(
        xacro_path,
        mappings={
            "arm_type": "openarm_v2.0",
            "bimanual": "true",
            "use_fake_hardware": use_fake_hardware_str,
            "ros2_control": "true",
            "mujoco_model_path": mujoco_model_path_str,
        },
    ).toprettyxml(indent="  ")


def robot_nodes_spawner(
    context: LaunchContext,
    description_package,
    arm_type,
    use_fake_hardware,
    mujoco_model_path,
    controllers_file,
):
    robot_description = generate_robot_description(
        context, description_package, arm_type, use_fake_hardware, mujoco_model_path
    )
    controllers_file_str = context.perform_substitution(controllers_file)
    robot_description_param = {"robot_description": robot_description}

    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[robot_description_param],
    )

    # MuJoCo-backed ros2_control_node. This hosts controller_manager and the MuJoCo viewer/physics thread.
    control_node = Node(
        package="mujoco_ros2_control",
        executable="ros2_control_node",
        output="both",
        parameters=[robot_description_param, controllers_file_str],
    )

    return [robot_state_pub_node, control_node]


def moveit_nodes_spawner(context: LaunchContext, use_fake_hardware, mujoco_model_path):
    use_fake_hardware_str = context.perform_substitution(use_fake_hardware)
    mujoco_model_path_str = context.perform_substitution(mujoco_model_path)

    description_pkg_path = get_package_share_directory("openarm_description")
    moveit_pkg_path = get_package_share_directory("openarm_bimanual_moveit_config")

    xacro_path = os.path.join(
        description_pkg_path,
        "assets",
        "robot",
        "openarm_v2.0",
        "urdf",
        "openarm_v20.bimanual.mujoco.urdf.xacro",
    )

    config_dir = "openarm_v2.0"

    moveit_config = (
        MoveItConfigsBuilder("openarm", package_name="openarm_bimanual_moveit_config")
        .robot_description(
            file_path=xacro_path,
            mappings={
                "arm_type": "openarm_v2.0",
                "bimanual": "true",
                "use_fake_hardware": use_fake_hardware_str,
                "ros2_control": "true",
                "mujoco_model_path": mujoco_model_path_str,
            },
        )
        .robot_description_semantic(file_path=f"config/{config_dir}/openarm_bimanual.srdf")
        .robot_description_kinematics(file_path=f"config/{config_dir}/kinematics.yaml")
        .joint_limits(file_path=f"config/{config_dir}/joint_limits.yaml")
        .trajectory_execution(file_path=f"config/{config_dir}/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"], default_planning_pipeline="ompl")
        .to_moveit_configs()
    )

    moveit_params = moveit_config.to_dict()

    pilz_cartesian_limits_path = os.path.join(
        moveit_pkg_path, "config", config_dir, "pilz_cartesian_limits.yaml"
    )
    if os.path.exists(pilz_cartesian_limits_path):
        try:
            import yaml  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            yaml = None

        if yaml is not None:
            with open(pilz_cartesian_limits_path, "r") as f:
                config_data = yaml.safe_load(f)
                if "cartesian_limits" in config_data:
                    moveit_params.setdefault("robot_description_planning", {}).update(config_data)

    rviz_cfg = os.path.join(moveit_pkg_path, "config", config_dir, "moveit.rviz")

    return [
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=[moveit_params],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="log",
            arguments=["-d", rviz_cfg],
            parameters=[moveit_params],
        ),
    ]


def generate_launch_description():
    default_mujoco_model_path = "/home/krishna/humanoid_arm/src/openarm_mujoco/v2/pedestal_pose1_cube.xml"
    declared_arguments = [
        DeclareLaunchArgument("description_package", default_value="openarm_description"),
        DeclareLaunchArgument(
            "arm_type",
            default_value="openarm_v2.0",
            description="OpenArm arm type (v2.0 only for this launch).",
        ),
        DeclareLaunchArgument("use_fake_hardware", default_value="false"),
        DeclareLaunchArgument(
            "mujoco_model_path",
            default_value=default_mujoco_model_path,
            description="Absolute path to the MJCF file for the MuJoCo simulation.",
        ),
        DeclareLaunchArgument("runtime_config_package", default_value="openarm_bringup"),
        DeclareLaunchArgument(
            "controllers_file",
            default_value="openarm_bimanual_moveit_controllers.yaml",
        ),
    ]

    description_package = LaunchConfiguration("description_package")
    arm_type = LaunchConfiguration("arm_type")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    mujoco_model_path = LaunchConfiguration("mujoco_model_path")
    runtime_config_package = LaunchConfiguration("runtime_config_package")
    controllers_file = LaunchConfiguration("controllers_file")

    controllers_file = PathJoinSubstitution(
        [FindPackageShare(runtime_config_package), "config", "controllers", controllers_file]
    )

    robot_nodes_spawner_func = OpaqueFunction(
        function=robot_nodes_spawner,
        args=[
            description_package,
            arm_type,
            use_fake_hardware,
            mujoco_model_path,
            controllers_file,
        ],
    )

    moveit_nodes_func = OpaqueFunction(function=moveit_nodes_spawner, args=[use_fake_hardware, mujoco_model_path])

    jsb_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    trajectory_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "left_joint_trajectory_controller",
            "right_joint_trajectory_controller",
            "-c",
            "/controller_manager",
        ],
    )

    gripper_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_gripper_controller", "right_gripper_controller", "-c", "/controller_manager"],
    )

    camera_publisher = Node(
        package="arm_commander",
        executable="camera_publisher.py",
        name="mujoco_camera_publisher",
        output="screen",
        parameters=[
            {
                "mujoco_model_path": mujoco_model_path,
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

    right_wrist_servo = Node(
        package="arm_commander",
        executable="right_wrist_servo.py",
        name="right_wrist_servo",
        output="screen",
        parameters=[{
            "pixel_threshold":    8.0,
            "servo_gain":         0.003,
            "use_color_fallback": True,
            "marker_size_m":      0.05,
            "marker_id":          0,
        }],
    )

    left_wrist_servo = Node(
        package="arm_commander",
        executable="left_wrist_servo.py",
        name="left_wrist_servo",
        output="screen",
        parameters=[{
            "pixel_threshold":    8.0,
            "servo_gain":         0.003,
            "use_color_fallback": True,
            "marker_size_m":      0.05,
            "marker_id":          0,
        }],
    )

    return LaunchDescription(
        declared_arguments
        + [
            robot_nodes_spawner_func,
            moveit_nodes_func,
            camera_publisher,
            TimerAction(period=4.0, actions=[aruco_detector]),
            TimerAction(period=4.0, actions=[right_wrist_servo]),
            TimerAction(period=4.0, actions=[left_wrist_servo]),
            TimerAction(period=2.0, actions=[jsb_spawner]),
            TimerAction(period=3.0, actions=[trajectory_spawner]),
            TimerAction(period=3.0, actions=[gripper_spawner]),
        ]
    )

