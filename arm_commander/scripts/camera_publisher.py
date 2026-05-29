#!/usr/bin/env python3
# NEW FILE — created for ArUco perception pipeline
# Does NOT replace or modify any existing file

from __future__ import annotations

import math
from typing import Dict, Tuple

import cv2
import mujoco
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class MujocoCameraPublisher(Node):
    """Publishes MuJoCo offscreen camera feeds as ROS Image topics."""

    def __init__(self) -> None:
        """
        WHAT: Initializes MuJoCo model, renderers, publishers, and timer loop.
        WHY: MuJoCo does not auto-publish ROS camera topics in this setup.
        INPUT: ROS parameters for model path, publish rate, and camera sizes.
        OUTPUT: Periodic image + camera_info publications and static camera TFs.
        """
        super().__init__("mujoco_camera_publisher")

        self.declare_parameter("mujoco_model_path", "")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("chest_cam_width", 640)
        self.declare_parameter("chest_cam_height", 480)
        self.declare_parameter("wrist_cam_width", 320)
        self.declare_parameter("wrist_cam_height", 240)

        self.model_path = str(self.get_parameter("mujoco_model_path").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        chest_w = int(self.get_parameter("chest_cam_width").value)
        chest_h = int(self.get_parameter("chest_cam_height").value)
        wrist_w = int(self.get_parameter("wrist_cam_width").value)
        wrist_h = int(self.get_parameter("wrist_cam_height").value)

        if not self.model_path:
            raise RuntimeError("Parameter 'mujoco_model_path' is required and cannot be empty.")

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.bridge = CvBridge()

        # key: ROS-facing camera frame/topic prefix, value: (candidate MJCF names, width, height)
        self.camera_cfg: Dict[str, Tuple[Tuple[str, ...], int, int]] = {
            "chest_cam": (("chest_cam",), chest_w, chest_h),
            "right_wrist_cam": (("right_wrist_cam", "camera_wrist_right"), wrist_w, wrist_h),
            "left_wrist_cam": (("left_wrist_cam", "camera_wrist_left"), wrist_w, wrist_h),
        }

        self.cam_source_name: Dict[str, str] = {}
        for ros_cam_name, (candidates, w, h) in self.camera_cfg.items():
            source_name = ""
            for candidate in candidates:
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, candidate)
                if cam_id >= 0:
                    source_name = candidate
                    break
            if not source_name:
                self.get_logger().warn(
                    f"Camera '{ros_cam_name}' not found in model "
                    f"(tried: {', '.join(candidates)}); skipping."
                )
                continue
            self.cam_source_name[ros_cam_name] = source_name

        # Single shared renderer for all cameras to prevent OpenGL context collision and view bleeding.
        # Initialize at chest camera's max resolution (640x480)
        self.renderer = mujoco.Renderer(self.model, height=chest_h, width=chest_w)

        # Publication: camera image data flowing to perception nodes.
        self.img_pubs = {
            "chest_cam": self.create_publisher(Image, "/chest_cam/image_raw", 10),
            "right_wrist_cam": self.create_publisher(Image, "/right_wrist_cam/image_raw", 10),
            "left_wrist_cam": self.create_publisher(Image, "/left_wrist_cam/image_raw", 10),
        }
        # Publication: camera intrinsics flowing to detector and visualization.
        self.info_pubs = {
            "chest_cam": self.create_publisher(CameraInfo, "/chest_cam/camera_info", 10),
            "right_wrist_cam": self.create_publisher(CameraInfo, "/right_wrist_cam/camera_info", 10),
            "left_wrist_cam": self.create_publisher(CameraInfo, "/left_wrist_cam/camera_info", 10),
        }

        self.tf_static_pub = StaticTransformBroadcaster(self)
        self.publish_camera_static_transforms()

        # Subscription to joint states to synchronize the offscreen renderer
        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 10
        )

        self.timer = self.create_timer(1.0 / max(self.publish_rate_hz, 1e-3), self.publish_tick)
        self.get_logger().info("mujoco_camera_publisher started")

    def joint_state_callback(self, msg: JointState) -> None:
        """
        WHAT: Updates offscreen simulation joints using the incoming /joint_states.
        WHY: Keeps the offscreen camera renderer synchronized with the active robot.
        INPUT: JointState message.
        OUTPUT: Updated self.data.qpos.
        """
        for name, position in zip(msg.name, msg.position):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                qpos_addr = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_addr] = position

                # Sync mirror gripper joint if updating finger_joint1
                if "finger_joint1" in name:
                    mirror_name = name.replace("finger_joint1", "finger_joint2")
                    mirror_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, mirror_name)
                    if mirror_id >= 0:
                        mirror_addr = self.model.jnt_qposadr[mirror_id]
                        self.data.qpos[mirror_addr] = position

    def build_camera_info(self, ros_cam_name: str, width: int, height: int) -> CameraInfo:
        """
        WHAT: Builds CameraInfo from MuJoCo camera FoV and image size.
        WHY: Detector needs intrinsic matrix for ArUco pose estimation.
        INPUT: Camera name and current publish dimensions.
        OUTPUT: CameraInfo message with plumb_bob model and zero distortion.
        """
        source_name = self.cam_source_name[ros_cam_name]
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, source_name)
        fovy_deg = float(self.model.cam_fovy[cam_id])
        fy = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
        fx = fy
        cx = width / 2.0
        cy = height / 2.0

        info = CameraInfo()
        info.width = width
        info.height = height
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def publish_camera_static_transforms(self) -> None:
        """
        WHAT: Publishes static TF from parent body frame to each camera frame.
        WHY: Perception pipeline needs camera->world transforms for pose fusion.
        INPUT: Camera definitions from MuJoCo model.
        OUTPUT: /tf_static transforms for available configured cameras.
        """
        transforms = []
        for ros_cam_name in self.camera_cfg:
            if ros_cam_name not in self.cam_source_name:
                continue
            source_name = self.cam_source_name[ros_cam_name]
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, source_name)
            if cam_id < 0:
                continue
            body_id = int(self.model.cam_bodyid[cam_id])
            parent_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            cam_pos = self.model.cam_pos[cam_id]
            cam_quat_wxyz = self.model.cam_quat[cam_id]

            tf_msg = TransformStamped()
            tf_msg.header.stamp = self.get_clock().now().to_msg()
            tf_msg.header.frame_id = parent_name if parent_name else "world"
            tf_msg.child_frame_id = ros_cam_name
            tf_msg.transform.translation.x = float(cam_pos[0])
            tf_msg.transform.translation.y = float(cam_pos[1])
            tf_msg.transform.translation.z = float(cam_pos[2])
            tf_msg.transform.rotation.w = float(cam_quat_wxyz[0])
            tf_msg.transform.rotation.x = float(cam_quat_wxyz[1])
            tf_msg.transform.rotation.y = float(cam_quat_wxyz[2])
            tf_msg.transform.rotation.z = float(cam_quat_wxyz[3])
            transforms.append(tf_msg)

        if transforms:
            self.tf_static_pub.sendTransform(transforms)

    def publish_tick(self) -> None:
        """
        WHAT: Renders each configured camera and publishes image + camera_info.
        WHY: Provides real-time camera streams for ArUco detector.
        INPUT: Timer callback at configured publish rate.
        OUTPUT: /<cam>/image_raw and /<cam>/camera_info publications.
        """
        mujoco.mj_forward(self.model, self.data)
        stamp = self.get_clock().now().to_msg()

        for ros_cam_name in self.camera_cfg:
            if ros_cam_name not in self.cam_source_name:
                continue
            _, width, height = self.camera_cfg[ros_cam_name]
            source_name = self.cam_source_name[ros_cam_name]

            self.renderer.update_scene(self.data, camera=source_name)
            rgb = self.renderer.render()
            rgb = np.flipud(rgb)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Downsample if target dimension is smaller than the 640x480 renderer buffer (e.g. wrist cams 320x240)
            if width != 640 or height != 480:
                bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)

            img_msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
            img_msg.header.stamp = stamp
            img_msg.header.frame_id = ros_cam_name
            self.img_pubs[ros_cam_name].publish(img_msg)

            info_msg = self.build_camera_info(ros_cam_name, width, height)
            info_msg.header.stamp = stamp
            info_msg.header.frame_id = ros_cam_name
            self.info_pubs[ros_cam_name].publish(info_msg)


def main(args=None) -> None:
    """
    WHAT: Entry point for MuJoCo camera bridge node.
    WHY: Starts ROS executor that publishes offscreen-rendered camera feeds.
    INPUT: Optional ROS CLI args.
    OUTPUT: Running bridge process until shutdown.
    """
    rclpy.init(args=args)
    node = MujocoCameraPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
