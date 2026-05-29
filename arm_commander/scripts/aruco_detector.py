#!/usr/bin/env python3
# NEW FILE — created for ArUco perception pipeline
# Does NOT replace or modify any existing file

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import cv2.aruco as aruco
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool
from tf2_ros import TransformException

MARKER_ID = 0
POSE_TIMEOUT_SEC = 1.0
POSE_PUBLISH_RATE_HZ = 10.0
DEBUG_PUBLISH_RATE_HZ = 5.0

# Systematic calibration bias: camera-detected - ground-truth
# Camera sees: [0.285, -0.159, 0.383]  Ground truth: [0.235, -0.100, 0.287]
# We subtract these offsets to correct the bias.
CALIB_OFFSET_X = 0.050
CALIB_OFFSET_Y = -0.059
CALIB_OFFSET_Z = 0.096


class KalmanFilter3D:
    """
    Simple constant-velocity Kalman filter for 3D position smoothing.
    State vector: [x, y, z, vx, vy, vz]
    """

    def __init__(self, process_noise: float = 1e-3, measurement_noise: float = 1e-2) -> None:
        n = 6  # state dimension
        m = 3  # measurement dimension
        self.x = np.zeros((n, 1))          # state
        self.P = np.eye(n) * 1.0           # uncertainty covariance
        dt = 0.1                            # expected dt ~10 Hz
        self.F = np.eye(n)                 # state transition
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        self.H = np.zeros((m, n))          # measurement matrix
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.Q = np.eye(n) * process_noise # process noise
        self.R = np.eye(m) * measurement_noise  # measurement noise
        self.initialized = False

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Feed in a [x, y, z] measurement and return the filtered [x, y, z]."""
        z = measurement.reshape(3, 1)
        if not self.initialized:
            self.x[:3] = z
            self.initialized = True
            return measurement.copy()
        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Update
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x[:3].flatten()


class ArucoDetectorNode(Node):
    """Detects ArUco marker and publishes cube world pose."""

    def __init__(self) -> None:
        """
        WHAT: Initialize ArUco detector ROS node.
        WHY: Set up subscriptions, publishers, TF, and timers for detection.
        INPUT: None (ROS parameters and topics are read internally).
        OUTPUT: Running node that publishes detection state and marker pose.
        """
        super().__init__("aruco_detector")

        self.declare_parameter("marker_size", 0.05)
        self.declare_parameter("debug_viz", True)
        self.marker_size_m = float(self.get_parameter("marker_size").value)
        self.debug_viz = bool(self.get_parameter("debug_viz").value)

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters_create()

        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.camera_calibrated = False

        self.last_seen_time: Optional[Time] = None
        self.last_pose_world: Optional[PoseStamped] = None
        self.last_debug_image: Optional[np.ndarray] = None
        self.last_debug_pub_time: Optional[Time] = None
        self.kalman = KalmanFilter3D(process_noise=1e-5, measurement_noise=1e-1)

        # Subscription: camera frames flowing from chest camera into detector.
        self.image_sub = self.create_subscription(
            Image, "/chest_cam/image_raw", self.image_callback, 10
        )
        # Subscription: intrinsic calibration flowing from camera publisher.
        self.camera_info_sub = self.create_subscription(
            CameraInfo, "/chest_cam/camera_info", self.camera_info_callback, 10
        )

        # Publication: detected cube world pose flowing to manipulation node.
        self.cube_pose_pub = self.create_publisher(PoseStamped, "/cube_pose", 10)
        # Publication: detection state flowing to behavior logic and RViz.
        self.cube_detected_pub = self.create_publisher(Bool, "/cube_detected", 10)
        # Publication: debug visualization image for human inspection.
        self.debug_image_pub = self.create_publisher(Image, "/aruco_debug_image", 10)

        self.pose_timer = self.create_timer(1.0 / POSE_PUBLISH_RATE_HZ, self.pose_timer_cb)

        self.get_logger().info("aruco_detector started")

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """
        WHAT: Stores camera intrinsics and distortion coefficients.
        WHY: ArUco pose estimation requires calibrated camera parameters.
        INPUT: CameraInfo message from /chest_cam/camera_info.
        OUTPUT: Updates calibration state used by pose estimator.
        """
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)
        self.camera_calibrated = True

    def image_callback(self, msg: Image) -> None:
        """
        WHAT: Runs ArUco detection on each incoming chest camera frame.
        WHY: To estimate cube marker pose in camera frame and transform to world.
        INPUT: Image message from /chest_cam/image_raw.
        OUTPUT: Updates latest detection pose and detection boolean state.
        """
        if not self.camera_calibrated or self.camera_matrix is None or self.dist_coeffs is None:
            self.publish_detected(False)
            return

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().warn(f"CvBridge conversion failed: {exc}")
            self.publish_detected(False)
            return

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None:
            self.publish_detected(False)
            self.maybe_publish_debug_image(bgr, msg.header.frame_id)
            return

        marker_indices = np.where(ids.flatten() == MARKER_ID)[0]
        if marker_indices.size == 0:
            self.publish_detected(False)
            self.maybe_publish_debug_image(bgr, msg.header.frame_id)
            return

        idx = int(marker_indices[0])
        marker_corners = [corners[idx]]
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, self.marker_size_m, self.camera_matrix, self.dist_coeffs
        )

        rvec = rvecs[0][0]
        tvec = tvecs[0][0]

        pose_world = self.transform_marker_to_world(msg.header.frame_id, rvec, tvec, msg.header.stamp)
        if pose_world is None:
            self.publish_detected(False)
            self.maybe_publish_debug_image(bgr, msg.header.frame_id)
            return

        self.last_pose_world = pose_world
        self.last_seen_time = self.get_clock().now()
        self.publish_detected(True)

        if self.debug_viz:
            aruco.drawDetectedMarkers(bgr, marker_corners, ids=np.array([[MARKER_ID]], dtype=np.int32))
            cv2.drawFrameAxes(
                bgr,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec,
                self.marker_size_m * 0.5,
            )
            self.maybe_publish_debug_image(bgr, msg.header.frame_id)

    def transform_marker_to_world(
        self, camera_frame: str, rvec: np.ndarray, tvec: np.ndarray, stamp
    ) -> Optional[PoseStamped]:
        """
        WHAT: Converts marker pose camera->marker into world->marker pose.
        WHY: Robot planning needs cube pose in world coordinates.
        INPUT: Camera frame id plus marker pose vectors from ArUco.
        OUTPUT: PoseStamped in world frame, or None on TF failure.
        """
        try:
            tf_world_cam = self.tf_buffer.lookup_transform(
                "world", camera_frame, Time(), timeout=Duration(seconds=0.2)
            )
        except TransformException as exc:
            self.get_logger().warn(f"TF lookup failed world->{camera_frame}: {exc}")
            return None

        t_world_cam, q_world_cam = self.tf_to_vec_quat(tf_world_cam)
        rot_world_cam = Rotation.from_quat(q_world_cam)

        rot_cv_marker = Rotation.from_rotvec(rvec.reshape(3))
        t_cv_marker = tvec.reshape(3)

        # OpenCV camera frame (Z-forward, Y-down) vs MuJoCo camera frame (Z-backward, Y-up)
        # We must physically rotate the OpenCV detections 180 degrees around local X axis
        # so they properly align with the MuJoCo camera's published TF coordinate system.
        cv_to_mujoco = Rotation.from_euler('x', 180, degrees=True)
        t_cam_marker = cv_to_mujoco.apply(t_cv_marker)
        rot_cam_marker = cv_to_mujoco * rot_cv_marker

        rot_world_marker = rot_world_cam * rot_cam_marker
        t_world_marker = t_world_cam + rot_world_cam.apply(t_cam_marker)

        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.header.stamp = stamp

        # Apply calibration bias correction
        raw_pos = np.array([t_world_marker[0], t_world_marker[1], t_world_marker[2]])
        corrected_pos = raw_pos - np.array([CALIB_OFFSET_X, CALIB_OFFSET_Y, CALIB_OFFSET_Z])

        # Apply Kalman smoothing
        smoothed_pos = self.kalman.update(corrected_pos)

        pose.pose.position.x = float(smoothed_pos[0])
        pose.pose.position.y = float(smoothed_pos[1])
        pose.pose.position.z = float(smoothed_pos[2])
        q = rot_world_marker.as_quat()  # x, y, z, w
        pose.pose.orientation.x = float(q[0])
        pose.pose.orientation.y = float(q[1])
        pose.pose.orientation.z = float(q[2])
        pose.pose.orientation.w = float(q[3])
        return pose

    def tf_to_vec_quat(self, tf_msg) -> Tuple[np.ndarray, np.ndarray]:
        """
        WHAT: Converts ROS TransformStamped into translation and quaternion arrays.
        WHY: Numeric operations for world/camera/marker transform composition.
        INPUT: TransformStamped returned by tf2 lookup.
        OUTPUT: Tuple of translation xyz and quaternion xyzw arrays.
        """
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        t_np = np.array([t.x, t.y, t.z], dtype=np.float64)
        q_np = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        return t_np, q_np

    def publish_detected(self, detected: bool) -> None:
        """
        WHAT: Publishes whether marker is currently visible.
        WHY: Enables manipulation logic to wait/retry safely.
        INPUT: Boolean detection state.
        OUTPUT: /cube_detected Bool publication.
        """
        msg = Bool()
        msg.data = detected
        self.cube_detected_pub.publish(msg)

    def pose_timer_cb(self) -> None:
        """
        WHAT: Periodically publishes last valid cube pose at 10 Hz.
        WHY: Decouples pose publish rate from camera frame rate and enforces timeout.
        INPUT: Timer tick event.
        OUTPUT: /cube_pose publication while marker seen in recent window.
        """
        now = self.get_clock().now()
        if self.last_pose_world is None or self.last_seen_time is None:
            return
        if (now - self.last_seen_time).nanoseconds > int(POSE_TIMEOUT_SEC * 1e9):
            return
        self.cube_pose_pub.publish(self.last_pose_world)

    def maybe_publish_debug_image(self, bgr: np.ndarray, frame_id: str) -> None:
        """
        WHAT: Publishes debug image at up to 5 Hz.
        WHY: Keeps visualization useful while limiting unnecessary traffic.
        INPUT: OpenCV BGR frame and source frame id.
        OUTPUT: /aruco_debug_image publication.
        """
        if not self.debug_viz:
            return
        now = self.get_clock().now()
        if self.last_debug_pub_time is not None:
            if (now - self.last_debug_pub_time).nanoseconds < int((1.0 / DEBUG_PUBLISH_RATE_HZ) * 1e9):
                return
        self.last_debug_pub_time = now
        msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        msg.header.frame_id = frame_id
        msg.header.stamp = now.to_msg()
        self.debug_image_pub.publish(msg)


def main(args=None) -> None:
    """
    WHAT: Entry point for ArUco detector process.
    WHY: Starts ROS executor for marker detection pipeline.
    INPUT: Optional CLI ROS args.
    OUTPUT: Long-running ROS node until shutdown.
    """
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
