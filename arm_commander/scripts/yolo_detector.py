#!/usr/bin/env python3
# NEW FILE — OpenCV contour-based cube detector (replaces aruco_detector.py)
# Does NOT modify any existing file.
#
# Detection strategy:
#   1. Convert chest-cam RGB to HSV and threshold for orange (the cube colour).
#   2. Find the largest contour in the mask — that is the cube face.
#   3. Compute the contour centroid (u, v) in pixels.
#   4. Sample median depth in a small patch around (u, v) from the registered
#      depth image → z_c (metres, in camera frame).
#   5. Back-project: x_c = (u - cx) * z_c / fx,  y_c = (v - cy) * z_c / fy.
#   6. Transform (x_c, y_c, z_c) to world frame via TF2.
#   7. Orientation = identity (cube sits flat — no need to detect rotation).

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool
from tf2_ros import TransformException

# ── Tuning constants ───────────────────────────────────────────────────────────

# Orange HSV range — tuned for the MuJoCo cube (rgba 1.0 0.45 0.05 in sRGB)
HSV_ORANGE_LOW  = np.array([8,  120, 100], dtype=np.uint8)
HSV_ORANGE_HIGH = np.array([25, 255, 255], dtype=np.uint8)

# Minimum blob area as fraction of image area — filters out tiny noise
MIN_BLOB_AREA_FRAC = 0.001

# Depth patch radius (pixels) — median over patch for robustness
DEPTH_PATCH_RADIUS = 5

# Valid depth range (metres)
DEPTH_MIN_M = 0.05
DEPTH_MAX_M = 3.0

# EMA smoothing factor for position (0 = frozen, 1 = raw)
EMA_ALPHA = 0.25

# Publish rates
POSE_PUBLISH_RATE_HZ  = 10.0
DEBUG_PUBLISH_RATE_HZ = 5.0
POSE_TIMEOUT_SEC      = 1.0

# Cube half-height (metres) — used to shift the detected top-face centroid
# down to the cube geometric centre for collision object placement
CUBE_HALF_Z = 0.03   # 6 cm cube → 3 cm half-height


class ContourCubeDetector(Node):
    """
    Detects the orange cube using OpenCV HSV contour detection on the chest
    RGB camera, lifts metric depth from the registered depth image, and
    publishes the cube world pose.  Drop-in replacement for aruco_detector.py.
    """

    def __init__(self) -> None:
        """
        WHAT: Initialize contour-based cube detector node.
        WHY:  No external ML library needed — pure OpenCV, zero extra deps.
        INPUT: None (reads ROS params internally).
        OUTPUT: Running node publishing /cube_pose and /cube_detected.
        """
        super().__init__("yolo_detector")   # keep node name for launch compat

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("debug_viz",          True)
        self.declare_parameter("ema_alpha",          EMA_ALPHA)
        self.declare_parameter("depth_patch_radius", DEPTH_PATCH_RADIUS)
        self.declare_parameter("hsv_low",  [8,  120, 100])
        self.declare_parameter("hsv_high", [25, 255, 255])

        self.debug_viz  = bool(self.get_parameter("debug_viz").value)
        self._ema_alpha = float(self.get_parameter("ema_alpha").value)
        self._patch_r   = int(self.get_parameter("depth_patch_radius").value)

        hsv_low_list  = self.get_parameter("hsv_low").value
        hsv_high_list = self.get_parameter("hsv_high").value
        self._hsv_low  = np.array(hsv_low_list,  dtype=np.uint8)
        self._hsv_high = np.array(hsv_high_list, dtype=np.uint8)

        # ── State ─────────────────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.camera_matrix:    Optional[np.ndarray] = None
        self.camera_calibrated = False

        self.last_pose_world:   Optional[PoseStamped] = None
        self.last_seen_time:    Optional[Time]        = None
        self.last_debug_pub_time: Optional[Time]      = None
        self.last_smoothed_pos: Optional[np.ndarray]  = None

        # Wrist camera refined pose (published by wrist servo nodes)
        self.last_wrist_pose:      Optional[PoseStamped] = None
        self.last_wrist_seen_time: Optional[Time]        = None

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            CameraInfo, "/chest_cam/camera_info", self._camera_info_cb, 10)

        # Time-synchronize RGB + depth
        rgb_sub   = Subscriber(self, Image, "/chest_cam/image_raw")
        depth_sub = Subscriber(self, Image, "/chest_cam/depth/image_raw")
        self._sync = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=0.05)
        self._sync.registerCallback(self._image_callback)

        self.create_subscription(
            PoseStamped, "/wrist_cube_pose", self._wrist_pose_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self.cube_pose_pub     = self.create_publisher(PoseStamped, "/cube_pose",         10)
        self.cube_detected_pub = self.create_publisher(Bool,         "/cube_detected",     10)
        self.debug_image_pub   = self.create_publisher(Image,        "/yolo_debug_image",  10)

        # ── Timer ─────────────────────────────────────────────────────────────
        self.create_timer(1.0 / POSE_PUBLISH_RATE_HZ, self._pose_timer_cb)

        self.get_logger().info(
            "contour_cube_detector started — "
            f"HSV [{self._hsv_low.tolist()}] to [{self._hsv_high.tolist()}]"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        """
        WHAT: Stores camera intrinsics.
        WHY:  Back-projection from pixel to 3D requires fx, fy, cx, cy.
        INPUT: CameraInfo from /chest_cam/camera_info.
        OUTPUT: Sets self.camera_matrix and calibrated flag.
        """
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.camera_calibrated = True

    def _image_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        """
        WHAT: Runs contour detection on RGB, samples depth at centroid,
              back-projects to 3D and transforms to world frame.
        WHY:  Core perception loop — generates cube world pose.
        INPUT: Synchronized RGB Image + depth Image (32FC1, metres).
        OUTPUT: Updates last_pose_world; publishes /cube_detected.
        """
        if not self.camera_calibrated or self.camera_matrix is None:
            self._publish_detected(False)
            return

        # ── Decode ────────────────────────────────────────────────────────────
        try:
            bgr   = self.bridge.imgmsg_to_cv2(rgb_msg,   desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().warn(f"Image decode failed: {exc}")
            self._publish_detected(False)
            return

        h, w = bgr.shape[:2]

        # ── Contour detection ─────────────────────────────────────────────────
        detection = self._detect_orange_contour(bgr, h, w)
        if detection is None:
            self._publish_detected(False)
            if self.debug_viz:
                self._maybe_publish_debug(bgr, None, None, rgb_msg.header.frame_id)
            return

        u, v, contour, bbox = detection   # centroid pixel + contour for debug

        # ── Depth sampling ────────────────────────────────────────────────────
        z_c = self._sample_depth(depth, int(u), int(v), h, w)
        if z_c is None:
            self.get_logger().warn(
                f"Depth at centroid ({int(u)},{int(v)}) is invalid — skipping",
                throttle_duration_sec=2.0)
            self._publish_detected(False)
            if self.debug_viz:
                self._maybe_publish_debug(bgr, contour, None, rgb_msg.header.frame_id)
            return

        # ── Back-projection to camera frame ───────────────────────────────────
        fx = float(self.camera_matrix[0, 0])
        fy = float(self.camera_matrix[1, 1])
        cx = float(self.camera_matrix[0, 2])
        cy = float(self.camera_matrix[1, 2])

        x_c = (u - cx) * z_c / fx
        y_c = (v - cy) * z_c / fy
        # z_c is already the forward distance in the OpenCV camera frame

        # ── Transform to world frame ──────────────────────────────────────────
        pose_world = self._transform_to_world(
            rgb_msg.header.frame_id,
            np.array([x_c, y_c, z_c]),
            rgb_msg.header.stamp,
        )
        if pose_world is None:
            self._publish_detected(False)
            return

        self.last_pose_world = pose_world
        self.last_seen_time  = self.get_clock().now()
        self._publish_detected(True)

        if self.debug_viz:
            self._maybe_publish_debug(bgr, contour, z_c, rgb_msg.header.frame_id,
                                       u=u, v=v, bbox=bbox)

    def _wrist_pose_cb(self, msg: PoseStamped) -> None:
        """Store refined wrist-camera pose."""
        self.last_wrist_pose      = msg
        self.last_wrist_seen_time = self.get_clock().now()

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detect_orange_contour(
        self,
        bgr: np.ndarray,
        h: int,
        w: int,
    ) -> Optional[Tuple[float, float, np.ndarray, Tuple]]:
        """
        WHAT: Finds the orange cube face using HSV colour segmentation + contours.
        WHY:  Simple, deterministic, zero-dependency detection for a single
              distinctively-coloured object in a controlled workspace.
        INPUT: BGR image, image height and width.
        OUTPUT: (centroid_u, centroid_v, contour, (x,y,bw,bh)) or None.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_low, self._hsv_high)

        # Morphological clean-up to remove isolated noise pixels
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Pick largest contour (the cube face should dominate the scene)
        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)
        min_area = MIN_BLOB_AREA_FRAC * w * h

        if area < min_area:
            return None   # too small → noise

        # Centroid via image moments
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        u = M["m10"] / M["m00"]
        v = M["m01"] / M["m00"]

        bbox = cv2.boundingRect(largest)  # (x, y, w, h) for debug overlay
        return float(u), float(v), largest, bbox

    # ── Depth ─────────────────────────────────────────────────────────────────

    def _sample_depth(
        self,
        depth: np.ndarray,
        u: int, v: int,
        h: int, w: int,
    ) -> Optional[float]:
        """
        WHAT: Returns median depth (metres) in a patch around (u, v).
        WHY:  Median over a patch is robust to isolated invalid (NaN/0) pixels.
        INPUT: float32 depth image, centroid pixel, image size.
        OUTPUT: Median depth in metres, or None if no valid pixels found.
        """
        r  = self._patch_r
        u0, u1 = max(0, u - r), min(w, u + r + 1)
        v0, v1 = max(0, v - r), min(h, v + r + 1)
        patch = depth[v0:v1, u0:u1].flatten()
        valid = patch[(patch > DEPTH_MIN_M) & (patch < DEPTH_MAX_M) & np.isfinite(patch)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    # ── TF transform ──────────────────────────────────────────────────────────

    def _transform_to_world(
        self,
        camera_frame: str,
        t_cam: np.ndarray,
        stamp,
    ) -> Optional[PoseStamped]:
        """
        WHAT: Transforms a 3D point from camera frame to world frame.
        WHY:  Robot planning needs the cube in world coordinates.
        INPUT: Camera frame id, 3D point [x,y,z] in camera frame, ROS stamp.
        OUTPUT: PoseStamped in world frame, or None on TF failure.

        NOTE: The depth image frame is 'chest_depth_cam', which has the same
              extrinsics as 'chest_cam'. The RGB image header frame_id is
              'chest_cam', so we use whichever is passed in.
        """
        try:
            tf_world_cam = self.tf_buffer.lookup_transform(
                "world", camera_frame, Time(), timeout=Duration(seconds=0.2)
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"TF lookup failed world→{camera_frame}: {exc}",
                throttle_duration_sec=2.0)
            return None

        t_world_cam = np.array([
            tf_world_cam.transform.translation.x,
            tf_world_cam.transform.translation.y,
            tf_world_cam.transform.translation.z,
        ], dtype=np.float64)
        q_world_cam = np.array([
            tf_world_cam.transform.rotation.x,
            tf_world_cam.transform.rotation.y,
            tf_world_cam.transform.rotation.z,
            tf_world_cam.transform.rotation.w,
        ], dtype=np.float64)
        rot_world_cam = Rotation.from_quat(q_world_cam)

        # MuJoCo camera convention: Z-backward, Y-up  vs  OpenCV: Z-forward, Y-down
        # Rotate 180° around X to flip Z and Y axes.
        cv_to_mujoco = Rotation.from_euler('x', 180, degrees=True)
        t_cam_corrected = cv_to_mujoco.apply(t_cam)

        t_world_cube = t_world_cam + rot_world_cam.apply(t_cam_corrected)

        # ── Orientation: identity (cube sits flat on table, yaw unknown/ignored) ──
        # We use the world-frame identity so the cube pose matches the planner's
        # expectation: orientation.w = 1, x = y = z = 0.
        q_identity = np.array([0.0, 0.0, 0.0, 1.0])

        # ── EMA smoothing ─────────────────────────────────────────────────────
        now = self.get_clock().now()
        if self.last_seen_time is not None:
            dt = (now - self.last_seen_time).nanoseconds / 1e9
            if dt > 1.5:
                self.last_smoothed_pos = None  # stale — reset filter

        if self.last_smoothed_pos is None:
            self.last_smoothed_pos = t_world_cube.copy()
        else:
            a = self._ema_alpha
            self.last_smoothed_pos = a * t_world_cube + (1.0 - a) * self.last_smoothed_pos

        pos = self.last_smoothed_pos

        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.header.stamp    = stamp
        pose.pose.position.x = float(pos[0])
        pose.pose.position.y = float(pos[1])
        pose.pose.position.z = float(pos[2])
        pose.pose.orientation.x = float(q_identity[0])
        pose.pose.orientation.y = float(q_identity[1])
        pose.pose.orientation.z = float(q_identity[2])
        pose.pose.orientation.w = float(q_identity[3])
        return pose

    # ── Timer / publishers ────────────────────────────────────────────────────

    def _pose_timer_cb(self) -> None:
        """
        WHAT: Publishes last valid cube pose at 10 Hz.
        WHY:  Decouples publish rate from camera frame rate; enforces staleness timeout.
        INPUT: Timer tick.
        OUTPUT: /cube_pose publication while cube was detected recently.
        """
        now = self.get_clock().now()

        # Prefer wrist camera if its reading is fresh (< 1.5 s)
        if (self.last_wrist_pose is not None
                and self.last_wrist_seen_time is not None
                and (now - self.last_wrist_seen_time).nanoseconds < int(1.5e9)):
            self.last_pose_world = self.last_wrist_pose
            self.last_seen_time  = now
            self.cube_pose_pub.publish(self.last_pose_world)
            self._publish_detected(True)
            return

        if self.last_pose_world is None or self.last_seen_time is None:
            return
        if (now - self.last_seen_time).nanoseconds > int(POSE_TIMEOUT_SEC * 1e9):
            return
        self.cube_pose_pub.publish(self.last_pose_world)

    def _publish_detected(self, detected: bool) -> None:
        msg = Bool()
        msg.data = detected
        self.cube_detected_pub.publish(msg)

    def _maybe_publish_debug(
        self,
        bgr: np.ndarray,
        contour,
        depth_m: Optional[float],
        frame_id: str,
        u: float = 0.0,
        v: float = 0.0,
        bbox: Optional[Tuple] = None,
    ) -> None:
        """
        WHAT: Draws diagnostic overlay on the camera image and publishes at ≤5 Hz.
        WHY:  Visual confirmation that contour detection is working correctly.
        INPUT: BGR frame, detected contour, depth, frame id, centroid, bbox.
        OUTPUT: Annotated Image on /yolo_debug_image at ≤5 Hz.
        """
        now = self.get_clock().now()
        if self.last_debug_pub_time is not None:
            if (now - self.last_debug_pub_time).nanoseconds < int((1.0 / DEBUG_PUBLISH_RATE_HZ) * 1e9):
                return
        self.last_debug_pub_time = now

        debug = bgr.copy()

        if contour is not None:
            # Draw contour outline in orange
            cv2.drawContours(debug, [contour], -1, (0, 165, 255), 2)

            if bbox is not None:
                bx, by, bw, bh = bbox
                # Draw bounding box
                cv2.rectangle(debug, (bx, by), (bx + bw, by + bh), (0, 200, 255), 1)

            # Draw centroid dot
            cv2.circle(debug, (int(u), int(v)), 6, (0, 0, 255), -1)

            # Draw crosshair at centroid
            h, w = debug.shape[:2]
            cv2.line(debug, (int(u) - 15, int(v)), (int(u) + 15, int(v)), (255, 255, 255), 1)
            cv2.line(debug, (int(u), int(v) - 15), (int(u), int(v) + 15), (255, 255, 255), 1)

            # Text label
            label = "CUBE DETECTED"
            if depth_m is not None:
                label += f"   z={depth_m:.3f}m"
            cv2.putText(debug, label,
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2)

            if self.last_smoothed_pos is not None:
                p = self.last_smoothed_pos
                world_txt = f"world: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"
                cv2.putText(debug, world_txt,
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)
        else:
            cv2.putText(debug, "NO DETECTION",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        try:
            msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            msg.header.frame_id = frame_id
            msg.header.stamp    = now.to_msg()
            self.debug_image_pub.publish(msg)
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().warn(f"Debug image publish failed: {exc}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None) -> None:
    """
    WHAT: Entry point for contour-based cube detector node.
    WHY:  Starts ROS executor for markerless detection pipeline.
    INPUT: Optional CLI ROS args.
    OUTPUT: Long-running ROS node until shutdown.
    """
    rclpy.init(args=args)
    node = ContourCubeDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
