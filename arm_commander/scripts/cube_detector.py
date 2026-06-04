#!/usr/bin/env python3
"""
cube_detector.py — Markerless cube detection node for the humanoid arm project.

Detection pipeline:
  1. Load the MuJoCo model at startup and read the cube's RGBA colour directly
     from the model geometry — no hardcoded colour params needed.
  2. Convert that RGBA to an HSV range (with tolerance) automatically.
  3. On each frame-pair (RGB + depth, time-synchronised):
       a. Threshold the RGB image with the auto-derived HSV range.
       b. Find the largest contour — that is the cube face.
       c. Compute the contour centroid (u, v) in pixels.
       d. Sample median depth at (u, v) from the registered depth image.
       e. Back-project: x = (u-cx)*z/fx,  y = (v-cy)*z/fy.
       f. Transform (x, y, z) to world frame via TF2.
       g. Apply EMA smoothing and publish /cube_pose.

Orientation: always identity (cube sits flat on table, yaw not needed).

ROS interfaces — identical to the old aruco_detector.py, C++ orchestrator unchanged:
  Subscribe:  /chest_cam/image_raw        (sensor_msgs/Image, BGR)
              /chest_cam/depth/image_raw  (sensor_msgs/Image, 32FC1, metres)
              /chest_cam/camera_info      (sensor_msgs/CameraInfo)
              /wrist_cube_pose            (geometry_msgs/PoseStamped)
  Publish:    /cube_pose                  (geometry_msgs/PoseStamped, world frame)
              /cube_detected              (std_msgs/Bool)
              /cube_debug_image           (sensor_msgs/Image, annotated RGB)
"""

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

# ── Fallback HSV range (orange) used when model colour cannot be read ──────────
# OpenCV HSV: H 0-179, S 0-255, V 0-255
_FALLBACK_HSV_LOW  = np.array([5,  100, 80],  dtype=np.uint8)
_FALLBACK_HSV_HIGH = np.array([25, 255, 255], dtype=np.uint8)

# HSV tolerances used when deriving range from model RGBA
_H_TOL = 12   # ± hue units (OpenCV scale, i.e. ±24° in real degrees)
_S_MIN = 80   # minimum saturation (filters greys/whites)
_V_MIN = 60   # minimum value     (filters blacks/shadows)

# Depth sampling
_DEPTH_PATCH_RADIUS = 5    # pixels around centroid for median depth
_DEPTH_MIN_M        = 0.05
_DEPTH_MAX_M        = 3.0

# Minimum blob area (fraction of image) to accept as cube, not noise
_MIN_BLOB_FRAC = 0.001

# EMA smoothing — higher α = faster tracking, more noise
_EMA_ALPHA = 0.25

# Publish rates
_POSE_HZ    = 10.0
_DEBUG_HZ   = 5.0
_TIMEOUT_S  = 1.0

# MuJoCo geometry name of the cube in the scene
_CUBE_GEOM_NAME = "test_cube_geom"


def rgba_to_hsv_range(
    rgba: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    WHAT: Converts a MuJoCo RGBA colour to an OpenCV HSV detection range.
    WHY:  Lets the detector adapt to any cube colour without hardcoded params.
    INPUT: RGBA array [r, g, b, a] with values in [0, 1].
    OUTPUT: (hsv_low, hsv_high) — uint8 arrays for cv2.inRange.
    """
    r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    # Build a 1x1 BGR pixel and convert to HSV
    bgr_px = np.array([[[b, g, r]]], dtype=np.uint8)
    hsv_px = cv2.cvtColor(bgr_px, cv2.COLOR_BGR2HSV)
    h, s, v = int(hsv_px[0, 0, 0]), int(hsv_px[0, 0, 1]), int(hsv_px[0, 0, 2])

    h_lo = max(0,   h - _H_TOL)
    h_hi = min(179, h + _H_TOL)
    s_lo = max(0,   s - 60)
    v_lo = max(0,   v - 80)

    # Enforce absolute floor so near-white surfaces are not matched
    s_lo = max(s_lo, _S_MIN)
    v_lo = max(v_lo, _V_MIN)

    hsv_low  = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    hsv_high = np.array([h_hi, 255,  255],  dtype=np.uint8)
    return hsv_low, hsv_high


def read_cube_color_from_model(model_path: str) -> Optional[np.ndarray]:
    """
    WHAT: Loads the MuJoCo model and returns the cube geom's RGBA.
    WHY:  Single source of truth for cube colour — comes straight from the scene.
    INPUT: Absolute path to the .xml model file.
    OUTPUT: RGBA array [r, g, b, a] or None on failure.
    """
    try:
        import mujoco  # type: ignore[import-untyped]
        model = mujoco.MjModel.from_xml_path(model_path)
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, _CUBE_GEOM_NAME)
        if geom_id < 0:
            return None
        return np.array(model.geom_rgba[geom_id], dtype=np.float32)
    except Exception:  # pylint: disable=broad-except
        return None


class CubeDetector(Node):
    """
    Detects the cube in the chest camera feed using depth-registered contour
    detection and publishes a world-frame pose.
    """

    def __init__(self) -> None:
        """
        WHAT: Initialise cube detector node.
        WHY:  Sets up auto-colour derivation, subscribers, publishers, TF.
        INPUT: None — reads ROS params internally.
        OUTPUT: Running node publishing /cube_pose and /cube_detected.
        """
        super().__init__("cube_detector")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("mujoco_model_path", "")
        self.declare_parameter("debug_viz",          True)
        self.declare_parameter("ema_alpha",          _EMA_ALPHA)
        self.declare_parameter("depth_patch_radius", _DEPTH_PATCH_RADIUS)

        model_path   = str(self.get_parameter("mujoco_model_path").value)
        self.debug_viz  = bool(self.get_parameter("debug_viz").value)
        self._ema_alpha = float(self.get_parameter("ema_alpha").value)
        self._patch_r   = int(self.get_parameter("depth_patch_radius").value)

        # ── Auto-derive cube colour from MuJoCo model ─────────────────────────
        self._hsv_low, self._hsv_high = _FALLBACK_HSV_LOW, _FALLBACK_HSV_HIGH
        if model_path:
            rgba = read_cube_color_from_model(model_path)
            if rgba is not None:
                self._hsv_low, self._hsv_high = rgba_to_hsv_range(rgba)
                self.get_logger().info(
                    f"Cube colour read from model: RGBA {rgba[:3].tolist()}  "
                    f"→  HSV range {self._hsv_low.tolist()} – {self._hsv_high.tolist()}"
                )
            else:
                self.get_logger().warn(
                    f"Could not find '{_CUBE_GEOM_NAME}' in model; "
                    "using fallback HSV orange range."
                )
        else:
            self.get_logger().warn(
                "mujoco_model_path not set — using fallback HSV orange range."
            )

        # ── State ─────────────────────────────────────────────────────────────
        self.bridge          = CvBridge()
        self.tf_buffer       = tf2_ros.Buffer()
        self.tf_listener     = tf2_ros.TransformListener(self.tf_buffer, self)
        self.camera_matrix:    Optional[np.ndarray]  = None
        self.camera_calibrated = False
        self.last_pose_world:   Optional[PoseStamped] = None
        self.last_seen_time:    Optional[Time]        = None
        self.last_debug_pub:    Optional[Time]        = None
        self.last_smoothed_pos: Optional[np.ndarray]  = None
        self.last_wrist_pose:   Optional[PoseStamped] = None
        self.last_wrist_time:   Optional[Time]        = None

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            CameraInfo, "/chest_cam/camera_info", self._camera_info_cb, 10)

        rgb_sub   = Subscriber(self, Image, "/chest_cam/image_raw")
        depth_sub = Subscriber(self, Image, "/chest_cam/depth/image_raw")
        self._sync = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=0.05)
        self._sync.registerCallback(self._frame_callback)

        self.create_subscription(
            PoseStamped, "/wrist_cube_pose", self._wrist_pose_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self.cube_pose_pub     = self.create_publisher(PoseStamped, "/cube_pose",        10)
        self.cube_detected_pub = self.create_publisher(Bool,         "/cube_detected",    10)
        self.debug_image_pub   = self.create_publisher(Image,        "/cube_debug_image", 10)

        # ── Pose publish timer ────────────────────────────────────────────────
        self.create_timer(1.0 / _POSE_HZ, self._pose_timer_cb)

        self.get_logger().info("cube_detector ready")

    # ── Camera intrinsics ─────────────────────────────────────────────────────

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        """Store camera intrinsic matrix from /chest_cam/camera_info."""
        if msg.k[0] < 10.0:
            return  # Ignore bogus normalized camera_info from mujoco_ros2_control
        self.camera_matrix     = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.camera_calibrated = True

    # ── Main perception callback ──────────────────────────────────────────────

    def _frame_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        """
        WHAT: Uses point cloud of color mask to find geometric center of top face.
        """
        if not self.camera_calibrated or self.camera_matrix is None:
            self._pub_detected(False)
            return

        try:
            bgr   = self.bridge.imgmsg_to_cv2(rgb_msg,   desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        except Exception as exc:
            self.get_logger().warn(f"Image decode failed: {exc}")
            self._pub_detected(False)
            return

        h, w = bgr.shape[:2]
        
        # Color mask
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_low, self._hsv_high)
        k    = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        v_idx, u_idx = np.where(mask > 0)
        if len(v_idx) < _MIN_BLOB_FRAC * w * h:
            self._pub_detected(False)
            if self.debug_viz:
                self._pub_debug(bgr, None, None, rgb_msg.header.frame_id)
            return

        z_c = depth[v_idx, u_idx]
        valid = (z_c > _DEPTH_MIN_M) & (z_c < _DEPTH_MAX_M) & np.isfinite(z_c)
        v_idx, u_idx, z_c = v_idx[valid], u_idx[valid], z_c[valid]

        if len(z_c) == 0:
            self._pub_detected(False)
            return

        # 3D back-projection
        fx = float(self.camera_matrix[0, 0])
        fy = float(self.camera_matrix[1, 1])
        cx = float(self.camera_matrix[0, 2])
        cy = float(self.camera_matrix[1, 2])
        
        x_c = (u_idx - cx) * z_c / fx
        y_c = (v_idx - cy) * z_c / fy
        pts_cam = np.vstack((x_c, y_c, z_c))

        # TF to world
        camera_frame = rgb_msg.header.frame_id
        try:
            tf_wc = self.tf_buffer.lookup_transform(
                "world", camera_frame, rgb_msg.header.stamp, timeout=Duration(seconds=0.2))
        except TransformException as exc:
            self.get_logger().warn(f"TF world←{camera_frame}: {exc}", throttle_duration_sec=2.0)
            return

        t_wc = np.array([tf_wc.transform.translation.x, tf_wc.transform.translation.y, tf_wc.transform.translation.z])
        q_wc = np.array([tf_wc.transform.rotation.x, tf_wc.transform.rotation.y, tf_wc.transform.rotation.z, tf_wc.transform.rotation.w])
        rot_wc = Rotation.from_quat(q_wc)
        flip = Rotation.from_euler('x', 180, degrees=True)
        
        pts_world = rot_wc.apply(flip.apply(pts_cam.T)).T + t_wc[:, np.newaxis]
        
        # Isolate top face (highest Z)
        z_world = pts_world[2, :]
        max_z = np.max(z_world)
        top_face_mask = z_world >= (max_z - 0.015) # Top 1.5 cm
        
        if np.sum(top_face_mask) == 0:
            self._pub_detected(False)
            return
            
        mean_x = float(np.mean(pts_world[0, top_face_mask]))
        mean_y = float(np.mean(pts_world[1, top_face_mask]))
        mean_z = float(np.mean(pts_world[2, top_face_mask]))
        t_cube = np.array([mean_x, mean_y, mean_z])

        # EMA smoothing
        now = self.get_clock().now()
        if (self.last_seen_time is not None
                and (now - self.last_seen_time).nanoseconds / 1e9 > 1.5):
            self.last_smoothed_pos = None

        if self.last_smoothed_pos is None:
            self.last_smoothed_pos = t_cube.copy()
        else:
            a = self._ema_alpha
            self.last_smoothed_pos = a * t_cube + (1.0 - a) * self.last_smoothed_pos
        
        pos = self.last_smoothed_pos

        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.header.stamp = rgb_msg.header.stamp
        pose.pose.position.x = float(pos[0])
        pose.pose.position.y = float(pos[1])
        pose.pose.position.z = float(pos[2])
        pose.pose.orientation.w = 1.0

        self.last_pose_world = pose
        self.last_seen_time  = now
        self._pub_detected(True)

        if self.debug_viz:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea) if contours else None
            # Project mean center back to 2D for debug visualization
            # Simplified projection, just plotting center of bounding box
            u_center = int(np.mean(u_idx[top_face_mask]))
            v_center = int(np.mean(v_idx[top_face_mask]))
            bbox = cv2.boundingRect(contour) if contour is not None else None
            self._pub_debug(bgr, contour, mean_z, rgb_msg.header.frame_id,
                             u=u_center, v=v_center, bbox=bbox)

    def _wrist_pose_cb(self, msg: PoseStamped) -> None:
        """Store the refined wrist-camera pose when available."""
        self.last_wrist_pose = msg
        self.last_wrist_time = self.get_clock().now()

    # ── Contour detection ─────────────────────────────────────────────────────

    def _find_cube_contour(
        self,
        bgr: np.ndarray,
        h: int,
        w: int,
    ) -> Optional[Tuple[float, float, np.ndarray, Tuple]]:
        """
        WHAT: HSV threshold → morphological clean-up → largest contour → centroid.
        WHY:  Deterministic, zero-dependency detection for a single coloured object.
        INPUT: BGR frame, image height and width.
        OUTPUT: (centroid_u, centroid_v, contour, bounding_rect) or None.
        """
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_low, self._hsv_high)

        # Morphological clean-up removes isolated noise pixels
        k    = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < _MIN_BLOB_FRAC * w * h:
            return None  # too small — reject as noise

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        u = M["m10"] / M["m00"]
        v = M["m01"] / M["m00"]

        bbox = cv2.boundingRect(largest)
        return float(u), float(v), largest, bbox

    # ── Depth sampling ────────────────────────────────────────────────────────

    def _sample_depth(
        self,
        depth: np.ndarray,
        u: int, v: int,
        h: int, w: int,
    ) -> Optional[float]:
        """
        WHAT: Median depth in a patch around (u, v).
        WHY:  Median is robust to NaN/zero pixels common at object edges.
        INPUT: float32 depth image (metres), centroid pixel, image size.
        OUTPUT: Median depth in metres, or None if no valid pixels in patch.
        """
        r  = self._patch_r
        u0, u1 = max(0, u - r), min(w, u + r + 1)
        v0, v1 = max(0, v - r), min(h, v + r + 1)
        patch  = depth[v0:v1, u0:u1].flatten()
        valid  = patch[(patch > _DEPTH_MIN_M) & (patch < _DEPTH_MAX_M)
                       & np.isfinite(patch)]
        return float(np.median(valid)) if valid.size > 0 else None

    # ── TF transform ──────────────────────────────────────────────────────────

    def _to_world_pose(
        self,
        camera_frame: str,
        t_cam: np.ndarray,
        stamp,
    ) -> Optional[PoseStamped]:
        """
        WHAT: Transforms a 3D camera-frame point to world-frame PoseStamped.
        WHY:  All downstream planning works in the world frame.
        INPUT: Camera frame id, 3D point [x,y,z] in camera frame, ROS stamp.
        OUTPUT: PoseStamped (world frame, identity orientation) or None.

        Frame note:
          OpenCV camera convention: +Z forward, +Y down, +X right.
          MuJoCo world convention:  +Z up, +Y forward, +X right.
          A 180° rotation around the camera X-axis aligns the two conventions.
        """
        try:
            tf_wc = self.tf_buffer.lookup_transform(
                "world", camera_frame, rgb_msg.header.stamp, timeout=Duration(seconds=0.2))
        except TransformException as exc:
            self.get_logger().warn(
                f"TF world←{camera_frame}: {exc}", throttle_duration_sec=2.0)
            return None

        t_wc = np.array([tf_wc.transform.translation.x,
                         tf_wc.transform.translation.y,
                         tf_wc.transform.translation.z], dtype=np.float64)
        q_wc = np.array([tf_wc.transform.rotation.x,
                         tf_wc.transform.rotation.y,
                         tf_wc.transform.rotation.z,
                         tf_wc.transform.rotation.w], dtype=np.float64)
        rot_wc = Rotation.from_quat(q_wc)

        # Flip OpenCV → MuJoCo camera convention
        flip   = Rotation.from_euler('x', 180, degrees=True)
        t_cube = t_wc + rot_wc.apply(flip.apply(t_cam))

        # EMA smoothing
        now = self.get_clock().now()
        if (self.last_seen_time is not None
                and (now - self.last_seen_time).nanoseconds / 1e9 > 1.5):
            self.last_smoothed_pos = None  # stale → reset

        if self.last_smoothed_pos is None:
            self.last_smoothed_pos = t_cube.copy()
        else:
            a = self._ema_alpha
            self.last_smoothed_pos = a * t_cube + (1.0 - a) * self.last_smoothed_pos
        pos = self.last_smoothed_pos

        pose = PoseStamped()
        pose.header.frame_id     = "world"
        pose.header.stamp        = stamp
        pose.pose.position.x     = float(pos[0])
        pose.pose.position.y     = float(pos[1])
        pose.pose.position.z     = float(pos[2])
        pose.pose.orientation.w  = 1.0  # identity — cube orientation fixed
        return pose

    # ── Pose publish timer ────────────────────────────────────────────────────

    def _pose_timer_cb(self) -> None:
        """Publish last valid pose at 10 Hz; prefer fresh wrist-camera estimate."""
        now = self.get_clock().now()

        # Wrist camera has priority when its reading is < 1.5 s old
        if (self.last_wrist_pose is not None
                and self.last_wrist_time is not None
                and (now - self.last_wrist_time).nanoseconds < int(1.5e9)):
            self.last_pose_world = self.last_wrist_pose
            self.last_seen_time  = now
            self.cube_pose_pub.publish(self.last_pose_world)
            self._pub_detected(True)
            return

        if self.last_pose_world is None or self.last_seen_time is None:
            return
        if (now - self.last_seen_time).nanoseconds > int(_TIMEOUT_S * 1e9):
            return
        self.cube_pose_pub.publish(self.last_pose_world)

    def _pub_detected(self, detected: bool) -> None:
        msg = Bool()
        msg.data = detected
        self.cube_detected_pub.publish(msg)

    # ── Debug visualisation ───────────────────────────────────────────────────

    def _pub_debug(
        self,
        bgr: np.ndarray,
        contour,
        depth_m: Optional[float],
        frame_id: str,
        u: float = 0.0,
        v: float = 0.0,
        bbox: Optional[Tuple] = None,
    ) -> None:
        """Draw detection overlay and publish to /cube_debug_image at ≤5 Hz."""
        now = self.get_clock().now()
        if (self.last_debug_pub is not None
                and (now - self.last_debug_pub).nanoseconds
                < int((1.0 / _DEBUG_HZ) * 1e9)):
            return
        self.last_debug_pub = now

        debug = bgr.copy()
        if contour is not None:
            # Contour outline
            cv2.drawContours(debug, [contour], -1, (0, 165, 255), 2)
            if bbox is not None:
                bx, by, bw, bh = bbox
                cv2.rectangle(debug, (bx, by), (bx + bw, by + bh), (0, 210, 255), 1)
            # Centroid
            cv2.circle(debug, (int(u), int(v)), 7, (0, 0, 255), -1)
            # Crosshair
            cv2.line(debug, (int(u) - 18, int(v)), (int(u) + 18, int(v)), (255, 255, 255), 1)
            cv2.line(debug, (int(u), int(v) - 18), (int(u), int(v) + 18), (255, 255, 255), 1)
            # Labels
            label = "CUBE"
            if depth_m is not None:
                label += f"  z={depth_m:.3f} m"
            cv2.putText(debug, label,
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
            if self.last_smoothed_pos is not None:
                p = self.last_smoothed_pos
                cv2.putText(debug,
                            f"world ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)
        else:
            cv2.putText(debug, "NO CUBE",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        try:
            msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            msg.header.frame_id = frame_id
            msg.header.stamp    = now.to_msg()
            self.debug_image_pub.publish(msg)
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().warn(f"Debug image publish failed: {exc}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None) -> None:
    """ROS2 entry point — start the cube detector node."""
    rclpy.init(args=args)
    node = CubeDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
