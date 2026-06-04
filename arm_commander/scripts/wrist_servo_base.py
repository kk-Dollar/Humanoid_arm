#!/usr/bin/env python3
# PERCEPTION PIPELINE — wrist visual servoing (YOLO edition)
# Part of arm_commander package
# ArUco detection replaced by YOLOv8 + orange colour filter

"""
wrist_servo_base.py — Shared base class for wrist-camera visual servoing.

══════════════════════════════════════════════════════════════════════════════
TUNING GUIDE
══════════════════════════════════════════════════════════════════════════════

1. servo_gain (default 0.003 m/px)
   Maps pixel error → Cartesian correction in metres.
   - Too HIGH: arm oscillates around the target, never settles.
               Symptom: /right_wrist/aligned flickers rapidly.
   - Too LOW:  alignment converges slowly, may time-out before grasping.
               Symptom: slow creep, timeout error in logs.
   Tune by watching the error_x / error_y readout on /right_wrist/debug_image.
   A good starting point: correction should be ≤ 5 mm per step.

2. alignment_threshold_m (default 0.005 m)
    Maximum XY alignment error (gripper-centre to cube-centre) in metres.
    - Too HIGH: gripper lands off-centre; accepts poor alignment.
    - Too LOW:  alignment may fail due to image/depth noise.
    Default 0.005 m = 5 mm.

3. FINGER_LENGTH_SCALE (in commander.cpp, default 0.09)
   Maps finger_joint1 radians → finger-tip gap metres.
     gap_m = joint_rad * FINGER_LENGTH_SCALE
   - Too LARGE: setGripperWidth opens LESS than requested (gripper too tight).
   - Too SMALL: setGripperWidth opens MORE than requested (gripper too wide).
   Measure the actual tip-to-tip gap when joint1 = 0.7 rad in MuJoCo
   (use a ruler in the simulation or tf2_echo on the finger links).
   Then set: FINGER_LENGTH_SCALE = measured_gap_m / 0.7

4. GRASP_SAFETY_MARGIN (in wrist_servo_pick_and_place.cpp, default 0.004 m)
   Extra gap added/subtracted on top of cube_width_m.
   - Too HIGH: gripper opens too wide; cube slips when closing.
   - Too LOW:  gripper crushes or clips cube edges.
   Start at 4 mm and adjust ±1 mm until reliable grip.

══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import Twist, PoseStamped, Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool, Float64
from std_srvs.srv import Trigger
from tf2_ros import TransformException


class WristServoBase(Node):
    """
    WHAT: Base class for wrist-camera visual servoing nodes.
    WHY:  Centralises all detection and servoing logic so the left and right
          nodes share a single implementation — only the topic strings differ.
    INPUT:  Constructor arguments define all ROS topic and service names.
    OUTPUT: Runs the servoing loop, publishes corrections, and provides
            the /*/align Trigger service used by the C++ orchestrator.
    """

    # ── Class-level constants ─────────────────────────────────────────────────

    # Minimum fraction of image area for a color blob to be considered valid.
    # Prevents tiny noise blobs from triggering the fallback detector.
    MIN_BLOB_AREA_FRAC = 0.002

    # Approximate depth (metres) used as fallback when YOLO gives no depth.
    FALLBACK_DEPTH_M = 0.20

    # Cube physical dimensions (full dimensions in metres)
    CUBE_SIZE_X = 0.07
    CUBE_SIZE_Y = 0.07
    CUBE_SIZE_Z = 0.06

    # Minimum blob area as fraction of image area — same as cube_detector.py
    MIN_BLOB_AREA_FRAC = 0.002

    # Fallback HSV range (orange) used when model colour cannot be read
    _FALLBACK_HSV_LOW  = np.array([5,  100, 80],  dtype=np.uint8)
    _FALLBACK_HSV_HIGH = np.array([25, 255, 255], dtype=np.uint8)

    def __init__(
        self,
        node_name: str,
        image_topic: str,
        camera_info_topic: str,
        correction_topic: str,
        width_topic: str,
        aligned_topic: str,
        debug_topic: str,
        align_service_name: str,
    ) -> None:
        """
        WHAT: Initialises the wrist servo node with all ROS interfaces.
        WHY:  One-time setup; after this the callbacks drive all behaviour.
        INPUT: All topic and service name strings (supplied by subclass).
        OUTPUT: Fully configured ROS node ready to run.
        """
        super().__init__(node_name)

        # ── Declare parameters ───────────────────────────────────────────────

        # Maximum XY error (metres) to consider the wrist aligned to cube centre
        self.declare_parameter("alignment_threshold_m", 0.005)
        # Scales pixel error to metres: correction_m = error_px * servo_gain
        self.declare_parameter("servo_gain", 0.003)
        # Seconds before the /align service gives up and returns failure
        self.declare_parameter("alignment_timeout", 10.0)
        # ArUco marker ID — kept for backward compat; not used in detection
        self.declare_parameter("marker_id", 0)
        # Physical size of the ArUco marker in metres — kept for backward compat
        self.declare_parameter("marker_size_m", 0.05)
        # If True, fall back to HSV color detection when contour fails
        self.declare_parameter("use_color_fallback", True)
        # HSV lower bound for color-blob fallback (used only if contour fails)
        self.declare_parameter("cube_color_hsv_lower", [20, 100, 100])
        # HSV upper bound for color-blob fallback
        self.declare_parameter("cube_color_hsv_upper", [30, 255, 255])
        # Path to the MuJoCo model — used to auto-derive cube colour
        self.declare_parameter("mujoco_model_path", "")

        # Read parameter values into instance variables
        self._alignment_threshold_m: float = float(
            self.get_parameter("alignment_threshold_m").value)
        self._servo_gain: float = float(
            self.get_parameter("servo_gain").value)
        self._alignment_timeout: float = float(
            self.get_parameter("alignment_timeout").value)
        self._marker_id: int = int(
            self.get_parameter("marker_id").value)
        self._marker_size_m: float = float(
            self.get_parameter("marker_size_m").value)
        self._use_color_fallback: bool = bool(
            self.get_parameter("use_color_fallback").value)
        hsv_lower_list = self.get_parameter("cube_color_hsv_lower").value
        hsv_upper_list = self.get_parameter("cube_color_hsv_upper").value
        self._hsv_lower = np.array(hsv_lower_list, dtype=np.uint8)
        self._hsv_upper = np.array(hsv_upper_list, dtype=np.uint8)

        # ── Auto-derive cube colour from MuJoCo model ─────────────────────────
        # Import the helper from cube_detector (same package directory)
        model_path = str(self.get_parameter("mujoco_model_path").value)
        self._hsv_low  = self._FALLBACK_HSV_LOW.copy()
        self._hsv_high = self._FALLBACK_HSV_HIGH.copy()
        if model_path:
            try:
                import mujoco  # type: ignore[import-untyped]
                from cube_detector import read_cube_color_from_model, rgba_to_hsv_range
                rgba = read_cube_color_from_model(model_path)
                if rgba is not None:
                    self._hsv_low, self._hsv_high = rgba_to_hsv_range(rgba)
                    self.get_logger().info(
                        f"Wrist servo: cube HSV range auto-derived from model "  
                        f"→ {self._hsv_low.tolist()} – {self._hsv_high.tolist()}")
            except Exception as exc:  # pylint: disable=broad-except
                self.get_logger().warn(
                    f"Could not auto-derive cube colour: {exc} — using fallback.")

        # ── State variables ──────────────────────────────────────────────────
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.camera_calibrated: bool = False

        # Servoing loop control
        self.servoing_active: bool = False
        self.aligned: bool = False
        self._aligned_event = threading.Event()  # Set by image_callback when aligned

        # Most recent detection results (written by image_callback)
        self.last_cube_width_m: float = 0.0
        self.last_error_x: float = 999.0
        self.last_error_y: float = 999.0
        self.last_error_m: float = 999.0
        self.last_depth_m: float = self.FALLBACK_DEPTH_M

        # Latest image for debug publishing
        self._latest_bgr: Optional[np.ndarray] = None

        # OpenCV bridge for ROS ↔ NumPy image conversion
        self.bridge = CvBridge()

        # TF2 listener for camera to world frame transform lookup
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._image_frame_id: Optional[str] = None

        # Debug image publish rate limiter
        self._last_debug_pub_time: float = 0.0
        self._debug_publish_interval: float = 0.2  # 5 Hz

        # ── Callback groups ──────────────────────────────────────────────────
        # CRITICAL: image/camera_info callbacks must NOT share a group with
        # align_callback, which blocks on threading.Event.wait(). With the
        # default MutuallyExclusiveCallbackGroup a blocking service callback
        # would starve image_callback, so _aligned_event would never fire.
        #
        # Solution: put subscriptions in one group and the service in another;
        # MultiThreadedExecutor can then run them concurrently.
        self._subscription_cbg = ReentrantCallbackGroup()   # concurrent image frames
        self._service_cbg = MutuallyExclusiveCallbackGroup()  # blocking align srv

        # ── Subscribers ──────────────────────────────────────────────────────

        # Incoming: synchronized RGB and Depth from wrist camera
        depth_topic = image_topic.replace("image_raw", "depth/image_raw")
        rgb_sub = Subscriber(self, Image, image_topic, callback_group=self._subscription_cbg)
        depth_sub = Subscriber(self, Image, depth_topic, callback_group=self._subscription_cbg)
        self._sync = ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.05)
        self._sync.registerCallback(self.image_callback)

        # Incoming: camera intrinsics → width-to-metres conversion and ArUco
        self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 10,
            callback_group=self._subscription_cbg)

        # ── Publishers ───────────────────────────────────────────────────────

        # Outgoing: Cartesian correction Twist → C++ orchestrator
        # linear.x/y hold corrections in metres; capped by MAX_SERVO_CORRECTION
        self._correction_pub = self.create_publisher(
            Twist, correction_topic, 10)

        # Outgoing: measured cube width in metres → C++ orchestrator (gripper width)
        self._width_pub = self.create_publisher(
            Float64, width_topic, 10)

        # Outgoing: alignment status (True = error < threshold) → orchestrator
        self._aligned_pub = self.create_publisher(
            Bool, aligned_topic, 10)

        # Outgoing: annotated debug image → human inspector / RViz
        self._debug_pub = self.create_publisher(
            Image, debug_topic, 10)

        # Outgoing: refined cube pose detected by wrist camera.
        self._wrist_cube_pose_pub = self.create_publisher(
            PoseStamped, "/wrist_cube_pose", 10)

        # ── Service ──────────────────────────────────────────────────────────

        # /*/align : Trigger → blocks until aligned or timeout; returns cube width
        # Must be in its own callback group so image_callback keeps running.
        self.create_service(
            Trigger, align_service_name, self.align_callback,
            callback_group=self._service_cbg)

        # ── Timer ────────────────────────────────────────────────────────────

        # 10 Hz correction publisher (active only when servoing_active is True)
        self.create_timer(0.1, self._correction_timer_cb)

        self.get_logger().info(
            f"{node_name} started — listening on {image_topic}")

    # ── Subscriber callbacks ──────────────────────────────────────────────────

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """
        WHAT: Stores camera intrinsic matrix and distortion coefficients.
        WHY:  ArUco pose estimation and pixel→metre conversion need these.
        INPUT: CameraInfo from /<cam>/camera_info topic.
        OUTPUT: Updates self.camera_matrix, self.dist_coeffs, sets calibrated flag.
        """
        if msg.k[0] < 10.0:
            return  # Ignore bogus normalized camera_info from mujoco_ros2_control
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)
        self.camera_calibrated = True

    def image_callback(self, msg: Image, depth_msg: Image) -> None:
        """
        WHAT: Runs cube detection on every incoming wrist camera frame.
        WHY:  Continuously updates error and width estimates for servoing loop.
        INPUT: Image message from /<cam>/image_raw.
        OUTPUT: Updates last_error_x/y, last_cube_width_m; publishes topics.
        """
        self._image_frame_id = msg.header.frame_id
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().warn(f"CvBridge failed: {exc}")
            return

        self._latest_bgr = bgr.copy()
        h, w = bgr.shape[:2]
        cx_img = w / 2.0
        cy_img = h / 2.0

        # ── Detection ────────────────────────────────────────────────────────
        result = self._detect_cube(bgr, depth, msg.header.frame_id, msg.header.stamp, cx_img, cy_img)

        if result is None:
            # No cube found — publish not-aligned, keep trying
            self.aligned = False
            self.last_error_x = 999.0
            self.last_error_y = 999.0
            self.last_error_m = 999.0
            self._publish_aligned(False)
            if self.servoing_active:
                self.get_logger().warn(
                    "Wrist servo: NO DETECTION — waiting for cube visibility",
                    throttle_duration_sec=2.0)
            self._maybe_publish_debug(bgr, cx_img, cy_img, detected=False)
            return

        center_u, center_v, width_px, depth_m, t_cube_world = result
        self.last_depth_m = depth_m

        # ── Pixel error from image centre ─────────────────────────────────
        error_x = center_u - cx_img   # +right in image  → world X correction
        error_y = center_v - cy_img   # +down  in image  → world Y correction

        self.last_error_x = error_x
        self.last_error_y = error_y

        # ── Width → metres conversion ─────────────────────────────────────
        # width_m = width_px * depth / fx
        if self.camera_calibrated and self.camera_matrix is not None:
            fx = float(self.camera_matrix[0, 0])
        else:
            fx = float(w) / (2.0 * np.tan(np.radians(60.0) / 2.0))  # 60° fallback

        cube_width_m = (width_px * depth_m) / fx if fx > 0 else 0.04
        self.last_cube_width_m = cube_width_m

        # Publish cube width for C++ orchestrator to read
        width_msg = Float64()
        width_msg.data = cube_width_m
        self._width_pub.publish(width_msg)

        # ── Convergence check ─────────────────────────────────────────────
        # Convert pixel centroid error into metric XY error at current depth:
        # error_m = error_px * depth / fx
        error_m_x = (error_x * depth_m) / fx if fx > 0 else 999.0
        error_m_y = (error_y * depth_m) / fx if fx > 0 else 999.0
        metric_error = float(np.hypot(error_m_x, error_m_y))
        self.last_error_m = metric_error
        is_aligned = metric_error < self._alignment_threshold_m
        self.aligned = is_aligned
        self._publish_aligned(is_aligned)
        if is_aligned:
            self._aligned_event.set()  # Wake up align_callback waiting thread

        # Publish the refined cube pose to /wrist_cube_pose when detected
        refined_pose = self._get_refined_pose_from_contour(bgr, depth, msg.header.frame_id, msg.header.stamp)
        if refined_pose is not None:
            self._wrist_cube_pose_pub.publish(refined_pose)

        self._maybe_publish_debug(
            bgr, cx_img, cy_img,
            detected=True,
            center_u=center_u, center_v=center_v,
            width_px=width_px, cube_width_m=cube_width_m,
            error_x=error_x, error_y=error_y,
            is_aligned=is_aligned,
        )

    # ── Detection pipeline ────────────────────────────────────────────────────

    def _detect_cube(
        self,
        bgr: np.ndarray,
        depth: np.ndarray,
        camera_frame: str,
        stamp,
        cx_img: float,
        cy_img: float,
    ) -> Optional[Tuple[float, float, float, float, np.ndarray]]:
        """
        WHAT: Tries YOLO detection first; falls back to color blob if enabled.
        WHY:  YOLO is more accurate; color blob handles cases where YOLO fails
              (e.g., during grasp approach when cube is too close / occluded).
        INPUT: BGR frame, image centre coordinates.
        OUTPUT: (center_u, center_v, width_px, depth_m) or None if no detection.
        """
        result = self._detect_contour(bgr, depth, camera_frame, stamp)
        if result is not None:
            return result

        if self._use_color_fallback:
            return self._detect_color_blob(bgr)

        return None

    def _detect_contour(
        self, bgr: np.ndarray, depth: np.ndarray, camera_frame: str, stamp
    ) -> Optional[Tuple[float, float, float, float, np.ndarray]]:
        h_img, w_img = bgr.shape[:2]
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_low, self._hsv_high)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        v_idx, u_idx = np.where(mask > 0)
        min_area = self.MIN_BLOB_AREA_FRAC * w_img * h_img
        if len(v_idx) < min_area:
            return None

        z_c = depth[v_idx, u_idx]
        valid = (z_c > 0.05) & (z_c < 0.80) & np.isfinite(z_c)
        v_idx, u_idx, z_c = v_idx[valid], u_idx[valid], z_c[valid]

        if len(z_c) == 0:
            return None

        if not self.camera_calibrated or self.camera_matrix is None:
            return None

        fx = float(self.camera_matrix[0, 0])
        fy = float(self.camera_matrix[1, 1])
        cx = float(self.camera_matrix[0, 2])
        cy = float(self.camera_matrix[1, 2])

        x_c = (u_idx - cx) * z_c / fx
        y_c = (v_idx - cy) * z_c / fy
        pts_cam = np.vstack((x_c, y_c, z_c))

        try:
            tf_wc = self._tf_buffer.lookup_transform(
                "world", camera_frame, stamp, timeout=Duration(seconds=0.2)
            )
        except Exception as exc:
            self.get_logger().warn(f"TF world<-{camera_frame} failed: {exc}", throttle_duration_sec=2.0)
            return None

        t_wc = np.array([tf_wc.transform.translation.x, tf_wc.transform.translation.y, tf_wc.transform.translation.z])
        q_wc = np.array([tf_wc.transform.rotation.x, tf_wc.transform.rotation.y, tf_wc.transform.rotation.z, tf_wc.transform.rotation.w])
        rot_wc = Rotation.from_quat(q_wc)
        flip = Rotation.from_euler('x', 180, degrees=True)
        
        pts_world = rot_wc.apply(flip.apply(pts_cam.T)).T + t_wc[:, np.newaxis]
        
        z_world = pts_world[2, :]
        max_z = np.max(z_world)
        top_face_mask = z_world >= (max_z - 0.015)
        
        if np.sum(top_face_mask) == 0:
            return None

        # Center in pixel coordinates for visual servoing
        center_u = float(np.mean(u_idx[top_face_mask]))
        center_v = float(np.mean(v_idx[top_face_mask]))
        
        # Depth in camera frame
        depth_m = float(np.mean(z_c[top_face_mask]))
        
        # Approximate width in pixels using bounding box of the top face mask
        min_u, max_u = np.min(u_idx[top_face_mask]), np.max(u_idx[top_face_mask])
        width_px = float(max_u - min_u)
        if width_px < 1.0:
            width_px = 1.0

        mean_x_world = float(np.mean(pts_world[0, top_face_mask]))
        mean_y_world = float(np.mean(pts_world[1, top_face_mask]))
        mean_z_world = float(np.mean(pts_world[2, top_face_mask]))
        t_cube_world = np.array([mean_x_world, mean_y_world, mean_z_world])

        return center_u, center_v, width_px, depth_m, t_cube_world

    def _detect_color_blob(
        self, bgr: np.ndarray
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        WHAT: Detects the cube by color segmentation in HSV space.
        WHY:  Fallback when ArUco marker is occluded (e.g., gripper approaching).
        INPUT: BGR image frame.
        OUTPUT: (center_u, center_v, width_px, FALLBACK_DEPTH_M) or None.
                Depth is the class fallback since color gives no Z information.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)

        # Morphological clean-up to remove isolated noise pixels
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Largest contour by area is assumed to be the cube
        largest = max(contours, key=cv2.contourArea)
        h_img, w_img = bgr.shape[:2]
        min_area = self.MIN_BLOB_AREA_FRAC * w_img * h_img

        if cv2.contourArea(largest) < min_area:
            return None  # Too small — likely noise

        x, y, w, h = cv2.boundingRect(largest)
        center_u = float(x + w / 2.0)
        center_v = float(y + h / 2.0)
        width_px = float(w)

        return center_u, center_v, width_px, self.FALLBACK_DEPTH_M

    # ── Correction publisher ──────────────────────────────────────────────────

    def _correction_timer_cb(self) -> None:
        """
        WHAT: Publishes Cartesian corrections at 10 Hz when servoing is active.
        WHY:  Decouples correction rate from image callback rate.
        INPUT: Timer event (10 Hz).
        OUTPUT: Twist on correction topic; linear.x/y hold metre deltas.
        """
        if not self.servoing_active:
            return

        twist = Twist()
        # Scale pixel error → metres using servo gain
        twist.linear.x = self.last_error_x * self._servo_gain
        twist.linear.y = self.last_error_y * self._servo_gain
        twist.linear.z = 0.0   # Z not corrected by wrist servoing
        self._correction_pub.publish(twist)

    # ── Alignment service ─────────────────────────────────────────────────────

    def align_callback(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """
        WHAT: Trigger service that blocks until the wrist is aligned to the cube.
        WHY:  C++ orchestrator calls this synchronously and waits for confirmation
              before commanding the approach move; avoids blind grasping.
        INPUT: std_srvs/Trigger request (no fields used).
        OUTPUT: response.success = True if aligned within timeout.
                response.message = "<cube_width_m as string>" on success
                response.message = "Alignment timeout after Xs" on failure.

        NOTE: We use threading.Event.wait() instead of time.sleep() so that
              the MultiThreadedExecutor thread pool can keep running
              image_callback concurrently to update self.aligned.
        """
        self.servoing_active = True
        self.aligned = False
        self._aligned_event.clear()  # Reset before starting

        self.get_logger().info(
            f"Align service called — starting servoing loop "
            f"(timeout={self._alignment_timeout:.0f}s, "
            f"threshold={self._alignment_threshold_m*1000:.1f}mm)")

        # Wait for image_callback to set _aligned_event, or timeout.
        # threading.Event.wait() releases the GIL so the executor thread pool
        # can run image_callback in a separate thread without being blocked.
        aligned = self._aligned_event.wait(timeout=self._alignment_timeout)

        self.servoing_active = False

        if aligned and rclpy.ok():
            self.get_logger().info(
                f"Wrist ALIGNED — cube width={self.last_cube_width_m:.4f} m "
                f"error=({self.last_error_x:.1f}, {self.last_error_y:.1f}) px "
                f"| {self.last_error_m*1000.0:.2f} mm")
            response.success = True
            # message field carries cube_width_m for the C++ caller to parse
            response.message = f"{self.last_cube_width_m:.6f}"
        else:
            self.get_logger().error(
                f"Wrist alignment TIMEOUT after {self._alignment_timeout:.1f}s "
                f"— last error={self.last_error_m*1000:.1f} mm")
            response.success = False
            response.message = f"Alignment timeout after {self._alignment_timeout:.1f}s"

        return response

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _publish_aligned(self, is_aligned: bool) -> None:
        """
        WHAT: Publishes Bool indicating whether gripper is centred on cube.
        WHY:  Lets external monitors observe servoing convergence.
        INPUT: Boolean alignment state.
        OUTPUT: Publication on /*/aligned topic.
        """
        msg = Bool()
        msg.data = is_aligned
        self._aligned_pub.publish(msg)

    def _maybe_publish_debug(
        self,
        bgr: np.ndarray,
        cx_img: float,
        cy_img: float,
        detected: bool,
        center_u: float = 0.0,
        center_v: float = 0.0,
        width_px: float = 0.0,
        cube_width_m: float = 0.0,
        error_x: float = 0.0,
        error_y: float = 0.0,
        is_aligned: bool = False,
    ) -> None:
        """
        WHAT: Draws diagnostic overlays on the camera image and publishes at 5 Hz.
        WHY:  Human inspection and debugging of the servoing loop.
        INPUT: BGR frame, detection results, error metrics.
        OUTPUT: Annotated Image published on /*/debug_image topic at ≤5 Hz.
        """
        now = time.monotonic()
        if now - self._last_debug_pub_time < self._debug_publish_interval:
            return
        self._last_debug_pub_time = now

        debug = bgr.copy()
        GREEN  = (0, 255, 0)
        RED    = (0, 0, 255)
        YELLOW = (0, 255, 255)
        WHITE  = (255, 255, 255)
        CYAN   = (255, 255, 0)

        cx = int(cx_img)
        cy = int(cy_img)

        # Green crosshair at image centre (where gripper is pointing)
        cv2.line(debug, (cx - 20, cy), (cx + 20, cy), GREEN, 2)
        cv2.line(debug, (cx, cy - 20), (cx, cy + 20), GREEN, 2)

        if detected:
            cu = int(center_u)
            cv_ = int(center_v)
            hw = int(width_px / 2)

            # Yellow rectangle around detected cube
            cv2.rectangle(debug,
                          (cu - hw, cv_ - hw), (cu + hw, cv_ + hw),
                          YELLOW, 2)

            # Red dot at detected cube centre
            cv2.circle(debug, (cu, cv_), 5, RED, -1)

            # Red line from crosshair to cube centre (error vector)
            cv2.line(debug, (cx, cy), (cu, cv_), RED, 2)

            # Text overlays
            cv2.putText(debug,
                        f"err_x:{error_x:+.1f}px  err_y:{error_y:+.1f}px",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
            cv2.putText(debug,
                        f"width: {cube_width_m:.4f} m",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

            if is_aligned:
                cv2.putText(debug, "ALIGNED",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
            elif self.servoing_active:
                cv2.putText(debug, "SERVOING",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
        else:
            cv2.putText(debug, "NO DETECTION",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)

        try:
            msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            self._debug_pub.publish(msg)
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().warn(f"Debug image publish failed: {exc}")

    def _get_refined_pose_from_contour(
        self, bgr: np.ndarray, depth: np.ndarray, camera_frame: str, stamp
    ) -> Optional[PoseStamped]:
        result = self._detect_contour(bgr, depth, camera_frame, stamp)
        if result is None:
            return None

        center_u, center_v, width_px, depth_m, t_world_cube = result

        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.header.stamp    = stamp
        pose.pose.position.x = float(t_world_cube[0])
        pose.pose.position.y = float(t_world_cube[1])
        pose.pose.position.z = float(t_world_cube[2])
        pose.pose.orientation.w = 1.0
        return pose

# ── Concrete subclasses ───────────────────────────────────────────────────────

class RightWristServoNode(WristServoBase):
    """
    WHAT: Right-arm wrist servo node.
    WHY:  Thin subclass; only supplies the right-arm topic name strings.
    """

    def __init__(self) -> None:
        super().__init__(
            node_name="right_wrist_servo",
            image_topic="/right_wrist_cam/image_raw",
            camera_info_topic="/right_wrist_cam/camera_info",
            correction_topic="/right_wrist/servo_correction",
            width_topic="/right_wrist/cube_width_m",
            aligned_topic="/right_wrist/aligned",
            debug_topic="/right_wrist/debug_image",
            align_service_name="/right_wrist/align",
        )


class LeftWristServoNode(WristServoBase):
    """
    WHAT: Left-arm wrist servo node.
    WHY:  Thin subclass; only supplies the left-arm topic name strings.
    """

    def __init__(self) -> None:
        super().__init__(
            node_name="left_wrist_servo",
            image_topic="/left_wrist_cam/image_raw",
            camera_info_topic="/left_wrist_cam/camera_info",
            correction_topic="/left_wrist/servo_correction",
            width_topic="/left_wrist/cube_width_m",
            aligned_topic="/left_wrist/aligned",
            debug_topic="/left_wrist/debug_image",
            align_service_name="/left_wrist/align",
        )
