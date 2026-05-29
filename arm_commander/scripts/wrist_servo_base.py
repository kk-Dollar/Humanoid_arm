#!/usr/bin/env python3
# PERCEPTION PIPELINE — wrist visual servoing
# Part of arm_commander package
# Created for precise pick-and-place with 3-camera system

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

2. pixel_threshold (default 8.0 px)
   Maximum pixel error considered "aligned".
   - Too HIGH: gripper lands off-centre; accepts poor alignment.
   - Too LOW:  alignment never converges due to camera noise; always times out.
   At 320×240 resolution a threshold of 8 px ≈ 2.5 % of frame width.

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

import time
from typing import Optional, Tuple

import cv2
import cv2.aruco as aruco
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float64
from std_srvs.srv import Trigger


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

    # ArUco dictionary used for cube marker detection
    ARUCO_DICT = aruco.DICT_4X4_50

    # Minimum fraction of image area for a color blob to be considered valid.
    # Prevents tiny noise blobs from triggering the fallback detector.
    MIN_BLOB_AREA_FRAC = 0.002

    # Approximate depth (metres) used as fallback when ArUco pose is unavailable.
    # Used in the width-to-metres conversion: width_m = width_px * depth / fx
    FALLBACK_DEPTH_M = 0.20

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

        # Maximum pixel error to consider the arm "aligned" to the cube
        self.declare_parameter("pixel_threshold", 8.0)
        # Scales pixel error to metres: correction_m = error_px * servo_gain
        self.declare_parameter("servo_gain", 0.003)
        # Seconds before the /align service gives up and returns failure
        self.declare_parameter("alignment_timeout", 10.0)
        # ArUco marker ID to track (must match the marker on the cube)
        self.declare_parameter("marker_id", 0)
        # Physical size of the ArUco marker in metres (for pose estimation)
        self.declare_parameter("marker_size_m", 0.05)
        # If True, fall back to HSV color detection when ArUco is not visible
        self.declare_parameter("use_color_fallback", True)
        # HSV lower bound for color-blob fallback (default: orange cube)
        self.declare_parameter("cube_color_hsv_lower", [20, 100, 100])
        # HSV upper bound for color-blob fallback
        self.declare_parameter("cube_color_hsv_upper", [30, 255, 255])

        # Read parameter values into instance variables
        self._pixel_threshold: float = float(
            self.get_parameter("pixel_threshold").value)
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

        # ── State variables ──────────────────────────────────────────────────
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.camera_calibrated: bool = False

        # Servoing loop control
        self.servoing_active: bool = False
        self.aligned: bool = False

        # Most recent detection results (written by image_callback)
        self.last_cube_width_m: float = 0.0
        self.last_error_x: float = 999.0
        self.last_error_y: float = 999.0
        self.last_depth_m: float = self.FALLBACK_DEPTH_M

        # Latest image for debug publishing
        self._latest_bgr: Optional[np.ndarray] = None

        # OpenCV bridge for ROS ↔ NumPy image conversion
        self.bridge = CvBridge()

        # ArUco detector setup
        self._aruco_dict = aruco.Dictionary_get(self.ARUCO_DICT)
        self._aruco_params = aruco.DetectorParameters_create()

        # Debug image publish rate limiter
        self._last_debug_pub_time: float = 0.0
        self._debug_publish_interval: float = 0.2  # 5 Hz

        # ── Subscribers ──────────────────────────────────────────────────────

        # Incoming: raw camera frames from wrist camera → detection pipeline
        self.create_subscription(
            Image, image_topic, self.image_callback, 10)

        # Incoming: camera intrinsics → width-to-metres conversion and ArUco
        self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 10)

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

        # ── Service ──────────────────────────────────────────────────────────

        # /*/align : Trigger → blocks until aligned or timeout; returns cube width
        self.create_service(Trigger, align_service_name, self.align_callback)

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
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)
        self.camera_calibrated = True

    def image_callback(self, msg: Image) -> None:
        """
        WHAT: Runs cube detection on every incoming wrist camera frame.
        WHY:  Continuously updates error and width estimates for servoing loop.
        INPUT: Image message from /<cam>/image_raw.
        OUTPUT: Updates last_error_x/y, last_cube_width_m; publishes topics.
        """
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().warn(f"CvBridge failed: {exc}")
            return

        self._latest_bgr = bgr.copy()
        h, w = bgr.shape[:2]
        cx_img = w / 2.0
        cy_img = h / 2.0

        # ── Detection ────────────────────────────────────────────────────────
        result = self._detect_cube(bgr, cx_img, cy_img)

        if result is None:
            # No cube found — publish not-aligned, keep trying
            self.aligned = False
            self.last_error_x = 999.0
            self.last_error_y = 999.0
            self._publish_aligned(False)
            if self.servoing_active:
                self.get_logger().warn(
                    "Wrist servo: NO DETECTION — waiting for cube visibility",
                    throttle_duration_sec=2.0)
            self._maybe_publish_debug(bgr, cx_img, cy_img, detected=False)
            return

        center_u, center_v, width_px, depth_m = result
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
        pixel_error = np.hypot(error_x, error_y)
        is_aligned = pixel_error < self._pixel_threshold
        self.aligned = is_aligned
        self._publish_aligned(is_aligned)

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
        cx_img: float,
        cy_img: float,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        WHAT: Tries ArUco detection first; falls back to color blob if enabled.
        WHY:  ArUco is more accurate; color blob handles cases where the marker
              is occluded (e.g., during grasp approach).
        INPUT: BGR frame, image centre coordinates.
        OUTPUT: (center_u, center_v, width_px, depth_m) or None if no detection.
        """
        result = self._detect_aruco(bgr)
        if result is not None:
            return result

        if self._use_color_fallback:
            return self._detect_color_blob(bgr)

        return None

    def _detect_aruco(
        self, bgr: np.ndarray
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        WHAT: Detects the configured ArUco marker in the image.
        WHY:  Primary detection method — gives sub-pixel accuracy and depth.
        INPUT: BGR image frame.
        OUTPUT: (center_u, center_v, width_px, depth_m) or None if not found.
                depth_m is taken from ArUco pose estimation (tvec Z component).
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(
            gray, self._aruco_dict, parameters=self._aruco_params)

        if ids is None:
            return None

        indices = np.where(ids.flatten() == self._marker_id)[0]
        if indices.size == 0:
            return None

        idx = int(indices[0])
        corner_pts = corners[idx][0]  # shape (4, 2)

        center_u = float(np.mean(corner_pts[:, 0]))
        center_v = float(np.mean(corner_pts[:, 1]))

        # Width = horizontal span of the marker in pixels
        width_px = float(
            np.linalg.norm(corner_pts[1] - corner_pts[0]) +
            np.linalg.norm(corner_pts[2] - corner_pts[3])
        ) / 2.0

        # Estimate depth from ArUco pose if camera is calibrated
        depth_m = self.FALLBACK_DEPTH_M
        if (self.camera_calibrated
                and self.camera_matrix is not None
                and self.dist_coeffs is not None):
            try:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    [corners[idx]], self._marker_size_m,
                    self.camera_matrix, self.dist_coeffs)
                # tvec is in camera frame; Z = distance along optical axis
                depth_m = float(abs(tvecs[0][0][2]))
            except Exception:  # pylint: disable=broad-except
                pass  # Fall through to FALLBACK_DEPTH_M

        return center_u, center_v, width_px, depth_m

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
        """
        self.servoing_active = True
        self.aligned = False
        start_time = self.get_clock().now()

        self.get_logger().info("Align service called — starting servoing loop")

        while rclpy.ok():
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1.0e9

            if elapsed > self._alignment_timeout:
                self.get_logger().error(
                    f"Wrist alignment TIMEOUT after {elapsed:.1f} s")
                response.success = False
                response.message = (
                    f"Alignment timeout after {elapsed:.1f}s")
                self.servoing_active = False
                return response

            if self.aligned:
                self.get_logger().info(
                    f"Wrist ALIGNED — cube width={self.last_cube_width_m:.4f} m "
                    f"error=({self.last_error_x:.1f}, {self.last_error_y:.1f}) px")
                response.success = True
                # message field carries cube_width_m for the C++ caller to parse
                response.message = f"{self.last_cube_width_m:.6f}"
                self.servoing_active = False
                return response

            # Poll at 20 Hz; image_callback runs asynchronously via executor
            time.sleep(0.05)

        # rclpy not ok — shut down gracefully
        response.success = False
        response.message = "Node shutdown during alignment"
        self.servoing_active = False
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
