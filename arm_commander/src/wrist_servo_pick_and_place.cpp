// PERCEPTION PIPELINE — wrist visual servoing
// Part of arm_commander package
// Created for precise pick-and-place with 3-camera system

/**
 * wrist_servo_pick_and_place.cpp
 *
 * WHAT: Full 3-camera perception-driven bimanual pick-and-place node.
 * WHY:  Supersedes aruco_pick_and_place.cpp — adds right-wrist and
 *       left-wrist visual servoing for precise grasping and handover.
 *
 * Camera pipeline:
 *   Chest cam  → coarse cube pose   → pre-grasp position (Phase 1-2)
 *   Right wrist → fine alignment    → precise pick + correct gripper width (Phase 3-4)
 *   Left wrist  → handover alignment → precise transfer (Phase 7)
 *
 * Execution phases:
 *   0  Home both arms, open grippers
 *   1  Wait for chest camera cube detection
 *   2  Right arm moves to pre-grasp above cube (downward grasp look position)
 *   3  Right wrist visual servo — align + measure cube width
 *   4  Open gripper to cube width, approach vertically, close to cube width, lift
 *   5  Right arm moves to handover pose
 *   6  Left arm opens gripper, moves to pre-handover pose
 *   7  Left wrist visual servo — align to cube held by right arm
 *   8  Transfer — left closes, right releases, right retreats
 *   9  Left arm places cube at fixed pose, opens gripper
 *  10  Both arms return home
 */

#include "arm_commander/commander.hpp"

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <algorithm>   // std::clamp
#include <chrono>
#include <cmath>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

using namespace std::chrono_literals;

// ── Downward-facing grasp orientation ─────────────────────────────────────────
// This orientation has the gripper pointing straight down at the table/marker
static constexpr double HANDS_UP_QX  = -0.007;
static constexpr double HANDS_UP_QY  = -0.104;
static constexpr double HANDS_UP_QZ  = 0.002;
static constexpr double HANDS_UP_QW  = 0.995;

static constexpr double GRASP_OFFSET_X = -0.02;
static constexpr double GRASP_OFFSET_Y = 0.0;
static constexpr double GRASP_OFFSET_Z = 0.006;

/// Phase-2 camera standoff in X (kept at 0 for straight above-cube pre-grasp).
static constexpr double WRIST_CAM_LOOK_X_OFFSET = 0.0;  // metres

/// Height of EE above grasp position when entering Phase 2 (pre-grasp pose).
/// Requirement: move to 15 cm above cube before wrist alignment.
static constexpr double WRIST_CAM_HOVER_HEIGHT  = 0.2;  // metres above grasp offset Z

/// After visual servo aligns XY, the arm descends this far to get close enough
/// for the gripper to close on the cube.
static constexpr double WRIST_CAM_DESCEND_DIST  = WRIST_CAM_HOVER_HEIGHT;

// ── Wrist servo safety ────────────────────────────────────────────────────────

/// Maximum Cartesian correction per servo step (caps wild first-frame errors).
static constexpr double MAX_SERVO_CORRECTION   = 0.02;   // metres

// ── Grasp geometry ────────────────────────────────────────────────────────────

/// Extra gap added to cube width when setting gripper before approach.
/// 3× margin = 12 mm total pre-approach clearance — prevents clipping.
static constexpr double GRASP_SAFETY_MARGIN    = 0.004;  // metres

/// After visual servo centres XY, arm descends this far to grasp height.
static constexpr double GRASP_APPROACH_DIST    = WRIST_CAM_DESCEND_DIST;  // metres

/// How far to lift after closing the gripper on the cube.
static constexpr double LIFT_HEIGHT            = 0.12;   // metres

// ── Handover poses (quaternion, verified reachable) ───────────────────────────

// Right arm handover position — where right arm holds cube for left to grasp
static constexpr double R_HAND_X  =  0.254, R_HAND_Y  = -0.062, R_HAND_Z  = 0.465;
static constexpr double R_HAND_QX =  0.492, R_HAND_QY = -0.462, R_HAND_QZ = -0.445, R_HAND_QW = 0.589;

// Left arm handover position — where left arm approaches to receive the cube
static constexpr double L_HAND_X  =  0.237, L_HAND_Y  =  0.051, L_HAND_Z  = 0.436;
static constexpr double L_HAND_QX =  0.517, L_HAND_QY =  0.544, L_HAND_QZ = -0.473, L_HAND_QW = -0.462;

// Right arm retreat position — where right arm moves after releasing the cube
static constexpr double R_RET_X   =  0.278, R_RET_Y   = -0.190, R_RET_Z   = 0.479;
static constexpr double R_RET_QX  =  0.492, R_RET_QY  = -0.462, R_RET_QZ  = -0.445, R_RET_QW  = 0.589;

// ── Place pose (fixed) ────────────────────────────────────────────────────────

// Where the left arm deposits the cube after the handover
static constexpr double PLACE_X   =  0.181, PLACE_Y   =  0.259, PLACE_Z   = 0.436;
static constexpr double PLACE_QX  =  0.045, PLACE_QY  = -0.749, PLACE_QZ  = -0.016, PLACE_QW  = 0.661;

// ── Timing ────────────────────────────────────────────────────────────────────

/// Delay between major phases — gives MoveIt time to settle and log messages
/// to remain legible.
static constexpr int STEP_DELAY_MS = 800;

// ── Timeouts ──────────────────────────────────────────────────────────────────

/// Seconds to wait for initial chest-cam cube detection before aborting.
static constexpr double CUBE_WAIT_TIMEOUT_SEC    = 30.0;
/// Seconds to wait for wrist alignment service before giving up.
static constexpr double ALIGN_SERVICE_TIMEOUT_SEC = 15.0;
/// Seconds to wait for a single cube-width message on a topic.
static constexpr double WIDTH_TOPIC_TIMEOUT_SEC   =  2.0;

// ─────────────────────────────────────────────────────────────────────────────
// Utility: stepDelay
// ─────────────────────────────────────────────────────────────────────────────

/**
 * WHAT: Logs a human-readable step label then sleeps STEP_DELAY_MS ms.
 * WHY:  Makes the log easy to follow and gives the controller time to settle.
 * INPUT:  node — used for logging; message — label printed to INFO.
 * OUTPUT: Sleep side-effect only.
 */
static void stepDelay(
  const rclcpp::Node::SharedPtr & node,
  const std::string & message)
{
  RCLCPP_INFO(node->get_logger(), "   ↳ %s", message.c_str());
  std::this_thread::sleep_for(std::chrono::milliseconds(STEP_DELAY_MS));
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: CubePoseTracker
// ─────────────────────────────────────────────────────────────────────────────

/**
 * WHAT: Thread-safe subscriber wrapper that caches the latest detected cube pose.
 * WHY:  Main execution thread blocks in waitForCubePose() and checks this cache.
 * INPUT:  ROS node; subscribes to /cube_pose and /cube_detected internally.
 * OUTPUT: getLatestDetectedPose() returns true + pose when cube is seen.
 */
class CubePoseTracker
{
public:
  explicit CubePoseTracker(const rclcpp::Node::SharedPtr & node)
  {
    // Incoming: chest-camera ArUco pose in world frame → caches for orchestrator
    cube_pose_sub_ = node->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/cube_pose", 10,
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_cube_pose_ = *msg;
      });

    // Incoming: chest-camera detection flag → gate on last_cube_pose_ freshness
    cube_detected_sub_ = node->create_subscription<std_msgs::msg::Bool>(
      "/cube_detected", 10,
      [this](const std_msgs::msg::Bool::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        cube_detected_ = msg->data;
      });
  }

  /**
   * WHAT: Returns the most recent cube pose if detection is currently active.
   * INPUT:  pose_out — filled with PoseStamped on success.
   * OUTPUT: true if cube is detected AND a valid pose has been received.
   */
  bool getLatestDetectedPose(geometry_msgs::msg::PoseStamped & pose_out) const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!cube_detected_.value_or(false) || !last_cube_pose_.has_value()) {
      return false;
    }
    pose_out = *last_cube_pose_;
    return true;
  }

private:
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr cube_pose_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr              cube_detected_sub_;
  mutable std::mutex mutex_;
  std::optional<bool>                                     cube_detected_;
  std::optional<geometry_msgs::msg::PoseStamped>          last_cube_pose_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Utility: waitForCubePose
// ─────────────────────────────────────────────────────────────────────────────

/**
 * WHAT: Blocks until a valid cube pose is received from the chest camera or
 *       timeout expires.
 * WHY:  Provides a safe blocking wait with log feedback every 2 seconds.
 * INPUT:
 *   node        — used for logging and clock.
 *   tracker     — CubePoseTracker to poll.
 *   timeout_sec — maximum wait time in seconds.
 * OUTPUT: std::optional<PoseStamped> — has value on success, nullopt on timeout.
 */
static std::optional<geometry_msgs::msg::PoseStamped>
waitForCubePose(
  const rclcpp::Node::SharedPtr & node,
  const CubePoseTracker & tracker,
  double timeout_sec,
  const rclcpp::Time & start_time)
{
  const auto start    = node->now();
  auto       next_log = start;

  while ((node->now() - start).seconds() < timeout_sec) {
    geometry_msgs::msg::PoseStamped cube_pose;
    if (tracker.getLatestDetectedPose(cube_pose)) {
      // Discard stale detections from before homing completed
      if (rclcpp::Time(cube_pose.header.stamp) >= start_time) {
        return cube_pose;
      }
    }

    if (node->now() >= next_log) {
      RCLCPP_INFO(node->get_logger(), "Waiting for cube detection from chest camera...");
      next_log = node->now() + rclcpp::Duration::from_seconds(2.0);
    }
    std::this_thread::sleep_for(100ms);
  }

  RCLCPP_ERROR(node->get_logger(),
    "Cube detection TIMEOUT after %.0f seconds — aborting", timeout_sec);
  return std::nullopt;
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: callAlignService
// ─────────────────────────────────────────────────────────────────────────────

/**
 * WHAT: Calls a wrist servo /align Trigger service synchronously.
 * WHY:  Blocks the pick-and-place sequence until the wrist is aligned,
 *       or aborts after timeout to prevent hanging the demo.
 * INPUT:
 *   node         — for logging and service client creation.
 *   service_name — full service name, e.g. "/right_wrist/align".
 *   timeout_sec  — how long to wait for the service to respond.
 *   out_cube_width_m — set to the cube width parsed from response.message.
 * OUTPUT: true if response.success == true; false on failure or timeout.
 *         out_cube_width_m is set when returning true.
 */
static bool callAlignService(
  const rclcpp::Node::SharedPtr & node,
  const std::string & service_name,
  double timeout_sec,
  double & out_cube_width_m)
{
  out_cube_width_m = 0.04;  // sensible default: 4 cm cube

  auto client = node->create_client<std_srvs::srv::Trigger>(service_name);

  RCLCPP_INFO(node->get_logger(),
    "Waiting for service '%s'...", service_name.c_str());

  if (!client->wait_for_service(
        std::chrono::duration<double>(timeout_sec / 2.0)))
  {
    RCLCPP_ERROR(node->get_logger(),
      "Service '%s' not available after %.0f s", service_name.c_str(),
      timeout_sec / 2.0);
    return false;
  }

  auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
  auto future  = client->async_send_request(request);

  // Spin the executor while waiting for the blocking align service response
  const auto deadline =
    std::chrono::steady_clock::now() +
    std::chrono::duration<double>(timeout_sec);

  while (std::chrono::steady_clock::now() < deadline) {
    if (future.wait_for(100ms) == std::future_status::ready) {
      auto response = future.get();
      RCLCPP_INFO(node->get_logger(),
        "Align service '%s' response: success=%s message='%s'",
        service_name.c_str(),
        response->success ? "true" : "false",
        response->message.c_str());

      if (response->success) {
        // Parse cube_width_m from response.message string
        try {
          out_cube_width_m = std::stod(response->message);
        } catch (const std::exception & e) {
          RCLCPP_WARN(node->get_logger(),
            "Could not parse cube_width from message '%s': %s — using %.3f m",
            response->message.c_str(), e.what(), out_cube_width_m);
          // out_cube_width_m stays at the default 4 cm
        }
      }
      return response->success;
    }
    std::this_thread::sleep_for(50ms);
  }

  RCLCPP_ERROR(node->get_logger(),
    "Align service '%s' TIMEOUT after %.0f s", service_name.c_str(), timeout_sec);
  return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: clamp helper
// ─────────────────────────────────────────────────────────────────────────────

/**
 * WHAT: Clamps value to [-limit, +limit].
 * WHY:  std::clamp requires C++17; explicit helper avoids ambiguity.
 */
static inline double clampAbs(double value, double limit)
{
  return std::max(-limit, std::min(limit, value));
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: abortToHome
// ─────────────────────────────────────────────────────────────────────────────

/**
 * WHAT: Logs a fatal error, homes both arms, shuts down ROS, joins spinner.
 * WHY:  Deduplicates the "something went wrong → safe home → exit" pattern
 *       that appears after every critical phase failure.
 * INPUT:  node, commander, spinner — the live runtime objects.
 *         phase — human label for the log (e.g. "Phase 2a").
 *         reason — why we are aborting.
 * OUTPUT: Does NOT return — calls std::exit(1) after cleanup.
 */
static void abortToHome(
  const rclcpp::Node::SharedPtr & node,
  Commander & commander,
  std::thread & spinner,
  const std::string & phase,
  const std::string & reason)
{
  RCLCPP_ERROR(node->get_logger(),
    "%s FAILED — %s. Returning home and aborting.",
    phase.c_str(), reason.c_str());
  commander.goToNamedTarget("hands_up", "right");
  commander.goToNamedTarget("hands_up", "left");
  rclcpp::shutdown();
  spinner.join();
  std::exit(1);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  // Node name used in all log messages
  auto node = std::make_shared<rclcpp::Node>("wrist_servo_pick_and_place");

  // Executor runs in a background thread — node stays responsive for topic
  // callbacks (cube pose, cube width) while main thread runs the sequence.
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  auto spinner = std::thread([&executor]() { executor.spin(); });

  Commander commander(node);
  CubePoseTracker tracker(node);

  RCLCPP_INFO(node->get_logger(),
    "════════════════════════════════════════");
  RCLCPP_INFO(node->get_logger(),
    " WRIST SERVO PICK AND PLACE — START");
  RCLCPP_INFO(node->get_logger(),
    "════════════════════════════════════════");

  // ── Phase 0: Home ─────────────────────────────────────────────────────────
  RCLCPP_INFO(node->get_logger(), "Phase 0: Homing both arms");

  commander.goToNamedTarget("hands_up", "right");
  stepDelay(node, "right arm homed");
  commander.goToNamedTarget("hands_up", "left");
  stepDelay(node, "left arm homed");
  commander.openGripper("right");
  commander.openGripper("left");
  stepDelay(node, "grippers opened — ready for demo");

  RCLCPP_INFO(node->get_logger(), "Phase 0 DONE — both arms home, grippers open");

  // ── Phase 1: Chest camera coarse cube detection ───────────────────────────
  RCLCPP_INFO(node->get_logger(),
    "Phase 1: Waiting for cube detection from chest camera");

  const auto phase1_start_time = node->now();
  auto cube_pose_opt = waitForCubePose(node, tracker, CUBE_WAIT_TIMEOUT_SEC, phase1_start_time);
  if (!cube_pose_opt.has_value()) {
    RCLCPP_ERROR(node->get_logger(),
      "Phase 1 FAILED — no cube detected. Returning home and aborting.");
    commander.goToNamedTarget("hands_up", "right");
    commander.goToNamedTarget("hands_up", "left");
    rclcpp::shutdown();
    spinner.join();
    return 1;
  }

  const auto & cube_pose = cube_pose_opt.value();
  RCLCPP_INFO(node->get_logger(),
    "Phase 1 DONE — cube at [x=%.3f, y=%.3f, z=%.3f]",
    cube_pose.pose.position.x,
    cube_pose.pose.position.y,
    cube_pose.pose.position.z);

  // ── Phase 2: Right arm → wrist-cam look position above cube ────────────────
  //
  // Wrist-parallel-to-table approach:
  //   Move to a pre-grasp pose directly above the cube with wrist level.
  //   Pre-grasp Z is 15 cm above grasp height for safe collision-free planning.
  //
  RCLCPP_INFO(node->get_logger(),
    "Phase 2: Moving right arm above cube for wrist-parallel view");

  // Pre-grasp pose: cube center + offsets + 15 cm hover.
  auto wrist_look_pose = commander.makePose(
    cube_pose.pose.position.x + GRASP_OFFSET_X + WRIST_CAM_LOOK_X_OFFSET,
    cube_pose.pose.position.y + GRASP_OFFSET_Y,
    cube_pose.pose.position.z + GRASP_OFFSET_Z + WRIST_CAM_HOVER_HEIGHT,
    HANDS_UP_QX, HANDS_UP_QY, HANDS_UP_QZ, HANDS_UP_QW);

  RCLCPP_INFO(node->get_logger(),
    "Phase 2: EE pre-grasp target → [x=%.3f, y=%.3f, z=%.3f]  (15 cm hover)",
    wrist_look_pose.position.x,
    wrist_look_pose.position.y,
    wrist_look_pose.position.z);

  commander.moveToPose(wrist_look_pose, "right");
  if (!commander.lastCommandSucceeded()) {
    abortToHome(node, commander, spinner,
      "Phase 2", "planning failed for 15 cm pre-grasp pose");
  }
  stepDelay(node, "right arm at 15 cm pre-grasp hover — wrist parallel");

  RCLCPP_INFO(node->get_logger(),
    "Phase 2 DONE — right arm at pre-grasp hover, wrist parallel");

  // ── Phase 3: Right wrist visual servoing ──────────────────────────────────
  RCLCPP_INFO(node->get_logger(),
    "Phase 3: Right wrist visual servoing — aligning to cube");

  // Subscribe to /right_wrist/servo_correction to apply visual servo corrections
  auto right_servo_sub = node->create_subscription<geometry_msgs::msg::Twist>(
    "/right_wrist/servo_correction", 10,
    [&commander, &node](const geometry_msgs::msg::Twist::SharedPtr msg) {
      static bool right_servo_moving = false;
      if (right_servo_moving) return;

      // Translate camera frame corrections to world frame (90-deg CW rotation when pointing down)
      double dx = msg->linear.y;   // Y_cam -> +X_world
      double dy = -msg->linear.x;  // X_cam -> -Y_world

      if (std::abs(dx) < 0.0005 && std::abs(dy) < 0.0005) return;

      dx = clampAbs(dx, MAX_SERVO_CORRECTION);
      dy = clampAbs(dy, MAX_SERVO_CORRECTION);

      right_servo_moving = true;
      RCLCPP_INFO(node->get_logger(), "Applying right wrist servo correction: [dx=%.4f, dy=%.4f]", dx, dy);
      commander.moveCartesianByAxis("right", dx, dy, 0.0);
      right_servo_moving = false;
    });

  double right_cube_width_m = 0.04;   // fallback if service parse fails
  bool right_aligned = callAlignService(
    node, "/right_wrist/align", ALIGN_SERVICE_TIMEOUT_SEC, right_cube_width_m);

  // Clean up subscription immediately after alignment finishes
  right_servo_sub.reset();

  if (!right_aligned) {
    abortToHome(node, commander, spinner,
      "Phase 3", "right wrist visual alignment timed out");
  }

  RCLCPP_INFO(node->get_logger(),
    "Phase 3 DONE — right wrist aligned to cube");
  RCLCPP_INFO(node->get_logger(),
    "  Cube width detected: %.4f m", right_cube_width_m);

  // Open gripper to cube width + generous pre-approach margin (3× safety)
  commander.setGripperWidth("right",
    right_cube_width_m + GRASP_SAFETY_MARGIN * 3.0);
  stepDelay(node, "right gripper pre-opened to cube width + margin");

  // ── Phase 4: Approach, close gripper, lift ────────────────────────────────
  RCLCPP_INFO(node->get_logger(),
    "Phase 4: Approaching cube and grasping");

  // Move toward cube along −Z axis (descend vertically to grasp height)
  commander.moveCartesianByZ("right", -GRASP_APPROACH_DIST);
  stepDelay(node, "right arm at grasp position — closing gripper");

  // Close to cube width − small safety margin so fingers make firm contact
  commander.setGripperWidth("right",
    right_cube_width_m - GRASP_SAFETY_MARGIN);
  stepDelay(node, "right gripper closed on cube — lifting");

  // Lift cube off table
  commander.moveCartesianByZ("right", LIFT_HEIGHT);
  stepDelay(node, "cube lifted — moving to handover pose");

  RCLCPP_INFO(node->get_logger(), "Phase 4 DONE — cube grasped and lifted");

  // ── Phase 5: Right arm → handover pose ───────────────────────────────────
  RCLCPP_INFO(node->get_logger(),
    "Phase 5: Right arm moving to handover pose");

  auto right_handover = commander.makePose(
    R_HAND_X, R_HAND_Y, R_HAND_Z,
    R_HAND_QX, R_HAND_QY, R_HAND_QZ, R_HAND_QW);
  commander.moveToPose(right_handover, "right");
  stepDelay(node, "right arm at handover — waiting for left arm");

  RCLCPP_INFO(node->get_logger(), "Phase 5 DONE — right arm at handover pose");

  // ── Phase 6: Left arm → pre-handover position ────────────────────────────
  RCLCPP_INFO(node->get_logger(),
    "Phase 6: Left arm moving to pre-handover position");

  commander.openGripper("left");
  stepDelay(node, "left gripper opened");

  auto left_handover = commander.makePose(
    L_HAND_X, L_HAND_Y, L_HAND_Z,
    L_HAND_QX, L_HAND_QY, L_HAND_QZ, L_HAND_QW);
  commander.moveToPose(left_handover, "left");
  stepDelay(node, "left arm at pre-handover — starting wrist servo");

  RCLCPP_INFO(node->get_logger(), "Phase 6 DONE — left arm at handover approach");

  // ── Phase 7: Left wrist visual servoing for handover ─────────────────────
  RCLCPP_INFO(node->get_logger(),
    "Phase 7: Left wrist visual servoing — aligning for handover");

  // Subscribe to /left_wrist/servo_correction to apply left wrist visual corrections
  auto left_servo_sub = node->create_subscription<geometry_msgs::msg::Twist>(
    "/left_wrist/servo_correction", 10,
    [&commander, &node](const geometry_msgs::msg::Twist::SharedPtr msg) {
      static bool left_servo_moving = false;
      if (left_servo_moving) return;

      // At L_HAND pose, X_cam maps to +Z_world and Y_cam maps to +Y_world
      double dy = msg->linear.y;  // Y_cam -> +Y_world
      double dz = msg->linear.x;  // X_cam -> +Z_world

      if (std::abs(dy) < 0.0005 && std::abs(dz) < 0.0005) return;

      dy = clampAbs(dy, MAX_SERVO_CORRECTION);
      dz = clampAbs(dz, MAX_SERVO_CORRECTION);

      left_servo_moving = true;
      RCLCPP_INFO(node->get_logger(), "Applying left wrist servo correction: [dy=%.4f, dz=%.4f]", dy, dz);
      commander.moveCartesianByAxis("left", 0.0, dy, dz);
      left_servo_moving = false;
    });

  double handover_width_m = right_cube_width_m;  // good fallback from Phase 3
  bool left_aligned = callAlignService(
    node, "/left_wrist/align", ALIGN_SERVICE_TIMEOUT_SEC, handover_width_m);

  // Clean up left subscription
  left_servo_sub.reset();

  if (!left_aligned) {
    RCLCPP_WARN(node->get_logger(),
      "Phase 7: Left wrist alignment failed — attempting handover with "
      "Phase 3 cube width (%.4f m)", handover_width_m);
    // Non-fatal: continue with Phase 3 measurement as best guess
  } else {
    RCLCPP_INFO(node->get_logger(),
      "Phase 7 DONE — left wrist aligned, cube width=%.4f m", handover_width_m);
  }

  // Open left gripper to cube width + generous margin before approaching
  commander.setGripperWidth("left",
    handover_width_m + GRASP_SAFETY_MARGIN * 3.0);
  stepDelay(node, "left gripper opened to cube width + margin");

  // ── Phase 8: Transfer — left closes, right releases, right retreats ───────
  RCLCPP_INFO(node->get_logger(), "Phase 8: Handover transfer");

  // Left closes onto cube (firm grip)
  commander.setGripperWidth("left",
    handover_width_m - GRASP_SAFETY_MARGIN);
  stepDelay(node, "left gripper closed — releasing right gripper");

  // Right releases (opens wide — 5× margin to ensure full clear)
  commander.setGripperWidth("right",
    right_cube_width_m + GRASP_SAFETY_MARGIN * 5.0);
  stepDelay(node, "right gripper released — right arm retreating");

  // Right arm retreats to a safe position away from the handover zone
  auto right_retreat = commander.makePose(
    R_RET_X, R_RET_Y, R_RET_Z,
    R_RET_QX, R_RET_QY, R_RET_QZ, R_RET_QW);
  commander.moveToPose(right_retreat, "right");
  stepDelay(node, "right arm retreated — left arm moving to place");

  RCLCPP_INFO(node->get_logger(), "Phase 8 DONE — cube transferred to left arm");

  // ── Phase 9: Left arm places cube ────────────────────────────────────────
  RCLCPP_INFO(node->get_logger(), "Phase 9: Left arm placing cube");

  auto place_pose = commander.makePose(
    PLACE_X, PLACE_Y, PLACE_Z,
    PLACE_QX, PLACE_QY, PLACE_QZ, PLACE_QW);
  commander.moveToPose(place_pose, "left");
  stepDelay(node, "left arm at place pose — opening gripper");

  commander.openGripper("left");
  stepDelay(node, "left gripper opened — cube placed — returning home");

  RCLCPP_INFO(node->get_logger(), "Phase 9 DONE — cube placed");

  // ── Phase 10: Home ────────────────────────────────────────────────────────
  RCLCPP_INFO(node->get_logger(), "Phase 10: Returning home");

  commander.goToNamedTarget("hands_up", "right");
  stepDelay(node, "right arm homed");
  commander.goToNamedTarget("hands_up", "left");

  RCLCPP_INFO(node->get_logger(),
    "════════════════════════════════════════");
  RCLCPP_INFO(node->get_logger(),
    " ═══ WRIST SERVO PICK AND PLACE COMPLETE ═══");
  RCLCPP_INFO(node->get_logger(),
    "════════════════════════════════════════");

  rclcpp::shutdown();
  spinner.join();
  return 0;
}
