// PERCEPTION PIPELINE — simplified pick-and-place (no visual servoing)
// Part of arm_commander package
//
// NOTE: Visual servoing alignment is BYPASSED.
// Pipeline: chest cam → hover → wrist cam refined pose → direct grasp.

/**
 * wrist_servo_pick_and_place.cpp
 *
 * WHAT: Bimanual pick-and-place (simplified — visual servoing bypassed).
 * WHY:  Removes wrist align service timeout issues; uses /wrist_cube_pose
 *       published by right_wrist_servo for accurate grasp position.
 *
 * Execution phases:
 *   0  Home both arms, open grippers
 *   1  Wait for chest camera cube detection (/cube_pose)
 *   2  Right arm moves to pre-grasp hover above cube
 *   3  Wait for wrist cam refined pose (/wrist_cube_pose, 5 s timeout)
 *      → moveToPose to grasp → close right gripper → attach cube → lift
 *   4  Right arm → handover pose (Pose 2)
 *   5  Left arm opens → handover approach (Pose 3)
 *   6  Left closes → transfer cube attachment → right opens → right retreats
 *      → left arm to place pose → detach cube → left opens
 *   Final: both arms home
 *
 * BYPASSED (kept under #if 0 for future re-enablement):
 *   - /right_wrist/align service  (visual servoing alignment)
 *   - /left_wrist/align  service  (handover alignment)
 *   - servo_correction subscription + moveCartesianByAxis
 *   - cube width measurement from wrist cam
 */

#include "arm_commander/commander.hpp"

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

using namespace std::chrono_literals;

// ── Downward-facing grasp orientation ─────────────────────────────────────────
// Gripper pointing straight down at the table / ArUco marker
static constexpr double HANDS_UP_QX  = -0.007;
static constexpr double HANDS_UP_QY  = -0.104;
static constexpr double HANDS_UP_QZ  =  0.002;
static constexpr double HANDS_UP_QW  =  0.995;

// ── Grasp offsets (applied on top of detected cube position) ──────────────────
// aruco_detector publishes cube CENTER z (~0.25 m in sim).
// Cube top is at center + half_height = 0.25 + 0.03 = 0.28 m.
// For a downward-facing gripper the EE must be ABOVE the cube top so that
// fingers wrap around the cube sides and do NOT extend below the table surface.
//   GRASP_OFFSET_Z = 0.050 m  →  EE at 0.25 + 0.050 = 0.300 m
//                               = 20 mm above cube top (0.28 m)
//   Finger tips (≈ 8 cm below EE)  →  z ≈ 0.220 m  ≈ table surface  ✓
// Tune ± 5 mm if fingers still clip the table or miss the cube.
static constexpr double GRASP_OFFSET_X =  0.033;   // m
static constexpr double GRASP_OFFSET_Y =  0.0;    // m
static constexpr double GRASP_OFFSET_Z =  0.136;  // m — EE above cube top


// ── Hover height above cube for wrist cam visibility ──────────────────────────
static constexpr double WRIST_CAM_LOOK_X_OFFSET = 0.0;  // m (computed for 25deg pitch view)
static constexpr double WRIST_CAM_LOOK_Y_OFFSET = 0.0;  // m
static constexpr double WRIST_CAM_HOVER_HEIGHT  = 0.24;  // m above grasp Z

// ── Post-grasp lift height ────────────────────────────────────────────────────
static constexpr double LIFT_HEIGHT = 0.1;  // m

// ── Handover poses ────────────────────────────────────────────────────────────
static constexpr double R_HAND_X  =  0.211, R_HAND_Y  = -0.210, R_HAND_Z  = 0.463;
static constexpr double R_HAND_QX =  0.495, R_HAND_QY = -0.502, R_HAND_QZ =  0.461, R_HAND_QW = 0.539;

// - Translation: [0.201, 0.292, 0.401]
// - Rotation: in Quaternion (xyzw) [-0.546, -0.450, -0.468, 0.529]

static constexpr double L_HAND_X_G  =  0.201, L_HAND_Y_G  =  0.292, L_HAND_Z_G  = 0.401;
static constexpr double L_HAND_QX_G = -0.546, L_HAND_QY_G = -0.450, L_HAND_QZ_G = -0.468, L_HAND_QW_G =  0.529;

static constexpr double L_HAND_X  =  0.213, L_HAND_Y  =  0.154, L_HAND_Z  = 0.463;
static constexpr double L_HAND_QX = -0.477, L_HAND_QY = -0.517, L_HAND_QZ = -0.444, L_HAND_QW =  0.555;

static constexpr double L_HAND_CORR_X=0.0;
static constexpr double L_HAND_CORR_Y=-0.04;
static constexpr double L_HAND_CORR_Z=0.02;

// ── Place pose ────────────────────────────────────────────────────────────────
// - Translation: [0.176, 0.350, 0.353]
// - Rotation: in Quaternion (xyzw) [0.026, -0.633, -0.014, 0.773]

// static constexpr double PLACE_X   =  0.131, PLACE_Y   =  0.297, PLACE_Z   = 0.34;
// static constexpr double PLACE_QX  =  0.223, PLACE_QY  = -0.666, PLACE_QZ  =  0.240, PLACE_QW  = 0.670;

static constexpr double PLACE_X   =  0.176, PLACE_Y   =  0.350, PLACE_Z   = 0.353;
static constexpr double PLACE_QX  =  0.026, PLACE_QY  = -0.633, PLACE_QZ  =  -0.014,
PLACE_QW = 0.773;
// ── Timing ────────────────────────────────────────────────────────────────────
static constexpr int    STEP_DELAY_MS        = 800;
static constexpr double CUBE_WAIT_TIMEOUT_SEC = 30.0;
static constexpr double WRIST_POSE_TIMEOUT_SEC =  5.0;

// ─────────────────────────────────────────────────────────────────────────────
// Utility: stepDelay
// ─────────────────────────────────────────────────────────────────────────────
static void stepDelay(const rclcpp::Node::SharedPtr & node, const std::string & msg)
{
  RCLCPP_INFO(node->get_logger(), "   ↳ %s", msg.c_str());
  std::this_thread::sleep_for(std::chrono::milliseconds(STEP_DELAY_MS));
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: abortToHome
// ─────────────────────────────────────────────────────────────────────────────
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
// CubePoseTracker — caches latest /cube_pose (chest cam, fused with wrist cam)
// ─────────────────────────────────────────────────────────────────────────────
class CubePoseTracker
{
public:
  explicit CubePoseTracker(const rclcpp::Node::SharedPtr & node)
  {
    pose_sub_ = node->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/cube_pose", 10,
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_pose_ = *msg;
      });
    det_sub_ = node->create_subscription<std_msgs::msg::Bool>(
      "/cube_detected", 10,
      [this](const std_msgs::msg::Bool::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        detected_ = msg->data;
      });
  }

  bool getLatestDetectedPose(geometry_msgs::msg::PoseStamped & out) const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!detected_.value_or(false) || !last_pose_.has_value()) return false;
    out = *last_pose_;
    return true;
  }

private:
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr              det_sub_;
  mutable std::mutex mutex_;
  std::optional<bool>                                     detected_;
  std::optional<geometry_msgs::msg::PoseStamped>          last_pose_;
};

// ─────────────────────────────────────────────────────────────────────────────
// WristPoseTracker — caches latest /wrist_cube_pose (right wrist cam refined)
// ─────────────────────────────────────────────────────────────────────────────
class WristPoseTracker
{
public:
  explicit WristPoseTracker(const rclcpp::Node::SharedPtr & node)
  {
    sub_ = node->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/wrist_cube_pose", 10,
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_pose_ = *msg;
      });
  }

  bool getLatestPose(geometry_msgs::msg::PoseStamped & out) const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!last_pose_.has_value()) return false;
    out = *last_pose_;
    return true;
  }

private:
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_;
  mutable std::mutex mutex_;
  std::optional<geometry_msgs::msg::PoseStamped> last_pose_;
};

// ─────────────────────────────────────────────────────────────────────────────
// waitForCubePose — block until /cube_pose arrives and is newer than start_time
// ─────────────────────────────────────────────────────────────────────────────
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
    geometry_msgs::msg::PoseStamped pose;
    if (tracker.getLatestDetectedPose(pose)) {
      if (rclcpp::Time(pose.header.stamp) >= start_time)
        return pose;
    }
    if (node->now() >= next_log) {
      RCLCPP_INFO(node->get_logger(), "Waiting for cube detection from chest camera...");
      next_log = node->now() + rclcpp::Duration::from_seconds(2.0);
    }
    std::this_thread::sleep_for(100ms);
  }
  RCLCPP_ERROR(node->get_logger(),
    "Cube detection TIMEOUT after %.0f s — aborting", timeout_sec);
  return std::nullopt;
}

// ─────────────────────────────────────────────────────────────────────────────
// waitForWristPose — block until /wrist_cube_pose arrives newer than not_before
// ─────────────────────────────────────────────────────────────────────────────
static std::optional<geometry_msgs::msg::PoseStamped>
waitForWristPose(
  const rclcpp::Node::SharedPtr & node,
  const WristPoseTracker & tracker,
  double timeout_sec,
  const rclcpp::Time & not_before)
{
  const auto deadline =
    std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_sec);

  while (std::chrono::steady_clock::now() < deadline) {
    geometry_msgs::msg::PoseStamped pose;
    if (tracker.getLatestPose(pose)) {
      if (rclcpp::Time(pose.header.stamp) >= not_before) {
        RCLCPP_INFO(node->get_logger(),
          "Wrist cam refined pose: [x=%.3f, y=%.3f, z=%.3f]",
          pose.pose.position.x, pose.pose.position.y, pose.pose.position.z);
        return pose;
      }
    }
    std::this_thread::sleep_for(100ms);
  }
  RCLCPP_WARN(node->get_logger(),
    "No wrist cam pose in %.1f s — falling back to chest cam", timeout_sec);
  return std::nullopt;
}

// ─────────────────────────────────────────────────────────────────────────────
// BYPASSED: callAlignService — kept for future re-enablement
// ─────────────────────────────────────────────────────────────────────────────
#if 0
static bool callAlignService(
  const rclcpp::Node::SharedPtr & node,
  const std::string & service_name,
  double timeout_sec,
  double & out_cube_width_m)
{
  out_cube_width_m = 0.04;
  auto client = node->create_client<std_srvs::srv::Trigger>(service_name);
  if (!client->wait_for_service(std::chrono::duration<double>(timeout_sec / 2.0))) {
    RCLCPP_ERROR(node->get_logger(), "Service '%s' not available", service_name.c_str());
    return false;
  }
  auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
  auto future  = client->async_send_request(request);
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration<double>(timeout_sec);
  while (std::chrono::steady_clock::now() < deadline) {
    if (future.wait_for(100ms) == std::future_status::ready) {
      auto response = future.get();
      if (response->success) {
        try { out_cube_width_m = std::stod(response->message); }
        catch (...) {}
      }
      return response->success;
    }
    std::this_thread::sleep_for(50ms);
  }
  RCLCPP_ERROR(node->get_logger(), "Align service '%s' TIMEOUT", service_name.c_str());
  return false;
}
#endif  // 0 — callAlignService bypassed

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>("wrist_servo_pick_and_place");

  // Background executor — keeps subscription callbacks alive while main
  // thread runs the sequential pick-and-place logic.
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  auto spinner = std::thread([&executor]() { executor.spin(); });

  Commander      commander(node);
  CubePoseTracker  chest_tracker(node);
  WristPoseTracker wrist_tracker(node);

  RCLCPP_INFO(node->get_logger(), "════════════════════════════════════════");
  RCLCPP_INFO(node->get_logger(), " PICK AND PLACE — START (no visual servoing)");
  RCLCPP_INFO(node->get_logger(), "════════════════════════════════════════");

  // ── Phase 0: Home ─────────────────────────────────────────────────────────
  RCLCPP_INFO(node->get_logger(), "Phase 0: Homing both arms");
  commander.goToNamedTarget("hands_up", "right");
  stepDelay(node, "right arm homed");
  commander.goToNamedTarget("hands_up", "left");
  stepDelay(node, "left arm homed");
  commander.openGripper("right");
  if (!commander.lastCommandSucceeded()) {
    abortToHome(node, commander, spinner, "Phase 0", "right gripper open failed");
  }
  stepDelay(node, "grippers opened — ready");
  RCLCPP_INFO(node->get_logger(), "Phase 0 DONE");

  // ── Phase 1: Chest camera cube detection ──────────────────────────────────
  RCLCPP_INFO(node->get_logger(), "Phase 1: Waiting for chest cam cube detection");
  const auto phase1_start = node->now();
  auto cube_pose_opt = waitForCubePose(node, chest_tracker, CUBE_WAIT_TIMEOUT_SEC, phase1_start);
  if (!cube_pose_opt.has_value()) {
    RCLCPP_ERROR(node->get_logger(), "Phase 1 FAILED — no cube detected. Aborting.");
    commander.goToNamedTarget("hands_up", "right");
    commander.goToNamedTarget("hands_up", "left");
    rclcpp::shutdown();
    spinner.join();
    return 1;
  }
  const auto & chest_pose = cube_pose_opt.value();
  RCLCPP_INFO(node->get_logger(),
    "Phase 1 DONE — cube at [x=%.3f, y=%.3f, z=%.3f] (chest cam)",
    chest_pose.pose.position.x, chest_pose.pose.position.y, chest_pose.pose.position.z);

  // ── Phase 2: Right arm → hover above cube (wrist cam can see marker) ───────
  RCLCPP_INFO(node->get_logger(), "Phase 2: Right arm to pre-grasp hover");
  auto hover_pose = commander.makePose(
    chest_pose.pose.position.x + WRIST_CAM_LOOK_X_OFFSET,
    chest_pose.pose.position.y + WRIST_CAM_LOOK_Y_OFFSET,
    chest_pose.pose.position.z + WRIST_CAM_HOVER_HEIGHT,
    HANDS_UP_QX, HANDS_UP_QY, HANDS_UP_QZ, HANDS_UP_QW);
  RCLCPP_INFO(node->get_logger(),
    "Phase 2: hover → [x=%.3f, y=%.3f, z=%.3f]",
    hover_pose.position.x, hover_pose.position.y, hover_pose.position.z);
  commander.moveToPose(hover_pose, "right");
  if (!commander.lastCommandSucceeded()) {
    abortToHome(node, commander, spinner, "Phase 2", "planning failed for hover pose");
  }
  stepDelay(node, "right arm at hover — waiting for wrist cam pose");
  RCLCPP_INFO(node->get_logger(), "Phase 2 DONE");

  // ── Phase 3: Wrist cam refined pose → grasp → lift ────────────────────────
  RCLCPP_INFO(node->get_logger(), "Phase 3: Wrist cam refined pose → grasp");

  // Wait for a fresh wrist cam pose published AFTER hover reached
  const auto hover_time = node->now();
  auto wrist_pose_opt = waitForWristPose(node, wrist_tracker, WRIST_POSE_TIMEOUT_SEC, hover_time);

  // Use wrist cam pose when available; abort if not
  geometry_msgs::msg::PoseStamped grasp_src;
  if (wrist_pose_opt.has_value()) {
    grasp_src = wrist_pose_opt.value();
    RCLCPP_INFO(node->get_logger(), "Phase 3: Using WRIST CAM refined pose");
  } else {
    RCLCPP_ERROR(node->get_logger(), "Phase 3: Wrist cam failed to detect marker. Aborting.");
    abortToHome(node, commander, spinner, "Phase 3", "wrist cam marker detection failed");
    return 1;
  }

  

  // Compute grasp pose from detected marker position
  auto grasp_pose = commander.makePose(
    grasp_src.pose.position.x + GRASP_OFFSET_X,
    grasp_src.pose.position.y + GRASP_OFFSET_Y,
    grasp_src.pose.position.z + GRASP_OFFSET_Z,
    HANDS_UP_QX, HANDS_UP_QY, HANDS_UP_QZ, HANDS_UP_QW);
  RCLCPP_INFO(node->get_logger(),
    "Phase 3: Grasp target [x=%.3f, y=%.3f, z=%.3f]",
    grasp_pose.position.x, grasp_pose.position.y, grasp_pose.position.z);

  // Attach cube to right gripper BEFORE planning the grasp move.
  // This removes the cube from MoveIt's collision world so the planner
  // can find a path that ends at the cube's position without treating it
  // as an obstacle. Without this, planning times out (goal in collision).
  // commander.attachCubeToGripper("detected_cube", "right");
  // stepDelay(node, "cube removed from collision world — planning grasp path");

  // Plan and execute grasp move (cube no longer an obstacle)
  commander.moveCartesianToPose(grasp_pose, "right");
  if (!commander.lastCommandSucceeded()) {
    abortToHome(node, commander, spinner, "Phase 3", "grasp moveCartesianToPose failed");
  }
  stepDelay(node, "right arm at grasp pose — closing gripper");

  // Close gripper fully — MuJoCo physics stops fingers when they contact the cube.
  // Using closeGripper (named target joint=0) instead of setGripperWidth because:
  //   right gripper closes toward joint=0, setGripperWidth always sent +0.111
  //   which is out of range [-0.7, 0] for right arm → was clamped to 0 anyway.
  commander.closeGripper("right");
  if (!commander.lastCommandSucceeded()) {
    abortToHome(node, commander, spinner, "Phase 3", "right gripper close failed");
  }
  stepDelay(node, "right gripper closed on cube");

   auto left_handover_lift = commander.makePose(
    L_HAND_X, L_HAND_Y, L_HAND_Z + LIFT_HEIGHT,
    L_HAND_QX, L_HAND_QY, L_HAND_QZ, L_HAND_QW);
  commander.moveToPose(left_handover_lift, "left");
  if (!commander.lastCommandSucceeded()) {
    abortToHome(node, commander, spinner, "Phase 3b", "left PRE-approaches moveToPose failed");
  }
  commander.openGripper("left");
  if (!commander.lastCommandSucceeded()) {
    abortToHome(node, commander, spinner, "Phase 3b", "left gripper open failed");
  }
  stepDelay(node, "left gripper opened");

  // ── Phase 4: Right arm → handover pose (Pose 2) ───────────────────────────
  RCLCPP_INFO(node->get_logger(), "Phase 4: Right arm to handover pose (Pose 2)");
  auto right_handover = commander.makePose(
    R_HAND_X, R_HAND_Y, R_HAND_Z,
    R_HAND_QX, R_HAND_QY, R_HAND_QZ, R_HAND_QW);
  commander.moveToPose(right_handover, "right");
  if (!commander.lastCommandSucceeded()) {
    abortToHome(node, commander, spinner, "Phase 4", "handover moveToPose failed");
  }
  stepDelay(node, "right arm at handover");
  RCLCPP_INFO(node->get_logger(), "Phase 4 DONE");

  // ── Phase 5: Left arm → handover approach (Pose 3) ───────────────────────
  RCLCPP_INFO(node->get_logger(), "Phase 5: Left arm to handover approach (Pose 3)");
  auto left_handover = commander.makePose(
    L_HAND_X + L_HAND_CORR_X, L_HAND_Y + L_HAND_CORR_Y, L_HAND_Z + L_HAND_CORR_Z,
    L_HAND_QX, L_HAND_QY, L_HAND_QZ, L_HAND_QW);
  commander.moveToPose(left_handover, "left");
  if (!commander.lastCommandSucceeded()) {
    abortToHome(node, commander, spinner, "Phase 5", "left approach moveToPose failed");
  }
  stepDelay(node, "left arm at handover approach");
  RCLCPP_INFO(node->get_logger(), "Phase 5 DONE");

  // ── Phase 6: Transfer — left closes, right opens, left places ─────────────
  RCLCPP_INFO(node->get_logger(), "Phase 6: Handover transfer");

  // Left gripper closes on cube
  // Left closes fully on cube — physics stops fingers at cube surface.
  commander.closeGripper("left");
  if (!commander.lastCommandSucceeded()) {
    abortToHome(node, commander, spinner, "Phase 6", "left gripper close failed");
  }
  stepDelay(node, "left gripper closed on cube");

  // Transfer cube attachment: now part of left arm in MoveIt scene
  // commander.attachCubeToGripper("detected_cube", "left");
  // commander.detachCubeFromGripper("detected_cube", "right");
  stepDelay(node, "cube transferred — attached to left, released from right");

  // Right releases and retreats
  commander.openGripper("right");
  if (!commander.lastCommandSucceeded()) {
    abortToHome(node, commander, spinner, "Phase 6", "right gripper open failed");
  }
  stepDelay(node, "right gripper released");
  
  // Left arm to place pose
  auto place_pose = commander.makePose(
    PLACE_X, PLACE_Y, PLACE_Z,
    PLACE_QX, PLACE_QY, PLACE_QZ, PLACE_QW);
  commander.moveToPose(place_pose, "left");
  if (!commander.lastCommandSucceeded()) {
    abortToHome(node, commander, spinner, "Phase 6", "place moveToPose failed");
  }
  stepDelay(node, "left arm at place pose");

  // Detach cube before opening — puts it back in world at current pose
  // commander.detachCubeFromGripper("detected_cube", "left");
  commander.openGripper("left");
  if (!commander.lastCommandSucceeded()) {
    abortToHome(node, commander, spinner, "Phase 6", "left gripper open (place) failed");
  }
  stepDelay(node, "left gripper opened — cube placed");
  RCLCPP_INFO(node->get_logger(), "Phase 6 DONE — cube placed");

  // ── Final: Home both arms ─────────────────────────────────────────────────
  RCLCPP_INFO(node->get_logger(), "Final: Returning home");
  commander.goToNamedTarget("hands_up", "right");
  stepDelay(node, "right arm homed");
  commander.goToNamedTarget("hands_up", "left");

  RCLCPP_INFO(node->get_logger(), "════════════════════════════════════════");
  RCLCPP_INFO(node->get_logger(), " ═══ PICK AND PLACE COMPLETE ═══");
  RCLCPP_INFO(node->get_logger(), "════════════════════════════════════════");

  rclcpp::shutdown();
  spinner.join();
  return 0;
}
