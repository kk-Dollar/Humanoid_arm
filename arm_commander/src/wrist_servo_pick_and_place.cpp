// PERCEPTION PIPELINE — dynamic bimanual handover pick-and-place
// Part of arm_commander package
//
// Pipeline:
//   0  Home both arms, open right gripper
//   1  Wait for chest camera cube detection (/cube_pose)
//   2  Right arm → pre-grasp hover above cube
//   3  Right wrist cam refined pose → grasp → close right gripper
//   4  Right arm → DYNAMIC handover pose
//        · Query left arm live EE pose via getCurrentEEPose("left")
//        · Target = left_EE + [HANDOVER_R_OFFSET_*] with sideways orientation
//        · On IK failure: retry with enlarged gap, then abort
//   5  Left arm → DYNAMIC scanning position (left wrist cam sees cube)
//        · Target = right_EE (≈ cube position) + [L_SCAN_OFFSET_*]
//        · Open left gripper
//        · Wait for fresh /left_wrist_cube_pose (LEFT wrist cam detects cube)
//   5b Left arm → precise grasp from wrist-cam pose
//        · Target = left_wrist_cube_pose + [L_GRASP_OFFSET_*]
//        · Fallback: right_EE + offset when wrist cam times out
//   6  Left closes → right opens → right retreats → left places → left opens
//   Final: home both arms
//
// Why dynamic?  Fixed handover constants only work when both arms land at a
// specific IK solution.  With the dynamic approach the right arm always meets
// the left arm wherever it actually is, removing brittle hand-tuned offsets.

/**
 * wrist_servo_pick_and_place.cpp
 *
 * Key design decisions
 * ────────────────────
 * • Right wrist cam cannot see the cube once the gripper is closed around it.
 *   After grasp we stop using the right wrist tracker.
 *
 * • Left wrist cam IS used during handover.  The left arm moves to a scanning
 *   position relative to the right EE (where the cube is), its wrist cam
 *   sees the cube, and the published /left_wrist_cube_pose drives the final
 *   left approach.  This gives sub-centimetre accuracy without fixed offsets.
 *
 * • Both wrist servo nodes now publish to separate topics:
 *     /right_wrist_cube_pose  (right wrist servo)
 *     /left_wrist_cube_pose   (left wrist servo)
 *   Previously both wrote to /wrist_cube_pose and overwrote each other.
 *
 * Tuning constants (search TUNE):
 *   HANDOVER_R_OFFSET_Y   — gap between arms at handover (~0.30 m nominal)
 *   L_SCAN_OFFSET_Y       — left scan position relative to cube (+Y side)
 *   L_GRASP_OFFSET_*      — left grasp target relative to wrist-cam cube pose
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

// ── Right arm handover orientation (sideways — presents cube to left arm) ──────
// Taken from previously hand-tuned R_HAND pose which was already sideways-facing.
static constexpr double R_HAND_QX =  0.495;
static constexpr double R_HAND_QY = -0.502;
static constexpr double R_HAND_QZ =  0.461;
static constexpr double R_HAND_QW =  0.539;

// ── Left arm receive orientation (fixed for now) ───────────────────────────────
// Taken from previously hand-tuned L_HAND pose.  Orientation only; position is
// computed dynamically from left wrist cam cube detection.
static constexpr double L_HAND_QX = -0.477;
static constexpr double L_HAND_QY = -0.517;
static constexpr double L_HAND_QZ = -0.444;
static constexpr double L_HAND_QW =  0.555;

// ── Grasp offsets (applied on top of detected cube position) ──────────────────
static constexpr double GRASP_OFFSET_X =  0.11;   // m
static constexpr double GRASP_OFFSET_Y =  0.0;    // m
static constexpr double GRASP_OFFSET_Z =  0.14;   // m — EE above cube top

// ── Hover height above cube for right wrist cam visibility ────────────────────
static constexpr double WRIST_CAM_LOOK_X_OFFSET = 0.0;   // m
static constexpr double WRIST_CAM_LOOK_Y_OFFSET = 0.0;   // m
static constexpr double WRIST_CAM_HOVER_HEIGHT  = 0.21;  // m above cube top

// ── Dynamic handover offsets ──────────────────────────────────────────────────
// Right arm target position = left_EE_position + HANDOVER_R_OFFSET_*
// Right arm is on -Y side, left arm is on +Y side of the robot body.
// TUNE: increase HANDOVER_R_OFFSET_Y (more negative) if arms collide,
//       decrease (less negative) if cube is too far from left wrist cam FOV.
static constexpr double HANDOVER_R_OFFSET_X =  0.0;    // m — no X shift
static constexpr double HANDOVER_R_OFFSET_Y = -0.40;   // m
static constexpr double HANDOVER_R_OFFSET_Z =  0.0;    // m — same height

// Fallback: if exact offset unreachable, enlarge gap by this factor and retry.
static constexpr double HANDOVER_R_FALLBACK_SCALE = 1.3;  // 30 % more gap

// ── Left arm scanning position offset from right EE (≈ cube position) ─────────
// Left arm moves here so its wrist cam can see the cube in the right gripper.
// GEOMETRY NOTE: The left arm must be ~30-36 cm to +Y of the right arm EE
// (same geometry as grasping). Scanning is from near the left arm home position.
// At ~32 cm separation the left wrist cam points inward toward the cube.
// TUNE: increase if wrist cam misses cube, decrease if arm reaches collision limit.
static constexpr double L_SCAN_OFFSET_X =  0.0;    // m
static constexpr double L_SCAN_OFFSET_Y =  0.36;   // m — left arm scans near home (36 cm +Y of cube)
static constexpr double L_SCAN_OFFSET_Z = -0.10;   // m — wrist LOWER than cube so camera above looks at cube

// Fallback scan offset if primary is unreachable.
static constexpr double L_SCAN_OFFSET_Y_FALLBACK = 0.40;  // m — further fallback

// ── Left arm grasp offset (applied to wrist-cam cube world pose) ───────────────
// Old working value: left EE y=+0.114 when right EE y=-0.210 → gap=0.324 m.
// The gripper FINGERS reach across the gap; the EE itself stays far from right arm.
// TUNE: decrease to move left arm closer to cube (but check IK and collision).
static constexpr double L_GRASP_OFFSET_X =  0.0;    // m
static constexpr double L_GRASP_OFFSET_Y =  0.32;   // m — ~32 cm +Y of right EE (gripper fingers reach cube)
static constexpr double L_GRASP_OFFSET_Z = +0.02;   // m — left EE 2 cm above right EE (matches old working geometry)

// ── Place pose ────────────────────────────────────────────────────────────────
static constexpr double PLACE_X   =  0.176, PLACE_Y   =  0.350, PLACE_Z   = 0.353;
static constexpr double PLACE_QX  =  0.026, PLACE_QY  = -0.633, PLACE_QZ  = -0.014;
static constexpr double PLACE_QW  =  0.773;

// ── Timing ────────────────────────────────────────────────────────────────────
static constexpr int    STEP_DELAY_MS              = 800;
static constexpr double CUBE_WAIT_TIMEOUT_SEC      = 30.0;
static constexpr double WRIST_POSE_TIMEOUT_SEC     =  5.0;   // right wrist (Phase 3)
static constexpr double LEFT_WRIST_POSE_TIMEOUT_SEC =  5.0;  // left  wrist (Phase 5)

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
// CubePoseTracker — caches latest /cube_pose (chest cam)
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
// WristPoseTracker — caches latest pose from a wrist camera topic.
// Instantiate once per arm:
//   WristPoseTracker right_wrist(node, "/right_wrist_cube_pose");
//   WristPoseTracker left_wrist (node, "/left_wrist_cube_pose");
// ─────────────────────────────────────────────────────────────────────────────
class WristPoseTracker
{
public:
  WristPoseTracker(const rclcpp::Node::SharedPtr & node, const std::string & topic)
  {
    sub_ = node->create_subscription<geometry_msgs::msg::PoseStamped>(
      topic, 10,
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
// waitForWristPose — block until the given wrist tracker has a pose newer than
// not_before.  Works for both left and right — pass a human-readable label.
// ─────────────────────────────────────────────────────────────────────────────
static std::optional<geometry_msgs::msg::PoseStamped>
waitForWristPose(
  const rclcpp::Node::SharedPtr & node,
  const WristPoseTracker & tracker,
  double timeout_sec,
  const rclcpp::Time & not_before,
  const std::string & label = "wrist")
{
  const auto deadline =
    std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_sec);

  while (std::chrono::steady_clock::now() < deadline) {
    geometry_msgs::msg::PoseStamped pose;
    if (tracker.getLatestPose(pose)) {
      if (rclcpp::Time(pose.header.stamp) >= not_before) {
        RCLCPP_INFO(node->get_logger(),
          "%s cam cube pose: [x=%.3f, y=%.3f, z=%.3f]",
          label.c_str(),
          pose.pose.position.x, pose.pose.position.y, pose.pose.position.z);
        return pose;
      }
    }
    std::this_thread::sleep_for(100ms);
  }
  RCLCPP_WARN(node->get_logger(),
    "No %s cam pose in %.1f s — will use fallback", label.c_str(), timeout_sec);
  return std::nullopt;
}

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
  CubePoseTracker chest_tracker(node);

  // Right wrist cam: used in Phase 3 to refine the grasp pose.
  // After the gripper closes the right wrist cam can no longer see the cube.
  WristPoseTracker right_wrist_tracker(node, "/right_wrist_cube_pose");

  // Left wrist cam: used in Phase 5 to see the cube held by the right arm
  // so the left arm can approach with sub-centimetre accuracy.
  WristPoseTracker left_wrist_tracker(node,  "/left_wrist_cube_pose");

  RCLCPP_INFO(node->get_logger(), "════════════════════════════════════════");
  RCLCPP_INFO(node->get_logger(), " PICK AND PLACE — DYNAMIC HANDOVER");
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

  // ── Phase 3: Right wrist cam refined pose → grasp ─────────────────────────
  RCLCPP_INFO(node->get_logger(), "Phase 3: Right wrist cam refined pose → grasp");

  const auto hover_time = node->now();
  auto right_wrist_opt = waitForWristPose(
    node, right_wrist_tracker, WRIST_POSE_TIMEOUT_SEC, hover_time, "right wrist");

  if (!right_wrist_opt.has_value()) {
    RCLCPP_ERROR(node->get_logger(), "Phase 3: Right wrist cam failed to detect marker. Aborting.");
    abortToHome(node, commander, spinner, "Phase 3", "right wrist cam marker detection failed");
    return 1;
  }

  const auto & grasp_src = right_wrist_opt.value();
  auto grasp_pose = commander.makePose(
    grasp_src.pose.position.x + GRASP_OFFSET_X,
    grasp_src.pose.position.y + GRASP_OFFSET_Y,
    grasp_src.pose.position.z + GRASP_OFFSET_Z,
    HANDS_UP_QX, HANDS_UP_QY, HANDS_UP_QZ, HANDS_UP_QW);
  RCLCPP_INFO(node->get_logger(),
    "Phase 3: Grasp target [x=%.3f, y=%.3f, z=%.3f]",
    grasp_pose.position.x, grasp_pose.position.y, grasp_pose.position.z);

  commander.moveCartesianToPose(grasp_pose, "right");
  if (!commander.lastCommandSucceeded()) {
    abortToHome(node, commander, spinner, "Phase 3", "grasp moveCartesianToPose failed");
  }
  stepDelay(node, "right arm at grasp pose — closing gripper");

  commander.closeGripper("right");
  if (!commander.lastCommandSucceeded()) {
    abortToHome(node, commander, spinner, "Phase 3", "right gripper close failed");
  }
  stepDelay(node, "right gripper closed on cube");
  RCLCPP_INFO(node->get_logger(), "Phase 3 DONE — cube grasped");
  // NOTE: right wrist cam can no longer see the cube once the gripper is closed.
  //       From here we rely on getCurrentEEPose() and the left wrist cam.

  // ── Phase 4: Dynamic right arm → handover position ────────────────────────
  // Query where the left arm EE currently is, then position the right arm so
  // the cube ends up close enough for the left wrist cam to detect it.
  RCLCPP_INFO(node->get_logger(), "Phase 4: Right arm → dynamic handover pose");
  {
    auto left_ee = commander.getCurrentEEPose("left");
    RCLCPP_INFO(node->get_logger(),
      "Phase 4: Left arm EE at [x=%.3f, y=%.3f, z=%.3f]",
      left_ee.position.x, left_ee.position.y, left_ee.position.z);

    // Primary target: left_EE + configurable offset.
    // The offset keeps a safe gap while putting the cube in the left wrist cam FOV.
    auto right_handover = commander.makePose(
      left_ee.position.x + HANDOVER_R_OFFSET_X,
      left_ee.position.y + HANDOVER_R_OFFSET_Y,
      left_ee.position.z + HANDOVER_R_OFFSET_Z,
      R_HAND_QX, R_HAND_QY, R_HAND_QZ, R_HAND_QW);

    RCLCPP_INFO(node->get_logger(),
      "Phase 4: Right handover target [x=%.3f, y=%.3f, z=%.3f]",
      right_handover.position.x, right_handover.position.y, right_handover.position.z);

    commander.moveToPose(right_handover, "right");

    if (!commander.lastCommandSucceeded()) {
      // IK might fail if the primary target is at the edge of the workspace.
      // Enlarge the gap slightly and retry — this pulls the target inward.
      RCLCPP_WARN(node->get_logger(),
        "Phase 4: Primary target unreachable — retrying with %.0f%% larger gap",
        (HANDOVER_R_FALLBACK_SCALE - 1.0) * 100.0);

      auto fallback = commander.makePose(
        left_ee.position.x + HANDOVER_R_OFFSET_X,
        left_ee.position.y + HANDOVER_R_OFFSET_Y * HANDOVER_R_FALLBACK_SCALE,
        left_ee.position.z + HANDOVER_R_OFFSET_Z,
        R_HAND_QX, R_HAND_QY, R_HAND_QZ, R_HAND_QW);

      commander.moveToPose(fallback, "right");
      if (!commander.lastCommandSucceeded()) {
        abortToHome(node, commander, spinner, "Phase 4",
          "right handover pose unreachable even with fallback gap");
      }
    }
  }
  stepDelay(node, "right arm at handover position — cube ready for left wrist cam");
  RCLCPP_INFO(node->get_logger(), "Phase 4 DONE");

  // ── Phase 5: Left arm → scanning position + left wrist cam detection ───────
  // The left arm moves to a position near the cube (currently in right gripper)
  // so its wrist cam can see the cube and publish an accurate world pose.
  RCLCPP_INFO(node->get_logger(), "Phase 5: Left arm → scanning position for left wrist cam");
  geometry_msgs::msg::Pose left_grasp_target;
  {
    // Cube is approximately at the right arm EE.
    auto right_ee = commander.getCurrentEEPose("right");
    RCLCPP_INFO(node->get_logger(),
      "Phase 5: Right EE (≈ cube) at [x=%.3f, y=%.3f, z=%.3f]",
      right_ee.position.x, right_ee.position.y, right_ee.position.z);

    // Open left gripper before moving so there is no risk of contact.
    commander.openGripper("left");
    if (!commander.lastCommandSucceeded()) {
      abortToHome(node, commander, spinner, "Phase 5", "left gripper open failed");
    }

    // Move left arm to scanning position: slightly to the +Y side of the cube
    // with the fixed receive orientation so the wrist cam faces the cube.
    auto left_scan = commander.makePose(
      right_ee.position.x + L_SCAN_OFFSET_X,
      right_ee.position.y + L_SCAN_OFFSET_Y,
      right_ee.position.z + L_SCAN_OFFSET_Z,
      L_HAND_QX, L_HAND_QY, L_HAND_QZ, L_HAND_QW);

    RCLCPP_INFO(node->get_logger(),
      "Phase 5: Left scan target [x=%.3f, y=%.3f, z=%.3f]",
      left_scan.position.x, left_scan.position.y, left_scan.position.z);

    commander.moveToPose(left_scan, "left");
    if (!commander.lastCommandSucceeded()) {
      // Primary scan target unreachable — try wider offset (pulls further from right arm).
      RCLCPP_WARN(node->get_logger(),
        "Phase 5: Primary scan target unreachable — retrying with %.0f cm gap",
        L_SCAN_OFFSET_Y_FALLBACK * 100.0);
      auto left_scan_fb = commander.makePose(
        right_ee.position.x + L_SCAN_OFFSET_X,
        right_ee.position.y + L_SCAN_OFFSET_Y_FALLBACK,
        right_ee.position.z + L_SCAN_OFFSET_Z,
        L_HAND_QX, L_HAND_QY, L_HAND_QZ, L_HAND_QW);
      commander.moveToPose(left_scan_fb, "left");
      if (!commander.lastCommandSucceeded()) {
        abortToHome(node, commander, spinner, "Phase 5",
          "left scan moveToPose failed even with fallback offset");
      }
    }
    stepDelay(node, "left arm at scanning position");

    // Wait for left wrist cam to see the cube and publish a world-frame pose.
    const auto scan_time = node->now();
    auto left_wrist_opt = waitForWristPose(
      node, left_wrist_tracker, LEFT_WRIST_POSE_TIMEOUT_SEC, scan_time, "left wrist");

    if (left_wrist_opt.has_value()) {
      // Wrist cam gives X refinement (lateral position across the arm axis).
      // Y and Z are taken from right arm EE instead of wrist cam because:
      //   - Wrist cam Y has significant parallax error from the 36 cm viewing angle.
      //   - Wrist cam Z reports the cube BOTTOM (z≈0.283, table level) not grasp height.
      // Right arm EE Y/Z are accurate because the right arm is physically holding the cube.
      const auto & cp = left_wrist_opt.value().pose.position;
      RCLCPP_INFO(node->get_logger(),
        "Phase 5: Left wrist cam sees cube at [x=%.3f, y=%.3f, z=%.3f] — using X for refinement",
        cp.x, cp.y, cp.z);
      left_grasp_target = commander.makePose(
        cp.x + L_GRASP_OFFSET_X,                   // X from wrist cam — lateral refinement
        right_ee.position.y + L_GRASP_OFFSET_Y,     // Y from right EE — reliable (parallax-free)
        right_ee.position.z + L_GRASP_OFFSET_Z,     // Z from right EE — reliable (not cube bottom)
        L_HAND_QX, L_HAND_QY, L_HAND_QZ, L_HAND_QW);
    } else {
      // Fallback: use right EE position + grasp offset.
      // Less accurate, but avoids full abort if wrist cam misses.
      RCLCPP_WARN(node->get_logger(),
        "Phase 5: Left wrist cam timeout — falling back to right EE + offset");
      left_grasp_target = commander.makePose(
        right_ee.position.x + L_GRASP_OFFSET_X,
        right_ee.position.y + L_GRASP_OFFSET_Y,
        right_ee.position.z + L_GRASP_OFFSET_Z,
        L_HAND_QX, L_HAND_QY, L_HAND_QZ, L_HAND_QW);
    }
  }
  RCLCPP_INFO(node->get_logger(), "Phase 5 DONE");

  // ── Phase 5b: Left arm → precise grasp ────────────────────────────────────
  RCLCPP_INFO(node->get_logger(), "Phase 5b: Left arm → precise grasp position");
  RCLCPP_INFO(node->get_logger(),
    "Phase 5b: Left grasp target [x=%.3f, y=%.3f, z=%.3f]",
    left_grasp_target.position.x, left_grasp_target.position.y, left_grasp_target.position.z);

  commander.moveToPose(left_grasp_target, "left");
  if (!commander.lastCommandSucceeded()) {
    abortToHome(node, commander, spinner, "Phase 5b", "left grasp moveToPose failed");
  }
  stepDelay(node, "left arm at grasp position — ready to close");
  RCLCPP_INFO(node->get_logger(), "Phase 5b DONE");

  // ── Phase 6: Transfer ─────────────────────────────────────────────────────
  RCLCPP_INFO(node->get_logger(), "Phase 6: Handover transfer");

  // Left closes on cube.
  commander.closeGripper("left");
  if (!commander.lastCommandSucceeded()) {
    abortToHome(node, commander, spinner, "Phase 6", "left gripper close failed");
  }
  stepDelay(node, "left gripper closed on cube");

  // Right releases the cube.
  commander.openGripper("right");
  if (!commander.lastCommandSucceeded()) {
    abortToHome(node, commander, spinner, "Phase 6", "right gripper open failed");
  }
  stepDelay(node, "right gripper released — cube transferred to left");

  // Left arm carries cube to the place pose.
  auto place_pose = commander.makePose(
    PLACE_X, PLACE_Y, PLACE_Z,
    PLACE_QX, PLACE_QY, PLACE_QZ, PLACE_QW);
  commander.moveToPose(place_pose, "left");
  if (!commander.lastCommandSucceeded()) {
    abortToHome(node, commander, spinner, "Phase 6", "place moveToPose failed");
  }
  stepDelay(node, "left arm at place pose");

  // Place the cube.
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
