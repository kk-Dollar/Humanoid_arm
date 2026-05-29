// NEW FILE — created for ArUco perception pipeline
// Does NOT replace or modify any existing file

#include "arm_commander/commander.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>

#include <chrono>
#include <cmath>
#include <mutex>
#include <optional>
#include <thread>

using namespace std::chrono_literals;

// Grasp geometry
static constexpr double GRASP_APPROACH_OFFSET = 0.05;
static constexpr double HANDOVER_LIFT_Z = 0.10;
static constexpr int STEP_DELAY_MS = 1000;
static constexpr int SHORT_STEP_DELAY_MS = 500;
static constexpr int WAIT_TIMEOUT_SEC = 30;
static constexpr int WAIT_LOG_PERIOD_SEC = 2;

// Fixed world-frame place pose
static constexpr double PLACE_X = 0.181;
static constexpr double PLACE_Y = 0.259;
static constexpr double PLACE_Z = 0.436;
static constexpr double PLACE_QX = 0.045;
static constexpr double PLACE_QY = -0.749;
static constexpr double PLACE_QZ = -0.016;
static constexpr double PLACE_QW = 0.661;

// Fixed world-frame right arm handover pose
static constexpr double R_HAND_X = 0.254;
static constexpr double R_HAND_Y = -0.062;
static constexpr double R_HAND_Z = 0.465;
static constexpr double R_HAND_QX = 0.492;
static constexpr double R_HAND_QY = -0.462;
static constexpr double R_HAND_QZ = -0.445;
static constexpr double R_HAND_QW = 0.589;

// Fixed world-frame left arm handover receive pose
static constexpr double L_HAND_X = 0.237;
static constexpr double L_HAND_Y = 0.051;
static constexpr double L_HAND_Z = 0.436;
static constexpr double L_HAND_QX = 0.517;
static constexpr double L_HAND_QY = 0.544;
static constexpr double L_HAND_QZ = -0.473;
static constexpr double L_HAND_QW = -0.462;

// Fixed world-frame right arm retreat pose
static constexpr double R_RET_X = 0.278;
static constexpr double R_RET_Y = -0.190;
static constexpr double R_RET_Z = 0.479;
static constexpr double R_RET_QX = 0.492;
static constexpr double R_RET_QY = -0.462;
static constexpr double R_RET_QZ = -0.445;
static constexpr double R_RET_QW = 0.589;

class CubePoseTracker
{
public:
  explicit CubePoseTracker(const rclcpp::Node::SharedPtr & node)
  {
    // Subscription: cube world pose flowing from aruco_detector to manipulator.
    cube_pose_sub_ = node->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/cube_pose", 10,
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_cube_pose_ = *msg;
      });

    // Subscription: cube visibility state flowing from aruco_detector.
    cube_detected_sub_ = node->create_subscription<std_msgs::msg::Bool>(
      "/cube_detected", 10,
      [this](const std_msgs::msg::Bool::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        cube_detected_ = msg->data;
      });
  }

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
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr cube_detected_sub_;
  mutable std::mutex mutex_;
  std::optional<bool> cube_detected_;
  std::optional<geometry_msgs::msg::PoseStamped> last_cube_pose_;
};

/**
 * WHAT: Sleep helper between manipulation steps.
 * WHY: Adds settle time for simulation/controllers between commands.
 * INPUT: Delay in milliseconds.
 * OUTPUT: Blocking delay only.
 */
static void stepDelay(int delay_ms)
{
  std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
}

/**
 * WHAT: Waits for a valid detected cube pose from perception topics.
 * WHY: Manipulation should begin only after reliable visual detection.
 * INPUT: ROS node, tracker object, timeout seconds.
 * OUTPUT: Optional PoseStamped; empty if timeout occurs.
 */
static std::optional<geometry_msgs::msg::PoseStamped> waitForCubePose(
  const rclcpp::Node::SharedPtr & node,
  const CubePoseTracker & tracker,
  int timeout_sec)
{
  const auto start = node->now();
  auto next_log = start;

  while ((node->now() - start).seconds() < timeout_sec) {
    geometry_msgs::msg::PoseStamped cube_pose;
    if (tracker.getLatestDetectedPose(cube_pose)) {
      return cube_pose;
    }
    if (node->now() >= next_log) {
      RCLCPP_INFO(node->get_logger(), "Waiting for cube detection...");
      next_log = node->now() + rclcpp::Duration::from_seconds(WAIT_LOG_PERIOD_SEC);
    }
    stepDelay(100);
  }

  RCLCPP_ERROR(node->get_logger(), "Cube detection timeout after %d seconds", timeout_sec);
  return std::nullopt;
}

/**
 * WHAT: Computes grasp target pose for right arm from detected cube pose.
 * WHY: Chest camera provides the cube pose, while the arm keeps a fixed grasp orientation.
 * INPUT: Commander for pose construction and cube PoseStamped.
 * OUTPUT: Pose for right gripper at grasp height.
 */
static geometry_msgs::msg::Pose computeRightPickPose(
  Commander & commander,
  const geometry_msgs::msg::PoseStamped & cube_pose)
{
  const double pick_x = cube_pose.pose.position.x;
  const double pick_y = cube_pose.pose.position.y;
  const double pick_z = cube_pose.pose.position.z;
  return commander.makePose(pick_x, pick_y, pick_z, 0.0, -M_PI_2, M_PI_2);
}

/**
 * WHAT: Computes pre-grasp approach pose above the chest-detected cube pose.
 * WHY: A short vertical approach is enough when the grasp orientation is fixed.
 * INPUT: Commander for pose construction and cube PoseStamped.
 * OUTPUT: Pose for right gripper above grasp by GRASP_APPROACH_OFFSET.
 */
static geometry_msgs::msg::Pose computeRightApproachPose(
  Commander & commander,
  const geometry_msgs::msg::PoseStamped & cube_pose)
{
  const double pick_x = cube_pose.pose.position.x;
  const double pick_y = cube_pose.pose.position.y;
  const double pick_z = cube_pose.pose.position.z + GRASP_APPROACH_OFFSET;
  return commander.makePose(pick_x, pick_y, pick_z, 0.0, -M_PI_2, M_PI_2);
}

/**
 * WHAT: Logs MoveIt command failure in a consistent format.
 * WHY: Every failed plan/execute should be visible without crashing the node.
 * INPUT: Node logger and description of the attempted command.
 * OUTPUT: Error log entry only.
 */
static void logIfFailed(const rclcpp::Node::SharedPtr & node, Commander & commander, const char * what)
{
  if (!commander.lastCommandSucceeded()) {
    RCLCPP_ERROR(node->get_logger(), "MoveIt command failed: %s", what);
  }
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("aruco_pick_and_place");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  auto spinner = std::thread([&executor]() {executor.spin();});

  Commander commander(node);
  CubePoseTracker tracker(node);

  RCLCPP_INFO(node->get_logger(), "[Phase 0] Home start");
  commander.goToNamedTarget("hands_up", "right");
  logIfFailed(node, commander, "goToNamedTarget(hands_up, right)");
  stepDelay(STEP_DELAY_MS);
  commander.goToNamedTarget("hands_up", "left");
  logIfFailed(node, commander, "goToNamedTarget(hands_up, left)");
  stepDelay(STEP_DELAY_MS);
  RCLCPP_INFO(node->get_logger(), "[Phase 0] Home complete");

  RCLCPP_INFO(node->get_logger(), "[Phase 1] Wait for cube detection start");
  const auto cube_pose_opt = waitForCubePose(node, tracker, WAIT_TIMEOUT_SEC);
  if (!cube_pose_opt.has_value()) {
    RCLCPP_ERROR(node->get_logger(), "Cube not detected, returning to home.");
    commander.goToNamedTarget("hands_up", "right");
    commander.goToNamedTarget("hands_up", "left");
    rclcpp::shutdown();
    spinner.join();
    return 1;
  }
  const auto cube_pose = cube_pose_opt.value();
  RCLCPP_INFO(
    node->get_logger(),
    "Detected cube pose: x=%.3f y=%.3f z=%.3f",
    cube_pose.pose.position.x, cube_pose.pose.position.y, cube_pose.pose.position.z);
  RCLCPP_INFO(node->get_logger(), "[Phase 1] Wait for cube detection complete");

  RCLCPP_INFO(node->get_logger(), "[Phase 2] Compute chest-camera grasp poses start");
  const auto approach_pose = computeRightApproachPose(commander, cube_pose);
  const auto pick_pose = computeRightPickPose(commander, cube_pose);
  RCLCPP_INFO(node->get_logger(), "[Phase 2] Compute chest-camera grasp poses complete");

  RCLCPP_INFO(node->get_logger(), "[Phase 3] Right arm pick start");
  commander.openGripper("right");
  logIfFailed(node, commander, "openGripper(right)");
  stepDelay(SHORT_STEP_DELAY_MS);
  commander.moveToPose(approach_pose, "right");
  logIfFailed(node, commander, "moveToPose(approach_pose, right)");
  stepDelay(SHORT_STEP_DELAY_MS);
  commander.moveToPose(pick_pose, "right");
  logIfFailed(node, commander, "moveToPose(pick_pose, right)");
  stepDelay(SHORT_STEP_DELAY_MS);
  commander.closeGripper("right");
  logIfFailed(node, commander, "closeGripper(right)");
  stepDelay(STEP_DELAY_MS);
  commander.moveCartesianByZ("right", HANDOVER_LIFT_Z);
  logIfFailed(node, commander, "moveCartesianByZ(right, HANDOVER_LIFT_Z)");
  stepDelay(STEP_DELAY_MS);
  RCLCPP_INFO(node->get_logger(), "[Phase 3] Right arm pick complete");

  RCLCPP_INFO(node->get_logger(), "[Phase 4] Right handover move start (fixed world-frame waypoint)");
  const auto right_handover = commander.makePose(
    R_HAND_X, R_HAND_Y, R_HAND_Z, R_HAND_QX, R_HAND_QY, R_HAND_QZ, R_HAND_QW);
  commander.moveToPose(right_handover, "right");
  logIfFailed(node, commander, "moveToPose(right_handover, right)");
  stepDelay(STEP_DELAY_MS);
  RCLCPP_INFO(node->get_logger(), "[Phase 4] Right handover move complete");

  RCLCPP_INFO(node->get_logger(), "[Phase 5] Left handover receive start (fixed world-frame waypoint)");
  commander.openGripper("left");
  logIfFailed(node, commander, "openGripper(left)");
  stepDelay(SHORT_STEP_DELAY_MS);
  const auto left_handover = commander.makePose(
    L_HAND_X, L_HAND_Y, L_HAND_Z, L_HAND_QX, L_HAND_QY, L_HAND_QZ, L_HAND_QW);
  commander.moveToPose(left_handover, "left");
  logIfFailed(node, commander, "moveToPose(left_handover, left)");
  stepDelay(STEP_DELAY_MS);
  commander.closeGripper("left");
  logIfFailed(node, commander, "closeGripper(left)");
  stepDelay(STEP_DELAY_MS);
  commander.openGripper("right");
  logIfFailed(node, commander, "openGripper(right)");
  stepDelay(SHORT_STEP_DELAY_MS);
  const auto right_retreat = commander.makePose(
    R_RET_X, R_RET_Y, R_RET_Z, R_RET_QX, R_RET_QY, R_RET_QZ, R_RET_QW);
  commander.moveToPose(right_retreat, "right");
  logIfFailed(node, commander, "moveToPose(right_retreat, right)");
  stepDelay(STEP_DELAY_MS);
  RCLCPP_INFO(node->get_logger(), "[Phase 5] Left handover receive complete");

  RCLCPP_INFO(node->get_logger(), "[Phase 6] Left place start (fixed world-frame waypoint)");
  const auto place_pose = commander.makePose(
    PLACE_X, PLACE_Y, PLACE_Z, PLACE_QX, PLACE_QY, PLACE_QZ, PLACE_QW);
  commander.moveToPose(place_pose, "left");
  logIfFailed(node, commander, "moveToPose(place_pose, left)");
  stepDelay(STEP_DELAY_MS);
  commander.openGripper("left");
  logIfFailed(node, commander, "openGripper(left)");
  stepDelay(SHORT_STEP_DELAY_MS);
  RCLCPP_INFO(node->get_logger(), "[Phase 6] Left place complete");

  RCLCPP_INFO(node->get_logger(), "[Phase 7] Return home start");
  commander.goToNamedTarget("hands_up", "right");
  logIfFailed(node, commander, "goToNamedTarget(hands_up, right)");
  stepDelay(STEP_DELAY_MS);
  commander.goToNamedTarget("hands_up", "left");
  logIfFailed(node, commander, "goToNamedTarget(hands_up, left)");
  RCLCPP_INFO(node->get_logger(), "[Phase 7] Return home complete");

  rclcpp::shutdown();
  spinner.join();
  return 0;
}
