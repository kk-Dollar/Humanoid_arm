#include "arm_commander/commander.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>

#include <chrono>
#include <mutex>
#include <optional>
#include <thread>

using namespace std::chrono_literals;

static constexpr double POSE1_APPROACH_HEIGHT = 0.04;
static constexpr double POSE1_APPROACH_Y = -0.1;  
static constexpr double POSE1_APPROACH_X = 0.0;
static constexpr int STEP_DELAY_MS = 1000;
static constexpr int SHORT_STEP_DELAY_MS = 500;
static constexpr int WAIT_TIMEOUT_SEC = 30;
static constexpr int WAIT_LOG_PERIOD_SEC = 2;

static constexpr double POSE1_QX = -0.007;
static constexpr double POSE1_QY = -0.104;
static constexpr double POSE1_QZ = 0.002;
static constexpr double POSE1_QW = 0.995;

// Offset from Cube Center to openarm_right_ee_base_link
static constexpr double GRASP_OFFSET_X = 0.01;
static constexpr double GRASP_OFFSET_Y = 0.015;
static constexpr double GRASP_OFFSET_Z = 0.148;

static constexpr double RIGHT_POSE_B_X = 0.211;
static constexpr double RIGHT_POSE_B_Y = -0.210;
static constexpr double RIGHT_POSE_B_Z = 0.463;
static constexpr double RIGHT_POSE_B_QX = 0.495;
static constexpr double RIGHT_POSE_B_QY = -0.502;
static constexpr double RIGHT_POSE_B_QZ = 0.461;
static constexpr double RIGHT_POSE_B_QW = 0.539;

static constexpr double LEFT_POSE_C_X = 0.213;
static constexpr double LEFT_POSE_C_Y = 0.154;
static constexpr double LEFT_POSE_C_Z = 0.463;
static constexpr double LEFT_POSE_C_QX = -0.477;
static constexpr double LEFT_POSE_C_QY = -0.517;
static constexpr double LEFT_POSE_C_QZ = -0.444;
static constexpr double LEFT_POSE_C_QW = 0.555;

static constexpr double LEFT_POSE_D_X = 0.148;
static constexpr double LEFT_POSE_D_Y = 0.305;
static constexpr double LEFT_POSE_D_Z = 0.430;
static constexpr double LEFT_POSE_D_QX = 0.512;
static constexpr double LEFT_POSE_D_QY = -0.379;
static constexpr double LEFT_POSE_D_QZ = 0.363;
static constexpr double LEFT_POSE_D_QW = 0.680;

class CubePoseTracker
{
public:
  explicit CubePoseTracker(const rclcpp::Node::SharedPtr & node)
  {
    cube_pose_sub_ = node->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/cube_pose", 10,
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_cube_pose_ = *msg;
      });

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

static void stepDelay(int delay_ms)
{
  std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
}

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

static geometry_msgs::msg::Pose computePose1Approach(
  Commander & commander,
  const geometry_msgs::msg::PoseStamped & cube_pose)
{
  // Restored dynamic ArUco tracking
  return commander.makePose(
    cube_pose.pose.position.x + GRASP_OFFSET_X,
    cube_pose.pose.position.y + GRASP_OFFSET_Y,
    cube_pose.pose.position.z + GRASP_OFFSET_Z + POSE1_APPROACH_HEIGHT,
    POSE1_QX, POSE1_QY, POSE1_QZ, POSE1_QW);
}

static geometry_msgs::msg::Pose computePose1Pick(
  Commander & commander,
  const geometry_msgs::msg::PoseStamped & cube_pose)
{
  // Restored dynamic ArUco tracking
  return commander.makePose(
    cube_pose.pose.position.x + GRASP_OFFSET_X,
    cube_pose.pose.position.y + GRASP_OFFSET_Y,
    cube_pose.pose.position.z + GRASP_OFFSET_Z,
    POSE1_QX, POSE1_QY, POSE1_QZ, POSE1_QW);
}

static void logIfFailed(const rclcpp::Node::SharedPtr & node, Commander & commander, const char * what)
{
  if (!commander.lastCommandSucceeded()) {
    RCLCPP_ERROR(node->get_logger(), "MoveIt command failed: %s", what);
  }
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>("openarm_commander");
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  auto spinner = std::thread([&executor]() { executor.spin(); });

  Commander commander(node);
  CubePoseTracker tracker(node);

  RCLCPP_INFO(node->get_logger(), "=== Step 0: return both arms home ===");
  commander.goToNamedTarget("hands_up", "right");
  logIfFailed(node, commander, "goToNamedTarget(hands_up, right)");
  stepDelay(STEP_DELAY_MS);
  commander.goToNamedTarget("hands_up", "left");
  logIfFailed(node, commander, "goToNamedTarget(hands_up, left)");
  stepDelay(STEP_DELAY_MS);

  RCLCPP_INFO(node->get_logger(), "=== Step 1: wait for chest-camera cube pose ===");
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

  RCLCPP_INFO(node->get_logger(), "=== Step 2: right_arm -> detected Pose 1 ===");
  const auto right_pose_A_approach = computePose1Approach(commander, cube_pose);
  const auto right_pose_A = computePose1Pick(commander, cube_pose);

  commander.openGripper("right");
  commander.openGripper("left");
  logIfFailed(node, commander, "openGripper(right)");
  stepDelay(SHORT_STEP_DELAY_MS);
  commander.moveToPose(right_pose_A_approach, "right");
  logIfFailed(node, commander, "moveToPose(right_pose_A_approach, right)");
  stepDelay(SHORT_STEP_DELAY_MS);
  commander.moveToPose(right_pose_A, "right");
  logIfFailed(node, commander, "moveToPose(right_pose_A, right)");
  stepDelay(SHORT_STEP_DELAY_MS);
  commander.closeGripper("right");
  logIfFailed(node, commander, "closeGripper(right)");
  stepDelay(STEP_DELAY_MS);

  RCLCPP_INFO(node->get_logger(), "=== Step 3: right_arm -> fixed Pose B ===");
  auto right_pose_B = commander.makePose(
    RIGHT_POSE_B_X, RIGHT_POSE_B_Y, RIGHT_POSE_B_Z,
    RIGHT_POSE_B_QX, RIGHT_POSE_B_QY, RIGHT_POSE_B_QZ, RIGHT_POSE_B_QW);
  commander.moveToPose(right_pose_B, "right");
  logIfFailed(node, commander, "moveToPose(right_pose_B, right)");
  stepDelay(STEP_DELAY_MS);

  RCLCPP_INFO(node->get_logger(), "=== Step 4: left_arm -> fixed Pose C ===");
  auto left_pose_C = commander.makePose(
    LEFT_POSE_C_X, LEFT_POSE_C_Y, LEFT_POSE_C_Z,
    LEFT_POSE_C_QX, LEFT_POSE_C_QY, LEFT_POSE_C_QZ, LEFT_POSE_C_QW);
  auto left_pose_C_approach = commander.makePose(
    LEFT_POSE_C_X  , LEFT_POSE_C_Y + POSE1_APPROACH_Y, LEFT_POSE_C_Z ,
    LEFT_POSE_C_QX, LEFT_POSE_C_QY, LEFT_POSE_C_QZ, LEFT_POSE_C_QW);
  commander.moveToPose(left_pose_C_approach, "left");
  logIfFailed(node, commander, "moveToPose(left_pose_C_approach, left)");
  commander.moveToPose(left_pose_C, "left");
  logIfFailed(node, commander, "moveToPose(left_pose_C, left)");
  commander.closeGripper("left");
  logIfFailed(node, commander, "closeGripper(left)");
  commander.openGripper("right");
  logIfFailed(node, commander, "openGripper(right)");
  commander.moveCartesianByZ("left", POSE1_APPROACH_HEIGHT + 0.02);
  logIfFailed(node, commander, "moveCartesianByZ(left, approach_height + 0.02)");

  RCLCPP_INFO(node->get_logger(), "Waiting 3 s...");
  std::this_thread::sleep_for(3s);

  RCLCPP_INFO(node->get_logger(), "=== Step 5: left_arm -> fixed Pose D ===");
  auto left_pose_D = commander.makePose(
    LEFT_POSE_D_X, LEFT_POSE_D_Y, LEFT_POSE_D_Z,
    LEFT_POSE_D_QX, LEFT_POSE_D_QY, LEFT_POSE_D_QZ, LEFT_POSE_D_QW);
  commander.moveToPose(left_pose_D, "left");
  logIfFailed(node, commander, "moveToPose(left_pose_D, left)");
  commander.openGripper("left");
  logIfFailed(node, commander, "openGripper(left)");

  commander.goToNamedTarget("hands_up", "left");
  logIfFailed(node, commander, "goToNamedTarget(hands_up, left)");
  commander.goToNamedTarget("hands_up", "right");
  logIfFailed(node, commander, "goToNamedTarget(hands_up, right)");

  RCLCPP_INFO(node->get_logger(), "=== Sequence complete ===");

  rclcpp::shutdown();
  spinner.join();
  return 0;
}