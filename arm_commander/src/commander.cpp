#include "arm_commander/commander.hpp"

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <example_interfaces/msg/bool.hpp>
#include <example_interfaces/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>

#include <cmath>
#include <thread>
#include <chrono>

using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;
using BoolMsg            = example_interfaces::msg::Bool;
using FloatArrayMsg      = example_interfaces::msg::Float64MultiArray;
using StringMsg          = std_msgs::msg::String;
using PoseMsg            = geometry_msgs::msg::Pose;

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────

Commander::Commander(std::shared_ptr<rclcpp::Node> node)
: node_(node)
{
  arm_right = std::make_shared<MoveGroupInterface>(node_, "right_arm");
  arm_right->setMaxAccelerationScalingFactor(1.0);
  arm_right->setMaxVelocityScalingFactor(1.0);
  gripper_right = std::make_shared<MoveGroupInterface>(node_, "right_gripper");

  arm_left = std::make_shared<MoveGroupInterface>(node_, "left_arm");
  arm_left->setMaxAccelerationScalingFactor(1.0);
  arm_left->setMaxVelocityScalingFactor(1.0);
  gripper_left = std::make_shared<MoveGroupInterface>(node_, "left_gripper");

  open_gripper_sub_ = node_->create_subscription<BoolMsg>(
    "open_gripper", 10,
    std::bind(&Commander::openGripperCallback, this, std::placeholders::_1));

  joint_command_sub_ = node_->create_subscription<FloatArrayMsg>(
    "joint_command", 10,
    std::bind(&Commander::jointCommandCallback, this, std::placeholders::_1));

  named_pose_sub_ = node_->create_subscription<StringMsg>(
    "named_pose", 10,
    std::bind(&Commander::namedPoseCallback, this, std::placeholders::_1));

  pose_target_sub_ = node_->create_subscription<PoseMsg>(
    "pose_target", 10,
    std::bind(&Commander::poseTargetCallback, this, std::placeholders::_1));
}

bool Commander::lastCommandSucceeded() const
{
  return last_command_succeeded_;
}

// ─────────────────────────────────────────────────────────────────────────────
// Named targets
// ─────────────────────────────────────────────────────────────────────────────

void Commander::goToNamedTarget(const std::string &name)
{
  goToNamedTarget(name, "right");
  goToNamedTarget(name, "left");
}

void Commander::goToNamedTarget(const std::string &name, const std::string &arm_name)
{
  auto arm = getArm(arm_name);
  if (!arm) {
    RCLCPP_ERROR(node_->get_logger(), "Unknown arm '%s' for named target", arm_name.c_str());
    last_command_succeeded_ = false;
    return;
  }
  arm->setStartStateToCurrentState();
  arm->setNamedTarget(name);
  planAndExecute(arm);
}

// ─────────────────────────────────────────────────────────────────────────────
// Gripper control
// ─────────────────────────────────────────────────────────────────────────────

void Commander::openGripper()
{
  openGripper("right");
  openGripper("left");
}

void Commander::openGripper(const std::string &arm_name)
{
  auto gripper = getGripper(arm_name);
  if (!gripper) {
    RCLCPP_ERROR(node_->get_logger(), "Unknown gripper '%s' to open", arm_name.c_str());
    last_command_succeeded_ = false;
    return;
  }
  gripper->setStartStateToCurrentState();
  gripper->setNamedTarget("open_gripper");
  planAndExecute(gripper);
}

void Commander::closeGripper()
{
  closeGripper("right");
  closeGripper("left");
}

void Commander::closeGripper(const std::string &arm_name)
{
  auto gripper = getGripper(arm_name);
  if (!gripper) {
    RCLCPP_ERROR(node_->get_logger(), "Unknown gripper '%s' to close", arm_name.c_str());
    last_command_succeeded_ = false;
    return;
  }
  gripper->setStartStateToCurrentState();
  gripper->setNamedTarget("close_gripper");
  planAndExecute(gripper);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pose helpers
// ─────────────────────────────────────────────────────────────────────────────

// RPY version — converts roll/pitch/yaw (radians) to quaternion internally.
//
// RPY → quaternion formula (intrinsic XYZ rotations):
//   q = Rz(yaw) * Ry(pitch) * Rx(roll)
//
// This is what you want to use when thinking about gripper orientation:
//   roll=0, pitch=-π/2, yaw=+π/2  → right arm gripper faces -Y (inward)
//   roll=0, pitch=-π/2, yaw=-π/2  → left  arm gripper faces +Y (inward)
PoseMsg Commander::makePose(
  double x, double y, double z,
  double roll, double pitch, double yaw)
{
  // Half angles
  double cr = std::cos(roll  / 2.0);
  double sr = std::sin(roll  / 2.0);
  double cp = std::cos(pitch / 2.0);
  double sp = std::sin(pitch / 2.0);
  double cy = std::cos(yaw   / 2.0);
  double sy = std::sin(yaw   / 2.0);

  PoseMsg p;
  p.position.x    = x;
  p.position.y    = y;
  p.position.z    = z;
  p.orientation.w = cr * cp * cy + sr * sp * sy;
  p.orientation.x = sr * cp * cy - cr * sp * sy;
  p.orientation.y = cr * sp * cy + sr * cp * sy;
  p.orientation.z = cr * cp * sy - sr * sp * cy;
  return p;
}

// Quaternion version — use when you have values from tf2_echo or python script.
PoseMsg Commander::makePose(
  double x, double y, double z,
  double qx, double qy, double qz, double qw)
{
  PoseMsg p;
  p.position.x    = x;
  p.position.y    = y;
  p.position.z    = z;
  p.orientation.x = qx;
  p.orientation.y = qy;
  p.orientation.z = qz;
  p.orientation.w = qw;
  return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// Motion — pose targets
// ─────────────────────────────────────────────────────────────────────────────

// Move to a full pose (position + orientation).
// This is what you use for pick/handover/place where orientation matters.
void Commander::moveToPose(const PoseMsg &pose, const std::string &arm_name)
{
  auto arm = getArm(arm_name);
  if (!arm) {
    RCLCPP_ERROR(node_->get_logger(), "Unknown arm '%s' for moveToPose", arm_name.c_str());
    last_command_succeeded_ = false;
    return;
  }
  arm->setStartStateToCurrentState();
  arm->setPoseTarget(pose);   // sets BOTH position AND orientation
  planAndExecute(arm);
}

// Move to a position only — IK solver picks the orientation.
// Useful for rough positioning where exact orientation doesn't matter.
void Commander::goToPoseTarget(const PoseMsg &pose)
{
  goToPoseTarget(pose, "right");
  goToPoseTarget(pose, "left");
}

void Commander::goToPoseTarget(const PoseMsg &pose, const std::string &arm_name)
{
  auto arm = getArm(arm_name);
  if (!arm) {
    RCLCPP_ERROR(node_->get_logger(), "Unknown arm '%s' for pose target", arm_name.c_str());
    last_command_succeeded_ = false;
    return;
  }
  arm->setStartStateToCurrentState();
  arm->setPositionTarget(
    pose.position.x, pose.position.y, pose.position.z);  // position only
  planAndExecute(arm);
}

// ─────────────────────────────────────────────────────────────────────────────
// Motion — Cartesian straight-line moves
//
// All three move the EE in a straight line along one world axis.
// The orientation is KEPT from the current pose — only position shifts.
// fraction check: if < 95% of the path is achievable, we warn and abort.
// ─────────────────────────────────────────────────────────────────────────────

void Commander::moveCartesianByAxis(
  const std::string &arm_name, double dx, double dy, double dz)
{
  auto arm = getArm(arm_name);
  if (!arm) {
    RCLCPP_ERROR(node_->get_logger(),
      "Unknown arm '%s' for Cartesian move", arm_name.c_str());
    last_command_succeeded_ = false;
    return;
  }

  auto current_pose = arm->getCurrentPose().pose;
  current_pose.position.x += dx;
  current_pose.position.y += dy;
  current_pose.position.z += dz;

  std::vector<PoseMsg> waypoints = { current_pose };

  moveit_msgs::msg::RobotTrajectory trajectory;
  const double fraction =
    arm->computeCartesianPath(waypoints, 0.01, 0.0, trajectory);

  if (fraction < 0.95) {
    RCLCPP_WARN(node_->get_logger(),
      "Cartesian path for '%s' only %.1f%% achieved — aborting",
      arm_name.c_str(), fraction * 100.0);
    last_command_succeeded_ = false;
    return;
  }

  MoveGroupInterface::Plan plan;
  plan.trajectory_ = trajectory;
  const auto exec_result = arm->execute(plan);
  last_command_succeeded_ = (exec_result == moveit::core::MoveItErrorCode::SUCCESS);
}

void Commander::moveCartesianByZ(const std::string &arm_name, double delta_z)
{
  moveCartesianByAxis(arm_name, 0.0, 0.0, delta_z);
}

void Commander::moveCartesianByY(const std::string &arm_name, double delta_y)
{
  moveCartesianByAxis(arm_name, 0.0, delta_y, 0.0);
}

void Commander::moveCartesianByX(const std::string &arm_name, double delta_x)
{
  moveCartesianByAxis(arm_name, delta_x, 0.0, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Planning + execution
// ─────────────────────────────────────────────────────────────────────────────

void Commander::planAndExecute(
  const std::shared_ptr<MoveGroupInterface> &interface)
{
  // Always sync start state before planning — prevents execute aborts when
  // prior motion failed or was partial. Ensures controller and planner agree
  // on the start state, even if a prior motion was only partially executed.
  interface->setStartStateToCurrentState();

  // Set longer planning timeout to allow more time for path finding
  interface->setPlanningTime(20.0);  // 20 seconds per planning attempt

  MoveGroupInterface::Plan plan;
  bool success =
    (interface->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (!success) {
    RCLCPP_ERROR(node_->get_logger(),
      "Planning failed for group '%s'.", interface->getName().c_str());
    last_command_succeeded_ = false;
    return;
  }

  // Small delay to let fake hardware controller catch up to start state
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Execute with automatic retry on abort
  auto exec_result = interface->execute(plan);
  if (exec_result.val == moveit::core::MoveItErrorCode::SUCCESS) {
    last_command_succeeded_ = true;
    return;  // Success on first attempt
  }

  // First execute failed; retry once with fresh planning
  RCLCPP_WARN(node_->get_logger(),
    "Execute failed for group '%s' (will retry with replan).",
    interface->getName().c_str());

  interface->setStartStateToCurrentState();
  interface->setPlanningTime(20.0);  // Set timeout again for retry
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  if (interface->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto retry_exec_result = interface->execute(plan);
    last_command_succeeded_ = (retry_exec_result == moveit::core::MoveItErrorCode::SUCCESS);
    if (!last_command_succeeded_) {
      RCLCPP_ERROR(node_->get_logger(),
        "Retry execute failed for group '%s'.", interface->getName().c_str());
    }
  } else {
    RCLCPP_ERROR(node_->get_logger(),
      "Retry planning failed for group '%s'.", interface->getName().c_str());
    last_command_succeeded_ = false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Topic callbacks
// ─────────────────────────────────────────────────────────────────────────────

void Commander::openGripperCallback(const BoolMsg &msg)
{
  if (msg.data) openGripper();
  else          closeGripper();
}

void Commander::jointCommandCallback(const FloatArrayMsg &msg)
{
  const auto &vals = msg.data;
  if (vals.size() == 7) {
    setJointTarget(vals, "right");
  } else if (vals.size() == 14) {
    setJointTarget(std::vector<double>(vals.begin(),      vals.begin() + 7),  "right");
    setJointTarget(std::vector<double>(vals.begin() + 7,  vals.end()),         "left");
  } else {
    RCLCPP_WARN(node_->get_logger(),
      "joint_command: expected 7 or 14 values, got %zu", vals.size());
  }
}

void Commander::setJointTarget(
  const std::vector<double> &vals, const std::string &arm_name)
{
  auto arm = getArm(arm_name);
  if (!arm) {
    RCLCPP_ERROR(node_->get_logger(),
      "Unknown arm '%s' for joint target", arm_name.c_str());
    last_command_succeeded_ = false;
    return;
  }
  arm->setStartStateToCurrentState();
  arm->setJointValueTarget(vals);
  planAndExecute(arm);
}

void Commander::namedPoseCallback(const StringMsg &msg)
{
  RCLCPP_INFO(node_->get_logger(), "Named pose: '%s'", msg.data.c_str());
  goToNamedTarget(msg.data);
}

void Commander::poseTargetCallback(const PoseMsg &msg)
{
  RCLCPP_INFO(node_->get_logger(),
    "Pose target: [x=%.3f, y=%.3f, z=%.3f]",
    msg.position.x, msg.position.y, msg.position.z);
  goToPoseTarget(msg);
}

// ─────────────────────────────────────────────────────────────────────────────
// Lookups
// ─────────────────────────────────────────────────────────────────────────────

std::shared_ptr<MoveGroupInterface>
Commander::getArm(const std::string &arm_name)
{
  if (arm_name == "right" || arm_name == "right_arm") return arm_right;
  if (arm_name == "left"  || arm_name == "left_arm")  return arm_left;
  return nullptr;
}

std::shared_ptr<MoveGroupInterface>
Commander::getGripper(const std::string &arm_name)
{
  if (arm_name == "right" || arm_name == "right_arm") return gripper_right;
  if (arm_name == "left"  || arm_name == "left_arm")  return gripper_left;
  return nullptr;
}