#pragma once

#include <example_interfaces/msg/bool.hpp>
#include <example_interfaces/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include <memory>
#include <string>
#include <vector>

class Commander
{
public:
  explicit Commander(std::shared_ptr<rclcpp::Node> node);

  // ── Named targets ────────────────────────────────────────────────────────
  void goToNamedTarget(const std::string &name);
  void goToNamedTarget(const std::string &name, const std::string &arm_name);
  bool lastCommandSucceeded() const;

  // ── Gripper control ──────────────────────────────────────────────────────
  void openGripper();
  void openGripper(const std::string &arm_name);
  void closeGripper();
  void closeGripper(const std::string &arm_name);

  // Sets gripper to a specific gap width in meters.
  // Solves the "closes completely" problem — gripper stops at cube width.
  // gap_meters: desired distance between finger tips in meters
  //             0.0 = fully closed, ~0.063 = fully open (0.7 rad * scale)
  // arm_name: "left" or "right"
  void setGripperWidth(const std::string &arm_name, double gap_meters);

  // Converts a gap in meters to a finger joint position in radians.
  // Uses FINGER_LENGTH_SCALE constant defined in commander.cpp.
  // Returns value clamped to [0.0, 0.7].
  double gapToJointValue(double gap_meters);

  // ── Pose helpers ─────────────────────────────────────────────────────────

  // Build a Pose from position + Roll-Pitch-Yaw (radians).
  // Use this when you want to specify orientation as human-readable angles.
  // Example: makePose(0.3, 0.0, 0.5, 0, -M_PI_2, M_PI_2)
  geometry_msgs::msg::Pose makePose(
    double x, double y, double z,
    double roll, double pitch, double yaw);

  // Build a Pose from position + explicit quaternion (x, y, z, w).
  // Use this when you already have quaternion values (e.g. from tf2_echo).
  // Example: makePose(0.3, 0.0, 0.5,  0.5, -0.5, 0.5, 0.5)
  geometry_msgs::msg::Pose makePose(
    double x, double y, double z,
    double qx, double qy, double qz, double qw);

  // ── Motion commands ──────────────────────────────────────────────────────

  // Move arm EE to an absolute pose (position + orientation).
  void moveToPose(const geometry_msgs::msg::Pose &pose, const std::string &arm_name);

  // Move arm EE to a position only (IK chooses orientation).
  void goToPoseTarget(const geometry_msgs::msg::Pose &pose);
  void goToPoseTarget(const geometry_msgs::msg::Pose &pose, const std::string &arm_name);

  // Cartesian straight-line moves — keeps current orientation, shifts one axis.
  void moveCartesianByZ(const std::string &arm_name, double delta_z);
  void moveCartesianByY(const std::string &arm_name, double delta_y);
  void moveCartesianByX(const std::string &arm_name, double delta_x);
  void moveCartesianByAxis(
    const std::string &arm_name, double dx, double dy, double dz);

private:
  void planAndExecute(
    const std::shared_ptr<moveit::planning_interface::MoveGroupInterface> &interface);

  // Topic callbacks
  void openGripperCallback(const example_interfaces::msg::Bool &msg);
  void jointCommandCallback(const example_interfaces::msg::Float64MultiArray &msg);
  void setJointTarget(const std::vector<double> &vals, const std::string &arm_name);
  void namedPoseCallback(const std_msgs::msg::String &msg);
  void poseTargetCallback(const geometry_msgs::msg::Pose &msg);

  // Arm / gripper lookup by name ("left" / "right")
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface>
    getArm(const std::string &arm_name);
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface>
    getGripper(const std::string &arm_name);

  // ── Members ──────────────────────────────────────────────────────────────
  std::shared_ptr<rclcpp::Node> node_;

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_right;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> gripper_right;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_left;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> gripper_left;

  rclcpp::Subscription<example_interfaces::msg::Bool>::SharedPtr            open_gripper_sub_;
  rclcpp::Subscription<example_interfaces::msg::Float64MultiArray>::SharedPtr joint_command_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr                    named_pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr                 pose_target_sub_;

  bool last_command_succeeded_{true};
};