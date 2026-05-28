#include "arm_commander/commander.hpp"
#include <rclcpp/rclcpp.hpp>
#include <thread>
#include <chrono>

using namespace std::chrono_literals;

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>("openarm_commander");
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  auto spinner = std::thread([&executor]() { executor.spin(); });

  Commander commander(node);

  // Pose A: right arm pose 1 (aligned with MuJoCo cube at pedestal_pose1_cube.xml)
  // Cube center in world (Pose 1 keyframe): [0.132, -0.220, 0.333]
  auto right_pose_A = commander.makePose(
    0.132, -0.220, 0.333,
    -0.270, -0.609, -0.178, 0.724);

  // approach above Pose A (cartesian descent): same x/y/orientation, higher z
  const double approach_height = 0.07; // meters above target before cartesian descent
  auto right_pose_A_approach = commander.makePose(
    0.132, -0.220, 0.333 + approach_height,
    -0.270, -0.609, -0.178, 0.724);

  // Pose B: right arm pose 2
  // tf2_echo world openarm_right_ee_base_link -> [0.193, -0.180, 0.325]
  auto right_pose_B = commander.makePose(
  0.211, -0.210, 0.463,
  0.495, -0.502, 0.461, 0.539);

  auto left_pose_C = commander.makePose(
  0.213, 0.154, 0.463,
  -0.477, -0.517, -0.444, 0.555);

  // approach above Pose C (cartesian descent)
  auto left_pose_C_approach = commander.makePose(
    0.213, 0.154, 0.463 + approach_height,
    -0.477, -0.517, -0.444, 0.555);

  // Pose D: left arm pose 4
  // tf2_echo world openarm_left_ee_base_link -> [0.148, 0.305, 0.430]
  auto left_pose_D = commander.makePose(
    0.148, 0.305, 0.430,
    0.512, -0.379, 0.363, 0.680);

  commander.openGripper();
  RCLCPP_INFO(node->get_logger(), "=== Step 1: right_arm -> Pose A (approach then descend) ===");
  // move above target first
  commander.moveToPose(right_pose_A_approach, "right");
  // descend straight down in z to target pose (cartesian)
  commander.moveCartesianByZ("right", -approach_height);
  // ensure final pose matches exact target
  commander.moveToPose(right_pose_A, "right");
  commander.closeGripper("right");

  RCLCPP_INFO(node->get_logger(), "Waiting 3 s...");
  std::this_thread::sleep_for(3s);

  RCLCPP_INFO(node->get_logger(), "=== Step 2: right_arm -> Pose B ===");
  commander.moveToPose(right_pose_B, "right");
  

  RCLCPP_INFO(node->get_logger(), "=== Step 3: left_arm -> Pose C (approach then descend) ===");
  commander.moveToPose(left_pose_C_approach, "left");
  commander.moveCartesianByZ("left", -approach_height);
  commander.moveToPose(left_pose_C, "left");
  commander.closeGripper("left");
  commander.openGripper("right");
   commander.moveCartesianByZ("left", approach_height+0.02);
  commander.goToNamedTarget("hands_up","right");

  RCLCPP_INFO(node->get_logger(), "Waiting 3 s...");
  std::this_thread::sleep_for(3s);

  RCLCPP_INFO(node->get_logger(), "=== Step 4: left_arm -> Pose D ===");
  commander.moveToPose(left_pose_D, "left");
  commander.openGripper("left");
  
  commander.goToNamedTarget("hands_up","left");
  commander.goToNamedTarget("hands_up","right");

  RCLCPP_INFO(node->get_logger(), "=== Sequence complete ===");

  rclcpp::shutdown();
  spinner.join();
  return 0;
}