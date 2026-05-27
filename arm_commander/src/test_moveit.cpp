#include "arm_commander/commander.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <functional>

// ─────────────────────────────────────────────────────────────────────────────
// ALL POSES — exact values from tf2_echo with robot positioned in RViz
// Format: Translation [x, y, z]  Quaternion [x, y, z, w]
// ─────────────────────────────────────────────────────────────────────────────

// Delay between each step (milliseconds) — increase if robot moves too fast
static constexpr int STEP_DELAY_MS = 1000;   // 1 second between steps

// ── RIGHT ARM POSES ──────────────────────────────────────────────────────────

// Step 1: Right arm pick pose — fingers at cube, ready to grasp
// tf2_echo: Translation [0.105, -0.320, 0.328]  Q [0, -0.682, 0, 0.732]
static constexpr double R_PICK_X  =  0.274, R_PICK_Y  = -0.271, R_PICK_Z  = 0.582;
static constexpr double R_PICK_QX =  0.000, R_PICK_QY = -0.682, R_PICK_QZ = 0.000, R_PICK_QW = 0.732;

// Step 3: Right arm handover pose — holds cube up for left arm to grab
// tf2_echo: Translation [0.254, -0.062, 0.465]  Q [0.492, -0.462, -0.445, 0.589]
static constexpr double R_HAND_X  =  0.272, R_HAND_Y  = -0.102, R_HAND_Z  = 0.563;
static constexpr double R_HAND_QX =  0.453, R_HAND_QY = -0.509, R_HAND_QZ = -0.486, R_HAND_QW = 0.547;

// Step 6: Right arm retreat pose — moves away after releasing cube
// tf2_echo: Translation [0.278, -0.190, 0.479]  Q [0.492, -0.462, -0.445, 0.589]
// The retreat pose has been removed.

// ── LEFT ARM POSES ───────────────────────────────────────────────────────────

// Step 4: Left arm handover pose — approaches to receive cube from right arm
// tf2_echo: Translation [0.237, 0.051, 0.436]  Q [0.517, 0.544, -0.473, -0.462]
static constexpr double L_HAND_X  =  0.237, L_HAND_Y  =  0.051, L_HAND_Z  = 0.436;
static constexpr double L_HAND_QX =  0.517, L_HAND_QY =  0.544, L_HAND_QZ = -0.473, L_HAND_QW = -0.462;

// Step 7: Left arm place pose — moves cube to final place location
// tf2_echo: Translation [0.181, 0.259, 0.436]  Q [0.045, -0.749, -0.016, 0.661]
static constexpr double L_PLACE_X  =  0.181, L_PLACE_Y  =  0.259, L_PLACE_Z  = 0.436;
static constexpr double L_PLACE_QX =  0.045, L_PLACE_QY = -0.749, L_PLACE_QZ = -0.016, L_PLACE_QW = 0.661;

// ─────────────────────────────────────────────────────────────────────────────
namespace
{

void stepDelay(const std::shared_ptr<rclcpp::Node> & node, const std::string & next_step)
{
  RCLCPP_INFO(node->get_logger(),
    "  ── waiting %d ms before: %s", STEP_DELAY_MS, next_step.c_str());
  std::this_thread::sleep_for(std::chrono::milliseconds(STEP_DELAY_MS));
}

std::string runCommand(const std::string & cmd)
{
  std::array<char, 256> buf{};
  std::string out;
  FILE * pipe = popen(cmd.c_str(), "r");
  if (!pipe) throw std::runtime_error("popen failed: " + cmd);
  while (fgets(buf.data(), static_cast<int>(buf.size()), pipe)) out += buf.data();
  if (pclose(pipe) != 0) throw std::runtime_error("Command failed: " + cmd);
  return out;
}

std::string buildRobotDescription()
{
  const auto share = ament_index_cpp::get_package_share_directory("openarm_description");
  const auto xacro = share + "/assets/robot/openarm_v2.0/urdf/openarm_v20.urdf.xacro";
  return runCommand(
    "xacro " + xacro +
    " arm_type:=openarm_v2.0"
    " bimanual:=true"
    " use_fake_hardware:=true"
    " ros2_control:=true"
    " left_can_interface:=can1"
    " right_can_interface:=can0");
}

std::string buildRobotSemanticDescription()
{
  const auto share = ament_index_cpp::get_package_share_directory("openarm_bimanual_moveit_config");
  return runCommand("cat " + share + "/config/openarm_v2.0/openarm_bimanual.srdf");
}

struct JointStateBroadcaster
{
  explicit JointStateBroadcaster(std::shared_ptr<rclcpp::Node> node)
  : node_(node), running_(true)
  {
    pub_ = node_->create_publisher<sensor_msgs::msg::JointState>(
      "joint_states", rclcpp::QoS(10));
    worker_ = std::thread([this] { loop(); });
  }
  ~JointStateBroadcaster()
  {
    running_ = false;
    if (worker_.joinable()) worker_.join();
  }
  void loop()
  {
    sensor_msgs::msg::JointState msg;
    msg.name = {
      "openarm_left_joint1",  "openarm_left_joint2",  "openarm_left_joint3",
      "openarm_left_joint4",  "openarm_left_joint5",  "openarm_left_joint6",
      "openarm_left_joint7",
      "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
      "openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6",
      "openarm_right_joint7",
      "openarm_left_finger_joint1", "openarm_right_finger_joint1"
    };
    msg.position = {
      0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0,
      0.7, -0.7
    };
    msg.velocity.resize(msg.name.size(), 0.0);
    msg.effort.resize(msg.name.size(),   0.0);
    while (rclcpp::ok() && running_.load()) {
      msg.header.stamp = node_->get_clock()->now();
      pub_->publish(msg);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
  std::shared_ptr<rclcpp::Node> node_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_;
  std::atomic<bool> running_;
  std::thread worker_;
};

struct NodeSpinner
{
  explicit NodeSpinner(std::shared_ptr<rclcpp::Node> node)
  {
    exec_.add_node(node);
    worker_ = std::thread([this] { exec_.spin(); });
  }
  ~NodeSpinner()
  {
    exec_.cancel();
    if (worker_.joinable()) worker_.join();
  }
  rclcpp::executors::SingleThreadedExecutor exec_;
  std::thread worker_;
};

bool waitForJointState(std::shared_ptr<rclcpp::Node> node, std::chrono::seconds timeout)
{
  bool received = false;
  auto sub = node->create_subscription<sensor_msgs::msg::JointState>(
    "joint_states", rclcpp::QoS(10),
    [&](sensor_msgs::msg::JointState::SharedPtr) { received = true; });
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (rclcpp::ok() && !received && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  return received;
}

// Step tracking and execution system
struct Step
{
  std::string name;
  bool completed = false;
  std::string error_msg;

  void mark_success()
  {
    completed = true;
    error_msg = "";
  }

  void mark_failure(const std::string & msg)
  {
    completed = false;
    error_msg = msg;
  }
};

struct Phase
{
  std::string name;
  std::vector<Step> steps;
  bool completed = false;
  int current_step_idx = 0;

  bool execute_next_step(const std::function<bool()> & action, const std::string & step_name)
  {
    if (current_step_idx >= static_cast<int>(steps.size())) {
      return false;  // All steps already completed
    }

    auto & step = steps[current_step_idx];
    step.name = step_name;

    if (!action()) {
      step.mark_failure("Action failed");
      return false;
    }

    step.mark_success();
    current_step_idx++;
    return true;
  }

  bool is_complete() const
  {
    return current_step_idx == static_cast<int>(steps.size());
  }

  void print_status(const std::shared_ptr<rclcpp::Node> & node) const
  {
    for (size_t i = 0; i < steps.size(); ++i) {
      const auto & s = steps[i];
      if (i < static_cast<size_t>(current_step_idx)) {
        RCLCPP_INFO(node->get_logger(), "    [✓] %s", s.name.c_str());
      } else if (i == static_cast<size_t>(current_step_idx)) {
        RCLCPP_INFO(node->get_logger(), "    [→] %s", s.name.c_str());
      } else {
        RCLCPP_INFO(node->get_logger(), "    [ ] %s", s.name.c_str());
      }
    }
  }
};

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>("handover_demo");

  const auto robot_description          = buildRobotDescription();
  const auto robot_description_semantic = buildRobotSemanticDescription();
  node->declare_parameter<std::string>("robot_description",          robot_description);
  node->declare_parameter<std::string>("robot_description_semantic", robot_description_semantic);
  node->set_parameter(rclcpp::Parameter("robot_description",          robot_description));
  node->set_parameter(rclcpp::Parameter("robot_description_semantic", robot_description_semantic));

  // Kinematics plugin — required for IK solving of Cartesian pose targets
  node->declare_parameter<std::string>(
    "robot_description_kinematics.left_arm.kinematics_solver",
    "kdl_kinematics_plugin/KDLKinematicsPlugin");
  node->declare_parameter<double>(
    "robot_description_kinematics.left_arm.kinematics_solver_search_resolution", 0.005);
  node->declare_parameter<double>(
    "robot_description_kinematics.left_arm.kinematics_solver_timeout", 0.005);
  node->declare_parameter<std::string>(
    "robot_description_kinematics.right_arm.kinematics_solver",
    "kdl_kinematics_plugin/KDLKinematicsPlugin");
  node->declare_parameter<double>(
    "robot_description_kinematics.right_arm.kinematics_solver_search_resolution", 0.005);
  node->declare_parameter<double>(
    "robot_description_kinematics.right_arm.kinematics_solver_timeout", 0.005);
  node->set_parameter(rclcpp::Parameter(
    "robot_description_kinematics.left_arm.kinematics_solver",
    std::string("kdl_kinematics_plugin/KDLKinematicsPlugin")));
  node->set_parameter(rclcpp::Parameter(
    "robot_description_kinematics.left_arm.kinematics_solver_search_resolution", 0.005));
  node->set_parameter(rclcpp::Parameter(
    "robot_description_kinematics.left_arm.kinematics_solver_timeout", 0.005));
  node->set_parameter(rclcpp::Parameter(
    "robot_description_kinematics.right_arm.kinematics_solver",
    std::string("kdl_kinematics_plugin/KDLKinematicsPlugin")));
  node->set_parameter(rclcpp::Parameter(
    "robot_description_kinematics.right_arm.kinematics_solver_search_resolution", 0.005));
  node->set_parameter(rclcpp::Parameter(
    "robot_description_kinematics.right_arm.kinematics_solver_timeout", 0.005));

  JointStateBroadcaster jsb(node);
  NodeSpinner spinner(node);

  if (!waitForJointState(node, std::chrono::seconds(5)))
    RCLCPP_WARN(node->get_logger(), "No joint_states within timeout — planning may fail.");

  Commander commander(node);

  // ── Build all poses from exact tf2_echo values ───────────────────────────
  const auto right_pick_pose     = commander.makePose(
    R_PICK_X,  R_PICK_Y,  R_PICK_Z,
    R_PICK_QX, R_PICK_QY, R_PICK_QZ, R_PICK_QW);

  const auto right_handover_pose = commander.makePose(
    R_HAND_X,  R_HAND_Y,  R_HAND_Z,
    R_HAND_QX, R_HAND_QY, R_HAND_QZ, R_HAND_QW);

  const auto left_handover_pose  = commander.makePose(
    L_HAND_X,  L_HAND_Y,  L_HAND_Z,
    L_HAND_QX, L_HAND_QY, L_HAND_QZ, L_HAND_QW);

  const auto left_place_pose     = commander.makePose(
    L_PLACE_X,  L_PLACE_Y,  L_PLACE_Z,
    L_PLACE_QX, L_PLACE_QY, L_PLACE_QZ, L_PLACE_QW);

  // ── SYSTEMATIC PHASE EXECUTION ────────────────────────────────────────────
  // Define all phases with their steps
  std::vector<Phase> phases;

  // Phase 0: Both arms to home
  phases.emplace_back();
  phases[0].name = "Phase 0: Home";
  phases[0].steps.resize(3);
  phases[0].steps[0].name = "Right arm to hands_up";
  phases[0].steps[1].name = "Left arm to hands_up";
  phases[0].steps[2].name = "Wait for next phase";

  // Phase 1: Right arm picks cube
  phases.emplace_back();
  phases[1].name = "Phase 1: Right arm pick";
  phases[1].steps.resize(4);
  phases[1].steps[0].name = "Open right gripper";
  phases[1].steps[1].name = "Move right arm to pick pose";
  phases[1].steps[2].name = "Close right gripper";
  phases[1].steps[3].name = "Wait for next phase";

  // Phase 2: Right arm moves to handover position
  phases.emplace_back();
  phases[2].name = "Phase 2: Right arm → handover pose";
  phases[2].steps.resize(2);
  phases[2].steps[0].name = "Move right arm to handover pose";
  phases[2].steps[1].name = "Wait for next phase";

  // Phase 3: Left arm receives cube
  phases.emplace_back();
  phases[3].name = "Phase 3: Left arm → handover pose";
  phases[3].steps.resize(4);
  phases[3].steps[0].name = "Open left gripper";
  phases[3].steps[1].name = "Move left arm to handover pose";
  phases[3].steps[2].name = "Close left gripper";
  phases[3].steps[3].name = "Wait for next phase";

  // Phase 4: Transfer — right releases and retreats
  phases.emplace_back();
  phases[4].name = "Phase 4: Transfer";
  phases[4].steps.resize(2);
  phases[4].steps[0].name = "Open right gripper";
  phases[4].steps[1].name = "Wait for next phase";

  // Phase 5: Left arm places cube
  phases.emplace_back();
  phases[5].name = "Phase 5: Left arm place";
  phases[5].steps.resize(3);
  phases[5].steps[0].name = "Move left arm to place pose";
  phases[5].steps[1].name = "Open left gripper";
  phases[5].steps[2].name = "Wait for next phase";

  // Phase 6: Both arms return home
  phases.emplace_back();
  phases[6].name = "Phase 6: Return home";
  phases[6].steps.resize(2);
  phases[6].steps[0].name = "Right arm to hands_up";
  phases[6].steps[1].name = "Left arm to hands_up";

  // ── EXECUTE ALL PHASES SYSTEMATICALLY ────────────────────────────────────
  RCLCPP_INFO(node->get_logger(), "═══ HANDOVER DEMO START ═══");
  bool demo_success = true;
  std::string failed_phase_name;

  for (size_t phase_idx = 0; phase_idx < phases.size() && demo_success; ++phase_idx) {
    auto & phase = phases[phase_idx];
    RCLCPP_INFO(node->get_logger(), "");
    RCLCPP_INFO(node->get_logger(), "────────────────────────────────────────");
    RCLCPP_INFO(node->get_logger(), "%s", phase.name.c_str());
    RCLCPP_INFO(node->get_logger(), "────────────────────────────────────────");
    phase.print_status(node);

    // Phase 0: Home
    if (phase_idx == 0) {
      if (!phase.execute_next_step([&]() { commander.goToNamedTarget("hands_up", "right"); return commander.lastCommandSucceeded(); }, "Right arm to hands_up")) { failed_phase_name = phase.name; goto phase_failed; }
      stepDelay(node, "left arm home");
      if (!phase.execute_next_step([&]() { commander.goToNamedTarget("hands_up", "left"); return commander.lastCommandSucceeded(); }, "Left arm to hands_up")) { failed_phase_name = phase.name; goto phase_failed; }
      if (!phase.execute_next_step([&]() { stepDelay(node, "right arm open"); return true; }, "Wait for next phase")) goto phase_failed;
    }
    // Phase 1: Right pick
    else if (phase_idx == 1) {
      if (!phase.execute_next_step([&]() { commander.openGripper("right"); return commander.lastCommandSucceeded(); }, "Open right gripper")) { failed_phase_name = phase.name; goto phase_failed; }
      stepDelay(node, "right arm move to pick");
      if (!phase.execute_next_step([&]() { commander.moveToPose(right_pick_pose, "right"); return commander.lastCommandSucceeded(); }, "Move right arm to pick pose")) { failed_phase_name = phase.name; goto phase_failed; }
      stepDelay(node, "close right gripper");
      if (!phase.execute_next_step([&]() { commander.closeGripper("right"); return commander.lastCommandSucceeded(); }, "Close right gripper")) { failed_phase_name = phase.name; goto phase_failed; }
      stepDelay(node, "right arm move to handover");
      if (!phase.execute_next_step([&]() { return true; }, "Wait for next phase")) goto phase_failed;
    }
    // Phase 2: Right to handover
    else if (phase_idx == 2) {
      if (!phase.execute_next_step([&]() { commander.moveToPose(right_handover_pose, "right"); return commander.lastCommandSucceeded(); }, "Move right arm to handover pose")) { failed_phase_name = phase.name; goto phase_failed; }
      stepDelay(node, "open left gripper");
      if (!phase.execute_next_step([&]() { commander.openGripper("left"); return commander.lastCommandSucceeded(); }, "Wait for next phase")) { failed_phase_name = phase.name; goto phase_failed; }
    }
    // Phase 3: Left handover
    else if (phase_idx == 3) {
      if (!phase.execute_next_step([&]() { commander.openGripper("left"); return commander.lastCommandSucceeded(); }, "Open left gripper")) { failed_phase_name = phase.name; goto phase_failed; }
      stepDelay(node, "left arm move to handover");
      if (!phase.execute_next_step([&]() { commander.moveToPose(left_handover_pose, "left"); return commander.lastCommandSucceeded(); }, "Move left arm to handover pose")) { failed_phase_name = phase.name; goto phase_failed; }
      stepDelay(node, "close left gripper");
      if (!phase.execute_next_step([&]() { commander.closeGripper("left"); return commander.lastCommandSucceeded(); }, "Close left gripper")) { failed_phase_name = phase.name; goto phase_failed; }
      stepDelay(node, "open right gripper and retreat");
      if (!phase.execute_next_step([&]() { commander.openGripper("right"); return commander.lastCommandSucceeded(); }, "Open right gripper")) { failed_phase_name = phase.name; goto phase_failed; }
      stepDelay(node, "left arm move to place");
      if (!phase.execute_next_step([&]() { return true; }, "Wait for next phase")) goto phase_failed;
    }
    // Phase 4: Transfer
    else if (phase_idx == 4) {
      if (!phase.execute_next_step([&]() { return true; }, "Wait for next phase")) goto phase_failed;
    }
    // Phase 5: Place
    else if (phase_idx == 5) {
      if (!phase.execute_next_step([&]() { commander.moveToPose(left_place_pose, "left"); return commander.lastCommandSucceeded(); }, "Move left arm to place pose")) { failed_phase_name = phase.name; goto phase_failed; }
      stepDelay(node, "open left gripper");
      if (!phase.execute_next_step([&]() { commander.openGripper("left"); return commander.lastCommandSucceeded(); }, "Open left gripper")) { failed_phase_name = phase.name; goto phase_failed; }
      stepDelay(node, "return home");
    }
    // Phase 6: Return home
    else if (phase_idx == 6) {
      if (!phase.execute_next_step([&]() { commander.goToNamedTarget("hands_up", "right"); return commander.lastCommandSucceeded(); }, "Right arm to hands_up")) { failed_phase_name = phase.name; goto phase_failed; }
      stepDelay(node, "left arm home");
      if (!phase.execute_next_step([&]() { commander.goToNamedTarget("hands_up", "left"); return commander.lastCommandSucceeded(); }, "Left arm to hands_up")) { failed_phase_name = phase.name; goto phase_failed; }
    }

    phase.completed = phase.is_complete();
    if (phase.completed) {
      RCLCPP_INFO(node->get_logger(), "✓ %s COMPLETED", phase.name.c_str());
    }
  }

  if (demo_success) {
    RCLCPP_INFO(node->get_logger(), "");
    RCLCPP_INFO(node->get_logger(), "═══ HANDOVER DEMO COMPLETE ═══");
  }

  goto demo_end;

  phase_failed:
  RCLCPP_ERROR(node->get_logger(), "✗ Phase %s FAILED - Step not completed", failed_phase_name.c_str());
  demo_success = false;

  demo_end:

  rclcpp::shutdown();
  return 0;
}