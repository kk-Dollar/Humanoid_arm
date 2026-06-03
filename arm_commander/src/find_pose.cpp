#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.hpp>
#include <vector>

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("find_pose_node");
    moveit::planning_interface::MoveGroupInterface move_group(node, "right_arm");
    
    // Straight down orientation
    double qx = -0.007, qy = -0.104, qz = 0.002, qw = 0.995;
    
    std::vector<double> xs = {0.15, 0.20, 0.22, 0.25, 0.28, 0.30};
    std::vector<double> ys = {-0.15, -0.10, -0.05, 0.0};
    
    for(double x : xs) {
        for(double y : ys) {
            geometry_msgs::msg::Pose hover;
            hover.position.x = x;
            hover.position.y = y;
            hover.position.z = 0.45;
            hover.orientation.x = qx;
            hover.orientation.y = qy;
            hover.orientation.z = qz;
            hover.orientation.w = qw;
            
            move_group.setPoseTarget(hover);
            moveit::planning_interface::MoveGroupInterface::Plan my_plan;
            bool success = (move_group.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
            
            if(success) {
                std::vector<geometry_msgs::msg::Pose> waypoints;
                geometry_msgs::msg::Pose grasp = hover;
                grasp.position.z = 0.25;
                waypoints.push_back(grasp);
                
                moveit_msgs::msg::RobotTrajectory trajectory;
                double fraction = move_group.computeCartesianPath(waypoints, 0.01, 0.0, trajectory);
                
                RCLCPP_INFO(node->get_logger(), "X: %.2f, Y: %.2f -> Hover: OK, Cartesian: %.1f%%", x, y, fraction * 100.0);
            } else {
                RCLCPP_INFO(node->get_logger(), "X: %.2f, Y: %.2f -> Hover: FAILED", x, y);
            }
        }
    }
    
    rclcpp::shutdown();
    return 0;
}
