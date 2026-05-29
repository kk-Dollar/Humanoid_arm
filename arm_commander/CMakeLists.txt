cmake_minimum_required(VERSION 3.8)
project(arm_commander)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(example_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)

include_directories(include)

add_library(commander_lib
  src/commander.cpp)
target_include_directories(commander_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(commander_lib
  ament_index_cpp
  rclcpp
  moveit_ros_planning_interface
  example_interfaces
  geometry_msgs
  std_msgs)

add_executable(pick_and_place src/pick_and_place.cpp)
target_link_libraries(pick_and_place commander_lib)
target_include_directories(pick_and_place PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
ament_target_dependencies(pick_and_place
  ament_index_cpp
  rclcpp
  moveit_ros_planning_interface
  example_interfaces
  geometry_msgs
  std_msgs)

# NOTE: aruco_pick_and_place.cpp does not exist in this workspace;
# the new wrist_servo_pick_and_place supersedes it.

add_executable(wrist_servo_pick_and_place
  src/wrist_servo_pick_and_place.cpp)
target_link_libraries(wrist_servo_pick_and_place commander_lib)
target_include_directories(wrist_servo_pick_and_place PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
ament_target_dependencies(wrist_servo_pick_and_place
  rclcpp
  moveit_ros_planning_interface
  geometry_msgs
  std_msgs
  std_srvs
  sensor_msgs
  tf2_ros
  tf2_geometry_msgs
  ament_index_cpp)


install(TARGETS
  commander_lib
  pick_and_place
  wrist_servo_pick_and_place
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  scripts/aruco_detector.py
  scripts/camera_publisher.py
  scripts/aruco_pose_visualizer.py
  scripts/wrist_servo_base.py
  scripts/right_wrist_servo.py
  scripts/left_wrist_servo.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include)
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY mujoco/
  DESTINATION share/${PROJECT_NAME}/mujoco)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()