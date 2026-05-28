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



install(TARGETS
  commander_lib
  pick_and_place
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include)
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()