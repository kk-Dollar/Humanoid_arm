#!/usr/bin/env python3
# NEW FILE — created for ArUco perception pipeline
# Does NOT replace or modify any existing file

from __future__ import annotations

from typing import Optional

import rclpy
from geometry_msgs.msg import Point, PoseStamped, Vector3
from rclpy.node import Node
from std_msgs.msg import Bool, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

MARKER_SIZE = 0.05
AXIS_LEN = 0.1
PUBLISH_RATE_HZ = 10.0
CUBE_HEIGHT = 0.06


class ArucoPoseVisualizer(Node):
    """Visualizes detected cube pose and local axes in RViz."""

    def __init__(self) -> None:
        """
        WHAT: Initializes visualizer subscriptions and marker publishers.
        WHY: Gives direct RViz feedback for detection validity and orientation.
        INPUT: /cube_pose and /cube_detected topic streams.
        OUTPUT: /cube_marker and /cube_axes marker publications.
        """
        super().__init__("aruco_pose_visualizer")

        self.last_pose: Optional[PoseStamped] = None
        self.detected = False

        # Subscription: detected cube pose flowing from aruco_detector.
        self.pose_sub = self.create_subscription(PoseStamped, "/cube_pose", self.pose_cb, 10)
        # Subscription: detection state flowing from aruco_detector.
        self.detect_sub = self.create_subscription(Bool, "/cube_detected", self.detect_cb, 10)

        # Publication: cube marker for RViz state visualization.
        self.marker_pub = self.create_publisher(Marker, "/cube_marker", 10)
        # Publication: axis arrows showing cube pose orientation.
        self.axes_pub = self.create_publisher(MarkerArray, "/cube_axes", 10)

        self.timer = self.create_timer(1.0 / PUBLISH_RATE_HZ, self.publish_markers)

    def pose_cb(self, msg: PoseStamped) -> None:
        """
        WHAT: Stores latest cube pose.
        WHY: Marker publisher needs most recent pose for RViz.
        INPUT: PoseStamped from /cube_pose.
        OUTPUT: Updates internal pose cache.
        """
        self.last_pose = msg

    def detect_cb(self, msg: Bool) -> None:
        """
        WHAT: Stores current cube visibility state.
        WHY: Marker color should reflect detection confidence.
        INPUT: Bool from /cube_detected.
        OUTPUT: Updates internal detected flag.
        """
        self.detected = bool(msg.data)

    def publish_markers(self) -> None:
        """
        WHAT: Publishes cube marker and XYZ axis markers at fixed rate.
        WHY: Continuous RViz updates simplify debugging during runtime.
        INPUT: Timer event and latest cached pose/detection state.
        OUTPUT: /cube_marker and /cube_axes marker publications.
        """
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "cube"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.scale = Vector3(x=MARKER_SIZE, y=MARKER_SIZE, z=MARKER_SIZE)
        marker.color = ColorRGBA(
            r=0.0 if self.detected else 1.0,
            g=1.0 if self.detected else 0.0,
            b=0.0,
            a=0.7,
        )
        if self.last_pose is not None:
            marker.pose = self.last_pose.pose
            marker.pose.position.z -= CUBE_HEIGHT * 0.5
        else:
            marker.pose.orientation.w = 1.0
        self.marker_pub.publish(marker)

        axes = MarkerArray()
        if self.last_pose is not None:
            for idx, (axis_name, color, direction) in enumerate(
                [
                    ("x", ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0), (AXIS_LEN, 0.0, 0.0)),
                    ("y", ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), (0.0, AXIS_LEN, 0.0)),
                    ("z", ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0), (0.0, 0.0, AXIS_LEN)),
                ]
            ):
                arrow = Marker()
                arrow.header.frame_id = "world"
                arrow.header.stamp = marker.header.stamp
                arrow.ns = "cube_axes"
                arrow.id = idx
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.color = color
                arrow.scale = Vector3(x=0.01, y=0.02, z=0.02)
                arrow.pose = self.last_pose.pose
                arrow.pose.position.z -= CUBE_HEIGHT * 0.5
                arrow.points = [Point(x=0.0, y=0.0, z=0.0), Point(x=direction[0], y=direction[1], z=direction[2])]
                axes.markers.append(arrow)
        self.axes_pub.publish(axes)


def main(args=None) -> None:
    """
    WHAT: Entry point for pose visualizer node.
    WHY: Runs periodic RViz marker publication loop.
    INPUT: Optional ROS CLI args.
    OUTPUT: Running visualizer process until shutdown.
    """
    rclpy.init(args=args)
    node = ArucoPoseVisualizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
