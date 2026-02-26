#!/usr/bin/env python3
from typing import Optional, Tuple, List, Dict
from argparse import ArgumentParser
from math import radians, inf, sqrt, atan2, pi, isinf, cos, sin, degrees
from time import sleep, time
import queue

import rospy
from geometry_msgs.msg import Twist, Point32, Vector3, Quaternion, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, PointCloud, ChannelFloat32
from visualization_msgs.msg import MarkerArray, Marker
from tf.transformations import euler_from_quaternion
from std_msgs.msg import ColorRGBA


OBS_FREE_WAYPOINTS = [
    {"x": 1, "y": 1},
    {"x": 2, "y": 1},
    {"x": 1, "y": 0},
]

W_OBS_WAYPOINTS = [
    {"x": 1.5, "y": 1.5},
    {"x": 4, "y": 1},
    {"x": 0, "y": 3.0},
]


def angle_to_0_to_2pi(angle: float) -> float:
    while angle < 0:
        angle += 2 * pi
    while angle > 2 * pi:
        angle -= 2 * pi
    return angle


def map_to_new_range(x: float, a_low: float, a_high: float, b_low: float, b_high: float):
    """Helper function to map a value from range [a_low, a_high] to [b_low, b_high]"""
    y = (x - a_low) / (a_high - a_low) * (b_high - b_low) + b_low
    return y


class PIDController:
    """
    Generates control action taking into account instantaneous error (proportional action),
    accumulated error (integral action) and rate of change of error (derivative action).
    """

    def __init__(self, kP, kI, kD, kS, u_min, u_max):
        assert u_min < u_max, "u_min should be less than u_max"
        # initialize PID variables here
        ######### Your code starts here #########
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.kS = kS
        self.u_min = u_min
        self.u_max = u_max
        self.t_prev = 0
        self.err_prev = 0
        self.integral = 0
        ######### Your code ends here #########

    def control(self, err, t):
        # compute PID control action here
        ######### Your code starts here #########
        dt = t - self.t_prev
        if dt <= 1e-6:
            return 0

        self.integral += err * dt
        self.integral = max(-5.0, min(5.0, self.integral))
        u_raw = self.kP * err + self.kI * self.integral + self.kD * (err - self.err_prev) / dt
        u = max(self.u_min, min(self.u_max, u_raw))
        if self.kS > 0:
            self.integral += (u - u_raw) / self.kS
        self.err_prev = err
        self.t_prev = t
        return u
        ######### Your code ends here #########


class PDController:
    """
    Generates control action taking into account instantaneous error (proportional action)
    and rate of change of error (derivative action).
    """

    def __init__(self, kP, kD, kS, u_min, u_max):
        assert u_min < u_max, "u_min should be less than u_max"
        # Initialize PD variables here
        ######### Your code starts here #########
        self.kP = kP
        self.kD = kD
        self.kS = kS
        self.u_min = u_min
        self.u_max = u_max
        self.t_prev = 0
        self.err_prev = 0
        ######### Your code ends here #########

    def control(self, err, t):
        dt = t - self.t_prev
        ######### Your code starts here #########
        if dt <= 1e-6:
            return 0
        u = self.kS * (self.kP * err + self.kD * (err - self.err_prev) / dt)
        u = max(self.u_min, min(self.u_max, u))
        self.err_prev = err
        self.t_prev = t
        return u
        ######### Your code ends here #########


def publish_waypoints(waypoints: List[Dict], publisher: rospy.Publisher):
    marker_array = MarkerArray()
    for i, waypoint in enumerate(waypoints):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "waypoints"
        marker.id = i
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position = Point(waypoint["x"], waypoint["y"], 0.0)
        marker.pose.orientation = Quaternion(0, 0, 0, 1)
        marker.scale = Vector3(0.1, 0.1, 0.1)
        marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.5)
        marker_array.markers.append(marker)
    publisher.publish(marker_array)


class ObstacleFreeWaypointController:
    def __init__(self, waypoints: List[Dict]):
        rospy.init_node("waypoint_follower", anonymous=True)
        self.waypoints = waypoints
        # Subscriber to the robot's current position (assuming you have Odometry data)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.robot_ctrl_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.waypoint_pub = rospy.Publisher("/waypoints", MarkerArray, queue_size=10)
        sleep(0.5)  # sleep to give time for rviz to subscribe to /waypoints
        publish_waypoints(self.waypoints, self.waypoint_pub)

        self.current_position = None

        # define linear and angular PID controllers here
        ######### Your code starts here #########
        self.linear_controller = PIDController(
            kP=0.5, kI=0.02, kD=0.1, kS=1.0,
            u_min=0.0, u_max=0.5,
        )
        self.angular_controller = PIDController(
            kP=0.5, kI=0.02, kD=0.45, kS=1.0,
            u_min=-2.84, u_max=2.84,
        )
        self.waypoint_tolerance = 0.1
        ######### Your code ends here #########

    def odom_callback(self, msg):
        # Extracting current position from Odometry message
        pose = msg.pose.pose
        orientation = pose.orientation
        _, _, theta = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.current_position = {"x": pose.position.x, "y": pose.position.y, "theta": theta}

    def calculate_error(self, goal_position: Dict) -> Optional[Tuple]:
        """Return distance and angle error between the current position and the provided goal_position. Returns None if
        the current position is not available.
        """
        if self.current_position is None:
            return None

        # Calculate error in position and orientation
        ######### Your code starts here #########
        dx = goal_position["x"] - self.current_position["x"]
        dy = goal_position["y"] - self.current_position["y"]
        distance_error = sqrt(dx * dx + dy * dy)
        angle_to_goal = atan2(dy, dx)
        angle_error = atan2(
            sin(angle_to_goal - self.current_position["theta"]),
            cos(angle_to_goal - self.current_position["theta"]),
        )
        ######### Your code ends here #########

        return distance_error, angle_error

    def control_robot(self):
        rate = rospy.Rate(20)  # 20 Hz
        ctrl_msg = Twist()

        # initialize first waypoint
        current_waypoint_idx = 0

        while not rospy.is_shutdown():

            # Travel through waypoints one at a time, checking if robot is close enough
            ######### Your code starts here #########
            if current_waypoint_idx >= len(self.waypoints):
                ctrl_msg.linear.x = 0.0
                ctrl_msg.angular.z = 0.0
                self.robot_ctrl_pub.publish(ctrl_msg)
                rate.sleep()
                continue

            goal = self.waypoints[current_waypoint_idx]
            err_tuple = self.calculate_error(goal)
            if err_tuple is None:
                rate.sleep()
                continue
            distance_error, angle_error = err_tuple

            if distance_error < self.waypoint_tolerance:
                current_waypoint_idx += 1
                ctrl_msg.linear.x = 0.0
                ctrl_msg.angular.z = 0.0
            else:
                t = time()
                ctrl_msg.linear.x = self.linear_controller.control(distance_error, t)
                ctrl_msg.angular.z = self.angular_controller.control(angle_error, t)
            self.robot_ctrl_pub.publish(ctrl_msg)
            ######### Your code ends here #########
            rate.sleep()


class ObstacleAvoidingWaypointController:
    def __init__(self, waypoints: List[Dict]):
        rospy.init_node("waypoint_follower", anonymous=True)
        self.waypoints = waypoints

        self.current_position = None
        self.laserscan: Optional[LaserScan] = None
        self.laserscan_angles: Optional[List[float]] = None
        self.ir_distance = None
        self.wall_following_desired_distance = 0.5  # set this to whatever you want

        # Subscriber to the robot's current position (assuming you have Odometry data)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.laserscan_sub = rospy.Subscriber("/scan", LaserScan, self.robot_laserscan_callback)

        self.robot_ctrl_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.waypoint_pub = rospy.Publisher("/waypoints", MarkerArray, queue_size=10)
        self.pointcloud_pub = rospy.Publisher("/scan_pointcloud", PointCloud, queue_size=10)

        sleep(0.5)  # sleep to give time for rviz to subscribe to /waypoints
        publish_waypoints(self.waypoints, self.waypoint_pub)

        # Add PID controllers here for obstacle avoidance and waypoint following
        ######### Your code starts here #########
        self.linear_controller = PIDController(
            kP=0.5, kI=0.02, kD=0.1, kS=1.0,
            u_min=0.0, u_max=0.5,
        )
        self.angular_controller = PIDController(
            kP=0.5, kI=0.02, kD=0.45, kS=1.0,
            u_min=-2.84, u_max=2.84,
        )
        self.wall_following_controller = PDController(
            kP=0.3, kD=0.6, kS=1.0,
            u_min=-2.84, u_max=2.84,
        )
        self.waypoint_tolerance = 0.1
        ######### Your code ends here #########

    def robot_laserscan_callback(self, msg: LaserScan):
        self.laserscan = msg
        if self.laserscan_angles is None:
            self.laserscan_angles = [
                self.laserscan.angle_min + i * self.laserscan.angle_increment for i in range(len(self.laserscan.ranges))
            ]
            # sanity check the angles
            assert (abs(self.laserscan.angle_min) < 1e-4) and (abs(self.laserscan_angles[0]) < 1e-4)
            assert abs(self.laserscan.angle_max - 2 * pi) < 1e-4 and (abs(self.laserscan_angles[-1] - 2 * pi) < 1e-4)

        left = msg.ranges[80:100]
        left = [x for x in left if x != inf]
        if len(left) > 0:
            self.ir_distance = sum(left) / len(left)
        else:
            self.ir_distance = None

    def odom_callback(self, msg):
        pose = msg.pose.pose
        orientation = pose.orientation
        _, _, theta = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.current_position = {"x": pose.position.x, "y": pose.position.y, "theta": theta}

    def waypoint_tracking_control(self, goal_position: Dict):

        if self.current_position is None:
            return None

        ######### Your code starts here #########
        dx = goal_position["x"] - self.current_position["x"]
        dy = goal_position["y"] - self.current_position["y"]
        distance_error = sqrt(dx * dx + dy * dy)
        angle_to_goal = atan2(dy, dx)
        angle_error = atan2(
            sin(angle_to_goal - self.current_position["theta"]),
            cos(angle_to_goal - self.current_position["theta"]),
        )
        t = time()
        cmd_linear_vel = self.linear_controller.control(distance_error, t)
        cmd_angular_vel = self.angular_controller.control(angle_error, t)
        ctrl_msg = Twist()
        ctrl_msg.linear.x = cmd_linear_vel
        ctrl_msg.angular.z = cmd_angular_vel
        self.robot_ctrl_pub.publish(ctrl_msg)
        ######### Your code ends here #########

        rospy.loginfo(
            f"distance to target: {distance_error:.2f}\tangle error: {angle_error:.2f}\tcommanded linear vel: {cmd_linear_vel:.2f}\tcommanded angular vel: {cmd_angular_vel:.2f}"
        )

    def obstacle_avoiding_control(self, visualize: bool = True):

        ctrl_msg = Twist()

        ######### Your code starts here #########
        # Wall-following (Lab 4): err = ir_distance - desired => too close gives negative err => turn right
        if self.ir_distance is None:
            u = 0.0
            ctrl_msg.linear.x = 0.1
            ctrl_msg.angular.z = 0.0
        else:
            err = self.ir_distance - self.wall_following_desired_distance
            u = self.wall_following_controller.control(err, time())
            ctrl_msg.linear.x = 0.15
            ctrl_msg.angular.z = u
        ######### Your code ends here #########

        self.robot_ctrl_pub.publish(ctrl_msg)
        if self.ir_distance is not None and u is not None:
            print(
                f"dist: {round(self.ir_distance, 4)}\ttgt: {round(self.wall_following_desired_distance, 4)}\tu: {round(u, 4)}"
            )

    def laserscan_distances_to_point(self, point: Dict, cone_angle: float, visualize: bool = False):
        """Returns the laserscan distances within the cone of angle `cone_angle` centered about the line pointing from
        the robots current position to the given point. Angles are in radians.

        Notes:
            1. Distances that are outside of the laserscan's minimum and maximum range are filterered out
        """
        curr_pos = self.current_position
        # angle to point in the local frame. this is the same as the lidar frame. Not neccessarily in [-pi, pi] because
        # of the theta subtraction
        angle_to_point_local = angle_to_0_to_2pi(
            atan2(point["y"] - curr_pos["y"], point["x"] - curr_pos["x"]) - curr_pos["theta"]
        )
        angle_low = angle_to_0_to_2pi(angle_to_point_local - cone_angle)
        angle_high = angle_to_0_to_2pi(angle_to_point_local + cone_angle)

        # This is the so called 'danger zone', because either the high or low angle has wrapped around. For example,
        # when low = 355 deg, and high = 20 deg. The solution is to set the low to 0 and use the high when angle is > 0,
        # or set the high to 2*pi and use the low when angle is < 2*pi
        if angle_to_point_local < cone_angle or angle_to_point_local > 2 * pi - cone_angle:
            if angle_to_point_local < cone_angle:
                angle_low = 0
                idx_low = 0
                idx_high = int(
                    map_to_new_range(
                        angle_high, self.laserscan.angle_min, self.laserscan.angle_max, 0, len(self.laserscan.ranges)
                    )
                )
            elif angle_to_point_local > 2 * pi - cone_angle:
                angle_high = 2 * pi
                idx_high = len(self.laserscan.ranges) - 1
                idx_low = int(
                    map_to_new_range(
                        angle_low, self.laserscan.angle_min, self.laserscan.angle_max, 0, len(self.laserscan.ranges)
                    )
                )
            else:
                assert False, "should not reach here"
        else:
            idx_low = int(
                map_to_new_range(
                    angle_low, self.laserscan.angle_min, self.laserscan.angle_max, 0, len(self.laserscan.ranges)
                )
            )
            idx_high = int(
                map_to_new_range(
                    angle_high, self.laserscan.angle_min, self.laserscan.angle_max, 0, len(self.laserscan.ranges)
                )
            )
        assert angle_low < angle_high, f"angle_low: {angle_low}, angle_high: {angle_high}"
        if idx_low > idx_high:
            idx_low, idx_high = idx_high, idx_low
        assert idx_low < idx_high, f"idx_low: {idx_low}, idx_high: {idx_high}"

        raw = self.laserscan.ranges[idx_low:idx_high]
        filtered = [r for r in raw if (r > self.laserscan.range_min and r < self.laserscan.range_max)]

        if visualize:
            # raw should include all ranges, even if they are inf, in the specified cone
            #   i.e. something like `raw = self.laserscan.ranges[idx_low:idx_high]`
            # `angle_low` and `angle_high` are the angles in the robots local frame
            pcd = PointCloud()
            pcd.header.frame_id = "odom"
            pcd.header.stamp = rospy.Time.now()
            for i, p in enumerate(raw):
                if isinf(p):
                    continue
                angle_local = map_to_new_range(i, 0, len(raw), angle_low, angle_high)
                angle = angle_local + curr_pos["theta"]
                x = p * cos(angle) + curr_pos["x"]
                y = p * sin(angle) + curr_pos["y"]
                z = 0.1
                pcd.points.append(Point32(x=x, y=y, z=z))
                pcd.channels.append(ChannelFloat32(name="rgb", values=(0.0, 1.0, 0.0)))
            self.pointcloud_pub.publish(pcd)
        return filtered

    def control_robot(self):
        rate = rospy.Rate(10)  # 20 Hz

        current_waypoint_idx = 0
        distance_from_wall_safety = 1.0
        cone_angle = radians(5)

        while not rospy.is_shutdown():

            if self.current_position is None or self.laserscan is None:
                sleep(0.01)
                continue

            # Travel through waypoints, checking if there is an obstacle in the way. Transition to obstacle avoidance if necessary
            ######### Your code starts here #########
            threshold_enter = 0.5   # switch to wall_following when obstacle in cone closer than this (m)
            threshold_leave = 1.0   # switch back to go_to_goal when cone clear beyond this (hysteresis)
            cone_angle_deg = 20   # wider cone (deg) so we detect obstacle earlier and return to waypoint safely
            mode = getattr(self, "_arb_mode", "go_to_goal")

            if current_waypoint_idx >= len(self.waypoints):
                ctrl_msg = Twist()
                ctrl_msg.linear.x = 0.0
                ctrl_msg.angular.z = 0.0
                self.robot_ctrl_pub.publish(ctrl_msg)
                rate.sleep()
                continue

            goal = self.waypoints[current_waypoint_idx]
            distances_in_cone = self.laserscan_distances_to_point(goal, radians(cone_angle_deg), visualize=False)
            min_dist_in_cone = min(distances_in_cone) if len(distances_in_cone) > 0 else inf

            # Arbitration: when obstacle blocks path to goal use Wall-Following; else Go-To-Goal
            if mode == "go_to_goal":
                if len(distances_in_cone) > 0 and min_dist_in_cone < threshold_enter:
                    mode = "wall_following"
            else:  # wall_following
                if len(distances_in_cone) == 0 or min_dist_in_cone > threshold_leave:
                    mode = "go_to_goal"

            self._arb_mode = mode

            if mode == "wall_following":
                self.obstacle_avoiding_control(visualize=True)
            else:
                self.waypoint_tracking_control(goal)
                dx = goal["x"] - self.current_position["x"]
                dy = goal["y"] - self.current_position["y"]
                dist = sqrt(dx * dx + dy * dy)
                if dist < self.waypoint_tolerance:
                    current_waypoint_idx += 1
            ######### Your code ends here #########
            rate.sleep()


""" Example usage

rosrun development lab6_7_sim.py --mode obstacle_free
rosrun development lab6_7_sim.py --mode obstacle_avoiding
"""


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--mode", type=str, required=True, help="Mode of operation: 'obstacle_free' or 'obstacle_avoiding'"
    )
    args = parser.parse_args()
    assert args.mode in {"obstacle_free", "obstacle_avoiding"}

    if args.mode == "obstacle_free":
        controller = ObstacleFreeWaypointController(OBS_FREE_WAYPOINTS)
    else:
        controller = ObstacleAvoidingWaypointController(W_OBS_WAYPOINTS)

    try:
        controller.control_robot()
    except rospy.ROSInterruptException:
        print("Shutting down...")
