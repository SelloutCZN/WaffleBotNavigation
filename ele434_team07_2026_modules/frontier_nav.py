#!/usr/bin/env python3
import math
import random
from enum import Enum

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry


class RobotState(Enum):
    SEEK_FRONTIER = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    ESCAPE = 4


class FrontierNavNode(Node):
    def __init__(self):
        super().__init__('frontier_nav')

        # Reuse existing launch parameters
        self.declare_parameter('forward_speed', 0.14)
        self.declare_parameter('turn_speed', 0.9)
        self.declare_parameter('slow_turn_forward_speed', 0.04)
        self.declare_parameter('caution_speed', 0.12)

        self.declare_parameter('stop_dist', 0.38)
        self.declare_parameter('caution_dist', 0.58)
        self.declare_parameter('side_block_dist', 0.30)
        self.declare_parameter('rear_block_dist', 0.25)

        self.declare_parameter('turn_duration', 0.9)
        self.declare_parameter('escape_duration', 1.7)

        self.declare_parameter('front_half_angle_deg', 45.0)
        self.declare_parameter('rear_half_angle_deg', 45.0)
        self.declare_parameter('side_half_angle_deg', 45.0)

        self.declare_parameter('max_runtime', 90.0)

        # Frontier-specific parameters
        self.declare_parameter('goal_reach_dist', 0.35)
        self.declare_parameter('frontier_min_unknown_neighbors', 1)
        self.declare_parameter('frontier_sample_stride', 2)
        self.declare_parameter('goal_heading_gain', 1.2)
        self.declare_parameter('max_goal_turn', 0.7)
        self.declare_parameter('goal_refresh_period', 2.0)
        self.declare_parameter('min_frontier_distance', 0.6)
        self.declare_parameter('max_frontier_distance', 6.0)

        self.forward_speed = self.get_parameter('forward_speed').value
        self.turn_speed = self.get_parameter('turn_speed').value
        self.slow_turn_forward_speed = self.get_parameter('slow_turn_forward_speed').value
        self.caution_speed = self.get_parameter('caution_speed').value

        self.stop_dist = self.get_parameter('stop_dist').value
        self.caution_dist = self.get_parameter('caution_dist').value
        self.side_block_dist = self.get_parameter('side_block_dist').value
        self.rear_block_dist = self.get_parameter('rear_block_dist').value

        self.turn_duration = self.get_parameter('turn_duration').value
        self.escape_duration = self.get_parameter('escape_duration').value

        self.front_half_angle_deg = self.get_parameter('front_half_angle_deg').value
        self.rear_half_angle_deg = self.get_parameter('rear_half_angle_deg').value
        self.side_half_angle_deg = self.get_parameter('side_half_angle_deg').value

        self.max_runtime = self.get_parameter('max_runtime').value

        self.goal_reach_dist = self.get_parameter('goal_reach_dist').value
        self.frontier_min_unknown_neighbors = self.get_parameter('frontier_min_unknown_neighbors').value
        self.frontier_sample_stride = self.get_parameter('frontier_sample_stride').value
        self.goal_heading_gain = self.get_parameter('goal_heading_gain').value
        self.max_goal_turn = self.get_parameter('max_goal_turn').value
        self.goal_refresh_period = self.get_parameter('goal_refresh_period').value
        self.min_frontier_distance = self.get_parameter('min_frontier_distance').value
        self.max_frontier_distance = self.get_parameter('max_frontier_distance').value

        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)

        self.timer = self.create_timer(0.1, self.control_loop)

        self.state = RobotState.SEEK_FRONTIER
        self.state_time = 0.0
        self.control_dt = 0.1

        self.scan_ready = False
        self.odom_ready = False
        self.map_ready = False
        self.run_finished = False

        self.front_dist = float('inf')
        self.rear_dist = float('inf')
        self.left_dist = float('inf')
        self.right_dist = float('inf')

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.map_msg = None
        self.goal_x = None
        self.goal_y = None
        self.last_goal_update_time = 0.0

        self.min_valid_range = 0.05
        self.max_valid_range = 3.5

        self.preferred_turn_left = True
        self.start_time = self.get_clock().now().nanoseconds

        self.get_logger().info('Frontier navigation node started.')

    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

        self.odom_ready = True

    def map_callback(self, msg: OccupancyGrid):
        self.map_msg = msg
        self.map_ready = True

    def scan_callback(self, msg: LaserScan):
        ranges = list(msg.ranges)
        if len(ranges) == 0:
            return

        angles = [msg.angle_min + i * msg.angle_increment for i in range(len(ranges))]

        def wrap_deg(deg):
            if deg < -180.0:
                deg += 360.0
            elif deg > 180.0:
                deg -= 360.0
            return deg

        def sector_min(angle_deg_min, angle_deg_max):
            vals = []
            for r, a in zip(ranges, angles):
                deg = wrap_deg(math.degrees(a))
                if angle_deg_min <= deg <= angle_deg_max:
                    if math.isfinite(r) and self.min_valid_range < r < self.max_valid_range:
                        vals.append(r)
            if not vals:
                return float('inf')
            return min(vals)

        def rear_sector_min():
            vals = []
            for r, a in zip(ranges, angles):
                deg = wrap_deg(math.degrees(a))

                in_rear_positive = 180.0 - self.rear_half_angle_deg <= deg <= 180.0
                in_rear_negative = -180.0 <= deg <= -180.0 + self.rear_half_angle_deg

                if in_rear_positive or in_rear_negative:
                    if math.isfinite(r) and self.min_valid_range < r < self.max_valid_range:
                        vals.append(r)

            if not vals:
                return float('inf')
            return min(vals)

        self.front_dist = sector_min(-self.front_half_angle_deg, self.front_half_angle_deg)
        self.left_dist = sector_min(90.0 - self.side_half_angle_deg, 90.0 + self.side_half_angle_deg)
        self.right_dist = sector_min(-90.0 - self.side_half_angle_deg, -90.0 + self.side_half_angle_deg)
        self.rear_dist = rear_sector_min()

        self.scan_ready = True

    def set_state(self, new_state: RobotState):
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            self.state_time = 0.0
            self.get_logger().info(
                f'STATE switched: {old_state.value} {old_state.name} -> {new_state.value} {new_state.name}'
            )

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def angle_wrap(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def map_index(self, mx, my, width):
        return my * width + mx

    def cell_to_world(self, mx, my):
        info = self.map_msg.info
        wx = info.origin.position.x + (mx + 0.5) * info.resolution
        wy = info.origin.position.y + (my + 0.5) * info.resolution
        return wx, wy

    def world_to_cell(self, wx, wy):
        info = self.map_msg.info
        mx = int((wx - info.origin.position.x) / info.resolution)
        my = int((wy - info.origin.position.y) / info.resolution)
        return mx, my

    def is_in_map(self, mx, my):
        info = self.map_msg.info
        return 0 <= mx < info.width and 0 <= my < info.height

    def is_frontier_cell(self, mx, my, data, width, height):
        idx = self.map_index(mx, my, width)
        if data[idx] != 0:
            return False

        unknown_neighbors = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = mx + dx
            ny = my + dy
            if 0 <= nx < width and 0 <= ny < height:
                nidx = self.map_index(nx, ny, width)
                if data[nidx] == -1:
                    unknown_neighbors += 1

        return unknown_neighbors >= self.frontier_min_unknown_neighbors

    def choose_frontier_goal(self):
        if not self.map_ready or not self.odom_ready:
            return

        info = self.map_msg.info
        width = info.width
        height = info.height
        data = self.map_msg.data

        best_goal = None
        best_score = float('inf')

        stride = max(1, int(self.frontier_sample_stride))

        for my in range(1, height - 1, stride):
            for mx in range(1, width - 1, stride):
                if not self.is_frontier_cell(mx, my, data, width, height):
                    continue

                wx, wy = self.cell_to_world(mx, my)

                dx = wx - self.robot_x
                dy = wy - self.robot_y
                dist = math.hypot(dx, dy)

                if dist < self.min_frontier_distance or dist > self.max_frontier_distance:
                    continue

                heading = math.atan2(dy, dx)
                heading_error = abs(self.angle_wrap(heading - self.robot_yaw))

                # Prefer nearby frontiers, but mildly prefer those roughly ahead
                score = dist + 0.4 * heading_error

                if score < best_score:
                    best_score = score
                    best_goal = (wx, wy)

        if best_goal is not None:
            self.goal_x, self.goal_y = best_goal
            self.get_logger().info(
                f'New frontier goal: x={self.goal_x:.2f}, y={self.goal_y:.2f}'
            )

    def control_loop(self):
        if not self.scan_ready:
            return

        elapsed_time = (self.get_clock().now().nanoseconds - self.start_time) * 1e-9
        if elapsed_time >= self.max_runtime:
            if not self.run_finished:
                stop = TwistStamped()
                self.cmd_pub.publish(stop)
                self.get_logger().info('Max runtime reached. Robot stopped.')
                self.run_finished = True
            return

        self.state_time += self.control_dt

        front_blocked = self.front_dist < self.stop_dist
        rear_blocked = self.rear_dist < self.rear_block_dist
        left_tight = self.left_dist < self.side_block_dist
        right_tight = self.right_dist < self.side_block_dist

        now_sec = elapsed_time
        if self.goal_x is None or self.goal_y is None or (now_sec - self.last_goal_update_time) > self.goal_refresh_period:
            self.choose_frontier_goal()
            self.last_goal_update_time = now_sec

        cmd = TwistStamped()

        if self.state == RobotState.SEEK_FRONTIER:
            if front_blocked:
                if self.left_dist > self.right_dist + 0.05:
                    self.preferred_turn_left = True
                    self.set_state(RobotState.TURN_LEFT)
                elif self.right_dist > self.left_dist + 0.05:
                    self.preferred_turn_left = False
                    self.set_state(RobotState.TURN_RIGHT)
                else:
                    self.preferred_turn_left = random.choice([True, False])
                    self.set_state(RobotState.ESCAPE)
            else:
                if self.goal_x is not None and self.goal_y is not None:
                    dx = self.goal_x - self.robot_x
                    dy = self.goal_y - self.robot_y
                    goal_dist = math.hypot(dx, dy)

                    if goal_dist < self.goal_reach_dist:
                        self.goal_x = None
                        self.goal_y = None
                    else:
                        goal_heading = math.atan2(dy, dx)
                        heading_error = self.angle_wrap(goal_heading - self.robot_yaw)

                        if self.front_dist < self.caution_dist:
                            cmd.twist.linear.x = self.caution_speed
                        else:
                            cmd.twist.linear.x = self.forward_speed

                        obstacle_steer = 0.0
                        if self.left_dist < self.right_dist - 0.08:
                            obstacle_steer = -0.25
                        elif self.right_dist < self.left_dist - 0.08:
                            obstacle_steer = 0.25

                        goal_steer = self.goal_heading_gain * heading_error
                        goal_steer = self.clamp(goal_steer, -self.max_goal_turn, self.max_goal_turn)

                        cmd.twist.angular.z = self.clamp(obstacle_steer + goal_steer, -1.0, 1.0)
                else:
                    if self.front_dist < self.caution_dist:
                        cmd.twist.linear.x = self.caution_speed
                    else:
                        cmd.twist.linear.x = self.forward_speed

                    if self.left_dist < self.right_dist - 0.08:
                        cmd.twist.angular.z = -0.25
                    elif self.right_dist < self.left_dist - 0.08:
                        cmd.twist.angular.z = 0.25

        elif self.state == RobotState.TURN_LEFT:
            cmd.twist.angular.z = self.turn_speed

            if front_blocked:
                cmd.twist.linear.x = 0.0
            else:
                cmd.twist.linear.x = self.slow_turn_forward_speed

            if (not front_blocked and self.left_dist > self.caution_dist) or self.state_time > self.turn_duration:
                self.set_state(RobotState.SEEK_FRONTIER)

        elif self.state == RobotState.TURN_RIGHT:
            cmd.twist.angular.z = -self.turn_speed

            if front_blocked:
                cmd.twist.linear.x = 0.0
            else:
                cmd.twist.linear.x = self.slow_turn_forward_speed

            if (not front_blocked and self.right_dist > self.caution_dist) or self.state_time > self.turn_duration:
                self.set_state(RobotState.SEEK_FRONTIER)

        elif self.state == RobotState.ESCAPE:
            if rear_blocked:
                cmd.twist.linear.x = 0.0
            else:
                cmd.twist.linear.x = -0.03

            if self.preferred_turn_left:
                cmd.twist.angular.z = self.turn_speed
            else:
                cmd.twist.angular.z = -self.turn_speed

            if self.state_time > self.escape_duration:
                self.set_state(RobotState.SEEK_FRONTIER)

        if self.state == RobotState.SEEK_FRONTIER and front_blocked and left_tight and right_tight:
            self.preferred_turn_left = random.choice([True, False])
            self.set_state(RobotState.ESCAPE)

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierNavNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = TwistStamped()
        node.cmd_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()