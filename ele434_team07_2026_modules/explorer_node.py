#!/usr/bin/env python3
import math
import random
from enum import Enum

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


class RobotState(Enum):
    FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    ESCAPE = 4


class ExplorerNode(Node):
    def __init__(self):
        super().__init__('explorer_node')

        self.declare_parameter('forward_speed', 0.24)
        self.declare_parameter('turn_speed', 0.9)
        self.declare_parameter('slow_turn_forward_speed', 0.14)
        self.declare_parameter('caution_speed', 0.16)

        self.declare_parameter('stop_dist', 0.25)
        self.declare_parameter('caution_dist', 0.4)
        self.declare_parameter('side_block_dist', 0.18)
        self.declare_parameter('rear_block_dist', 0.17)

        self.declare_parameter('turn_duration', 0.9)
        self.declare_parameter('escape_duration', 1.7)

        self.declare_parameter('front_half_angle_deg', 30.0)
        self.declare_parameter('rear_half_angle_deg', 30.0)
        self.declare_parameter('side_half_angle_deg', 60.0)

        self.declare_parameter('max_runtime', 90.0)

        # Visited-memory parameters
        self.declare_parameter('visited_cell_size', 0.40)
        self.declare_parameter('visited_lookahead', 0.80)
        self.declare_parameter('visited_weight', 0.25)
        self.declare_parameter('visit_turn_gain', 0.18)
        self.declare_parameter('visit_mark_radius_cells', 1)

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

        self.visited_cell_size = self.get_parameter('visited_cell_size').value
        self.visited_lookahead = self.get_parameter('visited_lookahead').value
        self.visited_weight = self.get_parameter('visited_weight').value
        self.visit_turn_gain = self.get_parameter('visit_turn_gain').value
        self.visit_mark_radius_cells = self.get_parameter('visit_mark_radius_cells').value

        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.timer = self.create_timer(0.1, self.control_loop)

        self.state = RobotState.FORWARD
        self.state_time = 0.0
        self.control_dt = 0.1

        self.scan_ready = False
        self.odom_ready = False
        self.run_finished = False

        self.front_dist = float('inf')
        self.rear_dist = float('inf')
        self.left_dist = float('inf')
        self.right_dist = float('inf')

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.min_valid_range = 0.05
        self.max_valid_range = 3.5

        self.preferred_turn_left = True
        self.start_time = self.get_clock().now().nanoseconds

        # key: (ix, iy), value: visit count
        self.visited_cells = {}

        self.get_logger().info(
            f'Explorer node started. '
            f'forward_speed={self.forward_speed}, '
            f'turn_speed={self.turn_speed}, '
            f'stop_dist={self.stop_dist}, '
            f'caution_dist={self.caution_dist}, '
            f'max_runtime={self.max_runtime}'
        )

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

    def scan_callback(self, msg: LaserScan):
        ranges = list(msg.ranges)
        n = len(ranges)

        if n == 0:
            return

        angles = [msg.angle_min + i * msg.angle_increment for i in range(n)]

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

    def world_to_cell(self, x, y):
        ix = int(round(x / self.visited_cell_size))
        iy = int(round(y / self.visited_cell_size))
        return ix, iy

    def mark_visited(self):
        if not self.odom_ready:
            return

        center_ix, center_iy = self.world_to_cell(self.robot_x, self.robot_y)

        for dx in range(-self.visit_mark_radius_cells, self.visit_mark_radius_cells + 1):
            for dy in range(-self.visit_mark_radius_cells, self.visit_mark_radius_cells + 1):
                key = (center_ix + dx, center_iy + dy)
                self.visited_cells[key] = self.visited_cells.get(key, 0) + 1

    def projected_visit_cost(self, relative_angle_deg):
        if not self.odom_ready:
            return 0.0

        samples = [0.6 * self.visited_lookahead, self.visited_lookahead, 1.4 * self.visited_lookahead]
        total = 0.0

        heading = self.robot_yaw + math.radians(relative_angle_deg)

        for d in samples:
            px = self.robot_x + d * math.cos(heading)
            py = self.robot_y + d * math.sin(heading)
            key = self.world_to_cell(px, py)
            total += self.visited_cells.get(key, 0)

        return total / len(samples)

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def set_state(self, new_state: RobotState):
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            self.state_time = 0.0
            self.get_logger().info(
                f'STATE switched: {old_state.value} {old_state.name} -> {new_state.value} {new_state.name}'
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

        if self.odom_ready:
            self.mark_visited()

        front_blocked = self.front_dist < self.stop_dist
        rear_blocked = self.rear_dist < self.rear_block_dist
        left_tight = self.left_dist < self.side_block_dist
        right_tight = self.right_dist < self.side_block_dist

        left_visit = self.projected_visit_cost(0)
        right_visit = self.projected_visit_cost(0)
        forward_visit = self.projected_visit_cost(0.0)

        cmd = TwistStamped()

        if self.state == RobotState.FORWARD:
            if front_blocked:
                left_score = self.left_dist - self.visited_weight * left_visit
                right_score = self.right_dist - self.visited_weight * right_visit

                if left_score > right_score + 0.05:
                    self.preferred_turn_left = True
                    self.set_state(RobotState.TURN_LEFT)
                elif right_score > left_score + 0.05:
                    self.preferred_turn_left = False
                    self.set_state(RobotState.TURN_RIGHT)
                else:
                    self.preferred_turn_left = random.choice([True, False])
                    self.set_state(RobotState.ESCAPE)
            else:
                if self.front_dist < self.caution_dist:
                    cmd.twist.linear.x = self.caution_speed
                else:
                    cmd.twist.linear.x = self.forward_speed

                obstacle_steer = 0.0
                if self.left_dist < self.right_dist - 0.08:
                    obstacle_steer = -0.25
                elif self.right_dist < self.left_dist - 0.08:
                    obstacle_steer = 0.25

                visit_steer = 0.0
                if left_visit + 0.5 < right_visit:
                    visit_steer = self.visit_turn_gain
                elif right_visit + 0.5 < left_visit:
                    visit_steer = -self.visit_turn_gain

                # If straight ahead is heavily revisited, encourage a gentle side bias
                if forward_visit > min(left_visit, right_visit) + 1.0:
                    if left_visit < right_visit:
                        visit_steer += 0.08
                    elif right_visit < left_visit:
                        visit_steer -= 0.08

                cmd.twist.angular.z = self.clamp(obstacle_steer + visit_steer, -0.5, 0.5)

        elif self.state == RobotState.TURN_LEFT:
            if front_blocked:
                cmd.twist.linear.x = 0.0
            else:
                cmd.twist.linear.x = self.slow_turn_forward_speed

            cmd.twist.angular.z = self.turn_speed

            if (not front_blocked and self.left_dist > self.caution_dist) or self.state_time > self.turn_duration:
                self.set_state(RobotState.FORWARD)

        elif self.state == RobotState.TURN_RIGHT:
            if front_blocked:
                cmd.twist.linear.x = 0.0
            else:
                cmd.twist.linear.x = self.slow_turn_forward_speed

            cmd.twist.angular.z = -self.turn_speed

            if (not front_blocked and self.right_dist > self.caution_dist) or self.state_time > self.turn_duration:
                self.set_state(RobotState.FORWARD)

        elif self.state == RobotState.ESCAPE:
            if rear_blocked:
                cmd.twist.linear.x = 0.0
            else:
                cmd.twist.linear.x = -0.03

            # Bias escape turn toward the less-visited side when possible
            if left_visit + 0.5 < right_visit:
                self.preferred_turn_left = True
            elif right_visit + 0.5 < left_visit:
                self.preferred_turn_left = False

            if self.preferred_turn_left:
                cmd.twist.angular.z = self.turn_speed
            else:
                cmd.twist.angular.z = -self.turn_speed

            if self.state_time > self.escape_duration:
                self.set_state(RobotState.FORWARD)

        if self.state == RobotState.FORWARD and front_blocked and left_tight and right_tight:
            if left_visit + 0.5 < right_visit:
                self.preferred_turn_left = True
            elif right_visit + 0.5 < left_visit:
                self.preferred_turn_left = False
            else:
                self.preferred_turn_left = random.choice([True, False])

            self.set_state(RobotState.ESCAPE)

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ExplorerNode()

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