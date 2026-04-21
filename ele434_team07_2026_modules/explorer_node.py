#!/usr/bin/env python3
import math
import random
from enum import Enum

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan


class RobotState(Enum):
    FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    ESCAPE = 4


class ExplorerNode(Node):
    def __init__(self):
        super().__init__('explorer_node')

        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.timer = self.create_timer(0.1, self.control_loop)

        self.state = RobotState.FORWARD
        self.state_time = 0.0
        self.control_dt = 0.1

        self.scan_ready = False

        self.front_dist = float('inf')
        self.front_left_dist = float('inf')
        self.front_right_dist = float('inf')
        self.left_dist = float('inf')
        self.right_dist = float('inf')

        self.forward_speed = 0.16
        self.turn_speed = 0.8
        self.slow_turn_forward_speed = 0.05

        self.stop_dist = 0.32
        self.caution_dist = 0.50
        self.side_block_dist = 0.28

        self.min_valid_range = 0.05
        self.max_valid_range = 3.5

        self.turn_duration = 0.8
        self.escape_duration = 1.5

        self.preferred_turn_left = True

        self.get_logger().info('Explorer node started.')

    def scan_callback(self, msg: LaserScan):
        ranges = list(msg.ranges)
        n = len(ranges)

        if n == 0:
            return

        angles = [msg.angle_min + i * msg.angle_increment for i in range(n)]

        def sector_min(angle_deg_min, angle_deg_max):
            vals = []
            for r, a in zip(ranges, angles):
                deg = math.degrees(a)

                if deg < -180.0:
                    deg += 360.0
                elif deg > 180.0:
                    deg -= 360.0

                if angle_deg_min <= deg <= angle_deg_max:
                    if math.isfinite(r) and self.min_valid_range < r < self.max_valid_range:
                        vals.append(r)

            if not vals:
                return float('inf')
            return min(vals)

        self.front_dist = min(
            sector_min(-20, 20),
            sector_min(-20, 0)
        )
        self.front_left_dist = sector_min(20, 60)
        self.front_right_dist = sector_min(-60, -20)
        self.left_dist = sector_min(60, 100)
        self.right_dist = sector_min(-100, -60)

        self.scan_ready = True

    def set_state(self, new_state: RobotState):
        if self.state != new_state:
            self.state = new_state
            self.state_time = 0.0

    def control_loop(self):
        if not self.scan_ready:
            return

        self.state_time += self.control_dt

        front_blocked = self.front_dist < self.stop_dist
        left_tight = self.front_left_dist < self.side_block_dist or self.left_dist < self.side_block_dist
        right_tight = self.front_right_dist < self.side_block_dist or self.right_dist < self.side_block_dist

        best_left_space = min(self.front_left_dist, self.left_dist)
        best_right_space = min(self.front_right_dist, self.right_dist)

        cmd = TwistStamped()

        if self.state == RobotState.FORWARD:
            if front_blocked:
                if best_left_space > best_right_space + 0.05:
                    self.preferred_turn_left = True
                    self.set_state(RobotState.TURN_LEFT)
                elif best_right_space > best_left_space + 0.05:
                    self.preferred_turn_left = False
                    self.set_state(RobotState.TURN_RIGHT)
                else:
                    self.preferred_turn_left = random.choice([True, False])
                    self.set_state(RobotState.ESCAPE)
            else:
                if self.front_dist < self.caution_dist:
                    cmd.twist.linear.x = 0.08
                else:
                    cmd.twist.linear.x = self.forward_speed

                steer = 0.0

                if self.left_dist < self.right_dist - 0.08:
                    steer = -0.25
                elif self.right_dist < self.left_dist - 0.08:
                    steer = 0.25

                cmd.twist.angular.z = steer

        elif self.state == RobotState.TURN_LEFT:
            cmd.twist.linear.x = self.slow_turn_forward_speed
            cmd.twist.angular.z = self.turn_speed

            if (not front_blocked and self.front_left_dist > self.caution_dist) or self.state_time > self.turn_duration:
                self.set_state(RobotState.FORWARD)

        elif self.state == RobotState.TURN_RIGHT:
            cmd.twist.linear.x = self.slow_turn_forward_speed
            cmd.twist.angular.z = -self.turn_speed

            if (not front_blocked and self.front_right_dist > self.caution_dist) or self.state_time > self.turn_duration:
                self.set_state(RobotState.FORWARD)

        elif self.state == RobotState.ESCAPE:
            cmd.twist.linear.x = -0.02

            if self.preferred_turn_left:
                cmd.twist.angular.z = self.turn_speed
            else:
                cmd.twist.angular.z = -self.turn_speed

            if self.state_time > self.escape_duration:
                self.set_state(RobotState.FORWARD)

        if self.state == RobotState.FORWARD and front_blocked and left_tight and right_tight:
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