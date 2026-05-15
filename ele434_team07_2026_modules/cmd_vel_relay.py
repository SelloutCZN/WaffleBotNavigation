#!/usr/bin/env python3
"""
cmd_vel_relay.py — convert Nav2's Twist output into the TwistStamped
that the Waffle's base controller expects.

Nav2's controller publishes geometry_msgs/Twist on /cmd_vel internally,
which is remapped to /cmd_vel_nav by the launch file so it does not
clash with the Waffle's TwistStamped /cmd_vel topic. This node listens
on /cmd_vel_nav and republishes as TwistStamped on /cmd_vel.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped


class CmdVelRelay(Node):
    def __init__(self):
        super().__init__('cmd_vel_relay')
        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_subscription(Twist, '/cmd_vel_nav', self._on_twist, 10)
        self.get_logger().info(
            'cmd_vel_relay: /cmd_vel_nav (Twist) -> /cmd_vel (TwistStamped).'
        )

    def _on_twist(self, msg: Twist):
        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'base_link'
        out.twist = msg
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # On Ctrl+C rclpy may already have shut down, so guard it.
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()