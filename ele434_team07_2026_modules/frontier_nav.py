#!/usr/bin/env python3
"""
Zone-visiting explorer for ELE434 Team 07.

Strategy: drive to the centre of each unvisited outer zone, then inner zones.
A zone is "visited" when the robot centre is within `zone_visit_radius` of
the zone centre (simple Euclidean check, not a margin-based boundary check).

Key fixes vs previous version:
  - Visit detection uses distance-to-zone-centre, not a margin inside the
    boundary. This is simpler and doesn't break at boundary edges.
  - xy_goal_tolerance tightened to 0.20 m so the robot actually reaches
    the zone centre before Nav2 declares success.
  - Goal switch margin reduced so the robot switches promptly after success.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus


class ZoneExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter('max_runtime', 90.0)
        self.declare_parameter('planning_period', 1.0)
        self.declare_parameter('goal_timeout', 30.0)

        self.declare_parameter('arena_half_size', 2.0)
        self.declare_parameter('zone_size', 1.0)
        # A zone is "visited" when robot centre is within this distance of
        # the zone centre. 0.40 m means clearly inside a 1 m² zone.
        self.declare_parameter('zone_visit_radius', 0.40)

        self.declare_parameter('outer_zone_bonus', 8.0)
        self.declare_parameter('visited_zone_penalty', 30.0)
        self.declare_parameter('heading_score_gain', 0.3)
        self.declare_parameter('distance_score_gain', 1.0)

        self.declare_parameter('goal_switch_margin', 1.5)
        self.declare_parameter('failed_goal_penalty', 10.0)
        self.declare_parameter('failed_goal_memory_seconds', 30.0)

        gp = lambda n: self.get_parameter(n).value
        self.max_runtime                = float(gp('max_runtime'))
        self.planning_period            = float(gp('planning_period'))
        self.goal_timeout               = float(gp('goal_timeout'))
        self.arena_half_size            = float(gp('arena_half_size'))
        self.zone_size                  = float(gp('zone_size'))
        self.zone_visit_radius          = float(gp('zone_visit_radius'))
        self.outer_zone_bonus           = float(gp('outer_zone_bonus'))
        self.visited_zone_penalty       = float(gp('visited_zone_penalty'))
        self.heading_score_gain         = float(gp('heading_score_gain'))
        self.distance_score_gain        = float(gp('distance_score_gain'))
        self.goal_switch_margin         = float(gp('goal_switch_margin'))
        self.failed_goal_penalty        = float(gp('failed_goal_penalty'))
        self.failed_goal_memory_seconds = float(gp('failed_goal_memory_seconds'))

        # All 16 zone ids: i,j in {-2,-1,0,1}
        self._all_zone_ids = [(i, j) for i in range(-2, 2) for j in range(-2, 2)]

        # ------------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------------
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(OccupancyGrid, '/map', self._map_callback, map_qos)
        self.create_subscription(Odometry, '/odom', self._odom_callback, 10)
        self.stop_pub   = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.map_ready  = False
        self.robot_x    = self.robot_y = self.robot_yaw = 0.0
        self.odom_ready = False

        self.origin_x = self.origin_y = self.origin_yaw = 0.0
        self.zone_origin_captured = False

        self.visited_zones = set()   # set of (i, j) zone ids

        self.goal_active            = False
        self.current_goal_handle    = None
        self.current_goal_zone      = None
        self.current_goal_xy        = None
        self.current_goal_score     = float('inf')
        self.current_goal_send_time = 0.0
        self.failed_goals           = []  # list of (wx, wy, expire_time)

        self.start_time   = self.get_clock().now().nanoseconds * 1e-9
        self.initialized  = False
        self.run_finished = False

        self.init_timer    = self.create_timer(0.5, self._init_tick)
        self.explore_timer = None

        self.get_logger().info('Zone explorer launched, waiting for Nav2 and SLAM...')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def elapsed(self):
        return (self.get_clock().now().nanoseconds * 1e-9) - self.start_time

    def _angle_wrap(self, a):
        while a >  math.pi: a -= 2.0 * math.pi
        while a < -math.pi: a += 2.0 * math.pi
        return a

    @staticmethod
    def is_outer_zone(zone_id):
        i, j = zone_id
        return not (i in (-1, 0) and j in (-1, 0))

    def zone_centre_map(self, zone_id):
        """Return (wx, wy) of zone centre in map/odom frame."""
        i, j = zone_id
        zx = (i + 0.5) * self.zone_size
        zy = (j + 0.5) * self.zone_size
        c  = math.cos(self.origin_yaw)
        s  = math.sin(self.origin_yaw)
        return self.origin_x + zx * c - zy * s, self.origin_y + zx * s + zy * c

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _init_tick(self):
        if self.initialized:
            return
        nav_ready  = self.nav_client.wait_for_server(timeout_sec=0.05)
        map_ready  = self.map_ready
        odom_ready = self.zone_origin_captured

        if nav_ready and map_ready and odom_ready:
            self.initialized = True
            self.init_timer.cancel()
            self.start_time = self.get_clock().now().nanoseconds * 1e-9
            self.explore_timer = self.create_timer(self.planning_period, self._explore_tick)
            self.get_logger().info('Nav2, map, and odom ready — zone exploration timer running.')
            return

        missing = [n for n, ok in [('Nav2', nav_ready), ('map', map_ready), ('odom', odom_ready)] if not ok]
        self.get_logger().info(f'Still waiting on: {", ".join(missing)}.')

    def _explore_tick(self):
        elapsed = self.elapsed()

        if elapsed >= self.max_runtime:
            self._end_run()
            return

        if not self.zone_origin_captured:
            return

        self.update_visited_zones()
        self._prune_failed_goals(elapsed)

        # If current goal's zone is now visited, cancel immediately.
        if (self.goal_active
                and self.current_goal_zone is not None
                and self.current_goal_zone in self.visited_zones):
            self.get_logger().info(
                f'[t={elapsed:.1f}s] Zone {self.current_goal_zone} visited — picking next.')
            self._cancel_current_goal()

        # Hard timeout.
        if self.goal_active and (elapsed - self.current_goal_send_time) > self.goal_timeout:
            self.get_logger().warn(f'[t={elapsed:.1f}s] Goal timeout — re-planning.')
            self._remember_failed_goal(elapsed)
            self._cancel_current_goal()

        if self.goal_active:
            target = self._choose_zone_goal(elapsed)
            if target is not None:
                zid, wx, wy, score = target
                if score + self.goal_switch_margin < self.current_goal_score:
                    self.get_logger().info(
                        f'[t={elapsed:.1f}s] Switching to better zone {zid} '
                        f'(score {score:.2f} vs {self.current_goal_score:.2f}).')
                    self._cancel_current_goal()
                    self._send_nav_goal(zid, wx, wy, score, elapsed)
            return

        target = self._choose_zone_goal(elapsed)
        if target is None:
            outer_left = [z for z in self._all_zone_ids
                          if z not in self.visited_zones and self.is_outer_zone(z)]
            self.get_logger().info(
                f'[t={elapsed:.1f}s] No zone goal. Outer remaining: {len(outer_left)}/12.')
            return

        zid, wx, wy, score = target
        self._send_nav_goal(zid, wx, wy, score, elapsed)

    def _choose_zone_goal(self, elapsed):
        best = None
        best_score = float('inf')

        for zid in self._all_zone_ids:
            wx, wy = self.zone_centre_map(zid)
            dx = wx - self.robot_x
            dy = wy - self.robot_y
            dist = math.hypot(dx, dy)

            heading_err = abs(self._angle_wrap(math.atan2(dy, dx) - self.robot_yaw))

            score = (self.distance_score_gain * dist
                     + self.heading_score_gain * heading_err)

            if zid in self.visited_zones:
                score += self.visited_zone_penalty
            elif self.is_outer_zone(zid):
                score -= self.outer_zone_bonus

            score += self._failed_goal_proximity_cost(wx, wy)

            if score < best_score:
                best_score = score
                best = (zid, wx, wy, score)

        return best

    def _end_run(self):
        if self.run_finished:
            return
        self.run_finished = True
        self._cancel_current_goal()
        self._publish_stop()
        outer = sum(1 for z in self.visited_zones if self.is_outer_zone(z))
        self.get_logger().info(
            f'[t=90.0s] Runtime elapsed. '
            f'Outer zones visited: {outer}/12. '
            f'All visited: {sorted(self.visited_zones)}'
        )
        if self.explore_timer is not None:
            self.explore_timer.cancel()

    def _publish_stop(self):
        zero = TwistStamped()
        zero.header.stamp    = self.get_clock().now().to_msg()
        zero.header.frame_id = 'base_link'
        self.stop_pub.publish(zero)

    # ------------------------------------------------------------------
    # Zone visit detection — distance to zone centre
    # ------------------------------------------------------------------

    def update_visited_zones(self):
        if not self.zone_origin_captured:
            return
        for zid in self._all_zone_ids:
            if zid in self.visited_zones:
                continue
            wx, wy = self.zone_centre_map(zid)
            dist = math.hypot(self.robot_x - wx, self.robot_y - wy)
            if dist <= self.zone_visit_radius:
                self.visited_zones.add(zid)
                kind  = 'OUTER' if self.is_outer_zone(zid) else 'inner'
                outer = sum(1 for z in self.visited_zones if self.is_outer_zone(z))
                outer_left = [z for z in self._all_zone_ids
                              if z not in self.visited_zones and self.is_outer_zone(z)]
                self.get_logger().info(
                    f'[t={self.elapsed():.1f}s] ✓ Visited {kind} zone {zid} '
                    f'(dist={dist:.2f}m). Outer: {outer}/12. '
                    f'Remaining: {outer_left}'
                )

    # ------------------------------------------------------------------
    # Failed-goal blacklist
    # ------------------------------------------------------------------

    def _remember_failed_goal(self, elapsed):
        if self.current_goal_xy is None:
            return
        self.failed_goals.append((self.current_goal_xy[0],
                                  self.current_goal_xy[1],
                                  elapsed + self.failed_goal_memory_seconds))

    def _prune_failed_goals(self, elapsed):
        self.failed_goals = [fg for fg in self.failed_goals if fg[2] > elapsed]

    def _failed_goal_proximity_cost(self, wx, wy):
        for fx, fy, _ in self.failed_goals:
            if math.hypot(wx - fx, wy - fy) < 0.6:
                return self.failed_goal_penalty
        return 0.0

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

        if not self.zone_origin_captured:
            self.origin_x   = self.robot_x
            self.origin_y   = self.robot_y
            self.origin_yaw = self.robot_yaw
            self.zone_origin_captured = True
            self.get_logger().info(
                f'Zone frame anchored at ({self.origin_x:.2f}, {self.origin_y:.2f}), '
                f'yaw={math.degrees(self.origin_yaw):.1f} deg.')
        self.odom_ready = True

    def _map_callback(self, msg: OccupancyGrid):
        self.map_ready = True

    # ------------------------------------------------------------------
    # Nav2 action client
    # ------------------------------------------------------------------

    def _send_nav_goal(self, zone_id, wx, wy, score, elapsed):
        yaw = math.atan2(wy - self.robot_y, wx - self.robot_x)

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp    = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(wx)
        goal_msg.pose.pose.position.y = float(wy)
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        kind = 'OUTER' if self.is_outer_zone(zone_id) else 'inner'
        outer = sum(1 for z in self.visited_zones if self.is_outer_zone(z))
        self.get_logger().info(
            f'[t={elapsed:.1f}s] → {kind} zone {zone_id} centre '
            f'({wx:.2f}, {wy:.2f}) score={score:.2f} [outer done: {outer}/12]'
        )

        self.goal_active            = True
        self.current_goal_zone      = zone_id
        self.current_goal_xy        = (wx, wy)
        self.current_goal_score     = score
        self.current_goal_send_time = elapsed

        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f'send_goal_async failed: {exc}')
            self.goal_active = False
            return

        if not goal_handle.accepted:
            self.get_logger().warn('Nav2 rejected the goal.')
            self._remember_failed_goal(self.elapsed())
            self.goal_active         = False
            self.current_goal_handle = None
            return

        self.current_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self._goal_result_callback)

    def _goal_result_callback(self, future):
        elapsed = self.elapsed()
        try:
            wrapped = future.result()
        except Exception as exc:
            self.get_logger().error(f'get_result_async failed: {exc}')
            self.goal_active         = False
            self.current_goal_handle = None
            return

        status = wrapped.status
        outer  = sum(1 for z in self.visited_zones if self.is_outer_zone(z))

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(
                f'[t={elapsed:.1f}s] Goal SUCCEEDED. Outer visited: {outer}/12.')
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warn(f'[t={elapsed:.1f}s] Goal ABORTED. Outer visited: {outer}/12.')
            self._remember_failed_goal(elapsed)
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info(f'[t={elapsed:.1f}s] Goal CANCELED.')
        else:
            self.get_logger().info(f'[t={elapsed:.1f}s] Goal status={status}.')

        self.goal_active          = False
        self.current_goal_handle  = None
        self.current_goal_zone    = None
        self.current_goal_xy      = None
        self.current_goal_score   = float('inf')

    def _cancel_current_goal(self):
        if self.current_goal_handle is not None:
            try:
                self.current_goal_handle.cancel_goal_async()
            except Exception:
                pass
        self.goal_active = False


def main(args=None):
    rclpy.init(args=args)
    node = ZoneExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._cancel_current_goal()
            node._publish_stop()
            rclpy.spin_once(node, timeout_sec=0.2)
            node._publish_stop()
            rclpy.spin_once(node, timeout_sec=0.2)
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()