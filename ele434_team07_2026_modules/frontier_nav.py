#!/usr/bin/env python3
"""
Zone-visiting explorer for ELE434 Team 07.

Goal-sending architecture:
  Goals are sent from EXACTLY TWO places:
    1. _init_done() — once, on startup.
    2. _goal_result_cb() — once per Nav2 result (SUCCEEDED/ABORTED/CANCELED).

  The tick timer (_explore_tick) NEVER sends goals. It only:
    a. Checks for the hard timeout and cancels if needed.
    b. Detects en-route zone visits and cancels so the callback picks next.
    c. Saves the map and stops the run at t=max_runtime in sim/real time.

Map saving:
  The node calls map_saver_cli itself at t=max_runtime (measured in
  whatever time source the node is using — sim time or wall time).
  This is more accurate than a TimerAction in the launch file, which
  always uses wall-clock time and fires far too early in simulation.

Zone visit radius:
  0.45 m from zone centre. The Waffle's radius is 0.14 m, so this
  requires the robot's centre to be within 0.45 m of the zone centre,
  meaning the far edge of the robot is at least (0.50 - 0.45 - 0.14)
  = -0.09 m from the zone boundary, i.e. the entire robot is inside.
"""
# last updated 18:38 16/05/2026

import math
import subprocess

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

        self.declare_parameter('max_runtime', 90.0)
        self.declare_parameter('goal_timeout', 35.0)
        self.declare_parameter('arena_half_size', 2.0)
        self.declare_parameter('zone_size', 1.0)
        # 0.45 m: robot centre must be 0.45 m from zone centre.
        # Zone half-width = 0.50 m, robot radius = 0.14 m.
        # This guarantees the entire robot body is inside the zone.
        self.declare_parameter('zone_visit_radius', 0.45)
        self.declare_parameter('outer_zone_bonus', 8.0)
        self.declare_parameter('visited_zone_penalty', 50.0)
        self.declare_parameter('heading_score_gain', 0.10)
        self.declare_parameter('distance_score_gain', 1.0)
        self.declare_parameter('failed_goal_penalty', 15.0)
        self.declare_parameter('failed_goal_memory_seconds', 40.0)
        # Filesystem path (without extension) for the saved map.
        # Leave blank to skip saving.
        self.declare_parameter('map_output_path', '')

        gp = lambda n: self.get_parameter(n).value
        self.max_runtime                = float(gp('max_runtime'))
        self.goal_timeout               = float(gp('goal_timeout'))
        self.arena_half_size            = float(gp('arena_half_size'))
        self.zone_size                  = float(gp('zone_size'))
        self.zone_visit_radius          = float(gp('zone_visit_radius'))
        self.outer_zone_bonus           = float(gp('outer_zone_bonus'))
        self.visited_zone_penalty       = float(gp('visited_zone_penalty'))
        self.heading_score_gain         = float(gp('heading_score_gain'))
        self.distance_score_gain        = float(gp('distance_score_gain'))
        self.failed_goal_penalty        = float(gp('failed_goal_penalty'))
        self.failed_goal_memory_seconds = float(gp('failed_goal_memory_seconds'))
        self.map_output_path            = str(gp('map_output_path'))

        self._all_zone_ids = [(i, j) for i in range(-2, 2) for j in range(-2, 2)]

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

        self.map_ready  = False
        self.robot_x    = self.robot_y = self.robot_yaw = 0.0
        self.odom_ready = False

        self.origin_x = self.origin_y = self.origin_yaw = 0.0
        self.zone_origin_captured = False

        self.visited_zones = set()
        self.goal_active            = False
        self.current_goal_handle    = None
        self.current_goal_zone      = None
        self.current_goal_xy        = None
        self.current_goal_send_time = 0.0
        self.failed_goals           = []

        self.start_time   = self.get_clock().now().nanoseconds * 1e-9
        self.initialized  = False
        self.run_finished = False
        self.map_saved    = False

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
    def is_outer(zid):
        i, j = zid
        return not (i in (-1, 0) and j in (-1, 0))

    def zone_centre(self, zid):
        i, j = zid
        zx = (i + 0.5) * self.zone_size
        zy = (j + 0.5) * self.zone_size
        c = math.cos(self.origin_yaw)
        s = math.sin(self.origin_yaw)
        return self.origin_x + zx * c - zy * s, self.origin_y + zx * s + zy * c

    def _outer_remaining(self):
        return [z for z in self._all_zone_ids
                if z not in self.visited_zones and self.is_outer(z)]

    def _score(self, zid):
        wx, wy = self.zone_centre(zid)
        dx, dy = wx - self.robot_x, wy - self.robot_y
        dist = math.hypot(dx, dy)
        heading_err = abs(self._angle_wrap(math.atan2(dy, dx) - self.robot_yaw))
        score = self.distance_score_gain * dist + self.heading_score_gain * heading_err
        if zid in self.visited_zones:
            score += self.visited_zone_penalty
        elif self.is_outer(zid):
            score -= self.outer_zone_bonus
        score += self._failed_cost(wx, wy)
        return score, wx, wy

    def _best_zone(self):
        best, best_score, best_wx, best_wy = None, float('inf'), 0.0, 0.0
        for zid in self._all_zone_ids:
            s, wx, wy = self._score(zid)
            if s < best_score:
                best_score, best, best_wx, best_wy = s, zid, wx, wy
        return (best, best_wx, best_wy, best_score) if best else None

    def _arrival_yaw(self, wx, wy):
        """Face toward the arena centre on arrival so departure arcs inward."""
        cx, cy = self.zone_centre((0, 0))
        return math.atan2(cy - wy, cx - wx)

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
            # Tick at 0.5 s for visit/timeout monitoring only.
            self.explore_timer = self.create_timer(0.5, self._explore_tick)
            self.get_logger().info(
                f'Nav2, map, and odom ready — starting zone exploration.\n'
                f'Outer zones: {self._outer_remaining()}'
            )
            self._pick_and_send(0.0)
            return

        missing = [n for n, ok in [('Nav2', nav_ready), ('map', map_ready), ('odom', odom_ready)] if not ok]
        self.get_logger().info(f'Still waiting on: {", ".join(missing)}.')

    def _explore_tick(self):
        """
        Tick handler — monitoring only. NEVER calls _pick_and_send directly.
        Cancels the current goal when necessary; cancellation triggers
        _goal_result_cb which then calls _pick_and_send.
        """
        elapsed = self.elapsed()

        # Save map at exactly t=max_runtime, before ending the run.
        if elapsed >= self.max_runtime and not self.map_saved:
            self._save_map(elapsed)

        if elapsed >= self.max_runtime:
            self._end_run()
            return

        if not self.zone_origin_captured:
            return

        self.update_visited_zones()
        self._prune_failed_goals(elapsed)

        if not self.goal_active:
            # Guard: no active goal outside startup — shouldn't happen normally.
            self.get_logger().warn(
                f'[t={elapsed:.1f}s] No active goal in tick — re-sending.')
            self._pick_and_send(elapsed)
            return

        # Cancel if target zone visited en-route.
        if (self.current_goal_zone is not None
                and self.current_goal_zone in self.visited_zones):
            self.get_logger().info(
                f'[t={elapsed:.1f}s] Zone {self.current_goal_zone} '
                f'visited en-route — cancelling.')
            self._cancel_current_goal()
            return

        # Hard timeout.
        if (elapsed - self.current_goal_send_time) > self.goal_timeout:
            self.get_logger().warn(f'[t={elapsed:.1f}s] Goal timeout — cancelling.')
            self._remember_failed_goal(elapsed)
            self._cancel_current_goal()
            return

    def _pick_and_send(self, elapsed):
        """Pick the best unvisited zone and send a Nav2 goal."""
        if self.goal_active:
            self.get_logger().warn(
                f'[t={elapsed:.1f}s] _pick_and_send called with goal active — ignoring.')
            return

        target = self._best_zone()
        if target is None:
            self.get_logger().info(f'[t={elapsed:.1f}s] No zone goal available.')
            return

        zid, wx, wy, score = target
        if zid in self.visited_zones:
            self.get_logger().info(
                f'[t={elapsed:.1f}s] Best zone {zid} already visited — all done?')
            return

        self._send_nav_goal(zid, wx, wy, score, elapsed)

    def _save_map(self, elapsed):
        """Call map_saver_cli to write the SLAM map. Runs once at t=max_runtime."""
        self.map_saved = True
        if not self.map_output_path:
            self.get_logger().warn(
                f'[t={elapsed:.1f}s] map_output_path not set — skipping map save.')
            return
        self.get_logger().info(
            f'[t={elapsed:.1f}s] Saving map to {self.map_output_path}...')
        try:
            subprocess.Popen([
                'ros2', 'run', 'nav2_map_server', 'map_saver_cli',
                '-f', self.map_output_path,
                '--fmt', 'png',
            ])
        except Exception as exc:
            self.get_logger().error(f'map_saver_cli failed to launch: {exc}')

    def _end_run(self):
        if self.run_finished:
            return
        self.run_finished = True
        self._cancel_current_goal()
        self._publish_stop()
        outer = sum(1 for z in self.visited_zones if self.is_outer(z))
        self.get_logger().info(
            f'[t=90.0s] DONE. Outer: {outer}/12. '
            f'Visited: {sorted(self.visited_zones)}'
        )
        if self.explore_timer is not None:
            self.explore_timer.cancel()

    def _publish_stop(self):
        zero = TwistStamped()
        zero.header.stamp    = self.get_clock().now().to_msg()
        zero.header.frame_id = 'base_link'
        self.stop_pub.publish(zero)

    # ------------------------------------------------------------------
    # Zone visit detection
    # ------------------------------------------------------------------

    def update_visited_zones(self):
        if not self.zone_origin_captured:
            return
        for zid in list(self._all_zone_ids):
            if zid in self.visited_zones:
                continue
            wx, wy = self.zone_centre(zid)
            dist = math.hypot(self.robot_x - wx, self.robot_y - wy)
            if dist <= self.zone_visit_radius:
                self.visited_zones.add(zid)
                kind  = 'OUTER' if self.is_outer(zid) else 'inner'
                outer = sum(1 for z in self.visited_zones if self.is_outer(z))
                self.get_logger().info(
                    f'[t={self.elapsed():.1f}s] ✓ {kind} zone {zid} '
                    f'(dist={dist:.2f}m). Outer: {outer}/12. '
                    f'Remaining: {self._outer_remaining()}'
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

    def _failed_cost(self, wx, wy):
        for fx, fy, _ in self.failed_goals:
            if math.hypot(wx - fx, wy - fy) < 0.5:
                return self.failed_goal_penalty
        return 0.0

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny, cosy)
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
        arrival_yaw = self._arrival_yaw(wx, wy)

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp    = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(wx)
        goal_msg.pose.pose.position.y = float(wy)
        goal_msg.pose.pose.orientation.z = math.sin(arrival_yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(arrival_yaw / 2.0)

        kind  = 'OUTER' if self.is_outer(zone_id) else 'inner'
        outer = sum(1 for z in self.visited_zones if self.is_outer(z))
        self.get_logger().info(
            f'[t={elapsed:.1f}s] → {kind} zone {zone_id} '
            f'({wx:.2f}, {wy:.2f}) score={score:.2f} '
            f'arrival={math.degrees(arrival_yaw):.0f}° '
            f'[outer done: {outer}/12]'
        )

        self.goal_active            = True
        self.current_goal_zone      = zone_id
        self.current_goal_xy        = (wx, wy)
        self.current_goal_send_time = elapsed

        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        try:
            gh = future.result()
        except Exception as exc:
            self.get_logger().error(f'send_goal_async failed: {exc}')
            self.goal_active = False
            self._pick_and_send(self.elapsed())
            return
        if not gh.accepted:
            self.get_logger().warn('Nav2 rejected goal.')
            self._remember_failed_goal(self.elapsed())
            self.goal_active         = False
            self.current_goal_handle = None
            self._pick_and_send(self.elapsed())
            return
        self.current_goal_handle = gh
        gh.get_result_async().add_done_callback(self._goal_result_cb)

    def _goal_result_cb(self, future):
        elapsed = self.elapsed()
        try:
            wrapped = future.result()
        except Exception as exc:
            self.get_logger().error(f'get_result_async failed: {exc}')
            self.goal_active         = False
            self.current_goal_handle = None
            self._pick_and_send(elapsed)
            return

        status = wrapped.status

        # Mark newly visited zones before scoring the next goal.
        self.update_visited_zones()

        outer = sum(1 for z in self.visited_zones if self.is_outer(z))

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(
                f'[t={elapsed:.1f}s] Goal SUCCEEDED. Outer: {outer}/12. '
                f'Remaining: {self._outer_remaining()}')
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warn(f'[t={elapsed:.1f}s] Goal ABORTED. Outer: {outer}/12.')
            self._remember_failed_goal(elapsed)
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info(f'[t={elapsed:.1f}s] Goal CANCELED.')
        else:
            self.get_logger().info(f'[t={elapsed:.1f}s] Goal status={status}.')

        self.goal_active          = False
        self.current_goal_handle  = None
        self.current_goal_zone    = None
        self.current_goal_xy      = None

        if not self.run_finished:
            self._pick_and_send(elapsed)

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