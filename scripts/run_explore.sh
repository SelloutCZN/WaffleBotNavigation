#!/usr/bin/env bash
# =====================================================================
# run_explore.sh — Development helper for ELE434 Team 07.
#
# Cleans up any zombie Nav2 / SLAM / RViz / our-own processes left over
# from previous launches on the laptop, then starts a fresh run of the
# exploration stack.
#
# Designed for the lab laptop: only kills laptop-side processes, leaves
# the Waffle's onboard processes (running on the robot itself, reached
# via Zenoh) completely alone.
#
# Usage:
#     ./scripts/run_explore.sh
#
# NOTE: This is a DEVELOPMENT convenience only. The launch file itself
# does not invoke any of these kills, so submission for marking remains
# safe and self-contained.
# =====================================================================

set -u

# ---------------------------------------------------------------------
# Cleanup phase
# ---------------------------------------------------------------------

echo "[run_explore] Killing laptop-side ROS processes from previous runs..."

# Target the specific things we and Nav2 spawn, NOT a blanket "ros2"
# match (which would also kill any "ros2 node list" running elsewhere).
PATTERNS=(
    "lifecycle_manager"
    "bt_navigator"
    "planner_server"
    "controller_server"
    "behavior_server"
    "smoother_server"
    "waypoint_follower"
    "velocity_smoother"
    "collision_monitor"
    "route_server"
    "opennav_docking"
    "docking_server"
    "cartographer_node"
    "cartographer_occupancy_grid_node"
    "frontier_nav"
    "cmd_vel_relay"
    "rviz2"
    "map_saver_cli"
    "nav2_map_server"
)

for pat in "${PATTERNS[@]}"; do
    pkill -9 -f "$pat" 2>/dev/null && echo "  - killed: $pat" || true
done

# Give DDS/Zenoh a moment to garbage-collect the killed processes'
# topic ownership before we start fresh ones.
echo "[run_explore] Waiting 3 s for the middleware to settle..."
sleep 3

# ---------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------

echo "[run_explore] Remaining ROS nodes on the graph:"
ros2 node list 2>/dev/null | sort | sed 's/^/  /'

# ---------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------

echo "[run_explore] Starting explore.launch.py..."
echo "[run_explore] Allow ~30–45 s for SLAM, Nav2 lifecycle activation,"
echo "[run_explore] and Zenoh discovery before the robot starts moving."
echo ""

exec ros2 launch ele434_team07_2026 explore.launch.py