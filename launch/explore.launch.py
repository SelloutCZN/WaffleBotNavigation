"""
ELE434 Team 07 — Real-world exploration launch file (Nav2 edition).

Brings up four cooperating systems:

  1. SLAM via the course-provided tuos_tb3_tools launch.
  2. The full Nav2 navigation stack, with /cmd_vel remapped to /cmd_vel_nav
     so it does not clash with the Waffle's TwistStamped /cmd_vel.
  3. cmd_vel_relay.py — converts Nav2's Twist output on /cmd_vel_nav back
     into TwistStamped on /cmd_vel.
  4. frontier_nav.py — sends NavigateToPose goals to Nav2 based on zone-
     aware frontier scoring.

After 92 s, map_saver_cli writes the SLAM map to <pkg>/maps/explore_map.{png,yaml}.

Launch arguments
----------------
use_sim_time : 'true' or 'false' (default: 'false')
    Set to 'true' when running in Gazebo simulation, 'false' on the real
    Waffle. The default matches the assessment configuration.

Examples
--------
Real robot (default):
    ros2 launch ele434_team07_2026 explore.launch.py

Simulation:
    ros2 launch ele434_team07_2026 explore.launch.py use_sim_time:=true
"""

from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    TimerAction,
    ExecuteProcess,
    GroupAction,
    DeclareLaunchArgument,
)
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, SetRemap
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_name = 'ele434_team07_2026'
    pkg_share = get_package_share_directory(package_name)
    nav2_params = os.path.join(pkg_share, 'config', 'nav2_params.yaml')

    package_source_dir = '/home/student/ros2_ws/src/ele434_team07_2026'
    map_output_base = os.path.join(package_source_dir, 'maps', 'explore_map')

    # ------------------------------------------------------------------
    # Launch arguments — let the caller override use_sim_time at the
    # command line, without editing this file or the YAML.
    # ------------------------------------------------------------------
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description=(
            "Use the simulation clock published on /clock. Set to 'true' "
            "for Gazebo, 'false' for the real robot."
        )
    )
    use_sim_time = LaunchConfiguration('use_sim_time')

    # ------------------------------------------------------------------
    # SLAM (course-provided, configured for the real Waffles)
    # ------------------------------------------------------------------
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('tuos_tb3_tools'),
                'launch',
                'slam.launch.py'
            )
        ),
        launch_arguments={'environment': 'real'}.items()
    )

    # ------------------------------------------------------------------
    # Nav2 — wrapped in a GroupAction with a SetRemap so the controller
    # publishes to /cmd_vel_nav (Twist), avoiding a type clash with the
    # Waffle's TwistStamped /cmd_vel.
    # ------------------------------------------------------------------
    nav2_group = GroupAction([
        SetRemap(src='/cmd_vel', dst='/cmd_vel_nav'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('nav2_bringup'),
                    'launch',
                    'navigation_launch.py'
                )
            ),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'params_file': nav2_params,
                'autostart': 'true',
            }.items()
        ),
    ])

    # ------------------------------------------------------------------
    # /cmd_vel_nav (Twist) -> /cmd_vel (TwistStamped) relay
    # ------------------------------------------------------------------
    cmd_vel_relay = Node(
        package=package_name,
        executable='cmd_vel_relay.py',
        name='cmd_vel_relay',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # ------------------------------------------------------------------
    # Frontier-driven exploration node
    # ------------------------------------------------------------------
    frontier_node = Node(
        package=package_name,
        executable='frontier_nav.py',
        name='frontier_explorer',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,

            # Runtime budget
            'max_runtime': 1000.0,
            'planning_period': 1.0,
            'goal_timeout': 18.0,

            # Frontier extraction
            'cluster_min_size': 1,
            'cluster_min_unknown_neighbours': 1,
            'frontier_sample_stride': 1,
            'clearance_radius_cells': 2,
            'min_frontier_distance': 0.10,
            'max_frontier_distance': 6.0,

            # Zone-aware scoring (4 x 4 m arena, 16 zones of 1 m square)
            'arena_half_size': 2.0,
            'zone_size': 1.0,
            'zone_visit_margin': 0.12,
            'outer_zone_bonus': 3.0,
            'visited_zone_penalty': 1.5,
            'inner_zone_penalty': 1.0,
            'radial_score_gain': 0.6,
            'heading_score_gain': 0.4,
        }]
    )

    # ------------------------------------------------------------------
    # Map saver — 2 s after the exploration window closes
    # ------------------------------------------------------------------
    save_map = TimerAction(
        period=92.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'run', 'nav2_map_server', 'map_saver_cli',
                    '-f', map_output_base,
                    '--fmt', 'png',
                ],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        use_sim_time_arg,
        slam_launch,
        nav2_group,
        cmd_vel_relay,
        frontier_node,
        save_map,
    ])