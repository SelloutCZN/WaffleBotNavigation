from launch import LaunchDescription
from launch.actions import TimerAction, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_name = 'ele434_team07_2026'

    # Change this to your actual package source path
    package_source_dir = '/home/student/ros2_ws/src/ele434_team07_2026'
    map_output_base = os.path.join(package_source_dir, 'maps', 'explore_map')

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('slam_toolbox'),
                'launch',
                'online_async_launch.py'
            )
        )
    )

    explorer_node = Node(
        package=package_name,
        executable='frontier_nav.py',
        name='explorer_node',
        output='screen',
        parameters=[
            {
                'forward_speed': 0.26,
                'turn_speed': 1.0,
                'slow_turn_forward_speed': 0.0,
                'caution_speed': 0.16,
                'stop_dist': 0.38,
                'caution_dist': 0.55,
                'side_block_dist': 0.30,
                'turn_duration': 0.7,
                'escape_duration': 1.5,
                'rear_block_dist': 0.25,
                'front_half_angle_deg': 45.0,
                'rear_half_angle_deg': 45.0,
                'side_half_angle_deg': 45.0,
                'max_runtime': 90.0,
                'goal_reach_dist': 0.35,
                'frontier_min_unknown_neighbors': 1,
                'frontier_sample_stride': 2,
                'goal_heading_gain': 1.2,
                'max_goal_turn': 0.7,
                'goal_refresh_period': 2.0,
                'min_frontier_distance': 0.6,
                'max_frontier_distance': 6.0,
            }
        ]
    )

    save_map = TimerAction(
        period=92.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'run', 'nav2_map_server', 'map_saver_cli',
                    '-f', map_output_base,
                    '--fmt', 'png'
                ],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        slam_launch,
        explorer_node,
        save_map,
    ])