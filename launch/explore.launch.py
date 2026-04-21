from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ele434_team07_2026',
            executable='explorer_node.py',
            name='explorer_node',
            output='screen'
        )
    ])