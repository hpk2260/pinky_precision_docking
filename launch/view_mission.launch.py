import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('pinky_precision_docking')
    default_rviz = os.path.join(pkg_share, 'config', 'arena_mission_view.rviz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'rviz_config',
            default_value=default_rviz,
            description='Full path to the RViz config file'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', LaunchConfiguration('rviz_config')]
        )
    ])
