import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    gz_sim_dir = get_package_share_directory('pinky_gz_sim')
    docking_dir = get_package_share_directory('pinky_precision_docking')

    world_name_arg = DeclareLaunchArgument(
        'world_name',
        default_value='arena_docking.world',
        description='Gazebo world file with arena walls and ArUco markers (ID 7, 4, 5)',
    )
    cam_tilt_deg_arg = DeclareLaunchArgument(
        'cam_tilt_deg',
        default_value='0',
        description='Camera pitch tilt angle in degrees',
    )
    spawn_x_arg = DeclareLaunchArgument(
        'spawn_x',
        default_value='0.0',
        description='Robot initial spawn x coordinate in world (Home Parking)',
    )
    spawn_y_arg = DeclareLaunchArgument(
        'spawn_y',
        default_value='0.0',
        description='Robot initial spawn y coordinate in world (Home Parking)',
    )
    spawn_yaw_arg = DeclareLaunchArgument(
        'spawn_yaw',
        default_value='0.0',
        description='Robot initial spawn yaw angle in radians',
    )
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='False',
        description='Run Gazebo in headless mode without GUI',
    )

    sim_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(gz_sim_dir, 'launch', 'launch_sim.launch.xml')
        ),
        launch_arguments={
            'world_name': LaunchConfiguration('world_name'),
            'cam_tilt_deg': LaunchConfiguration('cam_tilt_deg'),
            'spawn_x': LaunchConfiguration('spawn_x'),
            'spawn_y': LaunchConfiguration('spawn_y'),
            'spawn_yaw': LaunchConfiguration('spawn_yaw'),
            'headless': LaunchConfiguration('headless'),
        }.items(),
    )

    return LaunchDescription([
        world_name_arg,
        cam_tilt_deg_arg,
        spawn_x_arg,
        spawn_y_arg,
        spawn_yaw_arg,
        headless_arg,
        sim_launch,
    ])
