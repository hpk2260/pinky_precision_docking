import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource, PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    gz_sim_dir = get_package_share_directory('pinky_gz_sim')
    docking_dir = get_package_share_directory('pinky_precision_docking')

    world_name_arg = DeclareLaunchArgument(
        'world_name',
        default_value='docking_test.world',
        description='Gazebo world file with docking wall and ArUco 5x5 markers (ID 0, 2, 4)',
    )
    cam_tilt_deg_arg = DeclareLaunchArgument(
        'cam_tilt_deg',
        default_value='0',
        description='Camera pitch tilt angle in degrees',
    )
    run_docking_arg = DeclareLaunchArgument(
        'run_docking',
        default_value='False',
        description='Set True to start precision_docking_node immediately alongside Gazebo',
    )
    target_marker_id_arg = DeclareLaunchArgument(
        'target_marker_id',
        default_value='4',
        description='Target ArUco marker ID in docking_test.world (0, 2, or 4; -1 for auto)',
    )
    target_distance_arg = DeclareLaunchArgument(
        'target_distance',
        default_value='0.10',
        description='Target stopping distance in meters',
    )

    spawn_x_arg = DeclareLaunchArgument(
        'spawn_x',
        default_value='0.0',
        description='Robot initial spawn x coordinate in world',
    )
    spawn_y_arg = DeclareLaunchArgument(
        'spawn_y',
        default_value='0.0',
        description='Robot initial spawn y coordinate in world',
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

    docking_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(docking_dir, 'launch', 'precision_docking.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('run_docking')),
        launch_arguments={
            'use_sim_time': 'True',
            'use_compressed': 'False',
            'target_marker_id': LaunchConfiguration('target_marker_id'),
            'target_distance': LaunchConfiguration('target_distance'),
        }.items(),
    )

    return LaunchDescription([
        world_name_arg,
        cam_tilt_deg_arg,
        run_docking_arg,
        target_marker_id_arg,
        target_distance_arg,
        spawn_x_arg,
        spawn_y_arg,
        spawn_yaw_arg,
        headless_arg,
        sim_launch,
        docking_launch,
    ])
