import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('pinky_precision_docking')
    default_params_file = os.path.join(pkg_dir, 'config', 'docking_params_v38.yaml')

    # Declare Launch Arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Use simulation time (/clock)'
    )

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Path to ROS 2 parameters YAML file'
    )

    target_distance_arg = DeclareLaunchArgument(
        'target_distance',
        default_value='0.20',
        description='Target stopping distance from marker (meters)'
    )

    use_compressed_arg = DeclareLaunchArgument(
        'use_compressed',
        default_value='True',
        description='Subscribe to compressed image stream'
    )

    target_marker_id_arg = DeclareLaunchArgument(
        'target_marker_id',
        default_value='-1',
        description='ArUco marker ID to track (-1 for auto-lockon)'
    )

    aruco_dict_arg = DeclareLaunchArgument(
        'aruco_dictionary_id',
        default_value='DICT_5X5_250',
        description='ArUco dictionary (e.g. DICT_5X5_250, DICT_4X4_50)'
    )

    wall_align_arg = DeclareLaunchArgument(
        'enable_wall_alignment',
        default_value='False',
        description='Align robot front parallel to marker wall'
    )

    t_maneuver_arg = DeclareLaunchArgument(
        'enable_t_maneuver',
        default_value='True',
        description='Use orthogonal T-maneuver for collinear alignment'
    )

    t_maneuver_min_dist_arg = DeclareLaunchArgument(
        't_maneuver_min_dist',
        default_value='0.20',
        description='Minimum distance from marker to perform T-maneuver (meters)'
    )

    blind_push_arg = DeclareLaunchArgument(
        'enable_blind_push',
        default_value='True',
        description='Use odometry for <10cm ultra-close docking without marker loss'
    )

    docking_node = Node(
        package='pinky_precision_docking',
        executable='precision_docking_node_v38',
        name='precision_docking_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'target_distance': LaunchConfiguration('target_distance'),
                'use_compressed': LaunchConfiguration('use_compressed'),
                'target_marker_id': LaunchConfiguration('target_marker_id'),
                'aruco_dictionary_id': LaunchConfiguration('aruco_dictionary_id'),
                'enable_t_maneuver': LaunchConfiguration('enable_t_maneuver'),
                't_maneuver_min_dist': LaunchConfiguration('t_maneuver_min_dist'),
                'enable_wall_alignment': LaunchConfiguration('enable_wall_alignment'),
                'enable_blind_push': LaunchConfiguration('enable_blind_push'),
            }
        ]
    )

    return LaunchDescription([
        use_sim_time_arg,
        params_file_arg,
        target_distance_arg,
        use_compressed_arg,
        target_marker_id_arg,
        aruco_dict_arg,
        wall_align_arg,
        t_maneuver_arg,
        t_maneuver_min_dist_arg,
        blind_push_arg,
        docking_node,
    ])
