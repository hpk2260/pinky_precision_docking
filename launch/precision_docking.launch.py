import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('pinky_precision_docking')
    default_params_file = os.path.join(pkg_dir, 'config', 'docking_params.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Path to ROS 2 parameters YAML file',
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='False',
        description='Use simulation clock from /clock',
    )

    start_in_idle_arg = DeclareLaunchArgument(
        'start_in_idle',
        default_value='False',
        description='Start in IDLE state waiting for /docking/start trigger service',
    )

    target_marker_id_arg = DeclareLaunchArgument(
        'target_marker_id',
        default_value='-1',
        description='Target ArUco marker ID (-1 for auto-lockon)',
    )

    targetmarkerid_arg = DeclareLaunchArgument(
        'targetmarkerid',
        default_value='-1',
        description='Alias for target_marker_id',
    )

    marker_id_arg = DeclareLaunchArgument(
        'marker_id',
        default_value='-1',
        description='Alias for target_marker_id',
    )

    target_id_arg = DeclareLaunchArgument(
        'target_id',
        default_value='-1',
        description='Alias for target_marker_id',
    )

    marker_size_arg = DeclareLaunchArgument(
        'marker_size',
        default_value='0.06',
        description='Physical ArUco marker size in meters (default 0.06m)',
    )

    aruco_dict_arg = DeclareLaunchArgument(
        'aruco_dictionary_id',
        default_value='DICT_5X5_250',
        description='ArUco dictionary (e.g. DICT_5X5_250)',
    )

    use_compressed_arg = DeclareLaunchArgument(
        'use_compressed',
        default_value='True',
        description='Subscribe to /camera/image_raw/compressed',
    )

    lidar_roi_angle_deg_arg = DeclareLaunchArgument(
        'lidar_roi_angle_deg',
        default_value='65.0',
        description='Front LiDAR ROI half-angle in degrees (+/- deg, wide FOV for close docking)',
    )

    alpha_arg = DeclareLaunchArgument(
        'alpha',
        default_value='0.6',
        description='Complementary filter weight for camera distance vs LiDAR distance',
    )

    target_distance_arg = DeclareLaunchArgument(
        'target_distance',
        default_value='0.10',
        description='Target stopping distance from marker/station (meters)',
    )

    distance_tolerance_arg = DeclareLaunchArgument(
        'distance_tolerance',
        default_value='0.01',
        description='Distance error tolerance for docking completion (meters)',
    )

    center_tolerance_arg = DeclareLaunchArgument(
        'center_tolerance',
        default_value='0.005',
        description='Lateral center error tolerance for docking completion (meters)',
    )

    angle_tolerance_deg_arg = DeclareLaunchArgument(
        'angle_tolerance_deg',
        default_value='2.0',
        description='Angular error tolerance for docking completion (degrees)',
    )

    target_wall_angle_deg_arg = DeclareLaunchArgument(
        'target_wall_angle_deg',
        default_value='0.0',
        description='Target relative angle to wall normal (0.0: front bumper parallel, 90.0: side parallel)',
    )

    wall_align_tol_deg_arg = DeclareLaunchArgument(
        'wall_align_tol_deg',
        default_value='0.5',
        description='Ultra-fine angular tolerance for wall parallel alignment (degrees)',
    )

    allow_retreat_arg = DeclareLaunchArgument(
        'allow_retreat',
        default_value='True',
        description='Allow retreating/backing off during recovery (capped by initial start position)',
    )

    osd_bg_alpha_arg = DeclareLaunchArgument(
        'osd_bg_alpha',
        default_value='0.35',
        description='OSD banner background opacity (0.0: fully transparent, 1.0: fully opaque)',
    )

    csv_log_path_arg = DeclareLaunchArgument(
        'csv_log_path',
        default_value='/home/user/pinky_pro/docking_experiment_log.csv',
        description='CSV file path to record docking completion metrics',
    )

    docking_node = Node(
        package='pinky_precision_docking',
        executable='precision_docking_node',
        name='precision_docking_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'start_in_idle': LaunchConfiguration('start_in_idle'),
                'target_marker_id': LaunchConfiguration('target_marker_id'),
                'targetmarkerid': LaunchConfiguration('targetmarkerid'),
                'marker_id': LaunchConfiguration('marker_id'),
                'target_id': LaunchConfiguration('target_id'),
                'marker_size': LaunchConfiguration('marker_size'),
                'aruco_dictionary_id': LaunchConfiguration('aruco_dictionary_id'),
                'use_compressed': LaunchConfiguration('use_compressed'),
                'lidar_roi_angle_deg': LaunchConfiguration('lidar_roi_angle_deg'),
                'alpha': LaunchConfiguration('alpha'),
                'target_distance': LaunchConfiguration('target_distance'),
                'distance_tolerance': LaunchConfiguration('distance_tolerance'),
                'center_tolerance': LaunchConfiguration('center_tolerance'),
                'angle_tolerance_deg': LaunchConfiguration('angle_tolerance_deg'),
                'target_wall_angle_deg': LaunchConfiguration('target_wall_angle_deg'),
                'wall_align_tol_deg': LaunchConfiguration('wall_align_tol_deg'),
                'allow_retreat': LaunchConfiguration('allow_retreat'),
                'osd_bg_alpha': LaunchConfiguration('osd_bg_alpha'),
                'csv_log_path': LaunchConfiguration('csv_log_path'),
            },
        ],
    )

    return LaunchDescription([
        params_file_arg,
        use_sim_time_arg,
        start_in_idle_arg,
        target_marker_id_arg,
        targetmarkerid_arg,
        marker_id_arg,
        target_id_arg,
        marker_size_arg,
        aruco_dict_arg,
        use_compressed_arg,
        lidar_roi_angle_deg_arg,
        alpha_arg,
        target_distance_arg,
        distance_tolerance_arg,
        center_tolerance_arg,
        angle_tolerance_deg_arg,
        target_wall_angle_deg_arg,
        wall_align_tol_deg_arg,
        allow_retreat_arg,
        osd_bg_alpha_arg,
        csv_log_path_arg,
        docking_node,
    ])
