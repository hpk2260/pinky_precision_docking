#!/usr/bin/env python3
"""
Pinky Pro Precision Auto-Docking Node (v2.0 Advanced).
Features:
- ArUco marker pose estimation (OpenCV 4.6+ / 4.7+ compatible via solvePnP)
- Multi-stage FSM with Wall-Parallel Alignment (ALIGN_WALL)
- Odometry-assisted Blind Final Drive for ultra-close docking (<3cm) without marker loss
- EMA Jitter Filter & Safety Watchdog
- Dynamic Reconfigure & Feature Flags (enable_wall_alignment, enable_blind_push)
"""

from enum import Enum
import math
import signal
import sys
import time
from typing import List, Optional

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Bool


class DockingState(Enum):
    SEARCH = "SEARCH"
    ALIGN_BEARING = "ALIGN_BEARING"
    T_TURN_PARALLEL = "T_TURN_PARALLEL"
    T_DRIVE_LATERAL = "T_DRIVE_LATERAL"
    T_RETURN_FACING = "T_RETURN_FACING"
    FINE_ALIGN = "FINE_ALIGN"
    APPROACH_NORMAL = "APPROACH_NORMAL"
    ALIGN_WALL = "ALIGN_WALL"
    FINAL_APPROACH = "FINAL_APPROACH"
    BLIND_PUSH = "BLIND_PUSH"
    DOCKED = "DOCKED"
    LOST = "LOST"


class PrecisionDockingNode(Node):

    def __init__(self):
        super().__init__('precision_docking_node')

        self._declare_and_get_parameters()
        self.bridge = CvBridge()

        # FSM State & tracking variables
        self.state = DockingState.SEARCH
        self.last_state = None
        self.last_detection_time = 0.0
        self.consecutive_detect_count = 0
        self.min_consecutive_frames = 1
        self.last_known_bearing: Optional[float] = None
        self.last_turn_dir: float = 1.0  # +1.0 for CCW (left), -1.0 for CW (right)

        # Filtered marker pose [x, y, z, wall_angle, signed_lat, dist_along_normal]
        self.filtered_pose = None
        self.raw_pose = None
        self.last_R: Optional[np.ndarray] = None
        self.last_normal_u = None
        self.last_closest_pt = None

        # Odometry tracking for blind final drive and T-maneuver
        self.current_odom_x = 0.0
        self.current_odom_y = 0.0
        self.current_odom_yaw = 0.0
        self.blind_start_x = 0.0
        self.blind_start_y = 0.0
        self.blind_target_dist = 0.0
        self.turn_start_yaw = 0.0
        self.t_turn_angle_1 = 0.0
        self.t_turn_angle_2 = 0.0
        self.lat_start_x = 0.0
        self.lat_start_y = 0.0
        self.lat_target_dist = 0.0
        self.t_lateral_offset = 0.0
        self.fine_align_start_time = 0.0
        self.odom_received = False

        # Camera intrinsic matrix & distortion coefficients
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.camera_info_received = False
        self._init_fallback_camera_matrix(640, 480)

        # ArUco detector setup (modern OpenCV compatible)
        self._init_aruco_detector()

        # QoS Profiles
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, reliable_qos)
        self.completed_pub = self.create_publisher(Bool, '/docking_completed', reliable_qos)
        self.debug_image_pub = self.create_publisher(
            CompressedImage, '/docking/debug_image/compressed', sensor_qos
        )

        # Subscribers
        if self.use_compressed:
            compressed_topic = self.image_topic if 'compressed' in self.image_topic else f"{self.image_topic}/compressed"
            self.get_logger().info(f"Subscribing to CompressedImage: {compressed_topic}")
            self.image_sub = self.create_subscription(
                CompressedImage, compressed_topic, self._compressed_image_callback, sensor_qos
            )
        else:
            self.get_logger().info(f"Subscribing to raw Image: {self.image_topic}")
            self.image_sub = self.create_subscription(
                Image, self.image_topic, self._image_callback, sensor_qos
            )

        self.info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self._camera_info_callback, sensor_qos
        )

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, reliable_qos
        )

        # Control Loop Timer (20Hz)
        self.control_timer = self.create_timer(0.05, self._control_loop)

        # Dynamic parameter callback
        self.add_on_set_parameters_callback(self._on_parameter_update)

        self.get_logger().info(
            f"PrecisionDockingNode v3.6 (T-Maneuver Collinear Docking) ready! Target ID: {self.target_marker_id}, "
            f"Goal: {self.target_distance}m, TManeuver: {self.enable_t_maneuver} (min_dist: {self.t_maneuver_min_dist}m), BlindPush: {self.enable_blind_push}"
        )

    def _declare_and_get_parameters(self):
        """Declare ROS 2 parameters with defaults and load them."""
        # Marker settings
        self.declare_parameter('marker_size', 0.06, ParameterDescriptor(description='Marker physical size (meters)'))
        self.declare_parameter('target_marker_id', 0, ParameterDescriptor(description='Target ArUco Marker ID (-1 for auto-lock)'))
        self.declare_parameter('aruco_dictionary_id', 'DICT_5X5_250', ParameterDescriptor(description='ArUco dictionary name'))

        # Topic settings
        self.declare_parameter('use_compressed', True, ParameterDescriptor(description='Use CompressedImage stream'))
        self.declare_parameter('image_topic', '/camera/image_raw', ParameterDescriptor(description='Base camera image topic'))
        self.declare_parameter('camera_info_topic', '/camera/camera_info', ParameterDescriptor(description='CameraInfo topic'))
        self.declare_parameter('cmd_vel_topic', '/cmd_vel', ParameterDescriptor(description='Twist command velocity topic'))

        # Goals and tolerances
        self.declare_parameter('target_distance', 0.20, ParameterDescriptor(description='Target stop distance (meters)'))
        self.declare_parameter('distance_tolerance', 0.02, ParameterDescriptor(description='Distance tolerance (meters)'))
        self.declare_parameter('yaw_tolerance', 0.04, ParameterDescriptor(description='Yaw tolerance (radians)'))
        self.declare_parameter('lateral_tolerance', 0.02, ParameterDescriptor(description='Lateral X tolerance (meters)'))

        # Advanced Solutions (Feature Flags)
        self.declare_parameter('enable_t_maneuver', True, ParameterDescriptor(description='Use orthogonal T-maneuver (90 deg turn -> lateral drive -> 90 deg return) for collinear alignment'))
        self.declare_parameter('t_maneuver_min_dist', 0.20, ParameterDescriptor(description='Minimum distance from marker to perform T-maneuver (m)'))
        self.declare_parameter('t_maneuver_lateral_speed', 0.05, ParameterDescriptor(description='Linear velocity during T-maneuver lateral drive (m/s)'))
        self.declare_parameter('enable_wall_alignment', False, ParameterDescriptor(description='Align robot parallel to marker wall'))
        self.declare_parameter('enable_blind_push', True, ParameterDescriptor(description='Use odometry for <10cm final approach'))
        self.declare_parameter('vision_min_distance', 0.14, ParameterDescriptor(description='Distance before switching to blind drive (m)'))
        self.declare_parameter('blind_push_speed', 0.04, ParameterDescriptor(description='Linear velocity during blind final drive (m/s)'))

        # Controller parameters
        self.declare_parameter('kp_linear', 0.5, ParameterDescriptor(description='Linear velocity proportional gain'))
        self.declare_parameter('kp_angular', 1.2, ParameterDescriptor(description='Angular velocity proportional gain'))
        self.declare_parameter('max_linear_vel', 0.10, ParameterDescriptor(description='Maximum linear velocity (m/s)'))
        self.declare_parameter('max_angular_vel', 0.35, ParameterDescriptor(description='Maximum angular velocity (rad/s)'))
        self.declare_parameter('min_linear_vel', 0.02, ParameterDescriptor(description='Minimum linear velocity to overcome friction'))
        self.declare_parameter('min_angular_vel', 0.05, ParameterDescriptor(description='Minimum angular velocity'))
        self.declare_parameter('invert_angular', False, ParameterDescriptor(description='Invert steering direction'))

        # Safety & filter
        self.declare_parameter('loss_timeout', 1.2, ParameterDescriptor(description='Timeout before treating marker as lost (s)'))
        self.declare_parameter('ema_alpha', 0.6, ParameterDescriptor(description='EMA filter weight (0.0 to 1.0)'))
        self.declare_parameter('search_angular_vel', 0.20, ParameterDescriptor(description='Search rotation velocity (rad/s)'))

        self._load_parameters()

    def _load_parameters(self):
        """Cache parameter values into instance variables."""
        self.marker_size = self.get_parameter('marker_size').value
        self.target_marker_id = self.get_parameter('target_marker_id').value
        self.aruco_dict_name = self.get_parameter('aruco_dictionary_id').value
        self.use_compressed = self.get_parameter('use_compressed').value
        self.image_topic = self.get_parameter('image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.target_distance = self.get_parameter('target_distance').value
        self.distance_tolerance = self.get_parameter('distance_tolerance').value
        self.yaw_tolerance = self.get_parameter('yaw_tolerance').value
        self.lateral_tolerance = self.get_parameter('lateral_tolerance').value
        self.enable_t_maneuver = self.get_parameter('enable_t_maneuver').value
        self.t_maneuver_min_dist = self.get_parameter('t_maneuver_min_dist').value
        self.t_maneuver_lateral_speed = self.get_parameter('t_maneuver_lateral_speed').value
        self.enable_wall_alignment = self.get_parameter('enable_wall_alignment').value
        self.enable_blind_push = self.get_parameter('enable_blind_push').value
        self.vision_min_distance = self.get_parameter('vision_min_distance').value
        self.blind_push_speed = self.get_parameter('blind_push_speed').value
        self.kp_linear = self.get_parameter('kp_linear').value
        self.kp_angular = self.get_parameter('kp_angular').value
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        self.min_linear_vel = self.get_parameter('min_linear_vel').value
        self.min_angular_vel = self.get_parameter('min_angular_vel').value
        self.invert_angular = self.get_parameter('invert_angular').value
        self.loss_timeout = self.get_parameter('loss_timeout').value
        self.ema_alpha = self.get_parameter('ema_alpha').value
        self.search_angular_vel = self.get_parameter('search_angular_vel').value

    def _on_parameter_update(self, params: List[Parameter]) -> SetParametersResult:
        """Handle dynamic parameter updates at runtime."""
        dict_changed = False
        for param in params:
            if hasattr(self, param.name):
                setattr(self, param.name, param.value)
                if param.name == 'aruco_dictionary_id':
                    self.aruco_dict_name = param.value
                    dict_changed = True
                self.get_logger().info(f"Parameter '{param.name}' dynamically updated to: {param.value}")
        if dict_changed:
            self._init_aruco_detector()
            self.get_logger().info(f"ArUco Dictionary reinitialized to: {self.aruco_dict_name}")
        return SetParametersResult(successful=True)

    def _init_fallback_camera_matrix(self, width: int, height: int):
        """Construct fallback intrinsic camera matrix when CameraInfo is not yet received."""
        fx = fy = 0.8 * width
        cx = width / 2.0
        cy = height / 2.0
        self.camera_matrix = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    def _init_aruco_detector(self):
        """Initialize OpenCV ArUco detector strictly for 5x5 markers."""
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.aruco_parameters = cv2.aruco.DetectorParameters_create()
        else:
            self.aruco_parameters = cv2.aruco.DetectorParameters()

        # Enhance detection across lighting, glare, and distance
        self.aruco_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_parameters.adaptiveThreshWinSizeMin = 3
        self.aruco_parameters.adaptiveThreshWinSizeMax = 23
        self.aruco_parameters.adaptiveThreshWinSizeStep = 10
        self.aruco_parameters.adaptiveThreshConstant = 7
        self.aruco_parameters.minMarkerPerimeterRate = 0.02
        self.aruco_parameters.maxMarkerPerimeterRate = 4.0

        # Strictly 5X5 dictionary
        dict_attr = getattr(cv2.aruco, self.aruco_dict_name, cv2.aruco.DICT_5X5_250)
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dict_attr)

    def _camera_info_callback(self, msg: CameraInfo):
        """Receive true camera intrinsics from camera driver."""
        if not self.camera_info_received:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d, dtype=np.float64).reshape((-1, 1))
            self.camera_info_received = True
            self.get_logger().info("Received CameraInfo! Updated intrinsic calibration matrix.")

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Calculate shortest directed angular difference (a - b) wrapped into [-pi, pi]."""
        return math.atan2(math.sin(a - b), math.cos(a - b))

    def _odom_callback(self, msg: Odometry):
        """Track current robot odometry for blind final drive and T-maneuver."""
        self.current_odom_x = msg.pose.pose.position.x
        self.current_odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_odom_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.odom_received = True

    def _compressed_image_callback(self, msg: CompressedImage):
        """Decode incoming compressed JPEG image."""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is not None:
                self._process_image(cv_image)
        except Exception as e:
            self.get_logger().error(f"Failed to decode CompressedImage: {e}")

    def _image_callback(self, msg: Image):
        """Convert incoming raw ROS Image to OpenCV BGR."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self._process_image(cv_image)
        except Exception as e:
            self.get_logger().error(f"Failed to convert Image: {e}")

    def _process_image(self, cv_image: np.ndarray):
        """Run ArUco marker detection, 3D solvePnP pose estimation, and EMA filter."""
        h, w = cv_image.shape[:2]
        if not self.camera_info_received:
            self._init_fallback_camera_matrix(w, h)

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Strictly detect 5x5 ArUco markers using 5x5 dictionary
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dictionary, parameters=self.aruco_parameters)

        target_detected = False
        target_rvec = None
        target_tvec = None

        if ids is not None and len(ids) > 0:
            # ALWAYS draw all detected markers so they are visible on debug feed
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)

            for idx, marker_id in enumerate(ids.flatten()):
                marker_id_int = int(marker_id)
                # Auto-lock onto first detected marker if target_marker_id is -1
                if self.target_marker_id == -1:
                    self.target_marker_id = marker_id_int
                    self.get_logger().info(
                        f"[Auto-Lock] Successfully locked onto detected Marker ID: {self.target_marker_id} ({self.aruco_dict_name})"
                    )

                if marker_id_int == self.target_marker_id:
                    target_detected = True
                    target_corner = corners[idx]

                    s = self.marker_size
                    obj_points = np.array([
                        [-s / 2.0,  s / 2.0, 0.0],
                        [ s / 2.0,  s / 2.0, 0.0],
                        [ s / 2.0, -s / 2.0, 0.0],
                        [-s / 2.0, -s / 2.0, 0.0]
                    ], dtype=np.float32)

                    success, rvec, tvec = cv2.solvePnP(
                        obj_points, target_corner, self.camera_matrix, self.dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                    if success:
                        target_rvec = rvec
                        target_tvec = tvec
                    break

            if not target_detected:
                self.get_logger().warn(
                    f"Detected Marker ID(s) {ids.flatten().tolist()}, but waiting for target_marker_id={self.target_marker_id}! "
                    f"(Pass target_marker_id:=-1 to auto-lock onto any ID)",
                    throttle_duration_sec=2.0
                )

        now = time.time()
        if target_detected and target_tvec is not None:
            self.last_detection_time = now
            self.consecutive_detect_count += 1

            x = float(target_tvec[0][0])
            y = float(target_tvec[1][0])
            z = float(target_tvec[2][0])

            # Marker plane normal vector in camera coords: R[:, 2]
            R, _ = cv2.Rodrigues(target_rvec)
            nx = float(R[0, 2])
            ny = float(R[1, 2])
            nz = float(R[2, 2])

            # Project normal vector onto horizontal floor plane (X_c, Z_c)
            # This is 100% invariant to marker roll or whether marker is mounted sideways/upright!
            norm_horiz = math.hypot(nx, nz)
            if norm_horiz > 1e-4:
                nx_u = nx / norm_horiz
                nz_u = nz / norm_horiz
            else:
                nx_u, nz_u = 0.0, -1.0

            # Inward wall normal pointing INTO the wall: [-nx_u, -nz_u]
            # wall_angle is 0.0 when camera directly faces the wall squarely
            wall_angle = math.atan2(-nx_u, -nz_u)

            # Distance along normal from marker to camera:
            # Marker normal points OUT of wall towards robot:
            dist_along_normal = -(x * nx_u + z * nz_u)

            # Closest point on normal ray in camera horizontal coordinates:
            closest_x = x + dist_along_normal * nx_u
            closest_z = z + dist_along_normal * nz_u

            # Signed lateral offset: positive when robot is to the RIGHT of normal axis
            # Vector pointing to the right of normal vector is [nz_u, -nx_u]
            signed_lat = (0.0 - closest_x) * nz_u + (0.0 - closest_z) * (-nx_u)

            self.last_R = R
            self.last_normal_u = (nx_u, nz_u)
            self.last_closest_pt = (closest_x, closest_z)
            self.last_known_bearing = math.atan2(x, z)
            self.raw_pose = (x, y, z, wall_angle, signed_lat, dist_along_normal)

            # Exponential Moving Average (EMA) Jitter Filter
            if self.filtered_pose is None:
                self.filtered_pose = [x, y, z, wall_angle, signed_lat, dist_along_normal]
            else:
                a = self.ema_alpha
                self.filtered_pose[0] = a * x + (1.0 - a) * self.filtered_pose[0]
                self.filtered_pose[1] = a * y + (1.0 - a) * self.filtered_pose[1]
                self.filtered_pose[2] = a * z + (1.0 - a) * self.filtered_pose[2]
                self.filtered_pose[3] = a * wall_angle + (1.0 - a) * self.filtered_pose[3]
                self.filtered_pose[4] = a * signed_lat + (1.0 - a) * self.filtered_pose[4]
                self.filtered_pose[5] = a * dist_along_normal + (1.0 - a) * self.filtered_pose[5]

            cv2.drawFrameAxes(
                cv_image, self.camera_matrix, self.dist_coeffs,
                target_rvec, target_tvec, self.marker_size * 0.8
            )
        else:
            self.consecutive_detect_count = 0

        self._draw_osd(cv_image, ids, target_detected)
        self._publish_debug_image(cv_image)

    def _draw_osd(self, image: np.ndarray, detected_ids=None, target_detected: bool = False):
        """Render informative On-Screen Display (OSD) overlay on camera feed."""
        h, w = image.shape[:2]
        cv2.rectangle(image, (0, 0), (w, 85), (20, 20, 20), -1)

        state_color = {
            DockingState.SEARCH: (0, 165, 255),
            DockingState.ALIGN_BEARING: (255, 200, 0),
            DockingState.T_TURN_PARALLEL: (255, 140, 0),
            DockingState.T_DRIVE_LATERAL: (255, 80, 20),
            DockingState.T_RETURN_FACING: (200, 100, 255),
            DockingState.FINE_ALIGN: (0, 200, 255),
            DockingState.APPROACH_NORMAL: (0, 255, 200),
            DockingState.ALIGN_WALL: (255, 50, 200),
            DockingState.FINAL_APPROACH: (0, 255, 255),
            DockingState.BLIND_PUSH: (255, 100, 255),
            DockingState.DOCKED: (0, 255, 0),
            DockingState.LOST: (0, 0, 255)
        }.get(self.state, (200, 200, 200))

        dict_info = f"Dict: {self.aruco_dict_name} | ID: {self.target_marker_id}"
        cv2.putText(image, f"PINKY DOCKING v3.5 ({dict_info})", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        cv2.putText(image, f"STATE: {self.state.value}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, state_color, 2)

        if self.state == DockingState.T_TURN_PARALLEL:
            turned = self._angle_diff(self.current_odom_yaw, self.turn_start_yaw)
            cv2.putText(image, f"T-Maneuver: Turn 1 ({math.degrees(turned):+.1f}/{math.degrees(self.t_turn_angle_1):+.1f} deg)", (w - 530, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 140, 0), 2)
        elif self.state == DockingState.T_DRIVE_LATERAL:
            moved = math.hypot(self.current_odom_x - self.lat_start_x, self.current_odom_y - self.lat_start_y)
            cv2.putText(image, f"T-Maneuver: Lateral Drive ({moved*100:.1f}/{self.lat_target_dist*100:.1f}cm)", (w - 530, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 20), 2)
        elif self.state == DockingState.T_RETURN_FACING:
            turned = self._angle_diff(self.current_odom_yaw, self.turn_start_yaw)
            cv2.putText(image, f"T-Maneuver: Return Turn ({math.degrees(turned):+.1f}/{math.degrees(self.t_turn_angle_2):+.1f} deg)", (w - 530, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 100, 255), 2)
        elif self.state == DockingState.FINE_ALIGN:
            cv2.putText(image, "T-Maneuver: Fine Centering on Normal Axis...", (w - 530, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        elif self.state == DockingState.BLIND_PUSH:
            moved = math.hypot(self.current_odom_x - self.blind_start_x, self.current_odom_y - self.blind_start_y)
            info_str = f"Blind Push: {moved*100:.1f}cm / {self.blind_target_dist*100:.1f}cm"
            cv2.putText(image, info_str, (w - 450, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        elif self.state == DockingState.LOST:
            cv2.putText(image, "LOST - Active Re-acquisition Scanning...", (w - 490, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 165, 255), 2)
        elif detected_ids is not None and len(detected_ids) > 0 and not target_detected:
            warn_str = f"Seen ID {detected_ids.flatten().tolist()} != Target {self.target_marker_id}"
            cv2.putText(image, warn_str, (w - 480, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 165, 255), 2)
        elif self.filtered_pose is not None:
            x, y, z, wall_angle, signed_lat, dist_along_normal = self.filtered_pose
            dist_err = z - self.target_distance
            pose_str = f"Dist: {z:.3f}m (Goal: {self.target_distance:.2f}m) | NormalDist: {dist_along_normal:.3f}m"
            lat_str = f"LatErr: {signed_lat*100:+.1f}cm | AngleErr: {math.degrees(wall_angle):+.1f} deg"
            cv2.putText(image, pose_str, (w - 530, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)
            cv2.putText(image, lat_str, (w - 530, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)
        else:
            cv2.putText(image, "Searching for ArUco marker...", (w - 450, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    def _publish_debug_image(self, image: np.ndarray):
        """Compress and publish debug visualization image."""
        if self.debug_image_pub.get_subscription_count() > 0:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]
            success, encoded_img = cv2.imencode('.jpg', image, encode_param)
            if success:
                msg = CompressedImage()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.format = "jpeg"
                msg.data = encoded_img.tobytes()
                self.debug_image_pub.publish(msg)

    def _control_loop(self):
        """Main 20Hz FSM Control Loop."""
        now = time.time()
        time_since_last_detection = now - self.last_detection_time

        # Watchdog: During T-maneuver open-loop phases, BLIND_PUSH or DOCKED, marker loss is expected and ignored.
        # In SEARCH or LOST, state-specific recovery handlers take over.
        if self.state not in [
            DockingState.SEARCH, DockingState.LOST,
            DockingState.T_TURN_PARALLEL, DockingState.T_DRIVE_LATERAL, DockingState.T_RETURN_FACING,
            DockingState.BLIND_PUSH, DockingState.DOCKED
        ]:
            if time_since_last_detection > self.loss_timeout:
                self.get_logger().warn(
                    f"Marker lost for {time_since_last_detection:.2f}s (>{self.loss_timeout}s) in state {self.state.value}! "
                    f"Initiating active recovery."
                )
                self.last_state = self.state
                self.state = DockingState.LOST

        twist = Twist()
        completed = False

        if self.state == DockingState.SEARCH:
            self._handle_search(twist)

        elif self.state == DockingState.LOST:
            self._handle_lost(twist, time_since_last_detection)

        elif self.state == DockingState.ALIGN_BEARING:
            self._handle_align_bearing(twist)

        elif self.state == DockingState.T_TURN_PARALLEL:
            self._handle_t_turn_parallel(twist)

        elif self.state == DockingState.T_DRIVE_LATERAL:
            self._handle_t_drive_lateral(twist)

        elif self.state == DockingState.T_RETURN_FACING:
            self._handle_t_return_facing(twist)

        elif self.state == DockingState.FINE_ALIGN:
            self._handle_fine_align(twist)

        elif self.state == DockingState.APPROACH_NORMAL:
            self._handle_approach_normal(twist)

        elif self.state == DockingState.ALIGN_WALL:
            self._handle_align_wall(twist)

        elif self.state == DockingState.FINAL_APPROACH:
            self._handle_final_approach(twist)

        elif self.state == DockingState.BLIND_PUSH:
            self._handle_blind_push(twist)

        elif self.state == DockingState.DOCKED:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            completed = True

        if self.state in [
            DockingState.T_TURN_PARALLEL, DockingState.T_DRIVE_LATERAL, DockingState.T_RETURN_FACING,
            DockingState.FINE_ALIGN, DockingState.APPROACH_NORMAL, DockingState.ALIGN_WALL,
            DockingState.FINAL_APPROACH, DockingState.BLIND_PUSH
        ]:
            if self.filtered_pose is not None:
                x, _, z, wall_angle, signed_lat, dist_along_normal = self.filtered_pose
                self.get_logger().info(
                    f"[{self.state.value}] Dist: {z:.2f}m, LatErr: {signed_lat*100:+.1f}cm, AngleErr: {math.degrees(wall_angle):+.1f} deg | Cmd: v={twist.linear.x:.2f}m/s, w={twist.angular.z:+.2f}rad/s",
                    throttle_duration_sec=0.5
                )

        self._publish_twist(twist)

        status_msg = Bool()
        status_msg.data = completed
        self.completed_pub.publish(status_msg)

    def _handle_search(self, twist: Twist):
        """Rotate slowly until 5x5 marker is detected."""
        if self.consecutive_detect_count >= 1 and self.filtered_pose is not None:
            self.get_logger().info("5x5 Marker acquired! Instantly stopping rotation and transitioning to ALIGN_BEARING.")
            self.state = DockingState.ALIGN_BEARING
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        else:
            twist.linear.x = 0.0
            twist.angular.z = -self.search_angular_vel if self.invert_angular else self.search_angular_vel

    def _handle_lost(self, twist: Twist, time_since_loss: float):
        """
        Smart Lost Handler with Active Re-acquisition:
        - If marker detected again, immediately re-acquire and resume ALIGN_BEARING.
        - Phase 1 (0.0s ~ 1.5s after loss_timeout): Gently reverse turn or scan toward last known bearing.
        - Phase 2 (1.5s ~ 3.0s): Stand still to let camera/network stabilize.
        - Phase 3 (> 3.0s): Reset to full 360 SEARCH mode.
        """
        twist.linear.x = 0.0

        if self.consecutive_detect_count >= 1:
            self.get_logger().info("Marker RE-ACQUIRED! Resuming ALIGN_BEARING.")
            self.state = DockingState.ALIGN_BEARING
            twist.angular.z = 0.0
            return

        lost_duration = time_since_loss - self.loss_timeout

        if lost_duration < 1.5:
            # Phase 1: Active recovery scan
            recovery_w = 0.12
            if self.last_known_bearing is not None and abs(self.last_known_bearing) > 0.03:
                # If marker was on the right (bearing > 0), turn right (angular.z < 0)
                scan_dir = -1.0 if self.last_known_bearing > 0 else 1.0
            else:
                # Reverse the last turn direction
                scan_dir = -self.last_turn_dir

            if self.invert_angular:
                scan_dir = -scan_dir

            twist.angular.z = scan_dir * recovery_w
            self.get_logger().info(
                f"[LOST Active Recovery] Reversing turn to re-acquire marker (w={twist.angular.z:+.2f} rad/s, t={lost_duration:.1f}s)",
                throttle_duration_sec=0.5
            )
        elif lost_duration < 3.0:
            # Phase 2: Stand still and give camera / Wi-Fi buffer a chance to catch up
            twist.angular.z = 0.0
            self.get_logger().info(
                f"[LOST Paused] Waiting for camera stream... (t={lost_duration:.1f}s)",
                throttle_duration_sec=0.5
            )
        else:
            # Phase 3: Give up active recovery, reset to full 360 SEARCH
            self.get_logger().warn("[LOST Timeout] Marker not found during active recovery. Resetting to SEARCH mode.")
            self.filtered_pose = None
            self.last_known_bearing = None
            self.state = DockingState.SEARCH
            twist.angular.z = 0.0

    def _handle_align_bearing(self, twist: Twist):
        """Rotate in place to center marker in camera view."""
        x, _, z, wall_angle, signed_lat, dist_along_normal = self.filtered_pose
        bearing_error = math.atan2(x, z)

        if abs(bearing_error) <= 0.03:  # ~1.7 deg
            twist.angular.z = 0.0

            # Check if T-Maneuver should be engaged:
            # Trigger if:
            # 1. enable_t_maneuver is True
            # 2. z >= t_maneuver_min_dist
            # 3. EITHER lateral offset > tolerance OR angle error > tolerance!
            lateral_err = abs(signed_lat)
            angle_err = abs(wall_angle)
            needs_t_maneuver = (lateral_err > self.lateral_tolerance) or (angle_err > self.yaw_tolerance)

            if self.enable_t_maneuver and (z >= self.t_maneuver_min_dist) and needs_t_maneuver and self.last_closest_pt is not None:
                closest_x, closest_z = self.last_closest_pt
                dist_to_normal = math.hypot(closest_x, closest_z)

                # Bearing in camera frame towards closest point on normal ray:
                bearing_to_normal = math.atan2(closest_x, closest_z)
                # In robot body frame, turning towards this bearing:
                delta_theta = -bearing_to_normal

                self.t_turn_angle_1 = delta_theta
                self.t_lateral_offset = dist_to_normal
                self.turn_start_yaw = self.current_odom_yaw

                turn_dir_str = "LEFT" if delta_theta > 0 else "RIGHT"
                self.get_logger().info(
                    f"=== T-MANEUVER INITIATED === Angle Error: {math.degrees(wall_angle):+.1f} deg, "
                    f"Lateral Error: {signed_lat*100:+.1f}cm, Dist: {z:.2f}m. "
                    f"Step 1: Turning {turn_dir_str} by {math.degrees(delta_theta):+.1f} deg parallel to wall."
                )
                self.state = DockingState.T_TURN_PARALLEL
                return
            else:
                self.get_logger().info(
                    f"Bearing centered! (z={z:.2f}m, lat_err={signed_lat*100:+.1f}cm, angle={math.degrees(wall_angle):+.1f} deg). "
                    f"Transitioning directly to APPROACH_NORMAL."
                )
                self.state = DockingState.APPROACH_NORMAL
        else:
            ang_cmd = self.kp_angular * bearing_error
            twist.angular.z = self._clamp_angular(-ang_cmd)
            twist.linear.x = 0.0
            if abs(twist.angular.z) > 0.02:
                self.last_turn_dir = 1.0 if twist.angular.z > 0 else -1.0

    def _handle_t_turn_parallel(self, twist: Twist):
        """Step 1 of T-Maneuver: Rotate in place until optical axis is parallel to marker wall."""
        turned = self._angle_diff(self.current_odom_yaw, self.turn_start_yaw)
        remaining = self.t_turn_angle_1 - turned

        if abs(remaining) <= 0.025:  # ~1.4 deg tolerance
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.lat_start_x = self.current_odom_x
            self.lat_start_y = self.current_odom_y
            self.lat_target_dist = abs(self.t_lateral_offset)
            self.get_logger().info(
                f"Step 1 Complete (turned {math.degrees(turned):+.1f} deg). "
                f"Step 2: Driving laterally {self.lat_target_dist*100:.1f}cm parallel to wall."
            )
            self.state = DockingState.T_DRIVE_LATERAL
        else:
            ang_cmd = self.kp_angular * remaining
            twist.linear.x = 0.0
            twist.angular.z = self._clamp_angular(ang_cmd)

    def _handle_t_drive_lateral(self, twist: Twist):
        """Step 2 of T-Maneuver: Drive straight along wall until reaching marker normal axis."""
        moved = math.hypot(self.current_odom_x - self.lat_start_x, self.current_odom_y - self.lat_start_y)
        remaining = self.lat_target_dist - moved

        if remaining <= 0.005:  # within 5mm
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.turn_start_yaw = self.current_odom_yaw
            # At P_closest, the robot heading is parallel to the wall (orthogonal to the normal axis).
            # To face directly along the marker normal axis towards the marker, the robot must turn exactly 90 deg!
            self.t_turn_angle_2 = -math.copysign(math.pi / 2.0, self.t_turn_angle_1)
            self.get_logger().info(
                f"Step 2 Complete (moved {moved*100:.1f}cm). "
                f"Step 3: Turning {math.degrees(self.t_turn_angle_2):+.1f} deg (orthogonal to wall) to face marker squarely on normal axis."
            )
            self.state = DockingState.T_RETURN_FACING
        else:
            speed = min(self.t_maneuver_lateral_speed, self.kp_linear * remaining)
            speed = max(speed, self.min_linear_vel)
            twist.linear.x = self._clamp_linear(speed)
            twist.angular.z = 0.0

    def _handle_t_return_facing(self, twist: Twist):
        """Step 3 of T-Maneuver: Rotate in place back to face the marker directly."""
        turned = self._angle_diff(self.current_odom_yaw, self.turn_start_yaw)
        remaining = self.t_turn_angle_2 - turned

        if abs(remaining) <= 0.025:  # ~1.4 deg tolerance
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.fine_align_start_time = time.time()
            self.last_detection_time = time.time()  # grant grace period for visual re-acquisition
            self.get_logger().info(
                f"Step 3 Complete (turned {math.degrees(turned):+.1f} deg). "
                f"Step 4: Performing fine visual centering on normal axis..."
            )
            self.state = DockingState.FINE_ALIGN
        else:
            ang_cmd = self.kp_angular * remaining
            twist.linear.x = 0.0
            twist.angular.z = self._clamp_angular(ang_cmd)

    def _handle_fine_align(self, twist: Twist):
        """Step 4 of T-Maneuver: Fine visual bearing centering on normal axis before straight drive."""
        twist.linear.x = 0.0
        now = time.time()

        # Check if marker detected
        if self.consecutive_detect_count >= 1 and self.filtered_pose is not None:
            x, _, z, wall_angle, signed_lat, dist_along_normal = self.filtered_pose
            bearing_error = math.atan2(x, z)

            if abs(bearing_error) <= 0.025:  # ~1.4 deg
                twist.angular.z = 0.0
                self.get_logger().info(
                    f"=== T-MANEUVER COMPLETE! Collinear Alignment achieved! === "
                    f"LatErr: {signed_lat*100:+.1f}cm, AngleErr: {math.degrees(wall_angle):+.1f} deg, Bearing: {math.degrees(bearing_error):+.1f} deg, Dist: {z:.2f}m. "
                    f"Proceeding to straight-line docking."
                )
                self.state = DockingState.APPROACH_NORMAL
                return
            else:
                ang_cmd = self.kp_angular * bearing_error
                twist.angular.z = self._clamp_angular(-ang_cmd)
                return

        # If not yet detected within grace period, stand still and wait
        if now - self.fine_align_start_time > 3.0:
            self.get_logger().warn("Fine align timeout! Marker not seen, resuming ALIGN_BEARING.")
            self.state = DockingState.ALIGN_BEARING

    def _handle_align_on_axis(self, twist: Twist):
        """Compatibility alias for ALIGN_ON_AXIS."""
        self.state = DockingState.APPROACH_NORMAL

    def _handle_approach_normal(self, twist: Twist):
        """
        Advance along collinear normal axis while maintaining pure bearing centering.
        Transitions smoothly to FINAL_APPROACH when within 0.25m.
        """
        x, _, z, wall_angle, signed_lat, dist_along_normal = self.filtered_pose
        dist_err = z - self.target_distance
        bearing_error = math.atan2(x, z)

        # When within 0.25m, transition smoothly to FINAL_APPROACH
        if z <= max(self.vision_min_distance + 0.10, 0.25):
            self.get_logger().info(f"Near wall ({z:.2f}m, LatErr: {signed_lat*100:+.1f}cm). Advancing to FINAL_APPROACH.")
            self.state = DockingState.FINAL_APPROACH
            return

        ang_cmd = self.kp_angular * bearing_error
        twist.angular.z = self._clamp_angular(-ang_cmd)

        if dist_err > 0.03:
            lin_cmd = self.kp_linear * 0.7 * dist_err
            twist.linear.x = self._clamp_linear(lin_cmd)
        else:
            twist.linear.x = 0.0

        if abs(twist.angular.z) > 0.02:
            self.last_turn_dir = 1.0 if twist.angular.z > 0 else -1.0

    def _handle_align_wall(self, twist: Twist):
        """Seamlessly hands off to continuous forward blended FINAL_APPROACH."""
        self.state = DockingState.FINAL_APPROACH
        self._handle_final_approach(twist)

    def _handle_final_approach(self, twist: Twist):
        """
        Final approach driving directly into docking target.
        Smoothly blends wall parallel orientation so robot arrives perpendicular to wall without stopping in place.
        If target_distance is < vision_min_distance, hands off to BLIND_PUSH.
        """
        x, _, z, wall_angle, signed_lat, dist_along_normal = self.filtered_pose
        dist_err = z - self.target_distance
        bearing_error = math.atan2(x, z)

        # Check if we should hand off to BLIND_PUSH for ultra-close docking (<10cm)
        if self.enable_blind_push and self.target_distance < self.vision_min_distance:
            if z <= self.vision_min_distance:
                push_dist = z - self.target_distance
                self.blind_start_x = self.current_odom_x
                self.blind_start_y = self.current_odom_y
                self.blind_target_dist = push_dist
                self.get_logger().info(
                    f"Reached vision limit ({z*100:.1f}cm, LatErr: {signed_lat*100:+.1f}cm). Switching to BLIND PUSH for final {push_dist*100:.1f}cm using Odometry!"
                )
                self.state = DockingState.BLIND_PUSH
                return

        # Check standard completion condition
        if abs(dist_err) <= self.distance_tolerance and abs(bearing_error) <= 0.05:
            self.get_logger().info(f"=== DOCKING COMPLETED! === Final Dist: {z:.3f}m, LatErr: {signed_lat*100:+.1f}cm, AngleErr: {math.degrees(wall_angle):+.1f} deg")
            self.state = DockingState.DOCKED
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            return

        # Pure Bearing Pursuit during final approach:
        # Strictly steer towards the center of the marker (bearing_error)
        # to ensure the camera never turns away from the marker and never causes lost state!
        ang_cmd = self.kp_angular * 1.2 * bearing_error
        twist.angular.z = self._clamp_angular(-ang_cmd)

        lin_cmd = self.kp_linear * dist_err
        twist.linear.x = self._clamp_linear(lin_cmd)

        if abs(twist.angular.z) > 0.02:
            self.last_turn_dir = 1.0 if twist.angular.z > 0 else -1.0

    def _handle_blind_push(self, twist: Twist):
        """
        Odometry-guided open-loop final drive.
        Drives straight forward without camera dependence to achieve <3cm docking.
        """
        moved = math.hypot(self.current_odom_x - self.blind_start_x, self.current_odom_y - self.blind_start_y)
        remaining = self.blind_target_dist - moved

        if remaining <= 0.005:  # within 5mm
            self.get_logger().info(
                f"=== BLIND DOCKING COMPLETED! === Traveled: {moved*100:.1f}cm, Target was: {self.blind_target_dist*100:.1f}cm"
            )
            self.state = DockingState.DOCKED
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        else:
            twist.linear.x = self.blind_push_speed
            twist.angular.z = 0.0

    def _clamp_linear(self, val: float) -> float:
        """Clamp linear velocity to [-max_linear_vel, max_linear_vel] with deadband."""
        if abs(val) < 0.005:
            return 0.0
        val = max(min(val, self.max_linear_vel), -self.max_linear_vel)
        if 0.0 < val < self.min_linear_vel:
            val = self.min_linear_vel
        elif -self.min_linear_vel < val < 0.0:
            val = -self.min_linear_vel
        return val

    def _clamp_angular(self, val: float) -> float:
        """Clamp angular velocity to [-max_angular_vel, max_angular_vel]."""
        if self.invert_angular:
            val = -val
        if abs(val) < 0.02:
            return 0.0
        val = max(min(val, self.max_angular_vel), -self.max_angular_vel)
        if 0.0 < abs(val) < self.min_angular_vel:
            val = math.copysign(self.min_angular_vel, val)
        return val

    def _publish_twist(self, twist: Twist):
        """Publish velocity command to robot base."""
        self.cmd_vel_pub.publish(twist)

    def _stop_robot(self):
        """Publish zero velocity repeatedly to reliably halt robot base over DDS Wi-Fi."""
        zero_twist = Twist()
        for _ in range(8):
            self.cmd_vel_pub.publish(zero_twist)
            time.sleep(0.02)


def main(args=None):
    rclpy.init(args=args)
    node = PrecisionDockingNode()

    def sig_handler(sig, frame):
        node.get_logger().info("Ctrl+C (SIGINT) received! Emergency stopping Pinky motors...")
        node._stop_robot()
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        node.get_logger().info("Shutting down PrecisionDockingNode...")
    finally:
        node._stop_robot()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
