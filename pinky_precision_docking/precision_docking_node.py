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
        self.min_consecutive_frames = 3
        self.last_known_bearing: Optional[float] = None
        self.last_turn_dir: float = 1.0  # +1.0 for CCW (left), -1.0 for CW (right)

        # Filtered marker pose [x, y, z, wall_yaw]
        self.filtered_pose = None  # [x, y, z, wall_yaw]
        self.raw_pose = None

        # Odometry tracking for blind final drive
        self.current_odom_x = 0.0
        self.current_odom_y = 0.0
        self.blind_start_x = 0.0
        self.blind_start_y = 0.0
        self.blind_target_dist = 0.0
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
            f"PrecisionDockingNode v2.0 ready! Target ID: {self.target_marker_id}, "
            f"Goal: {self.target_distance}m, WallAlign: {self.enable_wall_alignment}, BlindPush: {self.enable_blind_push}"
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
        self.declare_parameter('yaw_tolerance', 0.05, ParameterDescriptor(description='Yaw tolerance (radians)'))
        self.declare_parameter('lateral_tolerance', 0.03, ParameterDescriptor(description='Lateral X tolerance (meters)'))

        # Advanced Solutions (Feature Flags)
        self.declare_parameter('enable_wall_alignment', True, ParameterDescriptor(description='Align robot parallel to marker wall'))
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
        """Initialize modern OpenCV ArUco dictionary and detector."""
        dict_attr = getattr(cv2.aruco, self.aruco_dict_name, cv2.aruco.DICT_5X5_250)
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dict_attr)

        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.aruco_parameters = cv2.aruco.DetectorParameters_create()
        else:
            self.aruco_parameters = cv2.aruco.DetectorParameters()

        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dictionary, self.aruco_parameters)
        else:
            self.aruco_detector = None

    def _camera_info_callback(self, msg: CameraInfo):
        """Receive true camera intrinsics from camera driver."""
        if not self.camera_info_received:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d, dtype=np.float64).reshape((-1, 1))
            self.camera_info_received = True
            self.get_logger().info("Received CameraInfo! Updated intrinsic calibration matrix.")

    def _odom_callback(self, msg: Odometry):
        """Track current robot odometry for blind final drive."""
        self.current_odom_x = msg.pose.pose.position.x
        self.current_odom_y = msg.pose.pose.position.y
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

        if self.aruco_detector is not None:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dictionary, parameters=self.aruco_parameters)

        target_detected = False
        target_rvec = None
        target_tvec = None

        if ids is not None and len(ids) > 0:
            for idx, marker_id in enumerate(ids.flatten()):
                if self.target_marker_id == -1:
                    self.target_marker_id = int(marker_id)
                    self.get_logger().info(f"[Auto-Lock] Locked onto detected Marker ID: {self.target_marker_id}")

                if marker_id == self.target_marker_id:
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

        now = time.time()
        if target_detected and target_tvec is not None:
            self.last_detection_time = now
            self.consecutive_detect_count += 1

            x = float(target_tvec[0][0])
            y = float(target_tvec[1][0])
            z = float(target_tvec[2][0])

            # Calculate wall normal yaw angle in horizontal plane
            # Marker plane normal vector in camera coords: [R[0, 2], R[1, 2], R[2, 2]]^T
            # Wall normal angle: atan2(R[0, 2], R[2, 2]) -> 0.0 when robot directly faces wall
            R, _ = cv2.Rodrigues(target_rvec)
            wall_yaw = math.atan2(R[0, 2], R[2, 2])

            self.last_known_bearing = math.atan2(x, z)
            self.raw_pose = (x, y, z, wall_yaw)

            # Exponential Moving Average (EMA) Jitter Filter
            if self.filtered_pose is None:
                self.filtered_pose = [x, y, z, wall_yaw]
            else:
                a = self.ema_alpha
                self.filtered_pose[0] = a * x + (1.0 - a) * self.filtered_pose[0]
                self.filtered_pose[1] = a * y + (1.0 - a) * self.filtered_pose[1]
                self.filtered_pose[2] = a * z + (1.0 - a) * self.filtered_pose[2]
                self.filtered_pose[3] = a * wall_yaw + (1.0 - a) * self.filtered_pose[3]

            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            cv2.drawFrameAxes(
                cv_image, self.camera_matrix, self.dist_coeffs,
                target_rvec, target_tvec, self.marker_size * 0.8
            )
        else:
            self.consecutive_detect_count = 0

        self._draw_osd(cv_image)
        self._publish_debug_image(cv_image)

    def _draw_osd(self, image: np.ndarray):
        """Render informative On-Screen Display (OSD) overlay on camera feed."""
        h, w = image.shape[:2]
        cv2.rectangle(image, (0, 0), (w, 85), (20, 20, 20), -1)

        state_color = {
            DockingState.SEARCH: (0, 165, 255),
            DockingState.ALIGN_BEARING: (255, 200, 0),
            DockingState.APPROACH_NORMAL: (255, 120, 0),
            DockingState.ALIGN_WALL: (255, 50, 200),
            DockingState.FINAL_APPROACH: (0, 255, 255),
            DockingState.BLIND_PUSH: (255, 100, 255),
            DockingState.DOCKED: (0, 255, 0),
            DockingState.LOST: (0, 0, 255)
        }.get(self.state, (200, 200, 200))

        cv2.putText(image, "PINKY PRO AUTO-DOCKING v2.0", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(image, f"STATE: {self.state.value}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, state_color, 2)

        if self.state == DockingState.BLIND_PUSH:
            moved = math.hypot(self.current_odom_x - self.blind_start_x, self.current_odom_y - self.blind_start_y)
            info_str = f"Blind Push: {moved*100:.1f}cm / {self.blind_target_dist*100:.1f}cm"
            cv2.putText(image, info_str, (w - 450, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        elif self.state == DockingState.LOST:
            cv2.putText(image, "LOST - Active Re-acquisition Scanning...", (w - 490, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 165, 255), 2)
        elif self.filtered_pose is not None:
            x, y, z, wall_yaw = self.filtered_pose
            dist_err = z - self.target_distance
            pose_str = f"Dist: {z:.3f}m (Goal: {self.target_distance:.2f}m, Err: {dist_err:+.3f}m)"
            lat_str = f"LatErr(X): {x:+.3f}m | WallAngle: {math.degrees(wall_yaw):+.1f} deg"
            cv2.putText(image, pose_str, (w - 480, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            cv2.putText(image, lat_str, (w - 480, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        else:
            cv2.putText(image, "Waiting for marker detection...", (w - 450, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

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

        # Watchdog: During BLIND_PUSH or DOCKED, marker loss is expected and ignored.
        # In SEARCH or LOST, state-specific recovery handlers take over.
        if self.state not in [DockingState.SEARCH, DockingState.LOST, DockingState.BLIND_PUSH, DockingState.DOCKED]:
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

        if self.state in [DockingState.APPROACH_NORMAL, DockingState.ALIGN_WALL, DockingState.FINAL_APPROACH, DockingState.BLIND_PUSH]:
            if self.filtered_pose is not None:
                x, _, z, _ = self.filtered_pose
                self.get_logger().info(
                    f"[{self.state.value}] Dist: {z:.2f}m, X: {x:+.2f}m | Cmd: v={twist.linear.x:.2f}m/s, w={twist.angular.z:+.2f}rad/s",
                    throttle_duration_sec=0.5
                )

        self._publish_twist(twist)

        status_msg = Bool()
        status_msg.data = completed
        self.completed_pub.publish(status_msg)

    def _handle_search(self, twist: Twist):
        """Rotate slowly until marker is stably found."""
        if self.consecutive_detect_count >= self.min_consecutive_frames and self.filtered_pose is not None:
            self.get_logger().info("Marker acquired! Transitioning to ALIGN_BEARING.")
            self.state = DockingState.ALIGN_BEARING
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
        x, _, z, _ = self.filtered_pose
        bearing_error = math.atan2(x, z)

        if abs(bearing_error) <= 0.04:  # ~2.3 deg
            twist.angular.z = 0.0
            self.get_logger().info("Bearing centered! Transitioning to APPROACH_NORMAL.")
            self.state = DockingState.APPROACH_NORMAL
        else:
            ang_cmd = self.kp_angular * bearing_error
            twist.angular.z = self._clamp_angular(-ang_cmd)
            twist.linear.x = 0.0
            if abs(twist.angular.z) > 0.02:
                self.last_turn_dir = 1.0 if twist.angular.z > 0 else -1.0

    def _handle_approach_normal(self, twist: Twist):
        """
        Advance along line of sight while maintaining centering.
        Transitions to ALIGN_WALL or FINAL_APPROACH.
        """
        x, _, z, wall_yaw = self.filtered_pose
        dist_err = z - self.target_distance
        bearing_error = math.atan2(x, z)

        # When close to wall or well-aligned, check if wall alignment is needed
        if z <= max(self.vision_min_distance + 0.10, 0.25):
            if self.enable_wall_alignment and abs(wall_yaw) > self.yaw_tolerance:
                if abs(bearing_error) < 0.08:  # centered within ~4.5 deg
                    self.get_logger().info(
                        f"Near wall ({z:.2f}m). Wall tilt: {math.degrees(wall_yaw):.1f}°. Transitioning to ALIGN_WALL."
                    )
                    self.state = DockingState.ALIGN_WALL
                    return
                # If bearing not yet centered, keep centering in APPROACH_NORMAL
            else:
                self.get_logger().info(
                    f"Near wall ({z:.2f}m). Wall aligned (tilt: {math.degrees(wall_yaw):.1f}°). Advancing to FINAL_APPROACH."
                )
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
        """
        Rotate slowly in place so robot front bumper is parallel to marker wall.
        Includes FOV Guard: aborts alignment turn if marker approaches camera frame boundary.
        """
        x, _, z, wall_yaw = self.filtered_pose
        bearing_error = math.atan2(x, z)
        align_error = wall_yaw

        # FOV Guard: Camera half-FOV is ~26.7 deg. If bearing_error exceeds 0.16 rad (~9.2 deg),
        # prioritize keeping the marker in view rather than turning away.
        max_safe_bearing = 0.16
        if abs(bearing_error) > max_safe_bearing:
            self.get_logger().warn(
                f"[ALIGN_WALL FOV Guard] Marker bearing ({math.degrees(bearing_error):.1f}°) near FOV limit! "
                f"Pausing wall alignment to re-center."
            )
            self.state = DockingState.ALIGN_BEARING
            return

        if abs(align_error) <= self.yaw_tolerance:
            twist.angular.z = 0.0
            twist.linear.x = 0.0
            self.get_logger().info(
                f"Robot is PARALLEL to wall! (error={math.degrees(align_error):.1f}°). Advancing to FINAL_APPROACH."
            )
            self.state = DockingState.FINAL_APPROACH
        else:
            ang_cmd = self.kp_angular * 0.8 * align_error
            # Limit maximum angular speed during wall alignment to 0.15 rad/s to prevent blur & overshooting
            max_align_w = 0.15
            ang_val = max(min(-ang_cmd, max_align_w), -max_align_w)
            if self.invert_angular:
                ang_val = -ang_val
            twist.angular.z = ang_val
            twist.linear.x = 0.0
            if abs(twist.angular.z) > 0.02:
                self.last_turn_dir = 1.0 if twist.angular.z > 0 else -1.0

    def _handle_final_approach(self, twist: Twist):
        """
        Final straight-line approach directly into docking target.
        If target_distance is < vision_min_distance, hands off to BLIND_PUSH.
        """
        x, _, z, _ = self.filtered_pose
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
                    f"Reached vision limit ({z*100:.1f}cm). Switching to BLIND PUSH for final {push_dist*100:.1f}cm using Odometry!"
                )
                self.state = DockingState.BLIND_PUSH
                return

        # Check standard completion condition
        if abs(dist_err) <= self.distance_tolerance and abs(bearing_error) <= 0.05:
            self.get_logger().info(f"=== DOCKING COMPLETED! === Final Dist: {z:.3f}m, Err: {dist_err:+.3f}m")
            self.state = DockingState.DOCKED
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            return

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
        """Publish zero velocity to immediately stop the robot."""
        zero_twist = Twist()
        self.cmd_vel_pub.publish(zero_twist)


def main(args=None):
    rclpy.init(args=args)
    node = PrecisionDockingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down PrecisionDockingNode...")
    finally:
        node._stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
