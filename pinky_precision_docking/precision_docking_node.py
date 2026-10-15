#!/usr/bin/env python3
"""
Pinky Pro Precision Auto-Docking Node.
Uses ArUco marker pose estimation (OpenCV 4.6+ / 4.7+ compatible via solvePnP),
finite state machine (FSM) multi-stage alignment, exponential moving average (EMA)
jitter filtering, and an active safety watchdog.
"""

from enum import Enum
import math
import time
from typing import List, Optional, Tuple

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
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
    FINAL_APPROACH = "FINAL_APPROACH"
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

        # Filtered marker pose [x, y, z, yaw]
        self.filtered_pose = None  # [x, y, z, yaw]
        self.raw_pose = None

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

        # Control Loop Timer (20Hz)
        self.control_timer = self.create_timer(0.05, self._control_loop)

        # Dynamic parameter callback
        self.add_on_set_parameters_callback(self._on_parameter_update)

        self.get_logger().info(
            f"PrecisionDockingNode initialized! Target ID: {self.target_marker_id}, "
            f"Distance: {self.target_distance}m, Size: {self.marker_size}m"
        )

    def _declare_and_get_parameters(self):
        """Declare ROS 2 parameters with defaults and load them."""
        # Marker settings
        self.declare_parameter('marker_size', 0.06, ParameterDescriptor(description='Marker physical size (meters)'))
        self.declare_parameter('target_marker_id', 0, ParameterDescriptor(description='Target ArUco Marker ID'))
        self.declare_parameter('aruco_dictionary_id', 'DICT_4X4_50', ParameterDescriptor(description='ArUco dictionary name'))

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

        # Controller parameters
        self.declare_parameter('kp_linear', 0.5, ParameterDescriptor(description='Linear velocity proportional gain'))
        self.declare_parameter('kp_angular', 1.2, ParameterDescriptor(description='Angular velocity proportional gain'))
        self.declare_parameter('max_linear_vel', 0.10, ParameterDescriptor(description='Maximum linear velocity (m/s)'))
        self.declare_parameter('max_angular_vel', 0.35, ParameterDescriptor(description='Maximum angular velocity (rad/s)'))
        self.declare_parameter('min_linear_vel', 0.02, ParameterDescriptor(description='Minimum linear velocity to overcome friction'))
        self.declare_parameter('min_angular_vel', 0.05, ParameterDescriptor(description='Minimum angular velocity'))
        self.declare_parameter('invert_angular', False, ParameterDescriptor(description='Invert steering direction'))

        # Safety & filter
        self.declare_parameter('loss_timeout', 0.8, ParameterDescriptor(description='Timeout before treating marker as lost (s)'))
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
        dict_attr = getattr(cv2.aruco, self.aruco_dict_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dict_attr)
        
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.aruco_parameters = cv2.aruco.DetectorParameters_create()
        else:
            self.aruco_parameters = cv2.aruco.DetectorParameters()

        # Use ArucoDetector if available (OpenCV 4.7+)
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

        # Detect markers
        if self.aruco_detector is not None:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dictionary, parameters=self.aruco_parameters)

        target_detected = False
        target_rvec = None
        target_tvec = None

        if ids is not None and len(ids) > 0:
            for idx, marker_id in enumerate(ids.flatten()):
                # If target_marker_id is -1, automatically lock-on to first detected marker
                if self.target_marker_id == -1:
                    self.target_marker_id = int(marker_id)
                    self.get_logger().info(f"[Auto-Lock] Automatically locked onto detected Marker ID: {self.target_marker_id}")

                if marker_id == self.target_marker_id:
                    target_detected = True
                    target_corner = corners[idx]

                    # 3D object points of planar marker centered at origin
                    s = self.marker_size
                    obj_points = np.array([
                        [-s / 2.0,  s / 2.0, 0.0],
                        [ s / 2.0,  s / 2.0, 0.0],
                        [ s / 2.0, -s / 2.0, 0.0],
                        [-s / 2.0, -s / 2.0, 0.0]
                    ], dtype=np.float32)

                    # SolvePnP for robust 6D pose estimation across all OpenCV versions
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

            # Extract 3D translations
            x = float(target_tvec[0][0])
            y = float(target_tvec[1][0])
            z = float(target_tvec[2][0])

            # Compute marker relative yaw angle in horizontal X-Z plane (0.0 rad when facing straight)
            R, _ = cv2.Rodrigues(target_rvec)
            yaw = math.atan2(R[2, 0], R[0, 0])

            self.raw_pose = (x, y, z, yaw)

            # Exponential Moving Average (EMA) Jitter Filter
            if self.filtered_pose is None:
                self.filtered_pose = [x, y, z, yaw]
            else:
                a = self.ema_alpha
                self.filtered_pose[0] = a * x + (1.0 - a) * self.filtered_pose[0]
                self.filtered_pose[1] = a * y + (1.0 - a) * self.filtered_pose[1]
                self.filtered_pose[2] = a * z + (1.0 - a) * self.filtered_pose[2]
                self.filtered_pose[3] = a * yaw + (1.0 - a) * self.filtered_pose[3]

            # Draw visual axes and contours
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            cv2.drawFrameAxes(
                cv_image, self.camera_matrix, self.dist_coeffs,
                target_rvec, target_tvec, self.marker_size * 0.8
            )
        else:
            self.consecutive_detect_count = 0

        # Draw OSD (On-Screen Display) Debug Info
        self._draw_osd(cv_image)

        # Publish debug compressed image
        self._publish_debug_image(cv_image)

    def _draw_osd(self, image: np.ndarray):
        """Render informative On-Screen Display (OSD) overlay on camera feed."""
        h, w = image.shape[:2]

        # Top banner background
        cv2.rectangle(image, (0, 0), (w, 85), (20, 20, 20), -1)

        # State color badge
        state_color = {
            DockingState.SEARCH: (0, 165, 255),       # Orange
            DockingState.ALIGN_BEARING: (255, 200, 0),# Cyan/Yellow
            DockingState.APPROACH_NORMAL: (255, 120, 0),
            DockingState.FINAL_APPROACH: (0, 255, 255),# Yellow
            DockingState.DOCKED: (0, 255, 0),         # Green
            DockingState.LOST: (0, 0, 255)            # Red
        }.get(self.state, (200, 200, 200))

        # Title and State
        cv2.putText(image, "PINKY PRO AUTO-DOCKING", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(image, f"STATE: {self.state.value}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, state_color, 2)

        # Pose and Errors
        if self.filtered_pose is not None and self.state != DockingState.LOST:
            x, y, z, yaw = self.filtered_pose
            dist_err = z - self.target_distance
            pose_str = f"Dist: {z:.3f}m (Goal: {self.target_distance:.2f}m, Err: {dist_err:+.3f}m)"
            lat_str = f"LatErr(X): {x:+.3f}m | YawErr: {math.degrees(yaw):+.1f} deg"
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

        # Watchdog: Handle marker loss
        if self.state != DockingState.SEARCH and time_since_last_detection > self.loss_timeout:
            if self.state != DockingState.DOCKED:
                if self.state != DockingState.LOST:
                    self.get_logger().warn(f"Marker lost for > {self.loss_timeout}s! Emergency stop.")
                    self.last_state = self.state
                    self.state = DockingState.LOST
                self._stop_robot()
                return

        # FSM Transition & Action Logic
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

        elif self.state == DockingState.FINAL_APPROACH:
            self._handle_final_approach(twist)

        elif self.state == DockingState.DOCKED:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            completed = True

        if self.state in [DockingState.APPROACH_NORMAL, DockingState.FINAL_APPROACH] and self.filtered_pose is not None:
            x, _, z, _ = self.filtered_pose
            self.get_logger().info(
                f"[{self.state.value}] Dist: {z:.2f}m, X: {x:+.2f}m | Cmd: v={twist.linear.x:.2f}m/s, w={twist.angular.z:+.2f}rad/s",
                throttle_duration_sec=0.5
            )

        # Publish Twist command
        self._publish_twist(twist)

        # Publish completion status
        status_msg = Bool()
        status_msg.data = completed
        self.completed_pub.publish(status_msg)

    def _handle_search(self, twist: Twist):
        """Search state: rotate slowly in place until marker is stably found."""
        if self.consecutive_detect_count >= self.min_consecutive_frames and self.filtered_pose is not None:
            self.get_logger().info("Marker acquired! Transitioning to ALIGN_BEARING.")
            self.state = DockingState.ALIGN_BEARING
            twist.angular.z = 0.0
        else:
            # Rotate slowly to find marker
            twist.linear.x = 0.0
            twist.angular.z = -self.search_angular_vel if self.invert_angular else self.search_angular_vel

    def _handle_lost(self, twist: Twist, time_since_loss: float):
        """Lost state: stop immediately. Return to SEARCH if lost for extended period."""
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        if self.consecutive_detect_count >= self.min_consecutive_frames:
            self.get_logger().info("Marker re-acquired! Resuming ALIGN_BEARING.")
            self.state = DockingState.ALIGN_BEARING
        elif time_since_loss > 3.0:
            self.get_logger().info("Extended loss timeout. Resetting to SEARCH.")
            self.filtered_pose = None
            self.state = DockingState.SEARCH

    def _handle_align_bearing(self, twist: Twist):
        """Align robot body so marker is centered in the camera frame."""
        x, y, z, yaw = self.filtered_pose
        bearing_error = math.atan2(x, z)

        if abs(bearing_error) <= 0.04:  # ~2.3 degrees
            twist.angular.z = 0.0
            self.get_logger().info("Bearing aligned! Transitioning to APPROACH_NORMAL.")
            self.state = DockingState.APPROACH_NORMAL
        else:
            # Proportional steering to center marker in camera view
            ang_cmd = self.kp_angular * bearing_error
            twist.angular.z = self._clamp_angular(-ang_cmd)
            twist.linear.x = 0.0

    def _handle_approach_normal(self, twist: Twist):
        """
        Approach the marker along the line of sight.
        Uses visual bearing error to keep the marker strictly centered in camera.
        """
        x, y, z, yaw = self.filtered_pose
        dist_err = z - self.target_distance
        bearing_error = math.atan2(x, z)

        # Transition to FINAL_APPROACH when close to target or well-centered
        if abs(bearing_error) <= 0.05:
            self.state = DockingState.FINAL_APPROACH
            return

        # Pure bearing proportional steering (Robust & immune to marker orientation tilt)
        ang_cmd = self.kp_angular * bearing_error
        twist.angular.z = self._clamp_angular(-ang_cmd)

        # Move forward slowly while converging
        if dist_err > 0.03:
            lin_cmd = self.kp_linear * 0.7 * dist_err
            twist.linear.x = self._clamp_linear(lin_cmd)
        else:
            twist.linear.x = 0.0

    def _handle_final_approach(self, twist: Twist):
        """
        Final straight-line approach directly into docking station.
        Maintains marker centered in camera view while driving forward to target_distance.
        """
        x, y, z, yaw = self.filtered_pose
        dist_err = z - self.target_distance
        bearing_error = math.atan2(x, z)

        # Check completion condition
        if abs(dist_err) <= self.distance_tolerance and abs(bearing_error) <= 0.05:
            self.get_logger().info(
                f"=== DOCKING COMPLETED! === Final Dist: {z:.3f}m, Err: {dist_err:+.3f}m"
            )
            self.state = DockingState.DOCKED
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            return

        # Fine steering correction using bearing
        ang_cmd = self.kp_angular * 1.2 * bearing_error
        twist.angular.z = self._clamp_angular(-ang_cmd)

        # Forward/Backward linear velocity control
        lin_cmd = self.kp_linear * dist_err
        twist.linear.x = self._clamp_linear(lin_cmd)

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
        if abs(val) < 0.02:  # Deadband to keep straight trajectory without jitter
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
