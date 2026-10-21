#!/usr/bin/env python3
"""
Pinky Pro Precision Auto-Docking Node.

Architecture:
  - Branch A (Camera): ArUco (DICT_5X5, 6cm) + IPPE_SQUARE + LiDAR-guided ambiguity resolution.
  - Branch B (LiDAR): Front 2D LaserScan + Camera-guided RANSAC & TLS line fitting.
  - Fusion Estimator: World-fixed static docking anchor in odom frame with Kalman filtering.
  - Staged Controller:
      1. Pre-docking Intercept (FOV-preserving deviated pursuit to centerline)
      2. Closed-loop Verification & Fine Adjustment (with recovery retreat guard)
      3. Terminal Straight Approach (parallel 11-shape entry to 10cm, <1cm dist, <1.5cm lat, <2 deg yaw)
"""

import csv
from datetime import datetime
import math
import os
import signal
import sys
import time
from typing import List, Optional, Tuple

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, LaserScan
from std_msgs.msg import Bool, Int32
from std_srvs.srv import Trigger

from .controller import ControllerParams, DockingController, Phase
from .estimator import DockingEstimate, DockingEstimator, EstimatorParams
from .geometry import Pose2D, wrap_angle
from .marker_detector import ArucoMarkerDetector, CameraExtrinsics, MarkerObservation
from .sound_effects import play_docking_complete
from .wall_fitting import LidarExtrinsics, WallFitParams, WallFitter, WallObservation


class PrecisionDockingNode(Node):
    """High-precision Camera-LiDAR fusion auto-docking node for differential drive robots."""

    def __init__(self):
        super().__init__('precision_docking_node')
        self.bridge = CvBridge()

        self._declare_and_get_parameters()

        # Instantiate modular components
        self.cam_ext = CameraExtrinsics(
            x=self.camera_z_offset,   # forward lever arm from wheel axle (m)
            y=self.camera_x_offset,
            z=0.0654,
            pitch=math.radians(self.cam_tilt_deg)
        )
        self.detector = ArucoMarkerDetector(
            dictionary=self.aruco_dict_name,
            marker_size=self.marker_size,
            extrinsics=self.cam_ext
        )

        self.lidar_ext = LidarExtrinsics(
            x=-0.017,  # URDF default rplidar_mount
            y=0.0,
            yaw=math.pi
        )
        self.wall_fitter = WallFitter(
            params=WallFitParams(
                fov_half_deg=self.lidar_roi_angle_deg,
                min_range=self.lidar_min_range,
                max_range=self.lidar_max_range
            ),
            extrinsics=self.lidar_ext
        )

        self.estimator = DockingEstimator(
            params=EstimatorParams(),
            cam_xy=(self.cam_ext.x, self.cam_ext.y),
            ref_xy=(self.cam_ext.x, self.cam_ext.y)
        )

        self.ctrl_params = ControllerParams(
            target_distance=self.target_distance,
            distance_tolerance=self.distance_tolerance,
            center_tolerance=self.center_tolerance,
            angle_tolerance_deg=self.angle_tolerance_deg,
            target_wall_angle_deg=self.target_wall_angle_deg,
            wall_align_tol_deg=self.wall_align_tol_deg,
            cam_x=self.cam_ext.x,
            ref_x=self.cam_ext.x,
            v_max=self.max_linear_vel,
            v_min=self.min_linear_vel,
            w_max=self.max_angular_vel,
            w_min=self.min_angular_vel,
            w_search=self.search_angular_vel,
            staging_distance=0.30,
            min_cross_distance=0.12,
            allow_retreat=self.allow_retreat,
            enable_continuous_curve=self.enable_continuous_curve
        )
        self.controller = DockingController(self.ctrl_params)

        # Internal state
        self.docking_logged = False
        self.last_logged_phase = Phase.IDLE
        self.last_img_stamp = 0.0
        self.last_scan_stamp = 0.0
        self.last_compressed_rx_time = 0.0
        self.last_rendered_debug: Optional[np.ndarray] = None
        self.last_est: Optional[DockingEstimate] = None
        self.last_marker_obs: Optional[MarkerObservation] = None
        self.last_wall_obs: Optional[WallObservation] = None

        # QoS
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
        self.debug_raw_pub = self.create_publisher(Image, '/docking/debug_image', reliable_qos)
        self.debug_image_pub = self.create_publisher(CompressedImage, '/docking/debug_image/compressed', reliable_qos)

        # Subscribers
        compressed_topic = self.image_topic if 'compressed' in self.image_topic else f'{self.image_topic}/compressed'
        raw_topic = self.image_topic.replace('/compressed', '')

        if self.use_compressed:
            self.get_logger().info(f'[Branch A] Subscribing to CompressedImage: {compressed_topic} with raw fallback {raw_topic}')
            self.image_sub = self.create_subscription(CompressedImage, compressed_topic, self._compressed_image_callback, sensor_qos)
            self.raw_fallback_sub = self.create_subscription(Image, raw_topic, self._raw_fallback_image_callback, sensor_qos)
        else:
            self.get_logger().info(f'[Branch A] Subscribing to raw Image: {raw_topic}')
            self.image_sub = self.create_subscription(Image, raw_topic, self._image_callback, sensor_qos)

        self.info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_callback, sensor_qos)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, sensor_qos)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self._odom_callback, reliable_qos)
        self.target_id_sub = self.create_subscription(Int32, '/docking/target_marker_id', self._target_marker_id_cb, reliable_qos)

        # Services
        self.start_srv = self.create_service(Trigger, self.trigger_service_name, self._handle_start_trigger)
        self.cancel_srv = self.create_service(Trigger, self.cancel_service_name, self._handle_cancel_trigger)

        # 20 Hz Control Timer
        self.control_timer = self.create_timer(0.05, self._control_loop)
        self.add_on_set_parameters_callback(self._on_parameter_update)

        if not self.start_in_idle:
            now = self._get_time_sec()
            self.controller.start(now)
            self.get_logger().info('PrecisionDockingNode started immediately in SEARCH state.')
        else:
            self.get_logger().info('PrecisionDockingNode initialized in IDLE state. Waiting for /docking/start.')

    def _get_time_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _declare_and_get_parameters(self):
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False, ParameterDescriptor(description='Use simulation clock from /clock'))
        self.declare_parameter('start_in_idle', True, ParameterDescriptor(description='Start in IDLE waiting for /docking/start'))
        self.declare_parameter('trigger_service_name', '/docking/start')
        self.declare_parameter('cancel_service_name', '/docking/cancel')

        self.declare_parameter('marker_size', 0.06)
        self.declare_parameter('target_marker_id', -1)
        self.declare_parameter('targetmarkerid', -1)
        self.declare_parameter('marker_id', -1)
        self.declare_parameter('target_id', -1)
        self.declare_parameter('aruco_dictionary_id', 'DICT_5X5_250')
        self.declare_parameter('use_compressed', True)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('camera_x_offset', 0.0)
        self.declare_parameter('camera_z_offset', 0.035)
        self.declare_parameter('cam_tilt_deg', 0.0)

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('lidar_roi_angle_deg', 70.0)
        self.declare_parameter('lidar_min_range', 0.03)
        self.declare_parameter('lidar_max_range', 1.50)

        self.declare_parameter('target_distance', 0.10)
        self.declare_parameter('distance_tolerance', 0.01)
        self.declare_parameter('center_tolerance', 0.005)
        self.declare_parameter('angle_tolerance_deg', 2.0)
        self.declare_parameter('target_wall_angle_deg', 0.0)
        self.declare_parameter('wall_align_tol_deg', 0.5)

        self.declare_parameter('max_linear_vel', 0.08)
        self.declare_parameter('min_linear_vel', 0.008)
        self.declare_parameter('max_angular_vel', 0.50)
        self.declare_parameter('min_angular_vel', 0.04)
        self.declare_parameter('search_angular_vel', 0.30)
        self.declare_parameter('allow_retreat', True)
        self.declare_parameter('enable_continuous_curve', True)
        self.declare_parameter('osd_bg_alpha', 0.35)
        self.declare_parameter('csv_log_path', '/home/user/pinky_pro/docking_experiment_log.csv')

        self.start_in_idle = bool(self.get_parameter('start_in_idle').value)
        self.enable_continuous_curve = bool(self.get_parameter('enable_continuous_curve').value)
        self.trigger_service_name = str(self.get_parameter('trigger_service_name').value)
        self.cancel_service_name = str(self.get_parameter('cancel_service_name').value)

        self.marker_size = float(self.get_parameter('marker_size').value)
        self.target_marker_id = int(self.get_parameter('target_marker_id').value)
        for alias in ('targetmarkerid', 'marker_id', 'target_id'):
            val = int(self.get_parameter(alias).value)
            if self.target_marker_id == -1 and val != -1:
                self.target_marker_id = val

        self.aruco_dict_name = str(self.get_parameter('aruco_dictionary_id').value)
        self.use_compressed = bool(self.get_parameter('use_compressed').value)
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.camera_x_offset = float(self.get_parameter('camera_x_offset').value)
        self.camera_z_offset = float(self.get_parameter('camera_z_offset').value)
        self.cam_tilt_deg = float(self.get_parameter('cam_tilt_deg').value)

        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.lidar_roi_angle_deg = float(self.get_parameter('lidar_roi_angle_deg').value)
        self.lidar_min_range = float(self.get_parameter('lidar_min_range').value)
        self.lidar_max_range = float(self.get_parameter('lidar_max_range').value)

        self.target_distance = float(self.get_parameter('target_distance').value)
        self.distance_tolerance = float(self.get_parameter('distance_tolerance').value)
        self.center_tolerance = float(self.get_parameter('center_tolerance').value)
        self.angle_tolerance_deg = float(self.get_parameter('angle_tolerance_deg').value)
        self.target_wall_angle_deg = float(self.get_parameter('target_wall_angle_deg').value)
        self.wall_align_tol_deg = float(self.get_parameter('wall_align_tol_deg').value)

        self.max_linear_vel = float(self.get_parameter('max_linear_vel').value)
        self.min_linear_vel = float(self.get_parameter('min_linear_vel').value)
        self.max_angular_vel = float(self.get_parameter('max_angular_vel').value)
        self.min_angular_vel = float(self.get_parameter('min_angular_vel').value)
        self.search_angular_vel = float(self.get_parameter('search_angular_vel').value)
        self.allow_retreat = bool(self.get_parameter('allow_retreat').value)
        self.osd_bg_alpha = float(self.get_parameter('osd_bg_alpha').value)
        self.csv_log_path = str(self.get_parameter('csv_log_path').value)

    def _on_parameter_update(self, params: List[Parameter]) -> SetParametersResult:
        for p in params:
            if hasattr(self, p.name):
                setattr(self, p.name, p.value)
                if p.name == 'lidar_roi_angle_deg':
                    self.wall_fitter.p.fov_half_deg = float(p.value)
                if hasattr(self.ctrl_params, p.name):
                    setattr(self.ctrl_params, p.name, p.value)
                self.get_logger().info(f"Updated parameter '{p.name}' = {p.value}")
        return SetParametersResult(successful=True)

    # -------------------------------------------------------------------------
    # ROS 2 Callbacks
    # -------------------------------------------------------------------------
    def _handle_start_trigger(self, req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        now = self._get_time_sec()
        self.docking_logged = False
        self.estimator.reset()
        self.last_marker_obs = None
        self.last_wall_obs = None
        self.controller.start(now)
        resp.success = True
        resp.message = f'Docking started in {self.controller.phase.value} (Target ID: {self.target_marker_id})'
        self.get_logger().info(resp.message)
        return resp

    def _handle_cancel_trigger(self, req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        now = self._get_time_sec()
        self.controller.cancel(now)
        self._stop_robot()
        resp.success = True
        resp.message = 'Docking cancelled'
        self.get_logger().info(resp.message)
        return resp

    def _target_marker_id_cb(self, msg: Int32):
        self.target_marker_id = int(msg.data)
        self.get_logger().info(f'[Target ID] Updated target_marker_id to {self.target_marker_id} via topic.')

    def _camera_info_callback(self, msg: CameraInfo):
        if not self.detector.intrinsics_from_info and len(msg.k) == 9:
            K = np.array(msg.k, dtype=np.float64).reshape((3, 3))
            if K[0, 0] > 1.0:
                D = np.array(msg.d, dtype=np.float64) if len(msg.d) >= 4 else np.zeros(5)
                self.detector.set_intrinsics(K, D, (msg.width, msg.height), from_info=True)
                self.get_logger().info(f'[Branch A] CameraInfo received: fx={K[0,0]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}')

    def _compressed_image_callback(self, msg: CompressedImage):
        try:
            now = self._get_time_sec()
            self.last_compressed_rx_time = now
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is not None:
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                if stamp <= 0.0 or abs(now - stamp) > 3.0:
                    stamp = now
                self._process_image(cv_image, stamp)
        except Exception as exc:
            self.get_logger().error(f'[Branch A] Decode error: {exc}', throttle_duration_sec=2.0)

    def _raw_fallback_image_callback(self, msg: Image):
        if (self._get_time_sec() - self.last_compressed_rx_time) < 1.0:
            return
        self._image_callback(msg)

    def _image_callback(self, msg: Image):
        try:
            now = self._get_time_sec()
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if stamp <= 0.0 or abs(now - stamp) > 3.0:
                stamp = now
            self._process_image(cv_image, stamp)
        except Exception as exc:
            self.get_logger().error(f'[Branch A] Image error: {exc}', throttle_duration_sec=2.0)

    def _process_image(self, cv_image: np.ndarray, stamp: float):
        h, w = cv_image.shape[:2]
        if self.detector.K is None:
            # Fallback pinhole FOV ~ 66 deg (1.15 rad in URDF)
            self.detector.set_intrinsics_from_hfov(w, h, 1.1519)

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        detected = self.detector.detect(gray)

        chosen_id = None
        chosen_corners = None
        if detected:
            if self.target_marker_id == -1:
                # auto-select marker closest to image horizontal center
                cx = w / 2.0
                best_dist = float('inf')
                for mid, c in detected:
                    dist = abs(float(np.mean(c[:, 0])) - cx)
                    if dist < best_dist:
                        best_dist = dist
                        chosen_id = mid
                        chosen_corners = c
                self.target_marker_id = chosen_id
                self.get_logger().info(f'[Branch A] Auto-locked onto ArUco ID {chosen_id}')
            else:
                for mid, c in detected:
                    if mid == self.target_marker_id:
                        chosen_id = mid
                        chosen_corners = c
                        break

        phi_hint = self.estimator.wall_phi_in_base()
        marker_obs = None
        if chosen_corners is not None and chosen_id is not None:
            marker_obs = self.detector.estimate(chosen_id, chosen_corners, stamp, phi_hint=phi_hint)
            if marker_obs is not None:
                self.estimator.update_camera(marker_obs)
                self.last_marker_obs = marker_obs

        self._render_debug_osd(cv_image, detected, marker_obs)
        self._publish_debug(cv_image)

    def _scan_callback(self, msg: LaserScan):
        now = self._get_time_sec()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0.0 or abs(now - stamp) > 3.0:
            stamp = now
        pts = self.wall_fitter.scan_to_points(
            ranges=msg.ranges,
            angle_min=msg.angle_min,
            angle_increment=msg.angle_increment,
            range_min=msg.range_min,
            range_max=msg.range_max
        )
        if len(pts) >= 12:
            hint_pt = self.estimator.marker_point_in_base()
            phi_hint = self.estimator.wall_phi_in_base()
            hint_arr = np.array(hint_pt) if hint_pt is not None else None
            wall_obs = self.wall_fitter.fit(pts, stamp=stamp, hint_point=hint_arr, phi_hint=phi_hint)
            if wall_obs is not None:
                self.estimator.update_lidar(wall_obs)
                self.last_wall_obs = wall_obs

    def _odom_callback(self, msg: Odometry):
        now = self._get_time_sec()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0.0 or abs(now - stamp) > 3.0:
            stamp = now
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.estimator.add_odom(stamp, Pose2D(pos.x, pos.y, yaw))

    # -------------------------------------------------------------------------
    # 20 Hz Control Loop
    # -------------------------------------------------------------------------
    def _control_loop(self):
        now = self._get_time_sec()
        est = self.estimator.estimate(now)
        self.last_est = est

        out = self.controller.step(
            now=now,
            est=est,
            cam_updates=self.estimator.cam_updates,
            lidar_updates=self.estimator.lidar_updates
        )

        if self.last_logged_phase != out.phase:
            self.get_logger().info(f"[Phase] {self.last_logged_phase.value} -> {out.phase.value} ({self.controller.reason})")
            self.last_logged_phase = out.phase

        # Only publish velocity commands when actively docking (never interfere with Nav2 in IDLE)
        if out.phase != Phase.IDLE:
            twist = Twist()
            twist.linear.x = float(out.v)
            twist.angular.z = float(out.w)
            self._publish_twist(twist)

        # Completion Status
        is_docked = (out.phase == Phase.DOCKED)
        comp_msg = Bool()
        comp_msg.data = is_docked
        self.completed_pub.publish(comp_msg)

        if is_docked and not self.docking_logged and self.controller.result is not None:
            self._log_docking_success(now, est, self.controller.result)
            self._stop_robot()
            self.controller.cancel(now)
            self.last_logged_phase = Phase.IDLE

    def _log_docking_success(self, now: float, est: DockingEstimate, res: dict):
        self.docking_logged = True
        try:
            play_docking_complete(blocking=False)
        except Exception:
            pass
        elapsed = res.get('elapsed', 0.0)
        d_err = res.get('distance_error', 0.0)
        y_err = res.get('lateral_error', 0.0)
        ang_err = res.get('angle_error_deg', 0.0)

        raw_cam_d = est.raw.get('cam_d', float('nan'))
        raw_lid_d = est.raw.get('lidar_d', float('nan'))

        self.get_logger().info(
            f"\n====================================================================\n"
            f"          [AUTODOCKING COMPLETED] PRECISION TOLERANCES MET!          \n"
            f"====================================================================\n"
            f" - Target Distance (d_target) : {self.target_distance:.4f} m\n"
            f" - Final Docked Dist (d_ref)  : {est.d_ref:.4f} m (Error: {d_err*100:+.2f} cm)\n"
            f" - Final Lateral Error (y)    : {y_err*100:+.2f} cm (Tol: +/-{self.center_tolerance*100:.1f} cm)\n"
            f" - Final Angular Error (psi)  : {ang_err:+.2f} deg (Tol: +/-{self.angle_tolerance_deg:.1f} deg)\n"
            f" - Camera Distance Raw        : {raw_cam_d:.4f} m\n"
            f" - LiDAR Distance Raw         : {raw_lid_d:.4f} m\n"
            f" - Elapsed Time               : {elapsed:.2f} s\n"
            f" - Retries / Intercepts       : {res.get('retries', 0)} / {res.get('intercepts', 0)}\n"
            f"===================================================================="
        )

        try:
            os.makedirs(os.path.dirname(self.csv_log_path), exist_ok=True)
            exists = os.path.isfile(self.csv_log_path)
            with open(self.csv_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not exists:
                    writer.writerow([
                        'timestamp', 'target_marker_id', 'cam_distance_m', 'lidar_distance_m',
                        'fused_distance_m', 'lateral_error_m', 'angular_error_deg', 'elapsed_time_s'
                    ])
                writer.writerow([
                    datetime.now().isoformat(timespec='seconds'),
                    self.target_marker_id,
                    f"{raw_cam_d:.4f}",
                    f"{raw_lid_d:.4f}",
                    f"{est.d_ref:.4f}",
                    f"{y_err:.4f}",
                    f"{ang_err:.3f}",
                    f"{elapsed:.2f}"
                ])
            self.get_logger().info(f"Saved docking metric to CSV: {self.csv_log_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to write CSV: {e}")

    # -------------------------------------------------------------------------
    # Debug OSD Visualization
    # -------------------------------------------------------------------------
    def _render_debug_osd(self, img: np.ndarray, detected, marker_obs: Optional[MarkerObservation]):
        h, w = img.shape[:2]
        banner_h = 98

        # Semi-transparent dark background for OSD (allows camera view to show through clearly)
        alpha = float(getattr(self, 'osd_bg_alpha', 0.35))
        if alpha > 0.0:
            roi = img[0:banner_h, 0:w]
            dark_mask = np.full_like(roi, 15)  # dark background tint
            cv2.addWeighted(dark_mask, alpha, roi, 1.0 - alpha, 0, dst=roi)

        def put_text(text: str, org: Tuple[int, int], scale: float, col: Tuple[int, int, int], thick: int = 1):
            # Crisp black outline/shadow for readability over transparent camera feed
            cv2.putText(img, text, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)

        # Crosshair at image optical center
        cx, cy = w // 2, h // 2
        cv2.line(img, (cx, 0), (cx, h), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 6, (0, 255, 255), 1, cv2.LINE_AA)

        # Phase color
        ph = self.controller.phase
        phase_colors = {
            Phase.IDLE: (180, 180, 180),
            Phase.SEARCH: (0, 165, 255),
            Phase.ACQUIRE: (255, 200, 0),
            Phase.VERIFY: (255, 255, 0),
            Phase.INTERCEPT_TURN: (255, 120, 0),
            Phase.INTERCEPT_DRIVE: (255, 80, 0),
            Phase.AXIS_ALIGN: (200, 180, 0),
            Phase.TERMINAL: (0, 255, 120),
            Phase.FINAL_ALIGN: (0, 255, 220),
            Phase.WALL_ALIGN: (0, 255, 255),
            Phase.RETREAT: (0, 0, 255),
            Phase.REACQUIRE: (0, 100, 255),
            Phase.DOCKED: (0, 255, 0),
            Phase.FAILED: (0, 0, 180)
        }
        color = phase_colors.get(ph, (255, 255, 255))

        # Line 1: Phase & Reason
        reason_txt = f" ({self.controller.reason[:32]})" if self.controller.reason else ""
        put_text(f"PHASE: {ph.value}{reason_txt}", (12, 22), 0.52, color, 2)
        put_text(f"ID: {self.target_marker_id} | Goal: {self.target_distance:.2f}m", (max(w - 240, 360), 22),
                 0.48, (220, 220, 220), 1)

        # Line 2: Camera Branch (solvePnP pose)
        if marker_obs is not None:
            c_txt = f"[Cam ArUco] range: {marker_obs.range_cam:.3f}m | bear: {math.degrees(marker_obs.bearing):+.1f}d | phi: {math.degrees(marker_obs.phi):+.1f}d | reproj: {marker_obs.reproj_err:.2f}px"
            c_color = (120, 255, 120)
            cv2.aruco.drawDetectedMarkers(img, [marker_obs.corners.astype(np.float32).reshape(1, 4, 2)],
                                          np.array([[marker_obs.marker_id]], dtype=np.int32))
            cv2.drawFrameAxes(img, self.detector.K, self.detector.D, marker_obs.rvec, marker_obs.tvec, self.marker_size * 0.75)
        else:
            c_txt = "[Cam ArUco] Waiting for ArUco 5x5 detection..."
            c_color = (150, 150, 150)
        put_text(c_txt, (12, 46), 0.44, c_color, 1)

        # Line 3: LiDAR Branch
        if self.last_wall_obs is not None and (self._get_time_sec() - self.last_wall_obs.stamp) < 1.0:
            wobs = self.last_wall_obs
            l_txt = f"[2D LiDAR ] wall_d: {wobs.distance:.3f}m | phi: {math.degrees(wobs.phi):+.1f}d | rms: {wobs.rms*1000:.1f}mm | inliers: {wobs.n_inliers} | span: {wobs.span*100:.1f}cm"
            l_color = (255, 220, 80)
        else:
            l_txt = "[2D LiDAR ] Waiting for scan wall points..."
            l_color = (150, 150, 150)
        put_text(l_txt, (12, 68), 0.44, l_color, 1)

        # Line 4: Fusion & Tolerances
        if self.last_est is not None and self.last_est.valid:
            e = self.last_est
            rem = e.d_ref - self.target_distance
            f_txt = f"[Fused Est] d_ref: {e.d_ref:.3f}m (rem: {rem*100:+.2f}cm) | y: {e.y*100:+.2f}cm | psi: {math.degrees(e.psi):+.1f}d | sig:(d={e.sigma_d*1000:.1f}mm, y={e.sigma_y*1000:.1f}mm)"
            f_color = (0, 255, 255)
        else:
            f_txt = "[Fused Est] Estimator initializing anchor..."
            f_color = (180, 180, 180)
        put_text(f_txt, (12, 90), 0.44, f_color, 1)

    def _publish_debug(self, img: np.ndarray):
        stamp = self.get_clock().now().to_msg()
        try:
            raw_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            raw_msg.header.stamp = stamp
            self.debug_raw_pub.publish(raw_msg)
        except Exception:
            pass

        if self.debug_image_pub.get_subscription_count() > 0:
            success, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if success:
                cmsg = CompressedImage()
                cmsg.header.stamp = stamp
                cmsg.format = 'jpeg'
                cmsg.data = enc.tobytes()
                self.debug_image_pub.publish(cmsg)

    def _publish_twist(self, twist: Twist):
        self.cmd_vel_pub.publish(twist)

    def _stop_robot(self):
        if not rclpy.ok():
            return
        z = Twist()
        for _ in range(5):
            try:
                self.cmd_vel_pub.publish(z)
                time.sleep(0.01)
            except Exception:
                break


def main(args=None):
    rclpy.init(args=args)
    node = PrecisionDockingNode()

    def sig_handler(sig, frame):
        node.get_logger().info('Stopping robot and shutting down...')
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
    except (KeyboardInterrupt, SystemExit, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node._stop_robot()
        if rclpy.ok():
            try:
                node.destroy_node()
            except Exception:
                pass
            rclpy.shutdown()


if __name__ == '__main__':
    main()
