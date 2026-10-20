"""
Camera-LiDAR complementary fusion estimator.

Key idea: the docking target (marker + wall) is STATIC in the world.  Instead of feeding raw,
noisy, intermittent sensor values to the controller, the estimator maintains a
"docking anchor" in the odometry frame and lets each sensor update only the component it is
good at:

    anchor = { theta_n : inward wall-normal direction in odom   <- LiDAR (primary)
                                                                   camera normal (weak prior)
               s       : wall/marker depth along the normal     <- LiDAR + camera (inv. var.)
               t       : marker position along the wall         <- camera ONLY }

Every component is a scalar Kalman filter (static state + odometry-drift process noise),
with Mahalanobis gating and automatic re-initialisation after persistent disagreement
(e.g. wheel slip, robot kidnapping).  Sensor measurements are time-aligned with the odometry
pose at the sensor timestamp (latency compensation).

The controller then reads, at the full control rate, the robot pose expressed in the docking
frame P (see geometry.py), computed from the CURRENT odometry pose and the anchor:
    d_base, y, psi  (+ d_ref : distance of the configured reference point to the wall)
This keeps the estimate smooth (no chattering), keeps working while the marker is briefly
out of view (turning, too close) and is exactly referenced to the wheel-axle rotation centre.
"""

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Deque, Optional, Tuple

from .geometry import interpolate_pose, marker_bearing_from_camera, Pose2D, wrap_angle
from .marker_detector import MarkerObservation
from .wall_fitting import WallObservation


@dataclass
class EstimatorParams:
    cam_lateral_sigma_k: float = 0.004   # sigma_lat = k * range + floor
    cam_lateral_sigma_floor: float = 0.0015
    cam_depth_sigma_k: float = 0.012     # sigma_depth = k * range + floor
    cam_depth_sigma_floor: float = 0.003
    cam_phi_sigma_deg: float = 6.0       # camera normal is weak (planar PnP ambiguity)
    cam_phi_sigma_ambiguous_deg: float = 20.0
    odom_trans_drift: float = 0.02       # [m / m travelled]
    odom_rot_drift: float = 0.03         # [rad / rad rotated]
    odom_heading_drift_per_m: float = 0.02  # [rad / m travelled]
    gate_sigma: float = 4.0
    gate_floor_pos: float = 0.015
    gate_floor_ang_deg: float = 2.5
    reset_after_rejects: int = 4
    marker_wall_offset: float = 0.0      # marker face protrusion from the wall plane [m]
    odom_history_sec: float = 3.0
    use_camera_distance: bool = True
    use_lidar_distance: bool = True
    use_lidar_angle: bool = True


class ScalarKF:

    def __init__(self, name: str, angular: bool = False):
        self.name = name
        self.angular = angular
        self.x: Optional[float] = None
        self.P = 1.0
        self.rejects = 0
        self.last_innovation = 0.0
        self.resets = 0

    def initialized(self) -> bool:
        return self.x is not None

    def reset(self, z: float, r: float):
        self.x = z
        self.P = r
        self.rejects = 0

    def predict(self, q: float):
        if self.x is not None:
            self.P += q

    def update(self, z: float, r: float, gate_sigma: float, gate_floor: float,
               reset_after: int) -> bool:
        if self.x is None:
            self.reset(z, r)
            return True
        innov = wrap_angle(z - self.x) if self.angular else z - self.x
        self.last_innovation = innov
        s = self.P + r
        if innov * innov > gate_sigma * gate_sigma * s and abs(innov) > gate_floor:
            self.rejects += 1
            if self.rejects >= reset_after:
                self.reset(z, r)
                self.resets += 1
                return True
            return False
        k = self.P / s
        self.x = self.x + k * innov
        if self.angular:
            self.x = wrap_angle(self.x)
        self.P = (1.0 - k) * self.P
        self.rejects = 0
        return True

    @property
    def sigma(self) -> float:
        return math.sqrt(max(self.P, 0.0))


@dataclass
class DockingEstimate:
    valid: bool = False
    d_base: float = float('nan')     # rotation centre -> wall plane [m]
    y: float = float('nan')          # lateral offset from the marker centreline [m] (+ left)
    psi: float = float('nan')        # heading error w.r.t. inward wall normal [rad] (+ CCW)
    d_ref: float = float('nan')      # reference point -> wall plane [m]
    sigma_d: float = float('inf')
    sigma_y: float = float('inf')
    sigma_psi: float = float('inf')
    cam_age: float = float('inf')
    lidar_age: float = float('inf')
    marker_bearing: float = float('nan')  # predicted camera bearing to the marker [rad]
    anchor: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (Mx, My, theta_n) in odom
    robot: Pose2D = field(default_factory=Pose2D)
    raw: dict = field(default_factory=dict)  # per-sensor values for debugging / logging


class DockingEstimator:

    def __init__(self, params: Optional[EstimatorParams] = None,
                 cam_xy: Tuple[float, float] = (0.035, 0.0),
                 ref_xy: Tuple[float, float] = (0.035, 0.0)):
        self.p = params or EstimatorParams()
        self.cam_xy = cam_xy
        self.ref_xy = ref_xy
        self.odom: Deque[Tuple[float, Pose2D]] = deque()
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.kf_theta = ScalarKF('theta_n', angular=True)
        self.kf_s = ScalarKF('s')
        self.kf_t = ScalarKF('t')
        self.last_cam_time = -math.inf
        self.last_lidar_time = -math.inf
        self.cam_updates = 0
        self.lidar_updates = 0
        self._last_pred_pose: Optional[Pose2D] = None
        self.raw = {}

    def initialized(self) -> bool:
        return self.kf_theta.initialized() and self.kf_s.initialized() and \
            self.kf_t.initialized()

    # ------------------------------------------------------------------
    # odometry
    # ------------------------------------------------------------------
    def add_odom(self, stamp: float, pose: Pose2D):
        if self.odom and stamp < self.odom[-1][0]:
            self.odom.clear()  # time jumped backwards (sim reset)
        self.odom.append((stamp, pose))
        while self.odom and (stamp - self.odom[0][0]) > self.p.odom_history_sec:
            self.odom.popleft()
        self._propagate(pose)

    def latest_pose(self) -> Optional[Pose2D]:
        return self.odom[-1][1] if self.odom else None

    def pose_at(self, stamp: float) -> Optional[Pose2D]:
        if not self.odom:
            return None
        if stamp >= self.odom[-1][0]:
            return self.odom[-1][1]
        if stamp <= self.odom[0][0]:
            return self.odom[0][1]
        prev = self.odom[0]
        for item in self.odom:
            if item[0] >= stamp:
                t0, p0 = prev
                t1, p1 = item
                r = 0.0 if t1 <= t0 else (stamp - t0) / (t1 - t0)
                return interpolate_pose(p0, p1, r)
            prev = item
        return self.odom[-1][1]

    def _propagate(self, pose: Pose2D):
        if self._last_pred_pose is None:
            self._last_pred_pose = pose
            return
        lp = self._last_pred_pose
        dist = math.hypot(pose.x - lp.x, pose.y - lp.y)
        drot = abs(wrap_angle(pose.theta - lp.theta))
        self._last_pred_pose = pose
        if dist < 1e-6 and drot < 1e-6:
            return
        p = self.p
        q_theta = (p.odom_rot_drift * drot) ** 2 + (p.odom_heading_drift_per_m * dist) ** 2
        q_pos = (p.odom_trans_drift * dist) ** 2
        # heading drift also displaces the anchor by lever arm x angle
        lever = 0.0
        if self.initialized():
            mx, my = self._marker_xy()
            lever = math.hypot(mx - pose.x, my - pose.y)
        q_lever = (lever * math.sqrt(q_theta)) ** 2
        self.kf_theta.predict(q_theta)
        self.kf_s.predict(q_pos + q_lever)
        self.kf_t.predict(q_pos + q_lever)

    # ------------------------------------------------------------------
    def _basis(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        th = self.kf_theta.x or 0.0
        return (math.cos(th), math.sin(th)), (-math.sin(th), math.cos(th))

    def _marker_xy(self) -> Tuple[float, float]:
        u, v = self._basis()
        s, t = self.kf_s.x or 0.0, self.kf_t.x or 0.0
        return s * u[0] + t * v[0], s * u[1] + t * v[1]

    def _rebase(self, new_theta: float):
        """Keep the marker POINT fixed while the normal direction changes."""
        if not (self.kf_s.initialized() and self.kf_t.initialized()):
            return
        mx, my = self._marker_xy()
        u = (math.cos(new_theta), math.sin(new_theta))
        v = (-math.sin(new_theta), math.cos(new_theta))
        self.kf_s.x = mx * u[0] + my * u[1]
        self.kf_t.x = mx * v[0] + my * v[1]

    def _update_theta(self, z: float, r: float) -> bool:
        old = self.kf_theta.x
        if old is not None:
            diff = abs(wrap_angle(z - old))
            if diff > math.radians(28.0):
                # Reject perpendicular/side walls: normal cannot deviate by >28 deg from anchor
                return False
        ok = self.kf_theta.update(z, r, self.p.gate_sigma,
                                  math.radians(self.p.gate_floor_ang_deg),
                                  self.p.reset_after_rejects)
        if old is not None and ok:
            # rebase about the old theta: recompute s,t with the new basis around fixed point
            new = self.kf_theta.x
            self.kf_theta.x = old
            self._rebase(new)
            self.kf_theta.x = new
        return ok

    # ------------------------------------------------------------------
    # sensor updates
    # ------------------------------------------------------------------
    def marker_point_in_base(self, pose: Optional[Pose2D] = None) -> Optional[Tuple[float, float]]:
        """Predicted marker position in the base frame (used as the LiDAR RANSAC hint)."""
        if not self.initialized():
            return None
        pose = pose or self.latest_pose()
        if pose is None:
            return None
        mx, my = self._marker_xy()
        return pose.inverse_transform_point(mx, my)

    def wall_phi_in_base(self, pose: Optional[Pose2D] = None) -> Optional[float]:
        if not self.kf_theta.initialized():
            return None
        pose = pose or self.latest_pose()
        if pose is None:
            return None
        return wrap_angle(self.kf_theta.x - pose.theta)

    def update_camera(self, obs: MarkerObservation) -> bool:
        pose = self.pose_at(obs.stamp)
        if pose is None:
            return False
        p = self.p
        mx, my = pose.transform_point(float(obs.p_base[0]), float(obs.p_base[1]))
        rng = math.hypot(obs.p_base[0] - self.cam_xy[0], obs.p_base[1] - self.cam_xy[1])
        sig_lat = p.cam_lateral_sigma_k * rng + p.cam_lateral_sigma_floor
        sig_dep = p.cam_depth_sigma_k * rng + p.cam_depth_sigma_floor
        # weaken measurements of a marker clipped by / very close to the image border
        if obs.edge_margin_px < 4.0:
            sig_lat *= 3.0
            sig_dep *= 3.0
        theta_cam = wrap_angle(pose.theta + obs.phi)
        sig_phi = math.radians(p.cam_phi_sigma_deg if obs.ambiguity_ratio > 3.0
                               else p.cam_phi_sigma_ambiguous_deg)

        if not self.kf_theta.initialized():
            self.kf_theta.reset(theta_cam, sig_phi ** 2)
        elif not self.p.use_lidar_angle or (obs.stamp - self.last_lidar_time) > 1.0:
            # camera normal is only used as a weak update when LiDAR is unavailable
            self._update_theta(theta_cam, sig_phi ** 2)

        u, v = self._basis()
        los = math.atan2(my - pose.y, mx - pose.x)
        a = los - self.kf_theta.x
        var_u = (sig_dep * math.cos(a)) ** 2 + (sig_lat * math.sin(a)) ** 2
        var_v = (sig_dep * math.sin(a)) ** 2 + (sig_lat * math.cos(a)) ** 2
        s_obs = mx * u[0] + my * u[1]
        t_obs = mx * v[0] + my * v[1]
        ok_t = self.kf_t.update(t_obs, var_v, p.gate_sigma, p.gate_floor_pos,
                                p.reset_after_rejects)
        if p.use_camera_distance or not self.kf_s.initialized():
            self.kf_s.update(s_obs, var_u, p.gate_sigma, p.gate_floor_pos,
                             p.reset_after_rejects)
        self.last_cam_time = obs.stamp
        self.cam_updates += 1

        # raw camera-only docking quantities (for debugging / logging)
        rx, ry = pose.x - mx, pose.y - my
        self.raw['cam_d'] = -(rx * u[0] + ry * u[1])
        self.raw['cam_y'] = rx * v[0] + ry * v[1]
        self.raw['cam_psi'] = -obs.phi
        self.raw['cam_range'] = obs.range_cam
        self.raw['cam_bearing'] = obs.bearing
        self.raw['cam_xb'] = float(obs.p_base[0])
        self.raw['cam_yb'] = float(obs.p_base[1])
        return ok_t

    def update_lidar(self, obs: WallObservation) -> bool:
        pose = self.pose_at(obs.stamp)
        if pose is None:
            return False
        p = self.p
        theta_l = wrap_angle(pose.theta + obs.phi)
        # Guard: if the wall normal deviates excessively from known docking wall anchor, reject
        if self.kf_theta.initialized():
            ang_diff = abs(wrap_angle(theta_l - self.kf_theta.x))
            if ang_diff > math.radians(25.0):
                return False

        if p.use_lidar_angle:
            if not self.kf_theta.initialized():
                self.kf_theta.reset(theta_l, obs.sigma_phi ** 2)
            else:
                self._update_theta(theta_l, obs.sigma_phi ** 2)
        if p.use_lidar_distance and self.kf_theta.initialized():
            u, _ = self._basis()
            cx = pose.x + obs.distance * math.cos(theta_l)
            cy = pose.y + obs.distance * math.sin(theta_l)
            s_wall = cx * u[0] + cy * u[1] - p.marker_wall_offset
            if self.kf_s.initialized():
                self.kf_s.update(s_wall, obs.sigma_distance ** 2, p.gate_sigma,
                                 p.gate_floor_pos, p.reset_after_rejects)
        self.last_lidar_time = obs.stamp
        self.lidar_updates += 1
        self.raw['lidar_d'] = obs.distance
        self.raw['lidar_psi'] = -obs.phi
        self.raw['lidar_rms'] = obs.rms
        self.raw['lidar_n'] = obs.n_inliers
        self.raw['lidar_span'] = obs.span
        return True

    # ------------------------------------------------------------------
    def estimate(self, now: float, pose: Optional[Pose2D] = None) -> DockingEstimate:
        pose = pose or self.latest_pose()
        est = DockingEstimate(raw=dict(self.raw))
        est.cam_age = now - self.last_cam_time
        est.lidar_age = now - self.last_lidar_time
        if pose is None or not self.initialized():
            return est
        u, v = self._basis()
        mx, my = self._marker_xy()
        rx, ry = pose.x - mx, pose.y - my
        x_p = rx * u[0] + ry * u[1]
        y_p = rx * v[0] + ry * v[1]
        psi = wrap_angle(pose.theta - self.kf_theta.x)
        refx, refy = pose.transform_point(*self.ref_xy)
        d_ref = -((refx - mx) * u[0] + (refy - my) * u[1])
        lever = math.hypot(rx, ry)
        est.valid = True
        est.d_base = -x_p
        est.y = y_p
        est.psi = psi
        est.d_ref = d_ref
        est.sigma_psi = self.kf_theta.sigma
        est.sigma_d = self.kf_s.sigma
        est.sigma_y = math.hypot(self.kf_t.sigma, lever * self.kf_theta.sigma)
        est.marker_bearing = marker_bearing_from_camera(est.d_base, est.y, est.psi,
                                                        self.cam_xy[0], self.cam_xy[1])
        est.anchor = (mx, my, self.kf_theta.x)
        est.robot = pose
        return est
