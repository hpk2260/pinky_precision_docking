"""
[LiDAR branch] Wall-plane extraction from a 2D LaserScan.

The docking wall is modelled as an infinite line in the base frame:
        n . p = d          (|n| = 1, n points from the robot INTO the wall, d > 0)
  * d   : perpendicular distance from the robot rotation centre to the wall plane
  * phi : atan2(n_y, n_x) = direction of the inward wall normal in the base frame
          -> heading error psi_lidar = -phi

Robust pipeline:
  1. LaserScan -> base-frame points (extrinsics: lidar mount x/y/yaw, Pinky: yaw = pi)
  2. forward-sector / range gating
  3. RANSAC line search (camera-guided: the line must pass close to the marker position
     reported by the camera, so neighbouring walls / furniture are never fitted)
  4. Total-least-squares (PCA) refinement on the inliers, iterated twice
  5. Quality metrics (rms, span, inlier count) -> measurement variances for fusion
"""

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np


@dataclass
class LidarExtrinsics:
    x: float = -0.017   # rplidar mount position in base frame (Pinky URDF)
    y: float = 0.0
    yaw: float = math.pi  # rplidar_link is rotated 180 deg about z on Pinky


@dataclass
class WallFitParams:
    fov_half_deg: float = 65.0
    min_range: float = 0.03
    max_range: float = 1.5
    ransac_iterations: int = 120
    inlier_threshold: float = 0.008
    min_inliers: int = 10
    min_span: float = 0.05            # reduced from 0.08 to 0.05m for ultra-close docking robustness
    hint_max_offset: float = 0.12     # increased from 0.05 to 0.12m for odometry drift tolerance
    hint_window: float = 0.50         # only use wall points within +/- this of the marker
    range_sigma: float = 0.002        # per-point noise (sim 2 mm, RPLIDAR C1 ~ 10-15 mm)
    distance_bias_sigma: float = 0.0015
    angle_bias_sigma_deg: float = 0.15
    distance_bias: float = 0.0        # calibration: subtract from the fitted distance
    angle_bias_deg: float = 0.0       # calibration: subtract from the fitted phi


@dataclass
class WallObservation:
    stamp: float
    distance: float
    phi: float
    rms: float
    n_inliers: int
    span: float
    sigma_distance: float
    sigma_phi: float
    inliers: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))


class WallFitter:

    def __init__(self, params: Optional[WallFitParams] = None,
                 extrinsics: Optional[LidarExtrinsics] = None, seed: int = 7):
        self.p = params or WallFitParams()
        self.ext = extrinsics or LidarExtrinsics()
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def scan_to_points(self, ranges, angle_min: float, angle_increment: float,
                       range_min: float, range_max: float) -> np.ndarray:
        r = np.asarray(ranges, dtype=np.float64)
        ang = angle_min + angle_increment * np.arange(r.size)
        r_lo = min(range_min, self.p.min_range) if range_min > 0.0 else self.p.min_range
        r_lo = max(r_lo, 0.02)
        r_hi = min(range_max, self.p.max_range)
        ok = np.isfinite(r) & (r >= r_lo) & (r <= r_hi)
        r, ang = r[ok], ang[ok]
        a = ang + self.ext.yaw
        px = self.ext.x + r * np.cos(a)
        py = self.ext.y + r * np.sin(a)
        bearing = np.arctan2(py, px)
        fwd = (px > 0.0) & (np.abs(bearing) <= math.radians(self.p.fov_half_deg))
        return np.stack([px[fwd], py[fwd]], axis=1)

    # ------------------------------------------------------------------
    @staticmethod
    def _tls_line(pts: np.ndarray):
        c = pts.mean(axis=0)
        cov = np.cov((pts - c).T)
        w, v = np.linalg.eigh(cov)
        n = v[:, 0]  # smallest eigenvalue -> normal
        d = float(n @ c)
        if d < 0.0:
            n, d = -n, -d
        return n, d, c

    def fit(self, pts: np.ndarray, stamp: float = 0.0,
            hint_point: Optional[np.ndarray] = None,
            phi_hint: Optional[float] = None) -> Optional[WallObservation]:
        p = self.p
        if pts is None or len(pts) < p.min_inliers:
            return None
        n_pts = len(pts)

        def _run_ransac(use_hint: bool):
            best = None
            best_score = -1.0
            hint = None if (hint_point is None or not use_hint) else np.asarray(hint_point[:2], dtype=np.float64)

            for _ in range(p.ransac_iterations):
                i, j = self._rng.choice(n_pts, 2, replace=False)
                a, b = pts[i], pts[j]
                t = b - a
                norm = math.hypot(t[0], t[1])
                if norm < 0.02:
                    continue
                n = np.array([-t[1], t[0]]) / norm
                d = float(n @ a)
                if d < 0.0:
                    n, d = -n, -d
                if d < 0.02:
                    continue
                # the wall must face the robot: normal roughly forward
                if n[0] < math.cos(math.radians(80.0)):
                    continue
                cand_phi = math.atan2(n[1], n[0])
                if phi_hint is not None:
                    # Reject side walls: candidate inward normal must roughly align with docking wall normal
                    diff = abs(math.atan2(math.sin(cand_phi - phi_hint), math.cos(cand_phi - phi_hint)))
                    if diff > math.radians(28.0):
                        continue
                if hint is not None and abs(float(n @ hint) - d) > p.hint_max_offset:
                    continue
                res = np.abs(pts @ n - d)
                inl = res < p.inlier_threshold
                score = float(np.count_nonzero(inl))
                if score > best_score:
                    best_score = score
                    best = inl
            return best, best_score, hint

        # If hint_point is given, require hint matching (prevents locking onto side corridor walls)
        best, best_score, hint = _run_ransac(use_hint=True)
        if (best is None or best_score < p.min_inliers) and hint_point is None:
            best, best_score, hint = _run_ransac(use_hint=False)

        if best is None or best_score < p.min_inliers:
            return None

        inl_pts = pts[best]
        n, d, c = self._tls_line(inl_pts)
        for _ in range(2):
            res = np.abs(pts @ n - d)
            mask = res < p.inlier_threshold
            if hint is not None:
                tang = np.array([-n[1], n[0]])
                along = (pts - hint) @ tang
                mask &= np.abs(along) < p.hint_window
            if np.count_nonzero(mask) < p.min_inliers:
                break
            inl_pts = pts[mask]
            n, d, c = self._tls_line(inl_pts)

        tang = np.array([-n[1], n[0]])
        s = inl_pts @ tang
        span = float(s.max() - s.min())
        if span < p.min_span:
            return None
        res = inl_pts @ n - d
        rms = float(np.sqrt(np.mean(res ** 2)))
        n_in = len(inl_pts)
        point_sigma = max(rms, p.range_sigma)
        sigma_d = math.hypot(point_sigma / math.sqrt(n_in), p.distance_bias_sigma)
        # slope std of a least-squares line through n_in points spread over `span`
        sigma_phi_rand = point_sigma * math.sqrt(12.0 / n_in) / max(span, 1e-3)
        sigma_phi = math.hypot(sigma_phi_rand, math.radians(p.angle_bias_sigma_deg))
        phi = math.atan2(n[1], n[0]) - math.radians(p.angle_bias_deg)

        if phi_hint is not None:
            diff = abs(math.atan2(math.sin(phi - phi_hint), math.cos(phi - phi_hint)))
            if diff > math.radians(28.0):
                return None
        return WallObservation(
            stamp=stamp,
            distance=d - p.distance_bias,
            phi=phi,
            rms=rms,
            n_inliers=n_in,
            span=span,
            sigma_distance=sigma_d,
            sigma_phi=sigma_phi,
            inliers=inl_pts,
            points=pts,
        )
