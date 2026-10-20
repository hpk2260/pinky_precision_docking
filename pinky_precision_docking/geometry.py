"""
Planar geometry helpers shared by the docking estimator / controller.

Frame conventions (REP-103):
  * base frame  B : origin = wheel-axle centre (physical rotation centre), +x forward, +y left
  * odom frame  O : world-fixed odometry frame
  * docking frame P : origin = marker centre projected on the floor,
                      +x_P = INTO the wall (along the inward wall normal),
                      +y_P = left along the wall when facing the wall.

  Robot pose expressed in P  ->  (x_P, y_P, psi)
      d_base = -x_P  : distance from the rotation centre to the wall plane
      y      =  y_P  : lateral offset of the rotation centre from the marker centreline
                       (+ = robot is LEFT of the centreline when facing the wall)
      psi            : heading error w.r.t. the inward wall normal (+ = rotated CCW / left)
"""

from dataclasses import dataclass
import math
from typing import Tuple


def wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return math.atan2(math.sin(a), math.cos(a))


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def sat(x: float, limit: float) -> float:
    """Symmetric saturation."""
    return clamp(x, -abs(limit), abs(limit))


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def transform_point(self, px: float, py: float) -> Tuple[float, float]:
        """Map a point from this pose's local frame into the parent frame."""
        c, s = math.cos(self.theta), math.sin(self.theta)
        return self.x + c * px - s * py, self.y + s * px + c * py

    def inverse_transform_point(self, px: float, py: float) -> Tuple[float, float]:
        """Map a point from the parent frame into this pose's local frame."""
        dx, dy = px - self.x, py - self.y
        c, s = math.cos(self.theta), math.sin(self.theta)
        return c * dx + s * dy, -s * dx + c * dy

    def compose(self, other: 'Pose2D') -> 'Pose2D':
        x, y = self.transform_point(other.x, other.y)
        return Pose2D(x, y, wrap_angle(self.theta + other.theta))


def interpolate_pose(p0: Pose2D, p1: Pose2D, ratio: float) -> Pose2D:
    ratio = clamp(ratio, 0.0, 1.0)
    return Pose2D(
        p0.x + (p1.x - p0.x) * ratio,
        p0.y + (p1.y - p0.y) * ratio,
        wrap_angle(p0.theta + wrap_angle(p1.theta - p0.theta) * ratio),
    )


def marker_bearing_from_camera(d_base: float, y: float, psi: float,
                               cam_x: float, cam_y: float = 0.0) -> float:
    """
    Bearing (rad, + = left) of the marker centre as seen from a camera mounted at
    (cam_x, cam_y) in the base frame, for a robot at docking-frame pose (-d_base, y, psi).
    """
    c, s = math.cos(psi), math.sin(psi)
    # camera position in P
    px = -d_base + c * cam_x - s * cam_y
    py = y + s * cam_x + c * cam_y
    # vector camera -> marker (marker at origin) rotated into the robot frame
    vx, vy = -px, -py
    rx = c * vx + s * vy
    ry = -s * vx + c * vy
    return math.atan2(ry, rx)


def max_heading_keeping_marker_in_fov(d_base: float, y: float, cam_x: float,
                                      edge_bearing: float) -> float:
    """
    Largest |psi| (pointing the robot TOWARD the centreline) for which the marker stays at
    a camera bearing <= edge_bearing.  Solved by bisection on the exact camera geometry,
    so the forward camera lever arm is taken into account.
    Returns a non-negative magnitude.
    """
    if abs(y) < 1e-6:
        return edge_bearing
    sgn = -1.0 if y > 0 else 1.0  # heading sign that moves the robot toward the centreline
    lo, hi = 0.0, math.radians(89.0)

    def bearing_mag(mag: float) -> float:
        return abs(marker_bearing_from_camera(d_base, y, sgn * mag, cam_x))

    # bearing as a function of mag first decreases (marker swings to the image centre)
    # then increases (marker swings to the opposite edge). find the largest feasible mag.
    if bearing_mag(hi) <= edge_bearing:
        return hi
    # find centre crossing (bearing ~ 0) first, then search above it
    centre = math.atan2(abs(y), max(1e-3, d_base))
    lo = clamp(centre, 0.0, hi)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if bearing_mag(mid) <= edge_bearing:
            lo = mid
        else:
            hi = mid
    return lo
