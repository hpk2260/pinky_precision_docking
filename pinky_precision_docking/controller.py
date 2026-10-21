"""
Multi-stage docking controller for a differential-drive robot.

        SEARCH ─► ACQUIRE ─► VERIFY ◄───────────────────────────────┐
                               │  (closed-loop decision point)      │
          ┌────────────────────┼─────────────────────┐              │
          ▼                    ▼                     ▼              │
   INTERCEPT_TURN         AXIS_ALIGN             RETREAT ───────────┤
          ▼                    │                                    │
   INTERCEPT_DRIVE ─► AXIS_ALIGN ─► VERIFY ─► TERMINAL ─► WALL_ALIGN ─► DOCKED
                                                 │  guard      │ guard
                                                 └──► RETREAT ◄─┘

Stage 1 - Pre-docking intercept (INTERCEPT_TURN / INTERCEPT_DRIVE / AXIS_ALIGN)
    The robot first merges onto the marker centreline at a staging distance, far from the
    wall.  The heading command is a pursuit of the staging point S = (d_stage, y = 0),
    saturated by a FOV constraint: the robot never turns further than the angle that puts the
    marker at the edge of the usable field of view (visual tracking).  When the constraint is
    active the path becomes a deviated-pursuit (log-spiral) that keeps the marker at the FOV
    edge.  The crossing distance of the manoeuvre is predicted beforehand by forward
    simulation; if it is too close to the wall, the robot backs off first (RETREAT).
    When the rotation centre reaches the centreline the robot rotates in place to face the
    wall (in-place rotation about the axle does not change the lateral offset).

Stage 2 - Closed-loop verification (VERIFY)
    Stop, let the sensors settle, re-measure.  Residual lateral offset is classified:
      * small (correctable within the remaining distance with <= fine_max_heading) -> TERMINAL
        with fine lateral correction
      * large but there is room for another intercept -> re-plan intercept
      * large and too close -> RETREAT (back off while correcting) then re-verify

Stage 3 - Terminal straight approach & Wall Parallel Alignment (TERMINAL / WALL_ALIGN)
    Decelerating approach. Small heading deviations (<= fine_max_heading) are allowed only
    while there is distance left; the deviation limit ramps to zero over the last
    `final_straight` metres so the final entry is parallel ("11-shape").
    Upon reaching target distance, WALL_ALIGN locks linear velocity to zero (v = 0.0)
    and performs ultra-fine in-place rotation using LiDAR wall-fitting to align the front bumper
    parallel to the wall (|psi| <= wall_align_tol_deg).
    Defensive guard: if the lateral/heading error grows beyond what can still be corrected
    before contact, stop and RETREAT instead of hitting the wall at an angle.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import List, Optional, Tuple

from .estimator import DockingEstimate
from .geometry import clamp, max_heading_keeping_marker_in_fov, sat, wrap_angle


class Phase(str, Enum):
    IDLE = 'IDLE'
    SEARCH = 'SEARCH'
    ACQUIRE = 'ACQUIRE'
    VERIFY = 'VERIFY'
    INTERCEPT_TURN = 'INTERCEPT_TURN'
    INTERCEPT_DRIVE = 'INTERCEPT_DRIVE'
    AXIS_ALIGN = 'AXIS_ALIGN'
    TERMINAL = 'TERMINAL'
    FINAL_ALIGN = 'FINAL_ALIGN'
    WALL_ALIGN = 'WALL_ALIGN'
    RETREAT = 'RETREAT'
    REACQUIRE = 'REACQUIRE'
    DOCKED = 'DOCKED'
    FAILED = 'FAILED'


ACTIVE_PHASES = {p for p in Phase} - {Phase.IDLE, Phase.DOCKED, Phase.FAILED}


@dataclass
class ControllerParams:
    # goal & tolerances
    target_distance: float = 0.10       # reference point -> wall [m]
    distance_tolerance: float = 0.01
    center_tolerance: float = 0.005     # strict 5mm lateral tolerance (was 15mm)
    angle_tolerance_deg: float = 2.0
    aim_ratio: float = 0.4              # internal aim = tolerance * aim_ratio
    # geometry
    cam_x: float = 0.035
    ref_x: float = 0.035
    camera_hfov_deg: float = 66.0
    fov_margin_deg: float = 7.0
    # velocity limits / dynamics
    v_max: float = 0.08
    v_terminal_max: float = 0.04
    v_min: float = 0.006
    v_retreat: float = 0.04
    w_max: float = 0.6
    w_min: float = 0.05
    w_search: float = 0.35
    a_lin: float = 0.25                 # command smoother limits
    a_ang: float = 1.5
    a_dec_lin: float = 0.06             # profile deceleration for stopping
    a_dec_ang: float = 0.8
    k_heading: float = 2.2
    # stage 1: intercept
    staging_distance: float = 0.30      # rotation centre -> wall where the merge should end
    min_cross_distance: float = 0.12    # crossing closer than this is rejected (0.12m gives 2cm straight margin before 0.10m target)
    intercept_lookahead_min: float = 0.04
    intercept_max_heading_deg: float = 65.0  # flexible turning up to 65 deg for large lateral offsets (was 35 deg)
    axis_reached_tol: float = 0.025     # 25mm realistic tolerance to merge onto centerline (was 3mm)
    max_retreat_distance: float = 0.65  # rotation centre -> wall, upper bound for backing off
    allow_retreat: bool = True          # True: allow backing off when near wall, capped by initial start position
    # stage 2: verification
    settle_time: float = 0.5
    verify_min_cam_updates: int = 3
    verify_timeout: float = 2.5
    # stage 3: terminal
    fine_max_heading_deg: float = 15.0  # flexible fine correction up to 15 deg (was 8.0)
    fine_lookahead: float = 0.08        # smooth lookahead for lateral error correction (was 0.04)
    final_straight: float = 0.03
    fine_max_lateral: float = 0.045     # 4.5cm fine correctable lateral envelope (was 0.03)
    abort_heading_deg: float = 25.0
    final_dwell: float = 0.6
    final_align_timeout: float = 10.0
    retreat_margin: float = 0.10        # how far to back off for a terminal retry
    # stage 4: wall parallel alignment
    target_wall_angle_deg: float = 0.0  # 0.0: front bumper parallel to wall ("11-shape"), 90.0: side parallel
    wall_align_tol_deg: float = 0.5     # ultra-fine angular tolerance for wall alignment
    # robustness
    max_retries: int = 5
    docking_timeout: float = 180.0
    search_timeout: float = 30.0
    cam_lost_timeout: float = 1.2       # camera age that triggers REACQUIRE in far phases
    blind_sigma_y: float = 0.006        # max lateral sigma tolerated while blind
    sensor_fresh_age: float = 0.35


@dataclass
class ControlOutput:
    v: float = 0.0
    w: float = 0.0
    phase: Phase = Phase.IDLE
    info: dict = field(default_factory=dict)


class CommandSmoother:
    """Acceleration-limited command shaping with motor dead-band compensation."""

    def __init__(self, p: ControllerParams):
        self.p = p
        self.v = 0.0
        self.w = 0.0

    def reset(self):
        self.v = 0.0
        self.w = 0.0

    def __call__(self, v_t: float, w_t: float, dt: float) -> Tuple[float, float]:
        p = self.p
        dt = clamp(dt, 1e-3, 0.2)
        dv = clamp(v_t - self.v, -p.a_lin * dt, p.a_lin * dt)
        dw = clamp(w_t - self.w, -p.a_ang * dt, p.a_ang * dt)
        self.v += dv
        self.w += dw
        # never keep moving when the target is zero and we are already slow
        if abs(v_t) < 1e-6 and abs(self.v) < p.v_min:
            self.v = 0.0
        if abs(w_t) < 1e-6 and abs(self.w) < p.w_min * 0.5:
            self.w = 0.0
        v, w = self.v, self.w
        if abs(v_t) > 1e-6 and 0.0 < abs(v) < p.v_min and v * v_t > 0:
            v = math.copysign(p.v_min, v_t)
        if abs(v_t) < 1e-6 and abs(w_t) > 1e-6 and 0.0 < abs(w) < p.w_min and w * w_t > 0:
            w = math.copysign(p.w_min, w_t)
        return v, w


class DockingController:

    def __init__(self, params: Optional[ControllerParams] = None):
        self.p = params or ControllerParams()
        self.smoother = CommandSmoother(self.p)
        self.events: List[str] = []
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, phase: Phase = Phase.IDLE):
        self.phase = phase
        self.phase_start = None
        self.start_time = None
        self.last_time = None
        self.retries = 0
        self.retreats = 0
        self.intercepts = 0
        self.align_loops = 0
        self.reason = ''
        self.target: dict = {}
        self.dwell_start = None
        self.last_bearing_sign = 1.0
        self.search_turned = 0.0
        self.y_at_drive_start = 0.0
        self.cam_updates_at_phase = 0
        self.lidar_updates_at_phase = 0
        self.smoother.reset()
        self.max_abs_w = 0.0
        self.path_length = 0.0
        self.result: Optional[dict] = None
        self.initial_d_base: Optional[float] = None

    def start(self, now: float):
        self.reset(Phase.SEARCH)
        self.start_time = now
        self.phase_start = now
        self._event(now, Phase.SEARCH, 'docking started')

    def cancel(self, now: float):
        self._event(now, Phase.IDLE, 'cancelled')
        self.phase = Phase.IDLE
        self.smoother.reset()

    @property
    def active(self) -> bool:
        return self.phase in ACTIVE_PHASES

    # ------------------------------------------------------------------
    def _event(self, now: float, phase: Phase, reason: str):
        t = 0.0 if self.start_time is None else now - self.start_time
        self.events.append(f'[t={t:6.2f}s] {self.phase.value} -> {phase.value}: {reason}')

    def _goto(self, now: float, phase: Phase, reason: str, est: Optional[DockingEstimate] = None,
              cam_updates: int = 0, lidar_updates: int = 0):
        self._event(now, phase, reason)
        self.phase = phase
        self.phase_start = now
        self.reason = reason
        self.dwell_start = None
        self.cam_updates_at_phase = cam_updates
        self.lidar_updates_at_phase = lidar_updates

    # ------------------------------------------------------------------
    # helper laws
    # ------------------------------------------------------------------
    @property
    def edge_bearing(self) -> float:
        return math.radians(self.p.camera_hfov_deg / 2.0 - self.p.fov_margin_deg)

    def remain(self, est: DockingEstimate) -> float:
        return est.d_ref - self.p.target_distance

    def get_staging_distance(self, d_base: float) -> float:
        """Adaptive staging distance: for 30cm start, staging ends at ~0.19m rather than 0.30m."""
        p = self.p
        d_goal = p.target_distance + p.ref_x  # ~0.135m
        min_stage = d_goal + 0.035           # at least 3.5cm straight buffer
        adaptive = d_goal + 0.35 * (d_base - d_goal)
        return max(min_stage, min(p.staging_distance, adaptive))

    def intercept_heading(self, d_base: float, y: float) -> Tuple[float, bool]:
        """Heading command toward the staging point, FOV constrained. Returns (psi, limited)."""
        p = self.p
        stage_d = self.get_staging_distance(d_base)
        look = max(d_base - stage_d, p.intercept_lookahead_min)
        psi = -math.atan2(y, look)
        psi = sat(psi, math.radians(p.intercept_max_heading_deg))
        fov_lim = max_heading_keeping_marker_in_fov(d_base, y, p.cam_x, self.edge_bearing)
        limited = abs(psi) > fov_lim
        return sat(psi, fov_lim), limited

    def predict_crossing(self, d_base: float, y: float) -> float:
        """Forward-simulate the intercept law; return the wall distance at centreline merge."""
        p = self.p
        d, yy = d_base, y
        ds = 0.002
        tol = max(p.axis_reached_tol, 0.003)
        for _ in range(2000):
            if abs(yy) <= tol or yy * y < 0:
                return d
            psi, _ = self.intercept_heading(d, yy)
            d -= ds * math.cos(psi)
            yy += ds * math.sin(psi)
            if d < 0.05:
                break
        return d

    def fine_correctable_lateral(self, remain: float) -> float:
        p = self.p
        room = max(0.0, remain - p.final_straight)
        return min(p.fine_max_lateral, max(0.025, room * math.tan(math.radians(p.fine_max_heading_deg))))

    def lateral_abort_limit(self, remain: float) -> float:
        p = self.p
        room = max(0.0, remain - p.final_straight)
        # Generous envelope during approach so small heading/lateral corrections don't trigger retreat
        return max(0.035, room * math.tan(math.radians(15.0)) + 0.02)

    def rotate_law(self, err: float, tol: float) -> float:
        p = self.p
        mag = min(p.w_max, p.k_heading * abs(err), math.sqrt(2.0 * p.a_dec_ang * abs(err)))
        if abs(err) > tol:
            mag = max(mag, p.w_min)
        else:
            mag = 0.0
        return math.copysign(mag, err)

    # ------------------------------------------------------------------
    def step(self, now: float, est: DockingEstimate, cam_updates: int = 0,
             lidar_updates: int = 0) -> ControlOutput:
        p = self.p
        if self.start_time is None or self.start_time <= 1e-3:
            self.start_time = now
            self.phase_start = now
        dt = 0.05 if self.last_time is None else now - self.last_time
        self.last_time = now
        out = ControlOutput(phase=self.phase)
        if not self.active:
            self.smoother.reset()
            out.info = self._info(est)
            return out

        if est.valid and self.initial_d_base is None and est.d_base > 0.12:
            self.initial_d_base = est.d_base

        v_t, w_t = self._phase_logic(now, est, cam_updates, lidar_updates)
        if not self.active:
            v_t, w_t = 0.0, 0.0
        v, w = self.smoother(v_t, w_t, dt)
        if not self.active:
            v, w = 0.0, 0.0
            self.smoother.reset()
        self.max_abs_w = max(self.max_abs_w, abs(w))
        self.path_length += abs(v) * dt
        out.v, out.w, out.phase = v, w, self.phase
        out.info = self._info(est)
        return out

    def _info(self, est: DockingEstimate) -> dict:
        info = {
            'phase': self.phase.value,
            'reason': self.reason,
            'retries': self.retries,
            'retreats': self.retreats,
            'intercepts': self.intercepts,
            'elapsed': 0.0 if (self.start_time is None or self.last_time is None)
            else self.last_time - self.start_time,
        }
        info.update(self.target)
        return info

    # ------------------------------------------------------------------
    def _phase_logic(self, now: float, est: DockingEstimate, cam_updates: int,
                     lidar_updates: int) -> Tuple[float, float]:
        p = self.p
        t_phase = now - self.phase_start
        go = lambda ph, why: self._goto(now, ph, why, est, cam_updates, lidar_updates)  # noqa

        if self.start_time is not None and now - self.start_time > p.docking_timeout:
            go(Phase.FAILED, 'docking timeout')
            return 0.0, 0.0
        if self.retries > p.max_retries:
            go(Phase.FAILED, f'too many retries ({self.retries})')
            return 0.0, 0.0

        cam_fresh = est.cam_age < p.sensor_fresh_age
        if est.valid and cam_fresh:
            self.last_bearing_sign = 1.0 if est.marker_bearing >= 0 else -1.0

        # ---------------------------------------------------------- SEARCH
        if self.phase == Phase.SEARCH:
            if est.valid and est.cam_age < 0.3:
                go(Phase.ACQUIRE, 'marker detected')
                return 0.0, 0.0
            if t_phase > p.search_timeout:
                go(Phase.FAILED, 'marker not found')
                return 0.0, 0.0
            if t_phase < 0.6:   # look first before turning
                return 0.0, 0.0
            return 0.0, self.last_bearing_sign * p.w_search

        # ---------------------------------------------------------- ACQUIRE
        if self.phase == Phase.ACQUIRE:
            n_cam = cam_updates - self.cam_updates_at_phase
            n_lid = lidar_updates - self.lidar_updates_at_phase
            if est.valid and t_phase > p.settle_time and n_cam >= p.verify_min_cam_updates \
                    and (n_lid >= 2 or t_phase > 2.0):
                go(Phase.VERIFY, 'anchor initialised')
            elif t_phase > 4.0:
                go(Phase.SEARCH, 'acquisition failed')
            return 0.0, 0.0

        if not est.valid:
            go(Phase.SEARCH, 'estimate lost')
            return 0.0, 0.0

        remain = self.remain(est)
        ang_tol = math.radians(p.angle_tolerance_deg)

        # ---------------------------------------------------------- VERIFY
        if self.phase == Phase.VERIFY:
            n_cam = cam_updates - self.cam_updates_at_phase
            n_lid = lidar_updates - self.lidar_updates_at_phase
            settled = t_phase > p.settle_time and n_cam >= p.verify_min_cam_updates \
                and (n_lid >= 2 or est.lidar_age > 1.0)
            if not settled:
                if t_phase > p.verify_timeout:
                    if est.cam_age > p.cam_lost_timeout:
                        go(Phase.REACQUIRE, 'marker not visible during verification')
                    else:
                        settled = True
                if not settled:
                    return 0.0, 0.0
            self._decide(now, est, go)
            return 0.0, 0.0

        # ---------------------------------------------------------- INTERCEPT_TURN
        if self.phase == Phase.INTERCEPT_TURN:
            psi_cmd, limited = self.intercept_heading(est.d_base, est.y)
            self.target = {'psi_cmd': psi_cmd, 'fov_limited': limited}
            err = psi_cmd - est.psi
            if abs(err) < math.radians(2.0):
                self.y_at_drive_start = est.y
                go(Phase.INTERCEPT_DRIVE, f'heading {math.degrees(psi_cmd):+.1f} deg reached')
                return 0.0, 0.0
            if t_phase > 12.0:
                go(Phase.VERIFY, 'intercept turn timeout')
                self.retries += 1
                return 0.0, 0.0
            return 0.0, self.rotate_law(err, math.radians(1.0))

        # ---------------------------------------------------------- INTERCEPT_DRIVE
        if self.phase == Phase.INTERCEPT_DRIVE:
            if est.cam_age > p.cam_lost_timeout and est.sigma_y > p.blind_sigma_y:
                go(Phase.REACQUIRE, 'marker lost during intercept')
                return 0.0, 0.0
            crossed = (est.y * self.y_at_drive_start <= 0 and abs(est.y) < 0.040)
            if abs(est.y) <= p.axis_reached_tol or crossed:
                go(Phase.AXIS_ALIGN, f'centreline reached (y={est.y * 100:+.2f} cm, '
                                     f'd={est.d_base:.3f} m)')
                return 0.0, 0.0
            if est.d_base <= p.min_cross_distance - 0.02:
                go(Phase.VERIFY, 'too close to wall before merging')
                return 0.0, 0.0
            psi_cmd, limited = self.intercept_heading(est.d_base, est.y)
            err = psi_cmd - est.psi
            w = sat(p.k_heading * err, p.w_max)
            stage_d = self.get_staging_distance(est.d_base)
            dist_s = math.hypot(max(est.d_base - stage_d, 0.0), est.y)
            v = clamp(math.sqrt(2.0 * p.a_dec_lin * dist_s), 0.025, p.v_max)
            # Gentle slowdown as we near the centerline to avoid overshooting
            if abs(est.y) < 0.08:
                v = min(v, 0.035)
            v *= clamp(1.0 - abs(err) / math.radians(35.0), 0.0, 1.0)
            self.target = {'psi_cmd': psi_cmd, 'fov_limited': limited,
                           'staging_distance': stage_d}
            return v, w

        # ---------------------------------------------------------- AXIS_ALIGN
        if self.phase == Phase.AXIS_ALIGN:
            err = -est.psi
            self.target = {'psi_cmd': 0.0}
            done = abs(err) < ang_tol * 0.5 or (t_phase > 2.5 and abs(err) < ang_tol * p.aim_ratio)
            if done:
                go(Phase.VERIFY, f'perpendicular (psi={math.degrees(est.psi):+.2f} deg)')
                return 0.0, 0.0
            if t_phase > 8.0:
                self.retries += 1
                go(Phase.VERIFY, 'axis align timeout')
                return 0.0, 0.0
            return 0.0, self.rotate_law(err, ang_tol * 0.25)

        # ---------------------------------------------------------- TERMINAL
        if self.phase == Phase.TERMINAL:
            if est.cam_age > p.cam_lost_timeout and est.sigma_y > p.blind_sigma_y:
                go(Phase.REACQUIRE, 'marker lost during terminal approach (sigma_y high)')
                return 0.0, 0.0
            if remain <= p.distance_tolerance * 0.5:
                if abs(est.y) > p.center_tolerance:
                    self.retries += 1
                    self._start_retreat(est)
                    go(Phase.RETREAT, f'guard: residual lateral error too large at goal '
                                      f'(y={est.y * 100:+.2f} cm > {p.center_tolerance * 100:.1f} cm)')
                    return 0.0, 0.0
                go(Phase.WALL_ALIGN, f'goal distance reached (remain={remain * 100:+.2f} cm)')
                return 0.0, 0.0
            is_lateral_too_large = abs(est.y) > self.lateral_abort_limit(remain)
            is_heading_too_large = abs(est.psi) > math.radians(p.abort_heading_deg)
            if is_lateral_too_large or is_heading_too_large:
                self.retries += 1
                if remain > 0.05 and est.d_base >= 0.18:
                    go(Phase.VERIFY, f'guard: re-intercepting forward (y={est.y * 100:+.2f} cm, '
                                     f'psi={math.degrees(est.psi):+.1f} deg)')
                else:
                    self._start_retreat(est)
                    go(Phase.RETREAT, f'guard: residual too large near wall (y={est.y * 100:+.2f} cm, '
                                      f'psi={math.degrees(est.psi):+.1f} deg)')
                return 0.0, 0.0
            ramp = clamp((remain - p.final_straight) / 0.04, 0.0, 1.0)
            psi_lim = math.radians(p.fine_max_heading_deg) * ramp
            psi_ref = sat(-est.y / p.fine_lookahead, psi_lim)
            w = sat(p.k_heading * (psi_ref - est.psi), p.w_max * 0.5)
            v = min(p.v_terminal_max, math.sqrt(2.0 * p.a_dec_lin * max(remain, 0.0)),
                    0.8 * remain + 0.004)
            v = max(v, p.v_min)
            self.target = {'psi_cmd': psi_ref, 'psi_lim': psi_lim, 'remain': remain}
            return v, w

        # ---------------------------------------------------------- FINAL_ALIGN
        if self.phase == Phase.FINAL_ALIGN:
            if abs(remain) <= p.distance_tolerance * 0.5:
                go(Phase.WALL_ALIGN, f'distance satisfied (remain={remain * 100:+.2f} cm) -> wall align')
                return 0.0, 0.0
            d_tol = p.distance_tolerance
            c_tol = p.center_tolerance
            self.target = {'remain': remain}
            if abs(est.y) > c_tol:
                self.retries += 1
                self._start_retreat(est)
                go(Phase.RETREAT, f'final lateral error {est.y * 100:+.2f} cm -> retry')
                return 0.0, 0.0
            v, w = 0.0, 0.0
            if abs(remain) > d_tol * p.aim_ratio:
                v = sat(0.8 * remain, 0.015)
                if abs(v) < p.v_min:
                    v = math.copysign(p.v_min, remain)
                w = sat(p.k_heading * (-est.psi), 0.2)
            else:
                go(Phase.WALL_ALIGN, 'fine distance settled -> wall align')
                return 0.0, 0.0
            return v, w

        # ---------------------------------------------------------- WALL_ALIGN
        if self.phase == Phase.WALL_ALIGN:
            # Pure in-place rotation: linear velocity strictly 0 to preserve docking distance
            v = 0.0
            target_psi = math.radians(p.target_wall_angle_deg)
            err = wrap_angle(target_psi - est.psi)
            wall_tol = math.radians(p.wall_align_tol_deg)
            c_tol = p.center_tolerance
            d_tol = p.distance_tolerance

            self.target = {
                'wall_err_deg': math.degrees(err),
                'remain': remain,
                'target_wall_angle_deg': p.target_wall_angle_deg
            }

            # Safety guard: if lateral error drifted excessively, back off and retry
            if abs(est.y) > c_tol * 1.5:
                self.retries += 1
                self._start_retreat(est)
                go(Phase.RETREAT, f'wall align lateral guard exceeded (y={est.y * 100:+.2f} cm)')
                return 0.0, 0.0

            # In-place rotation law:
            # Pinky Pro requires at least 0.18 rad/s to overcome wheel static friction when v=0
            if abs(err) > wall_tol:
                w_cmd = self.rotate_law(err, wall_tol * 0.25)
                if abs(w_cmd) > 1e-4 and abs(w_cmd) < 0.18:
                    w_cmd = math.copysign(0.18, err)
                w = w_cmd
            else:
                w = 0.0

            # Settled check: aligned within tolerance, stopped, and LiDAR wall measurement fresh
            is_stopped = abs(self.smoother.v) < 1e-4 and abs(self.smoother.w) < 1e-4 and abs(w) < 1e-4
            is_aligned = abs(err) <= wall_tol and abs(remain) <= d_tol * 1.5 and abs(est.y) <= c_tol
            lidar_fresh = est.lidar_age < p.sensor_fresh_age

            # In WALL_ALIGN, require fresh LiDAR measurement so we do not falsely finish while LiDAR is blind
            if is_aligned and is_stopped and lidar_fresh:
                if self.dwell_start is None:
                    self.dwell_start = now
                if now - self.dwell_start >= p.final_dwell:
                    self._finish(now, est)
                    return 0.0, 0.0
            else:
                self.dwell_start = None

            # Timeout guard: if timeout expires, accept reasonable alignment (<1.5 deg, tight y) or retry
            if t_phase > p.final_align_timeout:
                if abs(err) <= math.radians(1.5) and abs(remain) <= d_tol * 1.5 and abs(est.y) <= c_tol * 1.2:
                    self._finish(now, est)
                else:
                    self.retries += 1
                    self._start_retreat(est)
                    go(Phase.RETREAT, f'wall align timeout (err={math.degrees(err):+.2f} deg, y={est.y * 100:+.2f} cm)')
                return 0.0, 0.0

            return v, w

        # ---------------------------------------------------------- RETREAT
        if self.phase == Phase.RETREAT:
            goal = self.target.get('retreat_goal', est.d_base + 0.1)
            max_dist = self.initial_d_base if self.initial_d_base is not None else p.max_retreat_distance
            if est.d_base >= goal or est.d_base >= max_dist or t_phase > 12.0:
                go(Phase.VERIFY, f'backed off to d={est.d_base:.3f} m')
                return 0.0, 0.0
            psi_ref = sat(est.y / 0.08, math.radians(8.0))   # reversing also reduces |y|
            w = sat(p.k_heading * (psi_ref - est.psi), p.w_max * 0.5)
            v = -clamp(math.sqrt(2.0 * p.a_dec_lin * (goal - est.d_base)), 0.01, p.v_retreat)
            return v, w

        # ---------------------------------------------------------- REACQUIRE
        if self.phase == Phase.REACQUIRE:
            if est.cam_age < 0.2:
                go(Phase.VERIFY, 'marker re-acquired')
                return 0.0, 0.0
            if t_phase < 3.0:
                # point the camera to where the marker should be
                return 0.0, self.rotate_law(est.marker_bearing, math.radians(1.0))
            if t_phase < 6.0:
                max_d = self.initial_d_base if self.initial_d_base is not None else 0.65
                v_back = -0.03 if (est.d_base < max_d) else 0.0
                return v_back, 0.0
            go(Phase.SEARCH, 'reacquire failed')
            return 0.0, 0.0

        return 0.0, 0.0

    # ------------------------------------------------------------------
    def _start_retreat(self, est: DockingEstimate, extra: Optional[float] = None):
        p = self.p
        extra = p.retreat_margin if extra is None else extra
        # Never retreat further back than the initial starting position!
        max_dist = self.initial_d_base if self.initial_d_base is not None else p.max_retreat_distance
        goal = min(max_dist, est.d_base + extra)
        self.target = {'retreat_goal': goal}
        self.retreats += 1

    def _decide(self, now: float, est: DockingEstimate, go):
        """Closed-loop verification decision (stage 2)."""
        p = self.p
        remain = self.remain(est)
        ang_tol = math.radians(p.angle_tolerance_deg)
        y_fine = max(self.fine_correctable_lateral(remain), p.axis_reached_tol)
        ay = abs(est.y)
        if remain <= p.distance_tolerance and ay < p.center_tolerance:
            go(Phase.WALL_ALIGN, 'at goal distance')
            return
        if ay <= max(y_fine, p.center_tolerance):
            if abs(est.psi) > max(ang_tol * 0.5, math.radians(p.fine_max_heading_deg)) \
                    and self.align_loops < 3:
                self.align_loops += 1
                go(Phase.AXIS_ALIGN, f'heading {math.degrees(est.psi):+.1f} deg before terminal')
                return
            mode = 'straight' if ay <= p.center_tolerance * p.aim_ratio else 'fine-correction'
            go(Phase.TERMINAL, f'residual y={est.y * 100:+.2f} cm '
                               f'(limit {max(y_fine, p.center_tolerance) * 100:.2f} cm) -> {mode}')
            return
        d_cross = self.predict_crossing(est.d_base, est.y)
        self.target = {'predicted_cross': d_cross}

        # Rule: Near the initial starting position, or when far enough from wall (d_base >= 0.18m),
        # NEVER retreat! Always move forward directly into INTERCEPT_TURN.
        is_near_start = (self.initial_d_base is not None and est.d_base >= self.initial_d_base - 0.05)
        has_room_forward = (est.d_base >= 0.18) or is_near_start or (d_cross >= p.min_cross_distance)

        if has_room_forward:
            self.intercepts += 1
            if self.intercepts > 1:
                self.retries += 1
            go(Phase.INTERCEPT_TURN, f'lateral {est.y * 100:+.1f} cm, merge forward at '
                                     f'd={d_cross:.3f} m')
            return

        max_retreat = self.initial_d_base if self.initial_d_base is not None else p.max_retreat_distance
        if not p.allow_retreat or est.d_base >= max_retreat - 0.02:
            self.intercepts += 1
            if self.intercepts > 1:
                self.retries += 1
            go(Phase.INTERCEPT_TURN, f'retreat disallowed or at start position, merge at d={d_cross:.3f} m')
            return

        # Mid-course recovery: if very close to the wall (d_base < 0.18m) and need room, retreat is allowed (capped by start pos)
        need = p.min_cross_distance - d_cross
        self._start_retreat(est, extra=clamp(need * 2.0 + 0.04, 0.06, 0.25))
        go(Phase.RETREAT, f'lateral {est.y * 100:+.1f} cm not mergeable near wall '
                          f'(predicted d={d_cross:.3f} m) -> back off')

    def _finish(self, now: float, est: DockingEstimate):
        self._goto(now, Phase.DOCKED, 'all tolerances satisfied')
        self.result = {
            'success': True,
            'elapsed': now - (self.start_time or now),
            'd_ref': est.d_ref,
            'distance_error': self.remain(est),
            'lateral_error': est.y,
            'angle_error_deg': math.degrees(est.psi),
            'retries': self.retries,
            'retreats': self.retreats,
            'intercepts': self.intercepts,
            'path_length': self.path_length,
            'max_abs_w': self.max_abs_w,
        }

    def fail_result(self, now: float, est: DockingEstimate) -> dict:
        return {
            'success': False, 'reason': self.reason,
            'elapsed': now - (self.start_time or now),
            'd_ref': est.d_ref if est.valid else float('nan'),
            'distance_error': self.remain(est) if est.valid else float('nan'),
            'lateral_error': est.y if est.valid else float('nan'),
            'angle_error_deg': math.degrees(est.psi) if est.valid else float('nan'),
            'retries': self.retries, 'retreats': self.retreats,
            'intercepts': self.intercepts, 'path_length': self.path_length,
            'max_abs_w': self.max_abs_w,
        }
