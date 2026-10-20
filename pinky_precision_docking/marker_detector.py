"""
[Camera branch] ArUco (DICT_5X5_*) detection and metric pose estimation.

  * cv2.aruco detection with sub-pixel corner refinement (OpenCV 4.6 / 4.7+ API compatible)
  * solvePnPGeneric(IPPE_SQUARE) -> both planar pose hypotheses are kept.  The rotation
    ambiguity of a small planar target is resolved with the LiDAR wall normal when it is
    available (LiDAR-guided disambiguation), otherwise by reprojection error.
  * the optical-frame translation is mapped into the robot base frame through the full camera
    extrinsics (forward lever arm, height, pitch tilt), so every downstream quantity refers
    to the physical rotation centre of the robot.
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .geometry import wrap_angle


ARUCO_DICTS = {
    name: getattr(cv2.aruco, name)
    for name in (
        'DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_4X4_1000',
        'DICT_5X5_50', 'DICT_5X5_100', 'DICT_5X5_250', 'DICT_5X5_1000',
        'DICT_6X6_50', 'DICT_6X6_250', 'DICT_ARUCO_ORIGINAL',
    )
    if hasattr(cv2.aruco, name)
}

# optical frame (x right, y down, z forward) -> camera body frame (x fwd, y left, z up)
R_LINK_OPT = np.array([[0.0, 0.0, 1.0],
                       [-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0]])


def _rot_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


@dataclass
class CameraExtrinsics:
    """Camera body frame (x fwd, y left, z up) expressed in the robot base frame."""
    x: float = 0.035     # forward lever arm from the wheel axle (Pinky: 0.020 + 0.015)
    y: float = 0.0
    z: float = 0.0654
    roll: float = 0.0
    pitch: float = 0.0   # + = tilted down (URDF cam_tilt_deg)
    yaw: float = 0.0


@dataclass
class MarkerObservation:
    marker_id: int
    stamp: float
    corners: np.ndarray            # 4x2 refined pixel corners
    p_base: np.ndarray             # marker centre in base frame (x, y, z) [m]
    phi: float                     # inward wall-normal direction in base frame (camera) [rad]
    bearing: float                 # horizontal bearing from the camera axis (+ left) [rad]
    range_cam: float               # optical z distance [m]
    side_px: float                 # mean side length in pixels
    reproj_err: float              # [px]
    ambiguity_ratio: float         # reproj(second) / reproj(best) (>> 1 -> unambiguous)
    edge_margin_px: float          # min distance of a corner to the image border
    rvec: np.ndarray
    tvec: np.ndarray


class ArucoMarkerDetector:

    def __init__(self, dictionary: str = 'DICT_5X5_250', marker_size: float = 0.06,
                 extrinsics: Optional[CameraExtrinsics] = None):
        self.marker_size = float(marker_size)
        self.ext = extrinsics or CameraExtrinsics()
        self.set_dictionary(dictionary)
        self.K: Optional[np.ndarray] = None
        self.D = np.zeros(5)
        self.image_size: Tuple[int, int] = (0, 0)
        self.intrinsics_from_info = False
        self.update_extrinsics(self.ext)

    # ------------------------------------------------------------------
    def set_dictionary(self, name: str):
        if name not in ARUCO_DICTS:
            name = 'DICT_5X5_250'
        self.dictionary_name = name
        self.dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[name])
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            params = cv2.aruco.DetectorParameters_create()
        else:
            params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 50
        params.cornerRefinementMinAccuracy = 0.01
        params.adaptiveThreshWinSizeMin = 5
        params.adaptiveThreshWinSizeMax = 45
        params.adaptiveThreshWinSizeStep = 8
        params.minMarkerPerimeterRate = 0.02
        params.maxMarkerPerimeterRate = 4.0
        self.params = params
        if hasattr(cv2.aruco, 'ArucoDetector'):
            self._detector = cv2.aruco.ArucoDetector(self.dictionary, self.params)
        else:
            self._detector = None

    def update_extrinsics(self, ext: CameraExtrinsics):
        self.ext = ext
        self.R_base_link = _rot_rpy(ext.roll, ext.pitch, ext.yaw)
        self.R_base_opt = self.R_base_link @ R_LINK_OPT
        self.t_base_cam = np.array([ext.x, ext.y, ext.z])

    def set_intrinsics(self, K: np.ndarray, D: np.ndarray, size: Tuple[int, int],
                       from_info: bool = True):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(D, dtype=np.float64).reshape(-1)
        self.image_size = size
        self.intrinsics_from_info = from_info

    def set_intrinsics_from_hfov(self, width: int, height: int, hfov_rad: float):
        fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
        K = np.array([[fx, 0, width / 2.0], [0, fx, height / 2.0], [0, 0, 1]])
        self.set_intrinsics(K, np.zeros(5), (width, height), from_info=False)

    # ------------------------------------------------------------------
    def detect(self, gray: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary,
                                                      parameters=self.params)
        if ids is None:
            return []
        return [(int(i[0]), c.reshape(4, 2).astype(np.float64)) for i, c in zip(ids, corners)]

    def estimate(self, marker_id: int, corners: np.ndarray, stamp: float,
                 phi_hint: Optional[float] = None) -> Optional[MarkerObservation]:
        if self.K is None:
            return None
        h = self.marker_size / 2.0
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
        img = corners.reshape(4, 1, 2).astype(np.float64)
        try:
            n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                obj, img, self.K, self.D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error:
            return None
        if n_sol < 1:
            return None
        errs = np.asarray(errs, dtype=np.float64).reshape(-1) if errs is not None \
            else np.zeros(n_sol)

        cands = []
        for k in range(n_sol):
            rvec, tvec = rvecs[k].reshape(3), tvecs[k].reshape(3)
            if tvec[2] <= 0.0:
                continue
            R, _ = cv2.Rodrigues(rvec)
            n_out_base = self.R_base_opt @ R[:, 2]      # marker +z (out of the wall)
            phi = math.atan2(-n_out_base[1], -n_out_base[0])
            cands.append((float(errs[k]) if k < len(errs) else 0.0, phi, rvec, tvec))
        if not cands:
            return None
        cands.sort(key=lambda c: c[0])
        ambiguity = (cands[1][0] / max(cands[0][0], 1e-6)) if len(cands) > 1 else 1e3
        chosen = cands[0]
        if phi_hint is not None and len(cands) > 1:
            chosen = min(cands, key=lambda c: abs(wrap_angle(c[1] - phi_hint)))
        err, phi, rvec, tvec = chosen

        p_base = self.t_base_cam + self.R_base_opt @ tvec
        p_link = R_LINK_OPT @ tvec
        bearing = math.atan2(p_link[1], p_link[0])
        sides = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
        w, hgt = self.image_size
        margin = float(min(corners[:, 0].min(), corners[:, 1].min(),
                           w - corners[:, 0].max(), hgt - corners[:, 1].max())) \
            if w > 0 else 1e3
        return MarkerObservation(
            marker_id=marker_id, stamp=stamp, corners=corners, p_base=p_base, phi=phi,
            bearing=bearing, range_cam=float(tvec[2]), side_px=float(np.mean(sides)),
            reproj_err=err, ambiguity_ratio=float(ambiguity), edge_margin_px=margin,
            rvec=rvec, tvec=tvec)
