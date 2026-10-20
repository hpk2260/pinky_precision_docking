# Pinky Precision Docking (v5.0)

[![ROS 2](https://img.shields.io/badge/ROS%202-Jazzy%20%7C%20Humble-blue.svg)](https://docs.ros.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Precision](https://img.shields.io/badge/Precision-%3C3.3mm%20%7C%20%3C0.11%C2%B0-success.svg)]()

High-precision autonomous docking system for differential drive mobile robots (**Pinky Pro**) fusing monocular vision (**ArUco DICT_5X5_250**) and 2D LiDAR (**LaserScan**).

Achieves sub-centimeter distance accuracy ($< 4.5\text{ mm}$), millimeter-level lateral alignment ($< 3.3\text{ mm}$), and ultra-precise wall parallelism ($< 0.11^\circ$), even inside a narrow **40 cm corridor** with multiple ArUco targets.

---

## 🎯 Key Capabilities & Architecture (v5.0)

```
[Monocular Camera] ---> [ArUco Detector (IPPE_SQUARE)] ---\
                                                            +---> [Docking Estimator (Kalman Filter)]
[2D RPLIDAR Scan]   ---> [Wall Fitter (RANSAC + TLS)]    ---/         | (World-fixed Static Anchor)
                                                                      v
                                                            [Staged Docking Controller]
                                                                      |
                                                                      +---> [cmd_vel (Twist)]
```

1. **Branch A: Camera Detector (`marker_detector.py`)**
   - OpenCV `SOLVEPNP_IPPE_SQUARE` for planar square markers with camera extrinsics calibration.
   - Ambiguity resolution using LiDAR wall-normal guidance (`phi_hint`).
   - Auto-lockon to closest target marker or designated ID selection.

2. **Branch B: LiDAR Wall Fitter (`wall_fitting.py`)**
   - Front 2D LaserScan point extraction within adaptive ROI ($\pm 65^\circ$).
   - Camera-guided RANSAC plane fitting with `phi_hint` side-wall rejection (prevents locking onto corridor side walls).
   - TLS (Total Least Squares) line refinement for sub-millimeter distance and $< 0.1^\circ$ normal angle estimation.

3. **Sensor Fusion Estimator (`estimator.py`)**
   - World-fixed static docking anchor $(M_x, M_y, \theta_n)$ tracked via scalar Kalman filters in the `/odom` frame.
   - Robust Mahalanobis gating and rejection of orthogonal / perpendicular wall readings ($> 28^\circ$).
   - Latency compensation via odom history buffer interpolation.

4. **Staged Motion Controller (`controller.py`)**
   - **Phase 1: Pre-docking Intercept (`INTERCEPT_TURN` & `INTERCEPT_DRIVE`)**: FOV-constrained deviated pursuit with adaptive staging distance. At 30cm start distances, immediately navigates forward without reversing.
   - **Phase 2: Closed-Loop Verification (`VERIFY`)**: Verifies alignment against tight lateral thresholds.
   - **Phase 3: Terminal Straight Approach (`TERMINAL`)**: Parallel 11-shape entry to 10cm stopping distance.
   - **Phase 4: Wall Parallel Alignment (`WALL_ALIGN`)**: In-place rotation to guarantee bumper-wall parallelism ($\le 0.5^\circ$).

5. **Corridor Navigation & Side-Wall Safety**
   - Navigates through 40cm narrow corridors (20cm clearance on each side) with zero side-wall collisions and $> 6.5\text{ cm}$ physical margin at all times.

---

## 📊 Evaluation & Verification Results

Tested and verified in Gazebo Sim across diverse challenging scenarios:

| Scenario | Start Pose ($x, y, \text{yaw}$) | Target Marker | Distance Error (Goal: 10cm, Tol: $\pm 1.0\text{cm}$) | Lateral Error (Tol: $\pm 0.5\text{cm}$) | Angular Error (Tol: $\pm 2.0^\circ$) | Min $v_x$ (Retreat Guard) | Elapsed Time | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. 30cm Corridor Start (Left)** | $x=0.23\text{m}, y=0.0\text{m}, 0^\circ$ | **ID 0** | **+0.44 cm** | **-0.30 cm** | **+0.04°** | **0.000 m/s (No retreat)** | 10.49 s | **PASS** |
| **2. 30cm Corridor Start (Right)** | $x=0.23\text{m}, y=0.0\text{m}, 0^\circ$ | **ID 2** | **+0.36 cm** | **+0.33 cm** | **-0.04°** | **0.000 m/s (No retreat)** | 10.45 s | **PASS** |
| **3. 50cm Full Corridor Traverse** | $x=0.03\text{m}, y=0.0\text{m}, 0^\circ$ | **ID 0** | **+0.44 cm** | **-0.12 cm** | **+0.11°** | **0.000 m/s** | 11.36 s | **PASS** |
| **4. 50cm + Initial Perturbation** | $x=0.03\text{m}, y=+0.03\text{m}, -5.7^\circ$ | **ID 2** | **+0.41 cm** | **+0.18 cm** | **-0.09°** | **0.000 m/s** | 11.66 s | **PASS** |

---

## 🚀 Quick Start

### 1. Build Package
```bash
cd ~/pinky_pro
colcon build --packages-select pinky_precision_docking --symlink-install
source install/setup.bash
```

### 2. Run Autonomous Docking on Physical Robot
```bash
ros2 launch pinky_precision_docking precision_docking.launch.py \
    target_marker_id:=0 \
    target_distance:=0.10 \
    distance_tolerance:=0.01 \
    center_tolerance:=0.005
```

### 3. Run Automated Gazebo Simulation Benchmark
```bash
# Run 40cm corridor multi-marker automated evaluation
python3 src/pinky_precision_docking/pinky_precision_docking/run_corridor_scenarios.py --scenario all

# Or run standard multi-scenario benchmark
python3 src/pinky_precision_docking/pinky_precision_docking/run_docking_scenarios.py --scenario all
```

---

## ⚙️ Parameters (`config/docking_params.yaml`)

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `target_marker_id` | int | `-1` | Target ArUco marker ID (`-1` for auto-lockon) |
| `marker_size` | float | `0.06` | Physical size of the ArUco marker in meters |
| `aruco_dictionary_id` | str | `DICT_5X5_250` | ArUco dictionary identifier |
| `target_distance` | float | `0.10` | Final docking target distance (10 cm) |
| `distance_tolerance` | float | `0.01` | Final distance error tolerance ($\pm 1.0\text{ cm}$) |
| `center_tolerance` | float | `0.005` | Final lateral error tolerance ($\pm 5\text{ mm}$) |
| `angle_tolerance_deg` | float | `2.0` | Normal angle tolerance ($\pm 2.0^\circ$) |
| `wall_align_tol_deg` | float | `0.5` | Precise wall parallel alignment tolerance ($\pm 0.5^\circ$) |
| `allow_retreat` | bool | `true` | Allows retreat during close-wall recovery, capped by start position |
| `osd_bg_alpha` | float | `0.35` | Transparency of debug camera overlay banner |
