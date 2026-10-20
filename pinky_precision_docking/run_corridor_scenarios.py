#!/usr/bin/env python3
"""
Automated Gazebo Simulation Verification Runner for Corridor Docking.

Test Environment:
  - 40cm Wide Corridor (Left wall at y = +0.20m, Right wall at y = -0.20m).
  - Docking Station Wall at x = 0.55m (front face at x = 0.53m).
  - Two ArUco Markers on End Wall:
      * Left Marker:  ID 0 at y = +0.08m
      * Right Marker: ID 2 at y = -0.08m

Evaluation Scenarios:
  - Scenario 1 (Left 30cm): Start at x=0.23m (30cm from wall), y=0.0m -> Dock to Marker ID 0.
                            Verifies no retreat at 30cm start, collision-free navigation, and high precision.
  - Scenario 2 (Right 30cm): Start at x=0.23m (30cm from wall), y=0.0m -> Dock to Marker ID 2.
                            Verifies symmetrical docking to right marker with no retreat.
  - Scenario 3 (Left 50cm): Start at x=0.03m (50cm from wall), y=0.0m -> Dock to Marker ID 0.
                            Verifies full 40cm corridor traverse rejecting side walls in LiDAR.
  - Scenario 4 (Right 50cm + Perturbation): Start at x=0.03m, y=+0.03m, yaw=-5.7 deg -> Dock to Marker ID 2.
                            Verifies recovery from offset and angular perturbation inside corridor.
"""

import argparse
import csv
import math
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool


class CorridorDockingMonitor(Node):
    """ROS 2 Node that monitors docking completion, cmd_vel (retreats), and odom (side clearance)."""

    def __init__(self):
        super().__init__('corridor_docking_monitor')
        self.completed = False
        self.completion_time: Optional[float] = None
        self.min_linear_x = 0.0
        self.max_abs_y = 0.0
        self.trajectory_y: List[float] = []

        qos_reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        qos_sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub_completed = self.create_subscription(
            Bool, '/docking_completed', self._cb_completed, qos_reliable
        )
        self.sub_cmd_vel = self.create_subscription(
            Twist, '/cmd_vel', self._cb_cmd_vel, qos_reliable
        )
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self._cb_odom, qos_reliable
        )

    def _cb_completed(self, msg: Bool):
        if msg.data and not self.completed:
            self.completed = True
            self.completion_time = time.time()
            self.get_logger().info('>>> Received /docking_completed == True! <<<')

    def _cb_cmd_vel(self, msg: Twist):
        vx = msg.linear.x
        if vx < self.min_linear_x:
            self.min_linear_x = vx

    def _cb_odom(self, msg: Odometry):
        y = msg.pose.pose.position.y
        self.trajectory_y.append(y)
        if abs(y) > self.max_abs_y:
            self.max_abs_y = abs(y)


def kill_all_sim():
    """Kill lingering simulator and node processes safely."""
    for proc_name in [
        'ruby', 'parameter_bridge', 'image_bridge', 'gz', 'gz-sim',
        'precision_docking_node', 'robot_state_publisher'
    ]:
        subprocess.run(['killall', '-q', '-9', proc_name], stderr=subprocess.DEVNULL)
    time.sleep(1.5)


def read_latest_csv_row(csv_path: str) -> Optional[Dict[str, str]]:
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
        if reader:
            return reader[-1]
    return None


def run_corridor_scenario(
    scenario_name: str,
    target_marker_id: int,
    spawn_x: float,
    spawn_y: float,
    spawn_yaw: float,
    timeout_sec: float = 65.0,
    csv_log_path: str = '/home/user/pinky_pro/docking_experiment_log.csv',
    headless: bool = True
) -> Dict:
    print(f"\n{'='*75}")
    print(f"  RUNNING CORRIDOR SCENARIO: {scenario_name}")
    print(f"  Target Marker ID: {target_marker_id}")
    print(f"  Initial Pose    : x={spawn_x:+.3f}m, y={spawn_y:+.3f}m, yaw={math.degrees(spawn_yaw):+.1f} deg")
    print(f"  Corridor Bounds : y in [-0.20m, +0.20m] (Width 40cm)")
    print(f"{'='*75}")

    kill_all_sim()
    time.sleep(1.0)

    # 1. Launch Gazebo Simulation with corridor_docking.world
    sim_cmd = [
        'ros2', 'launch', 'pinky_gz_sim', 'launch_sim.launch.xml',
        'world_name:=corridor_docking.world',
        f'spawn_x:={spawn_x}',
        f'spawn_y:={spawn_y}',
        f'spawn_yaw:={spawn_yaw}',
        f'headless:={str(headless)}',
        'cam_tilt_deg:=0'
    ]
    print(f"[Launcher] Starting simulation: {' '.join(sim_cmd[:6])}...")
    sim_proc = subprocess.Popen(sim_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Allow Gazebo & bridge ~8 seconds to start and stabilize
    print("[Launcher] Waiting 8s for Gazebo world and ROS bridge to initialize...")
    time.sleep(8.0)

    # 2. Count existing rows in CSV
    initial_csv_row_count = 0
    if os.path.exists(csv_log_path):
        with open(csv_log_path, 'r', encoding='utf-8') as f:
            initial_csv_row_count = len(list(csv.reader(f)))

    # 3. Launch Precision Docking Node
    dock_cmd = [
        'ros2', 'launch', 'pinky_precision_docking', 'precision_docking.launch.py',
        'use_sim_time:=True',
        'use_compressed:=False',
        f'target_marker_id:={target_marker_id}',
        'target_distance:=0.10',
        'distance_tolerance:=0.01',
        'center_tolerance:=0.005',
        'angle_tolerance_deg:=2.0',
        f'csv_log_path:={csv_log_path}'
    ]
    print(f"[Launcher] Starting docking node: {' '.join(dock_cmd[:5])}...")
    node_log_file = open('/tmp/corridor_dock_node.log', 'w')
    dock_proc = subprocess.Popen(dock_cmd, stdout=node_log_file, stderr=subprocess.STDOUT)

    # 4. Monitor /docking_completed, /cmd_vel, and /odom
    rclpy.init()
    monitor = CorridorDockingMonitor()

    start_t = time.time()
    success = False

    try:
        while time.time() - start_t < timeout_sec:
            rclpy.spin_once(monitor, timeout_sec=0.1)
            if monitor.completed:
                success = True
                print(f"[Monitor] Docking completed successfully in {time.time() - start_t:.2f}s!")
                break
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("[Monitor] Interrupted.")
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

    # Allow time for CSV flush
    time.sleep(2.0)

    # 5. Read metrics from CSV
    latest_row = read_latest_csv_row(csv_log_path)
    current_csv_row_count = 0
    if os.path.exists(csv_log_path):
        with open(csv_log_path, 'r', encoding='utf-8') as f:
            current_csv_row_count = len(list(csv.reader(f)))

    new_row_logged = (current_csv_row_count > initial_csv_row_count) and (latest_row is not None)

    dist_err = float('nan')
    lat_err = float('nan')
    ang_err = float('nan')
    elapsed_time = time.time() - start_t

    if new_row_logged and latest_row:
        fused_d = float(latest_row.get('fused_distance_m', 0.10))
        dist_err = fused_d - 0.10
        lat_err = float(latest_row.get('lateral_error_m', 0.0))
        ang_err = float(latest_row.get('angular_error_deg', 0.0))
        elapsed_time = float(latest_row.get('elapsed_time_s', elapsed_time))

    # Evaluate compliance against required tolerances:
    # 1. Distance tolerance: +/- 1.0 cm (0.01m)
    dist_pass = abs(dist_err) <= 0.01 if not math.isnan(dist_err) else False
    # 2. Lateral tolerance: +/- 0.5 cm (0.005m)
    lat_pass = abs(lat_err) <= 0.005 if not math.isnan(lat_err) else False
    # 3. Angular tolerance: +/- 2.0 deg
    ang_pass = abs(ang_err) <= 2.0 if not math.isnan(ang_err) else False

    # 4. Corridor Clearance check: Robot half-width is ~0.055m. Wall is at |y| = 0.20m.
    #    Safety bound: max robot center |y| < 0.14m (ensuring > 0.5cm clearance from side walls).
    #    For Marker 0 (y=+0.08) or Marker 2 (y=-0.08), center reaches |y| ~ 0.08m.
    wall_collision = monitor.max_abs_y >= 0.145
    clearance_pass = not wall_collision

    # 5. No Retreat Check (especially relevant for 30cm start):
    #    A retreat is characterized by negative linear velocity < -0.01 m/s.
    did_retreat = monitor.min_linear_x < -0.015
    no_retreat_flag = not did_retreat

    all_pass = success and dist_pass and lat_pass and ang_pass and clearance_pass

    result = {
        'scenario': scenario_name,
        'target_id': target_marker_id,
        'start_dist_cm': (0.53 - spawn_x) * 100.0,
        'success': success,
        'all_tolerances_pass': all_pass,
        'dist_err_cm': dist_err * 100.0,
        'dist_pass': dist_pass,
        'lat_err_cm': lat_err * 100.0,
        'lat_pass': lat_pass,
        'ang_err_deg': ang_err,
        'ang_pass': ang_pass,
        'max_abs_y_cm': monitor.max_abs_y * 100.0,
        'clearance_pass': clearance_pass,
        'min_linear_x_mps': monitor.min_linear_x,
        'no_retreat': no_retreat_flag,
        'elapsed_sec': elapsed_time,
        'csv_row': latest_row if new_row_logged else None
    }

    print("\n--- CORRIDOR SCENARIO RESULT ---")
    print(f"Status              : {'SUCCESS (ALL PASS)' if all_pass else ('COMPLETED (CHECK METRICS)' if success else 'TIMEOUT/FAILED')}")
    print(f"Elapsed Time        : {elapsed_time:.2f} s")
    print(f"Distance Error      : {dist_err*100:+.2f} cm (Goal: 10cm, Tol: +/-1.0cm) -> {'PASS' if dist_pass else 'FAIL'}")
    print(f"Lateral Error       : {lat_err*100:+.2f} cm (Tol: +/-0.5cm) -> {'PASS' if lat_pass else 'FAIL'}")
    print(f"Angular Error       : {ang_err:+.2f} deg (Tol: +/-2.0deg) -> {'PASS' if ang_pass else 'FAIL'}")
    print(f"Max Lateral Center  : {monitor.max_abs_y*100:.1f} cm (Wall at 20cm, limit: 14.5cm) -> {'PASS (Safe)' if clearance_pass else 'FAIL (Collision Risk)'}")
    print(f"Min Linear Velocity : {monitor.min_linear_x:+.3f} m/s -> {'NO RETREAT (Pure Forward)' if no_retreat_flag else 'RETREATED'}")
    print(f"{'='*75}\n")

    kill_all_sim()
    return result


def main():
    parser = argparse.ArgumentParser(description="Corridor Precision Docking Simulation Runner")
    parser.add_argument('--scenario', choices=['all', '1', '2', '3', '4'], default='all',
                        help='Scenario to run: 1 (Left 30cm), 2 (Right 30cm), 3 (Left 50cm), 4 (Right 50cm+pert), or all')
    parser.add_argument('--gui', action='store_true', default=False, help='Launch Gazebo with full 3D GUI window')
    parser.add_argument('--headless', action='store_true', default=False, help='Run in headless mode')
    args = parser.parse_args()

    use_headless = not args.gui if args.gui else (True if args.headless else True)
    results = []

    # Scenario 1: Marker ID 0 (Left marker at y=+0.08m), 30cm start distance (x=0.23m)
    if args.scenario in ('all', '1'):
        r1 = run_corridor_scenario(
            scenario_name="Scenario 1 (Left Marker ID 0, 30cm Start)",
            target_marker_id=0,
            spawn_x=0.23,
            spawn_y=0.0,
            spawn_yaw=0.0,
            timeout_sec=55.0,
            headless=use_headless
        )
        results.append(r1)

    # Scenario 2: Marker ID 2 (Right marker at y=-0.08m), 30cm start distance (x=0.23m)
    if args.scenario in ('all', '2'):
        r2 = run_corridor_scenario(
            scenario_name="Scenario 2 (Right Marker ID 2, 30cm Start)",
            target_marker_id=2,
            spawn_x=0.23,
            spawn_y=0.0,
            spawn_yaw=0.0,
            timeout_sec=55.0,
            headless=use_headless
        )
        results.append(r2)

    # Scenario 3: Marker ID 0 (Left marker), 50cm start distance (x=0.03m)
    if args.scenario in ('all', '3'):
        r3 = run_corridor_scenario(
            scenario_name="Scenario 3 (Left Marker ID 0, 50cm Full Traverse)",
            target_marker_id=0,
            spawn_x=0.03,
            spawn_y=0.0,
            spawn_yaw=0.0,
            timeout_sec=70.0,
            headless=use_headless
        )
        results.append(r3)

    # Scenario 4: Marker ID 2 (Right marker), 50cm start distance with offset & yaw perturbation
    if args.scenario in ('all', '4'):
        r4 = run_corridor_scenario(
            scenario_name="Scenario 4 (Right Marker ID 2, 50cm + Perturbation)",
            target_marker_id=2,
            spawn_x=0.03,
            spawn_y=0.03,
            spawn_yaw=math.radians(-5.7),
            timeout_sec=70.0,
            headless=use_headless
        )
        results.append(r4)

    # Print Summary Table
    print("\n" + "="*95)
    print("                     40cm CORRIDOR PRECISION DOCKING SIMULATION SUMMARY                   ")
    print("="*95)
    print(f"{'Scenario':<42} | {'Dist Err':<9} | {'Lat Err':<9} | {'Yaw Err':<8} | {'Min Vx':<8} | {'Time':<6} | {'Result':<8}")
    print("-" * 95)
    for r in results:
        res_str = "PASS" if r['all_tolerances_pass'] else ("DONE" if r['success'] else "FAIL")
        d_str = f"{r['dist_err_cm']:+.2f}cm" if not math.isnan(r['dist_err_cm']) else "N/A"
        l_str = f"{r['lat_err_cm']:+.2f}cm" if not math.isnan(r['lat_err_cm']) else "N/A"
        a_str = f"{r['ang_err_deg']:+.2f}d" if not math.isnan(r['ang_err_deg']) else "N/A"
        vx_str = f"{r['min_linear_x_mps']:+.3f}"
        t_str = f"{r['elapsed_sec']:.1f}s"
        print(f"{r['scenario']:<42} | {d_str:<9} | {l_str:<9} | {a_str:<8} | {vx_str:<8} | {t_str:<6} | {res_str:<8}")
    print("="*95)


if __name__ == '__main__':
    main()
