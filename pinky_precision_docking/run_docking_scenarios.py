#!/usr/bin/env python3
"""
Automated Simulation Verification Runner for Precision Docking.

Executes 3 Evaluation Scenarios in Gazebo:
  - Scenario A (Straight Entry): Centered approach to Marker ID 2 (y=0.0m) from ~50cm.
  - Scenario B (Lateral Bias Entry): Offset start (20cm+ lateral bias) testing Pre-docking Intercept & Fine Alignment.
  - Scenario C (Distorted Heading Entry): Oblique start (heading twisted ~18 deg) testing visual tracking & heading recovery.

Logs final metrics [distance_err, lateral_err, angular_err, elapsed_time, success] and generates report.
"""

import argparse
import csv
from datetime import datetime
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Bool


class DockingCompletionMonitor(Node):
    """ROS 2 Node that monitors /docking_completed."""

    def __init__(self):
        super().__init__('docking_completion_monitor')
        self.completed = False
        self.completion_time: Optional[float] = None
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.sub = self.create_subscription(Bool, '/docking_completed', self._cb, qos)

    def _cb(self, msg: Bool):
        if msg.data and not self.completed:
            self.completed = True
            self.completion_time = time.time()
            self.get_logger().info('>>> Received /docking_completed == True! <<<')


def kill_all_sim():
    """Kill lingering simulator and node processes safely."""
    for proc_name in ['ruby', 'parameter_bridge', 'image_bridge', 'gz', 'gz-sim', 'precision_docking_node', 'robot_state_publisher']:
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


def run_scenario(
    scenario_name: str,
    target_marker_id: int,
    spawn_x: float,
    spawn_y: float,
    spawn_yaw: float,
    timeout_sec: float = 65.0,
    csv_log_path: str = '/home/user/pinky_pro/docking_experiment_log.csv',
    headless: bool = True
) -> Dict:
    print(f"\n{'='*70}")
    print(f"  RUNNING {scenario_name}")
    print(f"  Target Marker ID: {target_marker_id}")
    print(f"  Initial Pose    : x={spawn_x:+.3f}m, y={spawn_y:+.3f}m, yaw={math.degrees(spawn_yaw):+.1f} deg")
    print(f"{'='*70}")

    kill_all_sim()
    time.sleep(1.0)

    # 1. Launch Gazebo Simulation
    sim_cmd = [
        'ros2', 'launch', 'pinky_gz_sim', 'launch_sim.launch.xml',
        'world_name:=docking_test.world',
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

    # 2. Launch Precision Docking Node
    initial_csv_row_count = 0
    if os.path.exists(csv_log_path):
        with open(csv_log_path, 'r', encoding='utf-8') as f:
            initial_csv_row_count = len(list(csv.reader(f)))

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
    dock_proc = subprocess.Popen(dock_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 3. Monitor /docking_completed topic
    rclpy.init()
    monitor = DockingCompletionMonitor()

    start_t = time.time()
    success = False

    try:
        while time.time() - start_t < timeout_sec:
            rclpy.spin_once(monitor, timeout_sec=0.2)
            if monitor.completed:
                success = True
                print(f"[Monitor] Docking completed successfully in {time.time() - start_t:.2f}s!")
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("[Monitor] Interrupted.")
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

    # Allow time for CSV flush
    time.sleep(2.0)

    # 4. Read metrics from CSV
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

    # Evaluate compliance against required tolerances
    # Required: 거리 1cm (0.01m), 좌우 0.5cm (0.005m), 각도 2° (2.0 deg)
    dist_pass = abs(dist_err) <= 0.01 if not math.isnan(dist_err) else False
    lat_pass = abs(lat_err) <= 0.005 if not math.isnan(lat_err) else False
    ang_pass = abs(ang_err) <= 2.0 if not math.isnan(ang_err) else False
    all_pass = success and dist_pass and lat_pass and ang_pass

    result = {
        'scenario': scenario_name,
        'target_id': target_marker_id,
        'success': success,
        'all_tolerances_pass': all_pass,
        'dist_err_cm': dist_err * 100.0,
        'dist_pass': dist_pass,
        'lat_err_cm': lat_err * 100.0,
        'lat_pass': lat_pass,
        'ang_err_deg': ang_err,
        'ang_pass': ang_pass,
        'elapsed_sec': elapsed_time,
        'csv_row': latest_row if new_row_logged else None
    }

    print("\n--- SCENARIO RESULT ---")
    print(f"Status           : {'SUCCESS (ALL PASS)' if all_pass else ('COMPLETED (CHECK TOL)' if success else 'TIMEOUT/FAILED')}")
    print(f"Elapsed Time     : {elapsed_time:.2f} s")
    print(f"Distance Error   : {dist_err*100:+.2f} cm (Goal: 10cm, Tol: +/-1.0cm) -> {'PASS' if dist_pass else 'FAIL'}")
    print(f"Lateral Error    : {lat_err*100:+.2f} cm (Tol: +/-1.5cm) -> {'PASS' if lat_pass else 'FAIL'}")
    print(f"Angular Error    : {ang_err:+.2f} deg (Tol: +/-2.0deg) -> {'PASS' if ang_pass else 'FAIL'}")
    print(f"{'='*70}\n")

    # Cleanup processes
    kill_all_sim()
    return result


def main():
    parser = argparse.ArgumentParser(description="Precision Auto-Docking Simulation Runner")
    parser.add_argument('--scenario', choices=['all', 'A', 'B', 'C', 'random'], default='all', help='Scenario to execute')
    parser.add_argument('--gui', action='store_true', default=False, help='Launch Gazebo with full 3D GUI window')
    parser.add_argument('--headless', action='store_true', default=False, help='Run in headless mode (no GUI window)')
    parser.add_argument('--rqt', action='store_true', default=False, help='Automatically open rqt_image_view for /docking/debug_image')
    parser.add_argument('--trials', type=int, default=5, help='Number of random trials for scenario=random (default: 5)')
    parser.add_argument('--radius', type=float, default=0.05, help='Position error radius in meters (default: 0.05m = 5cm)')
    parser.add_argument('--yaw-noise-deg', type=float, default=10.0, help='Heading error in degrees (+/- deg, default: 10.0)')
    parser.add_argument('--target-id', type=int, default=2, help='Target ArUco marker ID (default: 2)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()

    # If --gui is specified, headless is False. Default is True unless --gui is given.
    use_headless = not args.gui if args.gui else (True if args.headless else True)

    # Optional auto-launch of rqt_image_view
    rqt_proc = None
    if args.rqt:
        print("[Launcher] Starting rqt_image_view on /docking/debug_image...")
        rqt_proc = subprocess.Popen(
            ['ros2', 'run', 'rqt_image_view', 'rqt_image_view', '/docking/debug_image'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    results = []

    try:
        # Scenario A: Straight Approach to ID 2 (y=0.0m)
        if args.scenario in ('all', 'A'):
            res_a = run_scenario(
                scenario_name="Scenario A (Straight Approach)",
                target_marker_id=2,
                spawn_x=0.0,
                spawn_y=0.0,
                spawn_yaw=0.0,
                timeout_sec=50.0,
                headless=use_headless
            )
            results.append(res_a)

        # Scenario B: Lateral Offset (20cm bias) Approach to ID 0 (y=+0.20m)
        if args.scenario in ('all', 'B'):
            res_b = run_scenario(
                scenario_name="Scenario B (Lateral Bias ~20cm Offset)",
                target_marker_id=0,
                spawn_x=0.0,
                spawn_y=0.0,  # robot starts at y=0, marker ID 0 is at y=+0.20m (20cm offset)
                spawn_yaw=0.0,
                timeout_sec=75.0,
                headless=use_headless
            )
            results.append(res_b)

        # Scenario C: Distorted Heading Approach (~18 deg yaw twist)
        if args.scenario in ('all', 'C'):
            res_c = run_scenario(
                scenario_name="Scenario C (Distorted Heading ~18 deg)",
                target_marker_id=2,
                spawn_x=0.0,
                spawn_y=-0.05,
                spawn_yaw=math.radians(18.0),  # heading twisted by 18 deg
                timeout_sec=65.0,
                headless=use_headless
            )
            results.append(res_c)

        # Scenario Random: Monte-Carlo Nav2 Waypoint Arrival Noise
        if args.scenario == 'random':
            if args.seed is not None:
                random.seed(args.seed)
            print(f"\n[Monte-Carlo] Starting {args.trials} Randomized Nav2 Waypoint Arrival Trials:")
            print(f"  Radius        : {args.radius * 100:.1f} cm disk")
            print(f"  Heading Noise : +/- {args.yaw_noise_deg:.1f} deg")
            print(f"  Target Marker : ID {args.target_id}")

            for i in range(args.trials):
                # Uniform disk sampling: r = R * sqrt(u), phi in [0, 2pi)
                u = random.random()
                r = args.radius * math.sqrt(u)
                phi = random.uniform(0.0, 2.0 * math.pi)
                dx = r * math.cos(phi)
                dy = r * math.sin(phi)
                dyaw_deg = random.uniform(-args.yaw_noise_deg, args.yaw_noise_deg)
                dyaw = math.radians(dyaw_deg)

                trial_name = f"Trial #{i+1:02d} (r={r*100:4.1f}cm, yaw={dyaw_deg:+5.1f}d)"
                res = run_scenario(
                    scenario_name=trial_name,
                    target_marker_id=args.target_id,
                    spawn_x=dx,
                    spawn_y=dy,
                    spawn_yaw=dyaw,
                    timeout_sec=65.0,
                    headless=use_headless
                )
                results.append(res)
    finally:
        if rqt_proc is not None:
            rqt_proc.terminate()

    print("\n" + "="*80)
    print("                     PRECISION DOCKING SIMULATION SUMMARY                   ")
    print("="*80)
    print(f"{'Scenario':<38} | {'Dist Err':<10} | {'Lat Err':<10} | {'Yaw Err':<10} | {'Time':<8} | {'Result':<10}")
    print("-" * 80)
    for r in results:
        res_str = "PASS" if r['all_tolerances_pass'] else ("DONE" if r['success'] else "FAIL")
        d_str = f"{r['dist_err_cm']:+.2f}cm" if not math.isnan(r['dist_err_cm']) else "N/A"
        l_str = f"{r['lat_err_cm']:+.2f}cm" if not math.isnan(r['lat_err_cm']) else "N/A"
        a_str = f"{r['ang_err_deg']:+.2f}d" if not math.isnan(r['ang_err_deg']) else "N/A"
        t_str = f"{r['elapsed_sec']:.1f}s"
        print(f"{r['scenario']:<38} | {d_str:<10} | {l_str:<10} | {a_str:<10} | {t_str:<8} | {res_str:<10}")
    print("="*80)

    # Monte-Carlo statistical summary
    if args.scenario == 'random' and results:
        pass_count = sum(1 for r in results if r['all_tolerances_pass'])
        success_rate = (pass_count / len(results)) * 100.0
        dist_errs = [abs(r['dist_err_cm']) for r in results if not math.isnan(r['dist_err_cm'])]
        lat_errs = [abs(r['lat_err_cm']) for r in results if not math.isnan(r['lat_err_cm'])]
        ang_errs = [abs(r['ang_err_deg']) for r in results if not math.isnan(r['ang_err_deg'])]
        times = [r['elapsed_sec'] for r in results if r['success']]

        print(f"\n[Monte-Carlo Statistics (Total Trials: {len(results)})]")
        print(f"  Success Rate (All Tolerances Met) : {pass_count}/{len(results)} ({success_rate:.1f}%)")
        if dist_errs:
            print(f"  Distance Error (Limit: 1.0 cm)    : Mean = {sum(dist_errs)/len(dist_errs):.2f} cm, Max = {max(dist_errs):.2f} cm")
        if lat_errs:
            print(f"  Lateral Error  (Limit: 1.5 cm)    : Mean = {sum(lat_errs)/len(lat_errs):.2f} cm, Max = {max(lat_errs):.2f} cm")
        if ang_errs:
            print(f"  Angular Error  (Limit: 2.0 deg)   : Mean = {sum(ang_errs)/len(ang_errs):.2f} deg, Max = {max(ang_errs):.2f} deg")
        if times:
            print(f"  Docking Time                      : Mean = {sum(times)/len(times):.1f} s, Max = {max(times):.1f} s")
        print("="*80 + "\n")


if __name__ == '__main__':
    main()
