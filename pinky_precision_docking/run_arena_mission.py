#!/usr/bin/env python3
"""
Full Autonomous Arena Mission Runner for Pinky Pro.

Scenario Sequence:
  Step 1: Nav2 from Home Parking -> Shelf Spot 1 Pre-docking ([0.5990, 1.4700, 3.1090])
  Step 2: Precision Docking to ArUco Marker ID 7 (Shelf 1)
  Step 3: Undock backup -> Nav2 to Shelf Spot 2 Pre-docking ([1.0210, 1.5320, -1.5680])
  Step 4: Precision Docking to ArUco Marker ID 4 (Shelf 2)
  Step 5: Undock backup -> Nav2 to Packaging Area Pre-docking ([0.7240, 0.2430, 0.1070])
  Step 6: Precision Docking to ArUco Marker ID 5 (Packaging / OMX)
  Step 7: Undock backup -> Nav2 Return Home to Parking ([0.0000, 0.0000, 0.0000])
"""

import argparse
import json
import math
import subprocess
import sys
import threading
import time
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from std_msgs.msg import Bool, Int32, String
from std_srvs.srv import Trigger

from pinky_precision_docking.sound_effects import (
    play_docking_complete,
    start_reverse_melody,
    stop_reverse_melody,
    get_sound_manager,
)


# Arena Waypoints [x(m), y(m), yaw(rad)]
WAYPOINTS = {
    'home': [0.1200, 0.0000, 0.0000],
    'shelf1': [0.6000, 1.4700, 3.1416],  # Shelf 1 Area Representative Pre-docking (Views all 3 bays: Left=6, Center=7, Right=8)
    'shelf2': [1.1100, 1.5000, -1.5708], # Shelf 2 Area Representative Pre-docking (Centered in Corridor 2: Left=3, Right=4)
    'packaging': [0.7240, 0.2430, 0.1070], # Packaging Area Pre-docking (Marker 5)
}

# Complete Arena Mission Stations: All 6 ArUco Markers across Area 1, 2, and 3
ALL_STATIONS = [
    # Area 1: Shelf 1 (Industrial Pallet Racks - 3 Bays)
    {
        'area': 'Shelf 1',
        'station_id': 'shelf1_bay1',
        'station_name': 'Shelf 1 Left Bay (선반 1번 좌측 베이, ID 6)',
        'marker_id': 6,
        'robot_name': '두산 로봇팔 (Doosan Collaborative Robot)',
        'task_type': '부품 A Pick & Place (피킹 및 핑키 적재)',
        'pre_docking': 'shelf1',
        'undock_dist': 0.50,
    },
    {
        'area': 'Shelf 1',
        'station_id': 'shelf1_bay2',
        'station_name': 'Shelf 1 Center Bay (선반 1번 중앙 베이, ID 7)',
        'marker_id': 7,
        'robot_name': '두산 로봇팔 (Doosan Collaborative Robot)',
        'task_type': '부품 B Pick & Place (피킹 및 핑키 적재)',
        'pre_docking': 'shelf1',
        'undock_dist': 0.50,
    },
    {
        'area': 'Shelf 1',
        'station_id': 'shelf1_bay3',
        'station_name': 'Shelf 1 Right Bay (선반 1번 우측 베이, ID 8)',
        'marker_id': 8,
        'robot_name': '두산 로봇팔 (Doosan Collaborative Robot)',
        'task_type': '부품 C Pick & Place (피킹 및 핑키 적재)',
        'pre_docking': 'shelf1',
        'undock_dist': 0.55,
    },
    # Area 2: Shelf 2 (Coolers & Refrigeration - 2 Coolers)
    {
        'area': 'Shelf 2',
        'station_id': 'shelf2_cooler1',
        'station_name': 'Shelf 2 Left Cooler (선반 2번 좌측 냉장고, ID 3)',
        'marker_id': 3,
        'robot_name': '두산 로봇팔 (Doosan Collaborative Robot)',
        'task_type': '냉장 부품 A Pick & Place (피킹 및 핑키 적재)',
        'pre_docking': 'shelf2',
        'undock_dist': 0.60,
    },
    {
        'area': 'Shelf 2',
        'station_id': 'shelf2_cooler2',
        'station_name': 'Shelf 2 Right Cooler (선반 2번 우측 냉장고, ID 4)',
        'marker_id': 4,
        'robot_name': '두산 로봇팔 (Doosan Collaborative Robot)',
        'task_type': '냉장 부품 B Pick & Place (피킹 및 핑키 적재)',
        'pre_docking': 'shelf2',
        'undock_dist': 0.65,
    },
    # Area 3: Packaging Station (ROBOTIS OpenMANIPULATOR-X)
    {
        'area': 'Packaging Station',
        'station_id': 'packaging',
        'station_name': 'Packaging Station (포장 구역, ID 5)',
        'marker_id': 5,
        'robot_name': 'OpenMANIPULATOR-X (OMX 매니퓰레이터)',
        'task_type': '적재물 전체 하역 및 자동 포장 (Packaging)',
        'pre_docking': 'packaging',
        'undock_dist': 0.35,
    },
]

# Legacy Sample Stations (1 per area for quick test)
SAMPLE_STATIONS = [
    ALL_STATIONS[2],  # Shelf 1 Right (ID 8)
    ALL_STATIONS[3],  # Shelf 2 Left (ID 3)
    ALL_STATIONS[5],  # Packaging (ID 5)
]


class ArenaMissionController:
    def __init__(self, enable_sound: bool = True, piezo_pin: int = None, use_sim_time: bool = False):
        self.enable_sound = enable_sound
        if self.enable_sound:
            self.sound_mgr = get_sound_manager()
            if piezo_pin is not None:
                self.sound_mgr.piezo_pin = piezo_pin
                self.sound_mgr._init_gpio_piezo()

        self.nav = BasicNavigator()
        if use_sim_time:
            self.nav.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        self.docking_completed = False

        # Subscriber for /docking_completed
        self.sub_dock = self.nav.create_subscription(
            Bool, '/docking_completed', self._docking_completed_cb, 10
        )
        self.cmd_vel_pub = self.nav.create_publisher(Twist, '/cmd_vel', 10)
        self.target_id_pub = self.nav.create_publisher(Int32, '/docking/target_marker_id', 10)
        self.buzzer_cmd_pub = self.nav.create_publisher(String, '/pinky/buzzer/command', 10)
        self.start_dock_cli = self.nav.create_client(Trigger, '/docking/start')

        # Inter-robot external manipulator task integration
        self.task_trigger_pub = self.nav.create_publisher(String, '/pinky/station_task/trigger', 10)
        self.task_complete_sub = self.nav.create_subscription(
            String, '/pinky/station_task/complete', self._station_task_complete_cb, 10
        )
        self.external_task_completed = False
        self.external_task_msg = ""

    def _station_task_complete_cb(self, msg: String):
        data = msg.data.strip()
        if data:
            self.external_task_completed = True
            self.external_task_msg = data

    def _trigger_buzzer(self, command: str):
        """Send sound command to real Pinky Pro hardware buzzer node over ROS 2."""
        if not self.enable_sound:
            return
        msg = String()
        msg.data = str(command)
        for _ in range(2):
            self.buzzer_cmd_pub.publish(msg)
            time.sleep(0.005)

    def _docking_completed_cb(self, msg: Bool):
        if msg.data:
            self.docking_completed = True

    def set_docking_marker_id(self, marker_id: int):
        """Set target_marker_id instantly via ROS topic (<5ms, zero subprocess delay)."""
        print(f"[Mission] Setting target_marker_id = {marker_id} via topic...")
        msg = Int32()
        msg.data = int(marker_id)
        for _ in range(3):
            self.target_id_pub.publish(msg)
            time.sleep(0.01)
        print(f"[Mission] Target ArUco Marker ID set to {marker_id} successfully.")

    def navigate_to(self, waypoint_name: str, x: float, y: float, yaw: float, timeout_sec: float = 180.0) -> bool:
        """Navigate to waypoint using Nav2 BasicNavigator with simulation-aware progress logging."""
        print(f"\n>>> [Nav2] Navigating to '{waypoint_name}': x={x:.3f}m, y={y:.3f}m, yaw={math.degrees(yaw):+.1f} deg (timeout: {timeout_sec:.0f}s)...")

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.nav.get_clock().now().to_msg()
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        goal.pose.position.z = 0.0
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)

        self.nav.goToPose(goal)

        start_clk = self.nav.get_clock().now()
        start_wall_t = time.time()
        last_log_wall_t = start_wall_t

        while not self.nav.isTaskComplete():
            feedback = self.nav.getFeedback()
            now_wall_t = time.time()
            now_clk = self.nav.get_clock().now()

            # Robust elapsed time calculation (handles uninitialized/zero clock when use_sim_time=True)
            if start_clk.nanoseconds > 0 and now_clk.nanoseconds >= start_clk.nanoseconds:
                elapsed_sim = (now_clk - start_clk).nanoseconds / 1e9
            else:
                elapsed_sim = now_wall_t - start_wall_t

            elapsed_wall = now_wall_t - start_wall_t

            # Periodic progress log every 3 seconds
            if now_wall_t - last_log_wall_t >= 3.0:
                last_log_wall_t = now_wall_t
                dist_str = f"{feedback.distance_remaining:.2f}m" if feedback else "measuring..."
                print(f"  [Nav2 Progress] To '{waypoint_name}' | Remaining: {dist_str} | Elapsed: {elapsed_sim:.1f}s (sim) / {elapsed_wall:.1f}s (wall)")

            time.sleep(0.1)

            # Timeout check against wall clock to guarantee safety against sim clock anomalies
            if elapsed_wall > timeout_sec:
                print(f"[Nav2 Error] Navigation to {waypoint_name} timed out ({timeout_sec:.0f}s wall time, {elapsed_sim:.1f}s sim time)!")
                self.nav.cancelTask()
                return False

        res = self.nav.getResult()
        elapsed_total = (self.nav.get_clock().now() - start_clk).nanoseconds / 1e9
        if res == TaskResult.SUCCEEDED:
            print(f"[Nav2] Successfully arrived at '{waypoint_name}' in {elapsed_total:.1f}s (sim).")
            return True
        else:
            print(f"[Nav2 Error] Navigation to '{waypoint_name}' failed with result: {res}")
            return False

    def execute_docking(self, station_name: str, marker_id: int, timeout_sec: float = 120.0) -> bool:
        """Execute precision docking to designated ArUco marker."""
        print(f"\n>>> [Docking] Starting Precision Docking at '{station_name}' to Marker ID {marker_id} (timeout: {timeout_sec:.0f}s)...")
        self.set_docking_marker_id(marker_id)
        time.sleep(0.1)

        if not self.start_dock_cli.wait_for_service(timeout_sec=5.0):
            print("[Docking Error] /docking/start service not available! Is precision_docking_node running?")
            return False

        self.docking_completed = False
        req = Trigger.Request()
        future = self.start_dock_cli.call_async(req)
        rclpy.spin_until_future_complete(self.nav, future, timeout_sec=2.0)
        if future.result():
            print(f"[Docking] Trigger response: {future.result().message}")

        start_clk = self.nav.get_clock().now()
        start_wall_t = time.time()
        last_log_wall_t = start_wall_t

        while not self.docking_completed:
            rclpy.spin_once(self.nav, timeout_sec=0.05)
            time.sleep(0.02)
            now_clk = self.nav.get_clock().now()
            now_wall_t = time.time()
            elapsed_wall = now_wall_t - start_wall_t
            if start_clk.nanoseconds > 0 and now_clk.nanoseconds >= start_clk.nanoseconds:
                elapsed_sim = (now_clk - start_clk).nanoseconds / 1e9
            else:
                elapsed_sim = elapsed_wall

            if now_wall_t - last_log_wall_t >= 3.0:
                last_log_wall_t = now_wall_t
                print(f"  [Docking Progress] Docking at '{station_name}' in progress... ({elapsed_sim:.1f}s sim)")

            if elapsed_wall > timeout_sec:
                print(f"[Docking Error] Docking at '{station_name}' timed out ({timeout_sec:.0f}s wall time)!")
                return False

        elapsed_total = (self.nav.get_clock().now() - start_clk).nanoseconds / 1e9
        print(f"[Docking] Precision Docking completed successfully at '{station_name}' in {elapsed_total:.1f}s (sim)!")
        if self.enable_sound:
            self._trigger_buzzer('docking_complete')
            play_docking_complete(blocking=False)
        return True

    def undock_backup(self, distance_m: float = 0.20, speed: float = -0.04, timeout_sec: float = 15.0):
        """
        Safely back up straight using Odometry closed-loop feedback.
        Measures actual euclidean distance traveled to prevent overshooting caused by simulation latency/RTF.
        """
        print(f"[Undock] Closed-loop backing up target: {distance_m*100:.1f}cm straight at {speed:.2f}m/s...")
        if self.enable_sound:
            self._trigger_buzzer('reverse_start')
            start_reverse_melody()

        start_pos = None
        current_pos = [0.0, 0.0]

        def _odom_cb(msg: Odometry):
            nonlocal start_pos, current_pos
            px = msg.pose.pose.position.x
            py = msg.pose.pose.position.y
            current_pos = [px, py]
            if start_pos is None:
                start_pos = [px, py]

        # Temporary subscription to /odom for precision distance tracking
        odom_sub = self.nav.create_subscription(Odometry, '/odom', _odom_cb, 10)

        # Wait for first valid odometry pose (up to 1.5s)
        wait_start = time.time()
        while start_pos is None and (time.time() - wait_start) < 1.5:
            rclpy.spin_once(self.nav, timeout_sec=0.05)

        start_clk = self.nav.get_clock().now()
        start_wall_t = time.time()
        last_log_wall_t = start_wall_t

        try:
            msg = Twist()
            msg.linear.x = float(speed)

            while rclpy.ok():
                rclpy.spin_once(self.nav, timeout_sec=0.02)

                # Calculate actual euclidean distance traveled from start
                if start_pos is not None:
                    dx = current_pos[0] - start_pos[0]
                    dy = current_pos[1] - start_pos[1]
                    dist_traveled = math.hypot(dx, dy)
                else:
                    dist_traveled = 0.0

                now_wall_t = time.time()
                if now_wall_t - last_log_wall_t >= 1.0:
                    last_log_wall_t = now_wall_t
                    print(f"  [Undock Progress] Traveled: {dist_traveled*100:.1f}cm / {distance_m*100:.1f}cm")

                # Check if exact target distance reached
                if dist_traveled >= distance_m:
                    print(f"[Undock] Target distance reached! Traveled: {dist_traveled*100:.1f}cm.")
                    break

                # Safety timeout check (wall clock)
                elapsed_wall = time.time() - start_wall_t
                if elapsed_wall > timeout_sec:
                    print(f"[Undock Warning] Safety timeout reached ({timeout_sec:.1f}s). Traveled: {dist_traveled*100:.1f}cm.")
                    break

                self.cmd_vel_pub.publish(msg)
                time.sleep(0.02)

            # Active braking to prevent inertia drift
            stop_msg = Twist()
            for _ in range(5):
                self.cmd_vel_pub.publish(stop_msg)
                time.sleep(0.05)

        finally:
            self.nav.destroy_subscription(odom_sub)
            if self.enable_sound:
                self._trigger_buzzer('reverse_stop')
                stop_reverse_melody()

        print("[Undock] Cleared obstacle successfully with precise distance.")

    def wait_for_station_task(
        self,
        station_id: str,
        station_name: str,
        robot_name: str,
        task_type: str,
        auto_mode: bool = False,
        auto_duration: float = 3.0
    ) -> bool:
        """
        Manage external task execution between docking and undocking.
        Triggers task start over ROS topic '/pinky/station_task/trigger' for external manipulators
        (Doosan collaborative robot, OpenMANIPULATOR-X), and waits for completion either via:
          1) Interactive terminal input ([Enter] or 'done')
          2) ROS topic '/pinky/station_task/complete' (sent by external robot arm node)
          3) Automated timer (if auto_mode=True)
        """
        print("\n" + "="*72)
        print("  🤖 [STATION TASK] External Manipulator Task Execution")
        print("="*72)
        print(f"  • Station:      {station_name} (ID: {station_id})")
        print(f"  • Manipulator:  {robot_name}")
        print(f"  • Task:         {task_type}")
        print(f"  • ROS Trigger:  Published START to '/pinky/station_task/trigger'")
        print("="*72)

        # 1. Publish task START trigger for external robot arm
        trigger_payload = {
            "station_id": station_id,
            "station_name": station_name,
            "robot": robot_name,
            "task": task_type,
            "action": "START",
            "timestamp": time.time()
        }
        start_msg = String()
        start_msg.data = json.dumps(trigger_payload)
        for _ in range(3):
            self.task_trigger_pub.publish(start_msg)
            time.sleep(0.01)

        if self.enable_sound:
            self._trigger_buzzer('alert')

        self.external_task_completed = False
        self.external_task_msg = ""
        task_start_t = time.time()

        if auto_mode:
            print(f"\n[Task Auto-Mode] Simulating task duration ({auto_duration:.1f}s)...")
            while (time.time() - task_start_t) < auto_duration and rclpy.ok():
                rclpy.spin_once(self.nav, timeout_sec=0.1)
            print(f"  [Task Auto-Mode] Simulated task completed ({auto_duration:.1f}s).")
        else:
            print(f"\n👉 [Action Required] Please perform or verify the task.")
            print(f"   • To complete via terminal: Press [Enter] or type 'done'")
            print(f"   • To complete via ROS 2:    ros2 topic pub --once /pinky/station_task/complete std_msgs/msg/String \"data: 'done'\"")
            print(f"\n   Waiting for task completion >> ", end='', flush=True)

            user_input_received = False
            user_input_val = ""

            def _read_stdin():
                nonlocal user_input_received, user_input_val
                try:
                    user_input_val = sys.stdin.readline()
                    user_input_received = True
                except Exception:
                    user_input_received = True

            input_thread = threading.Thread(target=_read_stdin, daemon=True)
            input_thread.start()

            while not user_input_received and not self.external_task_completed and rclpy.ok():
                rclpy.spin_once(self.nav, timeout_sec=0.1)

            elapsed = time.time() - task_start_t
            if self.external_task_completed:
                print(f"\n✅ [Task Complete] Received ROS signal from '{robot_name}' on '/pinky/station_task/complete': '{self.external_task_msg}' (took {elapsed:.1f}s)")
            else:
                user_str = user_input_val.strip() if user_input_val else "OK"
                print(f"\n✅ [Task Complete] Confirmed by user input: '{user_str}' (took {elapsed:.1f}s)")

        # 2. Publish task COMPLETE confirmation
        complete_payload = {
            "station_id": station_id,
            "station_name": station_name,
            "robot": robot_name,
            "task": task_type,
            "action": "COMPLETED",
            "timestamp": time.time()
        }
        done_msg = String()
        done_msg.data = json.dumps(complete_payload)
        for _ in range(3):
            self.task_trigger_pub.publish(done_msg)
            time.sleep(0.01)

        if self.enable_sound:
            self._trigger_buzzer('success')

        print(f"[Mission] Station task '{task_type}' finished. Proceeding to Undock.\n")
        return True


def run_full_mission(stations: list = None, enable_sound: bool = True, piezo_pin: int = None, use_sim_time: bool = False,
                     nav_timeout: float = 180.0, dock_timeout: float = 120.0,
                     auto_task: bool = False, task_delay: float = 3.0,
                     return_home: bool = True):
    if stations is None:
        stations = ALL_STATIONS

    rclpy.init()
    ctrl = ArenaMissionController(enable_sound=enable_sound, piezo_pin=piezo_pin, use_sim_time=use_sim_time)

    num_stations = len(stations)
    total_stages = num_stations + (1 if return_home else 0)

    print("\n" + "="*75)
    print("      🚀 STARTING PINKY PRO ARENA E2E MULTI-STATION MISSION 🚀")
    print("="*75)
    print(f"Mission Plan ({num_stations} Stations Total):")
    for idx, st in enumerate(stations, 1):
        print(f"  [{idx}/{total_stages}] {st['station_name']} | Marker ID {st['marker_id']} | Area: {st['area']}")
    if return_home:
        print(f"  [{total_stages}/{total_stages}] Return to Home Parking {WAYPOINTS['home']}")
    print("="*75 + "\n")

    mission_start_t = time.time()

    try:
        for idx, station in enumerate(stations, 1):
            pre_dock_key = station['pre_docking']
            pre_dock_pose = WAYPOINTS[pre_dock_key]

            print("\n" + "#"*72)
            print(f"  📍 [STAGE {idx}/{total_stages}] {station['station_name']}")
            print("#"*72)

            # Step 1: Navigate to Pre-docking Waypoint
            print(f"\n>>> [Stage {idx}-1] Navigating to Pre-docking Waypoint '{pre_dock_key}'...")
            if not ctrl.navigate_to(f"{pre_dock_key} Pre-docking", *pre_dock_pose, timeout_sec=nav_timeout):
                raise RuntimeError(f"Navigation to {pre_dock_key} pre-docking failed for {station['station_name']}")
            time.sleep(1.0)

            # Step 2: Precision Docking to ArUco Marker
            print(f"\n>>> [Stage {idx}-2] Precision Docking to Marker ID {station['marker_id']}...")
            if not ctrl.execute_docking(station['station_name'], station['marker_id'], timeout_sec=dock_timeout):
                raise RuntimeError(f"Precision docking failed at {station['station_name']} (ID {station['marker_id']})")

            # Step 3: External Manipulator Task Execution
            print(f"\n>>> [Stage {idx}-3] External Manipulator Task...")
            ctrl.wait_for_station_task(
                station_id=station['station_id'],
                station_name=station['station_name'],
                robot_name=station['robot_name'],
                task_type=station['task_type'],
                auto_mode=auto_task,
                auto_duration=task_delay
            )

            # Step 4: Closed-loop Undock Backup
            undock_dist = station.get('undock_dist', 0.50)
            print(f"\n>>> [Stage {idx}-4] Undock Backing Up ({undock_dist*100:.1f}cm)...")
            ctrl.undock_backup(distance_m=undock_dist, speed=-0.05, timeout_sec=30.0)
            time.sleep(0.5)

        # Final Stage: Return to Home Parking
        if return_home:
            home_idx = total_stages
            print("\n" + "#"*72)
            print(f"  🏁 [STAGE {home_idx}/{total_stages}] Return to Home Parking")
            print("#"*72)
            if not ctrl.navigate_to('Home Parking', *WAYPOINTS['home'], timeout_sec=nav_timeout):
                raise RuntimeError("Return navigation to Home Parking failed")

        total_elapsed = time.time() - mission_start_t
        print("\n" + "="*75)
        print(f"  🎉 [SUCCESS] ENTIRE MISSION ({num_stations} STATIONS) COMPLETED IN {total_elapsed:.1f} SECONDS! 🎉")
        print("="*75 + "\n")

    except Exception as e:
        print(f"\n❌ [MISSION ABORTED] Reason: {e}")
    finally:
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Pinky Pro Arena Full E2E Mission")
    parser.add_argument('--step', default='all', help="Run specific step/station: 'all', 'home', station index (1-6), or marker ID (3,4,5,6,7,8)")
    parser.add_argument('--sample', action='store_true', help="Run 3 representative stations (Shelf 1 Right ID 8, Shelf 2 Left ID 3, Packaging ID 5)")
    parser.add_argument('--no-home', action='store_true', help="Do not return to home parking after finishing stations")
    parser.add_argument('--no-sound', action='store_true', help="Disable sound effects")
    parser.add_argument('--buzzer-pin', type=int, default=None, help="Raspberry Pi GPIO pin for piezo buzzer")
    parser.add_argument('--sim', action='store_true', help="Enable use_sim_time for Gazebo simulation")
    parser.add_argument('--nav-timeout', type=float, default=180.0, help="Navigation timeout in seconds (default: 180.0)")
    parser.add_argument('--dock-timeout', type=float, default=120.0, help="Docking timeout in seconds (default: 120.0)")
    parser.add_argument('--auto-task', action='store_true', help="Automatically complete station tasks without interactive terminal prompt (useful for headless testing)")
    parser.add_argument('--task-delay', type=float, default=3.0, help="Simulated task duration in seconds if --auto-task is enabled (default: 3.0)")
    args = parser.parse_args()

    # Determine station selection
    base_stations = SAMPLE_STATIONS if args.sample else ALL_STATIONS

    if args.step == 'all':
        stations_to_run = base_stations
        return_home = not args.no_home
    elif args.step == 'home':
        stations_to_run = []
        return_home = True
    elif args.step.isdigit():
        val = int(args.step)
        # Check if val is a 1-based station index
        if 1 <= val <= len(base_stations):
            stations_to_run = [base_stations[val - 1]]
            return_home = not args.no_home
        else:
            # Check if val is a marker ID
            matching = [s for s in ALL_STATIONS if s['marker_id'] == val]
            if matching:
                stations_to_run = matching
                return_home = not args.no_home
            else:
                print(f"[Error] Unknown station step or marker ID: {args.step}")
                sys.exit(1)
    else:
        # Check by station_id or name substring
        matching = [s for s in ALL_STATIONS if args.step.lower() in s['station_id'].lower() or args.step.lower() in s['station_name'].lower()]
        if matching:
            stations_to_run = matching
            return_home = not args.no_home
        else:
            print(f"[Error] Unknown step argument: {args.step}")
            sys.exit(1)

    run_full_mission(
        stations=stations_to_run,
        enable_sound=not args.no_sound,
        piezo_pin=args.buzzer_pin,
        use_sim_time=args.sim,
        nav_timeout=args.nav_timeout,
        dock_timeout=args.dock_timeout,
        auto_task=args.auto_task,
        task_delay=args.task_delay,
        return_home=return_home
    )


if __name__ == '__main__':
    main()
