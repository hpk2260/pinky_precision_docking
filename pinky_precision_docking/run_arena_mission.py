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
    'shelf1': [0.6000, 1.4700, 3.1416],  # Shelf 1 Area Representative Pre-docking (Views all 3 bays)
    'shelf2': [1.1100, 1.5000, -1.5708], # Shelf 2 Area Representative Pre-docking (Centered in Corridor 2)
    'packaging': [0.7240, 0.2430, 0.1070], # ArUco Marker 5
}

MARKER_IDS = {
    'shelf1': 8,  # Shelf 1 Right Bay Marker (ID 8)
    'shelf2': 3,  # Shelf 2 Left Cooler Marker (ID 3)
    'packaging': 5,
}


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


def run_full_mission(enable_sound: bool = True, piezo_pin: int = None, use_sim_time: bool = False,
                     nav_timeout: float = 180.0, dock_timeout: float = 120.0,
                     auto_task: bool = False, task_delay: float = 3.0):
    rclpy.init()
    ctrl = ArenaMissionController(enable_sound=enable_sound, piezo_pin=piezo_pin, use_sim_time=use_sim_time)

    print("\n" + "="*70)
    print("      🚀 STARTING PINKY PRO ARENA E2E MULTI-STATION MISSION 🚀")
    print("="*70)
    print("Mission Plan:")
    print("  1. Home Parking -> Shelf 1 Pre-docking")
    print("  2. Dock to Shelf 1 Right Bay (ArUco ID 8) & Doosan Arm Pick & Place")
    print("  3. Undock -> Shelf 2 Pre-docking")
    print("  4. Dock to Shelf 2 Left Cooler (ArUco ID 3) & Doosan Arm Pick & Place")
    print("  5. Undock -> Packaging Area Pre-docking")
    print("  6. Dock to Packaging Station (ArUco ID 5) & OMX Packaging Task")
    print("  7. Undock -> Return to Home Parking")
    print("="*70 + "\n")

    mission_start_t = time.time()

    try:
        # Step 1: Home -> Shelf 1 Pre-docking
        print("\n------------------------------------------------------------")
        print(" [Step 1/7] Home Parking -> Shelf Spot 1 Pre-docking (Right Bay)")
        print("------------------------------------------------------------")
        if not ctrl.navigate_to('Shelf 1 Pre-docking', *WAYPOINTS['shelf1'], timeout_sec=nav_timeout):
            raise RuntimeError("Step 1 failed")
        time.sleep(1.0)

        # Step 2: Docking at Shelf 1 (Marker ID 8)
        print("\n------------------------------------------------------------")
        print(" [Step 2/7] Precision Docking at Shelf 1 Right Bay (Marker ID 8)")
        print("------------------------------------------------------------")
        if not ctrl.execute_docking('Shelf 1', MARKER_IDS['shelf1'], timeout_sec=dock_timeout):
            raise RuntimeError("Step 2 failed")

        # External Arm Task (Shelf 1)
        ctrl.wait_for_station_task(
            station_id='shelf1',
            station_name='Shelf 1 Right Bay (선반 1번 우측 베이)',
            robot_name='두산 로봇팔 (Doosan Collaborative Robot)',
            task_type='부품 Pick & Place (피킹 및 핑키 적재)',
            auto_mode=auto_task,
            auto_duration=task_delay
        )

        # Step 3: Undock -> Shelf 2 Pre-docking
        print("\n------------------------------------------------------------")
        print(" [Step 3/7] Undock -> Shelf Spot 2 Pre-docking (Left Cooler)")
        print("------------------------------------------------------------")
        # Step 3-1: Back up into the open central aisle (X from ~0.08m to ~0.65m)
        ctrl.undock_backup(distance_m=0.55, speed=-0.05, timeout_sec=25.0)
        time.sleep(0.5)

        if not ctrl.navigate_to('Shelf 2 Pre-docking', *WAYPOINTS['shelf2'], timeout_sec=nav_timeout):
            raise RuntimeError("Step 3 failed")
        time.sleep(1.0)

        # Step 4: Docking at Shelf 2 (Marker ID 3)
        print("\n------------------------------------------------------------")
        print(" [Step 4/7] Precision Docking at Shelf 2 Left Cooler (Marker ID 3)")
        print("------------------------------------------------------------")
        if not ctrl.execute_docking('Shelf 2', MARKER_IDS['shelf2'], timeout_sec=dock_timeout):
            raise RuntimeError("Step 4 failed")

        # External Arm Task (Shelf 2)
        ctrl.wait_for_station_task(
            station_id='shelf2',
            station_name='Shelf 2 Left Cooler (선반 2번 좌측 냉장고)',
            robot_name='두산 로봇팔 (Doosan Collaborative Robot)',
            task_type='추가 부품 Pick & Place (피킹 및 핑키 적재)',
            auto_mode=auto_task,
            auto_duration=task_delay
        )

        # Step 5: Undock -> Packaging Pre-docking
        print("\n------------------------------------------------------------")
        print(" [Step 5/7] Undock -> Packaging Station Pre-docking")
        print("------------------------------------------------------------")
        # Step 5-1: Back up straight out of Shelf 2 fully into the open north aisle (Y from ~1.00m to ~1.65m)
        ctrl.undock_backup(distance_m=0.65, speed=-0.05, timeout_sec=30.0)
        time.sleep(0.5)

        # Step 5-2: Smoothly navigate from North Corridor exit to Packaging Area Pre-docking
        if not ctrl.navigate_to('Packaging Pre-docking', *WAYPOINTS['packaging'], timeout_sec=nav_timeout):
            raise RuntimeError("Step 5 failed")
        time.sleep(1.0)

        # Step 6: Docking at Packaging Station (Marker ID 5)
        print("\n------------------------------------------------------------")
        print(" [Step 6/7] Precision Docking at Packaging Station (Marker ID 5)")
        print("------------------------------------------------------------")
        if not ctrl.execute_docking('Packaging Station', MARKER_IDS['packaging'], timeout_sec=dock_timeout):
            raise RuntimeError("Step 6 failed")

        # External Arm Task (Packaging)
        ctrl.wait_for_station_task(
            station_id='packaging',
            station_name='Packaging Station (포장 구역)',
            robot_name='OpenMANIPULATOR-X (OMX 매니퓰레이터)',
            task_type='적재물 하역 및 자동 포장 (Packaging)',
            auto_mode=auto_task,
            auto_duration=task_delay
        )

        # Step 7: Undock -> Return Home
        print("\n------------------------------------------------------------")
        print(" [Step 7/7] Undock -> Return to Home Parking")
        print("------------------------------------------------------------")
        # Step 7-1: Back up straight out of Packaging station into the central aisle (X from ~0.83m to ~0.50m)
        ctrl.undock_backup(distance_m=0.35, speed=-0.05, timeout_sec=20.0)
        time.sleep(0.5)

        if not ctrl.navigate_to('Home Parking', *WAYPOINTS['home'], timeout_sec=nav_timeout):
            raise RuntimeError("Step 7 failed")

        total_elapsed = time.time() - mission_start_t
        print("\n" + "="*70)
        print(f"  🎉 [SUCCESS] ENTIRE MISSION COMPLETED IN {total_elapsed:.1f} SECONDS! 🎉")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ [MISSION ABORTED] Reason: {e}")
    finally:
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Pinky Pro Arena Full E2E Mission")
    parser.add_argument('--step', default='all', help="Specific step to run: all, 1, 2, 3, 4, 5, 6, 7")
    parser.add_argument('--no-sound', action='store_true', help="Disable sound effects")
    parser.add_argument('--buzzer-pin', type=int, default=None, help="Raspberry Pi GPIO pin for piezo buzzer")
    parser.add_argument('--sim', action='store_true', help="Enable use_sim_time for Gazebo simulation")
    parser.add_argument('--nav-timeout', type=float, default=180.0, help="Navigation timeout in seconds (default: 180.0)")
    parser.add_argument('--dock-timeout', type=float, default=120.0, help="Docking timeout in seconds (default: 120.0)")
    parser.add_argument('--auto-task', action='store_true', help="Automatically complete station tasks without interactive terminal prompt (useful for headless testing)")
    parser.add_argument('--task-delay', type=float, default=3.0, help="Simulated task duration in seconds if --auto-task is enabled (default: 3.0)")
    args = parser.parse_args()

    run_full_mission(
        enable_sound=not args.no_sound,
        piezo_pin=args.buzzer_pin,
        use_sim_time=args.sim,
        nav_timeout=args.nav_timeout,
        dock_timeout=args.dock_timeout,
        auto_task=args.auto_task,
        task_delay=args.task_delay
    )


if __name__ == '__main__':
    main()
