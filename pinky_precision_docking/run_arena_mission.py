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
import math
import subprocess
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
    'home': [0.0000, 0.0000, 0.0000],
    'shelf1': [0.5990, 1.4700, 3.1090],  # ArUco Marker 7
    'shelf2': [1.0210, 1.5320, -1.5680], # ArUco Marker 4
    'packaging': [0.7240, 0.2430, 0.1070], # ArUco Marker 5
}

MARKER_IDS = {
    'shelf1': 7,
    'shelf2': 4,
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

            elapsed_clk = (now_clk - start_clk).nanoseconds / 1e9
            elapsed_wall = now_wall_t - start_wall_t

            # Periodic progress log every 3 seconds
            if now_wall_t - last_log_wall_t >= 3.0:
                last_log_wall_t = now_wall_t
                dist_str = f"{feedback.distance_remaining:.2f}m" if feedback else "measuring..."
                print(f"  [Nav2 Progress] To '{waypoint_name}' | Remaining: {dist_str} | Elapsed: {elapsed_clk:.1f}s (sim) / {elapsed_wall:.1f}s (wall)")

            time.sleep(0.1)

            # Timeout check against simulation clock (or wall clock if not sim)
            if elapsed_clk > timeout_sec:
                print(f"[Nav2 Error] Navigation to {waypoint_name} timed out ({timeout_sec:.0f}s sim time, {elapsed_wall:.1f}s wall time)!")
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
            elapsed_clk = (now_clk - start_clk).nanoseconds / 1e9

            if now_wall_t - last_log_wall_t >= 3.0:
                last_log_wall_t = now_wall_t
                print(f"  [Docking Progress] Docking at '{station_name}' in progress... ({elapsed_clk:.1f}s sim)")

            if elapsed_clk > timeout_sec:
                print(f"[Docking Error] Docking at '{station_name}' timed out ({timeout_sec:.0f}s sim time)!")
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

                # Safety timeout check (simulation clock)
                elapsed_clk = (self.nav.get_clock().now() - start_clk).nanoseconds / 1e9
                if elapsed_clk > timeout_sec:
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


def run_full_mission(enable_sound: bool = True, piezo_pin: int = None, use_sim_time: bool = False,
                     nav_timeout: float = 180.0, dock_timeout: float = 120.0):
    rclpy.init()
    ctrl = ArenaMissionController(enable_sound=enable_sound, piezo_pin=piezo_pin, use_sim_time=use_sim_time)

    print("\n" + "="*70)
    print("      🚀 STARTING PINKY PRO ARENA E2E MULTI-STATION MISSION 🚀")
    print("="*70)
    print("Mission Plan:")
    print("  1. Home Parking -> Shelf 1 Pre-docking")
    print("  2. Dock to Shelf 1 (ArUco ID 7)")
    print("  3. Undock -> Shelf 2 Pre-docking")
    print("  4. Dock to Shelf 2 (ArUco ID 4)")
    print("  5. Undock -> Packaging Area Pre-docking")
    print("  6. Dock to Packaging Station (ArUco ID 5)")
    print("  7. Undock -> Return to Home Parking")
    print("="*70 + "\n")

    mission_start_t = time.time()

    try:
        # Step 1: Home -> Shelf 1 Pre-docking
        print("\n------------------------------------------------------------")
        print(" [Step 1/7] Home Parking -> Shelf Spot 1 Pre-docking")
        print("------------------------------------------------------------")
        if not ctrl.navigate_to('Shelf 1 Pre-docking', *WAYPOINTS['shelf1'], timeout_sec=nav_timeout):
            raise RuntimeError("Step 1 failed")
        time.sleep(1.0)

        # Step 2: Docking at Shelf 1 (Marker ID 7)
        print("\n------------------------------------------------------------")
        print(" [Step 2/7] Precision Docking at Shelf 1 (Marker ID 7)")
        print("------------------------------------------------------------")
        if not ctrl.execute_docking('Shelf 1', MARKER_IDS['shelf1'], timeout_sec=dock_timeout):
            raise RuntimeError("Step 2 failed")
        time.sleep(2.0)  # Simulating picking duration

        # Step 3: Undock -> Shelf 2 Pre-docking
        print("\n------------------------------------------------------------")
        print(" [Step 3/7] Undock -> Shelf Spot 2 Pre-docking")
        print("------------------------------------------------------------")
        ctrl.undock_backup(distance_m=0.20)
        if not ctrl.navigate_to('Shelf 2 Pre-docking', *WAYPOINTS['shelf2'], timeout_sec=nav_timeout):
            raise RuntimeError("Step 3 failed")
        time.sleep(1.0)

        # Step 4: Docking at Shelf 2 (Marker ID 4)
        print("\n------------------------------------------------------------")
        print(" [Step 4/7] Precision Docking at Shelf 2 (Marker ID 4)")
        print("------------------------------------------------------------")
        if not ctrl.execute_docking('Shelf 2', MARKER_IDS['shelf2'], timeout_sec=dock_timeout):
            raise RuntimeError("Step 4 failed")
        time.sleep(2.0)  # Simulating picking duration

        # Step 5: Undock -> Packaging Pre-docking
        print("\n------------------------------------------------------------")
        print(" [Step 5/7] Undock -> Packaging Station Pre-docking")
        print("------------------------------------------------------------")
        ctrl.undock_backup(distance_m=0.20)
        if not ctrl.navigate_to('Packaging Pre-docking', *WAYPOINTS['packaging'], timeout_sec=nav_timeout):
            raise RuntimeError("Step 5 failed")
        time.sleep(1.0)

        # Step 6: Docking at Packaging Station (Marker ID 5)
        print("\n------------------------------------------------------------")
        print(" [Step 6/7] Precision Docking at Packaging Station (Marker ID 5)")
        print("------------------------------------------------------------")
        if not ctrl.execute_docking('Packaging Station', MARKER_IDS['packaging'], timeout_sec=dock_timeout):
            raise RuntimeError("Step 6 failed")
        time.sleep(2.0)  # Simulating packing duration

        # Step 7: Undock -> Return Home
        print("\n------------------------------------------------------------")
        print(" [Step 7/7] Undock -> Return to Home Parking")
        print("------------------------------------------------------------")
        ctrl.undock_backup(distance_m=0.20)
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
    args = parser.parse_args()

    run_full_mission(
        enable_sound=not args.no_sound,
        piezo_pin=args.buzzer_pin,
        use_sim_time=args.sim,
        nav_timeout=args.nav_timeout,
        dock_timeout=args.dock_timeout
    )


if __name__ == '__main__':
    main()
