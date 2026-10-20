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
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from std_msgs.msg import Bool
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
    def __init__(self, enable_sound: bool = True, piezo_pin: int = None):
        self.enable_sound = enable_sound
        if self.enable_sound:
            self.sound_mgr = get_sound_manager()
            if piezo_pin is not None:
                self.sound_mgr.piezo_pin = piezo_pin
                self.sound_mgr._init_gpio_piezo()

        self.nav = BasicNavigator()
        self.docking_completed = False

        # Subscriber for /docking_completed
        self.sub_dock = self.nav.create_subscription(
            Bool, '/docking_completed', self._docking_completed_cb, 10
        )
        self.cmd_vel_pub = self.nav.create_publisher(Twist, '/cmd_vel', 10)
        self.start_dock_cli = self.nav.create_client(Trigger, '/docking/start')

    def _docking_completed_cb(self, msg: Bool):
        if msg.data:
            self.docking_completed = True

    def set_docking_marker_id(self, marker_id: int):
        """Set target_marker_id parameter on precision_docking_node."""
        print(f"[Mission] Setting precision_docking_node target_marker_id = {marker_id}...")
        try:
            subprocess.run(
                ['ros2', 'param', 'set', '/precision_docking_node', 'target_marker_id', str(marker_id)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"[Mission] Parameter target_marker_id set to {marker_id} successfully.")
        except Exception as e:
            print(f"[Mission Warning] Could not set param via CLI: {e}")

    def navigate_to(self, waypoint_name: str, x: float, y: float, yaw: float, timeout_sec: float = 60.0) -> bool:
        """Navigate to waypoint using Nav2 BasicNavigator."""
        print(f"\n>>> [Nav2] Navigating to '{waypoint_name}': x={x:.3f}m, y={y:.3f}m, yaw={math.degrees(yaw):+.1f} deg...")

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.nav.get_clock().now().to_msg()
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        goal.pose.position.z = 0.0
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)

        self.nav.goToPose(goal)

        start_t = time.time()
        while not self.nav.isTaskComplete():
            feedback = self.nav.getFeedback()
            time.sleep(0.2)
            if time.time() - start_t > timeout_sec:
                print(f"[Nav2 Error] Navigation to {waypoint_name} timed out ({timeout_sec}s)!")
                self.nav.cancelTask()
                return False

        res = self.nav.getResult()
        if res == TaskResult.SUCCEEDED:
            print(f"[Nav2] Successfully arrived at '{waypoint_name}' in {time.time() - start_t:.1f}s.")
            return True
        else:
            print(f"[Nav2 Error] Navigation to '{waypoint_name}' failed with result: {res}")
            return False

    def execute_docking(self, station_name: str, marker_id: int, timeout_sec: float = 40.0) -> bool:
        """Execute precision docking to designated ArUco marker."""
        print(f"\n>>> [Docking] Starting Precision Docking at '{station_name}' to Marker ID {marker_id}...")
        self.set_docking_marker_id(marker_id)
        time.sleep(0.5)

        if not self.start_dock_cli.wait_for_service(timeout_sec=5.0):
            print("[Docking Error] /docking/start service not available! Is precision_docking_node running?")
            return False

        self.docking_completed = False
        req = Trigger.Request()
        future = self.start_dock_cli.call_async(req)

        start_t = time.time()
        while not self.docking_completed:
            rclpy.spin_once(self.nav, timeout_sec=0.1)
            time.sleep(0.05)
            if time.time() - start_t > timeout_sec:
                print(f"[Docking Error] Docking at '{station_name}' timed out ({timeout_sec}s)!")
                return False

        print(f"[Docking] Precision Docking completed successfully at '{station_name}' in {time.time() - start_t:.1f}s!")
        if self.enable_sound:
            play_docking_complete(blocking=False)
        return True

    def undock_backup(self, distance_m: float = 0.20, speed: float = -0.04):
        """Safely back up straight to clear wall/shelf before handing control back to Nav2."""
        duration = abs(distance_m / speed)
        print(f"[Undock] Backing up {distance_m*100:.0f}cm straight ({speed:.2f}m/s for {duration:.1f}s) to clear obstacle...")
        if self.enable_sound:
            start_reverse_melody()
        try:
            msg = Twist()
            msg.linear.x = float(speed)
            t_end = time.time() + duration
            while time.time() < t_end:
                self.cmd_vel_pub.publish(msg)
                time.sleep(0.05)
            # Stop
            self.cmd_vel_pub.publish(Twist())
            time.sleep(0.5)
        finally:
            if self.enable_sound:
                stop_reverse_melody()
        print("[Undock] Cleared obstacle successfully.")


def run_full_mission(enable_sound: bool = True, piezo_pin: int = None):
    rclpy.init()
    ctrl = ArenaMissionController(enable_sound=enable_sound, piezo_pin=piezo_pin)

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
        if not ctrl.navigate_to('Shelf 1 Pre-docking', *WAYPOINTS['shelf1']):
            raise RuntimeError("Step 1 failed")
        time.sleep(1.0)

        # Step 2: Docking at Shelf 1 (Marker ID 7)
        print("\n------------------------------------------------------------")
        print(" [Step 2/7] Precision Docking at Shelf 1 (Marker ID 7)")
        print("------------------------------------------------------------")
        if not ctrl.execute_docking('Shelf 1', MARKER_IDS['shelf1']):
            raise RuntimeError("Step 2 failed")
        time.sleep(2.0)  # Simulating picking duration

        # Step 3: Undock -> Shelf 2 Pre-docking
        print("\n------------------------------------------------------------")
        print(" [Step 3/7] Undock -> Shelf Spot 2 Pre-docking")
        print("------------------------------------------------------------")
        ctrl.undock_backup(distance_m=0.20)
        if not ctrl.navigate_to('Shelf 2 Pre-docking', *WAYPOINTS['shelf2']):
            raise RuntimeError("Step 3 failed")
        time.sleep(1.0)

        # Step 4: Docking at Shelf 2 (Marker ID 4)
        print("\n------------------------------------------------------------")
        print(" [Step 4/7] Precision Docking at Shelf 2 (Marker ID 4)")
        print("------------------------------------------------------------")
        if not ctrl.execute_docking('Shelf 2', MARKER_IDS['shelf2']):
            raise RuntimeError("Step 4 failed")
        time.sleep(2.0)  # Simulating picking duration

        # Step 5: Undock -> Packaging Pre-docking
        print("\n------------------------------------------------------------")
        print(" [Step 5/7] Undock -> Packaging Station Pre-docking")
        print("------------------------------------------------------------")
        ctrl.undock_backup(distance_m=0.20)
        if not ctrl.navigate_to('Packaging Pre-docking', *WAYPOINTS['packaging']):
            raise RuntimeError("Step 5 failed")
        time.sleep(1.0)

        # Step 6: Docking at Packaging Station (Marker ID 5)
        print("\n------------------------------------------------------------")
        print(" [Step 6/7] Precision Docking at Packaging Station (Marker ID 5)")
        print("------------------------------------------------------------")
        if not ctrl.execute_docking('Packaging Station', MARKER_IDS['packaging']):
            raise RuntimeError("Step 6 failed")
        time.sleep(2.0)  # Simulating packing duration

        # Step 7: Undock -> Return Home
        print("\n------------------------------------------------------------")
        print(" [Step 7/7] Undock -> Return to Home Parking")
        print("------------------------------------------------------------")
        ctrl.undock_backup(distance_m=0.20)
        if not ctrl.navigate_to('Home Parking', *WAYPOINTS['home']):
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
    args = parser.parse_args()

    run_full_mission(enable_sound=not args.no_sound, piezo_pin=args.buzzer_pin)


if __name__ == '__main__':
    main()
