import os
import signal
import subprocess
import time
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def kill_sim():
    for name in ['parameter_bridge', 'image_bridge', 'gz', 'precision_docking_node', 'robot_state_publisher']:
        subprocess.run(['killall', '-q', '-9', name], stderr=subprocess.DEVNULL)
    time.sleep(1.0)

kill_sim()

print("Launching Gazebo simulation...")
cmd = ['ros2', 'launch', 'pinky_gz_sim', 'launch_sim.launch.xml', 'world_name:=docking_test.world', 'headless:=true']
proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"Simulation PID: {proc.pid}. Waiting 8 seconds...")
time.sleep(8.0)

rclpy.init()
node = Node('cam_check_node')
bridge = CvBridge()
frame = None

def cb(msg):
    global frame
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    print("Frame received! Resolution:", frame.shape)
    rclpy.shutdown()

sub = node.create_subscription(Image, '/camera/image_raw', cb, 10)
print("Spinning node to wait for /camera/image_raw (timeout 10s)...")
start_t = time.time()
while time.time() - start_t < 10.0 and frame is None:
    rclpy.spin_once(node, timeout_sec=0.5)

if frame is not None:
    save_path = '/home/user/pinky_pro/scratch_sim_cam.jpg'
    cv2.imwrite(save_path, frame)
    print(f"Saved frame to {save_path}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, d)
    print("Detected IDs:", ids)
    print("Number of rejected candidates:", len(rejected))
else:
    print("FAILED: No image received on /camera/image_raw!")

kill_sim()
print("Done.")
