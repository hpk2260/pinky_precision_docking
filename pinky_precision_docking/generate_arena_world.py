#!/usr/bin/env python3
"""
Arena Docking World Generator for Pinky Pro Gazebo Simulation.
Generates an SDF 1.8 world corresponding to the 1.5m x 2.25m arena map (arena_map.yaml),
including arena walls and ArUco markers:
  - Shelf 1 (Marker ID 7)
  - Shelf 2 (Marker ID 4)
  - Packaging Station (Marker ID 5)
"""

import os
import cv2
import numpy as np


def generate_marker_xml(marker_id: int, x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> str:
    dict_5x5 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    marker_img = cv2.aruco.drawMarker(dict_5x5, marker_id, 7)
    
    xml = []
    xml.append(f'    <!-- ArUco Marker DICT_5X5_250 ID {marker_id} -->')
    xml.append(f'    <model name="aruco_5x5_id{marker_id}">')
    xml.append('      <static>true</static>')
    xml.append(f'      <pose>{x:.4f} {y:.4f} {z:.4f} {roll:.4f} {pitch:.4f} {yaw:.4f}</pose>')
    xml.append('      <link name="link">')
    xml.append('        <visual name="white_bg">')
    xml.append('          <pose>0 0 0 0 0 0</pose>')
    xml.append('          <geometry>')
    xml.append('            <box><size>0.002 0.07714 0.07714</size></box>')
    xml.append('          </geometry>')
    xml.append('          <material>')
    xml.append('            <ambient>1 1 1 1</ambient>')
    xml.append('            <diffuse>1 1 1 1</diffuse>')
    xml.append('            <specular>0 0 0 1</specular>')
    xml.append('            <emissive>0.92 0.92 0.92 1</emissive>')
    xml.append('          </material>')
    xml.append('        </visual>')

    cell_size = 0.06 / 7.0
    for r in range(7):
        for c in range(7):
            if marker_img[r, c] < 128:  # Black cell
                y_pos = (3 - c) * cell_size
                z_pos = (3 - r) * cell_size
                xml.append(f'        <visual name="cell_{r}_{c}">')
                xml.append(f'          <pose>-0.0008 {y_pos:.5f} {z_pos:.5f} 0 0 0</pose>')
                xml.append('          <geometry>')
                xml.append('            <box><size>0.001 0.00866 0.00866</size></box>')
                xml.append('          </geometry>')
                xml.append('          <material>')
                xml.append('            <ambient>0 0 0 1</ambient>')
                xml.append('            <diffuse>0 0 0 1</diffuse>')
                xml.append('            <specular>0 0 0 1</specular>')
                xml.append('            <emissive>0 0 0 1</emissive>')
                xml.append('          </material>')
                xml.append('        </visual>')
    xml.append('      </link>')
    xml.append('    </model>')
    return '\n'.join(xml)


def generate_wall_xml(name: str, x: float, y: float, z: float, dx: float, dy: float, dz: float) -> str:
    return f"""    <model name="{name}">
      <static>true</static>
      <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>{dx:.4f} {dy:.4f} {dz:.4f}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{dx:.4f} {dy:.4f} {dz:.4f}</size></box></geometry>
          <material>
            <ambient>0.7 0.72 0.75 1</ambient>
            <diffuse>0.7 0.72 0.75 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>"""


def build_world_xml() -> str:
    walls = [
        # (name, cx, cy, cz, dx, dy, dz)
        ('wall_north', 0.60, 1.88, 0.125, 1.56, 0.04, 0.25),
        ('wall_south', 0.60, -0.32, 0.125, 1.56, 0.04, 0.25),
        ('wall_west_lower', -0.14, 0.38, 0.125, 0.04, 1.40, 0.25),
        ('wall_shelf1_docking', -0.06, 1.47, 0.125, 0.04, 0.78, 0.25),
        ('wall_shelf1_divider', 0.15, 1.08, 0.125, 0.58, 0.04, 0.25),
        ('wall_center_pillar', 0.51, 0.04, 0.125, 0.04, 0.64, 0.25),
        ('wall_east_upper', 1.34, 1.36, 0.125, 0.04, 1.00, 0.25),
        ('wall_shelf2_divider', 0.88, 1.11, 0.125, 0.04, 0.50, 0.25),
        ('wall_shelf2_docking', 1.11, 0.86, 0.125, 0.46, 0.04, 0.25),
        ('wall_packaging_top', 1.15, 0.48, 0.125, 0.34, 0.04, 0.25),
        ('wall_packaging_bottom', 1.15, 0.06, 0.125, 0.34, 0.04, 0.25),
        ('wall_packaging_docking', 1.32, 0.27, 0.125, 0.04, 0.42, 0.25),
        ('wall_east_lower', 1.34, -0.13, 0.125, 0.04, 0.38, 0.25),
    ]

    wall_xmls = [generate_wall_xml(*w) for w in walls]

    # Markers:
    # 1. Shelf 1: Marker 7 at front surface of wall_shelf1_docking facing +x
    #    wall center x = -0.06, east face x = -0.040. Marker white bg at x = -0.039.
    marker7_xml = generate_marker_xml(7, -0.0390, 1.4700, 0.0680, 0.0, 0.0, 3.14159)

    # 2. Shelf 2: Marker 4 at north surface of wall_shelf2_docking facing +y
    #    wall center y = 0.86, north face y = 0.880. Marker white bg at y = 0.881.
    marker4_xml = generate_marker_xml(4, 1.0210, 0.8810, 0.0680, 0.0, 0.0, -1.5708)

    # 3. Packaging: Marker 5 at west surface of wall_packaging_docking facing -x
    #    wall center x = 1.32, west face x = 1.300. Marker white bg at x = 1.299.
    marker5_xml = generate_marker_xml(5, 1.2990, 0.2430, 0.0680, 0.0, 0.0, 0.0000)

    world_str = f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="default">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>

    <gui fullscreen="0">
      <camera name="user_camera">
        <!-- Overhead view of the entire 1.5m x 2.25m arena -->
        <pose>0.60 0.80 2.80 0.0 1.25 1.57</pose>
        <view_controller>orbit</view_controller>
        <projection_type>perspective</projection_type>
      </camera>
    </gui>

    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>0.6 0.8 5 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>0 0 -1</direction>
    </light>

    <!-- Dedicated Spotlights for High-Contrast ArUco Recognition -->
    <light type="point" name="shelf1_light">
      <cast_shadows>false</cast_shadows>
      <pose>0.30 1.47 0.80 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <attenuation>
        <range>3.0</range>
        <linear>0.2</linear>
        <constant>0.8</constant>
      </attenuation>
    </light>

    <light type="point" name="shelf2_light">
      <cast_shadows>false</cast_shadows>
      <pose>1.02 1.20 0.80 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <attenuation>
        <range>3.0</range>
        <linear>0.2</linear>
        <constant>0.8</constant>
      </attenuation>
    </light>

    <light type="point" name="packaging_light">
      <cast_shadows>false</cast_shadows>
      <pose>1.00 0.24 0.80 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <attenuation>
        <range>3.0</range>
        <linear>0.2</linear>
        <constant>0.8</constant>
      </attenuation>
    </light>

    <!-- Ground Plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>10 10</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>10 10</size></plane></geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- ========================================== -->
    <!-- Arena Walls (Matching arena_map.yaml)     -->
    <!-- ========================================== -->
{chr(10).join(wall_xmls)}

    <!-- ========================================== -->
    <!-- Precision Docking ArUco Markers           -->
    <!-- ========================================== -->
{marker7_xml}

{marker4_xml}

{marker5_xml}

  </world>
</sdf>
"""
    return world_str


def main():
    content = build_world_xml()
    destinations = [
        "/home/user/pinky_pro/src/pinky_precision_docking/worlds/arena_docking.world",
        "/home/user/pinky_pro/src/pinky_pro/pinky_gz_sim/worlds/arena_docking.world",
    ]
    for dest in destinations:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as f:
            f.write(content)
        print(f"[+] Successfully wrote: {dest}")


if __name__ == '__main__':
    main()
