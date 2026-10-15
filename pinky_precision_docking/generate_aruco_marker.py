#!/usr/bin/env python3
"""
ArUco Marker Generator for Pinky Pro Auto-Docking.
Generates printable ArUco markers (DICT_4X4_50, IDs 0-3) with quiet zone and text label.
"""

import argparse
import os
import cv2
import numpy as np


def generate_marker(
    marker_id: int = 0,
    dict_type: int = cv2.aruco.DICT_4X4_50,
    marker_size_px: int = 400,
    border_px: int = 80,
    output_dir: str = '.'
) -> str:
    """Generate a single ArUco marker with a white quiet zone border and text label."""
    dictionary = cv2.aruco.getPredefinedDictionary(dict_type)

    # Generate marker image
    if hasattr(cv2.aruco, 'generateImageMarker'):
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size_px)
    else:
        marker_img = cv2.aruco.drawMarker(dictionary, marker_id, marker_size_px)

    # Create white canvas with quiet zone (border) and extra space at bottom for text
    label_height = 60
    total_w = marker_size_px + 2 * border_px
    total_h = marker_size_px + 2 * border_px + label_height

    canvas = np.full((total_h, total_w), 255, dtype=np.uint8)

    # Place marker in center
    canvas[border_px:border_px + marker_size_px, border_px:border_px + marker_size_px] = marker_img

    # Draw label text at bottom
    text = f"ArUco DICT_4X4_50 | ID: {marker_id} | 60x60mm"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (total_w - text_size[0]) // 2
    text_y = total_h - 20

    cv2.putText(canvas, text, (text_x, text_y), font, font_scale, 0, thickness, cv2.LINE_AA)

    # Save image
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"aruco_marker_id{marker_id}.png")
    cv2.imwrite(filename, canvas)
    return filename


def main():
    parser = argparse.ArgumentParser(description="Generate ArUco markers for docking")
    parser.add_argument("--output-dir", type=str, default="./markers", help="Output directory")
    parser.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2, 3], help="Marker IDs to generate")
    parser.add_argument("--size", type=int, default=500, help="Marker size in pixels")
    parser.add_argument("--border", type=int, default=100, help="White border (quiet zone) in pixels")
    args = parser.parse_args()

    print(f"Generating ArUco markers in: {os.path.abspath(args.output_dir)}")
    for mid in args.ids:
        path = generate_marker(
            marker_id=mid,
            marker_size_px=args.size,
            border_px=args.border,
            output_dir=args.output_dir
        )
        print(f"  [+] Saved Marker ID {mid} -> {path}")
    print("Done! You can print these images and place them on your docking station.")


if __name__ == '__main__':
    main()
