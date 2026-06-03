#!/usr/bin/env python3
"""
Xiaomi 17T Stereo Camera Setup
================================
This script helps configure the Xiaomi 17T phone as a stereo camera pair
for the Balloon Catch project.

Phone camera specifications:
  - Main camera (Leica 23mm):       50 MP, 1/1.55" sensor, f/1.7, OIS
  - Ultra-wide camera (Leica 15mm): 12 MP,          f/2.2, 120-deg HFOV

Streaming the phone cameras to the Mac:
---------------------------------------
Two approaches are available. For stereo, BOTH must be running simultaneously.

=== APPROACH A: One IP Webcam instance per camera (recommended) ===

  1. Install TWO camera streaming apps on the phone:
     a) "IP Webcam" (free, Pavel Khlebovich)
        -> Configure to use the MAIN camera (23mm)
        -> Set port to 8080
     b) "DroidCam" (free)
        -> Configure to use the ULTRA-WIDE camera (15mm)
        -> Set port to 4747 if using WiFi mode

     Or use two instances of the same app if supported.

  NOTE: Most Android streaming apps only expose ONE camera at a time.
  If dual-camera simultaneous streaming is not possible with free apps,
  use APPROACH B instead.

=== APPROACH B: Phone main cam + Mac built-in cam (simplest) ===

  Install "IP Webcam" on the phone:
  1. Open IP Webcam app on the Xiaomi 17T
  2. Select the MAIN camera (23mm Leica) in settings
  3. Note the IP address shown at the bottom of the app (e.g. 192.168.1.100)
  4. The stream URL will be: http://<IP>:8080/video

  Then run with:
    python main.py --camera-left 0 --camera-right "http://192.168.1.100:8080/video"

  This uses Mac camera as left stereo cam + phone main cam as right stereo cam.
  Mac built-in camera is also used for gesture detection.

=== APPROACH C: DroidCam USB + IP Webcam WiFi ===

  DroidCam (USB):
  1. Install DroidCam on phone AND DroidCam Client on Mac
  2. Connect phone via USB
  3. DroidCam creates virtual webcam device (e.g., /dev/video2)
  4. In DroidCam phone app, select the camera you want

  IP Webcam (WiFi):
  1. Install IP Webcam on phone and select the OTHER camera
  2. Stream at http://<IP>:8080/video

  Then run with:
    python main.py --camera-left 2 --camera-right "http://192.168.1.100:8080/video"

Phone mounting for stereo:
--------------------------
For proper horizontal stereo baseline:
  - Mount the phone in LANDSCAPE orientation
  - The main camera should be on the LEFT side
  - The ultra-wide camera should be on the RIGHT side
  - Estimate baseline: ~12 mm between the two lenses

Calibration considerations:
---------------------------
  - DISABLE OIS on the main camera (optical stabilization changes intrinsics)
  - Set manual focus if possible (avoids intrinsics changing per frame)
  - The main and ultra-wide have very different focal lengths (~5.4mm vs ~2.0mm)
    This means they have different FOVs. After rectification, the usable
    overlap region may be smaller. Use a larger checkerboard to cover the overlap.
  - The phone must be RIGIDLY mounted. Any movement ruins stereo calibration.
  - Capture 30-40 checkerboard pairs from different angles/distances.
"""

import sys
import os
import argparse
import numpy as np
import urllib.request
import socket
import json


def compute_initial_intrinsics(camera_spec: dict,
                                image_width: int = 640,
                                image_height: int = 480) -> np.ndarray:
    fl_mm = camera_spec["focal_length_mm"]
    sw_mm = camera_spec["sensor_width_mm"]
    sh_mm = camera_spec["sensor_height_mm"]

    fx = fl_mm * image_width / sw_mm
    fy = fl_mm * image_height / sh_mm
    cx = image_width / 2.0
    cy = image_height / 2.0

    K = np.array([[fx, 0, cx],
                   [0, fy, cy],
                   [0,  0,  1]], dtype=np.float64)

    return K


def print_camera_specs(image_width: int = 640, image_height: int = 480):
    specs = {
        "main": {
            "focal_length_mm": 5.4,
            "focal_length_eq_mm": 23,
            "sensor_width_mm": 8.25,
            "sensor_height_mm": 6.19,
            "aperture": "f/1.7",
            "h_fov_deg": 81,
            "resolution": "50 MP (8192 x 6144) binned to 12.5 MP"
        },
        "ultrawide": {
            "focal_length_mm": 2.0,
            "focal_length_eq_mm": 15,
            "sensor_width_mm": 5.6,
            "sensor_height_mm": 4.2,
            "aperture": "f/2.2",
            "h_fov_deg": 120,
            "resolution": "12 MP"
        }
    }

    print("=" * 60)
    print("  Xiaomi 17T Camera Specifications")
    print("=" * 60)
    print(f"  Capture resolution: {image_width} x {image_height}")
    print(f"  Stereo baseline: ~12 mm (phone landscape)")
    print()

    for name, spec in specs.items():
        K = compute_initial_intrinsics(spec, image_width, image_height)
        print(f"  --- {name.upper()} Camera ---")
        print(f"  Physical focal length:  {spec['focal_length_mm']:.1f} mm")
        print(f"  35mm equiv focal:       {spec['focal_length_eq_mm']:.0f} mm")
        print(f"  Sensor size:            {spec['sensor_width_mm']:.2f} x {spec['sensor_height_mm']:.2f} mm")
        print(f"  Aperture:               {spec['aperture']}")
        print(f"  Horizontal FOV:         {spec['h_fov_deg']} degrees")
        print(f"  Native resolution:      {spec['resolution']}")
        print()
        print(f"  Estimated intrinsics at {image_width}x{image_height}:")
        print(f"    fx = {K[0, 0]:.1f} px")
        print(f"    fy = {K[1, 1]:.1f} px")
        print(f"    cx = {K[0, 2]:.1f} px")
        print(f"    cy = {K[1, 2]:.1f} px")
        print()

    K_main = compute_initial_intrinsics(specs["main"], image_width, image_height)
    K_uw  = compute_initial_intrinsics(specs["ultrawide"], image_width, image_height)

    baseline = 0.012  # 12 mm
    disparity_at_1m = baseline * K_main[0, 0] / 1.0
    depth_res_at_1m = 1.0 / (K_main[0, 0] * baseline) * 1.0 ** 2

    print(f"  --- Stereo depth estimates ---")
    print(f"  Baseline:               {baseline*1000:.0f} mm")
    print(f"  Disparity at 1m range:  {disparity_at_1m:.1f} px")
    print(f"  Depth resolution at 1m: ~{depth_res_at_1m:.4f} m/px")
    print(f"  Min reliable range:     ~{3 * depth_res_at_1m:.2f} m")
    print("=" * 60)


def test_phone_connection(ip: str, port: int = 8080, timeout: float = 3.0) -> bool:
    url = f"http://{ip}:{port}"
    try:
        req = urllib.request.Request(url)
        urllib.request.urlopen(req, timeout=timeout)
        print(f"  [OK] Phone reachable at {url}")
        return True
    except Exception as e:
        print(f"  [FAIL] Cannot reach phone at {url}: {e}")
        return False


def scan_network(base_port: int = 8080, timeout: float = 1.0) -> list:
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    prefix = ".".join(local_ip.split(".")[:3])
    print(f"Scanning LAN {prefix}.x for IP cameras on port {base_port}...")

    found = []
    for last in range(1, 255):
        ip = f"{prefix}.{last}"
        url = f"http://{ip}:{base_port}"
        try:
            urllib.request.urlopen(url, timeout=timeout)
            found.append(ip)
            print(f"  Found: {ip}:{base_port}")
        except Exception:
            pass

    if not found:
        print("  No IP cameras found on the network.")
        print("  Make sure the phone is connected to WiFi and the streaming app is running.")
    return found


def print_setup_guide():
    guide = __doc__
    print(guide)


def main():
    parser = argparse.ArgumentParser(
        description="Xiaomi 17T Stereo Camera Setup for Balloon Catch")
    parser.add_argument("--specs", action="store_true",
                        help="Show camera specifications and estimated intrinsics")
    parser.add_argument("--scan", action="store_true",
                        help="Scan LAN for IP cameras")
    parser.add_argument("--test", type=str, default=None,
                        help="Test connection to phone IP")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port for IP camera stream (default: 8080)")
    parser.add_argument("--width", type=int, default=640,
                        help="Capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="Capture height (default: 480)")
    parser.add_argument("--guide", action="store_true",
                        help="Show full setup guide")
    parser.add_argument("--export-config", type=str, default=None,
                        help="Export a config snippet for the phone setup")
    args = parser.parse_args()

    if args.guide:
        print_setup_guide()
        return

    if args.scan:
        scan_network(base_port=args.port)
        return

    if args.test:
        test_phone_connection(args.test, args.port)
        return

    if args.specs:
        print_camera_specs(args.width, args.height)
        return

    if args.export_config:
        host = args.export_config
        snippet = f"""
# Add to config.yaml or use via --camera-left, --camera-right
camera_left: {0}
camera_right: "http://{host}:8080/video"
camera_gesture: {0}
camera_width: {args.width}
camera_height: {args.height}
"""
        print(snippet)
        return

    print_camera_specs(args.width, args.height)
    print()
    print("Run with --guide for full setup instructions.")
    print("Run with --scan to find the phone on the network.")
    print("Run with --test <PHONE_IP> to check connectivity.")


if __name__ == "__main__":
    main()
