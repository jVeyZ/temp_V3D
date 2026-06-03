import argparse
import yaml
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game.app import App, load_calibration


def _parse_camera_source(val: str):
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return val


def main():
    parser = argparse.ArgumentParser(
        description="Balloon Catch - 3D Gesture-Controlled Robot Game")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration YAML file")
    parser.add_argument("--camera-left", type=str, default=None,
                        help="Left camera: int=index, str=URL or video path (overrides config)")
    parser.add_argument("--camera-right", type=str, default=None,
                        help="Right camera: int=index, str=URL or video path (overrides config)")
    parser.add_argument("--simulate", action="store_true", default=None,
                        help="Run with prerecorded videos")
    parser.add_argument("--video-left", type=str, default=None,
                        help="Path to left video file for simulation")
    parser.add_argument("--video-right", type=str, default=None,
                        help="Path to right video file for simulation")
    parser.add_argument("--debug", action="store_true", default=None,
                        help="Enable debug mode with extra windows")
    parser.add_argument("--calib", type=str, default=None,
                        help="Path to calibration directory")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    cam_left = _parse_camera_source(args.camera_left)
    if cam_left is not None:
        config["camera_left"] = cam_left
    cam_right = _parse_camera_source(args.camera_right)
    if cam_right is not None:
        config["camera_right"] = cam_right
    if args.simulate is not None:
        config["simulate"] = args.simulate
    if args.video_left is not None:
        config["simulate_video_left"] = args.video_left
    if args.video_right is not None:
        config["simulate_video_right"] = args.video_right
    if args.debug is not None:
        config["debug"] = args.debug

    calib_dir = args.calib if args.calib else config["calib_dir"]
    calib_params, calib_maps = load_calibration(calib_dir)

    print("=" * 50)
    print("  Balloon Catch - 3D Gesture-Controlled Robot")
    print("=" * 50)
    print(f"  Left camera:  {config['camera_left']}")
    print(f"  Right camera: {config['camera_right']}")
    print(f"  Simulation:   {config['simulate']}")
    print(f"  Camera matrix L focal: {calib_params['K1'][0, 0]:.1f}")
    print(f"  Camera matrix R focal: {calib_params['K2'][0, 0]:.1f}")
    print(f"  Baseline: {np.linalg.norm(calib_params['T']):.3f}m")
    print("=" * 50)
    print("  Controls:")
    print("    q - Quit")
    print("    r - Reset game")
    print("  Gestures:")
    print("    Move hand LEFT/RIGHT  -> robot moves sideways")
    print("    Move hand FWD/BACK   -> robot moves depth")
    print("    Pinch thumb+index     -> GRAB (manual catch)")
    print("=" * 50)

    app = App(config, calib_params, calib_maps)

    try:
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
