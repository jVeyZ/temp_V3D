import cv2
import numpy as np
import os
import glob
import yaml
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capture.camera_stream import CameraStream, IPCameraStream, VideoFileStream


def _open_camera(source, width=640, height=480):
    if isinstance(source, str) and source.startswith("http"):
        return IPCameraStream(source, width=width, height=height)
    elif isinstance(source, str):
        return VideoFileStream(source)
    elif isinstance(source, int):
        return CameraStream(source, width=width, height=height)
    else:
        raise TypeError(f"Invalid camera source type: {type(source)}")


def capture_calibration_images(camera_left, camera_right,
                                chessboard_size: tuple, save_dir: str,
                                width: int = 640, height: int = 480):
    os.makedirs(save_dir, exist_ok=True)

    cap_left = _open_camera(camera_left, width, height)
    cap_right = _open_camera(camera_right, width, height)

    cap_left.start()
    cap_right.start()

    count = 0
    print("Press 'c' to capture a stereo pair, 'q' to quit")
    time.sleep(1.0)

    while True:
        frame_l = cap_left.read()
        frame_r = cap_right.read()

        if frame_l is None or frame_r is None:
            time.sleep(0.01)
            continue

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        ret_cl, corners_l = cv2.findChessboardCorners(
            gray_l, chessboard_size, None)
        ret_cr, corners_r = cv2.findChessboardCorners(
            gray_r, chessboard_size, None)

        if ret_cl:
            cv2.drawChessboardCorners(frame_l, chessboard_size, corners_l, ret_cl)
        if ret_cr:
            cv2.drawChessboardCorners(frame_r, chessboard_size, corners_r, ret_cr)

        display = np.hstack((frame_l, frame_r))
        cv2.putText(display, f"Pairs captured: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "L", (10, frame_l.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "R", (frame_r.shape[1] + 10, frame_r.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if ret_cl and ret_cr:
            cv2.putText(display, "READY - press 'c'", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Stereo Calibration Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and ret_cl and ret_cr:
            cv2.imwrite(os.path.join(save_dir, f"left_{count:03d}.png"), frame_l)
            cv2.imwrite(os.path.join(save_dir, f"right_{count:03d}.png"), frame_r)
            count += 1
            print(f"Captured pair {count}")
        elif key == ord('q'):
            break

    cap_left.stop()
    cap_right.stop()
    cv2.destroyAllWindows()
    print(f"Saved {count} stereo pairs to {save_dir}")


def _parse_source(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return val


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    capture_calibration_images(
        camera_left=_parse_source(config["camera_left"]),
        camera_right=_parse_source(config["camera_right"]),
        chessboard_size=tuple(config["chessboard_size"]),
        save_dir=config["calib_dir"],
        width=config.get("camera_width", 640),
        height=config.get("camera_height", 480)
    )


if __name__ == "__main__":
    main()
