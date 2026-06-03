import cv2
import numpy as np
import os
import glob
import yaml


def load_image_pairs(calib_dir: str) -> tuple:
    left_files = sorted(glob.glob(os.path.join(calib_dir, "left_*.png")))
    right_files = sorted(glob.glob(os.path.join(calib_dir, "right_*.png")))

    if len(left_files) == 0 or len(right_files) == 0:
        raise FileNotFoundError(
            f"No calibration images found in {calib_dir}. "
            f"Run capture_calib.py first.")

    if len(left_files) != len(right_files):
        raise ValueError(
            f"Mismatched pairs: {len(left_files)} left, {len(right_files)} right")

    left_imgs = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in left_files]
    right_imgs = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in right_files]

    return left_imgs, right_imgs


def calibrate_stereo(left_images: list, right_images: list,
                     chessboard_size: tuple, square_size: float,
                     output_dir: str) -> dict:

    pattern_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3),
                               np.float32)
    pattern_points[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points_l = []
    img_points_r = []

    h, w = left_images[0].shape[:2]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for img_l, img_r in zip(left_images, right_images):
        ret_l, corners_l = cv2.findChessboardCorners(img_l, chessboard_size, None)
        ret_r, corners_r = cv2.findChessboardCorners(img_r, chessboard_size, None)

        if ret_l and ret_r:
            corners_l = cv2.cornerSubPix(img_l, corners_l, (5, 5), (-1, -1), criteria)
            corners_r = cv2.cornerSubPix(img_r, corners_r, (5, 5), (-1, -1), criteria)
            obj_points.append(pattern_points)
            img_points_l.append(corners_l)
            img_points_r.append(corners_r)

    if len(obj_points) < 5:
        raise RuntimeError(
            f"Only {len(obj_points)} valid pairs found. Need at least 5.")

    camera_matrix_l = cv2.initCameraMatrix2D(
        [pattern_points], [img_points_l[0]], (w, h), 0)
    camera_matrix_r = cv2.initCameraMatrix2D(
        [pattern_points], [img_points_r[0]], (w, h), 0)

    flags = cv2.CALIB_FIX_INTRINSIC
    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_l, img_points_r,
        camera_matrix_l, None,
        camera_matrix_r, None,
        (w, h), criteria=criteria, flags=flags)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, (w, h), R, T, alpha=0)

    map_l1, map_l2 = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
    map_r1, map_r2 = cv2.initUndistortRectifyMap(
        K2, D2, R2, P2, (w, h), cv2.CV_32FC1)

    os.makedirs(output_dir, exist_ok=True)

    params_path = os.path.join(output_dir, "stereo_params.npz")
    np.savez(params_path,
             K1=K1, D1=D1, K2=K2, D2=D2,
             R=R, T=T, E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)

    maps_path = os.path.join(output_dir, "rectify_maps.npz")
    np.savez(maps_path,
             map_l1=map_l1, map_l2=map_l2,
             map_r1=map_r1, map_r2=map_r2)

    total_error = 0.0
    for i in range(len(obj_points)):
        proj_l, _ = cv2.projectPoints(
            obj_points[i], cv2.Rodrigues(np.eye(3))[0], np.zeros(3),
            K1, D1)
        proj_r, _ = cv2.projectPoints(
            obj_points[i], R, T, K2, D2)
        error_l = cv2.norm(img_points_l[i], proj_l, cv2.NORM_L2) / len(proj_l)
        error_r = cv2.norm(img_points_r[i], proj_r, cv2.NORM_L2) / len(proj_r)
        total_error += (error_l + error_r) / 2.0

    mean_error = total_error / len(obj_points)

    print(f"Stereo calibration completed.")
    print(f"  Image pairs used: {len(obj_points)}")
    print(f"  Image size: {w}x{h}")
    print(f"  RMS reprojection error: {ret:.4f} px")
    print(f"  Mean per-corner error: {mean_error:.4f} px")
    print(f"  Saved to: {output_dir}")
    print(f"  Translation vector T (m): {T.ravel()}")

    return {
        "K1": K1, "D1": D1, "K2": K2, "D2": D2,
        "R": R, "T": T, "E": E, "F": F,
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
        "rms_error": ret,
        "mean_corner_error": mean_error
    }


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    left_imgs, right_imgs = load_image_pairs(config["calib_dir"])
    calibrate_stereo(
        left_images=left_imgs,
        right_images=right_imgs,
        chessboard_size=tuple(config["chessboard_size"]),
        square_size=config["square_size"],
        output_dir=config["calib_dir"]
    )


if __name__ == "__main__":
    main()
