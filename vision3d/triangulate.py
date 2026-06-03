import cv2
import numpy as np


def triangulate_points(pt_left: tuple, pt_right: tuple,
                       P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    if pt_left is None or pt_right is None:
        return None

    pts_l = np.array([[float(pt_left[0]), float(pt_left[1])]], dtype=np.float32).T
    pts_r = np.array([[float(pt_right[0]), float(pt_right[1])]], dtype=np.float32).T

    points_4d = cv2.triangulatePoints(P1, P2, pts_l, pts_r)
    points_3d = points_4d[:3] / points_4d[3]

    return points_3d.ravel()


def rectify_points(pt_left: tuple, pt_right: tuple,
                   K1: np.ndarray, D1: np.ndarray,
                   K2: np.ndarray, D2: np.ndarray,
                   R1: np.ndarray, R2: np.ndarray,
                   P1: np.ndarray, P2: np.ndarray,
                   map_l1: np.ndarray, map_l2: np.ndarray,
                   map_r1: np.ndarray, map_r2: np.ndarray) -> tuple:

    x_l, y_l = pt_left[:2]
    x_r, y_r = pt_right[:2]

    x_l_rect = map_l1[y_l, x_l]
    y_l_rect = map_l2[y_l, x_l]
    x_r_rect = map_r1[y_r, x_r]
    y_r_rect = map_r2[y_r, x_r]

    return (x_l_rect, y_l_rect), (x_r_rect, y_r_rect)


def compute_disparity(pt_left: tuple, pt_right: tuple) -> float:
    if pt_left is None or pt_right is None:
        return 0.0
    return float(pt_left[0] - pt_right[0])


def compute_depth(disparity: float, focal_length: float,
                  baseline: float) -> float:
    if abs(disparity) < 1e-6:
        return float('inf')
    return (focal_length * baseline) / disparity
