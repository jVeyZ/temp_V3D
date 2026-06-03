import numpy as np


def camera_to_world(point_3d: np.ndarray, R: np.ndarray,
                    T: np.ndarray) -> np.ndarray:
    if point_3d is None:
        return None
    R_inv = R.T
    return R_inv @ (point_3d - T.ravel())


def world_to_camera(point_world: np.ndarray, R: np.ndarray,
                    T: np.ndarray) -> np.ndarray:
    return R @ point_world + T.ravel()


def normalize_point(point: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(point)
    if norm < 1e-10:
        return point
    return point / norm


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def clamp_to_floor(point_3d: np.ndarray, floor_z: float = 0.0) -> np.ndarray:
    if point_3d is None:
        return None
    point = point_3d.copy()
    point[2] = max(point[2], floor_z)
    return point
