import cv2
import numpy as np


def project_3d_to_image(point_3d: np.ndarray,
                         K: np.ndarray, D: np.ndarray,
                         R: np.ndarray, T: np.ndarray) -> tuple:
    if point_3d is None:
        return None

    rvec, _ = cv2.Rodrigues(R)

    point_2d, _ = cv2.projectPoints(
        np.array([point_3d], dtype=np.float32),
        rvec, T, K, D)

    px, py = point_2d[0][0]
    return (int(px), int(py))


def draw_robot_overlay(frame: np.ndarray, robot_pos: np.ndarray,
                        K: np.ndarray, D: np.ndarray,
                        R: np.ndarray, T: np.ndarray,
                        alpha: float = 0.5) -> np.ndarray:

    pos_center = project_3d_to_image(robot_pos, K, D, R, T)
    if pos_center is None:
        return frame

    h = 100
    w = 60

    x1 = int(pos_center[0] - w / 2)
    y1 = int(pos_center[1] - h)
    x2 = int(pos_center[0] + w / 2)
    y2 = int(pos_center[1])

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y1 + h // 2), (50, 50, 200), -1)
    cv2.rectangle(overlay, (x1, y1 + h // 2), (x2, y2), (50, 50, 200), -1)
    cv2.circle(overlay, (pos_center[0], y1 + h // 4), 20, (100, 100, 255), -1)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame


def draw_balloon_overlay(frame: np.ndarray, balloon_pos: np.ndarray,
                          K: np.ndarray, D: np.ndarray,
                          R: np.ndarray, T: np.ndarray) -> np.ndarray:

    pos_center = project_3d_to_image(balloon_pos, K, D, R, T)
    if pos_center is None:
        return frame

    cv2.circle(frame, pos_center, 30, (0, 200, 0), 2)
    cv2.circle(frame, pos_center, 5, (0, 255, 0), -1)

    return frame


def draw_ar_overlay(frame: np.ndarray,
                     robot_pos: np.ndarray,
                     balloon_pos: np.ndarray,
                     K: np.ndarray, D: np.ndarray,
                     R: np.ndarray, T: np.ndarray) -> np.ndarray:

    frame = draw_robot_overlay(frame, robot_pos, K, D, R, T)
    if balloon_pos is not None:
        frame = draw_balloon_overlay(frame, balloon_pos, K, D, R, T)

    return frame
