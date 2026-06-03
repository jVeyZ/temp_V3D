import numpy as np
import cv2


def find_correspondence_balloon(pt_left: tuple, frame_right: np.ndarray,
                                 balloon_detector,
                                 epipolar_line_y_tolerance: int = 30) -> tuple:
    if pt_left[0] is None:
        return None, None, None

    cx_right, cy_right, radius_right, mask_right = balloon_detector.detect(
        frame_right)

    if cx_right is not None and cy_right is not None:
        if abs(cy_right - pt_left[1]) <= epipolar_line_y_tolerance:
            return cx_right, cy_right, radius_right

    return cx_right, cy_right, None


def check_epipolar_consistency(pt_left: tuple, pt_right: tuple,
                                min_disparity: float = 0.0,
                                max_vertical_diff: float = 5.0) -> bool:
    if pt_left is None or pt_right is None:
        return False

    disparity = pt_left[0] - pt_right[0]
    vertical_diff = abs(pt_left[1] - pt_right[1])

    if vertical_diff > max_vertical_diff:
        return False
    if disparity < min_disparity:
        return False
    return True
