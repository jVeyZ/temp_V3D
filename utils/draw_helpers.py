import cv2
import numpy as np


def overlay_text(frame: np.ndarray, text: str, position: tuple,
                 font_scale: float = 0.6,
                 color: tuple = (255, 255, 255),
                 thickness: int = 1) -> np.ndarray:
    cv2.putText(frame, text, position,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return frame


def draw_crosshair(frame: np.ndarray, center: tuple, size: int = 10,
                   color: tuple = (0, 255, 255),
                   thickness: int = 2) -> np.ndarray:
    x, y = center
    cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
    cv2.line(frame, (x, y - size), (x, y + size), color, thickness)
    return frame


def draw_bounding_box(frame: np.ndarray, top_left: tuple,
                      bottom_right: tuple,
                      color: tuple = (0, 255, 0),
                      thickness: int = 2) -> np.ndarray:
    cv2.rectangle(frame, top_left, bottom_right, color, thickness)
    return frame


def draw_hud_background(frame: np.ndarray, rect: tuple,
                        alpha: float = 0.5) -> np.ndarray:
    x, y, w, h = rect
    sub = frame[y:y+h, x:x+w].copy()
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    result = cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0)
    frame[y:y+h, x:x+w] = result
    return frame


def draw_epipolar_line(frame: np.ndarray, line_coeff: np.ndarray,
                       color: tuple = (255, 255, 0)) -> np.ndarray:
    h, w = frame.shape[:2]
    a, b, c = line_coeff
    x0 = 0
    y0 = int(-c / b) if abs(b) > 1e-6 else 0
    x1 = w
    y1 = int(-(a * w + c) / b) if abs(b) > 1e-6 else h
    cv2.line(frame, (x0, y0), (x1, y1), color, 1)
    return frame


def create_info_overlay(text_lines: list, frame_width: int,
                        line_height: int = 25) -> np.ndarray:
    h = len(text_lines) * line_height + 20
    overlay = np.zeros((h, frame_width, 3), dtype=np.uint8)

    for i, text in enumerate(text_lines):
        cv2.putText(overlay, text, (10, 20 + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return overlay
