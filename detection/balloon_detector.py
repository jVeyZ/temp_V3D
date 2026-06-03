import cv2
import numpy as np
from collections import deque


class BalloonDetector:
    def __init__(self, hue_low: int = 35, hue_high: int = 85,
                 sat_low: int = 80, val_low: int = 50,
                 min_area: int = 200):
        self.hue_low = hue_low
        self.hue_high = hue_high
        self.sat_low = sat_low
        self.val_low = val_low
        self.min_area = min_area

        self._kalman = cv2.KalmanFilter(4, 2)
        self._kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)
        self._kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)
        self._kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self._kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
        self._kalman_initialized = False

    def detect(self, frame: np.ndarray) -> tuple:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([self.hue_low, self.sat_low, self.val_low])
        upper = np.array([self.hue_high, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx, cy, radius = None, None, None
        mask_debug = mask

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                (x, y), r = cv2.minEnclosingCircle(contour)
                radius = int(r)
                break

        return cx, cy, radius, mask_debug

    def track(self, cx: float, cy: float) -> tuple:
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])

        if not self._kalman_initialized:
            self._kalman.statePre = np.array(
                [[np.float32(cx)], [np.float32(cy)],
                 [np.float32(0)], [np.float32(0)]])
            self._kalman.statePost = np.array(
                [[np.float32(cx)], [np.float32(cy)],
                 [np.float32(0)], [np.float32(0)]])
            self._kalman_initialized = True

        predicted = self._kalman.predict()
        corrected = self._kalman.correct(measurement)

        px, py = predicted[0][0], predicted[1][0]
        kx, ky = corrected[0][0], corrected[1][0]

        return kx, ky, px, py

    def reset_kalman(self):
        self._kalman_initialized = False
