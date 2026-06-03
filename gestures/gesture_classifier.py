import numpy as np
import time
from gestures.gesture_defs import GestureType

Z_INDEX = 2


class GestureClassifier:
    def __init__(self,
                 move_threshold_x: float = 50.0,
                 move_threshold_z: float = 40.0,
                 pinch_threshold: float = 30.0,
                 neutral_zone_duration: float = 2.0):

        self.move_threshold_x = move_threshold_x
        self.move_threshold_z = move_threshold_z
        self.pinch_threshold = pinch_threshold
        self.neutral_zone_duration = neutral_zone_duration

        self._neutral_wrist = None
        self._neutral_start_time = time.time()
        self._neutral_calibrated = False
        self._wrist_buffer = []

        self.current_gesture = GestureType.IDLE
        self.robot_command = (0.0, 0.0, 0.0, None)

        self._alpha = 0.7

    def classify(self, landmarks_list: list) -> tuple:
        dominant_hand = self._get_dominant_hand(landmarks_list)
        if dominant_hand is None:
            self.current_gesture = GestureType.IDLE
            self.robot_command = (0.0, 0.0, 0.0, None)
            return self.robot_command

        lm = dominant_hand["landmarks"]
        wrist = lm[0]

        self._calibrate_neutral(wrist)

        discrete_gesture = self._check_discrete_gestures(lm)
        if discrete_gesture is not None:
            self.current_gesture = discrete_gesture
            if discrete_gesture == GestureType.GRAB:
                self.robot_command = (0.0, 0.0, 0.0, GestureType.GRAB)
            elif discrete_gesture == GestureType.RESET:
                self.robot_command = (0.0, 0.0, 0.0, GestureType.RESET)
            return self.robot_command

        dx = wrist["px"] - self._neutral_wrist["px"]
        dz = wrist["z"] - self._neutral_wrist["z"]

        if abs(dx) > self.move_threshold_x:
            if dx < -self.move_threshold_x:
                self.current_gesture = GestureType.MOVE_LEFT
                self.robot_command = (-0.05, 0.0, 0.0, None)
            elif dx > self.move_threshold_x:
                self.current_gesture = GestureType.MOVE_RIGHT
                self.robot_command = (0.05, 0.0, 0.0, None)
        elif abs(dz) > self.move_threshold_z:
            if dz < -self.move_threshold_z:
                self.current_gesture = GestureType.MOVE_FORWARD
                self.robot_command = (0.0, 0.0, -0.05, None)
            elif dz > self.move_threshold_z:
                self.current_gesture = GestureType.MOVE_BACKWARD
                self.robot_command = (0.0, 0.0, 0.05, None)
        else:
            self.current_gesture = GestureType.IDLE
            self.robot_command = (0.0, 0.0, 0.0, None)

        return self.robot_command

    def _get_dominant_hand(self, landmarks_list: list) -> dict:
        if not landmarks_list:
            return None

        right_hands = [h for h in landmarks_list if h["hand"] == "Right"]
        if right_hands:
            return right_hands[0]

        return landmarks_list[0]

    def _calibrate_neutral(self, wrist: dict):
        if self._neutral_calibrated:
            self._neutral_wrist["px"] = (
                self._alpha * self._neutral_wrist["px"] +
                (1 - self._alpha) * wrist["px"])
            self._neutral_wrist["py"] = (
                self._alpha * self._neutral_wrist["py"] +
                (1 - self._alpha) * wrist["py"])
            self._neutral_wrist["z"]  = (
                self._alpha * self._neutral_wrist["z"]  +
                (1 - self._alpha) * wrist["z"])
            return

        self._wrist_buffer.append({
            "px": wrist["px"],
            "py": wrist["py"],
            "z":  wrist["z"]
        })

        elapsed = time.time() - self._neutral_start_time
        if elapsed >= self.neutral_zone_duration and len(self._wrist_buffer) > 10:
            px = np.mean([w["px"] for w in self._wrist_buffer])
            py = np.mean([w["py"] for w in self._wrist_buffer])
            z  = np.mean([w["z"]  for w in self._wrist_buffer])
            self._neutral_wrist = {"px": px, "py": py, "z": z}
            self._neutral_calibrated = True

    def _check_discrete_gestures(self, lm: dict) -> GestureType:
        thumb_tip = lm[4]
        index_tip = lm[8]
        thumb_mcp = lm[2]
        index_mcp = lm[5]
        wrist = lm[0]

        pinch_dist = np.sqrt(
            (thumb_tip["px"] - index_tip["px"]) ** 2 +
            (thumb_tip["py"] - index_tip["py"]) ** 2)

        if pinch_dist < self.pinch_threshold:
            return GestureType.GRAB

        palm_x = (thumb_mcp["px"] + index_mcp["px"]) / 2
        palm_y = (thumb_mcp["py"] + index_mcp["py"]) / 2
        palm_to_wrist = np.sqrt((palm_x - wrist["px"]) ** 2 +
                                 (palm_y - wrist["py"]) ** 2)
        if palm_to_wrist > 200:
            return GestureType.RESET

        return None

    def get_neutral_zone(self) -> tuple:
        if self._neutral_wrist is not None:
            return (int(self._neutral_wrist["px"]),
                    int(self._neutral_wrist["py"]))
        return None

    def is_calibrated(self) -> bool:
        return self._neutral_calibrated

    def reset_neutral(self):
        self._neutral_calibrated = False
        self._neutral_wrist = None
        self._wrist_buffer.clear()
        self._neutral_start_time = time.time()
