import cv2
import numpy as np
import mediapipe as mp


class HandDetector:

    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    INDEX_MCP = 5

    def __init__(self, static_image_mode: bool = False,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)

    def detect(self, frame: np.ndarray) -> list:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        landmarks_list = []

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(
                    results.multi_hand_landmarks):
                handedness = results.multi_handedness[hand_idx]
                label = handedness.classification[0].label

                lm = {}
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    h, w, _ = frame.shape
                    lm[idx] = {
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "px": int(landmark.x * w),
                        "py": int(landmark.y * h)
                    }

                landmarks_list.append({
                    "hand": label,
                    "landmarks": lm
                })

        return landmarks_list

    @staticmethod
    def draw_landmarks(frame: np.ndarray, landmarks_list: list) -> np.ndarray:
        for hand_data in landmarks_list:
            lm = hand_data["landmarks"]
            color = (0, 255, 0) if hand_data["hand"] == "Left" else (0, 0, 255)

            for idx in range(21):
                if idx in lm:
                    pt = lm[idx]
                    cv2.circle(frame, (pt["px"], pt["py"]), 4, color, -1)

            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (0, 17)
            ]
            for start_idx, end_idx in connections:
                if start_idx in lm and end_idx in lm:
                    cv2.line(frame,
                             (lm[start_idx]["px"], lm[start_idx]["py"]),
                             (lm[end_idx]["px"], lm[end_idx]["py"]),
                             color, 2)

            wrist = lm[0]
            cv2.putText(frame, hand_data["hand"],
                        (wrist["px"] + 10, wrist["py"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return frame

    def release(self):
        self.hands.close()
