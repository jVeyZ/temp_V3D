import cv2
import numpy as np
import time
import queue
import threading

from capture.stereo_stream import StereoStream
from detection.balloon_detector import BalloonDetector
from detection.hand_detector import HandDetector
from detection.correspondence import find_correspondence_balloon
from vision3d.triangulate import triangulate_points
from scene.virtual_scene import VirtualScene
from scene.ar_renderer import draw_ar_overlay, project_3d_to_image
from gestures.gesture_classifier import GestureClassifier
from gestures.gesture_defs import GestureType, GESTURE_NAMES
from game.game_state import GameState, GamePhase
from game.score_manager import ScoreManager
class App:
    def __init__(self, config: dict, calib_params: dict, calib_maps: dict):

        self.config = config
        self.calib = calib_params
        self.maps = calib_maps

        self.stereo = StereoStream(
            camera_left=config["camera_left"],
            camera_right=config["camera_right"],
            simulate=config["simulate"],
            video_left=config["simulate_video_left"],
            video_right=config["simulate_video_right"],
            width=config.get("camera_width", 640),
            height=config.get("camera_height", 480))

        self.balloon_detector_l = BalloonDetector(
            hue_low=config["balloon"]["hue_low"],
            hue_high=config["balloon"]["hue_high"],
            sat_low=config["balloon"]["sat_low"],
            val_low=config["balloon"]["val_low"],
            min_area=config["balloon"]["min_contour_area"])

        self.balloon_detector_r = BalloonDetector(
            hue_low=config["balloon"]["hue_low"],
            hue_high=config["balloon"]["hue_high"],
            sat_low=config["balloon"]["sat_low"],
            val_low=config["balloon"]["val_low"],
            min_area=config["balloon"]["min_contour_area"])

        self.hand_detector = HandDetector()

        self.gesture_classifier = GestureClassifier(
            move_threshold_x=config["gesture"]["move_threshold_x"],
            move_threshold_z=config["gesture"]["move_threshold_z"],
            pinch_threshold=config["gesture"]["pinch_threshold"],
            neutral_zone_duration=config["gesture"]["neutral_zone_duration"])

        self.virtual_scene = VirtualScene(
            floor_size=config["rendering"]["floor_size"],
            fps=config["rendering"]["scene_fps"])

        self.game_state = GameState(
            balloon_min_height=config["game"]["balloon_min_height"],
            balloon_launch_height=config["game"]["balloon_launch_height"],
            hitbox_radius=config["game"]["hitbox_radius"],
            score_reset_delay=config["game"]["score_reset_delay"])

        self.score_manager = ScoreManager(
            hitbox_radius=config["game"]["hitbox_radius"])

        self.running = False
        self.balloon_3d = None
        self.robot_cmd = (0.0, 0.0, 0.0, None)
        self.current_gesture = GestureType.IDLE

        self._display_lock = threading.Lock()
        self._display_frame_l = None
        self._display_frame_r = None

    def run(self):
        self.stereo.start()
        self.virtual_scene.start()
        self.running = True

        threads = [
            threading.Thread(target=self._balloon_thread, daemon=True),
            threading.Thread(target=self._gesture_thread, daemon=True),
            threading.Thread(target=self._ar_thread, daemon=True),
            threading.Thread(target=self._display_thread, daemon=True),
        ]

        for t in threads:
            t.start()

        fps_timer = time.time()
        frame_count = 0

        try:
            while self.running:
                frame_l, frame_r = self.stereo.get_frames(timeout=0.1)

                if frame_l is None or frame_r is None:
                    time.sleep(0.001)
                    continue

                self._display_lock.acquire()
                self._display_frame_l = frame_l.copy()
                self._display_frame_r = frame_r.copy()
                self._display_lock.release()

                self.game_state.update(
                    balloon_3d=self.balloon_3d,
                    robot_hitbox_center=self.virtual_scene.get_hitbox_center())

                dx, dy, dz, discrete = self.robot_cmd
                if discrete == GestureType.RESET:
                    self.game_state.reset()
                    self.balloon_detector_l.reset_kalman()
                    self.balloon_detector_r.reset_kalman()
                elif discrete == GestureType.GRAB:
                    robot_pos = self.virtual_scene.get_hitbox_center()
                    balloon_pos = self.virtual_scene.get_balloon_pos()
                    if balloon_pos is not None:
                        from vision3d.transforms import distance
                        if distance(robot_pos, balloon_pos) < self.config["game"]["hitbox_radius"]:
                            self.score_manager.evaluate(balloon_pos, robot_pos)
                else:
                    speed = self.config["rendering"]["robot_move_speed"]
                    self.virtual_scene.update_robot(
                        dx * speed, dy * speed, dz * speed)

                frame_count += 1
                now = time.time()
                if now - fps_timer >= 1.0:
                    fps = frame_count / (now - fps_timer)
                    fps_timer = now
                    frame_count = 0

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('r'):
                    self.game_state.reset()
                    self.balloon_detector_l.reset_kalman()
                    self.balloon_detector_r.reset_kalman()

        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            for t in threads:
                t.join(timeout=1.0)
            self.stereo.stop()
            self.virtual_scene.stop()
            self.hand_detector.release()
            cv2.destroyAllWindows()

    def _balloon_thread(self):
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = np.hstack((self.calib["R"], self.calib["T"]))

        while self.running:
            frame_l, frame_r = self.stereo.get_frames(timeout=0.1)
            if frame_l is None or frame_r is None:
                time.sleep(0.001)
                continue

            cx_l, cy_l, r_l, mask_l = self.balloon_detector_l.detect(frame_l)
            pt_l = (cx_l, cy_l) if cx_l is not None else None

            cx_r, cy_r, r_r, mask_r = self.balloon_detector_r.detect(frame_r)
            pt_r = (cx_r, cy_r) if cx_r is not None else None

            if pt_l is not None and pt_r is not None:
                if abs(cy_l - cy_r) > 40:
                    self.balloon_3d = None
                    continue
                self.balloon_3d = triangulate_points(pt_l, pt_r, P1, P2)
            else:
                self.balloon_3d = None

            if self.balloon_3d is not None:
                self.virtual_scene.update_balloon(self.balloon_3d)

    def _gesture_thread(self):
        while self.running:
            with self._display_lock:
                if self._display_frame_l is not None:
                    frame = self._display_frame_l.copy()
                else:
                    frame = None

            if frame is None:
                time.sleep(0.001)
                continue

            hands = self.hand_detector.detect(frame)
            self.robot_cmd = self.gesture_classifier.classify(hands)
            self.current_gesture = self.gesture_classifier.current_gesture

    def _ar_thread(self):
        K2 = self.calib["K2"]
        D2 = self.calib["D2"]
        R = self.calib["R"]
        T = self.calib["T"]

        while self.running:
            with self._display_lock:
                if self._display_frame_r is not None:
                    frame = self._display_frame_r.copy()
                else:
                    frame = None

            if frame is None:
                time.sleep(0.001)
                continue

            robot_pos = self.virtual_scene.get_hitbox_center()
            balloon_pos = self.virtual_scene.get_balloon_pos()

            frame = draw_ar_overlay(frame, robot_pos, balloon_pos,
                                     K2, D2, R, T)

            self._draw_ar_hud(frame)

            cv2.imshow("AR View (Camera Right)", frame)

            with self._display_lock:
                if self._display_frame_l is not None:
                    frame_l = self._display_frame_l.copy()
                else:
                    continue

            hands = self.hand_detector.detect(frame_l)
            frame_l = self.hand_detector.draw_landmarks(frame_l, hands)

            self._draw_gesture_hud(frame_l)

            cv2.imshow("Gesture View (Camera Left)", frame_l)

    def _display_thread(self):
        return

    def _draw_ar_hud(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        phase_names = {
            GamePhase.IDLE: "IDLE - Waiting for toss",
            GamePhase.CALIBRATING: "CALIBRATING...",
            GamePhase.TRACKING: "TRACKING balloon",
            GamePhase.SCORING: "SCORING...",
            GamePhase.RESULT: "RESULT"
        }
        phase_str = phase_names.get(self.game_state.phase, "")

        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 1.0, 0, frame)

        cv2.putText(frame, phase_str, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Score: {self.game_state.score}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        if self.balloon_3d is not None:
            pos_str = (f"Balloon: ({self.balloon_3d[0]:.2f}, "
                       f"{self.balloon_3d[1]:.2f}, {self.balloon_3d[2]:.2f})m")
            cv2.putText(frame, pos_str, (w // 2, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if self.game_state.phase == GamePhase.RESULT:
            if self.game_state.last_hit:
                hit_text = "CAUGHT! +1"
                color = (0, 255, 0)
            else:
                hit_text = "MISSED!"
                color = (0, 0, 255)

            text_size = cv2.getTextSize(hit_text, cv2.FONT_HERSHEY_SIMPLEX,
                                        2.0, 4)[0]
            tx = (w - text_size[0]) // 2
            ty = h // 2
            cv2.putText(frame, hit_text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)

    def _draw_gesture_hud(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 1.0, 0, frame)

        gesture_name = GESTURE_NAMES.get(
            self.current_gesture, "UNKNOWN")
        cv2.putText(frame, f"Gesture: {gesture_name}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        neutral = self.gesture_classifier.get_neutral_zone()
        calibrating = not self.gesture_classifier.is_calibrated()
        if calibrating:
            cv2.putText(frame, "Calibrating neutral zone...",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        elif neutral is not None:
            cv2.circle(frame, neutral, 10, (0, 255, 255), 2)
            cv2.line(frame,
                     (neutral[0] - 15, neutral[1]),
                     (neutral[0] + 15, neutral[1]),
                     (0, 255, 255), 1)
            cv2.line(frame,
                     (neutral[0], neutral[1] - 15),
                     (neutral[0], neutral[1] + 15),
                     (0, 255, 255), 1)


def load_calibration(calib_dir: str) -> tuple:
    import os
    params_path = os.path.join(calib_dir, "stereo_params.npz")
    maps_path = os.path.join(calib_dir, "rectify_maps.npz")

    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"Calibration not found at {params_path}. "
            f"Run: python calibration/capture_calib.py && "
            f"python calibration/stereo_calibrate.py")

    params = dict(np.load(params_path))
    maps = dict(np.load(maps_path)) if os.path.exists(maps_path) else {}

    return params, maps
