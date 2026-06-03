from enum import Enum, auto
import time
import numpy as np
from vision3d.transforms import distance


class GamePhase(Enum):
    IDLE = auto()
    CALIBRATING = auto()
    TRACKING = auto()
    SCORING = auto()
    RESULT = auto()


class GameState:
    def __init__(self,
                 balloon_min_height: float = 0.3,
                 balloon_launch_height: float = 0.5,
                 hitbox_radius: float = 0.2,
                 score_reset_delay: float = 3.0):

        self.balloon_min_height = balloon_min_height
        self.balloon_launch_height = balloon_launch_height
        self.hitbox_radius = hitbox_radius
        self.score_reset_delay = score_reset_delay

        self.phase = GamePhase.IDLE
        self.score = 0
        self.streak = 0
        self.last_result_time = 0.0
        self.last_hit = False

        self.balloon_pos = None
        self.robot_hitbox_center = None
        self._phase_start_time = time.time()

    def update(self, balloon_3d: np.ndarray,
               robot_hitbox_center: np.ndarray):

        self.balloon_pos = balloon_3d
        self.robot_hitbox_center = robot_hitbox_center

        if self.phase == GamePhase.IDLE:
            if balloon_3d is not None and balloon_3d[2] > self.balloon_launch_height:
                self.transition_to(GamePhase.TRACKING)

        elif self.phase == GamePhase.TRACKING:
            if balloon_3d is not None and balloon_3d[2] <= self.balloon_min_height:
                self.transition_to(GamePhase.SCORING)
            elif balloon_3d is None:
                self.transition_to(GamePhase.IDLE)

        elif self.phase == GamePhase.SCORING:
            self._evaluate_hit()
            self.transition_to(GamePhase.RESULT)
            self.last_result_time = time.time()

        elif self.phase == GamePhase.RESULT:
            if time.time() - self.last_result_time > self.score_reset_delay:
                self.transition_to(GamePhase.IDLE)

    def _evaluate_hit(self):
        if self.balloon_pos is None or self.robot_hitbox_center is None:
            self.last_hit = False
            self.streak = 0
            return

        dist = distance(self.balloon_pos, self.robot_hitbox_center)
        if dist <= self.hitbox_radius:
            self.score += 1
            self.streak += 1
            self.last_hit = True
        else:
            self.last_hit = False
            self.streak = 0

    def transition_to(self, phase: GamePhase):
        self.phase = phase
        self._phase_start_time = time.time()

    def reset(self):
        self.phase = GamePhase.IDLE
        self.score = 0
        self.streak = 0
        self.last_hit = False
        self.balloon_pos = None
