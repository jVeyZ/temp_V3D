import numpy as np
from vision3d.transforms import distance


class ScoreManager:
    def __init__(self, hitbox_radius: float = 0.2):
        self.hitbox_radius = hitbox_radius
        self.score = 0
        self.streak = 0
        self.best_streak = 0
        self.total_attempts = 0
        self.hit_history = []

    def evaluate(self, balloon_pos: np.ndarray,
                 hitbox_center: np.ndarray) -> bool:

        self.total_attempts += 1

        if balloon_pos is None or hitbox_center is None:
            self._miss()
            return False

        dist = distance(balloon_pos, hitbox_center)
        hit = dist <= self.hitbox_radius

        if hit:
            self.score += 1
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
        else:
            self._miss()

        self.hit_history.append({"hit": hit, "distance": dist})
        if len(self.hit_history) > 100:
            self.hit_history.pop(0)

        return hit

    def _miss(self):
        self.streak = 0

    def get_accuracy(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        hits = sum(1 for h in self.hit_history[-20:] if h["hit"])
        return hits / min(20, self.total_attempts) * 100

    def reset(self):
        self.score = 0
        self.streak = 0
        self.best_streak = 0
        self.total_attempts = 0
        self.hit_history.clear()
