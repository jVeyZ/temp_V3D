from enum import Enum, auto


class GestureType(Enum):
    IDLE = auto()
    MOVE_LEFT = auto()
    MOVE_RIGHT = auto()
    MOVE_FORWARD = auto()
    MOVE_BACKWARD = auto()
    GRAB = auto()
    RESET = auto()


GESTURE_NAMES = {
    GestureType.IDLE: "IDLE",
    GestureType.MOVE_LEFT: "LEFT",
    GestureType.MOVE_RIGHT: "RIGHT",
    GestureType.MOVE_FORWARD: "FWD",
    GestureType.MOVE_BACKWARD: "BACK",
    GestureType.GRAB: "GRAB",
    GestureType.RESET: "RESET"
}
