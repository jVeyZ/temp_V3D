import cv2
import numpy as np
import queue
import threading
import time

from capture.camera_stream import CameraStream, IPCameraStream, VideoFileStream


def _create_stream(source, simulate=False, video_path="", width=640, height=480):
    if simulate:
        if not video_path:
            raise ValueError("Video path required for simulation mode")
        return VideoFileStream(video_path)
    elif isinstance(source, str) and source.startswith("http"):
        return IPCameraStream(source, width=width, height=height)
    elif isinstance(source, str):
        return VideoFileStream(source)
    elif isinstance(source, int):
        return CameraStream(source, width=width, height=height)
    else:
        raise TypeError(f"Invalid camera source type: {type(source)}")


class StereoStream:
    def __init__(self, camera_left, camera_right,
                 simulate: bool = False,
                 video_left: str = "", video_right: str = "",
                 width: int = 640, height: int = 480):
        self.simulate = simulate

        self.cam_left = _create_stream(
            camera_left, simulate=simulate, video_path=video_left,
            width=width, height=height)
        self.cam_right = _create_stream(
            camera_right, simulate=simulate, video_path=video_right,
            width=width, height=height)

        self.frame_queue = queue.Queue(maxsize=1)
        self.running = False

    def start(self):
        self.cam_left.start()
        self.cam_right.start()
        self.running = True
        threading.Thread(target=self._sync_loop, daemon=True).start()
        return self

    def _sync_loop(self):
        while self.running:
            frame_l = self.cam_left.read()
            frame_r = self.cam_right.read()

            if frame_l is not None and frame_r is not None:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put((frame_l, frame_r))

            time.sleep(0.001)

    def get_frames(self, timeout: float = 0.1):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None

    def stop(self):
        self.running = False
        self.cam_left.stop()
        self.cam_right.stop()

    @property
    def size(self):
        return self.cam_left.size
