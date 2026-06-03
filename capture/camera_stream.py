import cv2
import numpy as np
import threading
import queue
import time
import urllib.request
import socket


class CameraStream:
    def __init__(self, camera_id: int, width: int = 640, height: int = 480):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")

        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame.copy()
            else:
                time.sleep(0.001)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.cap.release()

    @property
    def size(self):
        return (self.width, self.height)


class IPCameraStream:
    def __init__(self, url: str, width: int = 640, height: int = 480,
                 reconnect_delay: float = 1.0, timeout: float = 5.0):

        self.url = url
        self.target_width = width
        self.target_height = height
        self.reconnect_delay = reconnect_delay
        self.timeout = timeout

        self._backend = cv2.CAP_FFMPEG

        self.cap = cv2.VideoCapture(url, self._backend)
        if width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open IP camera stream at {url}\n"
                f"Check: (1) phone is on same WiFi, "
                f"(2) IP Webcam app is running, (3) URL is correct")

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = actual_w if actual_w > 0 else width
        self.height = actual_h if actual_h > 0 else height

        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self._thread = None
        self._reconnect_count = 0

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        consecutive_failures = 0
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame.copy()
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    self._reconnect()
                    consecutive_failures = 0
                time.sleep(0.001)

    def _reconnect(self):
        self._reconnect_count += 1
        old_cap = self.cap
        time.sleep(self.reconnect_delay)
        self.cap = cv2.VideoCapture(self.url, self._backend)
        if self.cap.isOpened():
            if self.target_width and self.target_height:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.cap.release()

    @property
    def size(self):
        return (self.width, self.height)

    @staticmethod
    def test_url(url: str, timeout: float = 3.0) -> bool:
        try:
            req = urllib.request.Request(url)
            urllib.request.urlopen(req, timeout=timeout)
            return True
        except Exception:
            return False

    @staticmethod
    def discover_phone(timeout: float = 2.0) -> list:
        common_ports = [8080, 4747, 8888, 8081, 5000]
        base_urls = []

        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        prefix = ".".join(local_ip.split(".")[:3])

        for last in range(1, 255):
            ip = f"{prefix}.{last}"
            if ip == local_ip:
                continue
            for port in common_ports:
                url = f"http://{ip}:{port}"
                try:
                    urllib.request.urlopen(url, timeout=timeout)
                    base_urls.append(url)
                except Exception:
                    pass

        return base_urls


class VideoFileStream:
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.running = False
        self.lock = threading.Lock()
        self.frame = None
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            with self.lock:
                self.frame = frame.copy()

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.cap.release()

    @property
    def size(self):
        return (self.width, self.height)
