import open3d as o3d
import numpy as np
import threading
import queue
import time


class VirtualScene:
    def __init__(self, floor_size: float = 3.0, fps: float = 30.0):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Virtual Scene - Digital Twin",
                                width=800, height=600)

        self.floor_size = floor_size
        self.fps = fps
        self.frame_delay = 1.0 / fps

        self.robot_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.balloon_pos = np.array([1.0, 0.0, 1.0], dtype=np.float64)
        self.balloon_visible = False

        self.command_queue = queue.Queue(maxsize=1)
        self.running = False
        self._thread = None

        self._create_scene()

    def _create_scene(self):
        floor = o3d.geometry.TriangleMesh.create_box(
            width=self.floor_size, height=0.02, depth=self.floor_size)
        floor.translate(np.array([-self.floor_size / 2, -0.02, -self.floor_size / 2]))
        floor.paint_uniform_color([0.3, 0.3, 0.3])
        self.vis.add_geometry(floor, reset_bounding_box=False)

        grid_size = int(self.floor_size)
        for i in range(grid_size + 1):
            offset = -self.floor_size / 2 + i
            p1 = np.array([offset, 0.0, -self.floor_size / 2])
            p2 = np.array([offset, 0.0, self.floor_size / 2])
            line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([p1, p2]),
                lines=o3d.utility.Vector2iVector([[0, 1]]))
            line.paint_uniform_color([0.5, 0.5, 0.5])
            self.vis.add_geometry(line, reset_bounding_box=False)

            p1 = np.array([-self.floor_size / 2, 0.0, offset])
            p2 = np.array([self.floor_size / 2, 0.0, offset])
            line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([p1, p2]),
                lines=o3d.utility.Vector2iVector([[0, 1]]))
            line.paint_uniform_color([0.5, 0.5, 0.5])
            self.vis.add_geometry(line, reset_bounding_box=False)

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0])
        self.vis.add_geometry(axis, reset_bounding_box=False)

        self._build_robot()

        self.balloon_sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=0.15)
        self.balloon_sphere.paint_uniform_color([0.1, 0.9, 0.1])
        self.balloon_sphere.translate(self.balloon_pos)
        self.vis.add_geometry(self.balloon_sphere, reset_bounding_box=False)

        self._robot_hitbox_vis = o3d.geometry.TriangleMesh.create_box(
            width=0.4, height=0.4, depth=0.4)
        self._robot_hitbox_vis.paint_uniform_color([1.0, 1.0, 0.0])
        self._robot_hitbox_vis.compute_vertex_normals()
        center = self.robot_pos + np.array([0.0, 0.8, 0.0])
        self._robot_hitbox_vis.translate(
            center - np.array([0.2, 0.2, 0.2]))
        self.vis.add_geometry(self._robot_hitbox_vis, reset_bounding_box=False)

        ctr = self.vis.get_view_control()
        ctr.set_front([0.0, -0.5, -1.0])
        ctr.set_up([0.0, 1.0, 0.0])
        ctr.set_zoom(0.8)

    def _build_robot(self):
        base = o3d.geometry.TriangleMesh.create_box(
            width=0.3, height=0.05, depth=0.3)
        base.paint_uniform_color([0.2, 0.2, 0.8])
        base.translate(np.array([-0.15, 0.0, -0.15]))

        body = o3d.geometry.TriangleMesh.create_box(
            width=0.2, height=0.5, depth=0.2)
        body.paint_uniform_color([0.3, 0.3, 0.9])
        body.translate(np.array([-0.1, 0.05, -0.1]))

        head = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
        head.paint_uniform_color([0.5, 0.5, 1.0])
        head.translate(np.array([0.0, 0.65, 0.0]))

        arm_l = o3d.geometry.TriangleMesh.create_cylinder(
            radius=0.04, height=0.3)
        arm_l.paint_uniform_color([0.4, 0.4, 0.8])
        arm_l.translate(np.array([-0.15, 0.4, 0.0]))

        arm_r = o3d.geometry.TriangleMesh.create_cylinder(
            radius=0.04, height=0.3)
        arm_r.paint_uniform_color([0.4, 0.4, 0.8])
        arm_r.translate(np.array([0.15, 0.4, 0.0]))

        hand_l = o3d.geometry.TriangleMesh.create_box(
            width=0.12, height=0.12, depth=0.12)
        hand_l.paint_uniform_color([0.2, 0.8, 0.2])
        hand_l.translate(np.array([-0.21, 0.55, -0.06]))

        hand_r = o3d.geometry.TriangleMesh.create_box(
            width=0.12, height=0.12, depth=0.12)
        hand_r.paint_uniform_color([0.2, 0.8, 0.2])
        hand_r.translate(np.array([0.09, 0.55, -0.06]))

        self.robot_parts = [base, body, head, arm_l, arm_r, hand_l, hand_r]
        self._robot_translation = np.zeros(3, dtype=np.float64)

        for part in self.robot_parts:
            self.vis.add_geometry(part, reset_bounding_box=False)

    def update_balloon(self, pos_3d: np.ndarray):
        if pos_3d is not None:
            self.balloon_pos = pos_3d.astype(np.float64)
            self.balloon_visible = True
        else:
            self.balloon_visible = False

        self.command_queue.put({
            "type": "balloon",
            "pos": self.balloon_pos.copy() if self.balloon_visible else None
        })

    def update_robot(self, dx: float, dy: float, dz: float):
        self.robot_pos += np.array([dx, dy, dz], dtype=np.float64)
        self.robot_pos = np.clip(
            self.robot_pos,
            -self.floor_size / 2,
            self.floor_size / 2)

        self.command_queue.put({
            "type": "robot",
            "pos": self.robot_pos.copy()
        })

    def get_hitbox_center(self) -> np.ndarray:
        return self.robot_pos + np.array([0.0, 0.6, 0.0])

    def get_balloon_pos(self) -> np.ndarray:
        return self.balloon_pos.copy()

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    def _render_loop(self):
        last_time = time.time()

        while self.running:
            try:
                cmd = self.command_queue.get_nowait()
                if cmd["type"] == "balloon":
                    if cmd["pos"] is not None:
                        self.balloon_sphere.translate(
                            cmd["pos"] - self.balloon_sphere.get_center(),
                            relative=False)
                        self._robot_hitbox_vis.translate(
                            cmd["pos"] - self._robot_hitbox_vis.get_center()
                            + np.array([0.4, 0.2, 0.4]),
                            relative=False)
                        if not self.balloon_visible:
                            self.balloon_sphere.paint_uniform_color(
                                [0.1, 0.9, 0.1])
                elif cmd["type"] == "robot":
                    translation = cmd["pos"] - self._robot_translation
                    for part in self.robot_parts:
                        part.translate(translation, relative=True)
                    self._robot_hitbox_vis.translate(
                        translation, relative=True)
                    self._robot_translation = cmd["pos"].copy()
            except queue.Empty:
                pass

            now = time.time()
            elapsed = now - last_time
            if elapsed < self.frame_delay:
                time.sleep(self.frame_delay - elapsed)
            last_time = time.time()

            self.vis.poll_events()
            self.vis.update_renderer()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.vis.destroy_window()
