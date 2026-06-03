import open3d as o3d
import numpy as np


def create_robot_model() -> o3d.geometry.TriangleMesh:
    parts = []

    base = o3d.geometry.TriangleMesh.create_box(
        width=0.3, height=0.05, depth=0.3)
    base.paint_uniform_color([0.2, 0.2, 0.8])
    base.translate(np.array([-0.15, 0.0, -0.15]))
    parts.append(base)

    body = o3d.geometry.TriangleMesh.create_box(
        width=0.2, height=0.5, depth=0.2)
    body.paint_uniform_color([0.3, 0.3, 0.9])
    body.translate(np.array([-0.1, 0.05, -0.1]))
    parts.append(body)

    head = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
    head.paint_uniform_color([0.5, 0.5, 1.0])
    head.translate(np.array([0.0, 0.65, 0.0]))
    parts.append(head)

    arm_l = o3d.geometry.TriangleMesh.create_cylinder(
        radius=0.04, height=0.3)
    arm_l.paint_uniform_color([0.4, 0.4, 0.8])
    arm_l.translate(np.array([-0.15, 0.4, 0.0]))
    parts.append(arm_l)

    arm_r = o3d.geometry.TriangleMesh.create_cylinder(
        radius=0.04, height=0.3)
    arm_r.paint_uniform_color([0.4, 0.4, 0.8])
    arm_r.translate(np.array([0.15, 0.4, 0.0]))
    parts.append(arm_r)

    hand_l = o3d.geometry.TriangleMesh.create_box(
        width=0.12, height=0.12, depth=0.12)
    hand_l.paint_uniform_color([0.2, 0.8, 0.2])
    hand_l.translate(np.array([-0.21, 0.55, -0.06]))
    parts.append(hand_l)

    hand_r = o3d.geometry.TriangleMesh.create_box(
        width=0.12, height=0.12, depth=0.12)
    hand_r.paint_uniform_color([0.2, 0.8, 0.2])
    hand_r.translate(np.array([0.09, 0.55, -0.06]))
    parts.append(hand_r)

    combined = parts[0]
    for part in parts[1:]:
        combined += part

    return combined
