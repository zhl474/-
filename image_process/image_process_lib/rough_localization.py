"""深度优先的高位粗定位，失败时可回退像素比例估计。"""

from typing import Callable, Sequence

import numpy as np


def rpy_degrees_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    """按 Rz(yaw) * Ry(pitch) * Rx(roll) 生成旋转矩阵。"""
    roll, pitch, yaw = np.deg2rad([float(roll_deg), float(pitch_deg), float(yaw_deg)])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    rotation_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rotation_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rotation_z @ rotation_y @ rotation_x


class RoughLocalizer:
    def __init__(
        self,
        shooting_pose: Sequence[float],
        servo_look_z: float,
        wrist_to_camera_mm,
        pixel_to_world_client: Callable,
        x_mm_per_pixel: float,
        y_mm_per_pixel: float,
        fallback_enabled: bool,
        warning_func: Callable[[str], None],
    ):
        self.shooting_pose = np.asarray(shooting_pose, dtype=float)
        self.wrist_to_camera_mm = np.asarray(wrist_to_camera_mm, dtype=float)
        if self.shooting_pose.shape != (6,) or not np.all(np.isfinite(self.shooting_pose)):
            raise ValueError("高位拍摄位姿必须包含 6 个有限数值")
        if self.wrist_to_camera_mm.shape != (4, 4) or not np.all(np.isfinite(self.wrist_to_camera_mm)):
            raise ValueError("手眼标定矩阵必须是有限的 4x4 矩阵")
        self.servo_look_z = float(servo_look_z)
        self.pixel_to_world_client = pixel_to_world_client
        self.x_mm_per_pixel = float(x_mm_per_pixel)
        self.y_mm_per_pixel = float(y_mm_per_pixel)
        self.fallback_enabled = bool(fallback_enabled)
        self.warning_func = warning_func

    def _query_world_position(self, px, py):
        pixel_x = int(round(float(px)))
        pixel_y = int(round(float(py)))
        response = self.pixel_to_world_client(pixel_x, pixel_y)
        if not response.success:
            raise ValueError(response.message)
        world_position = np.asarray(response.world_position, dtype=float)
        if world_position.shape != (3,) or not np.all(np.isfinite(world_position)):
            raise ValueError(f"深度服务返回无效世界坐标: {world_position.tolist()}")
        return world_position

    def _make_depth_pose(self, world_position):
        tool_rotation = rpy_degrees_to_rotation_matrix(*self.shooting_pose[3:6])
        camera_offset_in_base = tool_rotation @ self.wrist_to_camera_mm[:3, 3]
        tool_position = np.array([
            world_position[0] - camera_offset_in_base[0],
            world_position[1] - camera_offset_in_base[1],
            self.servo_look_z,
        ])
        return [*tool_position.tolist(), *self.shooting_pose[3:6].tolist()]

    def _make_pixel_fallback_pose(self, px, py, image_shape):
        height, width = image_shape[:2]
        predicted_x = self.shooting_pose[0] + (float(py) - height / 2.0) * self.y_mm_per_pixel
        predicted_y = self.shooting_pose[1] + (float(px) - width / 2.0) * self.x_mm_per_pixel
        return [
            float(predicted_x),
            float(predicted_y),
            self.servo_look_z,
            *self.shooting_pose[3:6].tolist(),
        ]

    def locate(self, px, py, image_shape, label):
        try:
            world_position = self._query_world_position(px, py)
            pose = self._make_depth_pose(world_position)
            return pose, "depth", world_position.tolist()
        except Exception as exc:
            if not self.fallback_enabled:
                raise
            self.warning_func(f"{label}深度定位失败，回退旧粗估: {exc}")
            return self._make_pixel_fallback_pose(px, py, image_shape), "fallback", []
