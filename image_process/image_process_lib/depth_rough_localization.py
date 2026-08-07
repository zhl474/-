"""标定模式下用深度世界坐标生成低位视觉伺服粗位姿。"""

from dataclasses import dataclass
from typing import Sequence

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


@dataclass(frozen=True)
class ZPlaneFit:
    """机器人 XY 到 TCP Z 的平面拟合结果。"""

    coefficients: np.ndarray
    residuals: np.ndarray
    rmse_mm: float

    def predict(self, x_mm, y_mm):
        a, b, c = self.coefficients
        return float(a * float(x_mm) + b * float(y_mm) + c)


def fit_z_plane(tcp_xyz: Sequence[Sequence[float]]) -> ZPlaneFit:
    """拟合 z=a*x+b*y+c，并拒绝少点、共线和非有限输入。"""
    points = np.asarray(tcp_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("TCP Z 平面至少需要 3 个三维点")
    if not np.all(np.isfinite(points)):
        raise ValueError("TCP Z 平面数据包含非有限数值")
    design = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
    coefficients, _residual_sum, rank, _singular = np.linalg.lstsq(
        design,
        points[:, 2],
        rcond=None,
    )
    if rank < 3:
        raise ValueError("TCP Z 平面的 XY 点共线，无法拟合二维平面")
    residuals = design @ coefficients - points[:, 2]
    return ZPlaneFit(
        coefficients=np.asarray(coefficients, dtype=float),
        residuals=np.asarray(residuals, dtype=float),
        rmse_mm=float(np.sqrt(np.mean(residuals**2))),
    )


class DepthRoughLocalizer:
    """把目标表面世界坐标转换为固定姿态的 TCP 粗定位位姿。"""

    def __init__(self, shooting_pose, wrist_to_camera_mm):
        shooting = np.asarray(shooting_pose, dtype=float)
        wrist_to_camera = np.asarray(wrist_to_camera_mm, dtype=float)
        if shooting.shape != (6,) or not np.all(np.isfinite(shooting)):
            raise ValueError("高位拍摄位姿必须包含 6 个有限数值")
        if wrist_to_camera.shape != (4, 4) or not np.all(np.isfinite(wrist_to_camera)):
            raise ValueError("手眼标定矩阵必须是有限的 4x4 矩阵")
        self.shooting_pose = shooting
        rotation = rpy_degrees_to_rotation_matrix(*shooting[3:6])
        self.camera_offset_in_base = rotation @ wrist_to_camera[:3, 3]

    def tcp_xy_from_world(self, world_position):
        """沿用历史算法，把表面世界 XY 转成相机对准时的 TCP XY。"""
        world = np.asarray(world_position, dtype=float)
        if world.shape != (3,) or not np.all(np.isfinite(world)):
            raise ValueError("目标表面世界坐标必须包含 3 个有限数值")
        return np.array(
            [
                world[0] - self.camera_offset_in_base[0],
                world[1] - self.camera_offset_in_base[1],
            ],
            dtype=float,
        )

    def block_observation_pose(self, world_position, observation_height_mm):
        """用方块上表面 XYZ 生成低位观察 TCP 位姿。"""
        world = np.asarray(world_position, dtype=float)
        tcp_xy = self.tcp_xy_from_world(world)
        height = float(observation_height_mm)
        if not np.isfinite(height) or height <= 0.0:
            raise ValueError("方块观察高度必须是大于 0 的有限数值")
        return [
            float(tcp_xy[0]),
            float(tcp_xy[1]),
            float(world[2] + height),
            *self.shooting_pose[3:6].tolist(),
        ]

    def tray_observation_pose(self, world_position, tcp_z_mm):
        """只使用托盘深度世界 XY，TCP Z 由方块观察平面提供。"""
        tcp_xy = self.tcp_xy_from_world(world_position)
        tcp_z = float(tcp_z_mm)
        if not np.isfinite(tcp_z):
            raise ValueError("托盘 TCP Z 必须是有限数值")
        return [
            float(tcp_xy[0]),
            float(tcp_xy[1]),
            tcp_z,
            *self.shooting_pose[3:6].tolist(),
        ]
