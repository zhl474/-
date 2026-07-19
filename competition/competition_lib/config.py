"""比赛执行配置读取与校验。"""

from dataclasses import dataclass
import os
from typing import Sequence

import numpy as np
import yaml


PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_EXECUTION_CONFIG_PATH = os.path.join(PACKAGE_DIR, "config", "execution.yaml")
DEFAULT_VISUAL_SERVO_CONFIG_PATH = os.path.join(PACKAGE_DIR, "config", "visual_servo.yaml")


@dataclass(frozen=True)
class ExecutionConfig:
    shooting_pose: tuple
    arm_speed: int
    pick_speed: int
    servo_speed: int
    pick_surface_offset_mm: float
    lift_z: float
    minimum_tcp_z_mm: float
    timing_debug: bool
    error_threshold_px: float
    min_step_mm: float
    max_step_mm: float
    max_iter: int
    success_stable_frames: int
    max_missed_frames: int
    settle_sec: float
    initial_motor_angle_deg: float
    motor_velocity_deg_per_sec: float
    motor_lower_margin_deg: float
    motor_upper_margin_deg: float


def _finite_pose(values: Sequence[float], name: str) -> tuple:
    pose = np.asarray(values, dtype=float)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} 必须包含 6 个有限数值")
    return tuple(float(value) for value in pose)


def load_execution_config(config_path: str = DEFAULT_EXECUTION_CONFIG_PATH) -> ExecutionConfig:
    with open(config_path, "r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    motion = data.get("motion", {})
    servo = data.get("servo", {})
    motor = data.get("tool_motor", {})
    config = ExecutionConfig(
        shooting_pose=_finite_pose(data.get("shooting_pose"), "shooting_pose"),
        arm_speed=int(motion["arm_speed"]),
        pick_speed=int(motion["pick_speed"]),
        servo_speed=int(motion["servo_speed"]),
        pick_surface_offset_mm=float(motion["pick_surface_offset_mm"]),
        lift_z=float(motion["lift_z"]),
        minimum_tcp_z_mm=float(motion["minimum_tcp_z_mm"]),
        timing_debug=bool(servo.get("timing_debug", False)),
        error_threshold_px=float(servo["error_threshold_px"]),
        min_step_mm=float(servo["min_step_mm"]),
        max_step_mm=float(servo["max_step_mm"]),
        max_iter=int(servo["max_iter"]),
        success_stable_frames=int(servo["success_stable_frames"]),
        max_missed_frames=int(servo["max_missed_frames"]),
        settle_sec=float(servo["settle_sec"]),
        initial_motor_angle_deg=float(motor["initial_angle_deg"]),
        motor_velocity_deg_per_sec=float(motor["velocity_deg_per_sec"]),
        motor_lower_margin_deg=float(motor["lower_margin_deg"]),
        motor_upper_margin_deg=float(motor["upper_margin_deg"]),
    )
    numeric_values = [
        config.arm_speed, config.pick_speed, config.servo_speed, config.pick_surface_offset_mm, config.lift_z,
        config.minimum_tcp_z_mm,
        config.error_threshold_px, config.min_step_mm, config.max_step_mm, config.max_iter,
        config.success_stable_frames, config.max_missed_frames, config.motor_velocity_deg_per_sec,
    ]
    if not np.isfinite(config.minimum_tcp_z_mm) or config.minimum_tcp_z_mm <= 0:
        raise ValueError("minimum_tcp_z_mm 必须是大于 0 的有限数值")
    if not np.all(np.isfinite(numeric_values)) or min(config.arm_speed, config.pick_speed, config.servo_speed) <= 0:
        raise ValueError("执行配置包含无效数值")
    if config.error_threshold_px < 0 or not 0 <= config.min_step_mm <= config.max_step_mm:
        raise ValueError("视觉伺服误差阈值和步长范围无效")
    if min(config.max_iter, config.success_stable_frames, config.max_missed_frames) <= 0:
        raise ValueError("视觉伺服迭代、稳定帧和丢失帧限制必须大于 0")
    if config.motor_velocity_deg_per_sec <= 0:
        raise ValueError("舵机速度必须大于 0")
    if not 0 <= config.motor_lower_margin_deg < config.motor_upper_margin_deg <= 360:
        raise ValueError("舵机安全角度边界无效")
    if config.shooting_pose[2] < config.minimum_tcp_z_mm:
        raise ValueError("shooting_pose 的 TCP Z 低于安全下限")
    if config.lift_z < config.minimum_tcp_z_mm:
        raise ValueError("lift_z 低于 TCP Z 安全下限")
    return config


def load_visual_servo_config(config_path: str = DEFAULT_VISUAL_SERVO_CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    matrix = np.asarray(config.get("pixel_to_robot_matrix"), dtype=float)
    offset = np.asarray(config.get("camera_to_sucker_offset_mm"), dtype=float)
    if matrix.shape != (2, 2) or offset.shape != (2,):
        raise ValueError("视觉伺服矩阵必须为 2x2，吸盘偏移必须包含 2 个数值")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(offset)):
        raise ValueError("视觉伺服配置包含非有限数值")
    return config
