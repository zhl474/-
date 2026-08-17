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
    calibration_mode: bool
    visual_servo_enabled: bool
    shooting_pose: tuple
    arm_speed: int
    pick_speed: int
    servo_speed: int
    pick_surface_offset_mm: float
    pick_approach_clearance_mm: float
    pick_approach_speed: int
    pick_retreat_blend_radius_mm: float
    place_descent_offset_mm: float
    place_descent_blend_radius_mm: float
    place_lift_blend_radius_mm: float
    minimum_tcp_z_mm: float
    pick_rotate_safe_lift_mm: float
    pick_safe_z_timeout_sec: float
    final_blow_hold_sec: float
    timing_debug: bool
    block_error_threshold_px: float
    tray_error_threshold_px: float
    min_step_mm: float
    max_step_mm: float
    max_iter: int
    success_stable_frames: int
    post_success_sample_frames: int
    max_missed_frames: int
    settle_sec: float
    sample_complete_ratio: float
    initial_motor_angle_deg: float
    motor_velocity_deg_per_sec: float
    motor_lower_margin_deg: float
    motor_upper_margin_deg: float


def _finite_pose(values: Sequence[float], name: str) -> tuple:
    pose = np.asarray(values, dtype=float)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} 必须包含 6 个有限数值")
    return tuple(float(value) for value in pose)


def _strict_bool(value, name: str) -> bool:
    """严格读取 YAML 布尔值，避免字符串 "false" 被判定为真。"""
    if not isinstance(value, bool):
        raise ValueError(f"{name} 必须是 YAML 布尔值 true 或 false")
    return value


def _nonnegative_int(value, name: str) -> int:
    """严格读取非负整数，禁止把小数帧数静默截断。"""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} 必须是大于等于 0 的整数")
    return int(value)


def _positive_int(value, name: str) -> int:
    """严格读取正整数，避免布尔值或小数被静默转换为运动速度。"""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是大于 0 的整数")
    return int(value)


def _nonnegative_finite_float(value, name: str) -> float:
    """严格读取有限非负浮点数，避免布尔值或非有限数进入运动配置。"""
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是大于等于 0 的有限数值")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是大于等于 0 的有限数值") from None
    if not np.isfinite(number) or number < 0:
        raise ValueError(f"{name} 必须是大于等于 0 的有限数值")
    return number


def _positive_finite_float(value, name: str) -> float:
    """严格读取有限正浮点数，供动态预抓取间隙等运动参数使用。"""
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是大于 0 的有限数值")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是大于 0 的有限数值") from None
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} 必须是大于 0 的有限数值")
    return number


def _servo_error_thresholds(servo: dict) -> tuple:
    """读取目标级阈值，并兼容仅包含旧版统一阈值的配置。"""
    block_key = "block_error_threshold_px"
    tray_key = "tray_error_threshold_px"
    has_block = block_key in servo
    has_tray = tray_key in servo
    if has_block != has_tray:
        raise ValueError(
            f"servo.{block_key} 和 servo.{tray_key} 必须同时配置"
        )
    if has_block:
        return (
            _nonnegative_finite_float(servo[block_key], f"servo.{block_key}"),
            _nonnegative_finite_float(servo[tray_key], f"servo.{tray_key}"),
        )
    if "error_threshold_px" not in servo:
        raise ValueError("servo 缺少方块和托盘视觉伺服误差阈值")
    legacy_threshold = _nonnegative_finite_float(
        servo["error_threshold_px"],
        "servo.error_threshold_px",
    )
    return legacy_threshold, legacy_threshold


def load_execution_config(config_path: str = DEFAULT_EXECUTION_CONFIG_PATH) -> ExecutionConfig:
    with open(config_path, "r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    motion = data.get("motion", {})
    servo = data.get("servo", {})
    motor = data.get("tool_motor", {})
    block_error_threshold_px, tray_error_threshold_px = _servo_error_thresholds(servo)
    config = ExecutionConfig(
        calibration_mode=_strict_bool(data.get("calibration_mode", False), "calibration_mode"),
        visual_servo_enabled=_strict_bool(
            servo.get("enabled", True),
            "servo.enabled",
        ),
        shooting_pose=_finite_pose(data.get("shooting_pose"), "shooting_pose"),
        arm_speed=int(motion["arm_speed"]),
        pick_speed=int(motion["pick_speed"]),
        servo_speed=int(motion["servo_speed"]),
        pick_surface_offset_mm=float(motion["pick_surface_offset_mm"]),
        pick_approach_clearance_mm=_positive_finite_float(
            motion["pick_approach_clearance_mm"],
            "motion.pick_approach_clearance_mm",
        ),
        pick_approach_speed=_positive_int(
            motion["pick_approach_speed"],
            "motion.pick_approach_speed",
        ),
        pick_retreat_blend_radius_mm=_nonnegative_finite_float(
            motion["pick_retreat_blend_radius_mm"],
            "motion.pick_retreat_blend_radius_mm",
        ),
        place_descent_offset_mm=_nonnegative_finite_float(
            motion["place_descent_offset_mm"],
            "motion.place_descent_offset_mm",
        ),
        place_descent_blend_radius_mm=_nonnegative_finite_float(
            motion["place_descent_blend_radius_mm"],
            "motion.place_descent_blend_radius_mm",
        ),
        place_lift_blend_radius_mm=_nonnegative_finite_float(
            motion["place_lift_blend_radius_mm"],
            "motion.place_lift_blend_radius_mm",
        ),
        minimum_tcp_z_mm=_positive_finite_float(
            motion["minimum_tcp_z_mm"],
            "motion.minimum_tcp_z_mm",
        ),
        pick_rotate_safe_lift_mm=_positive_finite_float(
            motion["pick_rotate_safe_lift_mm"],
            "motion.pick_rotate_safe_lift_mm",
        ),
        pick_safe_z_timeout_sec=_positive_finite_float(
            motion["pick_safe_z_timeout_sec"],
            "motion.pick_safe_z_timeout_sec",
        ),
        final_blow_hold_sec=_nonnegative_finite_float(
            motion.get("final_blow_hold_sec", 1.0),
            "motion.final_blow_hold_sec",
        ),
        timing_debug=bool(servo.get("timing_debug", False)),
        block_error_threshold_px=block_error_threshold_px,
        tray_error_threshold_px=tray_error_threshold_px,
        min_step_mm=float(servo["min_step_mm"]),
        max_step_mm=float(servo["max_step_mm"]),
        max_iter=int(servo["max_iter"]),
        success_stable_frames=int(servo["success_stable_frames"]),
        post_success_sample_frames=_nonnegative_int(
            servo.get("post_success_sample_frames", 0),
            "servo.post_success_sample_frames",
        ),
        max_missed_frames=int(servo["max_missed_frames"]),
        settle_sec=float(servo["settle_sec"]),
        sample_complete_ratio=_positive_finite_float(
            servo.get("sample_complete_ratio", 0.8),
            "servo.sample_complete_ratio",
        ),
        initial_motor_angle_deg=float(motor["initial_angle_deg"]),
        motor_velocity_deg_per_sec=float(motor["velocity_deg_per_sec"]),
        motor_lower_margin_deg=float(motor["lower_margin_deg"]),
        motor_upper_margin_deg=float(motor["upper_margin_deg"]),
    )
    numeric_values = [
        config.arm_speed, config.pick_speed, config.servo_speed,
        config.pick_surface_offset_mm, config.pick_approach_clearance_mm,
        config.pick_approach_speed, config.pick_retreat_blend_radius_mm,
        config.place_descent_offset_mm, config.place_descent_blend_radius_mm,
        config.place_lift_blend_radius_mm,
        config.minimum_tcp_z_mm,
        config.pick_rotate_safe_lift_mm,
        config.pick_safe_z_timeout_sec,
        config.final_blow_hold_sec,
        config.sample_complete_ratio,
        config.block_error_threshold_px, config.tray_error_threshold_px,
        config.min_step_mm, config.max_step_mm, config.max_iter,
        config.success_stable_frames, config.post_success_sample_frames,
        config.max_missed_frames, config.motor_velocity_deg_per_sec,
    ]
    if not np.isfinite(config.minimum_tcp_z_mm) or config.minimum_tcp_z_mm <= 0:
        raise ValueError("minimum_tcp_z_mm 必须是大于 0 的有限数值")
    if not np.all(np.isfinite(numeric_values)) or min(
        config.arm_speed,
        config.pick_speed,
        config.servo_speed,
        config.pick_approach_speed,
    ) <= 0:
        raise ValueError("执行配置包含无效数值")
    if not 0 <= config.min_step_mm <= config.max_step_mm:
        raise ValueError("视觉伺服步长范围无效")
    if not 0.0 < config.sample_complete_ratio <= 1.0:
        raise ValueError("servo.sample_complete_ratio 必须在 (0, 1] 之间")
    if config.pick_retreat_blend_radius_mm > 1000.0:
        raise ValueError("motion.pick_retreat_blend_radius_mm 必须小于等于 1000 mm")
    if config.place_descent_blend_radius_mm > 1000.0:
        raise ValueError("motion.place_descent_blend_radius_mm 必须小于等于 1000 mm")
    if config.place_lift_blend_radius_mm > 1000.0:
        raise ValueError("motion.place_lift_blend_radius_mm 必须小于等于 1000 mm")
    if (
        config.place_descent_offset_mm > 0.0
        and config.place_descent_blend_radius_mm >= config.place_descent_offset_mm
    ):
        raise ValueError(
            "motion.place_descent_blend_radius_mm 必须小于 "
            "motion.place_descent_offset_mm"
        )
    if (
        config.place_descent_offset_mm > 0.0
        and config.place_lift_blend_radius_mm >= config.place_descent_offset_mm
    ):
        raise ValueError(
            "motion.place_lift_blend_radius_mm 必须小于 "
            "motion.place_descent_offset_mm"
        )
    if min(config.max_iter, config.success_stable_frames, config.max_missed_frames) <= 0:
        raise ValueError("视觉伺服迭代、稳定帧和丢失帧限制必须大于 0")
    if config.motor_velocity_deg_per_sec <= 0:
        raise ValueError("舵机速度必须大于 0")
    if not 0 <= config.motor_lower_margin_deg < config.motor_upper_margin_deg <= 360:
        raise ValueError("舵机安全角度边界无效")
    if config.shooting_pose[2] < config.minimum_tcp_z_mm:
        raise ValueError("shooting_pose 的 TCP Z 低于安全下限")
    return config


_SUCKER_OFFSET_MODEL_TYPES = ("none", "linear_1d_x_residual", "linear_2d")


def _validate_sucker_offset_model(model: dict) -> dict:
    """校验 sucker_offset_model 并把参数归一化为 float。

    type: none                   仅使用固定偏移，无需其他参数。
    type: linear_1d_x_residual   Δvx = k_x·(X − x0)，Δvy = 0。
    type: linear_2d              Δvx = kx·(X − x0) + ky·(Y − y0)，
                                 Δvy = lx·(X − x0) + ly·(Y − y0)，lx/ly 可省略默认 0。
    所有类型的位置参数都必须带 clamp 采样范围，禁止外推。
    """
    kind = model.get("type", "none")
    if kind not in _SUCKER_OFFSET_MODEL_TYPES:
        raise ValueError(f"未知 sucker_offset_model.type: {kind!r}")
    normalized = {"type": kind}
    if kind == "none":
        return normalized
    keys = ["clamp_min_x", "clamp_max_x"]
    if kind == "linear_2d":
        keys += ["kx", "ky", "x0", "y0", "clamp_min_y", "clamp_max_y"]
        keys += [key for key in ("lx", "ly") if key in model]
    else:
        keys += ["k_x", "x0"]
    missing = [key for key in keys if key not in model]
    if missing:
        raise ValueError(f"sucker_offset_model 缺少参数: {', '.join(missing)}")
    try:
        for key in keys:
            normalized[key] = float(model[key])
    except (TypeError, ValueError):
        raise ValueError("sucker_offset_model 参数无效") from None
    if not np.all(np.isfinite([normalized[key] for key in keys])):
        raise ValueError("sucker_offset_model 参数无效")
    if normalized["clamp_min_x"] >= normalized["clamp_max_x"]:
        raise ValueError("sucker_offset_model X 采样范围无效")
    if (
        kind == "linear_2d"
        and normalized["clamp_min_y"] >= normalized["clamp_max_y"]
    ):
        raise ValueError("sucker_offset_model Y 采样范围无效")
    return normalized


def load_visual_servo_config(config_path: str = DEFAULT_VISUAL_SERVO_CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    matrix = np.asarray(config.get("pixel_to_robot_matrix"), dtype=float)
    offset = np.asarray(config.get("camera_to_sucker_offset_mm"), dtype=float)
    if matrix.shape != (2, 2) or offset.shape != (2,):
        raise ValueError("视觉伺服矩阵必须为 2x2，吸盘偏移必须包含 2 个数值")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(offset)):
        raise ValueError("视觉伺服配置包含非有限数值")
    model = config.get("sucker_offset_model")
    if model is not None:
        if not isinstance(model, dict):
            raise ValueError("sucker_offset_model 必须是字典")
        config["sucker_offset_model"] = _validate_sucker_offset_model(model)
    return config
