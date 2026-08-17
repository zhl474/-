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


_SAMPLE_AXIS_TOL_MM = 1.0
_LOOCV_SIMPLEST_SLACK = 0.05


def _build_sucker_offset_samples(samples):
    """解析并校验 sucker_offset_samples，返回自动拟合的偏移模型；至少需要 3 个样本。

    样本格式: [{x, y, vx, vy, anchor}]，(x, y) 为测量时步骤A（相机对准）的位姿 XY，
    (vx, vy) 为实测吸盘偏移（步骤B 位姿减步骤A 位姿的 XY）。
    anchor: true 的样本作为基线锚点（建议工作区中心），模型强制穿过该点。
    点数灵活：3 点即可，7/9 点直接加行，维度由代码按样本覆盖范围自动判定。
    """
    if not isinstance(samples, list):
        raise ValueError("sucker_offset_samples 必须是列表")
    if len(samples) < 3:
        raise ValueError("sucker_offset_samples 至少需要 3 个样本")
    parsed = []
    for index, raw in enumerate(samples):
        if not isinstance(raw, dict):
            raise ValueError(f"sucker_offset_samples[{index}] 必须是字典")
        for key in ("x", "y", "vx", "vy"):
            if key not in raw:
                raise ValueError(f"sucker_offset_samples[{index}] 缺少 {key}")
        parsed.append({
            "x": float(raw["x"]),
            "y": float(raw["y"]),
            "vx": float(raw["vx"]),
            "vy": float(raw["vy"]),
            "anchor": bool(raw.get("anchor", False)),
        })
    if sum(1 for sample in parsed if sample["anchor"]) > 1:
        raise ValueError("sucker_offset_samples 最多只能有 1 个锚点（anchor: true）")
    return _fit_sucker_offset_model(parsed)


def _clamp_for_kind(kind, xs, ys):
    """按模型维度生成 clamp 范围（禁止外推）。"""
    if kind == "1d_x":
        return [float(np.min(xs)), float(np.max(xs))]
    if kind == "1d_y":
        return [float(np.min(ys)), float(np.max(ys))]
    # 2d / quad2 均用四维 clamp
    return [
        float(np.min(xs)), float(np.max(xs)),
        float(np.min(ys)), float(np.max(ys)),
    ]


def _eval_offset_model(model, x_mm, y_mm):
    """本地计算模型预测，与 visual_servo._predict_sucker_offset 等价（避免循环导入）。"""
    clamp = model["clamp"]
    kind = model["kind"]
    if kind == "1d_x":
        x = min(max(float(x_mm), clamp[0]), clamp[1])
        row = np.array([1.0, x], dtype=float)
    elif kind == "1d_y":
        y = min(max(float(y_mm), clamp[0]), clamp[1])
        row = np.array([1.0, y], dtype=float)
    elif kind == "quad2":
        x = min(max(float(x_mm), clamp[0]), clamp[1])
        y = min(max(float(y_mm), clamp[2]), clamp[3])
        row = np.array([1.0, x, y, x * x, x * y, y * y], dtype=float)
    else:  # 2d
        x = min(max(float(x_mm), clamp[0]), clamp[1])
        y = min(max(float(y_mm), clamp[2]), clamp[3])
        row = np.array([1.0, x, y], dtype=float)
    return (
        float(np.dot(row, np.asarray(model["coef_vx"], dtype=float))),
        float(np.dot(row, np.asarray(model["coef_vy"], dtype=float))),
    )


def _report_sucker_offset_fit(model, samples):
    """打印每个样本的实测与预测偏移及残差，用于判断拟合质量和线性假设是否成立。"""
    print("sucker_offset_samples 拟合残差 (mm)：")
    for sample in samples:
        px, py = _eval_offset_model(model, sample["x"], sample["y"])
        anchor_mark = "  <- 锚点" if sample["anchor"] else ""
        print(
            f"  X={sample['x']:8.1f} Y={sample['y']:8.1f}"
            f" 实测=({sample['vx']:+8.3f},{sample['vy']:+8.3f})"
            f" 预测=({px:+8.3f},{py:+8.3f})"
            f" 残差=({sample['vx'] - px:+7.3f},{sample['vy'] - py:+7.3f}){anchor_mark}"
        )


def _fit_plane_through_anchor(samples, anchor):
    """以锚点实测偏移为基线，其余样本最小二乘定平面斜率（2 个自由参数）。

    模型 v(X,Y) = v_anchor + kx·(X-Xc) + ky·(Y-Yc)，锚点处严格等于实测值。
    返回 (kind, coef_vx, coef_vy)，coef 格式与 _eval_offset_model 兼容。
    """
    xc, yc = anchor["x"], anchor["y"]
    vxc, vyc = anchor["vx"], anchor["vy"]
    others = [sample for sample in samples if not sample["anchor"]]
    xs = np.asarray([sample["x"] for sample in others], dtype=float)
    ys = np.asarray([sample["y"] for sample in others], dtype=float)
    dvx = np.asarray([sample["vx"] - vxc for sample in others], dtype=float)
    dvy = np.asarray([sample["vy"] - vyc for sample in others], dtype=float)
    if float(np.ptp(ys)) < _SAMPLE_AXIS_TOL_MM:
        kind = "1d_x"
        design = np.column_stack([xs - xc])
    elif float(np.ptp(xs)) < _SAMPLE_AXIS_TOL_MM:
        kind = "1d_y"
        design = np.column_stack([ys - yc])
    else:
        kind = "2d"
        design = np.column_stack([xs - xc, ys - yc])
    coef_vx, _, _, _ = np.linalg.lstsq(design, dvx, rcond=None)
    coef_vy, _, _, _ = np.linalg.lstsq(design, dvy, rcond=None)
    if kind == "1d_x":
        return kind, [vxc - coef_vx[0] * xc, coef_vx[0]], [vyc - coef_vy[0] * xc, coef_vy[0]]
    if kind == "1d_y":
        return kind, [vxc - coef_vx[0] * yc, coef_vx[0]], [vyc - coef_vy[0] * yc, coef_vy[0]]
    return (
        kind,
        [vxc - coef_vx[0] * xc - coef_vx[1] * yc, coef_vx[0], coef_vx[1]],
        [vyc - coef_vy[0] * xc - coef_vy[1] * yc, coef_vy[0], coef_vy[1]],
    )


def _fit_quad2_through_anchor(samples, anchor):
    """以锚点实测偏移为基线，其余样本最小二乘定二次曲面斜率（5 个自由参数）。

    模型 v(X,Y) = v_anchor + slopes @ [dx,dy,dx²,dx·dy,dy²]，
    dx=X-Xc, dy=Y-Yc；正确展开为截距式6系数格式。
    """
    xc, yc = anchor["x"], anchor["y"]
    vxc, vyc = anchor["vx"], anchor["vy"]
    others = [s for s in samples if not s["anchor"]]
    dx = np.asarray([s["x"] - xc for s in others], dtype=float)
    dy = np.asarray([s["y"] - yc for s in others], dtype=float)
    dvx = np.asarray([s["vx"] - vxc for s in others], dtype=float)
    dvy = np.asarray([s["vy"] - vyc for s in others], dtype=float)
    design = np.column_stack([dx, dy, dx * dx, dx * dy, dy * dy])
    slopes_vx, _, _, _ = np.linalg.lstsq(design, dvx, rcond=None)
    slopes_vy, _, _, _ = np.linalg.lstsq(design, dvy, rcond=None)
    # 正确展开为截距式：coef @ [1, x, y, x², xy, y²]
    # v = v_anchor + a·dx + b·dy + c·dx² + d·dx·dy + e·dy²
    # 展开 (x-xc)² 等项后：
    #   c0 = v_anchor - a·xc - b·yc + c·xc² + d·xc·yc + e·yc²
    #   c1 = a - 2c·xc - d·yc
    #   c2 = b - d·xc - 2e·yc
    #   c3 = c,  c4 = d,  c5 = e
    a, b, c, d, e = slopes_vx
    c0_vx = vxc - a * xc - b * yc + c * xc * xc + d * xc * yc + e * yc * yc
    c1_vx = a - 2 * c * xc - d * yc
    c2_vx = b - d * xc - 2 * e * yc
    coef_vx = [c0_vx, c1_vx, c2_vx, float(c), float(d), float(e)]

    a, b, c, d, e = slopes_vy
    c0_vy = vyc - a * xc - b * yc + c * xc * xc + d * xc * yc + e * yc * yc
    c1_vy = a - 2 * c * xc - d * yc
    c2_vy = b - d * xc - 2 * e * yc
    coef_vy = [c0_vy, c1_vy, c2_vy, float(c), float(d), float(e)]
    return coef_vx, coef_vy


def _compute_loocv_for_kind(kind, samples, anchor):
    """对指定 kind 做留一交叉验证，返回 2D 距离 RMSE（mm）。"""
    others = [s for s in samples if not s["anchor"]]
    n_others = len(others)
    n_params = {"2d": 2, "quad2": 5}.get(kind)
    if n_params is None:
        return float("inf")
    if n_others - 1 < n_params:
        return float("inf")
    xc, yc = anchor["x"], anchor["y"]
    vxc, vyc = anchor["vx"], anchor["vy"]
    errors = []
    for i in range(n_others):
        training = [others[j] for j in range(n_others) if j != i]
        test = others[i]
        dx = np.asarray([s["x"] - xc for s in training], dtype=float)
        dy = np.asarray([s["y"] - yc for s in training], dtype=float)
        dvx = np.asarray([s["vx"] - vxc for s in training], dtype=float)
        dvy = np.asarray([s["vy"] - vyc for s in training], dtype=float)
        if kind == "2d":
            design = np.column_stack([dx, dy])
        else:
            design = np.column_stack([dx, dy, dx * dx, dx * dy, dy * dy])
        coef_vx = np.linalg.lstsq(design, dvx, rcond=None)[0]
        coef_vy = np.linalg.lstsq(design, dvy, rcond=None)[0]
        tdx = test["x"] - xc
        tdy = test["y"] - yc
        if kind == "2d":
            feat = np.array([tdx, tdy])
        else:
            feat = np.array([tdx, tdy, tdx * tdx, tdx * tdy, tdy * tdy])
        err_vx = test["vx"] - (vxc + coef_vx @ feat)
        err_vy = test["vy"] - (vyc + coef_vy @ feat)
        errors.append(err_vx * err_vx + err_vy * err_vy)
    return float(np.sqrt(np.mean(errors)))


def _select_best_anchored_kind(samples, anchor):
    """在候选 kind 中选 LOOCV 误差在最优 5% 以内的最简模型。"""
    candidates = ["2d"]
    n_others = sum(1 for s in samples if not s["anchor"])
    if n_others >= 6:
        candidates.append("quad2")
    loo_rmses = {k: _compute_loocv_for_kind(k, samples, anchor) for k in candidates}
    best_kind = min(loo_rmses, key=loo_rmses.get)
    best_rmse = loo_rmses[best_kind]
    limit = best_rmse * (1.0 + _LOOCV_SIMPLEST_SLACK) if np.isfinite(best_rmse) else 0.0
    for kind in ("2d", "quad2"):  # 从简到繁
        if kind in loo_rmses and loo_rmses[kind] <= limit:
            return kind, loo_rmses
    return best_kind, loo_rmses


def _fit_sucker_offset_model(samples):
    """拟合吸盘偏移随位姿 XY 的模型，并给出 clamp 范围（禁止外推）。

    候选模型：constant / 1d_x / 1d_y / 2d（平面） / quad2（二次曲面）。
    有锚点时通过 LOOCV 在 2d 与 quad2 之间自动选型；
    无锚点时退化为按覆盖范围选1d/2d。≥6 个非锚点时才尝试 quad2。
    """
    xs = np.asarray([sample["x"] for sample in samples], dtype=float)
    ys = np.asarray([sample["y"] for sample in samples], dtype=float)
    x_range = float(np.ptp(xs))
    y_range = float(np.ptp(ys))
    if x_range < _SAMPLE_AXIS_TOL_MM and y_range < _SAMPLE_AXIS_TOL_MM:
        raise ValueError("sucker_offset_samples 在 X 和 Y 方向都没有覆盖范围")
    anchors = [sample for sample in samples if sample["anchor"]]
    if anchors:
        loocv_kind, loo = _select_best_anchored_kind(samples, anchors[0])
        if loocv_kind == "quad2":
            coef_vx, coef_vy = _fit_quad2_through_anchor(samples, anchors[0])
            kind = "quad2"
        else:
            kind, coef_vx, coef_vy = _fit_plane_through_anchor(samples, anchors[0])
    else:
        vxs = np.asarray([sample["vx"] for sample in samples], dtype=float)
        vys = np.asarray([sample["vy"] for sample in samples], dtype=float)
        if y_range < _SAMPLE_AXIS_TOL_MM:
            kind = "1d_x"
            design = np.column_stack([np.ones_like(xs), xs])
        elif x_range < _SAMPLE_AXIS_TOL_MM:
            kind = "1d_y"
            design = np.column_stack([np.ones_like(ys), ys])
        else:
            kind = "2d"
            design = np.column_stack([np.ones_like(xs), xs, ys])
        coef_vx = np.linalg.lstsq(design, vxs, rcond=None)[0].tolist()
        coef_vy = np.linalg.lstsq(design, vys, rcond=None)[0].tolist()
    if not np.all(np.isfinite(coef_vx)) or not np.all(np.isfinite(coef_vy)):
        raise ValueError("sucker_offset_samples 拟合失败，系数非有限")
    model = {
        "kind": kind,
        "coef_vx": coef_vx,
        "coef_vy": coef_vy,
        "clamp": _clamp_for_kind(kind, xs, ys),
    }
    _report_sucker_offset_fit(model, samples)
    return model


def load_visual_servo_config(config_path: str = DEFAULT_VISUAL_SERVO_CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    matrix = np.asarray(config.get("pixel_to_robot_matrix"), dtype=float)
    offset = np.asarray(config.get("camera_to_sucker_offset_mm"), dtype=float)
    if matrix.shape != (2, 2) or offset.shape != (2,):
        raise ValueError("视觉伺服矩阵必须为 2x2，吸盘偏移必须包含 2 个数值")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(offset)):
        raise ValueError("视觉伺服配置包含非有限数值")
    samples = config.get("sucker_offset_samples")
    if samples is not None:
        config["sucker_offset_model"] = _build_sucker_offset_samples(samples)
    return config
