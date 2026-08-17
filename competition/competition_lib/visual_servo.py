"""方块和托盘共用的像素偏差视觉伺服闭环。"""

import time
from typing import Callable, Optional, Sequence

import numpy as np

from image_process_lib.sucker_offset import resolve_sucker_offset


def _finite_float(value):
    """把服务字段转成有限数值；缺失或无效时留空以便写入 CSV。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number if np.isfinite(number) else ""


def _pose_fields(pose):
    """把六维末端命令位姿转换为 CSV 字段。"""
    values = [_finite_float(value) for value in pose]
    if len(values) != 6 or "" in values:
        values = [""] * 6
    return dict(zip(
        ("末端命令X", "末端命令Y", "末端命令Z", "末端命令R", "末端命令P", "末端命令YAW"),
        values,
    ))


def _response_fields(response):
    """从低位检测响应提取像素与误差，并还原相机图像中心。"""
    target_x = _finite_float(getattr(response, "px", None))
    target_y = _finite_float(getattr(response, "py", None))
    error_x = _finite_float(getattr(response, "dx_px", None))
    error_y = _finite_float(getattr(response, "dy_px", None))
    center_x = target_x - error_x if target_x != "" and error_x != "" else ""
    center_y = target_y - error_y if target_y != "" and error_y != "" else ""
    if error_x == "" or error_y == "":
        maximum_error = ""
    else:
        maximum_error = max(abs(error_x), abs(error_y))
    return {
        "低位目标像素X": target_x,
        "低位目标像素Y": target_y,
        "低位图像中心X": center_x,
        "低位图像中心Y": center_y,
        "像素误差X": error_x,
        "像素误差Y": error_y,
        "最大像素误差": maximum_error,
        "低位检测角度deg": _finite_float(
            getattr(response, "detected_angle_deg", None)
        ),
        "低位匹配得分": _finite_float(getattr(response, "score", None)),
    }


def _emit_event(
    event_callback,
    event_name,
    iteration,
    response,
    pose,
    *,
    correction_xy=None,
    detection_ms="",
    control_ms="",
    arm_motion_ms="",
    settle_ms="",
    total_ms="",
    stable_frame_index="",
    static_sample_index="",
    message_override="",
):
    """向任务层发送单轮结构化事件，视觉闭环本身不直接输出 print。"""
    if event_callback is None:
        return
    correction = ["", ""] if correction_xy is None else [_finite_float(value) for value in correction_xy]
    row = {
        "事件": event_name,
        "伺服轮次": int(iteration) + 1,
        "连续稳定帧序号": stable_frame_index,
        "静止采样序号": static_sample_index,
        "识别成功": bool(getattr(response, "found", False)),
        "消息": str(message_override or getattr(response, "message", "") or ""),
        "XY修正X毫米": correction[0],
        "XY修正Y毫米": correction[1],
        "图像服务耗时毫秒": detection_ms,
        "控制计算耗时毫秒": control_ms,
        "机械臂到位耗时毫秒": arm_motion_ms,
        "稳定等待耗时毫秒": settle_ms,
        "本轮总耗时毫秒": total_ms,
    }
    row.update(_response_fields(response))
    row.update(_pose_fields(pose))
    event_callback(row)

def limit_xy_step(delta_xy, max_step_mm, min_step_mm=0.0):
    delta = np.asarray(delta_xy, dtype=float)
    norm = float(np.linalg.norm(delta))
    maximum = float(max_step_mm)
    minimum = min(max(0.0, float(min_step_mm)), maximum)
    if maximum <= 0.0 or norm <= 1e-9:
        return np.zeros(2, dtype=float)
    if norm > maximum:
        return delta / norm * maximum
    if norm < minimum:
        return delta / norm * minimum
    return delta


def pixel_error_to_robot_delta(dx_px, dy_px, matrix, max_step_mm, min_step_mm=0.0):
    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (2, 2) or not np.all(np.isfinite(transform)):
        raise ValueError("pixel_to_robot_matrix 必须是有限的 2x2 矩阵")
    raw_delta = transform @ np.array([float(dx_px), float(dy_px)], dtype=float)
    return limit_xy_step(raw_delta, max_step_mm, min_step_mm)


def apply_camera_to_sucker_offset(
    camera_pose: Sequence[float],
    config: dict,
    subject: str = "tray",
    pixel_xy: Optional[Sequence[float]] = None,
):
    """把相机对准位姿转换为吸盘对准位姿。

    偏移按 sucker_offset_strategy 选择（与感知端共用 image_process_lib.sucker_offset）：
      SINGLE_CALIBRATION（默认）：方块与托盘全部使用中间 camera_to_sucker_offset_mm；
      THREE_CALIBRATION：托盘恒用中间；方块（subject="block"）按高位检测像素
        pixel_xy 的 u 相对图像中线（640）分左右，分别用 left/right 两套偏移。
    subject 缺省为 "tray"，因此不传新参数的旧调用行为与旧版完全一致。

    位置相关偏移残差模型（sucker_offset_model）保留但当前不使用（type 必须为 none）：
      none                  仅使用固定偏移（未标定时兜底）
      linear_1d_x_residual  Δvx = k_x·(X − x0)，Δvy = 0
      linear_2d             Δvx = kx·(X − x0) + ky·(Y − y0)
                            Δvy = lx·(X − x0) + ly·(Y − y0)
    X/Y 为相机对准位姿的 XY，计算前先 clamp 到采样范围，禁止外推。
    """
    pose = list(float(value) for value in camera_pose)
    if len(pose) != 6:
        raise ValueError("相机位姿必须包含 6 个数值")
    offset, _side = resolve_sucker_offset(config, subject, pixel_xy)
    vx, vy = offset
    model = config.get("sucker_offset_model") or {}
    kind = model.get("type", "none")
    if kind == "linear_1d_x_residual":
        x = min(max(pose[0], model["clamp_min_x"]), model["clamp_max_x"])
        vx += model["k_x"] * (x - model["x0"])
    elif kind == "linear_2d":
        x = min(max(pose[0], model["clamp_min_x"]), model["clamp_max_x"])
        y = min(max(pose[1], model["clamp_min_y"]), model["clamp_max_y"])
        dx, dy = x - model["x0"], y - model["y0"]
        vx += model["kx"] * dx + model["ky"] * dy
        vy += model.get("lx", 0.0) * dx + model.get("ly", 0.0) * dy
    pose[0] += vx
    pose[1] += vy
    return pose


def run_offset_visual_servo_alignment(
    get_offset_func: Callable,
    move_pose_func: Callable,
    start_pose: Sequence[float],
    config: dict,
    speed: int,
    error_threshold_px: float,
    max_step_mm: float,
    max_iter: int,
    success_stable_frames: int,
    max_missed_frames: int,
    settle_sec: float,
    min_step_mm: float = 0.0,
    timing_debug: bool = False,
    log_label: str = "视觉伺服",
    event_callback: Optional[Callable[[dict], None]] = None,
    post_success_sample_frames: int = 0,
    abort_check: Optional[Callable[[], None]] = None,
):
    pose = list(float(value) for value in start_pose)
    stable_count = 0
    missed_count = 0
    last_response = None
    for iteration in range(int(max_iter)):
        if abort_check is not None:
            abort_check()
        round_started_at = time.perf_counter()
        response = get_offset_func()
        if abort_check is not None:
            abort_check()
        detection_finished_at = time.perf_counter()
        last_response = response
        if not response.found:
            missed_count += 1
            stable_count = 0
            print(
                f"[{log_label}] 第 {iteration + 1} 轮未识别，"
                f"连续丢失 {missed_count}/{max_missed_frames}: {response.message}"
            )
            if missed_count < max_missed_frames and settle_sec > 0:
                time.sleep(settle_sec)
                if abort_check is not None:
                    abort_check()
            round_finished_at = time.perf_counter()
            _emit_event(
                event_callback,
                "目标丢失",
                iteration,
                response,
                pose,
                detection_ms=(detection_finished_at - round_started_at) * 1000.0,
                settle_ms=(round_finished_at - detection_finished_at) * 1000.0,
                total_ms=(round_finished_at - round_started_at) * 1000.0,
            )
            if timing_debug:
                print(
                    f"[{log_label}耗时] 第 {iteration + 1} 轮 "
                    f"图像服务={(detection_finished_at - round_started_at) * 1000:.1f}ms，"
                    f"稳定等待={(round_finished_at - detection_finished_at) * 1000:.1f}ms，"
                    f"本轮总={(round_finished_at - round_started_at) * 1000:.1f}ms"
                )
            if missed_count >= max_missed_frames:
                return False, pose, last_response, "连续多帧未识别到目标"
            continue

        missed_count = 0
        error = max(abs(response.dx_px), abs(response.dy_px))
        if error <= error_threshold_px:
            stable_count += 1
            print(
                f"[{log_label}] 第 {iteration + 1} 轮"
                f"误差=({response.dx_px:+.2f},{response.dy_px:+.2f})px，"
                f"满足阈值 {error_threshold_px:.2f}px，"
                f"稳定帧 {stable_count}/{success_stable_frames}"
            )
            if stable_count < success_stable_frames and settle_sec > 0:
                time.sleep(settle_sec)
                if abort_check is not None:
                    abort_check()
            round_finished_at = time.perf_counter()
            _emit_event(
                event_callback,
                "稳定帧",
                iteration,
                response,
                pose,
                detection_ms=(detection_finished_at - round_started_at) * 1000.0,
                settle_ms=(round_finished_at - detection_finished_at) * 1000.0,
                total_ms=(round_finished_at - round_started_at) * 1000.0,
                stable_frame_index=stable_count,
            )
            if timing_debug:
                print(
                    f"[{log_label}耗时] 第 {iteration + 1} 轮 "
                    f"图像服务={(detection_finished_at - round_started_at) * 1000:.1f}ms，"
                    f"稳定等待={(round_finished_at - detection_finished_at) * 1000:.1f}ms，"
                    f"本轮总={(round_finished_at - round_started_at) * 1000:.1f}ms"
                )
            if stable_count >= success_stable_frames:
                # 成功后的附加帧仅用于估计检测噪声，绝不再发送运动命令，
                # 也不因诊断帧丢失或服务异常撤销已经成立的对准结果。
                for sample_index in range(max(0, int(post_success_sample_frames))):
                    if abort_check is not None:
                        abort_check()
                    sample_started_at = time.perf_counter()
                    try:
                        sample_response = get_offset_func()
                        sample_finished_at = time.perf_counter()
                        sample_event = (
                            "成功后静止帧"
                            if bool(getattr(sample_response, "found", False))
                            else "成功后静止丢失"
                        )
                        _emit_event(
                            event_callback,
                            sample_event,
                            iteration,
                            sample_response,
                            pose,
                            detection_ms=(sample_finished_at - sample_started_at) * 1000.0,
                            total_ms=(sample_finished_at - sample_started_at) * 1000.0,
                            static_sample_index=sample_index + 1,
                        )
                    except Exception as exc:  # noqa: BLE001
                        sample_finished_at = time.perf_counter()
                        _emit_event(
                            event_callback,
                            "成功后静止异常",
                            iteration,
                            None,
                            pose,
                            detection_ms=(sample_finished_at - sample_started_at) * 1000.0,
                            total_ms=(sample_finished_at - sample_started_at) * 1000.0,
                            static_sample_index=sample_index + 1,
                            message_override=str(exc),
                        )
                        print(f"[{log_label}] 成功后静止采样 {sample_index + 1} 异常: {exc}")
                    if abort_check is not None:
                        abort_check()
                return True, pose, last_response, "视觉伺服对准成功"
            continue

        stable_count = 0
        control_calculation_started_at = time.perf_counter()
        delta_xy = pixel_error_to_robot_delta(
            response.dx_px,
            response.dy_px,
            config["pixel_to_robot_matrix"],
            max_step_mm,
            min_step_mm,
        )
        control_calculation_finished_at = time.perf_counter()
        pose[0] += float(delta_xy[0])
        pose[1] += float(delta_xy[1])
        print(
            f"[{log_label}] 第 {iteration + 1} 轮"
            f"误差=({response.dx_px:+.2f},{response.dy_px:+.2f})px，"
            f"修正=({delta_xy[0]:+.3f},{delta_xy[1]:+.3f})mm"
        )
        control_started_at = time.perf_counter()
        move_pose_func(pose, speed=speed, wait_sec=0.0, wait_until_stable=True)
        if abort_check is not None:
            abort_check()
        arm_arrived_at = time.perf_counter()
        if settle_sec > 0:
            time.sleep(settle_sec)
            if abort_check is not None:
                abort_check()
        round_finished_at = time.perf_counter()
        _emit_event(
            event_callback,
            "执行修正",
            iteration,
            response,
            pose,
            correction_xy=delta_xy,
            detection_ms=(detection_finished_at - round_started_at) * 1000.0,
            control_ms=(control_calculation_finished_at - control_calculation_started_at) * 1000.0,
            arm_motion_ms=(arm_arrived_at - control_started_at) * 1000.0,
            settle_ms=(round_finished_at - arm_arrived_at) * 1000.0,
            total_ms=(round_finished_at - round_started_at) * 1000.0,
        )
        if timing_debug:
            print(
                f"[{log_label}耗时] 第 {iteration + 1} 轮 "
                f"图像服务={(detection_finished_at - round_started_at) * 1000:.1f}ms，"
                f"控制计算={(control_calculation_finished_at - control_calculation_started_at) * 1000:.1f}ms，"
                f"机械臂控制及到位={(arm_arrived_at - control_started_at) * 1000:.1f}ms，"
                f"稳定等待={(round_finished_at - arm_arrived_at) * 1000:.1f}ms，"
                f"本轮总={(round_finished_at - round_started_at) * 1000:.1f}ms"
            )
    return False, pose, last_response, "达到最大迭代次数仍未连续稳定"
