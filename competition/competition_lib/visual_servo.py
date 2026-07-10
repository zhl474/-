"""方块和托盘共用的像素偏差视觉伺服闭环。"""

import time
from typing import Callable, Sequence

import numpy as np


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


def apply_camera_to_sucker_offset(camera_pose: Sequence[float], config: dict):
    pose = list(float(value) for value in camera_pose)
    if len(pose) != 6:
        raise ValueError("相机位姿必须包含 6 个数值")
    offset = config["camera_to_sucker_offset_mm"]
    pose[0] += float(offset[0])
    pose[1] += float(offset[1])
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
):
    pose = list(float(value) for value in start_pose)
    stable_count = 0
    missed_count = 0
    last_response = None
    for iteration in range(int(max_iter)):
        round_started_at = time.perf_counter()
        response = get_offset_func()
        detection_finished_at = time.perf_counter()
        last_response = response
        if not response.found:
            missed_count += 1
            stable_count = 0
            print(
                f"[视觉伺服] 第 {iteration + 1} 轮未识别，"
                f"连续丢失 {missed_count}/{max_missed_frames}: {response.message}"
            )
            if missed_count >= max_missed_frames:
                if timing_debug:
                    print(
                        f"[视觉伺服耗时] 第 {iteration + 1} 轮 "
                        f"图像服务={(detection_finished_at - round_started_at) * 1000:.1f}ms，"
                        f"本轮总={(time.perf_counter() - round_started_at) * 1000:.1f}ms"
                    )
                return False, pose, last_response, "连续多帧未识别到目标"
            time.sleep(settle_sec)
            if timing_debug:
                round_finished_at = time.perf_counter()
                print(
                    f"[视觉伺服耗时] 第 {iteration + 1} 轮 "
                    f"图像服务={(detection_finished_at - round_started_at) * 1000:.1f}ms，"
                    f"稳定等待={(round_finished_at - detection_finished_at) * 1000:.1f}ms，"
                    f"本轮总={(round_finished_at - round_started_at) * 1000:.1f}ms"
                )
            continue

        missed_count = 0
        error = max(abs(response.dx_px), abs(response.dy_px))
        if error <= error_threshold_px:
            stable_count += 1
            if stable_count >= success_stable_frames:
                if timing_debug:
                    print(
                        f"[视觉伺服耗时] 第 {iteration + 1} 轮 "
                        f"图像服务={(detection_finished_at - round_started_at) * 1000:.1f}ms，"
                        f"本轮总={(time.perf_counter() - round_started_at) * 1000:.1f}ms"
                    )
                return True, pose, last_response, "视觉伺服对准成功"
            time.sleep(settle_sec)
            if timing_debug:
                round_finished_at = time.perf_counter()
                print(
                    f"[视觉伺服耗时] 第 {iteration + 1} 轮 "
                    f"图像服务={(detection_finished_at - round_started_at) * 1000:.1f}ms，"
                    f"稳定等待={(round_finished_at - detection_finished_at) * 1000:.1f}ms，"
                    f"本轮总={(round_finished_at - round_started_at) * 1000:.1f}ms"
                )
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
            f"[视觉伺服] 第 {iteration + 1} 轮误差=({response.dx_px:.2f},{response.dy_px:.2f})px，"
            f"修正=({delta_xy[0]:.3f},{delta_xy[1]:.3f})mm"
        )
        control_started_at = time.perf_counter()
        move_pose_func(pose, speed=speed, wait_sec=0.0)
        arm_arrived_at = time.perf_counter()
        if settle_sec > 0:
            time.sleep(settle_sec)
        round_finished_at = time.perf_counter()
        if timing_debug:
            print(
                f"[视觉伺服耗时] 第 {iteration + 1} 轮 "
                f"图像服务={(detection_finished_at - round_started_at) * 1000:.1f}ms，"
                f"控制计算={(control_calculation_finished_at - control_calculation_started_at) * 1000:.1f}ms，"
                f"误差日志输出={(control_started_at - control_calculation_finished_at) * 1000:.1f}ms，"
                f"机械臂控制及到位={(arm_arrived_at - control_started_at) * 1000:.1f}ms，"
                f"稳定等待={(round_finished_at - arm_arrived_at) * 1000:.1f}ms，"
                f"本轮总={(round_finished_at - round_started_at) * 1000:.1f}ms"
            )
    return False, pose, last_response, "达到最大迭代次数仍未连续稳定"
