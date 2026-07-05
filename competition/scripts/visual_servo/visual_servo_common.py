#!/home/zhl/fr3env/fr3env/bin/python
import os
import time

import numpy as np
import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
VISUAL_SERVO_CONFIG_PATH = os.path.join(SRC_DIR, "competition", "config", "visual_servo.yaml")

DEFAULT_CONFIG = {
    "pixel_to_robot_matrix": [[0.0, 0.5], [0.5, 0.0]],
    "camera_to_sucker_offset_mm": [0.0, 0.0],
    "last_calibration": {
        "pixel_motion": "未标定",
        "camera_sucker_offset": "未标定",
    },
}


def load_visual_servo_config(config_path=VISUAL_SERVO_CONFIG_PATH):
    """读取视觉伺服配置；缺字段时用默认值补齐，正式运行统一从这里拿参数。"""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    config = DEFAULT_CONFIG.copy()
    config["last_calibration"] = DEFAULT_CONFIG["last_calibration"].copy()
    config.update(data)
    if "last_calibration" in data and isinstance(data["last_calibration"], dict):
        config["last_calibration"].update(data["last_calibration"])
    return config


def save_visual_servo_config(config, config_path=VISUAL_SERVO_CONFIG_PATH):
    """保存视觉伺服配置，标定工具或人工调参脚本统一写这个文件。"""
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)


def limit_xy_step(delta_xy, max_step_mm):
    """限制单次 XY 修正量，防止一次识别错误导致机械臂大幅运动。"""
    delta_xy = np.array(delta_xy, dtype=float)
    norm = float(np.linalg.norm(delta_xy))
    if norm <= max_step_mm or norm <= 1e-9:
        return delta_xy
    return delta_xy / norm * max_step_mm


def pixel_error_to_robot_delta(dx_px, dy_px, pixel_to_robot_matrix, max_step_mm):
    """把像素误差换算成机械臂 XY 修正量，并做限幅。

    pixel_to_robot_matrix 表示 [dx_px, dy_px] 到 [dx_mm, dy_mm] 的局部线性映射。
    方块和托盘暂时共用同一矩阵；如果后续实测发现高度或目标不同导致映射不同，
    再把配置拆成 block_pixel_to_robot_matrix 和 board_pixel_to_robot_matrix。
    """
    matrix = np.array(pixel_to_robot_matrix, dtype=float)
    pixel_error = np.array([float(dx_px), float(dy_px)], dtype=float)
    delta_xy = matrix @ pixel_error
    return limit_xy_step(delta_xy, max_step_mm)


def run_offset_visual_servo_alignment(
    get_offset_func,
    move_pose_func,
    start_pose,
    config,
    speed,
    error_threshold_px,
    max_step_mm,
    max_iter,
    success_stable_frames,
    max_missed_frames,
    settle_sec,
):
    """通用视觉伺服闭环。

    get_offset_func 是无参数函数，内部可以请求方块或托盘图像服务，但返回对象必须至少包含
    found、dx_px、dy_px、message 字段。
    move_pose_func 是主流程传入的机械臂移动函数，签名为 move_pose_func(pose, speed, wait_sec)，
    这样闭环运动仍然走 competition.py 的统一运动入口。
    """
    pose = list(start_pose)
    stable_count = 0
    missed_count = 0
    last_resp = None
    pixel_to_robot_matrix = config["pixel_to_robot_matrix"]

    for iter_idx in range(max_iter):
        resp = get_offset_func()
        last_resp = resp
        if not resp.found:
            missed_count += 1
            stable_count = 0
            print(f"[视觉伺服] 第 {iter_idx + 1} 轮未识别，连续丢失 {missed_count}/{max_missed_frames}: {resp.message}")
            if missed_count >= max_missed_frames:
                return False, pose, last_resp, "连续多帧未识别到目标"
            time.sleep(settle_sec)
            continue

        missed_count = 0
        err = max(abs(resp.dx_px), abs(resp.dy_px))
        if err <= error_threshold_px:
            stable_count += 1
            print(
                f"[视觉伺服] 第 {iter_idx + 1} 轮满足阈值，"
                f"目标={getattr(resp, 'target_type', '')}，"
                f"误差=({resp.dx_px:.2f},{resp.dy_px:.2f})px，"
                f"连续成功 {stable_count}/{success_stable_frames}"
            )
            if stable_count >= success_stable_frames:
                return True, pose, last_resp, "视觉伺服对准成功"
            time.sleep(settle_sec)
            continue

        stable_count = 0
        delta_xy = pixel_error_to_robot_delta(
            resp.dx_px,
            resp.dy_px,
            pixel_to_robot_matrix,
            max_step_mm,
        )
        pose[0] += float(delta_xy[0])
        pose[1] += float(delta_xy[1])
        print(
            f"[视觉伺服] 第 {iter_idx + 1} 轮目标={getattr(resp, 'target_type', '')}，"
            f"误差=({resp.dx_px:.2f},{resp.dy_px:.2f})px，"
            f"修正=({delta_xy[0]:.3f},{delta_xy[1]:.3f})mm，目标pose={pose}"
        )
        move_pose_func(pose, speed=speed, wait_sec=settle_sec)

    return False, pose, last_resp, "达到最大迭代次数仍未连续稳定"


def apply_camera_to_sucker_offset(camera_pose, config):
    """相机中心已对准目标时，按标定偏移生成吸盘高位姿态。"""
    pose = list(camera_pose)
    offset_xy = config.get("camera_to_sucker_offset_mm", [0.0, 0.0])
    pose[0] += float(offset_xy[0])
    pose[1] += float(offset_xy[1])
    return pose
