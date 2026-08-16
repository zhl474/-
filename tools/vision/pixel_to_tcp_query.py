#!/home/zhl/fr3env/fr3env/bin/python
# -*- coding: utf-8 -*-
"""输入方块或托盘的高位检测像素，用正式代码的标定结果换算 TCP 世界坐标。

直接修改下方 QUERY_POINTS 后运行，不需要命令行参数：
    /home/zhl/fr3env/fr3env/bin/python tools/vision/pixel_to_tcp_query.py

装配逻辑与 image_process 的 ImageProcessor 保持一致：
- 标定 YAML：perception.yaml 的 calibration.block/tray_pixel_to_tcp（相对 src 解析）；
- 拍摄姿态：execution.yaml 的 shooting_pose，后 3 位作为观察 R/P/YAW；
- 安全边界：perception.yaml 的 high_tcp_localization 安全 XY 范围，
  Z 下限取 execution.yaml 的 motion.minimum_tcp_z_mm；
- 安全 XY 偏移：execution.yaml 的 servo.enabled=false 时按开环模式
  使用 visual_servo.yaml 的 camera_to_sucker_offset_mm；
- 固定 TCP Z：perception.yaml 的 high_tcp_localization.fixed_tcp_z 开启时，
  Z 用常数覆盖标定 z_plane（工作台为平面、深度相机 Z 不可信），XY 仍走标定模型。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

SRC_DIR = Path(__file__).resolve().parents[2]
PACKAGE_DIR = SRC_DIR / "image_process"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from image_process_lib.high_pixel_to_tcp_localizer import (  # noqa: E402
    HighPixelToTcpLocalizer,
)
from image_process_lib.pixel_to_tcp_calibration import (  # noqa: E402
    PixelToTcpCalibration,
    load_pixel_to_tcp_calibration,
)

PERCEPTION_CONFIG_PATH = PACKAGE_DIR / "config" / "perception.yaml"
EXECUTION_CONFIG_PATH = SRC_DIR / "competition" / "config" / "execution.yaml"
VISUAL_SERVO_CONFIG_PATH = SRC_DIR / "competition" / "config" / "visual_servo.yaml"
DEFAULT_BLOCK_CALIBRATION_PATH = (
    PACKAGE_DIR / "config" / "block_pixel_to_tcp_calibration.yaml"
)
DEFAULT_TRAY_CALIBRATION_PATH = (
    PACKAGE_DIR / "config" / "tray_pixel_to_tcp_calibration.yaml"
)
SUBJECT_LABELS = {"block": "方块", "tray": "托盘"}

# ----------------------------- 直接运行配置 -----------------------------
# 要查询的主体与像素点：主体为 "block"（方块）或 "tray"（托盘）。
# 像素为高位检测的 (u, v)，u=列、v=行，可写小数；同一主体可写多个点。
# 默认点是各自标定凸包顶点的均值，保证落在标定覆盖范围内。
QUERY_POINTS: Dict[str, List[Tuple[float, float]]] = {
    "block": [(601.3, 292.8)],
    "tray": [(569.5, 353.6)],
}


@dataclass(frozen=True)
class FormalContext:
    """与正式节点一致的定位器及其装配来源。"""

    localizer: HighPixelToTcpLocalizer
    calibration_paths: Dict[str, Path]
    shooting_rpy: Tuple[float, float, float]
    servo_enabled: bool
    safety_xy_offset: Tuple[float, float]


def _load_yaml(path: Path, name: str) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as file_handle:
            data = yaml.safe_load(file_handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"无法读取{name}（{path}）：{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{name} 必须是字典：{path}")
    return data


def _src_path(value, fallback: Path) -> Path:
    """与 ImageProcessor.src_path 一致：相对路径按 src 目录解析。"""
    raw_path = str(value or fallback)
    return (
        Path(raw_path).expanduser()
        if os.path.isabs(raw_path)
        else SRC_DIR / raw_path
    )


def build_formal_context() -> FormalContext:
    """按正式节点的顺序读取三份配置并装配 HighPixelToTcpLocalizer。"""
    perception = _load_yaml(PERCEPTION_CONFIG_PATH, "perception.yaml")
    execution = _load_yaml(EXECUTION_CONFIG_PATH, "execution.yaml")
    visual_servo = _load_yaml(VISUAL_SERVO_CONFIG_PATH, "visual_servo.yaml")

    calibration_config = perception.get("calibration", {})
    block_path = _src_path(
        calibration_config.get("block_pixel_to_tcp"),
        DEFAULT_BLOCK_CALIBRATION_PATH,
    )
    tray_path = _src_path(
        calibration_config.get("tray_pixel_to_tcp"),
        DEFAULT_TRAY_CALIBRATION_PATH,
    )

    shooting_pose = [float(value) for value in execution["shooting_pose"]]
    if len(shooting_pose) != 6 or not np.all(np.isfinite(shooting_pose)):
        raise ValueError("execution.yaml 的 shooting_pose 必须包含 6 个有限数值")

    localization_config = perception.get("high_tcp_localization", {})
    safe_x = [float(value) for value in localization_config.get(
        "safe_x_range_mm", [-444.224, -148.17]
    )]
    safe_y = [float(value) for value in localization_config.get(
        "safe_y_range_mm", [-263.279, 315.925]
    )]
    motion_config = execution.get("motion", {})
    minimum_tcp_z = float(motion_config["minimum_tcp_z_mm"])

    servo_enabled = bool(execution.get("servo", {}).get("enabled", True))
    if servo_enabled:
        safety_xy_offset = (0.0, 0.0)
    else:
        safety_xy_offset = tuple(
            float(value) for value in visual_servo["camera_to_sucker_offset_mm"]
        )

    # 与 image_node 一致：固定 Z 模式开启时把 TCP Z 常数传给定位器。
    fixed_tcp_z_config = localization_config.get("fixed_tcp_z", {})
    fixed_tcp_z_mm = None
    if fixed_tcp_z_config.get("enabled", False):
        fixed_tcp_z_mm = {
            "block": fixed_tcp_z_config["block_observation_z_mm"],
            "tray": fixed_tcp_z_config["tray_z_mm"],
        }

    localizer = HighPixelToTcpLocalizer(
        block_calibration_path=block_path,
        tray_calibration_path=tray_path,
        shooting_pose=shooting_pose,
        tcp_min_xyz=(safe_x[0], safe_y[0], minimum_tcp_z),
        tcp_max_xyz=(safe_x[1], safe_y[1], None),
        safety_xy_offset=safety_xy_offset,
        fixed_tcp_z_mm=fixed_tcp_z_mm,
    )
    return FormalContext(
        localizer=localizer,
        calibration_paths={"block": block_path, "tray": tray_path},
        shooting_rpy=tuple(shooting_pose[3:6]),
        servo_enabled=servo_enabled,
        safety_xy_offset=safety_xy_offset,
    )


def load_subject_calibration(context: FormalContext, subject: str) -> PixelToTcpCalibration:
    """按主体加载部署标定，用于批次信息和凸包覆盖诊断。"""
    return load_pixel_to_tcp_calibration(
        context.calibration_paths[subject],
        expected_subject=subject,
    )


def print_calibration_info(
    subject: str,
    calibration: PixelToTcpCalibration,
    calibration_path: Path,
    fixed_tcp_z_mm=None,
) -> None:
    """打印部署标定的批次、模型和像素覆盖范围。"""
    document = calibration.metadata
    hull = calibration.pixel_convex_hull
    label = SUBJECT_LABELS[subject]
    if calibration.schema_version == 2:
        model_name = document["xy_model"]["name"]
        z_source = document["z_plane"]["source"]
    else:
        model_name = document["model"]["name"]
        z_source = "同平面直接输出（schema v1）"
    if fixed_tcp_z_mm is not None:
        # 固定 Z 模式下 yaml 里的 z_plane 只是留档对照，实际 Z 是常数。
        z_source += f"（已被固定常数 {fixed_tcp_z_mm[subject]:.2f} 覆盖）"
    print(f"[{label}] 标定文件：{calibration_path}")
    print(
        f"       批次 {document.get('generation_id', '无')}，"
        f"XY 模型 {model_name}，Z 平面来源 {z_source}"
    )
    print(
        f"       像素覆盖：u {hull[:, 0].min():.1f}~{hull[:, 0].max():.1f}，"
        f"v {hull[:, 1].min():.1f}~{hull[:, 1].max():.1f}"
    )


def format_xyz(xyz) -> str:
    return f"X={xyz[0]:.3f}, Y={xyz[1]:.3f}, Z={xyz[2]:.3f}"


def print_point_result(
    context: FormalContext,
    subject: str,
    calibration: PixelToTcpCalibration,
    pixel: Tuple[float, float],
) -> None:
    """查询一个像素点并打印 TCP、覆盖、安全和观察位姿结果。"""
    label = SUBJECT_LABELS[subject]
    print(f"  [{label}] 像素 (u={pixel[0]:.1f}, v={pixel[1]:.1f})")
    try:
        assessment = context.localizer.assess(subject, pixel)
    except ValueError as exc:
        print(f"    预测失败：{exc}")
        return

    print(f"    预测 TCP XYZ（mm）：{format_xyz(assessment.predicted_tcp_xyz)}")
    if not np.allclose(assessment.safety_tcp_xyz, assessment.predicted_tcp_xyz):
        print(
            f"    待执行 TCP XYZ（mm，含安全偏置 "
            f"{context.safety_xy_offset}）：{format_xyz(assessment.safety_tcp_xyz)}"
        )
    if calibration.is_pixel_within_coverage(pixel):
        print("    凸包覆盖：在标定覆盖范围内")
    else:
        print("    凸包覆盖：警告，像素在标定覆盖范围外，结果属于外推")

    if assessment.safe:
        print("    安全评估：在安全范围内")
    else:
        print(f"    安全评估：越界轴 {list(assessment.violated_axes)}")
        print(f"      待执行 {format_xyz(assessment.safety_tcp_xyz)}")
        print(
            f"      允许最小 {format_xyz(assessment.safety_min_xyz)}，"
            f"最大 {list(assessment.safety_max_xyz)}"
        )
    pose = [
        *assessment.predicted_tcp_xyz,
        *context.shooting_rpy,
    ]
    pose_text = ", ".join(f"{value:.3f}" for value in pose)
    print(f"    观察位姿 [X,Y,Z,R,P,YAW]：[{pose_text}]")


def main() -> None:
    """按直接运行配置查询所有像素点。"""
    context = build_formal_context()
    mode_text = "闭环（视觉伺服开）" if context.servo_enabled else "开环（视觉伺服关）"
    fixed_tcp_z_mm = context.localizer.fixed_tcp_z_mm
    print("使用正式部署配置装配 HighPixelToTcpLocalizer：")
    print(f"  安全校验模式：{mode_text}，安全 XY 偏移 {list(context.safety_xy_offset)}")
    print(f"  拍摄姿态 R/P/YAW：{list(context.shooting_rpy)}")
    if fixed_tcp_z_mm is not None:
        print(
            "  TCP Z 固定模式：已开启——"
            f"方块观察 Z={fixed_tcp_z_mm['block']:.2f}，"
            f"托盘 Z={fixed_tcp_z_mm['tray']:.2f}（标定 z_plane 已禁用）"
        )
    for subject, points in QUERY_POINTS.items():
        if subject not in SUBJECT_LABELS:
            print(f"\n未知主体 {subject!r}，仅支持 {sorted(SUBJECT_LABELS)}，已跳过")
            continue
        calibration = load_subject_calibration(context, subject)
        print()
        print_calibration_info(
            subject,
            calibration,
            context.calibration_paths[subject],
            fixed_tcp_z_mm=fixed_tcp_z_mm,
        )
        for pixel in points:
            print_point_result(context, subject, calibration, pixel)


if __name__ == "__main__":
    main()
