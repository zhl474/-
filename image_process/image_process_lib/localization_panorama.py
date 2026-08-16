"""prepare_task 高位定位全景 JSON 组装：正式/标定两模式的纯数据序列化。"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from image_process_lib.board_scene_detector import (
    BOARD_COL_COUNT,
    BOARD_ROW_COUNT,
    interpolate_grid_point,
)
from image_process_lib.depth_rough_localization import ZPlaneFit
from image_process_lib.task_planner import ObservedBlock, PlacementTarget


def _finite(value, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} 不是有限数值: {value!r}")
    return number


def _image_document(image_shape) -> dict:
    shape = tuple(int(size) for size in image_shape[:2])
    return {"宽": shape[1], "高": shape[0]}


def _board_angle_document(board_angle_deg, board_grid_points) -> Optional[float]:
    if board_grid_points is None:
        return None
    return None if board_angle_deg is None else float(board_angle_deg)


def _grid_pixel_entries(board_grid_points) -> list[dict]:
    """遍历全部 140 格点像素；单点失败记录错误，不影响其他点。"""
    if board_grid_points is None:
        return []
    entries = []
    for row in range(1, BOARD_ROW_COUNT + 1):
        for col in range(1, BOARD_COL_COUNT + 1):
            entry = {"行": row, "列": col, "像素u": None, "像素v": None, "错误": ""}
            try:
                point = interpolate_grid_point(board_grid_points, row=row, col=col)
                entry["像素u"] = _finite(point[0], "格点像素u")
                entry["像素v"] = _finite(point[1], "格点像素v")
            except Exception as exc:
                entry["错误"] = f"格点像素插值失败: {exc}"
            entries.append(entry)
    return entries


def _observed_block_entries(
    observed_blocks: Sequence[ObservedBlock],
    pick_surface_offset_mm: float,
    localizer=None,
) -> list[dict]:
    entries = []
    for block in observed_blocks:
        pixel = (float(block.high_detected_pixel_xy[0]), float(block.high_detected_pixel_xy[1]))
        entry = {
            "类别": str(block.category),
            "识别角度deg": float(block.detected_angle_deg),
            "像素u": _finite(pixel[0], "方块像素u"),
            "像素v": _finite(pixel[1], "方块像素v"),
            "观察TCP_X": _finite(block.observation_pose[0], "方块观察TCP_X"),
            "观察TCP_Y": _finite(block.observation_pose[1], "方块观察TCP_Y"),
            "观察TCP_Z": _finite(block.observation_pose[2], "方块观察TCP_Z"),
            "表面Z毫米": _finite(block.pick_surface_z_mm, "方块表面Z"),
            "抓取Z毫米": _finite(
                float(block.pick_surface_z_mm) + float(pick_surface_offset_mm),
                "方块抓取Z",
            ),
            "凸包内": None,
            "错误": "",
        }
        if localizer is not None:
            try:
                entry["凸包内"] = bool(
                    localizer.is_pixel_within_coverage("block", pixel)
                )
            except Exception as exc:
                entry["错误"] = f"方块凸包判断失败: {exc}"
        entries.append(entry)
    return entries


def build_formal_panorama_document(
    *,
    generated_at: str,
    image_shape,
    board_angle_deg,
    board_grid_points,
    observed_blocks: Sequence[ObservedBlock],
    localizer,
    pick_surface_offset_mm: float,
    block_calibration_sha256: str,
    tray_calibration_sha256: str,
) -> dict:
    """正式模式：140 格点 + 方块全部走标定预测并附标定元信息。"""
    grid_entries = _grid_pixel_entries(board_grid_points)
    for entry in grid_entries:
        if entry["错误"]:
            entry.update(
                {
                    "TCP_X": None,
                    "TCP_Y": None,
                    "TCP_Z": None,
                    "凸包内": None,
                    "安全越界轴": [],
                }
            )
            continue
        pixel = (entry["像素u"], entry["像素v"])
        try:
            assessment = localizer.assess("tray", pixel)
            entry.update(
                {
                    "TCP_X": float(assessment.predicted_tcp_xyz[0]),
                    "TCP_Y": float(assessment.predicted_tcp_xyz[1]),
                    "TCP_Z": float(assessment.predicted_tcp_xyz[2]),
                    "凸包内": bool(localizer.is_pixel_within_coverage("tray", pixel)),
                    "安全越界轴": list(assessment.violated_axes),
                }
            )
        except Exception as exc:
            entry.update(
                {
                    "TCP_X": None,
                    "TCP_Y": None,
                    "TCP_Z": None,
                    "凸包内": None,
                    "安全越界轴": [],
                    "错误": f"托盘标定预测失败: {exc}",
                }
            )
    block_entries = _observed_block_entries(
        observed_blocks,
        pick_surface_offset_mm,
        localizer=localizer,
    )
    return {
        "生成时间": str(generated_at),
        "模式": "正式",
        "图像": _image_document(image_shape),
        "托盘旋转角deg": _board_angle_document(board_angle_deg, board_grid_points),
        "标定": {
            "方块": {
                **localizer.calibration_summary("block"),
                "文件sha256": str(block_calibration_sha256),
            },
            "托盘": {
                **localizer.calibration_summary("tray"),
                "文件sha256": str(tray_calibration_sha256),
            },
        },
        "托盘格点": grid_entries,
        "方块": block_entries,
    }


def build_calibration_panorama_document(
    *,
    generated_at: str,
    image_shape,
    board_angle_deg,
    board_grid_points,
    observed_blocks: Sequence[ObservedBlock],
    placement_targets: Sequence[PlacementTarget],
    block_plane: Optional[ZPlaneFit],
    pick_surface_offset_mm: float,
) -> dict:
    """标定模式：深度实测世界坐标与拟合 Z 平面全量留档。"""
    block_entries = _observed_block_entries(observed_blocks, pick_surface_offset_mm)
    for block, entry in zip(observed_blocks, block_entries):
        if bool(block.high_world_position_valid):
            entry.update(
                {
                    "深度世界X": _finite(block.high_world_position[0], "方块深度X"),
                    "深度世界Y": _finite(block.high_world_position[1], "方块深度Y"),
                    "深度世界Z": _finite(block.high_world_position[2], "方块深度Z"),
                    "深度中值毫米": float(block.depth_median_mm),
                    "深度MAD毫米": float(block.depth_mad_mm),
                }
            )
        else:
            entry.update(
                {
                    "深度世界X": None,
                    "深度世界Y": None,
                    "深度世界Z": None,
                    "深度中值毫米": float(block.depth_median_mm),
                    "深度MAD毫米": float(block.depth_mad_mm),
                }
            )
    tray_entries = []
    for target in placement_targets:
        world_valid = bool(target.high_world_position_valid)
        tray_entries.append(
            {
                "行": float(target.row),
                "列": float(target.col),
                "像素u": _finite(target.high_detected_pixel_xy[0], "托盘像素u"),
                "像素v": _finite(target.high_detected_pixel_xy[1], "托盘像素v"),
                "深度世界X": (
                    _finite(target.high_world_position[0], "托盘深度X")
                    if world_valid
                    else None
                ),
                "深度世界Y": (
                    _finite(target.high_world_position[1], "托盘深度Y")
                    if world_valid
                    else None
                ),
                "深度世界Z": (
                    _finite(target.high_world_position[2], "托盘深度Z")
                    if world_valid
                    else None
                ),
                "托盘TCP_X": _finite(target.observation_pose[0], "托盘TCP_X"),
                "托盘TCP_Y": _finite(target.observation_pose[1], "托盘TCP_Y"),
                "托盘TCP_Z": _finite(target.observation_pose[2], "托盘TCP_Z"),
                "深度中值毫米": float(target.depth_median_mm),
                "深度MAD毫米": float(target.depth_mad_mm),
            }
        )
    plane_document = None
    if isinstance(block_plane, ZPlaneFit):
        plane_document = {
            "系数a_b_c": [float(value) for value in block_plane.coefficients],
            "RMSE毫米": float(block_plane.rmse_mm),
        }
    return {
        "生成时间": str(generated_at),
        "模式": "标定",
        "图像": _image_document(image_shape),
        "托盘旋转角deg": _board_angle_document(board_angle_deg, board_grid_points),
        "方块观察Z平面": plane_document,
        "托盘格点像素": _grid_pixel_entries(board_grid_points),
        "方块": block_entries,
        "托盘采样点": tray_entries,
    }
