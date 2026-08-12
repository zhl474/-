"""任务布局与方块分配。

本模块不依赖 ROS，也不保存跨调用状态。重复规划同一组输入应得到同一结果。
"""

from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Sequence

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from image_process_lib.block_category import normalize_category_name
from image_process_lib.task_geometry import build_support_graph, normalize_cells


# 与 board_scene_detector 的 BOARD_ROW_COUNT / BOARD_COL_COUNT 保持一致，
# 低位托盘检测只支持整数或 .5 的行列坐标。
CALIBRATION_TRAY_ROW_COUNT = 14
CALIBRATION_TRAY_COL_COUNT = 10

# 标定托盘目标总数及四种模式的抽取数量：整数点、左右中点、上下中点、四点中心。
CALIBRATION_TRAY_POINT_COUNT = 34
CALIBRATION_TRAY_MODE_COUNTS = {
    "dot": 9,
    "horizontal_mid": 9,
    "vertical_mid": 8,
    "cell_center": 8,
}
if sum(CALIBRATION_TRAY_MODE_COUNTS.values()) != CALIBRATION_TRAY_POINT_COUNT:
    raise ValueError("标定托盘各模式抽取数量之和必须等于目标总数")


def select_calibration_tray_points(seed=None):
    """无重复随机挑选 34 个托盘标定点，覆盖整数与 .5 四种目标模式。

    模式名与低位托盘检测的 dot/horizontal_mid/vertical_mid/cell_center 对应。
    各模式的坐标池互斥，因此总体天然无重复；返回顺序是随机打乱的。
    """
    row_count = CALIBRATION_TRAY_ROW_COUNT
    col_count = CALIBRATION_TRAY_COL_COUNT
    pools = {
        "dot": [
            (float(row), float(col))
            for row in range(1, row_count + 1)
            for col in range(1, col_count + 1)
        ],
        "horizontal_mid": [
            (float(row), float(col) + 0.5)
            for row in range(1, row_count + 1)
            for col in range(1, col_count)
        ],
        "vertical_mid": [
            (float(row) + 0.5, float(col))
            for row in range(1, row_count)
            for col in range(1, col_count + 1)
        ],
        "cell_center": [
            (float(row) + 0.5, float(col) + 0.5)
            for row in range(1, row_count)
            for col in range(1, col_count)
        ],
    }
    rng = random.Random(seed)
    selected = []
    for mode, count in CALIBRATION_TRAY_MODE_COUNTS.items():
        candidates = rng.sample(pools[mode], count)
        selected.extend(
            {"row": float(row), "col": float(col), "mode": mode}
            for row, col in candidates
        )
    rng.shuffle(selected)
    return selected


@dataclass(frozen=True)
class ObservedBlock:
    category: str
    observation_pose: Sequence[float]
    detected_angle_deg: float
    # 本轮正式规划内唯一的实体编号；旧调用未编号时保持 -1。
    source_id: int = -1
    # 所选高度策略得到的方块上表面绝对 Z，单位 mm。
    pick_surface_z_mm: float = 0.0
    # false 表示高度策略失败，执行端禁止继续下探抓取。
    pick_surface_z_valid: bool = False
    # 以下字段保留现有服务兼容性，并保存高位像素到 TCP 标定诊断。
    high_detected_pixel_xy: Sequence[float] = (0.0, 0.0)
    high_depth_sample_pixel_xy: Sequence[float] = (0.0, 0.0)
    high_image_center_xy: Sequence[float] = (0.0, 0.0)
    high_world_position: Sequence[float] = (0.0, 0.0, 0.0)
    high_world_position_valid: bool = False
    rough_localization_source: str = ""
    depth_valid_frame_count: int = 0
    depth_median_mm: float = 0.0
    depth_mad_mm: float = 0.0
    calibration_target_tcp_z_mm: float = 0.0


@dataclass(frozen=True)
class PlacementTarget:
    index: int
    row: float
    col: float
    desired_angle_deg: float
    category: str
    observation_pose: Sequence[float]
    # 该目标在 14×10 托盘上占据的四个权威整数格，格式为 (列, 行)。
    cells: Sequence[Sequence[int]] = ()
    # 以下字段保留现有服务兼容性，并保存高位托盘像素标定诊断。
    high_detected_pixel_xy: Sequence[float] = (0.0, 0.0)
    high_depth_sample_pixel_xy: Sequence[float] = (0.0, 0.0)
    high_image_center_xy: Sequence[float] = (0.0, 0.0)
    high_world_position: Sequence[float] = (0.0, 0.0, 0.0)
    high_world_position_valid: bool = False
    rough_localization_source: str = ""
    depth_valid_frame_count: int = 0
    depth_median_mm: float = 0.0
    depth_mad_mm: float = 0.0
    calibration_target_tcp_z_mm: float = 0.0


@dataclass(frozen=True)
class TaskTarget:
    index: int
    category: str
    row: float
    col: float
    pick_observation_pose: Sequence[float]
    place_observation_pose: Sequence[float]
    detected_angle_deg: float
    rotation_delta_deg: float
    pick_surface_z_mm: float
    pick_surface_z_valid: bool
    # 被分配到该目标的本轮实体编号；标定任务和旧调用可保持 -1。
    source_id: int = -1
    # 目标种类：正式任务为 pick_place，标定方块为 block，标定托盘为 tray。
    target_type: str = "pick_place"
    # 抓取侧与摆放侧分别保留，避免任务分配后丢失高位标定诊断数据。
    pick_high_detected_pixel_xy: Sequence[float] = (0.0, 0.0)
    pick_high_depth_sample_pixel_xy: Sequence[float] = (0.0, 0.0)
    pick_high_image_center_xy: Sequence[float] = (0.0, 0.0)
    pick_high_world_position: Sequence[float] = (0.0, 0.0, 0.0)
    pick_high_world_position_valid: bool = False
    pick_rough_localization_source: str = ""
    place_high_detected_pixel_xy: Sequence[float] = (0.0, 0.0)
    place_high_depth_sample_pixel_xy: Sequence[float] = (0.0, 0.0)
    place_high_image_center_xy: Sequence[float] = (0.0, 0.0)
    place_high_world_position: Sequence[float] = (0.0, 0.0, 0.0)
    place_high_world_position_valid: bool = False
    place_rough_localization_source: str = ""
    pick_depth_valid_frame_count: int = 0
    pick_depth_median_mm: float = 0.0
    pick_depth_mad_mm: float = 0.0
    pick_calibration_target_tcp_z_mm: float = 0.0
    place_depth_valid_frame_count: int = 0
    place_depth_median_mm: float = 0.0
    place_depth_mad_mm: float = 0.0
    place_calibration_target_tcp_z_mm: float = 0.0


def load_task_layout(config_path: str) -> List[dict]:
    """读取并校验基础任务布局。"""
    with open(config_path, "r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    raw_targets = data.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("任务布局必须包含非空 targets 列表")

    targets = []
    for index, item in enumerate(raw_targets):
        try:
            category = normalize_category_name(item["category"])
            row = float(item["row"])
            col = float(item["col"])
            angle_deg = float(item["angle_deg"])
            cells = normalize_cells(item["cells"], f"任务布局第 {index} 项 cells")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"任务布局第 {index} 项无效: {item}") from exc
        if not category or not np.all(np.isfinite([row, col, angle_deg])):
            raise ValueError(f"任务布局第 {index} 项包含无效值: {item}")
        targets.append({
            "index": index,
            "row": row,
            "col": col,
            "angle_deg": angle_deg,
            "category": category,
            "cells": cells,
        })
    # 基础盘面在加载阶段即完成完整几何和支撑可达性检查，禁止错误盘面进入定位。
    build_support_graph(targets)
    return targets


def normalize_rotation_delta(category: str, desired_deg: float, detected_deg: float, board_deg: float) -> float:
    """按方块旋转对称性计算最短末端旋转量。"""
    category = normalize_category_name(category)
    delta = float(desired_deg) - float(detected_deg) + float(board_deg)
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    if category in ("z_green", "z_blue", "line"):
        if delta > 90.0:
            delta -= 180.0
        elif delta < -90.0:
            delta += 180.0
    elif category == "square":
        delta = delta % 90.0 if delta > 0 else delta % -90.0
        if delta > 45.0:
            delta -= 90.0
        elif delta < -45.0:
            delta += 90.0
    return float(delta)


def _validate_pose(pose: Sequence[float], label: str) -> np.ndarray:
    values = np.asarray(pose, dtype=float)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{label}必须是包含 6 个有限数值的位姿")
    return values


def _group_by_category(items: Iterable, category_getter) -> Dict[str, list]:
    grouped: Dict[str, list] = {}
    for item in items:
        category = normalize_category_name(category_getter(item))
        grouped.setdefault(category, []).append(item)
    return grouped


def make_task_target(
    block: ObservedBlock,
    target: PlacementTarget,
    board_angle_deg: float,
) -> TaskTarget:
    """由一个已经确定的实体—目标对应生成完整执行任务。"""
    block_category = normalize_category_name(block.category)
    target_category = normalize_category_name(target.category)
    if block_category != target_category:
        raise ValueError(
            f"实体类别 {block_category} 与目标类别 {target_category} 不一致"
        )
    _validate_pose(block.observation_pose, f"{block_category} 方块观察位")
    _validate_pose(target.observation_pose, f"{target_category} 摆放观察位")
    return TaskTarget(
        index=int(target.index),
        category=target_category,
        row=float(target.row),
        col=float(target.col),
        pick_observation_pose=tuple(float(value) for value in block.observation_pose),
        place_observation_pose=tuple(float(value) for value in target.observation_pose),
        detected_angle_deg=float(block.detected_angle_deg),
        rotation_delta_deg=normalize_rotation_delta(
            target_category,
            target.desired_angle_deg,
            block.detected_angle_deg,
            board_angle_deg,
        ),
        pick_surface_z_mm=float(block.pick_surface_z_mm),
        pick_surface_z_valid=bool(block.pick_surface_z_valid),
        source_id=int(block.source_id),
        pick_high_detected_pixel_xy=tuple(float(value) for value in block.high_detected_pixel_xy),
        pick_high_depth_sample_pixel_xy=tuple(
            float(value) for value in block.high_depth_sample_pixel_xy
        ),
        pick_high_image_center_xy=tuple(float(value) for value in block.high_image_center_xy),
        pick_high_world_position=tuple(float(value) for value in block.high_world_position),
        pick_high_world_position_valid=bool(block.high_world_position_valid),
        pick_rough_localization_source=str(block.rough_localization_source),
        pick_depth_valid_frame_count=int(block.depth_valid_frame_count),
        pick_depth_median_mm=float(block.depth_median_mm),
        pick_depth_mad_mm=float(block.depth_mad_mm),
        pick_calibration_target_tcp_z_mm=float(block.calibration_target_tcp_z_mm),
        place_high_detected_pixel_xy=tuple(
            float(value) for value in target.high_detected_pixel_xy
        ),
        place_high_depth_sample_pixel_xy=tuple(
            float(value) for value in target.high_depth_sample_pixel_xy
        ),
        place_high_image_center_xy=tuple(
            float(value) for value in target.high_image_center_xy
        ),
        place_high_world_position=tuple(float(value) for value in target.high_world_position),
        place_high_world_position_valid=bool(target.high_world_position_valid),
        place_rough_localization_source=str(target.rough_localization_source),
        place_depth_valid_frame_count=int(target.depth_valid_frame_count),
        place_depth_median_mm=float(target.depth_median_mm),
        place_depth_mad_mm=float(target.depth_mad_mm),
        place_calibration_target_tcp_z_mm=float(target.calibration_target_tcp_z_mm),
    )


def assign_blocks_to_targets(
    blocks: Sequence[ObservedBlock],
    targets: Sequence[PlacementTarget],
    board_angle_deg: float,
) -> List[TaskTarget]:
    """按类别使用匈牙利算法分配方块，并返回按目标序号排序的任务。"""
    blocks_by_category = _group_by_category(blocks, lambda item: item.category)
    targets_by_category = _group_by_category(targets, lambda item: item.category)
    target_by_index = {target.index: target for target in targets}
    if len(target_by_index) != len(targets):
        raise ValueError("摆放目标序号不能重复")

    result_by_index: Dict[int, TaskTarget] = {}
    for category, category_targets in targets_by_category.items():
        category_blocks = blocks_by_category.get(category, [])
        if len(category_blocks) < len(category_targets):
            raise ValueError(
                f"{category} 方块数量不足：检测到 {len(category_blocks)}，需要 {len(category_targets)}"
            )

        block_xy = np.array([
            _validate_pose(block.observation_pose, f"{category} 方块观察位")[:2]
            for block in category_blocks
        ])
        target_xy = np.array([
            _validate_pose(target.observation_pose, f"{category} 摆放观察位")[:2]
            for target in category_targets
        ])
        cost = np.linalg.norm(block_xy[:, None, :] - target_xy[None, :, :], axis=2)

        # 保留旧规划器的回程代价：目标 0 无回程，
        # 其余目标考虑到前一个全局目标的距离。
        for target_offset, target in enumerate(category_targets):
            if target.index <= 0:
                continue
            previous_target = target_by_index.get(target.index - 1)
            if previous_target is None:
                continue
            previous_xy = _validate_pose(previous_target.observation_pose, "前一摆放观察位")[:2]
            cost[:, target_offset] += np.linalg.norm(block_xy - previous_xy, axis=1)

        row_indices, col_indices = linear_sum_assignment(cost)
        assignment = {col: row for row, col in zip(row_indices, col_indices)}
        for target_offset, target in enumerate(category_targets):
            block = category_blocks[assignment[target_offset]]
            result_by_index[target.index] = make_task_target(
                block,
                target,
                board_angle_deg,
            )

    expected_indices = sorted(target_by_index)
    if sorted(result_by_index) != expected_indices:
        raise ValueError("任务规划结果不完整")
    return [result_by_index[index] for index in expected_indices]
