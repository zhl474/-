"""任务布局与方块分配。

本模块不依赖 ROS，也不保存跨调用状态。重复规划同一组输入应得到同一结果。
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from image_process_lib.block_category import normalize_category_name


@dataclass(frozen=True)
class ObservedBlock:
    category: str
    observation_pose: Sequence[float]
    detected_angle_deg: float
    # 高位深度相机测得的方块上表面绝对 Z；无有效深度时为 0。
    pick_surface_z_mm: float = 0.0
    # false 表示本次粗定位使用了无深度的像素比例回退，禁止按深度高度抓取。
    pick_surface_z_valid: bool = False


@dataclass(frozen=True)
class PlacementTarget:
    index: int
    row: float
    col: float
    desired_angle_deg: float
    category: str
    observation_pose: Sequence[float]


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
        })
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
            result_by_index[target.index] = TaskTarget(
                index=target.index,
                category=category,
                row=float(target.row),
                col=float(target.col),
                pick_observation_pose=tuple(float(value) for value in block.observation_pose),
                place_observation_pose=tuple(float(value) for value in target.observation_pose),
                detected_angle_deg=float(block.detected_angle_deg),
                rotation_delta_deg=normalize_rotation_delta(
                    category,
                    target.desired_angle_deg,
                    block.detected_angle_deg,
                    board_angle_deg,
                ),
                pick_surface_z_mm=float(block.pick_surface_z_mm),
                pick_surface_z_valid=bool(block.pick_surface_z_valid),
            )

    expected_indices = sorted(target_by_index)
    if sorted(result_by_index) != expected_indices:
        raise ValueError("任务规划结果不完整")
    return [result_by_index[index] for index in expected_indices]
