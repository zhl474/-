"""V5 动态盘面选择的精确回放报告序列化。"""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime
from typing import Mapping, Sequence

import numpy as np

from image_process_lib.board_candidate_selector import (
    BoardCandidateSelectionResult,
    CoarseSelectionResult,
    SourceTargetAssignment,
)
from image_process_lib.final_board_selector import (
    BoardOptimizationAttempt,
    FinalBoardDecision,
    build_unique_board_manifest,
)
from image_process_lib.task_planner import ObservedBlock
from image_process_lib.task_sequence_optimizer import build_task_plan_report
from image_process_lib.v5_board_library import V5BoardLibrary


DYNAMIC_BOARD_REPORT_PROTOCOL_VERSION = 1


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


_OBSERVED_FIELD_NAMES = {
    "category": "类别",
    "observation_pose": "观察位姿",
    "detected_angle_deg": "识别角度度",
    "source_id": "实体ID",
    "pick_surface_z_mm": "抓取表面Z毫米",
    "pick_surface_z_valid": "抓取表面Z有效",
    "high_detected_pixel_xy": "高位识别像素XY",
    "high_depth_sample_pixel_xy": "高位深度采样像素XY",
    "high_image_center_xy": "高位图像中心XY",
    "high_world_position": "高位世界坐标",
    "high_world_position_valid": "高位世界坐标有效",
    "rough_localization_source": "粗定位来源",
    "depth_valid_frame_count": "深度有效帧数",
    "depth_median_mm": "深度中位数毫米",
    "depth_mad_mm": "深度MAD毫米",
    "calibration_target_tcp_z_mm": "标定目标TCP_Z毫米",
}


def serialize_observed_block(block: ObservedBlock) -> dict:
    """保存 ObservedBlock 的全部字段，不丢弃回放输入。"""
    if not isinstance(block, ObservedBlock):
        raise TypeError("实体必须是 ObservedBlock")
    return {
        _OBSERVED_FIELD_NAMES[item.name]: _json_value(getattr(block, item.name))
        for item in fields(ObservedBlock)
    }


def deserialize_observed_block(document: Mapping) -> ObservedBlock:
    """从动态报告精确恢复 ObservedBlock。"""
    reverse = {chinese: english for english, chinese in _OBSERVED_FIELD_NAMES.items()}
    try:
        values = {
            reverse[key]: value
            for key, value in document.items()
            if key in reverse
        }
        missing = [item.name for item in fields(ObservedBlock) if item.name not in values]
        if missing:
            raise KeyError("、".join(missing))
        for name in (
            "observation_pose",
            "high_detected_pixel_xy",
            "high_depth_sample_pixel_xy",
            "high_image_center_xy",
            "high_world_position",
        ):
            values[name] = tuple(values[name])
        return ObservedBlock(**values)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"动态报告实体字段不完整：{exc}") from exc


def _assignment_document(assignment: SourceTargetAssignment) -> dict:
    return {
        "类别": assignment.category,
        "实体ID": int(assignment.source_id),
        "placement_PID": int(assignment.placement_id),
        "source区域位": int(assignment.source_region_bit),
        "target允许区域掩码": int(assignment.target_region_mask),
        "target匹配区域位": int(assignment.matched_target_region_bit),
        "距离毫米": (
            None if assignment.distance_mm is None else float(assignment.distance_mm)
        ),
        "等价旋转度": (
            None if assignment.rotation_deg is None else float(assignment.rotation_deg)
        ),
    }


def _coarse_document(result: CoarseSelectionResult | None):
    if result is None:
        return None
    return {
        "全库盘面数": int(result.total_board_count),
        "耗时秒": float(result.elapsed_seconds),
        "source签名": [[int(item[0]), item[1]] for item in result.source_signature],
        "source区域": [
            [int(source_id), int(region)]
            for source_id, region in result.source_region_by_id
        ],
        "候选": [
            {
                "盘面库下标": int(candidate.board_index),
                "盘面ID": candidate.board_id,
                "global_id": int(candidate.global_id),
                "layout_index": int(candidate.layout_index),
                "缺少类别": candidate.missing_category,
                "N_LR": int(candidate.n_lr),
                "N_UD": int(candidate.n_ud),
                "舍弃实体ID": int(candidate.unused_source_id),
                "配对": [
                    _assignment_document(item) for item in candidate.assignment
                ],
            }
            for candidate in result.candidates
        ],
    }


def _relaxed_document(result: BoardCandidateSelectionResult | None):
    if result is None:
        return None
    return {
        "全库盘面数": int(result.total_board_count),
        "粗筛候选数": int(result.coarse_candidate_count),
        "粗筛耗时秒": float(result.coarse_elapsed_seconds),
        "relaxed耗时秒": float(result.relaxed_elapsed_seconds),
        "候选": [
            {
                "盘面库下标": int(candidate.board_index),
                "盘面ID": candidate.board_id,
                "global_id": int(candidate.global_id),
                "layout_index": int(candidate.layout_index),
                "缺少类别": candidate.missing_category,
                "分数元组": [
                    int(candidate.n_lr),
                    int(candidate.n_ud),
                    float(candidate.relaxed_distance_mm),
                    float(candidate.relaxed_rotation_deg),
                ],
                "粗筛舍弃实体ID": int(candidate.coarse_unused_source_id),
                "relaxed舍弃实体ID": int(candidate.unused_source_id),
                "粗筛配对": [
                    _assignment_document(item) for item in candidate.coarse_assignment
                ],
                "relaxed配对": [
                    _assignment_document(item) for item in candidate.relaxed_assignment
                ],
            }
            for candidate in result.candidates
        ],
    }


def _attempt_document(attempt: BoardOptimizationAttempt) -> dict:
    return {
        "盘面库下标": int(attempt.board_index),
        "盘面ID": attempt.board_id,
        "global_id": int(attempt.global_id),
        "layout_index": int(attempt.layout_index),
        "缺少类别": attempt.missing_category,
        "筛选分数元组": _json_value(attempt.selector_score),
        "成功": bool(attempt.succeeded),
        "简化成本秒": attempt.simplified_cost_seconds,
        "舵机重放总时间秒": attempt.servo_replay_total_seconds,
        "选中方案来源": attempt.selected_origin,
        "PID执行顺序": list(attempt.target_pid_sequence),
        "source执行顺序": list(attempt.source_id_sequence),
        "未使用source_ID": list(attempt.unused_source_ids),
        "耗时秒": float(attempt.elapsed_seconds),
        "失败信息": attempt.error_message,
    }


def _grid_document(board_grid_points) -> list:
    rows = []
    for row in range(1, 15):
        points = []
        for col in range(1, 11):
            try:
                point = board_grid_points[row][col]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError(f"托盘格点缺少 ({row}, {col})") from exc
            values = np.asarray(point, dtype=float)
            if values.shape != (2,) or not np.all(np.isfinite(values)):
                raise ValueError(f"托盘格点 ({row}, {col}) 无效")
            points.append([float(values[0]), float(values[1])])
        rows.append(points)
    return rows


def deserialize_board_grid(document: Sequence) -> dict:
    """把报告中的 14×10 像素数组恢复为现有格点字典。"""
    array = np.asarray(document, dtype=float)
    if array.shape != (14, 10, 2) or not np.all(np.isfinite(array)):
        raise ValueError("动态报告托盘格点必须是 [14,10,2] 有限数组")
    return {
        row: {
            col: (float(array[row - 1, col - 1, 0]), float(array[row - 1, col - 1, 1]))
            for col in range(1, 11)
        }
        for row in range(1, 15)
    }


def build_dynamic_board_selection_report(
    *,
    mode: str,
    outcome: str,
    observed_blocks: Sequence[ObservedBlock],
    board_grid_points,
    tray_center_pixel_xy,
    board_angle_deg: float,
    image_shape,
    target_center_cache: Sequence[Mapping],
    library: V5BoardLibrary,
    runtime_config: Mapping,
    calibration_sha256: Mapping[str, str],
    coarse_result: CoarseSelectionResult | None = None,
    relaxed_result: BoardCandidateSelectionResult | None = None,
    decision: FinalBoardDecision | None = None,
    comparison_attempts: Sequence[BoardOptimizationAttempt] = (),
    error_message: str = "",
    human_failure_choice: str = "",
    actual_planner_message: str = "",
) -> dict:
    """构造可精确回放“粗筛→确认”链路的中文 JSON 文档。"""
    attempts = (
        decision.comparison_attempts
        if decision is not None
        else tuple(comparison_attempts)
    )
    document = {
        "协议版本": DYNAMIC_BOARD_REPORT_PROTOCOL_VERSION,
        "报告类型": "V5动态盘面选择报告",
        "生成时间": datetime.now().astimezone().isoformat(),
        "运行模式": str(mode),
        "本轮结果": str(outcome),
        "实际规划结果": str(actual_planner_message),
        "失败原因": str(error_message),
        "人工失败选择": str(human_failure_choice),
        "运行参数": _json_value(runtime_config),
        "高位输入快照": {
            "实体": [serialize_observed_block(item) for item in observed_blocks],
            "托盘中心像素XY": _json_value(tray_center_pixel_xy),
            "托盘格点": _grid_document(board_grid_points),
            "盘面角度度": float(board_angle_deg),
            "图像尺寸": [int(value) for value in image_shape[:2]],
        },
        "实际转换的目标中心缓存": _json_value(target_center_cache),
        "文件身份": {
            "V5盘面库路径": (
                "" if library.source_path is None else str(library.source_path)
            ),
            "V5盘面库SHA256": library.source_sha256,
            "标定文件SHA256": _json_value(calibration_sha256),
        },
        "四区粗筛": _coarse_document(coarse_result),
        "relaxed筛选": _relaxed_document(relaxed_result),
        "20盘比较": [_attempt_document(item) for item in attempts],
        "唯一盘面清单": (
            None if decision is None else build_unique_board_manifest(decision)
        ),
        "唯一盘面确认路径报告": (
            None
            if decision is None
            else build_task_plan_report(decision.confirmation_result)
        ),
    }
    return document
