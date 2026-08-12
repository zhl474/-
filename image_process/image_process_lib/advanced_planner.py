"""进阶任务 IDBS 动态库的版本化四格几何包装。"""

from ctypes import CDLL, c_char_p, c_int
import math
from typing import List, Sequence, Tuple

from image_process_lib.block_category import BLOCK_CATEGORY_NAMES, normalize_category_name
from image_process_lib.task_geometry import build_support_graph, normalize_cells


IDBS_WITH_CELLS_PROTOCOL = "IDBS_WITH_CELLS_V1"
IDBS_WITH_CELLS_FIELDS_PER_BLOCK = 12


def parse_idbs_with_cells_result(raw_result: bytes, expected_count: int) -> Tuple[List[dict], str]:
    """严格解析版本、数量、中心和四个运行时托盘格。"""
    if not raw_result:
        raise RuntimeError("IDBSWithCells 未返回规划结果")
    try:
        fields = raw_result.decode("gbk").split(",")
    except UnicodeDecodeError as exc:
        raise ValueError("IDBSWithCells 返回值不是有效 GBK 文本") from exc
    if len(fields) < 3 or fields[0] != IDBS_WITH_CELLS_PROTOCOL:
        raise ValueError("IDBSWithCells 协议版本错误")
    try:
        block_count = int(fields[1])
    except (TypeError, ValueError) as exc:
        raise ValueError("IDBSWithCells 方块数量字段无效") from exc
    if block_count < 0 or block_count != int(expected_count):
        raise ValueError(
            f"IDBSWithCells 方块数量错误：返回 {block_count}，预期 {expected_count}"
        )
    expected_field_count = 3 + block_count * IDBS_WITH_CELLS_FIELDS_PER_BLOCK
    if len(fields) != expected_field_count:
        raise ValueError(
            f"IDBSWithCells 字段数错误：返回 {len(fields)}，"
            f"预期 {expected_field_count}"
        )

    layout = []
    for index in range(block_count):
        offset = 2 + index * IDBS_WITH_CELLS_FIELDS_PER_BLOCK
        category = normalize_category_name(fields[offset])
        if category not in BLOCK_CATEGORY_NAMES:
            raise ValueError(f"IDBSWithCells 第 {index} 块类别无效：{category}")
        try:
            angle_deg = float(fields[offset + 1])
            center_col = float(fields[offset + 2])
            center_row = float(fields[offset + 3])
            raw_cells = [
                (int(fields[offset + 4 + cell_index * 2]),
                 int(fields[offset + 5 + cell_index * 2]))
                for cell_index in range(4)
            ]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"IDBSWithCells 第 {index} 块数值字段无效") from exc
        if not all(math.isfinite(value) for value in (angle_deg, center_col, center_row)):
            raise ValueError(f"IDBSWithCells 第 {index} 块包含非有限数值")
        if not 1.0 <= center_col <= 10.0 or not 1.0 <= center_row <= 14.0:
            raise ValueError(f"IDBSWithCells 第 {index} 块中心超出 10×14 托盘范围")
        cells = normalize_cells(raw_cells, f"IDBSWithCells 第 {index} 块 cells")
        layout.append({
            "index": index,
            "category": category,
            "angle_deg": angle_deg,
            "col": center_col,
            "row": center_row,
            "cells": cells,
        })
    # 动态库输出在进入图像插值前即校验重叠、支撑候选和全盘可达性。
    build_support_graph(layout)
    return layout, fields[-1]


class AdvancedPlanner:
    def __init__(self, library_path: str):
        self.library_path = library_path

    def build_layout(self, cube_counts: Sequence[int], place_order: Sequence[int]) -> Tuple[List[dict], str]:
        if len(cube_counts) != 7 or len(place_order) != 7:
            raise ValueError("进阶任务需要 7 类方块数量和 7 项摆放顺序")
        try:
            count_values = [int(value) for value in cube_counts]
            order_values = [int(value) for value in place_order]
        except (TypeError, ValueError) as exc:
            raise ValueError("进阶任务数量和顺序必须是整数") from exc
        if any(value < 0 for value in count_values):
            raise ValueError("进阶任务方块数量不能为负数")
        if sorted(order_values) != list(range(7)):
            raise ValueError("进阶任务摆放顺序必须是 0～6 的无重复排列")
        library = CDLL(self.library_path)
        library.IDBSWithCells.argtypes = [c_int] * 14
        library.IDBSWithCells.restype = c_char_p
        raw_result = library.IDBSWithCells(*count_values, *order_values)
        return parse_idbs_with_cells_result(raw_result, sum(count_values))
