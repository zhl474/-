"""进阶任务 IDBS 动态库的版本化四格几何包装。"""

from __future__ import annotations

from ctypes import CDLL, POINTER, c_char_p, c_double, c_int
import math
from typing import List, Sequence, Tuple

from image_process_lib.block_category import BLOCK_CATEGORY_NAMES, normalize_category_name
from image_process_lib.task_geometry import build_support_graph, normalize_cells


IDBS_WITH_CELLS_PROTOCOL = "IDBS_WITH_CELLS_V1"
IDBS_WITH_CELLS_FIELDS_PER_BLOCK = 12
IDBS_CONFIG_FIELDS_PER_BLOCK = 5

# 与 C++ 动态库内部 brick 定义保持一致，用于把新版 IDBS_Config 的输出中心
# 还原成项目后续需要的四个占用格 (列, 行)。
_TETRIS_ROWS = 14
_TETRIS_COLS = 10
_TETRIS_BRICKS = [
    [  # L_blue / LR
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, -1)],
    ],
    [  # L_yellow / LL
        [(0, 0), (1, 0), (1, -1), (1, -2)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (1, 0), (0, 1), (0, 2)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ],
    [  # z_blue / ZL
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (1, -1), (2, -1)],
    ],
    [  # z_green / ZR
        [(0, 0), (0, 1), (1, 0), (1, -1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ],
    [  # square / O
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],
    [  # T
        [(0, 0), (1, 0), (1, -1), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 0), (1, 0), (2, 0), (1, -1)],
    ],
    [  # line
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
    ],
]
_TETRIS_ROTATION_COUNTS = [len(shape) for shape in _TETRIS_BRICKS]


def _output_angle(category_index: int, rotation: int) -> float:
    """复刻 C++ 动态库从 rotation 到输出角度的换算。"""
    raw_angle = rotation * 90
    angle = raw_angle % 360
    if angle > 180:
        angle -= 360
    if angle <= -180:
        angle += 360
    if category_index in (0, 1, 5):  # L_blue / L_yellow / T
        angle = angle + 180
        if angle > 180:
            angle -= 360
        if angle <= -180 and category_index in (0, 1):
            angle += 360
    elif category_index in (2, 3, 6):  # Z / Line
        angle = (rotation % 2) * 90
    elif category_index == 4:  # square
        angle = 0.0
    return angle


def _calculate_center_col_row(
    category_index: int,
    rotation: int,
    grid_row: int,
    grid_col: int,
) -> Tuple[float, float, float]:
    """复刻 C++ calculate_center + 智能角度/坐标补偿，返回 (col, row, angle)。"""
    offsets = _TETRIS_BRICKS[category_index][rotation]
    min_dx = min(dx for dx, _ in offsets)
    max_dx = max(dx for dx, _ in offsets)
    min_dy = min(dy for _, dy in offsets)
    max_dy = max(dy for _, dy in offsets)

    cx = grid_col + 0.5 + (min_dy + max_dy) / 2.0
    cy = (_TETRIS_ROWS - 1) - grid_row + 0.5 - (min_dx + max_dx) / 2.0
    x = _TETRIS_COLS - cx
    y = _TETRIS_ROWS - cy
    angle = _output_angle(category_index, rotation)

    # L 形智能坐标补偿
    if category_index in (0, 1):
        if math.isclose(angle, 0.0, abs_tol=1e-9):
            y -= 0.5
        elif math.isclose(angle, 90.0, abs_tol=1e-9):
            x -= 0.5
        elif math.isclose(angle, 180.0, abs_tol=1e-9):
            y += 0.5
        elif math.isclose(angle, -90.0, abs_tol=1e-9):
            x += 0.5

    return x + 0.5, y + 0.5, angle


def _cells_from_config_item(
    category: str,
    angle_deg: float,
    col: float,
    row: float,
) -> Tuple[Tuple[int, int], ...]:
    """根据新版输出的中心/角度反查四个占用格。

    新版 IDBS_Config 不再直接返回 cells，这里用与 C++ 相同的几何定义，
    枚举所有合法 rotation/锚点，找到唯一与输出中心/角度一致的摆放。
    """
    if category not in BLOCK_CATEGORY_NAMES:
        raise ValueError(f"IDBS_Config 类别无效：{category}")
    category_index = BLOCK_CATEGORY_NAMES.index(category)
    wanted_angle = float(angle_deg)
    wanted_col = float(col)
    wanted_row = float(row)

    for rotation in range(_TETRIS_ROTATION_COUNTS[category_index]):
        for grid_row in range(_TETRIS_ROWS):
            for grid_col in range(_TETRIS_COLS):
                offsets = _TETRIS_BRICKS[category_index][rotation]
                cells = []
                valid = True
                for dx, dy in offsets:
                    internal_row = grid_row + dx
                    internal_col = grid_col + dy
                    if not (0 <= internal_row < _TETRIS_ROWS and 0 <= internal_col < _TETRIS_COLS):
                        valid = False
                        break
                    cells.append((_TETRIS_COLS - internal_col, internal_row + 1))
                if not valid:
                    continue
                center_col, center_row, center_angle = _calculate_center_col_row(
                    category_index,
                    rotation,
                    grid_row,
                    grid_col,
                )
                if (
                    math.isclose(center_col, wanted_col, abs_tol=1e-6)
                    and math.isclose(center_row, wanted_row, abs_tol=1e-6)
                    and math.isclose(center_angle, wanted_angle, abs_tol=1e-6)
                ):
                    return normalize_cells(cells, f"IDBS_Config {category} cells")

    raise ValueError(
        f"IDBS_Config 第 {category} 块无法还原四格几何："
        f"angle={angle_deg}, col={col}, row={row}"
    )


def parse_idbs_config_result(raw_result: bytes, expected_count: int) -> Tuple[List[dict], str]:
    """解析新版 IDBS/IDBS_Config 输出。

    新版输出格式：每个方块 5 个字段（名称,角度,x,y,方块索引），末尾满行数。
    x/y 为 C++ 的 0 基中心，这里按旧版 Python 调用习惯 +0.5 转成 1 基行列。
    """
    if not raw_result:
        raise RuntimeError("IDBS_Config 未返回规划结果")
    try:
        fields = raw_result.decode("gbk").split(",")
    except UnicodeDecodeError as exc:
        raise ValueError("IDBS_Config 返回值不是有效 GBK 文本") from exc

    expected_field_count = expected_count * IDBS_CONFIG_FIELDS_PER_BLOCK + 1
    if len(fields) != expected_field_count:
        raise ValueError(
            f"IDBS_Config 字段数错误：返回 {len(fields)}，"
            f"预期 {expected_field_count}"
        )

    layout = []
    for index in range(expected_count):
        offset = index * IDBS_CONFIG_FIELDS_PER_BLOCK
        category = normalize_category_name(fields[offset])
        if category not in BLOCK_CATEGORY_NAMES:
            raise ValueError(f"IDBS_Config 第 {index} 块类别无效：{category}")
        try:
            angle_deg = float(fields[offset + 1])
            center_x = float(fields[offset + 2])
            center_y = float(fields[offset + 3])
            block_index = int(fields[offset + 4])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"IDBS_Config 第 {index} 块数值字段无效") from exc
        if not all(math.isfinite(value) for value in (angle_deg, center_x, center_y)):
            raise ValueError(f"IDBS_Config 第 {index} 块包含非有限数值")

        center_col = center_x + 0.5
        center_row = center_y + 0.5
        if not 1.0 <= center_col <= 10.0 or not 1.0 <= center_row <= 14.0:
            raise ValueError(f"IDBS_Config 第 {index} 块中心超出 10×14 托盘范围")
        if block_index < 0 or block_index > 4:
            raise ValueError(f"IDBS_Config 第 {index} 块方块索引无效：{block_index}")

        cells = _cells_from_config_item(category, angle_deg, center_col, center_row)
        layout.append({
            "index": index,
            "category": category,
            "angle_deg": angle_deg,
            "col": center_col,
            "row": center_row,
            "cells": cells,
            "block_index": block_index,
        })

    build_support_graph(layout)
    return layout, fields[-1]


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


def _coerce_coord_shape(coords, rows, cols, label):
    """兼容嵌套 [rows][cols][2] 和扁平 [rows*cols*2] 两种传入方式。"""
    if (
        len(coords) == rows * cols * 2
        and all(not isinstance(value, (list, tuple)) for value in coords)
    ):
        return [
            [
                [float(coords[(row * cols + col) * 2]),
                 float(coords[(row * cols + col) * 2 + 1])]
                for col in range(cols)
            ]
            for row in range(rows)
        ]
    return coords


def _flatten_coords(coords):
    """把 [7][5][2] 或 [14][10][2] 坐标展平成 ctypes double 数组。"""
    flattened = []
    for row in coords:
        for point in row:
            flattened.append(float(point[0]))
            flattened.append(float(point[1]))
    return flattened


class AdvancedPlanner:
    def __init__(self, library_path: str):
        self.library_path = library_path

    def build_layout(
        self,
        cube_counts: Sequence[int],
        place_order: Sequence[int],
        block_coord: Sequence[Sequence[Sequence[float]]] | None = None,
        grid_coord: Sequence[Sequence[Sequence[float]]] | None = None,
        shooting_pose: Sequence[float] | None = None,
    ) -> Tuple[List[dict], str]:
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

        expected_count = sum(count_values)
        library = CDLL(self.library_path)

        if block_coord is not None or grid_coord is not None:
            if block_coord is None or grid_coord is None or shooting_pose is None:
                raise ValueError(
                    "使用 IDBS_Config 必须同时提供 block_coord/grid_coord/shooting_pose"
                )
            return self._build_with_config(
                library,
                count_values,
                order_values,
                block_coord,
                grid_coord,
                shooting_pose,
                expected_count,
            )

        # 旧版动态库仍保留 IDBSWithCells，直接走原来的 cells 协议。
        if hasattr(library, "IDBSWithCells"):
            library.IDBSWithCells.argtypes = [c_int] * 14
            library.IDBSWithCells.restype = c_char_p
            raw_result = library.IDBSWithCells(*count_values, *order_values)
            return parse_idbs_with_cells_result(raw_result, expected_count)

        # 新版动态库的 IDBS 接口（内置默认坐标）也返回带方块索引的新格式。
        if hasattr(library, "IDBS"):
            library.IDBS.argtypes = [POINTER(c_int), POINTER(c_int)]
            library.IDBS.restype = c_char_p
            counts_arr = (c_int * 7)(*count_values)
            orders_arr = (c_int * 7)(*order_values)
            raw_result = library.IDBS(counts_arr, orders_arr)
            return parse_idbs_config_result(raw_result, expected_count)

        raise AttributeError("动态库缺少 IDBS/IDBS_Config/IDBSWithCells")

    @staticmethod
    def _build_with_config(
        library,
        count_values,
        order_values,
        block_coord,
        grid_coord,
        shooting_pose,
        expected_count,
    ):
        if not hasattr(library, "IDBS_Config"):
            raise AttributeError("动态库缺少 IDBS_Config")

        if len(shooting_pose) == 6:
            # 兼容传入完整 6 维位姿，只取前 3 个位置分量。
            shooting_pose = shooting_pose[:3]
        if len(block_coord) != 7 or len(grid_coord) != 14 or len(shooting_pose) != 3:
            # 扁平数组也接受，先转换成嵌套格式再继续。
            if len(block_coord) == 70 and len(grid_coord) == 280:
                block_coord = _coerce_coord_shape(block_coord, 7, 5, "block_coord")
                grid_coord = _coerce_coord_shape(grid_coord, 14, 10, "grid_coord")
            else:
                raise ValueError(
                    "IDBS_Config 参数形状错误：需要 block_coord[7][5][2]、"
                    "grid_coord[14][10][2]、shooting_pose[3]"
                )
        if any(len(row) != 5 or len(point) != 2 for row in block_coord for point in row):
            raise ValueError("block_coord 必须为 [7][5][2]")
        if any(len(row) != 10 or len(point) != 2 for row in grid_coord for point in row):
            raise ValueError("grid_coord 必须为 [14][10][2]")

        library.IDBS_Config.argtypes = [
            POINTER(c_int),
            POINTER(c_int),
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
        ]
        library.IDBS_Config.restype = c_char_p

        counts_arr = (c_int * 7)(*count_values)
        orders_arr = (c_int * 7)(*order_values)
        block_arr = (c_double * 70)(*_flatten_coords(block_coord))
        grid_arr = (c_double * 280)(*_flatten_coords(grid_coord))
        pose_arr = (c_double * 3)(*[float(value) for value in shooting_pose])
        raw_result = library.IDBS_Config(
            counts_arr,
            orders_arr,
            block_arr,
            grid_arr,
            pose_arr,
        )
        return parse_idbs_config_result(raw_result, expected_count)
