"""V5 盘面 JSONL 到现场快速读取 NPZ 的转换与加载。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from image_process_lib.block_category import (
    BLOCK_CATEGORY_NAMES,
    normalize_category_name,
)


V5_BOARD_LIBRARY_FORMAT_VERSION = 1
V5_BOARD_TARGET_COUNT = 34
V5_TARGETS_PER_CATEGORY = 5
V5_REGION_MASK_ORDER = (1, 2, 4, 8, 3, 12, 5, 10, 15)
V5_L_CATEGORIES = frozenset(("L_blue", "L_yellow"))

# V5 搜索内部以逆时针方向为正，现有机械臂接口要求顺时针为正。
V5_CLOCKWISE_ANGLE_OUTPUT_MAP = {
    0: 0,
    90: -90,
    180: 180,
    -90: 90,
}

_REQUIRED_ARRAY_DTYPES = {
    "format_version": np.dtype(np.uint16),
    "board_global_id": np.dtype(np.int32),
    "board_layout_index": np.dtype(np.uint16),
    "board_rank": np.dtype(np.int32),
    "board_missing_category": np.dtype(np.uint8),
    "board_target_pid": np.dtype(np.int16),
    "board_region_counts": np.dtype(np.uint8),
    "placement_category": np.dtype(np.uint8),
    "placement_row": np.dtype(np.float32),
    "placement_col": np.dtype(np.float32),
    "placement_yaw_clockwise_deg": np.dtype(np.int16),
    "placement_region_mask": np.dtype(np.uint8),
    "placement_cells": np.dtype(np.uint8),
    "region_mask_order": np.dtype(np.uint8),
}


@dataclass(frozen=True)
class V5BoardLibrary:
    """已加载到内存的 V5 盘面库。"""

    format_version: int
    category_names: Tuple[str, ...]
    region_mask_order: np.ndarray
    board_global_id: np.ndarray
    board_layout_index: np.ndarray
    board_rank: np.ndarray
    board_missing_category: np.ndarray
    board_target_pid: np.ndarray
    board_region_counts: np.ndarray
    placement_category: np.ndarray
    placement_row: np.ndarray
    placement_col: np.ndarray
    placement_yaw_clockwise_deg: np.ndarray
    placement_region_mask: np.ndarray
    placement_cells: np.ndarray
    source_path: Path | None = None
    source_sha256: str = ""

    @property
    def board_count(self) -> int:
        return int(self.board_global_id.shape[0])

    @property
    def placement_count(self) -> int:
        return int(self.placement_category.shape[0])

    def board_id(self, board_index: int) -> str:
        """返回不依赖文件名的稳定盘面编号。"""
        index = int(board_index)
        if index < 0 or index >= self.board_count:
            raise IndexError(f"盘面索引越界: {index}")
        return (
            f"v5_g{int(self.board_global_id[index]):05d}"
            f"_l{int(self.board_layout_index[index]):02d}"
        )

    def missing_category_name(self, board_index: int) -> str:
        category_index = int(self.board_missing_category[int(board_index)])
        return self.category_names[category_index]


@dataclass(frozen=True)
class V5BoardLibraryBuildSummary:
    """转换后用于命令行报告的统计。"""

    output_path: Path
    board_count: int
    signature_count: int
    placement_count: int
    input_file_count: int
    output_bytes: int


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        with path.open("r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 JSON 文件 {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 文件顶层必须是对象: {path}")
    return payload


def _as_bounded_integer(
    value: object,
    label: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label}必须是整数")
    try:
        integer = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}必须是整数") from exc
    if not np.isfinite(numeric) or numeric != integer:
        raise ValueError(f"{label}必须是整数")
    if integer < minimum or integer > maximum:
        raise ValueError(f"{label}超出范围 [{minimum}, {maximum}]: {integer}")
    return integer


def _l_placement_reference(
    cells: np.ndarray,
    label: str,
) -> Tuple[float, float]:
    """由 L 方块占格恢复机械臂实际使用的长边中间格。"""
    # V5 cells 的每一项顺序为 [col, row]，返回值统一为 (row, col)。
    cell_tuples = tuple((int(cell[0]), int(cell[1])) for cell in cells)
    if len(set(cell_tuples)) != 4:
        raise ValueError(f"{label}的 L 方块 cells 必须包含 4 个不同格子")

    candidates = []
    rows_to_cols: Dict[int, list] = {}
    cols_to_rows: Dict[int, list] = {}
    for col, row in cell_tuples:
        rows_to_cols.setdefault(row, []).append(col)
        cols_to_rows.setdefault(col, []).append(row)

    for row, cols in rows_to_cols.items():
        sorted_cols = sorted(cols)
        if len(sorted_cols) == 3 and sorted_cols == list(
            range(sorted_cols[0], sorted_cols[0] + 3)
        ):
            candidates.append((float(row), float(sorted_cols[1])))
    for col, rows in cols_to_rows.items():
        sorted_rows = sorted(rows)
        if len(sorted_rows) == 3 and sorted_rows == list(
            range(sorted_rows[0], sorted_rows[0] + 3)
        ):
            candidates.append((float(sorted_rows[1]), float(col)))

    if len(candidates) != 1:
        raise ValueError(
            f"{label}无法从 cells 唯一确定 L 方块三格长边的中间格"
        )
    return candidates[0]


def _runtime_placement_reference(
    category: str,
    catalog_row: float,
    catalog_col: float,
    cells: np.ndarray,
    label: str,
) -> Tuple[float, float]:
    """返回实际摆放参考点；L 方块不能使用包围盒几何中心。"""
    if category in V5_L_CATEGORIES:
        return _l_placement_reference(cells, label)
    return float(catalog_row), float(catalog_col)


def _catalog_signature_order(
    catalog: Mapping[str, object],
) -> Tuple[Tuple[str, int], ...]:
    raw_order = catalog.get("spatial_signature_order")
    if not isinstance(raw_order, list):
        raise ValueError("placement catalog 缺少 spatial_signature_order 列表")

    parsed = []
    for index, item in enumerate(raw_order):
        if not isinstance(item, dict):
            raise ValueError(f"spatial_signature_order[{index}] 必须是对象")
        category = normalize_category_name(str(item.get("category", "")))
        try:
            region_mask = int(item["region_mask"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"spatial_signature_order[{index}] 缺少合法 region_mask"
            ) from exc
        parsed.append((category, region_mask))

    expected = {
        (category, mask)
        for category in BLOCK_CATEGORY_NAMES
        for mask in V5_REGION_MASK_ORDER
    }
    if len(parsed) != len(expected) or set(parsed) != expected:
        raise ValueError("placement catalog 的空间签名顺序与 V5 格式不一致")
    if len(set(parsed)) != len(parsed):
        raise ValueError("placement catalog 的空间签名顺序包含重复项")
    return tuple(parsed)


def _parse_catalog(catalog_path: Path) -> Tuple[Dict[str, np.ndarray], Tuple[Tuple[str, int], ...]]:
    catalog = _read_json(catalog_path)
    if catalog.get("version") != 5:
        raise ValueError(
            f"placement catalog 版本必须为 5，实际为 {catalog.get('version')!r}"
        )
    signature_order = _catalog_signature_order(catalog)

    raw_placements = catalog.get("placements")
    if not isinstance(raw_placements, list) or not raw_placements:
        raise ValueError("placement catalog 必须包含非空 placements 列表")
    placement_count = len(raw_placements)
    if placement_count > np.iinfo(np.int16).max:
        raise ValueError("placement 数量超出 int16 可表示范围")

    category_to_index = {
        category: index for index, category in enumerate(BLOCK_CATEGORY_NAMES)
    }
    placement_category = np.empty(placement_count, dtype=np.uint8)
    placement_row = np.empty(placement_count, dtype=np.float32)
    placement_col = np.empty(placement_count, dtype=np.float32)
    placement_yaw = np.empty(placement_count, dtype=np.int16)
    placement_region_mask = np.empty(placement_count, dtype=np.uint8)
    placement_cells = np.empty((placement_count, 4, 2), dtype=np.uint8)
    seen_pids = set()

    for source_index, item in enumerate(raw_placements):
        if not isinstance(item, dict):
            raise ValueError(f"placements[{source_index}] 必须是对象")
        pid = _as_bounded_integer(
            item.get("id"),
            f"placements[{source_index}].id",
            0,
            placement_count - 1,
        )
        if pid in seen_pids:
            raise ValueError(f"placement ID 重复: {pid}")
        seen_pids.add(pid)

        category = normalize_category_name(str(item.get("category", "")))
        if category not in category_to_index:
            raise ValueError(f"placement {pid} 包含未知类别: {category!r}")
        try:
            row = float(item["row"])
            col = float(item["col"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"placement {pid} 的 row/col 无效") from exc
        if not np.all(np.isfinite([row, col])):
            raise ValueError(f"placement {pid} 的 row/col 必须是有限数")

        internal_angle = _as_bounded_integer(
            item.get("angle_deg_internal"),
            f"placement {pid} angle_deg_internal",
            -180,
            180,
        )
        if internal_angle not in V5_CLOCKWISE_ANGLE_OUTPUT_MAP:
            raise ValueError(
                f"placement {pid} 包含未支持的内部角度: {internal_angle}"
            )
        region_mask = _as_bounded_integer(
            item.get("region_mask"),
            f"placement {pid} region_mask",
            0,
            255,
        )
        if region_mask not in V5_REGION_MASK_ORDER:
            raise ValueError(f"placement {pid} 包含未知区域掩码: {region_mask}")

        raw_cells = np.asarray(item.get("cells"))
        if raw_cells.shape != (4, 2):
            raise ValueError(f"placement {pid} cells 必须是 [4,2]")
        try:
            numeric_cells = raw_cells.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"placement {pid} cells 必须是整数") from exc
        if (
            not np.all(np.isfinite(numeric_cells))
            or not np.all(numeric_cells == np.floor(numeric_cells))
            or np.any(numeric_cells < 0)
            or np.any(numeric_cells > np.iinfo(np.uint8).max)
        ):
            raise ValueError(f"placement {pid} cells 必须是 uint8 范围内的整数")

        numeric_cells_uint8 = numeric_cells.astype(np.uint8)
        runtime_row, runtime_col = _runtime_placement_reference(
            category,
            row,
            col,
            numeric_cells_uint8,
            f"placement {pid}",
        )

        placement_category[pid] = category_to_index[category]
        # catalog 的 row/col 是包围盒中心；L 方块必须改用长边中间格，
        # 才与视觉抓取参考点和固定 task_layout.yaml 保持一致。
        placement_row[pid] = runtime_row
        placement_col[pid] = runtime_col
        placement_yaw[pid] = V5_CLOCKWISE_ANGLE_OUTPUT_MAP[internal_angle]
        placement_region_mask[pid] = region_mask
        placement_cells[pid] = numeric_cells_uint8

    if seen_pids != set(range(placement_count)):
        raise ValueError("placement ID 必须从 0 开始连续，才能直接作为查表下标")

    arrays = {
        "placement_category": placement_category,
        "placement_row": placement_row,
        "placement_col": placement_col,
        "placement_yaw_clockwise_deg": placement_yaw,
        "placement_region_mask": placement_region_mask,
        "placement_cells": placement_cells,
    }
    return arrays, signature_order


def _read_layout_records(layout_paths: Iterable[Path]) -> Tuple[list, int]:
    records = []
    input_file_count = 0
    for path in sorted(Path(item) for item in layout_paths):
        input_file_count += 1
        try:
            input_file = path.open("r", encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"无法读取盘面文件 {path}: {exc}") from exc
        with input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"盘面文件 {path} 第 {line_number} 行 JSON 无效: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(f"盘面文件 {path} 第 {line_number} 行必须是对象")
                records.append((record, path, line_number))
    if input_file_count == 0:
        raise ValueError("未找到 V5 盘面 JSONL 文件")
    if not records:
        raise ValueError("V5 盘面 JSONL 文件中没有有效记录")
    return records, input_file_count


def _record_integer(record: Mapping[str, object], key: str, label: str, maximum: int) -> int:
    if key not in record:
        raise ValueError(f"{label} 缺少 {key}")
    return _as_bounded_integer(record[key], f"{label}.{key}", 0, maximum)


def _build_board_arrays(
    records_with_origin: Sequence[tuple],
    placement_arrays: Mapping[str, np.ndarray],
    signature_order: Sequence[Tuple[str, int]],
) -> Tuple[Dict[str, np.ndarray], int]:
    category_to_index = {
        category: index for index, category in enumerate(BLOCK_CATEGORY_NAMES)
    }
    mask_to_index = {mask: index for index, mask in enumerate(V5_REGION_MASK_ORDER)}
    placement_category = placement_arrays["placement_category"]
    placement_region_mask = placement_arrays["placement_region_mask"]
    placement_count = int(placement_category.shape[0])

    normalized_records = []
    seen_board_keys = set()
    # global_id 对应 V5 的现场分布 signature，同一 signature 可以有多张 layout。
    signature_global_ids = set()
    for record, path, line_number in records_with_origin:
        label = f"{path} 第 {line_number} 行"
        if record.get("version") != 5:
            raise ValueError(f"{label}的 version 必须为 5")
        global_id = _record_integer(record, "global_id", label, np.iinfo(np.int32).max)
        layout_index = _record_integer(
            record, "layout_index", label, np.iinfo(np.uint16).max
        )
        rank = _record_integer(record, "rank", label, np.iinfo(np.int32).max)
        board_key = (global_id, layout_index)
        if board_key in seen_board_keys:
            raise ValueError(
                f"盘面编号重复: global_id={global_id}, layout_index={layout_index}"
            )
        seen_board_keys.add(board_key)

        missing_category = normalize_category_name(str(record.get("missing_category", "")))
        if missing_category not in category_to_index:
            raise ValueError(f"{label}包含未知 missing_category: {missing_category!r}")

        raw_ids = record.get("ids")
        if not isinstance(raw_ids, list) or len(raw_ids) != V5_BOARD_TARGET_COUNT:
            raise ValueError(f"{label}必须包含正好 34 个 placement ID")
        pids = [
            _as_bounded_integer(value, f"{label}.ids[{index}]", 0, placement_count - 1)
            for index, value in enumerate(raw_ids)
        ]
        if len(set(pids)) != V5_BOARD_TARGET_COUNT:
            raise ValueError(f"{label}的 placement ID 必须互不重复")

        ids_by_category = [[] for _ in BLOCK_CATEGORY_NAMES]
        region_counts = np.zeros(
            (len(BLOCK_CATEGORY_NAMES), len(V5_REGION_MASK_ORDER)), dtype=np.uint8
        )
        for pid in pids:
            category_index = int(placement_category[pid])
            ids_by_category[category_index].append(pid)
            mask = int(placement_region_mask[pid])
            region_counts[category_index, mask_to_index[mask]] += 1

        missing_index = category_to_index[missing_category]
        for category_index, category_pids in enumerate(ids_by_category):
            expected_count = 4 if category_index == missing_index else 5
            if len(category_pids) != expected_count:
                category = BLOCK_CATEGORY_NAMES[category_index]
                raise ValueError(
                    f"{label}中 {category} 数量为 {len(category_pids)}，"
                    f"应为 {expected_count}"
                )
            category_pids.sort()

        raw_signature = record.get("spatial_signature_vector")
        if not isinstance(raw_signature, list) or len(raw_signature) != len(signature_order):
            raise ValueError(f"{label}的 spatial_signature_vector 长度错误")
        signature_values = []
        signature_counts = {}
        for index, ((category, mask), value) in enumerate(zip(signature_order, raw_signature)):
            count = _as_bounded_integer(
                value,
                f"{label}.spatial_signature_vector[{index}]",
                0,
                V5_TARGETS_PER_CATEGORY,
            )
            signature_values.append(count)
            signature_counts[(category, mask)] = count
        for category_index, category in enumerate(BLOCK_CATEGORY_NAMES):
            for mask_index, mask in enumerate(V5_REGION_MASK_ORDER):
                if int(region_counts[category_index, mask_index]) != signature_counts[
                    (category, mask)
                ]:
                    raise ValueError(f"{label}的区域计数与 placement 不一致")

        signature_global_ids.add(global_id)
        normalized_records.append(
            (
                rank,
                global_id,
                layout_index,
                missing_index,
                ids_by_category,
                region_counts,
            )
        )

    # 库内存储顺序固定，与 JSONL 的拼接顺序无关。
    normalized_records.sort(key=lambda item: (item[0], item[1], item[2]))
    board_count = len(normalized_records)
    board_global_id = np.empty(board_count, dtype=np.int32)
    board_layout_index = np.empty(board_count, dtype=np.uint16)
    board_rank = np.empty(board_count, dtype=np.int32)
    board_missing_category = np.empty(board_count, dtype=np.uint8)
    board_target_pid = np.full(
        (board_count, len(BLOCK_CATEGORY_NAMES), V5_TARGETS_PER_CATEGORY),
        -1,
        dtype=np.int16,
    )
    board_region_counts = np.empty(
        (board_count, len(BLOCK_CATEGORY_NAMES), len(V5_REGION_MASK_ORDER)),
        dtype=np.uint8,
    )

    for board_index, item in enumerate(normalized_records):
        rank, global_id, layout_index, missing_index, ids_by_category, region_counts = item
        board_global_id[board_index] = global_id
        board_layout_index[board_index] = layout_index
        board_rank[board_index] = rank
        board_missing_category[board_index] = missing_index
        board_region_counts[board_index] = region_counts
        for category_index, pids in enumerate(ids_by_category):
            board_target_pid[board_index, category_index, : len(pids)] = pids

    arrays = {
        "board_global_id": board_global_id,
        "board_layout_index": board_layout_index,
        "board_rank": board_rank,
        "board_missing_category": board_missing_category,
        "board_target_pid": board_target_pid,
        "board_region_counts": board_region_counts,
    }
    return arrays, len(signature_global_ids)


def convert_v5_board_library(
    input_directory: Path | str,
    output_path: Path | str,
    layout_paths: Sequence[Path | str] | None = None,
) -> V5BoardLibraryBuildSummary:
    """校验 V5 JSONL 并原子生成未压缩 NPZ。"""
    input_directory = Path(input_directory)
    output_path = Path(output_path)
    catalog_path = input_directory / "placement_catalog.json"
    if layout_paths is None:
        resolved_layout_paths = sorted(input_directory.glob("library_*.jsonl"))
    else:
        resolved_layout_paths = [Path(path) for path in layout_paths]

    placement_arrays, signature_order = _parse_catalog(catalog_path)
    records, input_file_count = _read_layout_records(resolved_layout_paths)
    board_arrays, signature_count = _build_board_arrays(
        records,
        placement_arrays,
        signature_order,
    )
    arrays = {
        "format_version": np.asarray(V5_BOARD_LIBRARY_FORMAT_VERSION, dtype=np.uint16),
        "category_names": np.asarray(BLOCK_CATEGORY_NAMES, dtype=np.str_),
        "region_mask_order": np.asarray(V5_REGION_MASK_ORDER, dtype=np.uint8),
        **board_arrays,
        **placement_arrays,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            # np.savez 而非 savez_compressed，换取现场更快的加载。
            np.savez(temporary_file, **arrays)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        # NamedTemporaryFile 默认是 0600，部署配置文件需要可被 ROS 运行用户读取。
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

    return V5BoardLibraryBuildSummary(
        output_path=output_path,
        board_count=int(board_arrays["board_global_id"].shape[0]),
        signature_count=signature_count,
        placement_count=int(placement_arrays["placement_category"].shape[0]),
        input_file_count=input_file_count,
        output_bytes=output_path.stat().st_size,
    )


def _require_array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    expected_dtype: np.dtype | None = None,
) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"V5 盘面库缺少数组: {name}")
    array = arrays[name]
    if expected_dtype is not None and array.dtype != expected_dtype:
        raise ValueError(
            f"V5 盘面库数组 {name} 类型应为 {expected_dtype}，"
            f"实际为 {array.dtype}"
        )
    return array


def _validate_loaded_arrays(arrays: Mapping[str, np.ndarray]) -> V5BoardLibrary:
    for name, dtype in _REQUIRED_ARRAY_DTYPES.items():
        _require_array(arrays, name, dtype)
    category_names_array = _require_array(arrays, "category_names")
    if category_names_array.dtype.kind != "U":
        raise ValueError("V5 盘面库 category_names 必须是 Unicode 数组")

    format_version_array = arrays["format_version"]
    if format_version_array.shape not in ((), (1,)):
        raise ValueError("V5 盘面库 format_version 必须是标量")
    format_version = int(format_version_array.reshape(-1)[0])
    if format_version != V5_BOARD_LIBRARY_FORMAT_VERSION:
        raise ValueError(
            f"V5 盘面库版本不支持: {format_version}，"
            f"当前仅支持 {V5_BOARD_LIBRARY_FORMAT_VERSION}"
        )

    category_names = tuple(str(value) for value in category_names_array.tolist())
    if category_names != tuple(BLOCK_CATEGORY_NAMES):
        raise ValueError(
            "V5 盘面库类别顺序与 BLOCK_CATEGORY_NAMES 不一致"
        )
    region_mask_order = arrays["region_mask_order"]
    if region_mask_order.shape != (len(V5_REGION_MASK_ORDER),) or tuple(
        int(value) for value in region_mask_order
    ) != V5_REGION_MASK_ORDER:
        raise ValueError("V5 盘面库 region_mask_order 不一致")

    board_count = int(arrays["board_global_id"].shape[0])
    placement_count = int(arrays["placement_category"].shape[0])
    if board_count <= 0 or placement_count <= 0:
        raise ValueError("V5 盘面库不能为空")
    one_dimensional_board_arrays = (
        "board_global_id",
        "board_layout_index",
        "board_rank",
        "board_missing_category",
    )
    for name in one_dimensional_board_arrays:
        if arrays[name].shape != (board_count,):
            raise ValueError(f"V5 盘面库数组 {name} 形状错误")
    one_dimensional_placement_arrays = (
        "placement_category",
        "placement_row",
        "placement_col",
        "placement_yaw_clockwise_deg",
        "placement_region_mask",
    )
    for name in one_dimensional_placement_arrays:
        if arrays[name].shape != (placement_count,):
            raise ValueError(f"V5 盘面库数组 {name} 形状错误")
    expected_pid_shape = (
        board_count,
        len(BLOCK_CATEGORY_NAMES),
        V5_TARGETS_PER_CATEGORY,
    )
    if arrays["board_target_pid"].shape != expected_pid_shape:
        raise ValueError("V5 盘面库 board_target_pid 形状错误")
    if arrays["board_region_counts"].shape != (
        board_count,
        len(BLOCK_CATEGORY_NAMES),
        len(V5_REGION_MASK_ORDER),
    ):
        raise ValueError("V5 盘面库 board_region_counts 形状错误")
    if arrays["placement_cells"].shape != (placement_count, 4, 2):
        raise ValueError("V5 盘面库 placement_cells 形状错误")

    if not np.all(np.isfinite(arrays["placement_row"])) or not np.all(
        np.isfinite(arrays["placement_col"])
    ):
        raise ValueError("V5 盘面库 placement row/col 包含非有限数")
    if np.any(arrays["placement_category"] >= len(BLOCK_CATEGORY_NAMES)):
        raise ValueError("V5 盘面库 placement_category 越界")
    if np.any(arrays["board_missing_category"] >= len(BLOCK_CATEGORY_NAMES)):
        raise ValueError("V5 盘面库 board_missing_category 越界")
    valid_masks = np.isin(
        arrays["placement_region_mask"], np.asarray(V5_REGION_MASK_ORDER, dtype=np.uint8)
    )
    if not np.all(valid_masks):
        raise ValueError("V5 盘面库 placement_region_mask 包含未知值")

    # 旧版 NPZ 把 L 方块包围盒中心误当作摆放点，会产生约半格（约 10 mm）
    # 的系统偏移。加载阶段显式拒绝旧数据，避免代码更新后仍静默执行坏盘面。
    for pid in range(placement_count):
        category = category_names[int(arrays["placement_category"][pid])]
        if category not in V5_L_CATEGORIES:
            continue
        expected_row, expected_col = _l_placement_reference(
            arrays["placement_cells"][pid],
            f"V5 盘面库 placement {pid}",
        )
        actual_row = float(arrays["placement_row"][pid])
        actual_col = float(arrays["placement_col"][pid])
        if not np.allclose(
            (actual_row, actual_col),
            (expected_row, expected_col),
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                f"V5 盘面库 placement {pid} 的 L 方块摆放参考点错误："
                f"实际=({actual_row}, {actual_col})，"
                f"应为长边中间格=({expected_row}, {expected_col})；"
                "请重新运行转换V5盘面库.py"
            )

    pids = arrays["board_target_pid"]
    if np.any(pids < -1) or np.any(pids >= placement_count):
        raise ValueError("V5 盘面库 board_target_pid 越界")
    if np.any(np.sum(pids >= 0, axis=(1, 2)) != V5_BOARD_TARGET_COUNT):
        raise ValueError("V5 盘面库每张盘面必须正好包含 34 个目标")

    mask_to_index = {mask: index for index, mask in enumerate(V5_REGION_MASK_ORDER)}
    stable_keys = []
    board_ids = []
    for board_index in range(board_count):
        missing_index = int(arrays["board_missing_category"][board_index])
        flat_valid_pids = pids[board_index][pids[board_index] >= 0]
        if len(np.unique(flat_valid_pids)) != V5_BOARD_TARGET_COUNT:
            raise ValueError(f"V5 盘面库第 {board_index} 张盘面包含重复 PID")
        computed_counts = np.zeros(
            (len(BLOCK_CATEGORY_NAMES), len(V5_REGION_MASK_ORDER)), dtype=np.uint8
        )
        for category_index in range(len(BLOCK_CATEGORY_NAMES)):
            category_pids = pids[board_index, category_index]
            valid_category_pids = category_pids[category_pids >= 0]
            expected_count = 4 if category_index == missing_index else 5
            if valid_category_pids.shape[0] != expected_count:
                raise ValueError(
                    f"V5 盘面库第 {board_index} 张盘面的类别数量错误"
                )
            if np.any(arrays["placement_category"][valid_category_pids] != category_index):
                raise ValueError(
                    f"V5 盘面库第 {board_index} 张盘面的 PID 类别错误"
                )
            for pid in valid_category_pids:
                mask = int(arrays["placement_region_mask"][pid])
                computed_counts[category_index, mask_to_index[mask]] += 1
        if not np.array_equal(computed_counts, arrays["board_region_counts"][board_index]):
            raise ValueError(
                f"V5 盘面库第 {board_index} 张盘面的区域计数不一致"
            )
        stable_keys.append(
            (
                int(arrays["board_rank"][board_index]),
                int(arrays["board_global_id"][board_index]),
                int(arrays["board_layout_index"][board_index]),
            )
        )
        board_ids.append(
            (
                int(arrays["board_global_id"][board_index]),
                int(arrays["board_layout_index"][board_index]),
            )
        )
    if stable_keys != sorted(stable_keys):
        raise ValueError("V5 盘面库未按 (rank, global_id, layout_index) 稳定排序")
    if len(set(board_ids)) != len(board_ids):
        raise ValueError("V5 盘面库包含重复盘面编号")

    immutable_arrays = {}
    for name, array in arrays.items():
        copied = np.array(array, copy=True)
        copied.setflags(write=False)
        immutable_arrays[name] = copied

    return V5BoardLibrary(
        format_version=format_version,
        category_names=category_names,
        region_mask_order=immutable_arrays["region_mask_order"],
        board_global_id=immutable_arrays["board_global_id"],
        board_layout_index=immutable_arrays["board_layout_index"],
        board_rank=immutable_arrays["board_rank"],
        board_missing_category=immutable_arrays["board_missing_category"],
        board_target_pid=immutable_arrays["board_target_pid"],
        board_region_counts=immutable_arrays["board_region_counts"],
        placement_category=immutable_arrays["placement_category"],
        placement_row=immutable_arrays["placement_row"],
        placement_col=immutable_arrays["placement_col"],
        placement_yaw_clockwise_deg=immutable_arrays[
            "placement_yaw_clockwise_deg"
        ],
        placement_region_mask=immutable_arrays["placement_region_mask"],
        placement_cells=immutable_arrays["placement_cells"],
    )


def load_v5_board_library(npz_path: Path | str) -> V5BoardLibrary:
    """一次性加载并完整校验 V5 盘面库，严禁 pickle。"""
    npz_path = Path(npz_path).expanduser()
    try:
        digest = hashlib.sha256()
        with npz_path.open("rb") as input_file:
            while True:
                chunk = input_file.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            input_file.seek(0)
            with np.load(input_file, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"无法加载 V5 盘面库 {npz_path}: {exc}") from exc
    validated = _validate_loaded_arrays(arrays)
    return V5BoardLibrary(
        **{
            field_name: getattr(validated, field_name)
            for field_name in validated.__dataclass_fields__
            if field_name not in ("source_path", "source_sha256")
        },
        source_path=npz_path.resolve(),
        source_sha256=digest.hexdigest(),
    )
