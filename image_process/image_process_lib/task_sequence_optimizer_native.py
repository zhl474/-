"""C++ 任务顺序优化器的严格 ctypes 包装。"""

from ctypes import (
    CDLL,
    POINTER,
    Structure,
    c_char,
    c_double,
    c_int32,
    c_size_t,
    c_uint32,
    c_uint64,
    create_string_buffer,
)
from ctypes.util import find_library
from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path
import time
from typing import Optional, Sequence, Tuple

import numpy as np


NATIVE_ABI_VERSION = 1
CATEGORY_COUNT = 7
MAX_SOURCES_PER_CATEGORY = 5
MAX_TARGET_COUNT = 63
MAX_BEAM_WIDTH = 5000
MAX_INT32 = (1 << 31) - 1
LIBRARY_BASENAME = "libtask_sequence_optimizer_native.so"
LIBRARY_ENVIRONMENT_VARIABLE = "TASK_SEQUENCE_OPTIMIZER_NATIVE_LIB"


class _NativeStatisticsV1(Structure):
    _fields_ = [
        ("expanded_parent_count", c_uint64),
        ("generated_child_count", c_uint64),
        ("peak_retained_node_count", c_uint64),
        ("final_candidate_count", c_uint64),
        ("beam_search_seconds", c_double),
        ("source_assignment_seconds", c_double),
    ]


@dataclass(frozen=True)
class NativeCandidate:
    target_sequence: Tuple[int, ...]
    source_sequence: Tuple[int, ...]
    prefix_score: float
    assignment_score: float


@dataclass(frozen=True)
class NativeSearchStatistics:
    expanded_parent_count: int
    generated_child_count: int
    peak_retained_node_count: int
    final_candidate_count: int
    beam_search_seconds: float
    source_assignment_seconds: float
    native_call_seconds: float
    candidate_conversion_seconds: float


@dataclass(frozen=True)
class NativeSearchResult:
    candidates: Tuple[NativeCandidate, ...]
    statistics: NativeSearchStatistics
    library_path: Path


def _candidate_library_paths(explicit_path: Optional[os.PathLike] = None):
    if explicit_path is not None:
        yield Path(explicit_path).expanduser()
        return
    environment_path = os.environ.get(LIBRARY_ENVIRONMENT_VARIABLE, "").strip()
    if environment_path:
        yield Path(environment_path).expanduser()

    module_dir = Path(__file__).resolve().parent
    yield module_dir / LIBRARY_BASENAME
    source_root = module_dir.parents[1]
    workspace_root = source_root.parent
    yield workspace_root / "devel" / "lib" / LIBRARY_BASENAME
    yield workspace_root / "install" / "lib" / LIBRARY_BASENAME
    for directory in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if directory:
            yield Path(directory) / LIBRARY_BASENAME
    system_path = find_library("task_sequence_optimizer_native")
    if system_path:
        yield Path(system_path)


def resolve_native_library_path(
    explicit_path: Optional[os.PathLike] = None,
) -> Path:
    """按显式路径、环境变量、源码旁和 catkin 输出顺序寻找动态库。"""
    checked = []
    for candidate in _candidate_library_paths(explicit_path):
        candidate_text = str(candidate)
        checked.append(candidate_text)
        if candidate.is_file():
            return candidate.resolve()
        # find_library 可能直接返回由系统加载器解释的 soname。
        if candidate.parent == Path(".") and os.path.sep not in candidate_text:
            return candidate
    raise RuntimeError(
        "找不到 C++ 任务顺序优化动态库；已检查：" + "，".join(checked)
    )


@lru_cache(maxsize=4)
def _load_native_library(resolved_path_text: str):
    try:
        library = CDLL(resolved_path_text)
    except OSError as exc:
        raise RuntimeError(
            f"无法加载 C++ 任务顺序优化动态库：{resolved_path_text}：{exc}"
        ) from exc
    library.task_sequence_optimizer_abi_version.argtypes = []
    library.task_sequence_optimizer_abi_version.restype = c_uint32
    actual_version = int(library.task_sequence_optimizer_abi_version())
    if actual_version != NATIVE_ABI_VERSION:
        raise RuntimeError(
            "C++ 任务顺序优化器 ABI 版本不一致："
            f"Python={NATIVE_ABI_VERSION}，动态库={actual_version}"
        )
    library.task_sequence_optimizer_search_v1.argtypes = [
        c_uint32,
        c_int32,
        c_int32,
        c_int32,
        c_uint64,
        POINTER(c_uint64),
        POINTER(c_int32),
        POINTER(c_int32),
        POINTER(c_int32),
        POINTER(c_int32),
        POINTER(c_double),
        c_int32,
        POINTER(c_int32),
        POINTER(c_int32),
        POINTER(c_int32),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(_NativeStatisticsV1),
        POINTER(c_char),
        c_size_t,
    ]
    library.task_sequence_optimizer_search_v1.restype = c_int32
    return library


def clear_native_library_cache() -> None:
    """仅供测试切换动态库路径时清空 CDLL 缓存。"""
    _load_native_library.cache_clear()


def _as_contiguous_array(values, dtype, shape, label):
    try:
        array = np.ascontiguousarray(values, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 无法转换为连续数组") from exc
    if array.shape != shape:
        raise ValueError(f"{label} 形状必须为 {shape}，实际为 {array.shape}")
    return array


def _validate_positive_integer(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label}必须是正整数")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是正整数") from exc
    if not math.isfinite(number) or not number.is_integer() or number <= 0.0:
        raise ValueError(f"{label}必须是正整数")
    return int(number)


def _strict_integer(value, label, minimum, maximum):
    if isinstance(value, bool):
        raise ValueError(f"{label}必须是整数")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}必须是整数") from exc
    if integer != value or not minimum <= integer <= maximum:
        raise ValueError(
            f"{label}必须是 [{minimum}, {maximum}] 范围内的整数"
        )
    return integer


def run_native_task_sequence_search(
    *,
    first_layer_mask: int,
    unlock_masks: Sequence[int],
    target_categories: Sequence[int],
    sources_by_category: Sequence[Sequence[int]],
    source_ids: Sequence[int],
    edge_cost_seconds,
    beam_width: int,
    library_path: Optional[os.PathLike] = None,
) -> NativeSearchResult:
    """调用 C++ 完成 Beam 和所有最终候选的 source assignment。"""
    target_count = len(target_categories)
    source_count = len(source_ids)
    beam_width = _validate_positive_integer(beam_width, "beam_width")
    if beam_width > MAX_BEAM_WIDTH:
        raise ValueError("beam_width 必须位于 [1, 5000]")
    if not 1 <= target_count <= MAX_TARGET_COUNT:
        raise ValueError("target 数量必须位于 [1, 63]")
    if source_count < target_count:
        raise ValueError("source 数量必须大于等于 target 数量")
    if len(sources_by_category) != CATEGORY_COUNT:
        raise ValueError("sources_by_category 必须包含 7 个类别")
    first_layer_mask = _strict_integer(
        first_layer_mask,
        "first_layer_mask",
        0,
        (1 << target_count) - 1,
    )

    if len(unlock_masks) != target_count:
        raise ValueError(f"unlock_masks 形状必须为 ({target_count},)")
    validated_unlock_masks = [
        _strict_integer(
            value,
            f"unlock_masks[{index}]",
            0,
            (1 << target_count) - 1,
        )
        for index, value in enumerate(unlock_masks)
    ]

    unlock_array = _as_contiguous_array(
        validated_unlock_masks,
        np.uint64,
        (target_count,),
        "unlock_masks",
    )
    if len(target_categories) != target_count:
        raise ValueError(f"target_categories 形状必须为 ({target_count},)")
    validated_target_categories = [
        _strict_integer(
            value,
            f"target_categories[{index}]",
            0,
            CATEGORY_COUNT - 1,
        )
        for index, value in enumerate(target_categories)
    ]
    target_category_array = _as_contiguous_array(
        validated_target_categories,
        np.int32,
        (target_count,),
        "target_categories",
    )
    if np.any(target_category_array < 0) or np.any(
        target_category_array >= CATEGORY_COUNT
    ):
        raise ValueError("target_categories 包含越界类别")
    validated_source_ids = [
        _strict_integer(
            value,
            f"source_ids[{index}]",
            0,
            MAX_INT32,
        )
        for index, value in enumerate(source_ids)
    ]
    source_id_array = _as_contiguous_array(
        validated_source_ids,
        np.int32,
        (source_count,),
        "source_ids",
    )
    if len(set(int(value) for value in source_id_array)) != source_count:
        raise ValueError("source_ids 不能重复")

    category_source_counts = np.zeros(CATEGORY_COUNT, dtype=np.int32)
    category_sources = np.full(
        (CATEGORY_COUNT, MAX_SOURCES_PER_CATEGORY),
        -1,
        dtype=np.int32,
    )
    flattened_sources = []
    for category, category_values in enumerate(sources_by_category):
        values = []
        for local_index, value in enumerate(category_values):
            values.append(_strict_integer(
                value,
                f"类别 {category} source 索引 {local_index}",
                0,
                source_count - 1,
            ))
        if len(values) > MAX_SOURCES_PER_CATEGORY:
            raise ValueError(f"类别 {category} source 数超过 5")
        category_source_counts[category] = len(values)
        if values:
            category_sources[category, :len(values)] = values
            flattened_sources.extend(values)
    if sorted(flattened_sources) != list(range(source_count)):
        raise ValueError("类别 source 索引必须无重复地覆盖全部实体")
    target_counts = np.bincount(
        target_category_array,
        minlength=CATEGORY_COUNT,
    )
    if np.any(target_counts > category_source_counts):
        raise ValueError("存在 target 数超过 source 数的类别")

    edge_cost_array = _as_contiguous_array(
        edge_cost_seconds,
        np.float64,
        (target_count + 1, source_count, target_count),
        "edge_cost_seconds",
    )
    for target, category in enumerate(target_category_array):
        category_sources_for_target = category_sources[
            int(category),
            :category_source_counts[int(category)],
        ]
        if category_sources_for_target.size == 0:
            raise ValueError(f"目标 {target} 所属类别没有 source")
        matching_costs = edge_cost_array[:, category_sources_for_target, target]
        if not np.all(np.isfinite(matching_costs)) or np.any(matching_costs < 0.0):
            raise ValueError("匹配类别的边成本必须全部为有限非负数")

    resolved_path = resolve_native_library_path(library_path)
    library = _load_native_library(str(resolved_path))
    output_targets = np.empty((beam_width, target_count), dtype=np.int32)
    output_sources = np.empty((beam_width, target_count), dtype=np.int32)
    output_prefix_scores = np.empty(beam_width, dtype=np.float64)
    output_assignment_scores = np.empty(beam_width, dtype=np.float64)
    output_count = c_int32(0)
    native_statistics = _NativeStatisticsV1()
    error_buffer = create_string_buffer(2048)

    native_call_started_at = time.perf_counter()
    return_code = library.task_sequence_optimizer_search_v1(
        NATIVE_ABI_VERSION,
        target_count,
        source_count,
        beam_width,
        first_layer_mask,
        unlock_array.ctypes.data_as(POINTER(c_uint64)),
        target_category_array.ctypes.data_as(POINTER(c_int32)),
        category_source_counts.ctypes.data_as(POINTER(c_int32)),
        category_sources.ctypes.data_as(POINTER(c_int32)),
        source_id_array.ctypes.data_as(POINTER(c_int32)),
        edge_cost_array.ctypes.data_as(POINTER(c_double)),
        beam_width,
        output_count,
        output_targets.ctypes.data_as(POINTER(c_int32)),
        output_sources.ctypes.data_as(POINTER(c_int32)),
        output_prefix_scores.ctypes.data_as(POINTER(c_double)),
        output_assignment_scores.ctypes.data_as(POINTER(c_double)),
        native_statistics,
        error_buffer,
        len(error_buffer),
    )
    native_call_seconds = time.perf_counter() - native_call_started_at
    if return_code != 0:
        message = error_buffer.value.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"C++ 任务顺序优化失败（错误码 {return_code}）：{message}"
        )
    candidate_count = int(output_count.value)
    if not 1 <= candidate_count <= beam_width:
        raise RuntimeError(
            f"C++ 任务顺序优化返回候选数无效：{candidate_count}"
        )
    if int(native_statistics.final_candidate_count) != candidate_count:
        raise RuntimeError("C++ 搜索统计中的候选数与实际输出不一致")

    conversion_started_at = time.perf_counter()
    candidates = tuple(
        NativeCandidate(
            target_sequence=tuple(int(value) for value in output_targets[index]),
            source_sequence=tuple(int(value) for value in output_sources[index]),
            prefix_score=float(output_prefix_scores[index]),
            assignment_score=float(output_assignment_scores[index]),
        )
        for index in range(candidate_count)
    )
    conversion_seconds = time.perf_counter() - conversion_started_at
    statistics = NativeSearchStatistics(
        expanded_parent_count=int(native_statistics.expanded_parent_count),
        generated_child_count=int(native_statistics.generated_child_count),
        peak_retained_node_count=int(native_statistics.peak_retained_node_count),
        final_candidate_count=int(native_statistics.final_candidate_count),
        beam_search_seconds=float(native_statistics.beam_search_seconds),
        source_assignment_seconds=float(native_statistics.source_assignment_seconds),
        native_call_seconds=native_call_seconds,
        candidate_conversion_seconds=conversion_seconds,
    )
    return NativeSearchResult(
        candidates=candidates,
        statistics=statistics,
        library_path=resolved_path,
    )
