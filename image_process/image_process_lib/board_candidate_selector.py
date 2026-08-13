"""V5 盘面的四区粗筛与 relaxed assignment。

本模块不依赖 ROS，不推断合法摆放顺序，也不执行完整路径搜索。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import math
import time
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from image_process_lib.block_category import (
    BLOCK_CATEGORY_NAMES,
    normalize_category_name,
)
from image_process_lib.task_planner import ObservedBlock, normalize_rotation_delta
from image_process_lib.v5_board_library import (
    V5BoardLibrary,
    V5_REGION_MASK_ORDER,
)


# 区域位与 V5 搜索器保持一致：LU=1、RU=2、LD=4、RD=8。
_QUADRANT_BITS = (1, 2, 4, 8)
_QUADRANT_NAMES = {1: "LU", 2: "RU", 4: "LD", 8: "RD"}
_QUADRANT_INDEX_BY_BIT = {bit: index for index, bit in enumerate(_QUADRANT_BITS)}

# 5 个 source 对 5 或 4 个 target 都恰好只有 120 种映射。
_SOURCE_PERMUTATIONS = {
    target_count: np.asarray(
        tuple(permutations(range(5), target_count)), dtype=np.uint8
    )
    for target_count in (4, 5)
}


def _build_region_pair_tables() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lr_table = np.zeros((4, 16), dtype=np.uint8)
    ud_table = np.zeros((4, 16), dtype=np.uint8)
    chosen_region_table = np.zeros((4, 16), dtype=np.uint8)
    for source_index, source_bit in enumerate(_QUADRANT_BITS):
        source_is_right = source_bit in (2, 8)
        source_is_down = source_bit in (4, 8)
        for target_mask in range(16):
            allowed_bits = [bit for bit in _QUADRANT_BITS if target_mask & bit]
            if not allowed_bits:
                continue
            choices = []
            for target_bit in allowed_bits:
                target_is_right = target_bit in (2, 8)
                target_is_down = target_bit in (4, 8)
                choices.append(
                    (
                        int(source_is_right != target_is_right),
                        int(source_is_down != target_is_down),
                        _QUADRANT_INDEX_BY_BIT[target_bit],
                        target_bit,
                    )
                )
            best = min(choices)
            lr_table[source_index, target_mask] = best[0]
            ud_table[source_index, target_mask] = best[1]
            chosen_region_table[source_index, target_mask] = best[3]
    return lr_table, ud_table, chosen_region_table


_PAIR_LR, _PAIR_UD, _PAIR_CHOSEN_REGION = _build_region_pair_tables()


@dataclass(frozen=True)
class BoardCandidateSelectorConfig:
    """候选数量配置。"""

    coarse_top_k: int = 300
    final_candidate_k: int = 20
    keep_coarse_boundary_ties: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.coarse_top_k, bool)
            or not isinstance(self.coarse_top_k, (int, np.integer))
            or int(self.coarse_top_k) <= 0
        ):
            raise ValueError("coarse_top_k 必须是正整数")
        if (
            isinstance(self.final_candidate_k, bool)
            or not isinstance(self.final_candidate_k, (int, np.integer))
            or int(self.final_candidate_k) <= 0
        ):
            raise ValueError("final_candidate_k 必须是正整数")


@dataclass(frozen=True)
class SourceTargetAssignment:
    """一个实体与 placement 的调试对应。"""

    category: str
    source_id: int
    placement_id: int
    source_region_bit: int
    target_region_mask: int
    matched_target_region_bit: int
    distance_mm: float | None = None
    rotation_deg: float | None = None


@dataclass(frozen=True)
class CoarseBoardCandidate:
    """一张盘面的粗筛结果。"""

    board_index: int
    board_id: str
    global_id: int
    layout_index: int
    missing_category: str
    score: Tuple[int, int]
    assignment: Tuple[SourceTargetAssignment, ...]
    unused_source_id: int

    @property
    def n_lr(self) -> int:
        return self.score[0]

    @property
    def n_ud(self) -> int:
        return self.score[1]


@dataclass(frozen=True)
class CoarseSelectionResult:
    """全库四区粗筛结果。"""

    total_board_count: int
    candidates: Tuple[CoarseBoardCandidate, ...]
    elapsed_seconds: float
    source_signature: Tuple[Tuple[int, str], ...]
    source_region_by_id: Tuple[Tuple[int, int], ...]
    library_format_version: int
    library_placement_count: int


@dataclass(frozen=True)
class FinalBoardCandidate:
    """完成 relaxed assignment 后的最终候选。"""

    board_index: int
    board_id: str
    global_id: int
    layout_index: int
    missing_category: str
    score: Tuple[int, int, float, float]
    coarse_assignment: Tuple[SourceTargetAssignment, ...]
    relaxed_assignment: Tuple[SourceTargetAssignment, ...]
    coarse_unused_source_id: int
    unused_source_id: int

    @property
    def n_lr(self) -> int:
        return self.score[0]

    @property
    def n_ud(self) -> int:
        return self.score[1]

    @property
    def relaxed_distance_mm(self) -> float:
        return self.score[2]

    @property
    def relaxed_rotation_deg(self) -> float:
        return self.score[3]


@dataclass(frozen=True)
class BoardCandidateSelectionResult:
    """relaxed assignment 严格 top-k 的输出。"""

    total_board_count: int
    coarse_candidate_count: int
    candidates: Tuple[FinalBoardCandidate, ...]
    coarse_elapsed_seconds: float
    relaxed_elapsed_seconds: float


@dataclass(frozen=True)
class _ValidatedSource:
    source_id: int
    category: str
    category_index: int
    xy_mm: Tuple[float, float]
    detected_angle_deg: float
    pixel_xy: Tuple[float, float]


def _finite_vector(value: object, shape: Tuple[int, ...], label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是有限数值数组") from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{label}必须是形状 {shape} 的有限数值数组")
    return array


def _source_id_as_integer(value: object, source_index: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"第 {source_index} 个 source 的 source_id 必须是整数")
    try:
        source_id = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"第 {source_index} 个 source 的 source_id 必须是整数") from exc
    if not np.isfinite(numeric) or numeric != source_id:
        raise ValueError(f"第 {source_index} 个 source 的 source_id 必须是整数")
    return source_id


def _validate_observed_blocks(
    observed_blocks: Iterable[ObservedBlock],
) -> Tuple[Tuple[_ValidatedSource, ...], ...]:
    blocks = list(observed_blocks)
    if len(blocks) != 35:
        raise ValueError(f"现场 source 必须正好为 35 个，实际为 {len(blocks)} 个")

    category_to_index = {
        category: index for index, category in enumerate(BLOCK_CATEGORY_NAMES)
    }
    grouped = [[] for _ in BLOCK_CATEGORY_NAMES]
    seen_source_ids = set()
    for source_index, block in enumerate(blocks):
        try:
            category = normalize_category_name(str(block.category))
            source_id = _source_id_as_integer(block.source_id, source_index)
        except AttributeError as exc:
            raise ValueError(f"第 {source_index} 个 source 缺少必需字段") from exc
        if category not in category_to_index:
            raise ValueError(f"第 {source_index} 个 source 包含未知类别: {category!r}")
        if source_id in seen_source_ids:
            raise ValueError(f"source_id 必须唯一，发现重复值: {source_id}")
        seen_source_ids.add(source_id)

        try:
            pose = np.asarray(block.observation_pose, dtype=np.float64)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"source {source_id} 的 observation_pose 无效") from exc
        if pose.ndim != 1 or pose.shape[0] < 2 or not np.all(np.isfinite(pose[:2])):
            raise ValueError(f"source {source_id} 的 TCP XY 必须是有限数")
        try:
            pixel = _finite_vector(
                block.high_detected_pixel_xy,
                (2,),
                f"source {source_id} 的 high_detected_pixel_xy",
            )
            detected_angle_deg = float(block.detected_angle_deg)
        except AttributeError as exc:
            raise ValueError(f"source {source_id} 缺少必需字段") from exc
        if not np.isfinite(detected_angle_deg):
            raise ValueError(f"source {source_id} 的检测角度必须是有限数")

        category_index = category_to_index[category]
        grouped[category_index].append(
            _ValidatedSource(
                source_id=source_id,
                category=category,
                category_index=category_index,
                xy_mm=(float(pose[0]), float(pose[1])),
                detected_angle_deg=detected_angle_deg,
                pixel_xy=(float(pixel[0]), float(pixel[1])),
            )
        )

    for category_index, category_sources in enumerate(grouped):
        if len(category_sources) != 5:
            raise ValueError(
                f"类别 {BLOCK_CATEGORY_NAMES[category_index]} 必须正好有 5 个 source，"
                f"实际为 {len(category_sources)} 个"
            )
        category_sources.sort(key=lambda item: item.source_id)
    return tuple(tuple(items) for items in grouped)


def _source_signature(
    grouped_sources: Sequence[Sequence[_ValidatedSource]],
) -> Tuple[Tuple[int, str], ...]:
    signature = [
        (source.source_id, source.category)
        for category_sources in grouped_sources
        for source in category_sources
    ]
    signature.sort()
    return tuple(signature)


def _region_index(pixel_xy: Tuple[float, float], center_xy: np.ndarray) -> int:
    is_right = pixel_xy[0] >= float(center_xy[0])
    is_down = pixel_xy[1] >= float(center_xy[1])
    if not is_right and not is_down:
        return 0
    if is_right and not is_down:
        return 1
    if not is_right and is_down:
        return 2
    return 3


def _target_masks_from_counts(region_counts: Sequence[int]) -> np.ndarray:
    counts = np.asarray(region_counts, dtype=np.int16)
    return np.repeat(np.asarray(V5_REGION_MASK_ORDER, dtype=np.uint8), counts)


def _minimum_region_score(
    source_region_indices: Sequence[int],
    region_counts: Sequence[int],
) -> Tuple[int, int]:
    target_masks = _target_masks_from_counts(region_counts)
    target_count = int(target_masks.shape[0])
    if target_count not in _SOURCE_PERMUTATIONS:
        raise ValueError(f"粗筛的单类 target 数量必须为 4 或 5，实际为 {target_count}")
    permutations_array = _SOURCE_PERMUTATIONS[target_count]
    source_regions = np.asarray(source_region_indices, dtype=np.uint8)
    assigned_regions = source_regions[permutations_array]
    lr_values = np.sum(
        _PAIR_LR[assigned_regions, target_masks[np.newaxis, :]],
        axis=1,
        dtype=np.int16,
    )
    ud_values = np.sum(
        _PAIR_UD[assigned_regions, target_masks[np.newaxis, :]],
        axis=1,
        dtype=np.int16,
    )
    best_index = int(np.lexsort((ud_values, lr_values))[0])
    return int(lr_values[best_index]), int(ud_values[best_index])


class BoardCandidateSelector:
    """独立的两阶段 V5 盘面候选筛选器。"""

    def __init__(
        self,
        library: V5BoardLibrary,
        config: BoardCandidateSelectorConfig | None = None,
    ) -> None:
        if not isinstance(library, V5BoardLibrary):
            raise TypeError("library 必须由 load_v5_board_library 加载")
        self.library = library
        self.config = config or BoardCandidateSelectorConfig()

    def _coarse_assignment_for_category(
        self,
        category_index: int,
        category_sources: Sequence[_ValidatedSource],
        source_region_indices: Sequence[int],
        target_pids: np.ndarray,
    ) -> Tuple[Tuple[SourceTargetAssignment, ...], Tuple[int, int], int | None]:
        target_count = int(target_pids.shape[0])
        permutations_array = _SOURCE_PERMUTATIONS[target_count]
        target_masks = self.library.placement_region_mask[target_pids]
        source_regions = np.asarray(source_region_indices, dtype=np.uint8)
        assigned_regions = source_regions[permutations_array]
        lr_values = np.sum(
            _PAIR_LR[assigned_regions, target_masks[np.newaxis, :]],
            axis=1,
            dtype=np.int16,
        )
        ud_values = np.sum(
            _PAIR_UD[assigned_regions, target_masks[np.newaxis, :]],
            axis=1,
            dtype=np.int16,
        )
        # permutations 本身按已排序 source_id 的字典序生成，最后一级平局可直接用其下标。
        permutation_order = np.arange(permutations_array.shape[0], dtype=np.int16)
        best_index = int(np.lexsort((permutation_order, ud_values, lr_values))[0])
        best_permutation = permutations_array[best_index]

        assignments = []
        for target_offset, source_offset_value in enumerate(best_permutation):
            source_offset = int(source_offset_value)
            source = category_sources[source_offset]
            source_region_index = int(source_regions[source_offset])
            source_region_bit = _QUADRANT_BITS[source_region_index]
            target_mask = int(target_masks[target_offset])
            assignments.append(
                SourceTargetAssignment(
                    category=BLOCK_CATEGORY_NAMES[category_index],
                    source_id=source.source_id,
                    placement_id=int(target_pids[target_offset]),
                    source_region_bit=source_region_bit,
                    target_region_mask=target_mask,
                    matched_target_region_bit=int(
                        _PAIR_CHOSEN_REGION[source_region_index, target_mask]
                    ),
                )
            )
        used_offsets = {int(value) for value in best_permutation}
        unused_offsets = [index for index in range(5) if index not in used_offsets]
        unused_source_id = (
            category_sources[unused_offsets[0]].source_id if unused_offsets else None
        )
        return (
            tuple(assignments),
            (int(lr_values[best_index]), int(ud_values[best_index])),
            unused_source_id,
        )

    def _make_coarse_candidate(
        self,
        board_index: int,
        expected_score: Tuple[int, int],
        grouped_sources: Sequence[Sequence[_ValidatedSource]],
        source_region_indices: Sequence[Sequence[int]],
    ) -> CoarseBoardCandidate:
        assignments = []
        unused_source_ids = []
        n_lr = 0
        n_ud = 0
        for category_index, category_sources in enumerate(grouped_sources):
            raw_pids = self.library.board_target_pid[board_index, category_index]
            target_pids = raw_pids[raw_pids >= 0]
            category_assignment, category_score, unused_source_id = (
                self._coarse_assignment_for_category(
                    category_index,
                    category_sources,
                    source_region_indices[category_index],
                    target_pids,
                )
            )
            assignments.extend(category_assignment)
            n_lr += category_score[0]
            n_ud += category_score[1]
            if unused_source_id is not None:
                unused_source_ids.append(unused_source_id)
        if (n_lr, n_ud) != expected_score:
            raise RuntimeError("粗筛计数得分与重建配对不一致")
        if len(unused_source_ids) != 1:
            raise RuntimeError("粗筛结果必须恰好舍弃一个 source")
        return CoarseBoardCandidate(
            board_index=board_index,
            board_id=self.library.board_id(board_index),
            global_id=int(self.library.board_global_id[board_index]),
            layout_index=int(self.library.board_layout_index[board_index]),
            missing_category=self.library.missing_category_name(board_index),
            score=(n_lr, n_ud),
            assignment=tuple(assignments),
            unused_source_id=unused_source_ids[0],
        )

    def select_coarse(
        self,
        observed_blocks: Iterable[ObservedBlock],
        tray_center_pixel_xy: Sequence[float],
    ) -> CoarseSelectionResult:
        """对全库执行字典序 (N_LR, N_UD) 四区粗筛。"""
        start = time.perf_counter()
        grouped_sources = _validate_observed_blocks(observed_blocks)
        center_xy = _finite_vector(
            tray_center_pixel_xy,
            (2,),
            "tray_center_pixel_xy",
        )
        source_region_indices = tuple(
            tuple(_region_index(source.pixel_xy, center_xy) for source in sources)
            for sources in grouped_sources
        )

        score_cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[int, int]] = {}
        ranked_boards = []
        for board_index in range(self.library.board_count):
            n_lr = 0
            n_ud = 0
            for category_index in range(len(BLOCK_CATEGORY_NAMES)):
                counts = tuple(
                    int(value)
                    for value in self.library.board_region_counts[
                        board_index, category_index
                    ]
                )
                cache_key = (source_region_indices[category_index], counts)
                category_score = score_cache.get(cache_key)
                if category_score is None:
                    category_score = _minimum_region_score(
                        source_region_indices[category_index], counts
                    )
                    score_cache[cache_key] = category_score
                n_lr += category_score[0]
                n_ud += category_score[1]
            ranked_boards.append((n_lr, n_ud, board_index))
        ranked_boards.sort()

        strict_count = min(int(self.config.coarse_top_k), len(ranked_boards))
        selected_ranked = ranked_boards[:strict_count]
        if self.config.keep_coarse_boundary_ties and strict_count < len(ranked_boards):
            boundary_score = ranked_boards[strict_count - 1][:2]
            next_index = strict_count
            while (
                next_index < len(ranked_boards)
                and ranked_boards[next_index][:2] == boundary_score
            ):
                selected_ranked.append(ranked_boards[next_index])
                next_index += 1

        candidates = tuple(
            self._make_coarse_candidate(
                board_index=board_index,
                expected_score=(n_lr, n_ud),
                grouped_sources=grouped_sources,
                source_region_indices=source_region_indices,
            )
            for n_lr, n_ud, board_index in selected_ranked
        )
        return CoarseSelectionResult(
            total_board_count=self.library.board_count,
            candidates=candidates,
            elapsed_seconds=time.perf_counter() - start,
            source_signature=_source_signature(grouped_sources),
            source_region_by_id=tuple(
                sorted(
                    (
                        source.source_id,
                        _QUADRANT_BITS[source_region_indices[category_index][source_offset]],
                    )
                    for category_index, category_sources in enumerate(grouped_sources)
                    for source_offset, source in enumerate(category_sources)
                )
            ),
            library_format_version=self.library.format_version,
            library_placement_count=self.library.placement_count,
        )

    def _validate_coarse_result(self, coarse_result: CoarseSelectionResult) -> None:
        if not isinstance(coarse_result, CoarseSelectionResult):
            raise TypeError("coarse_result 必须由 select_coarse 生成")
        if (
            coarse_result.library_format_version != self.library.format_version
            or coarse_result.total_board_count != self.library.board_count
            or coarse_result.library_placement_count != self.library.placement_count
        ):
            raise ValueError("coarse_result 与当前 V5 盘面库不匹配")
        for candidate in coarse_result.candidates:
            if candidate.board_index < 0 or candidate.board_index >= self.library.board_count:
                raise ValueError("coarse_result 包含越界盘面索引")
            if candidate.board_id != self.library.board_id(candidate.board_index):
                raise ValueError("coarse_result 的盘面编号与当前库不匹配")

    def required_placement_ids(
        self,
        coarse_result: CoarseSelectionResult,
    ) -> Tuple[int, ...]:
        """返回 relaxed 阶段需要转换为 TCP XY 的去重 PID。"""
        self._validate_coarse_result(coarse_result)
        required = set()
        for candidate in coarse_result.candidates:
            board_pids = self.library.board_target_pid[candidate.board_index]
            required.update(int(pid) for pid in board_pids.flat if int(pid) >= 0)
        return tuple(sorted(required))

    def _relaxed_assignment_for_category(
        self,
        category_index: int,
        category_sources: Sequence[_ValidatedSource],
        target_pids: np.ndarray,
        placement_xy_by_pid: np.ndarray,
        board_angle_deg: float,
        source_region_by_id: Mapping[int, int],
    ) -> Tuple[Tuple[SourceTargetAssignment, ...], float, float, int | None]:
        target_count = int(target_pids.shape[0])
        permutations_array = _SOURCE_PERMUTATIONS[target_count]
        source_xy = np.asarray([source.xy_mm for source in category_sources], dtype=np.float64)
        target_xy = placement_xy_by_pid[target_pids]
        distances = np.hypot(
            target_xy[:, np.newaxis, 0] - source_xy[np.newaxis, :, 0],
            target_xy[:, np.newaxis, 1] - source_xy[np.newaxis, :, 1],
        )
        rotations = np.empty((target_count, 5), dtype=np.float64)
        for target_offset, pid_value in enumerate(target_pids):
            pid = int(pid_value)
            target_yaw = float(self.library.placement_yaw_clockwise_deg[pid])
            for source_offset, source in enumerate(category_sources):
                rotations[target_offset, source_offset] = abs(
                    normalize_rotation_delta(
                        BLOCK_CATEGORY_NAMES[category_index],
                        target_yaw,
                        source.detected_angle_deg,
                        board_angle_deg,
                    )
                )

        row_indices = np.arange(target_count, dtype=np.intp)[np.newaxis, :]
        distance_values = np.sum(
            distances[row_indices, permutations_array], axis=1, dtype=np.float64
        )
        rotation_values = np.sum(
            rotations[row_indices, permutations_array], axis=1, dtype=np.float64
        )
        permutation_order = np.arange(permutations_array.shape[0], dtype=np.int16)
        best_index = int(
            np.lexsort((permutation_order, rotation_values, distance_values))[0]
        )
        best_permutation = permutations_array[best_index]

        assignments = []
        target_masks = self.library.placement_region_mask[target_pids]
        for target_offset, source_offset_value in enumerate(best_permutation):
            source_offset = int(source_offset_value)
            source = category_sources[source_offset]
            source_region_bit = int(source_region_by_id[source.source_id])
            source_region_index = _QUADRANT_INDEX_BY_BIT[source_region_bit]
            target_mask = int(target_masks[target_offset])
            assignments.append(
                SourceTargetAssignment(
                    category=BLOCK_CATEGORY_NAMES[category_index],
                    source_id=source.source_id,
                    placement_id=int(target_pids[target_offset]),
                    source_region_bit=source_region_bit,
                    target_region_mask=target_mask,
                    matched_target_region_bit=int(
                        _PAIR_CHOSEN_REGION[source_region_index, target_mask]
                    ),
                    distance_mm=float(distances[target_offset, source_offset]),
                    rotation_deg=float(rotations[target_offset, source_offset]),
                )
            )
        used_offsets = {int(value) for value in best_permutation}
        unused_offsets = [index for index in range(5) if index not in used_offsets]
        unused_source_id = (
            category_sources[unused_offsets[0]].source_id if unused_offsets else None
        )
        return (
            tuple(assignments),
            float(distance_values[best_index]),
            float(rotation_values[best_index]),
            unused_source_id,
        )

    def select_relaxed(
        self,
        coarse_result: CoarseSelectionResult,
        observed_blocks: Iterable[ObservedBlock],
        placement_xy_by_pid: Sequence[Sequence[float]],
        board_angle_deg: float,
    ) -> BoardCandidateSelectionResult:
        """对粗筛候选执行 XY 距离优先、等价旋转次之的独立配对。"""
        start = time.perf_counter()
        self._validate_coarse_result(coarse_result)
        grouped_sources = _validate_observed_blocks(observed_blocks)
        if _source_signature(grouped_sources) != coarse_result.source_signature:
            raise ValueError("relaxed 阶段的 source ID/类别与粗筛输入不一致")
        source_region_by_id = dict(coarse_result.source_region_by_id)
        if set(source_region_by_id) != {
            source.source_id
            for category_sources in grouped_sources
            for source in category_sources
        }:
            raise ValueError("coarse_result 缺少 source 区域信息")
        try:
            placement_xy = np.asarray(placement_xy_by_pid, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("placement_xy_by_pid 必须是数值数组") from exc
        expected_shape = (self.library.placement_count, 2)
        if placement_xy.shape != expected_shape:
            raise ValueError(
                f"placement_xy_by_pid 形状必须为 {expected_shape}，"
                f"实际为 {placement_xy.shape}"
            )
        required_pids = self.required_placement_ids(coarse_result)
        if required_pids and not np.all(
            np.isfinite(placement_xy[np.asarray(required_pids, dtype=np.intp)])
        ):
            raise ValueError("placement_xy_by_pid 中候选盘面所需 PID 包含非有限坐标")
        try:
            board_angle = float(board_angle_deg)
        except (TypeError, ValueError) as exc:
            raise ValueError("board_angle_deg 必须是有限数") from exc
        if not np.isfinite(board_angle):
            raise ValueError("board_angle_deg 必须是有限数")

        relaxed_candidates = []
        for coarse_candidate in coarse_result.candidates:
            board_index = coarse_candidate.board_index
            assignments = []
            unused_source_ids = []
            category_distances = []
            category_rotations = []
            for category_index, category_sources in enumerate(grouped_sources):
                raw_pids = self.library.board_target_pid[board_index, category_index]
                target_pids = raw_pids[raw_pids >= 0]
                (
                    category_assignment,
                    category_distance,
                    category_rotation,
                    unused_source_id,
                ) = self._relaxed_assignment_for_category(
                    category_index,
                    category_sources,
                    target_pids,
                    placement_xy,
                    board_angle,
                    source_region_by_id,
                )
                assignments.extend(category_assignment)
                category_distances.append(category_distance)
                category_rotations.append(category_rotation)
                if unused_source_id is not None:
                    unused_source_ids.append(unused_source_id)
            if len(unused_source_ids) != 1:
                raise RuntimeError("relaxed assignment 必须恰好舍弃一个 source")
            distance = math.fsum(category_distances)
            rotation = math.fsum(category_rotations)
            score = (
                coarse_candidate.n_lr,
                coarse_candidate.n_ud,
                distance,
                rotation,
            )
            relaxed_candidates.append(
                FinalBoardCandidate(
                    board_index=board_index,
                    board_id=coarse_candidate.board_id,
                    global_id=coarse_candidate.global_id,
                    layout_index=coarse_candidate.layout_index,
                    missing_category=coarse_candidate.missing_category,
                    score=score,
                    coarse_assignment=coarse_candidate.assignment,
                    relaxed_assignment=tuple(assignments),
                    coarse_unused_source_id=coarse_candidate.unused_source_id,
                    unused_source_id=unused_source_ids[0],
                )
            )

        # 最后平分只使用库内稳定顺序，且严格截断 top-k。
        relaxed_candidates.sort(key=lambda item: (*item.score, item.board_index))
        final_count = min(int(self.config.final_candidate_k), len(relaxed_candidates))
        selected = tuple(relaxed_candidates[:final_count])
        return BoardCandidateSelectionResult(
            total_board_count=self.library.board_count,
            coarse_candidate_count=len(coarse_result.candidates),
            candidates=selected,
            coarse_elapsed_seconds=coarse_result.elapsed_seconds,
            relaxed_elapsed_seconds=time.perf_counter() - start,
        )


def _region_text(bit: int) -> str:
    return _QUADRANT_NAMES.get(int(bit), f"未知({bit})")


def format_candidate_selection_report(
    coarse_result: CoarseSelectionResult,
    final_result: BoardCandidateSelectionResult,
    coarse_preview_count: int = 10,
) -> str:
    """生成包含最终全部配对的中文调试报告。"""
    preview_count = max(0, int(coarse_preview_count))
    lines = [
        f"总盘面数 = {coarse_result.total_board_count}",
        f"粗筛耗时 = {coarse_result.elapsed_seconds:.4f} 秒",
        "",
        f"粗筛候选（共 {len(coarse_result.candidates)} 张）:",
    ]
    for candidate in coarse_result.candidates[:preview_count]:
        lines.append(
            f"{candidate.board_id}  LR={candidate.n_lr} UD={candidate.n_ud}  "
            f"缺少={candidate.missing_category}  舍弃source={candidate.unused_source_id}"
        )
    if len(coarse_result.candidates) > preview_count:
        lines.append(f"……其余 {len(coarse_result.candidates) - preview_count} 张省略")

    lines.extend(
        [
            "",
            f"relaxed 耗时 = {final_result.relaxed_elapsed_seconds:.4f} 秒",
            f"最终选中 {len(final_result.candidates)} 张盘面:",
        ]
    )
    for candidate in final_result.candidates:
        lines.append(
            f"{candidate.board_id}  LR={candidate.n_lr} UD={candidate.n_ud} "
            f"D={candidate.relaxed_distance_mm:.3f} mm "
            f"R={candidate.relaxed_rotation_deg:.3f}°  "
            f"缺少={candidate.missing_category}  舍弃source={candidate.unused_source_id}"
        )
        for assignment in candidate.relaxed_assignment:
            lines.append(
                f"  source {assignment.source_id} -> placement {assignment.placement_id} "
                f"[{assignment.category}]  距离={assignment.distance_mm:.3f} mm "
                f"旋转={assignment.rotation_deg:.3f}°  "
                f"区域={_region_text(assignment.source_region_bit)}"
            )
    return "\n".join(lines)


# 简短别名，方便离线脚本调用。
format_selection_report = format_candidate_selection_report
