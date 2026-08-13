"""V5 relaxed 候选的完整路径比较与唯一盘面确认。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import json
import math
import time
from typing import Callable, Mapping, Optional, Sequence, Tuple

import numpy as np

from image_process_lib.block_category import normalize_category_name
from image_process_lib.board_candidate_selector import (
    BoardCandidateSelectionResult,
    FinalBoardCandidate,
)
from image_process_lib.task_geometry import (
    build_stable_legal_target_sequence,
    normalize_cells,
)
from image_process_lib.task_planner import ObservedBlock, PlacementTarget, TaskTarget
from image_process_lib.task_sequence_optimizer import (
    TaskPlanResult,
    TaskSequenceOptimizerConfig,
    optimize_task_sequence,
)
from image_process_lib.v5_board_library import V5BoardLibrary


@dataclass(frozen=True)
class FinalBoardSelectorConfig:
    """跨盘面比较与唯一盘面确认参数。"""

    comparison_beam_width: int = 1000
    comparison_returned_candidates: int = 1
    comparison_worker_count: int = 12
    confirmation_candidate_k: int = 10
    confirmation_beam_width: int = 5000
    confirmation_returned_candidates: int = 20
    confirmation_worker_count: int = 10
    soft_time_budget_sec: float = 10.0

    def __post_init__(self) -> None:
        for label, value in (
            ("comparison_beam_width", self.comparison_beam_width),
            ("comparison_returned_candidates", self.comparison_returned_candidates),
            ("comparison_worker_count", self.comparison_worker_count),
            ("confirmation_candidate_k", self.confirmation_candidate_k),
            ("confirmation_beam_width", self.confirmation_beam_width),
            ("confirmation_returned_candidates", self.confirmation_returned_candidates),
            ("confirmation_worker_count", self.confirmation_worker_count),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{label} 必须是正整数")
            if int(value) <= 0:
                raise ValueError(f"{label} 必须是正整数")
        if int(self.comparison_beam_width) > 50000:
            raise ValueError("comparison_beam_width 不能超过 50000")
        if int(self.confirmation_beam_width) > 50000:
            raise ValueError("confirmation_beam_width 不能超过 50000")
        if self.comparison_returned_candidates > self.comparison_beam_width:
            raise ValueError("比较阶段返回数不能大于 Beam 宽度")
        if self.confirmation_returned_candidates > self.confirmation_beam_width:
            raise ValueError("确认阶段返回数不能大于 Beam 宽度")
        try:
            budget = float(self.soft_time_budget_sec)
        except (TypeError, ValueError) as exc:
            raise ValueError("soft_time_budget_sec 必须是有限非负数") from exc
        if not math.isfinite(budget) or budget < 0.0:
            raise ValueError("soft_time_budget_sec 必须是有限非负数")


@dataclass(frozen=True)
class BoardOptimizationAttempt:
    """一张盘面在某一级完整路径优化中的轻量结果。"""

    board_index: int
    board_id: str
    global_id: int
    layout_index: int
    missing_category: str
    selector_score: Tuple[int, int, float, float]
    succeeded: bool
    simplified_cost_seconds: Optional[float]
    servo_replay_total_seconds: Optional[float]
    selected_origin: str
    target_pid_sequence: Tuple[int, ...]
    source_id_sequence: Tuple[int, ...]
    unused_source_ids: Tuple[int, ...]
    elapsed_seconds: float
    error_message: str = ""

    @property
    def ranking_key(self) -> Tuple[float, float, int, int, float, float, int]:
        """返回跨盘面的固定字典序排名键。"""
        if not self.succeeded:
            raise RuntimeError("失败的盘面没有排名键")
        return (
            float(self.simplified_cost_seconds),
            float(self.servo_replay_total_seconds),
            int(self.selector_score[0]),
            int(self.selector_score[1]),
            float(self.selector_score[2]),
            float(self.selector_score[3]),
            int(self.board_index),
        )


@dataclass(frozen=True)
class FinalBoardDecision:
    """被唯一确认并可直接执行的 V5 盘面。"""

    board_index: int
    board_id: str
    global_id: int
    layout_index: int
    missing_category: str
    selector_score: Tuple[int, int, float, float]
    library_sha256: str
    placement_pids: Tuple[int, ...]
    placement_pid_sha256: str
    decision_fingerprint: str
    comparison_attempt: BoardOptimizationAttempt
    comparison_attempts: Tuple[BoardOptimizationAttempt, ...]
    confirmation_attempt: BoardOptimizationAttempt
    confirmation_attempts: Tuple[BoardOptimizationAttempt, ...]
    confirmation_result: TaskPlanResult
    tasks: Tuple[TaskTarget, ...]
    final_source_to_pid: Tuple[Tuple[int, int], ...]
    final_pid_sequence: Tuple[int, ...]
    unused_source_id: int
    comparison_worker_count: int
    confirmation_worker_count: int
    comparison_elapsed_seconds: float
    confirmation_elapsed_seconds: float
    total_elapsed_seconds: float
    soft_time_budget_exceeded: bool


class DynamicBoardSelectionError(RuntimeError):
    """动态盘面链路无法得到唯一可执行结果。"""

    def __init__(
        self,
        message: str,
        attempts: Sequence[BoardOptimizationAttempt] = (),
        confirmation_attempts: Sequence[BoardOptimizationAttempt] = (),
    ) -> None:
        super().__init__(message)
        # attempts 保留为低 Beam 结果，兼容现有失败处理调用方。
        self.attempts = tuple(attempts)
        self.comparison_attempts = self.attempts
        self.confirmation_attempts = tuple(confirmation_attempts)


@dataclass(frozen=True)
class _OptimizationOutcome:
    """线程内部结果；完整 TaskPlanResult 只在高 Beam 阶段暂存。"""

    candidate: FinalBoardCandidate
    attempt: BoardOptimizationAttempt
    placement_pids: Tuple[int, ...] = ()
    ordered_targets: Tuple[PlacementTarget, ...] = ()
    plan_result: Optional[TaskPlanResult] = None
    unused_source_id: Optional[int] = None
    source_to_pid: Tuple[Tuple[int, int], ...] = ()


def _sha256_pid_list(placement_pids: Sequence[int]) -> str:
    values = np.asarray(tuple(int(pid) for pid in placement_pids), dtype="<i2")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _logical_library_sha256(library: V5BoardLibrary) -> str:
    """仅供无源文件的单元测试库生成可重现指纹。"""
    if library.source_sha256:
        digest = str(library.source_sha256).lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("V5 盘面库 source_sha256 无效")
        return digest
    hasher = hashlib.sha256()
    for array in (
        library.board_global_id,
        library.board_layout_index,
        library.board_target_pid,
        library.placement_category,
        library.placement_row,
        library.placement_col,
        library.placement_yaw_clockwise_deg,
        library.placement_cells,
    ):
        contiguous = np.ascontiguousarray(array)
        hasher.update(str(contiguous.dtype).encode("ascii"))
        hasher.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        hasher.update(contiguous.tobytes(order="C"))
    return hasher.hexdigest()


def _decision_fingerprint(
    library_sha256: str,
    board_id: str,
    placement_pids: Sequence[int],
) -> str:
    payload = json.dumps(
        [library_sha256, str(board_id), [int(pid) for pid in placement_pids]],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _board_pids(library: V5BoardLibrary, board_index: int) -> Tuple[int, ...]:
    raw = library.board_target_pid[int(board_index)]
    pids = tuple(sorted(int(pid) for pid in raw.flat if int(pid) >= 0))
    if len(pids) != 34 or len(set(pids)) != 34:
        raise ValueError(f"盘面 {board_index} 必须包含 34 个唯一 PID")
    return pids


def _validate_candidate_identity(
    library: V5BoardLibrary,
    candidate: FinalBoardCandidate,
) -> None:
    index = int(candidate.board_index)
    if index < 0 or index >= library.board_count:
        raise ValueError("最终候选包含越界盘面索引")
    if candidate.board_id != library.board_id(index):
        raise ValueError("最终候选的 board_id 与盘面库不一致")
    if int(candidate.global_id) != int(library.board_global_id[index]):
        raise ValueError("最终候选的 global_id 与盘面库不一致")
    if int(candidate.layout_index) != int(library.board_layout_index[index]):
        raise ValueError("最终候选的 layout_index 与盘面库不一致")
    if candidate.missing_category != library.missing_category_name(index):
        raise ValueError("最终候选的 missing_category 与盘面库不一致")


def _validate_placement_target(
    library: V5BoardLibrary,
    pid: int,
    target: PlacementTarget,
) -> None:
    if int(target.index) != int(pid):
        raise ValueError(f"PID {pid} 的 PlacementTarget.index 不一致")
    expected_category = library.category_names[int(library.placement_category[pid])]
    if normalize_category_name(target.category) != expected_category:
        raise ValueError(f"PID {pid} 的类别与 V5 库不一致")
    expected_numeric = (
        float(library.placement_row[pid]),
        float(library.placement_col[pid]),
        float(library.placement_yaw_clockwise_deg[pid]),
    )
    actual_numeric = (
        float(target.row),
        float(target.col),
        float(target.desired_angle_deg),
    )
    if not np.array_equal(np.asarray(actual_numeric), np.asarray(expected_numeric)):
        raise ValueError(f"PID {pid} 的行列或角度与 V5 库不一致")
    expected_cells = normalize_cells(library.placement_cells[pid], f"PID {pid} cells")
    if normalize_cells(target.cells, f"PID {pid} target cells") != expected_cells:
        raise ValueError(f"PID {pid} 的四格几何与 V5 库不一致")


def _stable_legal_targets(
    library: V5BoardLibrary,
    candidate: FinalBoardCandidate,
    placement_targets_by_pid: Mapping[int, PlacementTarget],
) -> Tuple[Tuple[int, ...], Tuple[PlacementTarget, ...]]:
    pids = _board_pids(library, candidate.board_index)
    missing = [pid for pid in pids if pid not in placement_targets_by_pid]
    if missing:
        raise ValueError(
            f"盘面 {candidate.board_id} 缺少 PlacementTarget PID：{missing}"
        )
    canonical = []
    for pid in pids:
        target = placement_targets_by_pid[pid]
        if not isinstance(target, PlacementTarget):
            raise TypeError(f"PID {pid} 必须对应 PlacementTarget")
        _validate_placement_target(library, pid, target)
        canonical.append(target)
    dense_sequence = build_stable_legal_target_sequence(canonical)
    ordered = tuple(canonical[dense_index] for dense_index in dense_sequence)
    return pids, ordered


def _stage_optimizer_config(
    base: TaskSequenceOptimizerConfig,
    beam_width: int,
    returned_candidates: int,
) -> TaskSequenceOptimizerConfig:
    return replace(
        base,
        beam_width=int(beam_width),
        returned_candidate_limit=int(returned_candidates),
        report_top_candidates=int(returned_candidates),
    )


class FinalBoardSelector:
    """relaxed top-k 之后的低 Beam 比较与前 K 名高 Beam 复核。"""

    def __init__(
        self,
        library: V5BoardLibrary,
        config: FinalBoardSelectorConfig | None = None,
        optimizer_callable: Callable[..., TaskPlanResult] = optimize_task_sequence,
    ) -> None:
        if not isinstance(library, V5BoardLibrary):
            raise TypeError("library 必须是 V5BoardLibrary")
        if not callable(optimizer_callable):
            raise TypeError("optimizer_callable 必须可调用")
        self.library = library
        self.config = config or FinalBoardSelectorConfig()
        self.optimizer_callable = optimizer_callable
        self.library_sha256 = _logical_library_sha256(library)

    def _run_optimizer(
        self,
        observed_blocks: Sequence[ObservedBlock],
        ordered_targets: Sequence[PlacementTarget],
        board_angle_deg: float,
        optimizer_config: TaskSequenceOptimizerConfig,
        motion_model,
        native_library_path,
    ) -> TaskPlanResult:
        result = self.optimizer_callable(
            observed_blocks,
            ordered_targets,
            board_angle_deg,
            optimizer_config,
            motion_model,
            legacy_tasks=None,
            native_library_path=native_library_path,
        )
        if not isinstance(result, TaskPlanResult):
            raise TypeError("完整路径优化器必须返回 TaskPlanResult")
        return result

    @staticmethod
    def _successful_attempt(
        candidate: FinalBoardCandidate,
        result: TaskPlanResult,
        elapsed_seconds: float,
    ) -> BoardOptimizationAttempt:
        """从完整结果提取跨线程传递和报告所需的轻量字段。"""
        return BoardOptimizationAttempt(
            board_index=int(candidate.board_index),
            board_id=candidate.board_id,
            global_id=int(candidate.global_id),
            layout_index=int(candidate.layout_index),
            missing_category=candidate.missing_category,
            selector_score=tuple(candidate.score),
            succeeded=True,
            simplified_cost_seconds=float(result.simplified_cost_seconds),
            servo_replay_total_seconds=float(
                result.selected_servo_replay.replay_total_seconds
            ),
            selected_origin=result.selected_origin,
            target_pid_sequence=tuple(int(value) for value in result.target_sequence),
            source_id_sequence=tuple(int(value) for value in result.source_sequence),
            unused_source_ids=tuple(
                int(value) for value in result.selected_plan.unused_source_ids
            ),
            elapsed_seconds=float(elapsed_seconds),
        )

    @staticmethod
    def _failed_attempt(
        candidate: FinalBoardCandidate,
        error: Exception,
        elapsed_seconds: float,
    ) -> BoardOptimizationAttempt:
        """把单盘异常转换为可继续排序其他盘面的失败记录。"""
        return BoardOptimizationAttempt(
            board_index=int(candidate.board_index),
            board_id=str(candidate.board_id),
            global_id=int(candidate.global_id),
            layout_index=int(candidate.layout_index),
            missing_category=str(candidate.missing_category),
            selector_score=tuple(candidate.score),
            succeeded=False,
            simplified_cost_seconds=None,
            servo_replay_total_seconds=None,
            selected_origin="",
            target_pid_sequence=(),
            source_id_sequence=(),
            unused_source_ids=(),
            elapsed_seconds=float(elapsed_seconds),
            error_message=f"{type(error).__name__}: {error}",
        )

    def _optimize_candidate(
        self,
        candidate: FinalBoardCandidate,
        observed_blocks: Sequence[ObservedBlock],
        placement_targets_by_pid: Mapping[int, PlacementTarget],
        board_angle_deg: float,
        optimizer_config: TaskSequenceOptimizerConfig,
        motion_model,
        native_library_path,
        *,
        prepared_outcome: Optional[_OptimizationOutcome] = None,
        retain_plan_result: bool = False,
        validate_final_plan: bool = False,
    ) -> _OptimizationOutcome:
        """独立优化一张盘面，线程间不写共享状态。"""
        started_at = time.perf_counter()
        try:
            if prepared_outcome is None:
                _validate_candidate_identity(self.library, candidate)
                placement_pids, ordered_targets = _stable_legal_targets(
                    self.library,
                    candidate,
                    placement_targets_by_pid,
                )
            else:
                placement_pids = prepared_outcome.placement_pids
                ordered_targets = prepared_outcome.ordered_targets
                if not placement_pids or not ordered_targets:
                    raise RuntimeError("高 Beam 复核缺少低 Beam 阶段的合法目标")
            result = self._run_optimizer(
                observed_blocks,
                ordered_targets,
                board_angle_deg,
                optimizer_config,
                motion_model,
                native_library_path,
            )
            unused_source_id = None
            source_to_pid = ()
            if validate_final_plan:
                unused_source_id, source_to_pid = self._validate_final_plan(
                    candidate,
                    placement_pids,
                    observed_blocks,
                    result,
                )
            attempt = self._successful_attempt(
                candidate,
                result,
                time.perf_counter() - started_at,
            )
            return _OptimizationOutcome(
                candidate=candidate,
                attempt=attempt,
                placement_pids=placement_pids,
                ordered_targets=ordered_targets,
                plan_result=result if retain_plan_result else None,
                unused_source_id=unused_source_id,
                source_to_pid=source_to_pid,
            )
        except Exception as exc:
            return _OptimizationOutcome(
                candidate=candidate,
                attempt=self._failed_attempt(
                    candidate,
                    exc,
                    time.perf_counter() - started_at,
                ),
            )

    @staticmethod
    def _run_parallel_stage(
        inputs: Sequence,
        worker_count: int,
        worker: Callable,
        thread_name_prefix: str,
    ) -> Tuple[Tuple[_OptimizationOutcome, ...], int]:
        """并行执行但按输入顺序返回，消除完成先后对结果的影响。"""
        values = tuple(inputs)
        if not values:
            return (), 0
        actual_worker_count = min(int(worker_count), len(values))
        if actual_worker_count == 1:
            return tuple(worker(value) for value in values), 1
        with ThreadPoolExecutor(
            max_workers=actual_worker_count,
            thread_name_prefix=thread_name_prefix,
        ) as executor:
            return tuple(executor.map(worker, values)), actual_worker_count

    @staticmethod
    def _validate_final_plan(
        candidate: FinalBoardCandidate,
        placement_pids: Sequence[int],
        observed_blocks: Sequence[ObservedBlock],
        result: TaskPlanResult,
    ) -> Tuple[int, Tuple[Tuple[int, int], ...]]:
        target_sequence = tuple(int(pid) for pid in result.target_sequence)
        source_sequence = tuple(int(source_id) for source_id in result.source_sequence)
        if len(target_sequence) != 34 or set(target_sequence) != set(placement_pids):
            raise RuntimeError("确认结果未且仅使用盘面的 34 个 PID")
        if len(source_sequence) != 34 or len(set(source_sequence)) != 34:
            raise RuntimeError("确认结果未使用 34 个不同 source")
        source_category = {
            int(block.source_id): normalize_category_name(block.category)
            for block in observed_blocks
        }
        if len(source_category) != 35:
            raise RuntimeError("动态盘面确认要求 35 个唯一 source")
        unused = sorted(set(source_category) - set(source_sequence))
        if len(unused) != 1:
            raise RuntimeError("确认结果必须恰好留下一个未使用 source")
        if source_category[unused[0]] != candidate.missing_category:
            raise RuntimeError("未使用 source 类别与 missing_category 不一致")
        return unused[0], tuple(zip(source_sequence, target_sequence))

    def select(
        self,
        relaxed_result: BoardCandidateSelectionResult,
        observed_blocks: Sequence[ObservedBlock],
        placement_targets_by_pid: Mapping[int, PlacementTarget],
        board_angle_deg: float,
        optimizer_config: TaskSequenceOptimizerConfig,
        motion_model,
        native_library_path=None,
    ) -> FinalBoardDecision:
        """并行比较全部 relaxed 候选，再并行复核低 Beam 前 K 名。"""
        if not isinstance(relaxed_result, BoardCandidateSelectionResult):
            raise TypeError("relaxed_result 必须由 BoardCandidateSelector 生成")
        if relaxed_result.total_board_count != self.library.board_count:
            raise ValueError("relaxed_result 与当前 V5 盘面库不匹配")
        candidates = tuple(relaxed_result.candidates)
        if not candidates:
            raise DynamicBoardSelectionError("relaxed 阶段没有候选盘面")
        blocks = tuple(observed_blocks)
        if len(blocks) != 35:
            raise ValueError(f"动态盘面要求正好 35 个 source，实际为 {len(blocks)}")
        try:
            board_angle = float(board_angle_deg)
        except (TypeError, ValueError) as exc:
            raise ValueError("board_angle_deg 必须是有限数") from exc
        if not math.isfinite(board_angle):
            raise ValueError("board_angle_deg 必须是有限数")

        comparison_config = _stage_optimizer_config(
            optimizer_config,
            self.config.comparison_beam_width,
            self.config.comparison_returned_candidates,
        )
        confirmation_config = _stage_optimizer_config(
            optimizer_config,
            self.config.confirmation_beam_width,
            self.config.confirmation_returned_candidates,
        )
        overall_started_at = time.perf_counter()
        comparison_started_at = overall_started_at
        comparison_outcomes, comparison_worker_count = self._run_parallel_stage(
            candidates,
            self.config.comparison_worker_count,
            lambda candidate: self._optimize_candidate(
                candidate,
                blocks,
                placement_targets_by_pid,
                board_angle,
                comparison_config,
                motion_model,
                native_library_path,
            ),
            "v5-low-beam",
        )
        comparison_elapsed = time.perf_counter() - comparison_started_at
        comparison_attempts = tuple(
            outcome.attempt for outcome in comparison_outcomes
        )
        successful_comparisons = sorted(
            (
                outcome
                for outcome in comparison_outcomes
                if outcome.attempt.succeeded
            ),
            key=lambda outcome: outcome.attempt.ranking_key,
        )
        if not successful_comparisons:
            raise DynamicBoardSelectionError(
                "relaxed 候选的比较级路径优化全部失败",
                comparison_attempts,
            )

        confirmation_inputs = tuple(
            successful_comparisons[: self.config.confirmation_candidate_k]
        )
        confirmation_started_at = time.perf_counter()
        confirmation_outcomes, confirmation_worker_count = self._run_parallel_stage(
            confirmation_inputs,
            self.config.confirmation_worker_count,
            lambda comparison_outcome: self._optimize_candidate(
                comparison_outcome.candidate,
                blocks,
                placement_targets_by_pid,
                board_angle,
                confirmation_config,
                motion_model,
                native_library_path,
                prepared_outcome=comparison_outcome,
                retain_plan_result=True,
                validate_final_plan=True,
            ),
            "v5-high-beam",
        )
        confirmation_elapsed = time.perf_counter() - confirmation_started_at
        confirmation_attempts = tuple(
            outcome.attempt for outcome in confirmation_outcomes
        )
        successful_confirmations = tuple(
            outcome
            for outcome in confirmation_outcomes
            if outcome.attempt.succeeded
        )
        if not successful_confirmations:
            raise DynamicBoardSelectionError(
                "低 Beam 前 K 名盘面的高 Beam 复核全部失败",
                comparison_attempts,
                confirmation_attempts,
            )

        winning_outcome = min(
            successful_confirmations,
            key=lambda outcome: outcome.attempt.ranking_key,
        )
        winning_candidate = winning_outcome.candidate
        winning_attempt = next(
            outcome.attempt
            for outcome in successful_comparisons
            if outcome.candidate.board_index == winning_candidate.board_index
        )
        confirmation_attempt = winning_outcome.attempt
        confirmation_result = winning_outcome.plan_result
        unused_source_id = winning_outcome.unused_source_id
        source_to_pid = winning_outcome.source_to_pid
        winning_pids = winning_outcome.placement_pids
        if confirmation_result is None or unused_source_id is None:
            raise RuntimeError("高 Beam 胜出盘面的内部结果不完整")
        total_elapsed = time.perf_counter() - overall_started_at
        pid_sha256 = _sha256_pid_list(winning_pids)
        fingerprint = _decision_fingerprint(
            self.library_sha256,
            winning_candidate.board_id,
            winning_pids,
        )
        return FinalBoardDecision(
            board_index=int(winning_candidate.board_index),
            board_id=winning_candidate.board_id,
            global_id=int(winning_candidate.global_id),
            layout_index=int(winning_candidate.layout_index),
            missing_category=winning_candidate.missing_category,
            selector_score=tuple(winning_candidate.score),
            library_sha256=self.library_sha256,
            placement_pids=winning_pids,
            placement_pid_sha256=pid_sha256,
            decision_fingerprint=fingerprint,
            comparison_attempt=winning_attempt,
            comparison_attempts=comparison_attempts,
            confirmation_attempt=confirmation_attempt,
            confirmation_attempts=confirmation_attempts,
            confirmation_result=confirmation_result,
            tasks=tuple(confirmation_result.tasks),
            final_source_to_pid=source_to_pid,
            final_pid_sequence=tuple(
                int(value) for value in confirmation_result.target_sequence
            ),
            unused_source_id=unused_source_id,
            comparison_worker_count=comparison_worker_count,
            confirmation_worker_count=confirmation_worker_count,
            comparison_elapsed_seconds=comparison_elapsed,
            confirmation_elapsed_seconds=confirmation_elapsed,
            total_elapsed_seconds=total_elapsed,
            soft_time_budget_exceeded=(
                total_elapsed > float(self.config.soft_time_budget_sec)
            ),
        )


def build_unique_board_manifest(decision: FinalBoardDecision) -> dict:
    """构造可原样写入 JSON 的唯一盘面清单。"""
    if not isinstance(decision, FinalBoardDecision):
        raise TypeError("decision 必须是 FinalBoardDecision")
    comparison_ranking = sorted(
        (item for item in decision.comparison_attempts if item.succeeded),
        key=lambda item: item.ranking_key,
    )
    confirmation_ranking = sorted(
        (item for item in decision.confirmation_attempts if item.succeeded),
        key=lambda item: item.ranking_key,
    )
    comparison_rank = next(
        index
        for index, item in enumerate(comparison_ranking, start=1)
        if item.board_index == decision.board_index
    )
    confirmation_rank = next(
        index
        for index, item in enumerate(confirmation_ranking, start=1)
        if item.board_index == decision.board_index
    )
    return {
        "library_sha256": decision.library_sha256,
        "board_id": decision.board_id,
        "global_id": decision.global_id,
        "layout_index": decision.layout_index,
        "missing_category": decision.missing_category,
        "placement_pids": list(decision.placement_pids),
        "placement_pid_sha256": decision.placement_pid_sha256,
        "decision_fingerprint": decision.decision_fingerprint,
        "final_source_to_pid": [
            {"source_id": source_id, "placement_pid": pid}
            for source_id, pid in decision.final_source_to_pid
        ],
        "final_pid_sequence": list(decision.final_pid_sequence),
        "unused_source_id": decision.unused_source_id,
        "comparison_rank": comparison_rank,
        "confirmation_rank": confirmation_rank,
        "comparison_candidate_count": len(decision.comparison_attempts),
        "confirmation_candidate_count": len(decision.confirmation_attempts),
        "comparison_worker_count": decision.comparison_worker_count,
        "confirmation_worker_count": decision.confirmation_worker_count,
        "comparison_simplified_cost_seconds": (
            decision.comparison_attempt.simplified_cost_seconds
        ),
        "comparison_servo_replay_total_seconds": (
            decision.comparison_attempt.servo_replay_total_seconds
        ),
        "confirmation_simplified_cost_seconds": (
            decision.confirmation_attempt.simplified_cost_seconds
        ),
        "confirmation_servo_replay_total_seconds": (
            decision.confirmation_attempt.servo_replay_total_seconds
        ),
        "timings_seconds": {
            "comparison": decision.comparison_elapsed_seconds,
            "confirmation": decision.confirmation_elapsed_seconds,
            "total": decision.total_elapsed_seconds,
        },
        "soft_time_budget_exceeded": decision.soft_time_budget_exceeded,
    }
