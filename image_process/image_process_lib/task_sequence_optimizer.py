"""固定盘面的合法摆放顺序与实体对应联合优化 V1。"""

from dataclasses import dataclass, replace
from datetime import datetime
import heapq
import math
import time
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from image_process_lib.arm_motion_time import ArmMotionTimeModel
from image_process_lib.block_category import BLOCK_CATEGORY_NAMES, normalize_category_name
from image_process_lib.servo_angle_model import plan_servo_angle_transition
from image_process_lib.task_geometry import (
    SupportGraph,
    build_support_graph,
    validate_dense_target_sequence,
)
from image_process_lib.task_planner import (
    ObservedBlock,
    PlacementTarget,
    TaskTarget,
    make_task_target,
    normalize_rotation_delta,
)
from image_process_lib.task_sequence_optimizer_native import (
    NativeSearchResult,
    run_native_task_sequence_search,
)


START_PREDECESSOR = -1
INFINITE_COST = float("inf")
_COST_TOLERANCE = 1e-10


@dataclass(frozen=True)
class TaskSequenceOptimizerConfig:
    """V1 搜索、位姿和舵机重放所需的固定参数。"""

    shooting_pose: Sequence[float]
    camera_to_sucker_offset_mm: Sequence[float]
    pick_surface_offset_mm: float
    pick_approach_clearance_mm: float
    motor_velocity_deg_per_sec: float
    initial_motor_angle_deg: float
    motor_lower_margin_deg: float
    motor_upper_margin_deg: float
    beam_width: int = 1000
    report_top_candidates: int = 20
    returned_candidate_limit: Optional[int] = None


@dataclass(frozen=True)
class MotionCostTables:
    """Beam 热循环只读的全部路程、时间与类别索引表。"""

    blocks: Tuple[ObservedBlock, ...]
    targets: Tuple[PlacementTarget, ...]
    support_graph: SupportGraph
    source_pre_pick_xyz: np.ndarray
    source_pick_xyz: np.ndarray
    target_place_xyz: np.ndarray
    start_xyz: np.ndarray
    empty_distance_mm: np.ndarray
    empty_time_seconds: np.ndarray
    loaded_distance_mm: np.ndarray
    loaded_time_seconds: np.ndarray
    rotation_delta_deg: np.ndarray
    rotation_time_seconds: np.ndarray
    edge_cost_seconds: np.ndarray
    category_index_by_target: Tuple[int, ...]
    sources_by_category: Tuple[Tuple[int, ...], ...]

    @property
    def start_predecessor_index(self) -> int:
        return len(self.targets)


@dataclass(frozen=True)
class PlanCostStep:
    """一块方块在简化时间模型中的完整单步明细。"""

    step: int
    predecessor_target_id: Optional[int]
    target_dense_index: int
    target_id: int
    source_index: int
    source_id: int
    category: str
    empty_distance_mm: float
    loaded_distance_mm: float
    empty_time_seconds: float
    loaded_time_seconds: float
    rotation_delta_deg: float
    rotation_time_seconds: float
    edge_cost_seconds: float


@dataclass(frozen=True)
class AssignmentPlan:
    """一条完整 target 顺序及其确定的 source 对应。"""

    target_dense_sequence: Tuple[int, ...]
    source_indices: Tuple[int, ...]
    target_id_sequence: Tuple[int, ...]
    source_id_sequence: Tuple[int, ...]
    unused_source_indices: Tuple[int, ...]
    unused_source_ids: Tuple[int, ...]
    simplified_cost_seconds: float
    steps: Tuple[PlanCostStep, ...]


@dataclass(frozen=True)
class ServoReplayStep:
    """共享舵机边界规则重放后的一步并行时间统计。"""

    step: int
    source_id: int
    target_id: int
    start_angle_deg: float
    pick_angle_deg: float
    place_angle_deg: float
    pre_pick_target_angle_deg: Optional[float]
    pre_pick_rotation_seconds: float
    loaded_rotation_seconds: float
    empty_motion_seconds: float
    loaded_motion_seconds: float
    servo_covered_seconds: float
    servo_naked_wait_seconds: float
    replay_step_seconds: float


@dataclass(frozen=True)
class ServoReplayResult:
    """完整方案的绝对角状态重放结果。"""

    pre_rotation_count: int
    servo_covered_seconds: float
    servo_naked_wait_seconds: float
    replay_total_seconds: float
    final_angle_deg: float
    steps: Tuple[ServoReplayStep, ...]


@dataclass(frozen=True)
class BeamSearchStatistics:
    elapsed_seconds: float
    expanded_parent_count: int
    generated_child_count: int
    peak_retained_node_count: int
    final_candidate_count: int
    returned_candidate_count: int = 0
    backend: str = "python_reference"
    source_assignment_elapsed_seconds: float = 0.0
    native_call_elapsed_seconds: float = 0.0
    candidate_conversion_elapsed_seconds: float = 0.0
    python_postprocessing_elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class BeamCandidateResult:
    plan: AssignmentPlan
    servo_replay: ServoReplayResult


@dataclass(frozen=True)
class TaskPlanResult:
    """V1 的生产任务、三组对照与完整性能/重放结果。"""

    tasks: Tuple[TaskTarget, ...]
    selected_origin: str
    selected_plan: AssignmentPlan
    selected_servo_replay: ServoReplayResult
    fixed_order_plan: AssignmentPlan
    fixed_order_servo_replay: ServoReplayResult
    beam_best_plan: AssignmentPlan
    beam_best_servo_replay: ServoReplayResult
    legacy_plan: Optional[AssignmentPlan]
    legacy_servo_replay: Optional[ServoReplayResult]
    final_beam_candidates: Tuple[BeamCandidateResult, ...]
    true_replay_best_candidate_index: int
    simplified_and_replay_ranking_differ: bool
    statistics: BeamSearchStatistics
    cost_tables: MotionCostTables
    report_top_candidates: int

    @property
    def target_sequence(self) -> Tuple[int, ...]:
        return self.selected_plan.target_id_sequence

    @property
    def source_sequence(self) -> Tuple[int, ...]:
        return self.selected_plan.source_id_sequence

    @property
    def simplified_cost_seconds(self) -> float:
        return self.selected_plan.simplified_cost_seconds


@dataclass(frozen=True)
class OptimizerModeDecision:
    """三种生产模式在不依赖 ROS 时的确定性选择结果。"""

    tasks: Tuple[TaskTarget, ...]
    plan_result: Optional[TaskPlanResult]
    status: str
    error_message: str = ""


def decide_optimizer_mode(
    mode: str,
    legacy_tasks: Sequence[TaskTarget],
    visual_servo_enabled: bool,
    optimize_callable: Callable[[], TaskPlanResult],
) -> OptimizerModeDecision:
    """执行 legacy/shadow/execute 的失败隔离与任务选择规则。"""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in ("legacy", "shadow", "execute"):
        raise ValueError("规划模式只能是 legacy、shadow 或 execute")
    legacy_tuple = tuple(legacy_tasks)
    if normalized_mode == "legacy":
        return OptimizerModeDecision(legacy_tuple, None, "legacy")
    if normalized_mode == "execute" and bool(visual_servo_enabled):
        raise RuntimeError(
            "V1 execute 只允许 servo.enabled=false 的开环执行；"
            "闭环视觉修正只能使用 shadow 估算"
        )
    try:
        result = optimize_callable()
    except Exception as exc:
        if normalized_mode == "shadow":
            return OptimizerModeDecision(
                legacy_tuple,
                None,
                "shadow_failed",
                str(exc),
            )
        raise
    if normalized_mode == "shadow":
        return OptimizerModeDecision(legacy_tuple, result, "shadow")
    return OptimizerModeDecision(tuple(result.tasks), result, "execute")


@dataclass(frozen=True)
class _BeamNode:
    placed_mask: int
    available_mask: int
    last_target: int
    target_sequence: Tuple[int, ...]
    category_dps: Tuple[Tuple[float, ...], ...]
    category_best_costs: Tuple[float, ...]
    prefix_score: float


class MotionModelSpeedMismatchError(ValueError):
    """路程时间标定速度与当前执行速度不一致。"""


def validate_motion_model_speeds(
    motion_model: ArmMotionTimeModel,
    arm_speed: float,
    pick_approach_speed: float,
) -> None:
    """确认搜索中的两类 MoveL 调用都匹配路程时间标定速度。"""
    expected = float(motion_model.move_speed_percent)
    values = (float(arm_speed), float(pick_approach_speed))
    if any(not math.isfinite(value) for value in values):
        raise ValueError("机械臂速度必须是有限数值")
    if any(abs(value - expected) > 1e-9 for value in values):
        raise MotionModelSpeedMismatchError(
            "路程时间标定速度与当前执行速度不一致："
            f"标定={expected:g}，arm_speed={values[0]:g}，"
            f"pick_approach_speed={values[1]:g}"
        )


def _finite_xyz_from_pose(pose: Sequence[float], label: str) -> np.ndarray:
    values = np.asarray(pose, dtype=float)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{label}必须是包含 6 个有限数值的位姿")
    return values[:3].copy()


def _nonnegative_integer_id(value, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label}必须是非负整数")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是非负整数") from exc
    if not math.isfinite(number) or not number.is_integer() or number < 0.0:
        raise ValueError(f"{label}必须是非负整数")
    return int(number)


def _validate_optimizer_config(config: TaskSequenceOptimizerConfig) -> Tuple[np.ndarray, np.ndarray]:
    start_xyz = _finite_xyz_from_pose(config.shooting_pose, "高位拍摄位")
    offset = np.asarray(config.camera_to_sucker_offset_mm, dtype=float)
    if offset.shape != (2,) or not np.all(np.isfinite(offset)):
        raise ValueError("相机到吸盘偏移必须包含 2 个有限数值")
    numeric = np.asarray([
        config.pick_surface_offset_mm,
        config.pick_approach_clearance_mm,
        config.motor_velocity_deg_per_sec,
        config.initial_motor_angle_deg,
        config.motor_lower_margin_deg,
        config.motor_upper_margin_deg,
    ], dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("任务顺序优化配置包含非有限数值")
    if config.pick_approach_clearance_mm <= 0.0:
        raise ValueError("预抓取间隙必须大于 0")
    if config.motor_velocity_deg_per_sec <= 0.0:
        raise ValueError("舵机速度必须大于 0")
    if not 0.0 <= config.initial_motor_angle_deg <= 360.0:
        raise ValueError("初始舵机角度必须位于 [0, 360]")
    if not (
        0.0
        <= config.motor_lower_margin_deg
        < config.motor_upper_margin_deg
        <= 360.0
    ):
        raise ValueError("舵机安全角度边界无效")
    if isinstance(config.beam_width, bool) or int(config.beam_width) != config.beam_width:
        raise ValueError("beam_width 必须是正整数")
    if int(config.beam_width) <= 0:
        raise ValueError("beam_width 必须是正整数")
    if (
        isinstance(config.report_top_candidates, bool)
        or int(config.report_top_candidates) != config.report_top_candidates
        or int(config.report_top_candidates) <= 0
    ):
        raise ValueError("report_top_candidates 必须是正整数")
    if config.returned_candidate_limit is not None:
        if (
            isinstance(config.returned_candidate_limit, bool)
            or int(config.returned_candidate_limit) != config.returned_candidate_limit
            or int(config.returned_candidate_limit) <= 0
        ):
            raise ValueError("returned_candidate_limit 必须是正整数或 None")
        if int(config.returned_candidate_limit) > int(config.beam_width):
            raise ValueError("returned_candidate_limit 不能大于 beam_width")
    return start_xyz, offset


def _validate_and_order_inputs(
    blocks: Sequence[ObservedBlock],
    targets: Sequence[PlacementTarget],
) -> Tuple[Tuple[ObservedBlock, ...], Tuple[PlacementTarget, ...]]:
    if not blocks:
        raise ValueError("当次识别实体不能为空")
    if not targets:
        raise ValueError("摆放目标不能为空")
    blocks_with_ids = [
        (_nonnegative_integer_id(block.source_id, "source_id"), block)
        for block in blocks
    ]
    blocks_with_ids.sort(key=lambda item: item[0])
    source_ids = [item[0] for item in blocks_with_ids]
    sorted_blocks = tuple(item[1] for item in blocks_with_ids)
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source_id 不能重复")
    # target 输入列表就是旧规划的固定执行顺序；ID 只负责身份，不得拿来重排。
    ordered_targets = tuple(targets)
    target_ids = [
        _nonnegative_integer_id(target.index, "target ID")
        for target in ordered_targets
    ]
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("target ID 不能重复")

    known_categories = set(BLOCK_CATEGORY_NAMES)
    source_counts = {category: 0 for category in BLOCK_CATEGORY_NAMES}
    target_counts = {category: 0 for category in BLOCK_CATEGORY_NAMES}
    for block in sorted_blocks:
        category = normalize_category_name(block.category)
        if category not in known_categories:
            raise ValueError(f"不支持的实体类别：{category}")
        _finite_xyz_from_pose(block.observation_pose, f"实体 {block.source_id} 观察位")
        if not math.isfinite(float(block.detected_angle_deg)):
            raise ValueError(f"实体 {block.source_id} 角度无效")
        if not math.isfinite(float(block.pick_surface_z_mm)):
            raise ValueError(f"实体 {block.source_id} 抓取表面 Z 无效")
        if not bool(block.pick_surface_z_valid):
            raise ValueError(f"实体 {block.source_id} 缺少有效抓取表面 Z")
        source_counts[category] += 1
    for target in ordered_targets:
        category = normalize_category_name(target.category)
        if category not in known_categories:
            raise ValueError(f"不支持的目标类别：{category}")
        _finite_xyz_from_pose(target.observation_pose, f"目标 {target.index} 观察位")
        if not math.isfinite(float(target.desired_angle_deg)):
            raise ValueError(f"目标 {target.index} 角度无效")
        target_counts[category] += 1
    for category in BLOCK_CATEGORY_NAMES:
        if source_counts[category] > 5:
            raise ValueError(
                f"{category} 实体数为 {source_counts[category]}，超过 V1 的 5-bit 上限"
            )
        if target_counts[category] > source_counts[category]:
            raise ValueError(
                f"{category} 实体数量不足：检测到 {source_counts[category]}，"
                f"目标需要 {target_counts[category]}"
            )
    return sorted_blocks, ordered_targets


def build_motion_cost_tables(
    blocks: Sequence[ObservedBlock],
    targets: Sequence[PlacementTarget],
    board_angle_deg: float,
    config: TaskSequenceOptimizerConfig,
    motion_model: ArmMotionTimeModel,
) -> MotionCostTables:
    """一次批量完成全部 XYZ 路程和边成本预计算。"""
    try:
        board_angle = float(board_angle_deg)
    except (TypeError, ValueError) as exc:
        raise ValueError("托盘角度必须是有限数值") from exc
    if not math.isfinite(board_angle):
        raise ValueError("托盘角度必须是有限数值")
    start_xyz, camera_offset = _validate_optimizer_config(config)
    sorted_blocks, ordered_targets = _validate_and_order_inputs(blocks, targets)
    support_graph = build_support_graph(ordered_targets)
    source_count = len(sorted_blocks)
    target_count = len(ordered_targets)

    source_pre_pick_xyz = np.empty((source_count, 3), dtype=float)
    source_pick_xyz = np.empty((source_count, 3), dtype=float)
    for source_index, block in enumerate(sorted_blocks):
        observation_xyz = _finite_xyz_from_pose(
            block.observation_pose,
            f"实体 {block.source_id} 观察位",
        )
        pick_xyz = observation_xyz.copy()
        pick_xyz[:2] += camera_offset
        pick_xyz[2] = float(block.pick_surface_z_mm) + float(config.pick_surface_offset_mm)
        pre_pick_xyz = pick_xyz.copy()
        pre_pick_xyz[2] += float(config.pick_approach_clearance_mm)
        source_pick_xyz[source_index] = pick_xyz
        source_pre_pick_xyz[source_index] = pre_pick_xyz

    target_place_xyz = np.empty((target_count, 3), dtype=float)
    for target_index, target in enumerate(ordered_targets):
        place_xyz = _finite_xyz_from_pose(
            target.observation_pose,
            f"目标 {target.index} 观察位",
        )
        place_xyz[:2] += camera_offset
        target_place_xyz[target_index] = place_xyz

    predecessor_xyz = np.vstack((target_place_xyz, start_xyz.reshape(1, 3)))
    empty_distance = np.linalg.norm(
        predecessor_xyz[:, None, :] - source_pre_pick_xyz[None, :, :],
        axis=2,
    )
    loaded_distance = np.full((source_count, target_count), np.nan, dtype=float)
    matching_pairs = []
    matching_loaded_distances = []
    category_to_index = {
        category: index for index, category in enumerate(BLOCK_CATEGORY_NAMES)
    }
    sources_by_category_lists = [[] for _ in BLOCK_CATEGORY_NAMES]
    for source_index, block in enumerate(sorted_blocks):
        category = normalize_category_name(block.category)
        sources_by_category_lists[category_to_index[category]].append(source_index)
        for target_index, target in enumerate(ordered_targets):
            if category != normalize_category_name(target.category):
                continue
            distance = float(np.linalg.norm(
                source_pick_xyz[source_index] - target_place_xyz[target_index]
            ))
            loaded_distance[source_index, target_index] = distance
            matching_pairs.append((source_index, target_index))
            matching_loaded_distances.append(distance)

    # 热循环前只调用一次批量接口；随后全部查询都是数组索引。
    all_distances = np.concatenate((
        empty_distance.ravel(),
        np.asarray(matching_loaded_distances, dtype=float),
    ))
    all_times = np.asarray(
        motion_model.predict_array_seconds(all_distances),
        dtype=float,
    )
    if (
        all_times.shape != all_distances.shape
        or not np.all(np.isfinite(all_times))
        or np.any(all_times < 0.0)
    ):
        raise ValueError("机械臂路程时间模型批量输出形状或数值无效")
    empty_size = empty_distance.size
    empty_time = all_times[:empty_size].reshape(empty_distance.shape)
    loaded_time = np.full((source_count, target_count), INFINITE_COST, dtype=float)
    rotation_delta = np.full((source_count, target_count), np.nan, dtype=float)
    rotation_time = np.full((source_count, target_count), INFINITE_COST, dtype=float)
    edge_cost = np.full(
        (target_count + 1, source_count, target_count),
        INFINITE_COST,
        dtype=float,
    )
    matching_times = all_times[empty_size:]
    for pair_offset, (source_index, target_index) in enumerate(matching_pairs):
        loaded = float(matching_times[pair_offset])
        delta = normalize_rotation_delta(
            ordered_targets[target_index].category,
            ordered_targets[target_index].desired_angle_deg,
            sorted_blocks[source_index].detected_angle_deg,
            board_angle,
        )
        rotation = abs(delta) / float(config.motor_velocity_deg_per_sec)
        loaded_time[source_index, target_index] = loaded
        rotation_delta[source_index, target_index] = delta
        rotation_time[source_index, target_index] = rotation
        edge_cost[:, source_index, target_index] = (
            empty_time[:, source_index] + max(loaded, rotation)
        )

    arrays = (
        source_pre_pick_xyz,
        source_pick_xyz,
        target_place_xyz,
        start_xyz,
        empty_distance,
        empty_time,
        loaded_distance,
        loaded_time,
        rotation_delta,
        rotation_time,
        edge_cost,
    )
    for array in arrays:
        array.setflags(write=False)
    category_index_by_target = tuple(
        category_to_index[normalize_category_name(target.category)]
        for target in ordered_targets
    )
    return MotionCostTables(
        blocks=sorted_blocks,
        targets=ordered_targets,
        support_graph=support_graph,
        source_pre_pick_xyz=source_pre_pick_xyz,
        source_pick_xyz=source_pick_xyz,
        target_place_xyz=target_place_xyz,
        start_xyz=start_xyz,
        empty_distance_mm=empty_distance,
        empty_time_seconds=empty_time,
        loaded_distance_mm=loaded_distance,
        loaded_time_seconds=loaded_time,
        rotation_delta_deg=rotation_delta,
        rotation_time_seconds=rotation_time,
        edge_cost_seconds=edge_cost,
        category_index_by_target=category_index_by_target,
        sources_by_category=tuple(
            tuple(indices) for indices in sources_by_category_lists
        ),
    )


def _predecessor_array_index(previous_target: int, tables: MotionCostTables) -> int:
    if previous_target == START_PREDECESSOR:
        return tables.start_predecessor_index
    return int(previous_target)


def _extend_category_dp(
    old_dp: Tuple[float, ...],
    source_indices: Tuple[int, ...],
    predecessor_index: int,
    target_index: int,
    tables: MotionCostTables,
) -> Tuple[Tuple[float, ...], float]:
    """只更新本次 target 所属类别的至多 32 个 DP 状态。"""
    new_dp = [INFINITE_COST] * len(old_dp)
    full_source_mask = (1 << len(source_indices)) - 1
    for used_mask, old_cost in enumerate(old_dp):
        if not math.isfinite(old_cost):
            continue
        unused_mask = full_source_mask ^ used_mask
        while unused_mask:
            source_bit = unused_mask & -unused_mask
            local_source_index = source_bit.bit_length() - 1
            source_index = source_indices[local_source_index]
            candidate_mask = used_mask | source_bit
            candidate_cost = old_cost + float(
                tables.edge_cost_seconds[
                    predecessor_index,
                    source_index,
                    target_index,
                ]
            )
            if candidate_cost < new_dp[candidate_mask]:
                new_dp[candidate_mask] = candidate_cost
            unused_mask ^= source_bit
    best_cost = min(new_dp)
    return tuple(new_dp), float(best_cost)


class _BoundedBeam:
    """以最大劣项为堆顶，流式保留排序键最小的 K 个节点。"""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._heap = []
        self._counter = 0

    @staticmethod
    def _reverse_sequence(sequence: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(-value for value in sequence)

    def add(self, node: _BeamNode) -> None:
        self._counter += 1
        item = (
            -float(node.prefix_score),
            self._reverse_sequence(node.target_sequence),
            self._counter,
            node,
        )
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, item)
            return
        worst = self._heap[0]
        worst_key = (
            -float(worst[0]),
            self._reverse_sequence(worst[1]),
        )
        candidate_key = (float(node.prefix_score), node.target_sequence)
        if candidate_key < worst_key:
            heapq.heapreplace(self._heap, item)

    def sorted_nodes(self) -> List[_BeamNode]:
        return sorted(
            (item[3] for item in self._heap),
            key=lambda node: (node.prefix_score, node.target_sequence),
        )

    def __len__(self) -> int:
        return len(self._heap)


def run_target_beam_search(
    tables: MotionCostTables,
    beam_width: int,
) -> Tuple[Tuple[_BeamNode, ...], BeamSearchStatistics]:
    """Python 参考搜索；生产路径使用 C++，本函数保留用于一致性测试。"""
    started_at = time.perf_counter()
    initial_dps = []
    for source_indices in tables.sources_by_category:
        state_count = 1 << len(source_indices)
        initial_dps.append((0.0,) + (INFINITE_COST,) * (state_count - 1))
    root = _BeamNode(
        placed_mask=0,
        available_mask=tables.support_graph.first_layer_mask,
        last_target=START_PREDECESSOR,
        target_sequence=(),
        category_dps=tuple(initial_dps),
        category_best_costs=(0.0,) * len(BLOCK_CATEGORY_NAMES),
        prefix_score=0.0,
    )
    current_layer = [root]
    expanded_parent_count = 0
    generated_child_count = 0
    peak_retained = 1
    all_targets_mask = tables.support_graph.all_targets_mask

    for _depth in range(len(tables.targets)):
        next_layer = _BoundedBeam(beam_width)
        for node in current_layer:
            expanded_parent_count += 1
            available = node.available_mask & ~node.placed_mask
            bits = available
            predecessor_index = _predecessor_array_index(node.last_target, tables)
            while bits:
                target_bit = bits & -bits
                target_index = target_bit.bit_length() - 1
                category_index = tables.category_index_by_target[target_index]
                new_category_dp, new_category_best = _extend_category_dp(
                    node.category_dps[category_index],
                    tables.sources_by_category[category_index],
                    predecessor_index,
                    target_index,
                    tables,
                )
                if math.isfinite(new_category_best):
                    new_dps = list(node.category_dps)
                    new_dps[category_index] = new_category_dp
                    new_best_costs = list(node.category_best_costs)
                    old_category_best = new_best_costs[category_index]
                    new_best_costs[category_index] = new_category_best
                    new_placed = node.placed_mask | target_bit
                    new_available = (
                        node.available_mask
                        | tables.support_graph.unlock_masks[target_index]
                    ) & ~new_placed
                    child = _BeamNode(
                        placed_mask=new_placed,
                        available_mask=new_available,
                        last_target=target_index,
                        target_sequence=node.target_sequence + (target_index,),
                        category_dps=tuple(new_dps),
                        category_best_costs=tuple(new_best_costs),
                        prefix_score=(
                            node.prefix_score
                            - old_category_best
                            + new_category_best
                        ),
                    )
                    next_layer.add(child)
                    generated_child_count += 1
                bits ^= target_bit
        if len(next_layer) == 0:
            raise RuntimeError(
                f"搜索在第 {_depth + 1} 步没有合法目标或可用实体，已终止新规划"
            )
        current_layer = next_layer.sorted_nodes()
        peak_retained = max(peak_retained, len(current_layer))

    final_nodes = tuple(
        node for node in current_layer if node.placed_mask == all_targets_mask
    )
    if not final_nodes:
        raise RuntimeError("Beam Search 未生成完整合法摆放顺序")
    statistics = BeamSearchStatistics(
        elapsed_seconds=time.perf_counter() - started_at,
        expanded_parent_count=expanded_parent_count,
        generated_child_count=generated_child_count,
        peak_retained_node_count=peak_retained,
        final_candidate_count=len(final_nodes),
        returned_candidate_count=len(final_nodes),
    )
    return final_nodes, statistics


def run_target_beam_search_native(
    tables: MotionCostTables,
    beam_width: int,
    returned_candidate_limit: Optional[int] = None,
    library_path=None,
) -> NativeSearchResult:
    """把只读成本表传给 C++，完成 Beam 和所有最终实体回溯。"""
    return run_native_task_sequence_search(
        first_layer_mask=tables.support_graph.first_layer_mask,
        unlock_masks=tables.support_graph.unlock_masks,
        target_categories=tables.category_index_by_target,
        sources_by_category=tables.sources_by_category,
        source_ids=[int(block.source_id) for block in tables.blocks],
        edge_cost_seconds=tables.edge_cost_seconds,
        beam_width=beam_width,
        returned_candidate_limit=returned_candidate_limit,
        library_path=library_path,
    )


def _make_cost_step(
    step_index: int,
    previous_target: int,
    target_index: int,
    source_index: int,
    tables: MotionCostTables,
) -> PlanCostStep:
    predecessor_index = _predecessor_array_index(previous_target, tables)
    target = tables.targets[target_index]
    block = tables.blocks[source_index]
    return PlanCostStep(
        step=step_index + 1,
        predecessor_target_id=(
            None
            if previous_target == START_PREDECESSOR
            else int(tables.targets[previous_target].index)
        ),
        target_dense_index=target_index,
        target_id=int(target.index),
        source_index=source_index,
        source_id=int(block.source_id),
        category=normalize_category_name(target.category),
        empty_distance_mm=float(
            tables.empty_distance_mm[predecessor_index, source_index]
        ),
        loaded_distance_mm=float(
            tables.loaded_distance_mm[source_index, target_index]
        ),
        empty_time_seconds=float(
            tables.empty_time_seconds[predecessor_index, source_index]
        ),
        loaded_time_seconds=float(
            tables.loaded_time_seconds[source_index, target_index]
        ),
        rotation_delta_deg=float(
            tables.rotation_delta_deg[source_index, target_index]
        ),
        rotation_time_seconds=float(
            tables.rotation_time_seconds[source_index, target_index]
        ),
        edge_cost_seconds=float(
            tables.edge_cost_seconds[predecessor_index, source_index, target_index]
        ),
    )


def _assemble_assignment_plan(
    target_sequence: Sequence[int],
    source_sequence: Sequence[int],
    tables: MotionCostTables,
) -> AssignmentPlan:
    validate_dense_target_sequence(target_sequence, tables.support_graph)
    if len(target_sequence) != len(source_sequence):
        raise ValueError("target/source 顺序长度不一致")
    if len(set(int(value) for value in source_sequence)) != len(source_sequence):
        raise ValueError("完整方案不能重复使用实体")
    steps = []
    previous_target = START_PREDECESSOR
    for step_index, (target_index, source_index) in enumerate(
        zip(target_sequence, source_sequence)
    ):
        target_index = int(target_index)
        source_index = int(source_index)
        target_category = normalize_category_name(tables.targets[target_index].category)
        source_category = normalize_category_name(tables.blocks[source_index].category)
        if target_category != source_category:
            raise ValueError(
                f"目标 {tables.targets[target_index].index} 与实体 "
                f"{tables.blocks[source_index].source_id} 类别不匹配"
            )
        steps.append(
            _make_cost_step(
                step_index,
                previous_target,
                target_index,
                source_index,
                tables,
            )
        )
        previous_target = target_index
    used_sources = set(int(value) for value in source_sequence)
    unused_source_indices = tuple(
        index for index in range(len(tables.blocks)) if index not in used_sources
    )
    return AssignmentPlan(
        target_dense_sequence=tuple(int(value) for value in target_sequence),
        source_indices=tuple(int(value) for value in source_sequence),
        target_id_sequence=tuple(
            int(tables.targets[index].index) for index in target_sequence
        ),
        source_id_sequence=tuple(
            int(tables.blocks[index].source_id) for index in source_sequence
        ),
        unused_source_indices=unused_source_indices,
        unused_source_ids=tuple(
            int(tables.blocks[index].source_id) for index in unused_source_indices
        ),
        simplified_cost_seconds=float(sum(step.edge_cost_seconds for step in steps)),
        steps=tuple(steps),
    )


def evaluate_target_sequence(
    target_sequence: Sequence[int],
    tables: MotionCostTables,
) -> AssignmentPlan:
    """对固定 target 顺序运行一次可回溯的每类 bitmask DP。"""
    sequence = tuple(int(value) for value in target_sequence)
    validate_dense_target_sequence(sequence, tables.support_graph)
    source_by_step = [None] * len(sequence)
    slots_by_category = [[] for _ in BLOCK_CATEGORY_NAMES]
    previous_target = START_PREDECESSOR
    for step_index, target_index in enumerate(sequence):
        category_index = tables.category_index_by_target[target_index]
        slots_by_category[category_index].append(
            (step_index, previous_target, target_index)
        )
        previous_target = target_index

    for category_index, slots in enumerate(slots_by_category):
        if not slots:
            continue
        local_sources = tables.sources_by_category[category_index]
        # 值为 (成本, 按该类别出现顺序排列的 source_id, source 全局下标序列)。
        states = {0: (0.0, (), ())}
        for _step_index, previous_target, target_index in slots:
            predecessor_index = _predecessor_array_index(previous_target, tables)
            new_states = {}
            full_mask = (1 << len(local_sources)) - 1
            for used_mask, (old_cost, old_ids, old_sources) in states.items():
                unused_mask = full_mask ^ used_mask
                while unused_mask:
                    source_bit = unused_mask & -unused_mask
                    local_source_index = source_bit.bit_length() - 1
                    source_index = local_sources[local_source_index]
                    new_mask = used_mask | source_bit
                    new_cost = old_cost + float(
                        tables.edge_cost_seconds[
                            predecessor_index,
                            source_index,
                            target_index,
                        ]
                    )
                    new_ids = old_ids + (int(tables.blocks[source_index].source_id),)
                    new_sources = old_sources + (source_index,)
                    previous = new_states.get(new_mask)
                    if previous is None or (new_cost, new_ids) < (
                        previous[0],
                        previous[1],
                    ):
                        new_states[new_mask] = (new_cost, new_ids, new_sources)
                    unused_mask ^= source_bit
            states = new_states
        if not states:
            raise RuntimeError(
                f"{BLOCK_CATEGORY_NAMES[category_index]} 固定顺序 assignment 无解"
            )
        _, _, selected_sources = min(
            states.values(),
            key=lambda value: (value[0], value[1]),
        )
        for (step_index, _previous, _target), source_index in zip(
            slots,
            selected_sources,
        ):
            source_by_step[step_index] = source_index
    if any(source is None for source in source_by_step):
        raise RuntimeError("固定顺序 assignment 回溯结果不完整")
    return _assemble_assignment_plan(sequence, source_by_step, tables)


def evaluate_explicit_assignment(
    target_id_sequence: Sequence[int],
    source_id_sequence: Sequence[int],
    tables: MotionCostTables,
) -> AssignmentPlan:
    """按明确 target/source ID 对评估旧算法方案，不重新分配。"""
    target_lookup = {
        int(target.index): index for index, target in enumerate(tables.targets)
    }
    source_lookup = {
        int(block.source_id): index for index, block in enumerate(tables.blocks)
    }
    try:
        target_sequence = tuple(target_lookup[int(value)] for value in target_id_sequence)
        source_sequence = tuple(source_lookup[int(value)] for value in source_id_sequence)
    except KeyError as exc:
        raise ValueError(f"旧方案包含未知 target/source ID：{exc.args[0]}") from exc
    return _assemble_assignment_plan(target_sequence, source_sequence, tables)


def replay_servo_plan(
    plan: AssignmentPlan,
    tables: MotionCostTables,
    config: TaskSequenceOptimizerConfig,
) -> ServoReplayResult:
    """以真实绝对角越界逻辑重放方案；只统计，不参与 V1 选优。"""
    current_angle = float(config.initial_motor_angle_deg)
    replay_steps = []
    covered_total = 0.0
    naked_wait_total = 0.0
    replay_total = 0.0
    pre_rotation_count = 0
    velocity = float(config.motor_velocity_deg_per_sec)
    for cost_step in plan.steps:
        transition = plan_servo_angle_transition(
            current_angle,
            cost_step.rotation_delta_deg,
            config.motor_lower_margin_deg,
            config.motor_upper_margin_deg,
        )
        pre_rotation_seconds = transition.pre_pick_rotation_deg / velocity
        loaded_rotation_seconds = transition.loaded_rotation_deg / velocity
        if transition.pre_pick_target_angle_deg is not None:
            pre_rotation_count += 1
        empty_covered = min(cost_step.empty_time_seconds, pre_rotation_seconds)
        loaded_covered = min(cost_step.loaded_time_seconds, loaded_rotation_seconds)
        covered = empty_covered + loaded_covered
        naked_wait = (
            max(0.0, pre_rotation_seconds - cost_step.empty_time_seconds)
            + max(0.0, loaded_rotation_seconds - cost_step.loaded_time_seconds)
        )
        replay_step_seconds = (
            max(cost_step.empty_time_seconds, pre_rotation_seconds)
            + max(cost_step.loaded_time_seconds, loaded_rotation_seconds)
        )
        replay_steps.append(ServoReplayStep(
            step=cost_step.step,
            source_id=cost_step.source_id,
            target_id=cost_step.target_id,
            start_angle_deg=transition.start_angle_deg,
            pick_angle_deg=transition.pick_angle_deg,
            place_angle_deg=transition.place_angle_deg,
            pre_pick_target_angle_deg=transition.pre_pick_target_angle_deg,
            pre_pick_rotation_seconds=pre_rotation_seconds,
            loaded_rotation_seconds=loaded_rotation_seconds,
            empty_motion_seconds=cost_step.empty_time_seconds,
            loaded_motion_seconds=cost_step.loaded_time_seconds,
            servo_covered_seconds=covered,
            servo_naked_wait_seconds=naked_wait,
            replay_step_seconds=replay_step_seconds,
        ))
        covered_total += covered
        naked_wait_total += naked_wait
        replay_total += replay_step_seconds
        current_angle = transition.place_angle_deg
    return ServoReplayResult(
        pre_rotation_count=pre_rotation_count,
        servo_covered_seconds=covered_total,
        servo_naked_wait_seconds=naked_wait_total,
        replay_total_seconds=replay_total,
        final_angle_deg=current_angle,
        steps=tuple(replay_steps),
    )


def _legacy_plan_from_tasks(
    legacy_tasks: Optional[Sequence[TaskTarget]],
    tables: MotionCostTables,
) -> Optional[AssignmentPlan]:
    if legacy_tasks is None:
        return None
    return evaluate_explicit_assignment(
        [task.index for task in legacy_tasks],
        [task.source_id for task in legacy_tasks],
        tables,
    )


def optimize_task_sequence(
    blocks: Sequence[ObservedBlock],
    targets: Sequence[PlacementTarget],
    board_angle_deg: float,
    config: TaskSequenceOptimizerConfig,
    motion_model: ArmMotionTimeModel,
    legacy_tasks: Optional[Sequence[TaskTarget]] = None,
    native_library_path=None,
) -> TaskPlanResult:
    """执行 V1 搜索，并保证最终简化成本不劣于固定 target 顺序。"""
    tables = build_motion_cost_tables(
        blocks,
        targets,
        board_angle_deg,
        config,
        motion_model,
    )
    fixed_sequence = tuple(range(len(tables.targets)))
    fixed_plan = evaluate_target_sequence(fixed_sequence, tables)
    fixed_replay = replay_servo_plan(fixed_plan, tables, config)

    native_result = run_target_beam_search_native(
        tables,
        int(config.beam_width),
        returned_candidate_limit=config.returned_candidate_limit,
        library_path=native_library_path,
    )
    native_statistics = native_result.statistics
    statistics = BeamSearchStatistics(
        elapsed_seconds=native_statistics.beam_search_seconds,
        expanded_parent_count=native_statistics.expanded_parent_count,
        generated_child_count=native_statistics.generated_child_count,
        peak_retained_node_count=native_statistics.peak_retained_node_count,
        final_candidate_count=native_statistics.final_candidate_count,
        returned_candidate_count=native_statistics.returned_candidate_count,
        backend="cpp_native",
        source_assignment_elapsed_seconds=(
            native_statistics.source_assignment_seconds
        ),
        native_call_elapsed_seconds=native_statistics.native_call_seconds,
        candidate_conversion_elapsed_seconds=(
            native_statistics.candidate_conversion_seconds
        ),
    )
    postprocessing_started_at = time.perf_counter()
    beam_candidates = []
    for native_candidate in native_result.candidates:
        plan = _assemble_assignment_plan(
            native_candidate.target_sequence,
            native_candidate.source_sequence,
            tables,
        )
        if (
            abs(
                plan.simplified_cost_seconds
                - native_candidate.assignment_score
            )
            > _COST_TOLERANCE
        ):
            raise RuntimeError(
                "C++ source assignment 与 Python 明细成本不一致："
                f"{native_candidate.assignment_score:.12f} != "
                f"{plan.simplified_cost_seconds:.12f}"
            )
        if (
            abs(plan.simplified_cost_seconds - native_candidate.prefix_score)
            > _COST_TOLERANCE
        ):
            raise RuntimeError(
                "C++ Beam 增量 DP 与最终回溯 DP 成本不一致："
                f"{native_candidate.prefix_score:.12f} != "
                f"{plan.simplified_cost_seconds:.12f}"
            )
        beam_candidates.append(
            BeamCandidateResult(
                plan=plan,
                servo_replay=replay_servo_plan(plan, tables, config),
            )
        )
    beam_candidates.sort(key=lambda item: (
        item.plan.simplified_cost_seconds,
        item.plan.target_id_sequence,
        item.plan.source_id_sequence,
    ))
    beam_best = beam_candidates[0]

    # 束剪枝可能丢失固定顺序；生产选中结果显式取二者较优，绝不倒退。
    selected_origin = "beam"
    selected_plan = beam_best.plan
    selected_replay = beam_best.servo_replay
    if (
        fixed_plan.simplified_cost_seconds,
        fixed_plan.target_id_sequence,
        fixed_plan.source_id_sequence,
    ) < (
        selected_plan.simplified_cost_seconds,
        selected_plan.target_id_sequence,
        selected_plan.source_id_sequence,
    ):
        selected_origin = "fixed_order_fallback"
        selected_plan = fixed_plan
        selected_replay = fixed_replay

    true_replay_best_index = min(
        range(len(beam_candidates)),
        key=lambda index: (
            beam_candidates[index].servo_replay.replay_total_seconds,
            beam_candidates[index].plan.target_id_sequence,
            beam_candidates[index].plan.source_id_sequence,
        ),
    )
    simplified_and_replay_differ = true_replay_best_index != 0
    legacy_plan = _legacy_plan_from_tasks(legacy_tasks, tables)
    legacy_replay = (
        replay_servo_plan(legacy_plan, tables, config)
        if legacy_plan is not None
        else None
    )
    tasks = tuple(
        make_task_target(
            tables.blocks[source_index],
            tables.targets[target_index],
            board_angle_deg,
        )
        for target_index, source_index in zip(
            selected_plan.target_dense_sequence,
            selected_plan.source_indices,
        )
    )
    statistics = replace(
        statistics,
        python_postprocessing_elapsed_seconds=(
            time.perf_counter() - postprocessing_started_at
        ),
    )
    return TaskPlanResult(
        tasks=tasks,
        selected_origin=selected_origin,
        selected_plan=selected_plan,
        selected_servo_replay=selected_replay,
        fixed_order_plan=fixed_plan,
        fixed_order_servo_replay=fixed_replay,
        beam_best_plan=beam_best.plan,
        beam_best_servo_replay=beam_best.servo_replay,
        legacy_plan=legacy_plan,
        legacy_servo_replay=legacy_replay,
        final_beam_candidates=tuple(beam_candidates),
        true_replay_best_candidate_index=true_replay_best_index,
        simplified_and_replay_ranking_differ=simplified_and_replay_differ,
        statistics=statistics,
        cost_tables=tables,
        report_top_candidates=int(config.report_top_candidates),
    )


def _cost_step_document(step: PlanCostStep) -> dict:
    return {
        "步骤": step.step,
        "前一目标ID": step.predecessor_target_id,
        "目标ID": step.target_id,
        "实体ID": step.source_id,
        "类别": step.category,
        "空载XYZ距离毫米": step.empty_distance_mm,
        "负载XYZ距离毫米": step.loaded_distance_mm,
        "空载时间秒": step.empty_time_seconds,
        "负载时间秒": step.loaded_time_seconds,
        "相对旋转角度": step.rotation_delta_deg,
        "相对旋转时间秒": step.rotation_time_seconds,
        "简化边成本秒": step.edge_cost_seconds,
    }


def _replay_step_document(step: ServoReplayStep) -> dict:
    return {
        "步骤": step.step,
        "实体ID": step.source_id,
        "目标ID": step.target_id,
        "起始绝对角度": step.start_angle_deg,
        "抓取绝对角度": step.pick_angle_deg,
        "摆放绝对角度": step.place_angle_deg,
        "抓前预旋转目标角度": step.pre_pick_target_angle_deg,
        "抓前预旋转时间秒": step.pre_pick_rotation_seconds,
        "负载旋转时间秒": step.loaded_rotation_seconds,
        "舵机被机械臂覆盖时间秒": step.servo_covered_seconds,
        "舵机裸等待秒": step.servo_naked_wait_seconds,
        "重放单步时间秒": step.replay_step_seconds,
    }


def _plan_document(
    plan: AssignmentPlan,
    replay: ServoReplayResult,
    expand_steps: bool,
) -> dict:
    document = {
        "目标序列": list(plan.target_id_sequence),
        "实体序列": list(plan.source_id_sequence),
        "未使用实体": list(plan.unused_source_ids),
        "简化成本秒": plan.simplified_cost_seconds,
        "舵机预旋转次数": replay.pre_rotation_count,
        "舵机覆盖时间秒": replay.servo_covered_seconds,
        "舵机裸等待秒": replay.servo_naked_wait_seconds,
        "真实逻辑重放总时间秒": replay.replay_total_seconds,
        "重放结束绝对角度": replay.final_angle_deg,
    }
    if expand_steps:
        document["简化成本逐步明细"] = [
            _cost_step_document(step) for step in plan.steps
        ]
        document["舵机重放逐步明细"] = [
            _replay_step_document(step) for step in replay.steps
        ]
    return document


def build_task_plan_report(result: TaskPlanResult) -> dict:
    """生成可直接写为 UTF-8 JSON 的中文规划报告。"""
    tables = result.cost_tables
    top_count = len(result.final_beam_candidates)
    # 全部候选都经过重放并保留摘要，只为配置指定的前 N 条展开逐步明细。
    expanded_limit = min(result.report_top_candidates, top_count)
    candidates = []
    for index, candidate in enumerate(result.final_beam_candidates):
        candidates.append({
            "候选排名": index + 1,
            **_plan_document(
                candidate.plan,
                candidate.servo_replay,
                expand_steps=index < expanded_limit,
            ),
        })
    legacy_document = None
    if result.legacy_plan is not None and result.legacy_servo_replay is not None:
        legacy_document = _plan_document(
            result.legacy_plan,
            result.legacy_servo_replay,
            expand_steps=True,
        )
    return {
        "协议版本": 1,
        "生成时间": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "V1边界": {
            "搜索维护舵机跨步绝对角状态": False,
            "最终候选使用真实舵机边界逻辑重放": True,
            "真实舵机重放改变V1选择": False,
            "统一下探抬升吸吹气计入": False,
        },
        "数量": {
            "实体数": len(tables.blocks),
            "目标数": len(tables.targets),
            "最终Beam候选数": len(result.final_beam_candidates),
        },
        "实体": [
            {
                "实体ID": int(block.source_id),
                "类别": normalize_category_name(block.category),
                "观察位": [float(value) for value in block.observation_pose],
                "识别角度": float(block.detected_angle_deg),
                "抓取表面Z毫米": float(block.pick_surface_z_mm),
            }
            for block in tables.blocks
        ],
        "目标": [
            {
                "目标ID": int(target.index),
                "类别": normalize_category_name(target.category),
                "行": float(target.row),
                "列": float(target.col),
                "期望角度": float(target.desired_angle_deg),
                "占用格": [[int(col), int(row)] for col, row in target.cells],
                "观察位": [float(value) for value in target.observation_pose],
            }
            for target in tables.targets
        ],
        "对照方案": {
            "当前旧匈牙利": legacy_document,
            "固定目标顺序加bitmask_DP": _plan_document(
                result.fixed_order_plan,
                result.fixed_order_servo_replay,
                expand_steps=True,
            ),
            "合法顺序Beam加bitmask_DP": _plan_document(
                result.beam_best_plan,
                result.beam_best_servo_replay,
                expand_steps=True,
            ),
        },
        "V1选中方案来源": result.selected_origin,
        "V1选中方案": _plan_document(
            result.selected_plan,
            result.selected_servo_replay,
            expand_steps=True,
        ),
        "简化第一与真实重放第一不同": result.simplified_and_replay_ranking_differ,
        "真实重放第一候选排名": result.true_replay_best_candidate_index + 1,
        "Beam性能": {
            "搜索后端": result.statistics.backend,
            "搜索耗时秒": result.statistics.elapsed_seconds,
            "实体回溯耗时秒": (
                result.statistics.source_assignment_elapsed_seconds
            ),
            "C++调用总耗时秒": result.statistics.native_call_elapsed_seconds,
            "候选跨语言转换耗时秒": (
                result.statistics.candidate_conversion_elapsed_seconds
            ),
            "Python后处理耗时秒": (
                result.statistics.python_postprocessing_elapsed_seconds
            ),
            "展开父节点数": result.statistics.expanded_parent_count,
            "生成子节点数": result.statistics.generated_child_count,
            "峰值保留节点数": result.statistics.peak_retained_node_count,
            # 保留 V1 旧字段，方便已有离线脚本继续读取。
            "最终候选数": result.statistics.final_candidate_count,
            "最终Beam保留数": result.statistics.final_candidate_count,
            "实际返回数": result.statistics.returned_candidate_count,
        },
        "最终Beam候选": candidates,
    }
