import itertools
import json
from ctypes import POINTER, c_double, c_int32, c_uint64, create_string_buffer
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

import image_process_lib.task_planner as task_planner_module
import image_process_lib.task_sequence_optimizer_native as native_optimizer_module
from image_process_lib.servo_angle_model import (
    plan_servo_angle_transition,
    worst_case_servo_reset_seconds,
)
from image_process_lib.task_geometry import validate_dense_target_sequence
from image_process_lib.task_planner import ObservedBlock, PlacementTarget
from image_process_lib.task_sequence_optimizer import (
    TaskSequenceOptimizerConfig,
    build_motion_cost_tables,
    build_task_plan_report,
    decide_optimizer_mode,
    evaluate_target_sequence,
    optimize_task_sequence,
    run_target_beam_search,
    run_target_beam_search_native,
    validate_motion_model_speeds,
)
from image_process_lib.task_sequence_optimizer_native import (
    run_native_task_sequence_search,
)


class 线性批量时间模型:
    move_speed_percent = 100.0

    def __init__(self):
        self.calls = 0
        self.forbid_calls = False

    def predict_array_seconds(self, distances_mm):
        if self.forbid_calls:
            raise AssertionError("Beam 热循环不应再次调用时间模型")
        self.calls += 1
        return np.asarray(distances_mm, dtype=float) / 200.0


def _pose(x, y, z=200.0):
    return (float(x), float(y), float(z), -180.0, 0.0, 90.0)


def _config(beam_width=1000, report_top_candidates=20):
    return TaskSequenceOptimizerConfig(
        shooting_pose=_pose(0, 0, 300),
        camera_to_sucker_offset_mm=(5.0, -7.0),
        pick_surface_offset_mm=100.0,
        pick_approach_clearance_mm=5.0,
        motor_velocity_deg_per_sec=90.0,
        initial_motor_angle_deg=180.0,
        motor_lower_margin_deg=10.0,
        motor_upper_margin_deg=350.0,
        beam_width=beam_width,
        report_top_candidates=report_top_candidates,
    )


def _first_layer_cells(col):
    return ((col, 1), (col, 2), (col, 3), (col, 4))


def _block(source_id, category, x, y, angle=0.0, surface_z=50.0):
    return ObservedBlock(
        category=category,
        observation_pose=_pose(x, y, 260),
        detected_angle_deg=angle,
        source_id=source_id,
        pick_surface_z_mm=surface_z,
        pick_surface_z_valid=True,
    )


def _target(index, category, x, y, cells, angle=0.0, z=180.0):
    return PlacementTarget(
        index=index,
        row=1.0,
        col=float(index + 1),
        desired_angle_deg=angle,
        category=category,
        observation_pose=_pose(x, y, z),
        cells=cells,
    )


def test_xyz_waypoints_and_edge_cost_follow_v1_formula():
    block = _block(7, "T", 10, 20, surface_z=50)
    target = _target(4, "T", 40, 50, _first_layer_cells(1), z=180)
    model = 线性批量时间模型()

    tables = build_motion_cost_tables(
        [block],
        [target],
        board_angle_deg=0.0,
        config=_config(),
        motion_model=model,
    )

    np.testing.assert_allclose(tables.source_pre_pick_xyz[0], [15, 13, 155])
    np.testing.assert_allclose(tables.source_pick_xyz[0], [15, 13, 150])
    np.testing.assert_allclose(tables.target_place_xyz[0], [45, 43, 180])
    np.testing.assert_allclose(tables.start_xyz, [0, 0, 300])
    empty_distance = np.linalg.norm(np.array([0, 0, 300]) - np.array([15, 13, 155]))
    loaded_distance = np.linalg.norm(np.array([15, 13, 150]) - np.array([45, 43, 180]))
    assert tables.empty_distance_mm[1, 0] == pytest.approx(empty_distance)
    assert tables.loaded_distance_mm[0, 0] == pytest.approx(loaded_distance)
    assert tables.edge_cost_seconds[1, 0, 0] == pytest.approx(
        empty_distance / 200.0 + loaded_distance / 200.0
    )
    assert model.calls == 1


@pytest.mark.parametrize("seed", range(8))
def test_rectangular_bitmask_dp_matches_bruteforce_and_hungarian(seed):
    rng = np.random.default_rng(seed)
    blocks = [
        _block(
            source_id,
            "T",
            *rng.uniform(-100, 100, size=2),
            angle=float(rng.uniform(-180, 180)),
        )
        for source_id in (10, 20, 30, 40)
    ]
    targets = [
        _target(
            index,
            "T",
            *rng.uniform(-100, 100, size=2),
            cells=_first_layer_cells(index + 1),
            angle=float(rng.uniform(-180, 180)),
        )
        for index in range(3)
    ]
    tables = build_motion_cost_tables(
        blocks,
        targets,
        board_angle_deg=13.0,
        config=_config(),
        motion_model=线性批量时间模型(),
    )
    sequence = (2, 0, 1)

    plan = evaluate_target_sequence(sequence, tables)
    matrix = np.empty((4, 3), dtype=float)
    previous = -1
    for slot, target_index in enumerate(sequence):
        predecessor = tables.start_predecessor_index if previous < 0 else previous
        matrix[:, slot] = tables.edge_cost_seconds[predecessor, :, target_index]
        previous = target_index
    rows, cols = linear_sum_assignment(matrix)
    hungarian_cost = float(matrix[rows, cols].sum())
    brute_cost, brute_sources = min(
        (
            sum(matrix[source, slot] for slot, source in enumerate(sources)),
            sources,
        )
        for sources in itertools.permutations(range(4), 3)
    )

    assert plan.simplified_cost_seconds == pytest.approx(hungarian_cost)
    assert plan.simplified_cost_seconds == pytest.approx(brute_cost)
    assert plan.source_indices == brute_sources
    assert len(plan.unused_source_ids) == 1


def _small_supported_problem():
    targets = [
        _target(0, "T", -80, 90, ((1, 1), (1, 2), (2, 1), (2, 2)), 0),
        _target(1, "line", 80, 90, ((3, 1), (3, 2), (4, 1), (4, 2)), 90),
        _target(2, "T", -20, 130, ((2, 3), (2, 4), (3, 3), (3, 4)), 90),
        _target(3, "line", 100, 150, ((5, 1), (5, 2), (5, 3), (5, 4)), 0),
    ]
    blocks = [
        _block(0, "T", -90, -80, -20),
        _block(1, "T", 20, -70, 60),
        _block(2, "line", 100, -60, 10),
        _block(3, "line", -10, -90, 80),
    ]
    return blocks, targets


def test_wide_beam_matches_global_exhaustive_optimum_and_is_deterministic():
    blocks, targets = _small_supported_problem()
    model = 线性批量时间模型()
    config = _config(beam_width=100)
    tables = build_motion_cost_tables(blocks, targets, 0.0, config, model)
    exhaustive = []
    for sequence in itertools.permutations(range(len(targets))):
        try:
            validate_dense_target_sequence(sequence, tables.support_graph)
        except ValueError:
            continue
        plan = evaluate_target_sequence(sequence, tables)
        exhaustive.append((
            plan.simplified_cost_seconds,
            plan.target_id_sequence,
            plan.source_id_sequence,
        ))
    expected = min(exhaustive)

    first = optimize_task_sequence(blocks, targets, 0.0, config, model)
    second = optimize_task_sequence(blocks, targets, 0.0, config, 线性批量时间模型())
    actual = (
        first.simplified_cost_seconds,
        first.target_sequence,
        first.source_sequence,
    )

    assert actual[0] == pytest.approx(expected[0])
    assert first.target_sequence == expected[1]
    assert first.source_sequence == expected[2]
    assert first.target_sequence == second.target_sequence
    assert first.source_sequence == second.source_sequence
    assert first.simplified_cost_seconds == second.simplified_cost_seconds
    assert len(set(first.target_sequence)) == len(targets)
    assert len(set(first.source_sequence)) == len(targets)
    validate_dense_target_sequence(
        first.selected_plan.target_dense_sequence,
        first.cost_tables.support_graph,
    )
    assert [task.index for task in first.tasks] == list(first.target_sequence)
    assert [task.source_id for task in first.tasks] == list(first.source_sequence)


def test_beam_hot_loop_calls_neither_time_model_nor_hungarian(monkeypatch):
    blocks, targets = _small_supported_problem()
    model = 线性批量时间模型()
    tables = build_motion_cost_tables(blocks, targets, 0.0, _config(), model)
    assert model.calls == 1
    model.forbid_calls = True
    monkeypatch.setattr(
        task_planner_module,
        "linear_sum_assignment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Beam 热循环不应调用匈牙利算法")
        ),
    )

    candidates, stats = run_target_beam_search(tables, beam_width=100)

    assert candidates
    assert stats.generated_child_count > 0
    assert model.calls == 1


@pytest.mark.parametrize("seed", range(8))
def test_cpp_beam_and_assignment_match_python_reference_for_every_candidate(seed):
    rng = np.random.default_rng(seed)
    categories = ("T", "line", "T", "line", "T")
    blocks = [
        _block(
            source_id=100 + source_index * 7,
            category=category,
            x=float(rng.uniform(-150.0, 150.0)),
            y=float(rng.uniform(-150.0, 150.0)),
            angle=float(rng.uniform(-180.0, 180.0)),
        )
        for source_index, category in enumerate(categories)
    ]
    targets = [
        _target(
            index=20 + target_index,
            category=category,
            x=float(rng.uniform(-150.0, 150.0)),
            y=float(rng.uniform(-150.0, 150.0)),
            cells=_first_layer_cells(target_index + 1),
            angle=float(rng.uniform(-180.0, 180.0)),
        )
        for target_index, category in enumerate(categories)
    ]
    tables = build_motion_cost_tables(
        blocks,
        targets,
        11.0,
        _config(beam_width=17),
        线性批量时间模型(),
    )

    python_nodes, python_statistics = run_target_beam_search(tables, 17)
    python_candidates = []
    for node in python_nodes:
        plan = evaluate_target_sequence(node.target_sequence, tables)
        python_candidates.append((
            plan.simplified_cost_seconds,
            plan.target_dense_sequence,
            plan.source_indices,
            node.prefix_score,
        ))
    python_candidates.sort(key=lambda item: (
        item[0],
        item[1],
        tuple(tables.blocks[index].source_id for index in item[2]),
    ))

    native_result = run_target_beam_search_native(tables, 17)
    assert native_result.statistics.expanded_parent_count == (
        python_statistics.expanded_parent_count
    )
    assert native_result.statistics.generated_child_count == (
        python_statistics.generated_child_count
    )
    assert native_result.statistics.peak_retained_node_count == (
        python_statistics.peak_retained_node_count
    )
    assert len(native_result.candidates) == len(python_candidates)
    for python_candidate, native_candidate in zip(
        python_candidates,
        native_result.candidates,
    ):
        assert native_candidate.assignment_score == pytest.approx(
            python_candidate[0], abs=1e-10
        )
        assert native_candidate.target_sequence == python_candidate[1]
        assert native_candidate.source_sequence == python_candidate[2]
        assert native_candidate.prefix_score == pytest.approx(
            python_candidate[3], abs=1e-10
        )


def test_cpp_ties_choose_smaller_target_then_smaller_source_id():
    blocks = [
        _block(30, "T", 0, 0),
        _block(10, "T", 0, 0),
        _block(20, "T", 0, 0),
    ]
    targets = [
        _target(index, "T", 0, 0, _first_layer_cells(index + 1))
        for index in range(3)
    ]
    tables = build_motion_cost_tables(
        blocks,
        targets,
        0.0,
        _config(beam_width=2),
        线性批量时间模型(),
    )
    # 强制所有可行边完全平分，专门验证两级确定性规则。
    tables.edge_cost_seconds.setflags(write=True)
    tables.edge_cost_seconds[:] = 0.0
    tables.edge_cost_seconds.setflags(write=False)

    result = run_target_beam_search_native(tables, 2)

    assert [item.target_sequence for item in result.candidates] == [
        (0, 1, 2),
        (0, 2, 1),
    ]
    assert all(item.source_sequence == (0, 1, 2) for item in result.candidates)


@pytest.mark.parametrize("returned_limit", [1, 20])
def test_native_return_limit_keeps_search_width_and_same_best_candidates(
    returned_limit,
):
    blocks = [
        _block(100 + index, "T", index * 15.0, index * 7.0)
        for index in range(5)
    ]
    targets = [
        _target(
            index,
            "T",
            index * 9.0,
            index * 11.0,
            _first_layer_cells(index + 1),
        )
        for index in range(5)
    ]
    tables = build_motion_cost_tables(
        blocks,
        targets,
        0.0,
        _config(beam_width=30),
        线性批量时间模型(),
    )

    full_result = run_target_beam_search_native(tables, 30)
    limited_result = run_target_beam_search_native(
        tables,
        30,
        returned_candidate_limit=returned_limit,
    )

    assert limited_result.candidates == full_result.candidates[:returned_limit]
    assert limited_result.statistics.final_candidate_count == (
        full_result.statistics.final_candidate_count
    )
    assert limited_result.statistics.returned_candidate_count == returned_limit
    assert full_result.statistics.returned_candidate_count == len(
        full_result.candidates
    )


def test_native_wrapper_rejects_missing_library_and_invalid_inputs(tmp_path):
    valid = {
        "first_layer_mask": 1,
        "unlock_masks": (0,),
        "target_categories": (0,),
        "sources_by_category": ((0,), (), (), (), (), (), ()),
        "source_ids": (1,),
        "edge_cost_seconds": np.zeros((2, 1, 1), dtype=np.float64),
        "beam_width": 1,
    }
    with pytest.raises(RuntimeError, match="找不到.*动态库"):
        run_native_task_sequence_search(
            **valid,
            library_path=tmp_path / "不存在.so",
        )

    invalid_cost = dict(valid)
    invalid_cost["edge_cost_seconds"] = np.full((2, 1, 1), np.nan)
    with pytest.raises(ValueError, match="有限非负"):
        run_native_task_sequence_search(**invalid_cost)

    too_many_sources = dict(valid)
    too_many_sources.update({
        "sources_by_category": ((0, 1, 2, 3, 4, 5), (), (), (), (), (), ()),
        "source_ids": tuple(range(6)),
        "edge_cost_seconds": np.zeros((2, 6, 1)),
    })
    with pytest.raises(ValueError, match="超过 5"):
        run_native_task_sequence_search(**too_many_sources)

    too_wide = dict(valid)
    too_wide["beam_width"] = 50001
    with pytest.raises(ValueError, match=r"\[1, 50000\]"):
        run_native_task_sequence_search(**too_wide)

    too_many_returned = dict(valid)
    too_many_returned["returned_candidate_limit"] = 2
    with pytest.raises(ValueError, match="不能大于 beam_width"):
        run_native_task_sequence_search(**too_many_returned)

    with pytest.raises(ValueError, match=r"\[1, 63\]"):
        run_native_task_sequence_search(
            first_layer_mask=1,
            unlock_masks=(0,) * 64,
            target_categories=(0,) * 64,
            sources_by_category=(tuple(range(64)), (), (), (), (), (), ()),
            source_ids=tuple(range(64)),
            edge_cost_seconds=np.zeros((65, 64, 64)),
            beam_width=1,
        )


def test_native_c_abi_rejects_wrong_version_and_small_output_capacity():
    library_path = native_optimizer_module.resolve_native_library_path()
    library = native_optimizer_module._load_native_library(str(library_path))
    unlock_masks = np.zeros(1, dtype=np.uint64)
    target_categories = np.zeros(1, dtype=np.int32)
    category_source_counts = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    category_sources = np.full((7, 5), -1, dtype=np.int32)
    category_sources[0, 0] = 0
    source_ids = np.array([7], dtype=np.int32)
    edge_costs = np.zeros((2, 1, 1), dtype=np.float64)
    output_count = c_int32()
    output_targets = np.empty((1, 1), dtype=np.int32)
    output_sources = np.empty((1, 1), dtype=np.int32)
    output_prefix = np.empty(1, dtype=np.float64)
    output_assignment = np.empty(1, dtype=np.float64)
    statistics = native_optimizer_module._NativeStatisticsV2()

    def 调用(abi_version, output_capacity, beam_width=1):
        error = create_string_buffer(256)
        return_code = library.task_sequence_optimizer_search_v2(
            abi_version,
            1,
            1,
            beam_width,
            1,
            unlock_masks.ctypes.data_as(POINTER(c_uint64)),
            target_categories.ctypes.data_as(POINTER(c_int32)),
            category_source_counts.ctypes.data_as(POINTER(c_int32)),
            category_sources.ctypes.data_as(POINTER(c_int32)),
            source_ids.ctypes.data_as(POINTER(c_int32)),
            edge_costs.ctypes.data_as(POINTER(c_double)),
            output_capacity,
            output_count,
            output_targets.ctypes.data_as(POINTER(c_int32)),
            output_sources.ctypes.data_as(POINTER(c_int32)),
            output_prefix.ctypes.data_as(POINTER(c_double)),
            output_assignment.ctypes.data_as(POINTER(c_double)),
            statistics,
            error,
            len(error),
        )
        return return_code, error.value.decode("utf-8")

    return_code, error = 调用(999, 1)
    assert return_code != 0
    assert "ABI" in error
    return_code, error = 调用(native_optimizer_module.NATIVE_ABI_VERSION, 0)
    assert return_code != 0
    assert "输出候选容量" in error
    return_code, error = 调用(
        native_optimizer_module.NATIVE_ABI_VERSION,
        1,
        beam_width=2,
    )
    assert return_code == 0, error
    assert statistics.final_candidate_count == 1
    assert statistics.returned_candidate_count == 1


def test_replay_uses_absolute_servo_state_but_does_not_rerank_v1_selection():
    blocks, targets = _small_supported_problem()
    result = optimize_task_sequence(
        blocks,
        targets,
        board_angle_deg=0.0,
        config=_config(beam_width=100, report_top_candidates=1),
        motion_model=线性批量时间模型(),
    )
    expected = min(
        (result.fixed_order_plan, result.beam_best_plan),
        key=lambda plan: (
            plan.simplified_cost_seconds,
            plan.target_id_sequence,
            plan.source_id_sequence,
        ),
    )

    assert result.selected_plan == expected
    assert result.selected_servo_replay.replay_total_seconds >= 0.0
    assert len(result.selected_servo_replay.steps) == len(targets)
    report = build_task_plan_report(result)
    json.dumps(report, ensure_ascii=False)
    assert report["V1边界"]["真实舵机重放改变V1选择"] is False
    assert report["Beam性能"]["搜索后端"] == "cpp_native"
    assert report["Beam性能"]["C++调用总耗时秒"] >= 0.0
    assert "简化成本逐步明细" in report["最终Beam候选"][0]
    if len(report["最终Beam候选"]) > 1:
        assert "简化成本逐步明细" not in report["最终Beam候选"][1]


def test_servo_pure_function_matches_existing_boundary_formula_and_reset_wait():
    upper = plan_servo_angle_transition(350.0, 20.0, 10.0, 350.0)
    lower = plan_servo_angle_transition(10.0, -20.0, 10.0, 350.0)
    direct = plan_servo_angle_transition(180.0, 45.0, 10.0, 350.0)

    assert (upper.pick_angle_deg, upper.place_angle_deg) == (330.0, 350.0)
    assert upper.pre_pick_target_angle_deg == 330.0
    assert (lower.pick_angle_deg, lower.place_angle_deg) == (30.0, 10.0)
    assert lower.pre_pick_target_angle_deg == 30.0
    assert direct.pre_pick_target_angle_deg is None
    assert worst_case_servo_reset_seconds(180.0, 270.0) == pytest.approx(2.0 / 3.0)


def test_optimizer_limits_and_motion_calibration_speed_are_explicit():
    six_blocks = [_block(index, "T", index * 10, 0) for index in range(6)]
    target = _target(0, "T", 0, 0, _first_layer_cells(1))
    with pytest.raises(ValueError, match="5-bit"):
        build_motion_cost_tables(
            six_blocks,
            [target],
            0.0,
            _config(),
            线性批量时间模型(),
        )

    with pytest.raises(ValueError, match="数量不足"):
        build_motion_cost_tables(
            [_block(0, "T", 0, 0)],
            [target, _target(1, "T", 10, 0, _first_layer_cells(2))],
            0.0,
            _config(),
            线性批量时间模型(),
        )
    with pytest.raises(ValueError, match="速度不一致"):
        validate_motion_model_speeds(线性批量时间模型(), 99, 100)


def test_legacy_shadow_execute_modes_have_expected_failure_isolation():
    calls = []
    legacy = ("旧任务",)
    optimized = SimpleNamespace(tasks=("新任务",))

    legacy_decision = decide_optimizer_mode(
        "legacy",
        legacy,
        False,
        lambda: calls.append("不应执行"),
    )
    assert legacy_decision.tasks == legacy
    assert calls == []

    shadow = decide_optimizer_mode("shadow", legacy, False, lambda: optimized)
    execute = decide_optimizer_mode("execute", legacy, False, lambda: optimized)
    failed_shadow = decide_optimizer_mode(
        "shadow",
        legacy,
        False,
        lambda: (_ for _ in ()).throw(RuntimeError("规划失败")),
    )
    assert shadow.tasks == legacy and shadow.plan_result is optimized
    assert execute.tasks == optimized.tasks
    assert failed_shadow.tasks == legacy
    assert failed_shadow.status == "shadow_failed"
    assert failed_shadow.error_message == "规划失败"

    with pytest.raises(RuntimeError, match="只允许.*开环"):
        decide_optimizer_mode("execute", legacy, True, lambda: optimized)
