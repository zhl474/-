import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

import image_process_lib.task_planner as task_planner_module
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
    validate_motion_model_speeds,
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
