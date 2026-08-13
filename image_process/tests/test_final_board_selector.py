"""V5 最终盘面比较、失败隔离和唯一指纹测试。"""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from image_process_lib.board_candidate_selector import (
    BoardCandidateSelectionResult,
    FinalBoardCandidate,
)
from image_process_lib.final_board_selector import (
    DynamicBoardSelectionError,
    FinalBoardSelector,
    FinalBoardSelectorConfig,
    build_unique_board_manifest,
)
from image_process_lib.task_planner import ObservedBlock, PlacementTarget
from image_process_lib.task_sequence_optimizer import (
    TaskSequenceOptimizerConfig,
    optimize_task_sequence,
)
from image_process_lib.v5_board_library import load_v5_board_library


LIBRARY_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "v5_board_library_v1.npz"
)


class 线性时间模型:
    """测试用的确定性批量路程时间模型。"""

    def predict_array_seconds(self, distances_mm):
        return np.asarray(distances_mm, dtype=np.float64) / 200.0


def _optimizer_config():
    return TaskSequenceOptimizerConfig(
        shooting_pose=(0.0, 0.0, 300.0, -180.0, 0.0, 90.0),
        camera_to_sucker_offset_mm=(0.0, 0.0),
        pick_surface_offset_mm=100.0,
        pick_approach_clearance_mm=5.0,
        motor_velocity_deg_per_sec=90.0,
        initial_motor_angle_deg=180.0,
        motor_lower_margin_deg=10.0,
        motor_upper_margin_deg=350.0,
        beam_width=10,
        report_top_candidates=2,
    )


@pytest.fixture(scope="module")
def library():
    if not LIBRARY_PATH.exists():
        pytest.skip("当前工作区没有部署用 V5 NPZ")
    return load_v5_board_library(LIBRARY_PATH)


def _candidate(library, board_index, score=(0, 0, 0.0, 0.0)):
    return FinalBoardCandidate(
        board_index=board_index,
        board_id=library.board_id(board_index),
        global_id=int(library.board_global_id[board_index]),
        layout_index=int(library.board_layout_index[board_index]),
        missing_category=library.missing_category_name(board_index),
        score=score,
        coarse_assignment=(),
        relaxed_assignment=(),
        coarse_unused_source_id=-1,
        unused_source_id=-1,
    )


def _result(library, candidates):
    return BoardCandidateSelectionResult(
        total_board_count=library.board_count,
        coarse_candidate_count=len(candidates),
        candidates=tuple(candidates),
        coarse_elapsed_seconds=0.01,
        relaxed_elapsed_seconds=0.02,
    )


def _board_pids(library, board_index):
    return tuple(sorted(
        int(pid)
        for pid in library.board_target_pid[board_index].flat
        if int(pid) >= 0
    ))


def _placement_targets(library, board_indices):
    pids = sorted({
        pid
        for board_index in board_indices
        for pid in _board_pids(library, board_index)
    })
    return {
        pid: PlacementTarget(
            index=pid,
            row=float(library.placement_row[pid]),
            col=float(library.placement_col[pid]),
            desired_angle_deg=float(library.placement_yaw_clockwise_deg[pid]),
            category=library.category_names[int(library.placement_category[pid])],
            observation_pose=(
                float(library.placement_col[pid]) * 10.0,
                float(library.placement_row[pid]) * 10.0,
                200.0,
                -180.0,
                0.0,
                90.0,
            ),
            cells=tuple(
                tuple(int(value) for value in cell)
                for cell in library.placement_cells[pid]
            ),
        )
        for pid in pids
    }


def _observed_blocks(library):
    blocks = []
    for category_index, category in enumerate(library.category_names):
        for local_index in range(5):
            blocks.append(ObservedBlock(
                category=category,
                observation_pose=(
                    category_index * 50.0 + local_index * 5.0,
                    local_index * 10.0,
                    250.0,
                    -180.0,
                    0.0,
                    90.0,
                ),
                detected_angle_deg=0.0,
                source_id=category_index * 5 + local_index,
                pick_surface_z_mm=50.0,
                pick_surface_z_valid=True,
            ))
    return tuple(blocks)


def _selector_config():
    return FinalBoardSelectorConfig(
        comparison_beam_width=10,
        comparison_returned_candidates=1,
        confirmation_beam_width=20,
        confirmation_returned_candidates=2,
        soft_time_budget_sec=10.0,
    )


def test_final_selector_locks_34_unique_pids_sources_and_manifest(library):
    candidate = _candidate(library, 0)
    selector = FinalBoardSelector(library, _selector_config())
    blocks = _observed_blocks(library)

    decision = selector.select(
        _result(library, (candidate,)),
        blocks,
        _placement_targets(library, (0,)),
        board_angle_deg=0.0,
        optimizer_config=_optimizer_config(),
        motion_model=线性时间模型(),
    )

    assert len(decision.placement_pids) == 34
    assert len(set(decision.placement_pids)) == 34
    assert len(decision.final_pid_sequence) == 34
    assert set(decision.final_pid_sequence) == set(decision.placement_pids)
    assert len({source_id for source_id, _ in decision.final_source_to_pid}) == 34
    unused_category = next(
        block.category for block in blocks if block.source_id == decision.unused_source_id
    )
    assert unused_category == decision.missing_category
    assert len(decision.library_sha256) == 64
    assert len(decision.placement_pid_sha256) == 64
    assert len(decision.decision_fingerprint) == 64
    manifest = build_unique_board_manifest(decision)
    assert manifest["final_pid_sequence"] == list(decision.final_pid_sequence)
    assert manifest["decision_fingerprint"] == decision.decision_fingerprint
    json.dumps(manifest, ensure_ascii=False)


def test_comparison_failure_is_recorded_and_other_board_continues(library):
    candidates = (_candidate(library, 0), _candidate(library, 1))
    failed_pid_set = set(_board_pids(library, 0))

    def optimizer_with_one_failure(
        blocks,
        targets,
        board_angle_deg,
        config,
        motion_model,
        **kwargs,
    ):
        if config.beam_width == 10 and {target.index for target in targets} == failed_pid_set:
            raise RuntimeError("注入的比较失败")
        return optimize_task_sequence(
            blocks,
            targets,
            board_angle_deg,
            config,
            motion_model,
            **kwargs,
        )

    selector = FinalBoardSelector(
        library,
        _selector_config(),
        optimizer_callable=optimizer_with_one_failure,
    )
    decision = selector.select(
        _result(library, candidates),
        _observed_blocks(library),
        _placement_targets(library, (0, 1)),
        0.0,
        _optimizer_config(),
        线性时间模型(),
    )

    assert decision.board_index == 1
    assert not decision.comparison_attempts[0].succeeded
    assert "注入的比较失败" in decision.comparison_attempts[0].error_message
    assert decision.comparison_attempts[1].succeeded


def test_exact_comparison_tie_uses_stable_board_index_not_input_order(library):
    candidates = (_candidate(library, 1), _candidate(library, 0))

    def optimizer_with_comparison_tie(
        blocks,
        targets,
        board_angle_deg,
        config,
        motion_model,
        **kwargs,
    ):
        result = optimize_task_sequence(
            blocks,
            targets,
            board_angle_deg,
            config,
            motion_model,
            **kwargs,
        )
        if config.beam_width == 10:
            return replace(
                result,
                selected_plan=replace(result.selected_plan, simplified_cost_seconds=100.0),
                selected_servo_replay=replace(
                    result.selected_servo_replay,
                    replay_total_seconds=120.0,
                ),
            )
        return result

    selector = FinalBoardSelector(
        library,
        _selector_config(),
        optimizer_callable=optimizer_with_comparison_tie,
    )
    decision = selector.select(
        _result(library, candidates),
        _observed_blocks(library),
        _placement_targets(library, (0, 1)),
        0.0,
        _optimizer_config(),
        线性时间模型(),
    )

    assert decision.board_index == 0
    assert decision.comparison_attempt.ranking_key[-1] == 0


def test_all_comparison_failures_raise_error_with_attempts(library):
    def always_fail(*_args, **_kwargs):
        raise RuntimeError("全部失败")

    candidate = _candidate(library, 0)
    selector = FinalBoardSelector(
        library,
        _selector_config(),
        optimizer_callable=always_fail,
    )
    with pytest.raises(DynamicBoardSelectionError, match="全部失败") as caught:
        selector.select(
            _result(library, (candidate,)),
            _observed_blocks(library),
            _placement_targets(library, (0,)),
            0.0,
            _optimizer_config(),
            线性时间模型(),
        )
    assert len(caught.value.attempts) == 1
    assert not caught.value.attempts[0].succeeded
