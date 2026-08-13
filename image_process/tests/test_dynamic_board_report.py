"""新 V5 动态报告的粗筛到唯一盘面精确回放测试。"""

import importlib.util
from pathlib import Path

import numpy as np

from image_process_lib.arm_motion_time import get_default_arm_motion_time_model
from image_process_lib.board_candidate_selector import (
    BoardCandidateSelector,
    BoardCandidateSelectorConfig,
)
from image_process_lib.dynamic_board_report import build_dynamic_board_selection_report
from image_process_lib.dynamic_board_runtime import atomic_write_json, sha256_file
from image_process_lib.final_board_selector import (
    FinalBoardSelector,
    FinalBoardSelectorConfig,
)
from image_process_lib.task_planner import ObservedBlock, PlacementTarget
from image_process_lib.task_sequence_optimizer import TaskSequenceOptimizerConfig
from image_process_lib.v5_board_library import load_v5_board_library


SOURCE_ROOT = Path(__file__).resolve().parents[2]
LIBRARY_PATH = SOURCE_ROOT / "image_process" / "config" / "v5_board_library_v1.npz"
REPLAY_SCRIPT_PATH = SOURCE_ROOT / "tools" / "找最优解" / "回放V5动态盘面选择.py"


def _load_replay_module():
    spec = importlib.util.spec_from_file_location(
        "dynamic_board_replay_test_module",
        REPLAY_SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observed_blocks(library):
    pixels = ((2.0, 2.0), (9.0, 2.0), (2.0, 12.0), (9.0, 12.0), (5.0, 7.0))
    blocks = []
    for category_index, category in enumerate(library.category_names):
        for local_index, pixel in enumerate(pixels):
            blocks.append(ObservedBlock(
                category=category,
                observation_pose=(
                    -400.0 + category_index * 25.0 + local_index * 3.0,
                    -100.0 + local_index * 40.0,
                    250.0,
                    -180.0,
                    0.0,
                    90.0,
                ),
                detected_angle_deg=float(local_index * 15),
                source_id=category_index * 5 + local_index,
                pick_surface_z_mm=50.0,
                pick_surface_z_valid=True,
                high_detected_pixel_xy=pixel,
                high_image_center_xy=(5.5, 7.5),
                rough_localization_source="报告回放测试",
            ))
    return tuple(blocks)


def _target_cache(library, required_pids):
    placement_xy = np.full((library.placement_count, 2), np.nan, dtype=np.float64)
    records = {}
    pose_by_center = {}
    for pid in required_pids:
        row = float(library.placement_row[pid])
        col = float(library.placement_col[pid])
        key = (row, col)
        if key not in records:
            pose = (col * 20.0, row * 15.0, 200.0, -180.0, 0.0, 90.0)
            pose_by_center[key] = pose
            records[key] = {
                "行": row,
                "列": col,
                "目标中心像素XY": [col, row],
                "TCP观察位姿": list(pose),
                "高位定位诊断": {},
                "placement_PID": [],
            }
        records[key]["placement_PID"].append(int(pid))
        placement_xy[int(pid)] = pose_by_center[key][:2]
    return placement_xy, pose_by_center, tuple(records[key] for key in sorted(records))


def _placement_targets(library, relaxed_result, pose_by_center):
    pids = sorted({
        int(pid)
        for candidate in relaxed_result.candidates
        for pid in library.board_target_pid[candidate.board_index].flat
        if int(pid) >= 0
    })
    return {
        pid: PlacementTarget(
            index=pid,
            row=float(library.placement_row[pid]),
            col=float(library.placement_col[pid]),
            desired_angle_deg=float(library.placement_yaw_clockwise_deg[pid]),
            category=library.category_names[int(library.placement_category[pid])],
            observation_pose=pose_by_center[(
                float(library.placement_row[pid]),
                float(library.placement_col[pid]),
            )],
            cells=tuple(
                tuple(int(value) for value in cell)
                for cell in library.placement_cells[pid]
            ),
        )
        for pid in pids
    }


def test_new_dynamic_report_replays_same_rank_scores_board_and_sequence(tmp_path):
    library = load_v5_board_library(LIBRARY_PATH)
    blocks = _observed_blocks(library)
    selector_config = BoardCandidateSelectorConfig(
        coarse_top_k=8,
        final_candidate_k=2,
        keep_coarse_boundary_ties=True,
    )
    selector = BoardCandidateSelector(library, selector_config)
    center = (5.5, 7.5)
    coarse_result = selector.select_coarse(blocks, center)
    required_pids = selector.required_placement_ids(coarse_result)
    placement_xy, pose_by_center, center_records = _target_cache(
        library,
        required_pids,
    )
    relaxed_result = selector.select_relaxed(
        coarse_result,
        blocks,
        placement_xy,
        board_angle_deg=3.25,
    )
    optimizer_config = TaskSequenceOptimizerConfig(
        shooting_pose=(-250.0, 20.0, 380.0, -180.0, 0.0, 90.0),
        camera_to_sucker_offset_mm=(-94.1, -13.8),
        pick_surface_offset_mm=165.0,
        pick_approach_clearance_mm=7.0,
        motor_velocity_deg_per_sec=90.0,
        initial_motor_angle_deg=180.0,
        motor_lower_margin_deg=10.0,
        motor_upper_margin_deg=350.0,
        beam_width=10,
        report_top_candidates=2,
    )
    final_config = FinalBoardSelectorConfig(
        comparison_beam_width=10,
        comparison_returned_candidates=1,
        confirmation_beam_width=20,
        confirmation_returned_candidates=2,
        soft_time_budget_sec=10.0,
    )
    motion_model = get_default_arm_motion_time_model()
    decision = FinalBoardSelector(library, final_config).select(
        relaxed_result,
        blocks,
        _placement_targets(library, relaxed_result, pose_by_center),
        3.25,
        optimizer_config,
        motion_model,
    )
    runtime_config = {
        "mode": "shadow",
        "library_path": str(LIBRARY_PATH),
        "coarse_top_k": selector_config.coarse_top_k,
        "final_candidate_k": selector_config.final_candidate_k,
        "keep_coarse_boundary_ties": selector_config.keep_coarse_boundary_ties,
        "comparison_beam_width": final_config.comparison_beam_width,
        "comparison_returned_candidates": final_config.comparison_returned_candidates,
        "confirmation_beam_width": final_config.confirmation_beam_width,
        "confirmation_returned_candidates": final_config.confirmation_returned_candidates,
        "soft_time_budget_sec": final_config.soft_time_budget_sec,
        "failure_prompt_timeout_sec": 60.0,
        "task_sequence_optimizer": {
            "shooting_pose": list(optimizer_config.shooting_pose),
            "camera_to_sucker_offset_mm": list(
                optimizer_config.camera_to_sucker_offset_mm
            ),
            "pick_surface_offset_mm": optimizer_config.pick_surface_offset_mm,
            "pick_approach_clearance_mm": optimizer_config.pick_approach_clearance_mm,
            "motor_velocity_deg_per_sec": optimizer_config.motor_velocity_deg_per_sec,
            "initial_motor_angle_deg": optimizer_config.initial_motor_angle_deg,
            "motor_lower_margin_deg": optimizer_config.motor_lower_margin_deg,
            "motor_upper_margin_deg": optimizer_config.motor_upper_margin_deg,
        },
        "motion_model_path": str(motion_model.source_path),
        "arm_speed": motion_model.move_speed_percent,
        "pick_approach_speed": motion_model.move_speed_percent,
    }
    grid = {
        row: {col: (float(col), float(row)) for col in range(1, 11)}
        for row in range(1, 15)
    }
    document = build_dynamic_board_selection_report(
        mode="shadow",
        outcome="回放测试",
        observed_blocks=blocks,
        board_grid_points=grid,
        tray_center_pixel_xy=center,
        board_angle_deg=3.25,
        image_shape=(720, 1280),
        target_center_cache=center_records,
        library=library,
        runtime_config=runtime_config,
        calibration_sha256={
            "机械臂运动时间": sha256_file(motion_model.source_path),
        },
        coarse_result=coarse_result,
        relaxed_result=relaxed_result,
        decision=decision,
        actual_planner_message="V5 shadow",
    )
    report_path = tmp_path / "动态盘面选择报告.json"
    atomic_write_json(report_path, document)

    replay_module = _load_replay_module()
    assert replay_module.replay_dynamic_report(report_path, library) == "ok"
