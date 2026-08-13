"""V5 盘面快速筛选器的字典序、missing 与 relaxed 配对测试。"""

from dataclasses import replace
import random

import numpy as np
import pytest

from image_process_lib.block_category import BLOCK_CATEGORY_NAMES
from image_process_lib.board_candidate_selector import (
    BoardCandidateSelector,
    BoardCandidateSelectorConfig,
    format_candidate_selection_report,
)
from image_process_lib.task_planner import ObservedBlock
from image_process_lib.v5_board_library import (
    V5BoardLibrary,
    V5_BOARD_LIBRARY_FORMAT_VERSION,
    V5_REGION_MASK_ORDER,
)


QUADRANT_PIXEL = {
    1: (-10.0, -10.0),
    2: (10.0, -10.0),
    4: (-10.0, 10.0),
    8: (10.0, 10.0),
}
DEFAULT_REGIONS = (1, 2, 4, 8, 1)
DEFAULT_XY = (
    (-10.0, -10.0),
    (10.0, -10.0),
    (-10.0, 10.0),
    (10.0, 10.0),
    (0.0, 0.0),
)


def _make_board_spec(
    global_id,
    missing_category="square",
    regions_by_category=None,
    xy_by_category=None,
    yaw_by_category=None,
):
    missing_index = BLOCK_CATEGORY_NAMES.index(missing_category)
    region_lists = []
    xy_lists = []
    yaw_lists = []
    regions_by_category = regions_by_category or {}
    xy_by_category = xy_by_category or {}
    yaw_by_category = yaw_by_category or {}
    for category_index, category in enumerate(BLOCK_CATEGORY_NAMES):
        count = 4 if category_index == missing_index else 5
        regions = list(regions_by_category.get(category, DEFAULT_REGIONS))
        xy_values = list(xy_by_category.get(category, DEFAULT_XY))
        yaw_values = list(yaw_by_category.get(category, (0.0,) * 5))
        if len(regions) != count:
            regions = regions[:count]
        if len(xy_values) != count:
            xy_values = xy_values[:count]
        if len(yaw_values) != count:
            yaw_values = yaw_values[:count]
        assert len(regions) == len(xy_values) == len(yaw_values) == count
        region_lists.append(regions)
        xy_lists.append(xy_values)
        yaw_lists.append(yaw_values)
    return {
        "global_id": global_id,
        "missing_index": missing_index,
        "regions": region_lists,
        "xy": xy_lists,
        "yaw": yaw_lists,
    }


def _make_library(board_specs):
    board_specs = list(board_specs)
    board_count = len(board_specs)
    category_count = len(BLOCK_CATEGORY_NAMES)
    board_target_pid = np.full((board_count, category_count, 5), -1, dtype=np.int16)
    board_region_counts = np.zeros(
        (board_count, category_count, len(V5_REGION_MASK_ORDER)), dtype=np.uint8
    )
    placement_categories = []
    placement_rows = []
    placement_cols = []
    placement_yaws = []
    placement_masks = []
    placement_xy = []
    mask_to_index = {mask: index for index, mask in enumerate(V5_REGION_MASK_ORDER)}

    for board_index, spec in enumerate(board_specs):
        for category_index in range(category_count):
            for target_offset, region_mask in enumerate(spec["regions"][category_index]):
                pid = len(placement_categories)
                board_target_pid[board_index, category_index, target_offset] = pid
                board_region_counts[
                    board_index, category_index, mask_to_index[region_mask]
                ] += 1
                placement_categories.append(category_index)
                placement_rows.append(float(target_offset))
                placement_cols.append(float(category_index))
                placement_yaws.append(int(spec["yaw"][category_index][target_offset]))
                placement_masks.append(region_mask)
                placement_xy.append(spec["xy"][category_index][target_offset])

    placement_count = len(placement_categories)
    library = V5BoardLibrary(
        format_version=V5_BOARD_LIBRARY_FORMAT_VERSION,
        category_names=tuple(BLOCK_CATEGORY_NAMES),
        region_mask_order=np.asarray(V5_REGION_MASK_ORDER, dtype=np.uint8),
        board_global_id=np.asarray(
            [spec["global_id"] for spec in board_specs], dtype=np.int32
        ),
        board_layout_index=np.zeros(board_count, dtype=np.uint16),
        board_rank=np.arange(board_count, dtype=np.int32),
        board_missing_category=np.asarray(
            [spec["missing_index"] for spec in board_specs], dtype=np.uint8
        ),
        board_target_pid=board_target_pid,
        board_region_counts=board_region_counts,
        placement_category=np.asarray(placement_categories, dtype=np.uint8),
        placement_row=np.asarray(placement_rows, dtype=np.float32),
        placement_col=np.asarray(placement_cols, dtype=np.float32),
        placement_yaw_clockwise_deg=np.asarray(placement_yaws, dtype=np.int16),
        placement_region_mask=np.asarray(placement_masks, dtype=np.uint8),
        placement_cells=np.zeros((placement_count, 4, 2), dtype=np.uint8),
    )
    return library, np.asarray(placement_xy, dtype=np.float64)


def _make_sources(
    regions_by_category=None,
    xy_by_category=None,
    yaw_by_category=None,
):
    regions_by_category = regions_by_category or {}
    xy_by_category = xy_by_category or {}
    yaw_by_category = yaw_by_category or {}
    sources = []
    for category_index, category in enumerate(BLOCK_CATEGORY_NAMES):
        regions = regions_by_category.get(category, DEFAULT_REGIONS)
        xy_values = xy_by_category.get(category, DEFAULT_XY)
        yaw_values = yaw_by_category.get(category, (0.0,) * 5)
        for local_index in range(5):
            x_mm, y_mm = xy_values[local_index]
            sources.append(
                ObservedBlock(
                    category=category,
                    observation_pose=(x_mm, y_mm, 200.0, -180.0, 0.0, 90.0),
                    detected_angle_deg=yaw_values[local_index],
                    source_id=category_index * 5 + local_index,
                    high_detected_pixel_xy=QUADRANT_PIXEL[regions[local_index]],
                )
            )
    return sources


def _run_both_stages(library, placement_xy, sources, coarse_top_k=300, final_k=20):
    selector = BoardCandidateSelector(
        library,
        BoardCandidateSelectorConfig(
            coarse_top_k=coarse_top_k,
            final_candidate_k=final_k,
            keep_coarse_boundary_ties=True,
        ),
    )
    coarse = selector.select_coarse(sources, (0.0, 0.0))
    final = selector.select_relaxed(coarse, sources, placement_xy, 0.0)
    return selector, coarse, final


def test_target_masks_cover_fixed_vertical_horizontal_and_center_regions():
    spec = _make_board_spec(
        100,
        regions_by_category={"L_blue": [1, 3, 5, 15, 1]},
    )
    library, _ = _make_library([spec])
    sources = _make_sources()
    selector = BoardCandidateSelector(library)

    coarse = selector.select_coarse(sources, (0.0, 0.0))

    assert coarse.candidates[0].score == (0, 0)
    assert len(coarse.candidates[0].assignment) == 34
    for assignment in coarse.candidates[0].assignment:
        assert assignment.matched_target_region_bit & assignment.target_region_mask


def test_lexicographic_region_score_never_trades_lr_for_ud():
    all_lu_sources = {
        category: [1, 1, 1, 1, 1] for category in BLOCK_CATEGORY_NAMES
    }
    board_with_lr = _make_board_spec(
        100,
        regions_by_category={
            category: ([2, 1, 1, 1, 1] if category == "L_blue" else [1, 1, 1, 1, 1])
            for category in BLOCK_CATEGORY_NAMES
        },
    )
    board_with_many_ud = _make_board_spec(
        101,
        regions_by_category={
            category: [4, 4, 4, 4, 4] for category in BLOCK_CATEGORY_NAMES
        },
    )
    library, _ = _make_library([board_with_lr, board_with_many_ud])
    selector = BoardCandidateSelector(
        library,
        BoardCandidateSelectorConfig(
            coarse_top_k=1,
            final_candidate_k=1,
            keep_coarse_boundary_ties=False,
        ),
    )

    coarse = selector.select_coarse(
        _make_sources(regions_by_category=all_lu_sources), (0.0, 0.0)
    )

    assert coarse.candidates[0].global_id == 101
    assert coarse.candidates[0].n_lr == 0
    assert coarse.candidates[0].n_ud > 1


def test_missing_category_automatically_discards_best_source_in_both_stages():
    t_regions = [1, 1, 1, 1, 8]
    t_source_xy = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0), (1000.0, 0.0)]
    spec = _make_board_spec(
        100,
        missing_category="T",
        regions_by_category={"T": [1, 1, 1, 1]},
        xy_by_category={"T": t_source_xy[:4]},
    )
    library, placement_xy = _make_library([spec])
    sources = _make_sources(
        regions_by_category={"T": t_regions},
        xy_by_category={"T": t_source_xy},
    )

    _, coarse, final = _run_both_stages(library, placement_xy, sources)

    assert coarse.candidates[0].unused_source_id == 29
    assert final.candidates[0].unused_source_id == 29


def test_relaxed_assignment_minimizes_distance_before_rotation():
    focus_xy = [(0.0, 0.0), (10.0, 0.0), (100.0, 0.0), (200.0, 0.0), (300.0, 0.0)]
    focus_source_yaw = [90.0, 0.0, 0.0, 0.0, 0.0]
    focus_target_yaw = [0.0, 90.0, 0.0, 0.0, 0.0]
    spec = _make_board_spec(
        100,
        xy_by_category={"L_blue": focus_xy},
        yaw_by_category={"L_blue": focus_target_yaw},
    )
    library, placement_xy = _make_library([spec])
    sources = _make_sources(
        xy_by_category={"L_blue": focus_xy},
        yaw_by_category={"L_blue": focus_source_yaw},
    )

    _, _, final = _run_both_stages(library, placement_xy, sources)
    candidate = final.candidates[0]
    l_blue_assignments = [
        item for item in candidate.relaxed_assignment if item.category == "L_blue"
    ]

    assert candidate.relaxed_distance_mm == pytest.approx(0.0)
    assert l_blue_assignments[0].source_id == 0
    assert l_blue_assignments[1].source_id == 1
    assert sum(item.rotation_deg for item in l_blue_assignments) == pytest.approx(180.0)


def test_relaxed_assignment_uses_rotation_to_break_distance_tie():
    same_xy = [(0.0, 0.0)] * 5
    focus_source_yaw = [90.0, 0.0, 0.0, 0.0, 0.0]
    focus_target_yaw = [0.0, 90.0, 0.0, 0.0, 0.0]
    spec = _make_board_spec(
        100,
        xy_by_category={"L_blue": same_xy},
        yaw_by_category={"L_blue": focus_target_yaw},
    )
    library, placement_xy = _make_library([spec])
    sources = _make_sources(
        xy_by_category={"L_blue": same_xy},
        yaw_by_category={"L_blue": focus_source_yaw},
    )

    _, _, final = _run_both_stages(library, placement_xy, sources)
    l_blue_assignments = [
        item for item in final.candidates[0].relaxed_assignment if item.category == "L_blue"
    ]

    assert l_blue_assignments[0].source_id == 1
    assert l_blue_assignments[1].source_id == 0
    assert sum(item.rotation_deg for item in l_blue_assignments) == pytest.approx(0.0)


def test_symmetric_categories_use_existing_rotation_equivalence_rules():
    target_yaws = {
        "z_blue": [180.0] * 5,
        "z_green": [180.0] * 5,
        "square": [90.0] * 4,
        "line": [180.0] * 5,
    }
    spec = _make_board_spec(100, yaw_by_category=target_yaws)
    library, placement_xy = _make_library([spec])

    _, _, final = _run_both_stages(library, placement_xy, _make_sources())

    assert final.candidates[0].relaxed_rotation_deg == pytest.approx(0.0)


def test_coarse_boundary_ties_expand_final_top_k_is_strict_and_shuffle_is_stable():
    specs = [_make_board_spec(100 + index) for index in range(25)]
    library, placement_xy = _make_library(specs)
    sources = _make_sources()
    selector, coarse, final = _run_both_stages(
        library, placement_xy, sources, coarse_top_k=3, final_k=20
    )

    assert len(coarse.candidates) == 25
    assert len(final.candidates) == 20
    assert [item.global_id for item in final.candidates] == list(range(100, 120))

    shuffled_sources = list(sources)
    random.Random(20260813).shuffle(shuffled_sources)
    shuffled_coarse = selector.select_coarse(shuffled_sources, (0.0, 0.0))
    shuffled_final = selector.select_relaxed(
        shuffled_coarse, shuffled_sources, placement_xy, 0.0
    )
    assert [item.board_id for item in shuffled_final.candidates] == [
        item.board_id for item in final.candidates
    ]
    assert [
        tuple((pair.source_id, pair.placement_id) for pair in item.relaxed_assignment)
        for item in shuffled_final.candidates
    ] == [
        tuple((pair.source_id, pair.placement_id) for pair in item.relaxed_assignment)
        for item in final.candidates
    ]

    report = format_candidate_selection_report(coarse, final, coarse_preview_count=2)
    assert "总盘面数 = 25" in report
    assert "最终选中 20 张盘面" in report
    assert "source 0 -> placement" in report


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda items: items[:-1], "35 个"),
        (
            lambda items: [replace(items[0], source_id=items[1].source_id)] + items[1:],
            "source_id 必须唯一",
        ),
        (
            lambda items: [replace(items[0], category="T")] + items[1:],
            "L_blue.*5 个",
        ),
        (
            lambda items: [replace(items[0], high_detected_pixel_xy=(np.nan, 0.0))]
            + items[1:],
            "high_detected_pixel_xy",
        ),
        (
            lambda items: [
                replace(items[0], observation_pose=(np.nan, 0.0, 0.0, 0.0, 0.0, 0.0))
            ]
            + items[1:],
            "TCP XY",
        ),
        (
            lambda items: [replace(items[0], detected_angle_deg=np.nan)] + items[1:],
            "检测角度",
        ),
    ],
)
def test_invalid_observations_fail_with_clear_errors(mutator, match):
    library, _ = _make_library([_make_board_spec(100)])
    selector = BoardCandidateSelector(library)

    with pytest.raises(ValueError, match=match):
        selector.select_coarse(mutator(_make_sources()), (0.0, 0.0))


def test_relaxed_rejects_bad_xy_and_nonfinite_board_angle():
    library, placement_xy = _make_library([_make_board_spec(100)])
    sources = _make_sources()
    selector = BoardCandidateSelector(library)
    coarse = selector.select_coarse(sources, (0.0, 0.0))

    with pytest.raises(ValueError, match="形状必须"):
        selector.select_relaxed(coarse, sources, placement_xy[:-1], 0.0)
    bad_xy = placement_xy.copy()
    bad_xy[selector.required_placement_ids(coarse)[0], 0] = np.nan
    with pytest.raises(ValueError, match="非有限坐标"):
        selector.select_relaxed(coarse, sources, bad_xy, 0.0)
    with pytest.raises(ValueError, match="board_angle_deg"):
        selector.select_relaxed(coarse, sources, placement_xy, np.nan)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
