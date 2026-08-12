import os

import pytest

from image_process_lib.task_planner import (
    ObservedBlock,
    PlacementTarget,
    assign_blocks_to_targets,
    load_task_layout,
    normalize_rotation_delta,
)
from image_process_lib.task_geometry import (
    build_support_graph,
    validate_dense_target_sequence,
)


PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _pose(x, y):
    return [x, y, 200, -180, 0, 90]


def test_base_layout_is_single_valid_source():
    layout = load_task_layout(os.path.join(PACKAGE_DIR, "config", "task_layout.yaml"))
    assert len(layout) == 34
    assert [item["index"] for item in layout] == list(range(34))
    cells = [cell for item in layout for cell in item["cells"]]
    assert len(cells) == 136
    assert len(set(cells)) == 136
    graph = build_support_graph(layout)
    validate_dense_target_sequence(range(34), graph)


def test_assignment_is_stateless_across_repeated_calls():
    blocks = [
        ObservedBlock("T", _pose(0, 0), 10),
        ObservedBlock("T", _pose(100, 0), 20),
    ]
    targets = [
        PlacementTarget(0, 1, 1, 0, "T", _pose(5, 0)),
        PlacementTarget(1, 1, 2, 90, "T", _pose(95, 0)),
    ]
    first = assign_blocks_to_targets(blocks, targets, board_angle_deg=0)
    second = assign_blocks_to_targets(blocks, targets, board_angle_deg=0)
    assert first == second
    assert first[0].pick_observation_pose[0] == 0
    assert first[1].pick_observation_pose[0] == 100


def test_assignment_rejects_insufficient_category_count():
    blocks = [ObservedBlock("T", _pose(0, 0), 0)]
    targets = [
        PlacementTarget(0, 1, 1, 0, "T", _pose(0, 0)),
        PlacementTarget(1, 1, 2, 0, "T", _pose(1, 0)),
    ]
    with pytest.raises(ValueError, match="数量不足"):
        assign_blocks_to_targets(blocks, targets, board_angle_deg=0)


def test_assignment_preserves_high_localization_diagnostics():
    block = ObservedBlock(
        "T",
        _pose(0, 0),
        10,
        high_detected_pixel_xy=(100.5, 200.5),
        high_depth_sample_pixel_xy=(0.0, 0.0),
        high_image_center_xy=(640.0, 360.0),
        high_world_position=(0.0, 0.0, 0.0),
        high_world_position_valid=False,
        rough_localization_source="tcp_calibration",
    )
    target = PlacementTarget(
        0,
        1,
        1,
        0,
        "T",
        _pose(5, 0),
        high_detected_pixel_xy=(300.5, 400.5),
        high_depth_sample_pixel_xy=(0.0, 0.0),
        high_image_center_xy=(640.0, 360.0),
        high_world_position=(0.0, 0.0, 0.0),
        high_world_position_valid=False,
        rough_localization_source="tcp_calibration",
    )

    result = assign_blocks_to_targets([block], [target], board_angle_deg=0)[0]

    assert result.pick_high_detected_pixel_xy == (100.5, 200.5)
    assert result.pick_high_world_position == (0.0, 0.0, 0.0)
    assert result.pick_rough_localization_source == "tcp_calibration"
    assert result.place_high_detected_pixel_xy == (300.5, 400.5)
    assert result.place_high_world_position == (0.0, 0.0, 0.0)
    assert result.place_rough_localization_source == "tcp_calibration"


@pytest.mark.parametrize(
    "category,expected_limit",
    [("T", 180), ("line", 90), ("z_blue", 90), ("square", 45)],
)
def test_rotation_delta_respects_block_symmetry(category, expected_limit):
    delta = normalize_rotation_delta(category, 355, 5, 0)
    assert -expected_limit <= delta <= expected_limit


def test_rotation_delta_preserves_legacy_boundary_direction():
    assert normalize_rotation_delta("T", 180, 0, 0) == 180
    assert normalize_rotation_delta("line", 90, 0, 0) == 90
    assert normalize_rotation_delta("square", 45, 0, 0) == 45


def test_select_calibration_tray_points_returns_34_valid_points():
    from image_process_lib.task_planner import (
        CALIBRATION_TRAY_COL_COUNT,
        CALIBRATION_TRAY_MODE_COUNTS,
        CALIBRATION_TRAY_POINT_COUNT,
        CALIBRATION_TRAY_ROW_COUNT,
        select_calibration_tray_points,
    )

    points = select_calibration_tray_points(seed=7)

    assert len(points) == CALIBRATION_TRAY_POINT_COUNT == 34
    coordinates = [(point["row"], point["col"]) for point in points]
    assert len(set(coordinates)) == len(coordinates)
    for point in points:
        assert 1.0 <= point["row"] <= CALIBRATION_TRAY_ROW_COUNT
        assert 1.0 <= point["col"] <= CALIBRATION_TRAY_COL_COUNT
        assert point["row"] * 2 == round(point["row"] * 2)
        assert point["col"] * 2 == round(point["col"] * 2)
    mode_counts = {}
    for point in points:
        mode_counts[point["mode"]] = mode_counts.get(point["mode"], 0) + 1
    assert mode_counts == CALIBRATION_TRAY_MODE_COUNTS
    assert sum(mode_counts.values()) == 34


def test_select_calibration_tray_points_is_reproducible_with_seed():
    from image_process_lib.task_planner import select_calibration_tray_points

    assert select_calibration_tray_points(seed=11) == select_calibration_tray_points(seed=11)
