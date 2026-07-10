import os

import pytest

from image_process_lib.task_planner import (
    ObservedBlock,
    PlacementTarget,
    assign_blocks_to_targets,
    load_task_layout,
    normalize_rotation_delta,
)


PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _pose(x, y):
    return [x, y, 200, -180, 0, 90]


def test_base_layout_is_single_valid_source():
    layout = load_task_layout(os.path.join(PACKAGE_DIR, "config", "task_layout.yaml"))
    assert len(layout) == 34
    assert [item["index"] for item in layout] == list(range(34))


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
