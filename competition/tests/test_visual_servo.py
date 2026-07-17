from types import SimpleNamespace

import numpy as np

from competition_lib.config import load_execution_config, load_visual_servo_config
from competition_lib.visual_servo import (
    limit_xy_step,
    pixel_error_to_robot_delta,
    run_offset_visual_servo_alignment,
)


def _response(found=True, dx=0.0, dy=0.0, message=""):
    return SimpleNamespace(found=found, dx_px=dx, dy_px=dy, message=message)


def test_current_execution_and_servo_configs_are_valid():
    execution = load_execution_config()
    visual = load_visual_servo_config()
    assert execution.arm_speed > 0
    assert execution.servo_speed > 0
    assert len(execution.shooting_pose) == 6
    assert np.asarray(visual["pixel_to_robot_matrix"]).shape == (2, 2)


def test_step_limit_applies_minimum_and_maximum():
    assert np.allclose(limit_xy_step([10, 0], 5, 0.1), [5, 0])
    assert np.allclose(limit_xy_step([0.01, 0], 5, 0.1), [0.1, 0])
    assert np.allclose(limit_xy_step([0, 0], 5, 0.1), [0, 0])


def test_pixel_error_matrix_mapping_is_preserved():
    delta = pixel_error_to_robot_delta(10, 20, [[0, 0.07], [0.07, 0]], 5)
    assert np.allclose(delta, [1.4, 0.7])


def test_alignment_requires_configured_stable_frames():
    responses = iter([_response(dx=1, dy=1), _response(dx=0.5, dy=0.5)])
    result = run_offset_visual_servo_alignment(
        lambda: next(responses),
        lambda *_args, **_kwargs: None,
        [0, 0, 200, 0, 0, 0],
        {"pixel_to_robot_matrix": [[1, 0], [0, 1]]},
        speed=25,
        error_threshold_px=2,
        max_step_mm=5,
        max_iter=2,
        success_stable_frames=2,
        max_missed_frames=2,
        settle_sec=0,
    )
    assert result[0] is True


def test_alignment_stops_after_consecutive_misses():
    result = run_offset_visual_servo_alignment(
        lambda: _response(found=False, message="未识别"),
        lambda *_args, **_kwargs: None,
        [0, 0, 200, 0, 0, 0],
        {"pixel_to_robot_matrix": [[1, 0], [0, 1]]},
        speed=25,
        error_threshold_px=2,
        max_step_mm=5,
        max_iter=5,
        success_stable_frames=2,
        max_missed_frames=2,
        settle_sec=0,
    )
    assert result[0] is False
    assert "连续多帧" in result[3]
