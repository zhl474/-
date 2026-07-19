from types import SimpleNamespace

import numpy as np

from competition_lib.config import load_execution_config, load_visual_servo_config
from competition_lib.visual_servo import (
    limit_xy_step,
    pixel_error_to_robot_delta,
    run_offset_visual_servo_alignment,
)


def _response(found=True, dx=0.0, dy=0.0, px=320.0, py=240.0, message=""):
    return SimpleNamespace(found=found, px=px, py=py, dx_px=dx, dy_px=dy, message=message)


def test_current_execution_and_servo_configs_are_valid():
    execution = load_execution_config()
    visual = load_visual_servo_config()
    assert execution.arm_speed > 0
    assert execution.servo_speed > 0
    assert len(execution.shooting_pose) == 6
    assert np.asarray(visual["pixel_to_robot_matrix"]).shape == (2, 2)
    assert visual["block_servo_height_offset_mm"] > 0
    assert visual["board_servo_height_offset_mm"] > 0
    assert "servo_height_offset_mm" not in visual


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


def test_alignment_writes_correction_event_without_terminal_detail(capsys):
    events = []
    run_offset_visual_servo_alignment(
        lambda: _response(dx=3, dy=1),
        lambda *_args, **_kwargs: None,
        [0, 0, 200, 0, 0, 0],
        {"pixel_to_robot_matrix": [[1, 0], [0, 1]]},
        speed=25,
        error_threshold_px=2,
        max_step_mm=5,
        max_iter=1,
        success_stable_frames=1,
        max_missed_frames=2,
        settle_sec=0,
        log_label="方块视觉伺服",
        event_callback=events.append,
    )

    assert capsys.readouterr().out == ""
    assert len(events) == 1
    assert events[0]["事件"] == "执行修正"
    assert events[0]["伺服轮次"] == 1
    assert events[0]["XY修正X毫米"] == 3.0
    assert events[0]["低位图像中心X"] == 317.0


def test_alignment_event_includes_low_target_and_camera_center_pixels():
    events = []
    run_offset_visual_servo_alignment(
        lambda: _response(dx=3, dy=-2, px=323, py=238),
        lambda *_args, **_kwargs: None,
        [0, 0, 200, 0, 0, 0],
        {"pixel_to_robot_matrix": [[1, 0], [0, 1]]},
        speed=25,
        error_threshold_px=1,
        max_step_mm=5,
        max_iter=1,
        success_stable_frames=1,
        max_missed_frames=2,
        settle_sec=0,
        event_callback=events.append,
    )

    assert events[0]["低位目标像素X"] == 323.0
    assert events[0]["低位目标像素Y"] == 238.0
    assert events[0]["低位图像中心X"] == 320.0
    assert events[0]["低位图像中心Y"] == 240.0


def test_alignment_emits_missing_target_event_before_failure():
    events = []
    result = run_offset_visual_servo_alignment(
        lambda: _response(found=False, message="未识别"),
        lambda *_args, **_kwargs: None,
        [0, 0, 200, 0, 0, 0],
        {"pixel_to_robot_matrix": [[1, 0], [0, 1]]},
        speed=25,
        error_threshold_px=2,
        max_step_mm=5,
        max_iter=5,
        success_stable_frames=1,
        max_missed_frames=2,
        settle_sec=0,
        event_callback=events.append,
    )

    assert result[0] is False
    assert [event["事件"] for event in events] == ["目标丢失", "目标丢失"]


def test_alignment_motion_requests_stability_wait():
    move_calls = []
    run_offset_visual_servo_alignment(
        lambda: _response(dx=3, dy=1),
        lambda *args, **kwargs: move_calls.append((args, kwargs)),
        [0, 0, 200, 0, 0, 0],
        {"pixel_to_robot_matrix": [[1, 0], [0, 1]]},
        speed=25,
        error_threshold_px=2,
        max_step_mm=5,
        max_iter=1,
        success_stable_frames=1,
        max_missed_frames=2,
        settle_sec=0,
    )

    assert len(move_calls) == 1
    assert move_calls[0][1]["wait_until_stable"] is True
