from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from competition_lib.config import (
    DEFAULT_EXECUTION_CONFIG_PATH,
    load_execution_config,
    load_visual_servo_config,
)
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
    assert execution.pick_approach_clearance_mm == 3.0
    # 运行参数允许由控制台按现场需要调整，这里只验证加载和取值范围。
    assert execution.pick_approach_speed > 0
    assert execution.pick_retreat_blend_radius_mm == 5.0
    # 下探/抬升参数可由参数中心按现场调整，这里只验证非负取值范围。
    assert execution.place_descent_offset_mm >= 0
    assert execution.place_descent_blend_radius_mm >= 0
    assert execution.place_lift_blend_radius_mm >= 0
    assert execution.minimum_tcp_z_mm == 162.0
    assert isinstance(execution.calibration_mode, bool)
    assert isinstance(execution.visual_servo_enabled, bool)
    assert execution.block_error_threshold_px == 1.0
    assert execution.tray_error_threshold_px == 0.5
    assert isinstance(execution.post_success_sample_frames, int)
    assert execution.post_success_sample_frames >= 0
    assert len(execution.shooting_pose) == 6
    assert not hasattr(execution, "lift_z")
    assert np.asarray(visual["pixel_to_robot_matrix"]).shape == (2, 2)
    assert "block_servo_height_offset_mm" not in visual
    assert "board_servo_height_offset_mm" not in visual


def test_执行配置兼容旧版统一视觉伺服阈值(tmp_path):
    config_data = yaml.safe_load(
        Path(DEFAULT_EXECUTION_CONFIG_PATH).read_text(encoding="utf-8")
    )
    config_data["servo"].pop("block_error_threshold_px")
    config_data["servo"].pop("tray_error_threshold_px")
    config_data["servo"]["error_threshold_px"] = 0.75
    config_path = tmp_path / "旧版统一阈值.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    config = load_execution_config(config_path)

    assert config.block_error_threshold_px == 0.75
    assert config.tray_error_threshold_px == 0.75


@pytest.mark.parametrize(
    "threshold_name,invalid_value",
    [
        ("block_error_threshold_px", -0.1),
        ("block_error_threshold_px", float("inf")),
        ("block_error_threshold_px", float("nan")),
        ("block_error_threshold_px", "不是数值"),
        ("tray_error_threshold_px", -0.1),
        ("tray_error_threshold_px", float("inf")),
        ("tray_error_threshold_px", float("nan")),
        ("tray_error_threshold_px", "不是数值"),
    ],
)
def test_执行配置拒绝非法目标级视觉伺服阈值(
    tmp_path,
    threshold_name,
    invalid_value,
):
    config_data = yaml.safe_load(
        Path(DEFAULT_EXECUTION_CONFIG_PATH).read_text(encoding="utf-8")
    )
    config_data["servo"][threshold_name] = invalid_value
    config_path = tmp_path / "非法目标级阈值.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=threshold_name):
        load_execution_config(config_path)


@pytest.mark.parametrize(
    "missing_name",
    ["block_error_threshold_px", "tray_error_threshold_px"],
)
def test_执行配置拒绝只配置一个目标级视觉伺服阈值(tmp_path, missing_name):
    config_data = yaml.safe_load(
        Path(DEFAULT_EXECUTION_CONFIG_PATH).read_text(encoding="utf-8")
    )
    config_data["servo"].pop(missing_name)
    config_path = tmp_path / "缺少目标级阈值.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="必须同时配置"):
        load_execution_config(config_path)


@pytest.mark.parametrize("calibration_mode", [False, True])
def test_execution_config_reads_strict_boolean_calibration_mode(tmp_path, calibration_mode):
    config_data = yaml.safe_load(
        Path(DEFAULT_EXECUTION_CONFIG_PATH).read_text(encoding="utf-8")
    )
    config_data["calibration_mode"] = calibration_mode
    config_path = tmp_path / "运行模式.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    assert load_execution_config(config_path).calibration_mode is calibration_mode


@pytest.mark.parametrize("invalid_value", ["false", "true", 0, 1, None])
def test_execution_config_rejects_nonboolean_calibration_mode(tmp_path, invalid_value):
    config_data = yaml.safe_load(
        Path(DEFAULT_EXECUTION_CONFIG_PATH).read_text(encoding="utf-8")
    )
    config_data["calibration_mode"] = invalid_value
    config_path = tmp_path / "错误运行模式.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="calibration_mode 必须是 YAML 布尔值"):
        load_execution_config(config_path)


@pytest.mark.parametrize("enabled", [False, True])
def test_执行配置读取视觉伺服开关(tmp_path, enabled):
    config_data = yaml.safe_load(
        Path(DEFAULT_EXECUTION_CONFIG_PATH).read_text(encoding="utf-8")
    )
    config_data["servo"]["enabled"] = enabled
    config_path = tmp_path / "视觉伺服开关.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    assert load_execution_config(config_path).visual_servo_enabled is enabled


def test_执行配置缺少视觉伺服开关时默认开启(tmp_path):
    config_data = yaml.safe_load(
        Path(DEFAULT_EXECUTION_CONFIG_PATH).read_text(encoding="utf-8")
    )
    config_data["servo"].pop("enabled")
    config_path = tmp_path / "旧版执行配置.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    assert load_execution_config(config_path).visual_servo_enabled is True


@pytest.mark.parametrize("invalid_value", [-1, 1.5, True, "20"])
def test_执行配置拒绝非法成功后静止采样帧数(tmp_path, invalid_value):
    config_data = yaml.safe_load(
        Path(DEFAULT_EXECUTION_CONFIG_PATH).read_text(encoding="utf-8")
    )
    config_data["servo"]["post_success_sample_frames"] = invalid_value
    config_path = tmp_path / "非法静止采样帧数.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="大于等于 0 的整数"):
        load_execution_config(config_path)


@pytest.mark.parametrize("invalid_value", ["false", "true", 0, 1, None])
def test_执行配置拒绝非布尔视觉伺服开关(
    tmp_path,
    invalid_value,
):
    config_data = yaml.safe_load(
        Path(DEFAULT_EXECUTION_CONFIG_PATH).read_text(encoding="utf-8")
    )
    config_data["servo"]["enabled"] = invalid_value
    config_path = tmp_path / "错误视觉伺服开关.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="servo.enabled 必须是 YAML 布尔值"):
        load_execution_config(config_path)


@pytest.mark.parametrize("invalid_minimum_tcp_z_mm", [0.0, -1.0])
def test_execution_config_rejects_non_positive_minimum_tcp_z(tmp_path, invalid_minimum_tcp_z_mm):
    config_data = yaml.safe_load(
        Path(DEFAULT_EXECUTION_CONFIG_PATH).read_text(encoding="utf-8")
    )
    config_data["motion"]["minimum_tcp_z_mm"] = invalid_minimum_tcp_z_mm
    config_path = tmp_path / "无效最低高度.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="minimum_tcp_z_mm 必须是大于 0 的有限数值"):
        load_execution_config(config_path)


@pytest.mark.parametrize(
    "invalid_value",
    [0.0, -1.0, float("inf"), float("nan"), True, "不是数值"],
)
def test_执行配置拒绝非法预抓取间隙(tmp_path, invalid_value):
    config_data = yaml.safe_load(
        Path(DEFAULT_EXECUTION_CONFIG_PATH).read_text(encoding="utf-8")
    )
    config_data["motion"]["pick_approach_clearance_mm"] = invalid_value
    config_path = tmp_path / "非法预抓取间隙.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="motion.pick_approach_clearance_mm"):
        load_execution_config(config_path)


@pytest.mark.parametrize(
    "invalid_value",
    [-1.0, 1000.1, float("inf"), float("nan"), True, "不是数值"],
)
def test_执行配置拒绝非法抓后抬升圆滑半径(tmp_path, invalid_value):
    config_data = yaml.safe_load(
        Path(DEFAULT_EXECUTION_CONFIG_PATH).read_text(encoding="utf-8")
    )
    config_data["motion"]["pick_retreat_blend_radius_mm"] = invalid_value
    config_path = tmp_path / "非法抓后抬升圆滑半径.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="motion.pick_retreat_blend_radius_mm"):
        load_execution_config(config_path)


def test_执行配置允许用零关闭抓后抬升圆滑(tmp_path):
    config_data = yaml.safe_load(
        Path(DEFAULT_EXECUTION_CONFIG_PATH).read_text(encoding="utf-8")
    )
    config_data["motion"]["pick_retreat_blend_radius_mm"] = 0
    config_path = tmp_path / "关闭抓后抬升圆滑.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    assert load_execution_config(config_path).pick_retreat_blend_radius_mm == 0.0


@pytest.mark.parametrize(
    "invalid_value",
    [-1.0, float("inf"), float("nan"), True, "不是数值"],
)
def test_执行配置拒绝非法摆放下探深度(tmp_path, invalid_value):
    config_data = yaml.safe_load(
        Path(DEFAULT_EXECUTION_CONFIG_PATH).read_text(encoding="utf-8")
    )
    config_data["motion"]["place_descent_offset_mm"] = invalid_value
    config_path = tmp_path / "非法摆放下探深度.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="motion.place_descent_offset_mm"):
        load_execution_config(config_path)


@pytest.mark.parametrize(
    "invalid_value",
    [-1.0, 1000.1, float("inf"), float("nan"), True, "不是数值"],
)
def test_执行配置拒绝非法摆放下探圆滑半径(tmp_path, invalid_value):
    config_data = yaml.safe_load(
        Path(DEFAULT_EXECUTION_CONFIG_PATH).read_text(encoding="utf-8")
    )
    config_data["motion"]["place_descent_blend_radius_mm"] = invalid_value
    config_path = tmp_path / "非法摆放下探圆滑半径.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="motion.place_descent_blend_radius_mm"):
        load_execution_config(config_path)


def test_执行配置拒绝圆滑半径不小于下探深度(tmp_path):
    config_data = yaml.safe_load(
        Path(DEFAULT_EXECUTION_CONFIG_PATH).read_text(encoding="utf-8")
    )
    config_data["motion"]["place_descent_offset_mm"] = 5.0
    config_data["motion"]["place_descent_blend_radius_mm"] = 5.0
    config_path = tmp_path / "圆滑半径不小于下探深度.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="必须小于"):
        load_execution_config(config_path)


def test_执行配置拒绝抬升圆滑半径不小于下探深度(tmp_path):
    config_data = yaml.safe_load(
        Path(DEFAULT_EXECUTION_CONFIG_PATH).read_text(encoding="utf-8")
    )
    config_data["motion"]["place_descent_offset_mm"] = 5.0
    config_data["motion"]["place_lift_blend_radius_mm"] = 5.0
    config_path = tmp_path / "抬升圆滑半径不小于下探深度.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="place_lift_blend_radius_mm"):
        load_execution_config(config_path)


@pytest.mark.parametrize(
    "invalid_value",
    [-1.0, 1000.1, float("inf"), float("nan"), True, "不是数值"],
)
def test_执行配置拒绝非法摆放后抬升圆滑半径(tmp_path, invalid_value):
    config_data = yaml.safe_load(
        Path(DEFAULT_EXECUTION_CONFIG_PATH).read_text(encoding="utf-8")
    )
    config_data["motion"]["place_lift_blend_radius_mm"] = invalid_value
    config_path = tmp_path / "非法摆放后抬升圆滑半径.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="motion.place_lift_blend_radius_mm"):
        load_execution_config(config_path)


def test_执行配置允许用零关闭摆放下探和圆滑(tmp_path):
    config_data = yaml.safe_load(
        Path(DEFAULT_EXECUTION_CONFIG_PATH).read_text(encoding="utf-8")
    )
    config_data["motion"]["place_descent_offset_mm"] = 0
    config_data["motion"]["place_descent_blend_radius_mm"] = 0
    config_data["motion"]["place_lift_blend_radius_mm"] = 0
    config_path = tmp_path / "关闭摆放下探和圆滑.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    config = load_execution_config(config_path)

    assert config.place_descent_offset_mm == 0.0
    assert config.place_descent_blend_radius_mm == 0.0
    assert config.place_lift_blend_radius_mm == 0.0


@pytest.mark.parametrize("invalid_value", [0, -1, 1.5, True, "30"])
def test_执行配置拒绝非法斜向接近速度(tmp_path, invalid_value):
    config_data = yaml.safe_load(
        Path(DEFAULT_EXECUTION_CONFIG_PATH).read_text(encoding="utf-8")
    )
    config_data["motion"]["pick_approach_speed"] = invalid_value
    config_path = tmp_path / "非法斜向接近速度.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="motion.pick_approach_speed"):
        load_execution_config(config_path)


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


def test_托盘小数阈值包含边界且超出时继续修正():
    boundary_moves = []
    boundary_result = run_offset_visual_servo_alignment(
        lambda: _response(dx=0.5, dy=-0.5),
        lambda *args, **kwargs: boundary_moves.append((args, kwargs)),
        [0, 0, 200, 0, 0, 0],
        {"pixel_to_robot_matrix": [[1, 0], [0, 1]]},
        speed=25,
        error_threshold_px=0.5,
        max_step_mm=5,
        max_iter=1,
        success_stable_frames=1,
        max_missed_frames=2,
        settle_sec=0,
        log_label="托盘视觉伺服",
    )
    over_limit_moves = []
    over_limit_result = run_offset_visual_servo_alignment(
        lambda: _response(dx=0.5001, dy=0.0),
        lambda *args, **kwargs: over_limit_moves.append((args, kwargs)),
        [0, 0, 200, 0, 0, 0],
        {"pixel_to_robot_matrix": [[1, 0], [0, 1]]},
        speed=25,
        error_threshold_px=0.5,
        max_step_mm=5,
        max_iter=1,
        success_stable_frames=1,
        max_missed_frames=2,
        settle_sec=0,
        log_label="托盘视觉伺服",
    )

    assert boundary_result[0] is True
    assert boundary_moves == []
    assert over_limit_result[0] is False
    assert len(over_limit_moves) == 1


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


def test_alignment_writes_correction_event_and_terminal_error_log(capsys):
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

    output = capsys.readouterr().out
    assert "[方块视觉伺服] 第 1 轮误差=(+3.00,+1.00)px，修正=(+3.000,+1.000)mm" in output
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


def test_alignment_logs_stable_frame_and_timing(capsys):
    result = run_offset_visual_servo_alignment(
        lambda: _response(dx=1, dy=-0.5),
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
        timing_debug=True,
        log_label="托盘视觉伺服",
    )

    output = capsys.readouterr().out
    assert result[0] is True
    assert "[托盘视觉伺服] 第 1 轮误差=(+1.00,-0.50)px，满足阈值 2.00px，稳定帧 1/1" in output
    assert "[托盘视觉伺服耗时] 第 1 轮 图像服务=" in output


def test_alignment_samples_twenty_static_frames_without_extra_motion():
    events = []
    moves = []
    responses = iter(
        [_response(dx=0.5, dy=-0.25)]
        + [_response(dx=0.1, dy=-0.2) for _index in range(20)]
    )

    result = run_offset_visual_servo_alignment(
        lambda: next(responses),
        lambda *args, **kwargs: moves.append((args, kwargs)),
        [0, 0, 200, 0, 0, 0],
        {"pixel_to_robot_matrix": [[1, 0], [0, 1]]},
        speed=25,
        error_threshold_px=2,
        max_step_mm=5,
        max_iter=2,
        success_stable_frames=1,
        max_missed_frames=2,
        settle_sec=0,
        event_callback=events.append,
        post_success_sample_frames=20,
    )

    assert result[0] is True
    assert result[1] == [0, 0, 200, 0, 0, 0]
    assert moves == []
    assert [row["事件"] for row in events].count("稳定帧") == 1
    assert [row["事件"] for row in events].count("成功后静止帧") == 20
    assert [row["静止采样序号"] for row in events[-20:]] == list(range(1, 21))


def test_alignment_logs_missing_target(capsys):
    run_offset_visual_servo_alignment(
        lambda: _response(found=False, message="未识别到方块"),
        lambda *_args, **_kwargs: None,
        [0, 0, 200, 0, 0, 0],
        {"pixel_to_robot_matrix": [[1, 0], [0, 1]]},
        speed=25,
        error_threshold_px=2,
        max_step_mm=5,
        max_iter=1,
        success_stable_frames=1,
        max_missed_frames=1,
        settle_sec=0,
        log_label="方块视觉伺服",
    )

    assert "[方块视觉伺服] 第 1 轮未识别，连续丢失 1/1: 未识别到方块" in capsys.readouterr().out


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
