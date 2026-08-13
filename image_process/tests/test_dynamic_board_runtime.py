"""动态盘面终端失败选择、原子报告和快照恢复测试。"""

import io
import json

from image_process_lib.dynamic_board_report import (
    deserialize_board_grid,
    deserialize_observed_block,
    serialize_observed_block,
)
from image_process_lib.dynamic_board_runtime import (
    atomic_write_json,
    prompt_dynamic_selection_failure,
    sha256_file,
)
from image_process_lib.task_planner import ObservedBlock


class _FakeTty(io.StringIO):
    def isatty(self):
        return True

    def fileno(self):
        return 123


def test_non_tty_eof_and_timeout_all_stop_safely():
    assert prompt_dynamic_selection_failure(
        "失败",
        input_stream=io.StringIO("f\n"),
        output_stream=io.StringIO(),
    ) == "stop"
    assert prompt_dynamic_selection_failure(
        "失败",
        input_stream=_FakeTty(""),
        output_stream=io.StringIO(),
        wait_readable=lambda *_args: ([_args[0][0]], [], []),
    ) == "stop"
    assert prompt_dynamic_selection_failure(
        "失败",
        input_stream=_FakeTty("f\n"),
        output_stream=io.StringIO(),
        wait_readable=lambda *_args: ([], [], []),
    ) == "stop"


def test_invalid_terminal_input_can_then_choose_fixed_yaml_fallback():
    output = io.StringIO()
    stream = _FakeTty("x\nf\n")

    choice = prompt_dynamic_selection_failure(
        "注入失败",
        input_stream=stream,
        output_stream=output,
        wait_readable=lambda *_args: ([stream], [], []),
    )

    assert choice == "fixed_yaml"
    assert "输入无效" in output.getvalue()


def test_atomic_json_and_sha256_roundtrip(tmp_path):
    path = tmp_path / "动态盘面选择报告.json"
    document = {"协议版本": 1, "结果": ["成功", 34]}

    written = atomic_write_json(path, document)

    assert written == path
    assert json.loads(path.read_text(encoding="utf-8")) == document
    assert len(sha256_file(path)) == 64
    assert not list(tmp_path.glob(".*.tmp"))


def test_observed_block_and_14_by_10_grid_snapshot_restore_exactly():
    block = ObservedBlock(
        category="T",
        observation_pose=(-300.0, 20.0, 200.0, -180.0, 0.0, 90.0),
        detected_angle_deg=17.5,
        source_id=9,
        pick_surface_z_mm=8.2,
        pick_surface_z_valid=True,
        high_detected_pixel_xy=(123.5, 456.5),
        high_depth_sample_pixel_xy=(124.0, 457.0),
        high_image_center_xy=(640.0, 360.0),
        high_world_position=(-1.0, 2.0, 3.0),
        high_world_position_valid=True,
        rough_localization_source="tcp_calibration",
        depth_valid_frame_count=15,
        depth_median_mm=500.0,
        depth_mad_mm=0.2,
        calibration_target_tcp_z_mm=200.0,
    )

    restored = deserialize_observed_block(serialize_observed_block(block))
    grid_array = [
        [[float(row), float(col)] for col in range(1, 11)]
        for row in range(1, 15)
    ]
    grid = deserialize_board_grid(grid_array)

    assert restored == block
    assert grid[1][1] == (1.0, 1.0)
    assert grid[14][10] == (14.0, 10.0)
