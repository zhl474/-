import importlib
import csv
import sys
import types
from dataclasses import replace

import pytest

from competition_lib.config import load_execution_config, load_visual_servo_config


def _load_task_runner(monkeypatch):
    rospy = types.ModuleType("rospy")
    rospy.loginfo = lambda *_args, **_kwargs: None
    rospy.logerr = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "rospy", rospy)

    clients_module = types.ModuleType("competition_lib.ros_clients")

    class RobotClients:
        SUCK = 0
        BLOW = 1
        OFF = 2

    clients_module.RobotClients = RobotClients
    monkeypatch.setitem(sys.modules, "competition_lib.ros_clients", clients_module)
    sys.modules.pop("competition_lib.task_runner", None)
    return importlib.import_module("competition_lib.task_runner")


def _read_csv_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file_handle:
        return list(csv.DictReader(file_handle))


class _FakeClients:
    def __init__(self):
        self.moves = []
        self.suction_states = []

    def get_task_target(self, _index):
        return types.SimpleNamespace(
            category="T",
            row=1.0,
            col=1.0,
            pick_observation_pose=[0, 0, 200, -180, 0, 90],
            place_observation_pose=[10, 10, 200, -180, 0, 90],
            detected_angle_deg=0.0,
            rotation_delta_deg=0.0,
            pick_surface_z_mm=180.0,
            pick_surface_z_valid=True,
            pick_high_detected_pixel_xy=[120.5, 220.5],
            pick_high_depth_sample_pixel_xy=[120.0, 220.0],
            pick_high_image_center_xy=[640.0, 360.0],
            pick_high_world_position=[-100.0, 20.0, 180.0],
            pick_high_world_position_valid=True,
            pick_rough_localization_source="depth",
            place_high_detected_pixel_xy=[320.5, 420.5],
            place_high_depth_sample_pixel_xy=[320.0, 420.0],
            place_high_image_center_xy=[640.0, 360.0],
            place_high_world_position=[-200.0, 30.0, 160.0],
            place_high_world_position_valid=True,
            place_rough_localization_source="depth",
        )

    def move_arm(self, pose, speed, wait_sec=0.0, wait_until_stable=False):
        self.moves.append((list(pose), speed, wait_sec, wait_until_stable))

    def rotate_tool(self, _angle):
        return None

    def set_suction(self, state):
        self.suction_states.append(state)

    def get_actual_pose(self):
        return types.SimpleNamespace(
            success=True,
            tcp_pose=[1, 2, 3, -180, 0, 90],
            camera_pose=[4, 5, 6, -180, 0, 90],
        )


def test_pick_alignment_failure_never_descends_or_starts_suction(monkeypatch, tmp_path):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    runner = module.TaskRunner(
        clients=clients,
        execution_config=load_execution_config(),
        visual_config=load_visual_servo_config(),
        servo_csv_output_dir=tmp_path,
    )
    runner._align = lambda *_args, **_kwargs: (False, [], None, "未识别")

    with pytest.raises(RuntimeError, match="方块视觉伺服失败"):
        runner.execute_all(1)

    assert runner.state is module.TaskState.FAILED
    assert clients.suction_states == []
    assert len(clients.moves) == 1
    rows = _read_csv_rows(tmp_path / "方块视觉伺服.csv")
    assert [row["事件"] for row in rows] == ["伺服开始", "伺服失败"]


def test_pick_uses_depth_surface_height_and_configured_offset(monkeypatch):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    execution_config = replace(load_execution_config(), pick_surface_offset_mm=-2.5)
    runner = module.TaskRunner(
        clients=clients,
        execution_config=execution_config,
        visual_config=load_visual_servo_config(),
    )
    runner._align = lambda *_args, **_kwargs: (True, [0, 0, 200, -180, 0, 90], None, "成功")

    runner._pick(clients.get_task_target(0))

    assert clients.moves[2][0][2] == 177.5


def test_place_keeps_dynamic_observation_height_and_directly_releases(monkeypatch):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    execution_config = load_execution_config()
    runner = module.TaskRunner(
        clients=clients,
        execution_config=execution_config,
        visual_config=load_visual_servo_config(),
    )
    runner._align = lambda *_args, **_kwargs: (True, [11, 12, 234, -180, 0, 90], None, "成功")
    target = clients.get_task_target(0)

    runner._place(target)

    assert clients.moves[0][0] == [10, 10, 200, -180, 0, 90]
    assert clients.moves[1][0] == pytest.approx([-83.1, -1.8, 234.0, -180.0, 0.0, 90.0])
    assert [move[3] for move in clients.moves] == [True, False]
    assert clients.suction_states == [module.RobotClients.BLOW]
    assert not hasattr(execution_config, "place_high_z")
    assert not hasattr(execution_config, "place_down_z")
    assert not hasattr(execution_config, "place_lift_step_mm")


def test_pick_rejects_missing_depth_height_before_moving(monkeypatch):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    target = clients.get_task_target(0)
    target.pick_surface_z_valid = False
    runner = module.TaskRunner(
        clients=clients,
        execution_config=load_execution_config(),
        visual_config=load_visual_servo_config(),
    )

    with pytest.raises(RuntimeError, match="缺少有效深度高度"):
        runner._pick(target)

    assert clients.moves == []
    assert clients.suction_states == []


def test_execute_all_writes_block_and_board_csv_diagnostics(monkeypatch, tmp_path, capsys):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    runner = module.TaskRunner(
        clients=clients,
        execution_config=load_execution_config(),
        visual_config=load_visual_servo_config(),
        servo_csv_output_dir=tmp_path,
    )
    low_response = types.SimpleNamespace(found=True, px=323.0, py=238.0, dx_px=3.0, dy_px=-2.0)
    runner._align = lambda *_args, **_kwargs: (
        True,
        [1, 2, 200, -180, 0, 90],
        low_response,
        "成功",
    )

    runner.execute_all(1)

    output = capsys.readouterr().out
    assert "视觉伺服 CSV 已覆盖创建" in output
    assert "最终低位目标像素" not in output
    block_rows = _read_csv_rows(tmp_path / "方块视觉伺服.csv")
    board_rows = _read_csv_rows(tmp_path / "托盘视觉伺服.csv")
    assert [row["事件"] for row in block_rows] == ["伺服开始", "伺服成功"]
    assert [row["事件"] for row in board_rows] == ["伺服开始", "伺服成功"]
    block_result = block_rows[-1]
    assert block_result["高位检测像素X"] == "120.5"
    assert block_result["高位世界坐标X"] == "-100.0"
    assert block_result["粗定位来源"] == "depth"
    assert block_result["低位目标像素X"] == "323.0"
    assert block_result["低位图像中心X"] == "320.0"
    assert block_result["实测TCP位置X"] == "1.0"
    assert block_result["实测相机光心位置X"] == "4.0"
    assert board_rows[-1]["高位检测像素X"] == "320.5"


def test_execute_all_records_correction_and_stable_frame_rows(monkeypatch, tmp_path):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    block_responses = iter([
        types.SimpleNamespace(found=True, px=323.0, py=238.0, dx_px=3.0, dy_px=-2.0, message="修正"),
        types.SimpleNamespace(found=True, px=320.0, py=240.0, dx_px=0.0, dy_px=0.0, message="对准"),
    ])
    board_responses = iter([
        types.SimpleNamespace(found=True, px=317.0, py=241.0, dx_px=-3.0, dy_px=1.0, message="修正"),
        types.SimpleNamespace(found=True, px=320.0, py=240.0, dx_px=0.0, dy_px=0.0, message="对准"),
    ])
    clients.detect_block_offset = lambda *_args: next(block_responses)
    clients.detect_board_offset = lambda *_args: next(board_responses)
    execution_config = replace(
        load_execution_config(),
        error_threshold_px=1.0,
        success_stable_frames=1,
        max_iter=2,
        settle_sec=0.0,
    )
    runner = module.TaskRunner(
        clients=clients,
        execution_config=execution_config,
        visual_config=load_visual_servo_config(),
        servo_csv_output_dir=tmp_path,
    )

    runner.execute_all(1)

    block_rows = _read_csv_rows(tmp_path / "方块视觉伺服.csv")
    board_rows = _read_csv_rows(tmp_path / "托盘视觉伺服.csv")
    assert [row["事件"] for row in block_rows] == ["伺服开始", "执行修正", "稳定帧", "伺服成功"]
    assert [row["事件"] for row in board_rows] == ["伺服开始", "执行修正", "稳定帧", "伺服成功"]
    assert block_rows[1]["XY修正X毫米"]
    assert block_rows[1]["低位图像中心X"] == "320.0"
    assert block_rows[1]["图像服务耗时毫秒"]
    assert board_rows[1]["低位图像中心Y"] == "240.0"


def test_servo_finish_pose_read_failure_writes_message_without_raising(monkeypatch):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    clients.get_actual_pose = lambda: (_ for _ in ()).throw(RuntimeError("控制器离线"))
    runner = module.TaskRunner(
        clients=clients,
        execution_config=load_execution_config(),
        visual_config=load_visual_servo_config(),
    )
    result = runner._read_actual_pose_for_csv()

    assert result["实测TCP位置X"] == ""
    assert result["实测相机光心位置X"] == ""
    assert result["实测位姿读取信息"] == "读取异常: 控制器离线"


def test_fallback_world_coordinates_are_blank_in_csv(monkeypatch, tmp_path):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    target = clients.get_task_target(0)
    target.pick_high_world_position_valid = False
    target.pick_rough_localization_source = "fallback"
    runner = module.TaskRunner(
        clients=clients,
        execution_config=load_execution_config(),
        visual_config=load_visual_servo_config(),
        servo_csv_output_dir=tmp_path,
    )
    runner._align = lambda *_args, **_kwargs: (True, [0, 0, 200, -180, 0, 90], None, "成功")
    clients.get_task_target = lambda _index: target

    runner.execute_all(1)

    row = _read_csv_rows(tmp_path / "方块视觉伺服.csv")[0]
    assert row["高位世界坐标有效"] == "False"
    assert row["粗定位来源"] == "fallback"
    assert row["高位世界坐标X"] == ""
    assert row["高位世界坐标Y"] == ""
    assert row["高位世界坐标Z"] == ""


def test_interactive_prepare_retries_after_failed_rough_localization(monkeypatch):
    module = _load_task_runner(monkeypatch)
    runner = module.TaskRunner(
        clients=_FakeClients(),
        execution_config=load_execution_config(),
        visual_config=load_visual_servo_config(),
    )
    responses = iter([
        types.SimpleNamespace(success=False, task_count=0, message="托盘格点不足"),
        types.SimpleNamespace(success=True, task_count=3, message="粗定位成功"),
    ])
    prepare_calls = []
    runner.prepare = lambda **kwargs: prepare_calls.append(kwargs) or next(responses)
    executed_counts = []
    runner.execute_all = executed_counts.append

    # 依次回答：基础任务、失败后重试、识别结果满意、开始抓放。
    answers = iter(["", "", "1", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    runner.run_interactive()

    assert len(prepare_calls) == 2
    assert executed_counts == [3]


def test_completed_state_logs_total_execution_time(monkeypatch):
    module = _load_task_runner(monkeypatch)
    logs = []
    monkeypatch.setattr(module.rospy, "loginfo", lambda message, *args: logs.append(message % args))
    monkeypatch.setattr(module.time, "monotonic", lambda: 125.25)
    runner = module.TaskRunner(
        clients=_FakeClients(),
        execution_config=load_execution_config(),
        visual_config=load_visual_servo_config(),
    )
    runner.execution_start_time = 120.0

    runner._set_state(module.TaskState.PICK_COARSE)
    assert logs == []

    runner._set_state(module.TaskState.COMPLETED)

    assert logs == ["任务状态: 完成", "全部方块抓放完成，总耗时: 5.25 秒"]
    assert runner.execution_start_time is None
