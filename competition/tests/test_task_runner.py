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
        self.prepare_requests = []
        self.actual_pose_calls = 0

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
            pick_high_depth_sample_pixel_xy=[0.0, 0.0],
            pick_high_image_center_xy=[640.0, 360.0],
            pick_high_world_position=[0.0, 0.0, 0.0],
            pick_high_world_position_valid=False,
            pick_rough_localization_source="tcp_calibration",
            place_high_detected_pixel_xy=[320.5, 420.5],
            place_high_depth_sample_pixel_xy=[0.0, 0.0],
            place_high_image_center_xy=[640.0, 360.0],
            place_high_world_position=[0.0, 0.0, 0.0],
            place_high_world_position_valid=False,
            place_rough_localization_source="tcp_calibration",
        )

    def move_arm(self, pose, speed, wait_sec=0.0, wait_until_stable=False):
        self.moves.append((list(pose), speed, wait_sec, wait_until_stable))

    def rotate_tool(self, _angle):
        return None

    def set_suction(self, state):
        self.suction_states.append(state)

    def prepare_task(self, advanced=False, place_order=()):
        self.prepare_requests.append((advanced, list(place_order)))
        return types.SimpleNamespace(success=True, task_count=1, message="成功")

    def get_actual_pose(self):
        self.actual_pose_calls += 1
        return types.SimpleNamespace(
            success=True,
            tcp_pose=[1, 2, 3, -180, 0, 90],
            camera_pose=[4, 5, 6, -180, 0, 90],
        )


def test_formal_pick_alignment_failure_never_writes_csv_or_reads_actual_pose(
    monkeypatch, tmp_path
):
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
    assert clients.actual_pose_calls == 0
    assert not (tmp_path / "方块视觉伺服.csv").exists()
    assert not (tmp_path / "托盘视觉伺服.csv").exists()


def test_pick_uses_surface_height_and_configured_offset(monkeypatch):
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


def test_pick_rejects_missing_surface_height_before_moving(monkeypatch):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    target = clients.get_task_target(0)
    target.pick_surface_z_valid = False
    runner = module.TaskRunner(
        clients=clients,
        execution_config=load_execution_config(),
        visual_config=load_visual_servo_config(),
    )

    with pytest.raises(RuntimeError, match="缺少有效抓取表面高度"):
        runner._pick(target)

    assert clients.moves == []
    assert clients.suction_states == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pick_observation_pose", [0, 0, float("nan"), -180, 0, 90], "方块观察位"),
        ("pick_observation_pose", [0, 0, 164.9, -180, 0, 90], "低于 TCP 安全下限"),
        ("pick_surface_z_mm", float("inf"), "方块抓取表面高度"),
        ("pick_surface_z_mm", 2.0, "最终抓取位"),
    ],
)
def test_pick_rejects_invalid_pose_or_height_before_moving(
    monkeypatch, field, value, message
):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    target = clients.get_task_target(0)
    setattr(target, field, value)
    runner = module.TaskRunner(
        clients=clients,
        execution_config=load_execution_config(),
        visual_config=load_visual_servo_config(),
    )

    with pytest.raises(RuntimeError, match=message):
        runner._pick(target)

    assert clients.moves == []
    assert clients.suction_states == []


def test_place_rejects_low_observation_height_before_moving(monkeypatch):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    target = clients.get_task_target(0)
    target.place_observation_pose[2] = 160.0
    runner = module.TaskRunner(
        clients=clients,
        execution_config=load_execution_config(),
        visual_config=load_visual_servo_config(),
    )

    with pytest.raises(RuntimeError, match="托盘观察位.*低于 TCP 安全下限"):
        runner._place(target)

    assert clients.moves == []
    assert clients.suction_states == []


def test_place_rejects_low_aligned_release_height_before_final_move(monkeypatch):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    runner = module.TaskRunner(
        clients=clients,
        execution_config=load_execution_config(),
        visual_config=load_visual_servo_config(),
    )
    runner._align = lambda *_args, **_kwargs: (
        True,
        [11, 12, 160, -180, 0, 90],
        None,
        "成功",
    )

    with pytest.raises(RuntimeError, match="最终摆放位.*低于 TCP 安全下限"):
        runner._place(clients.get_task_target(0))

    assert len(clients.moves) == 1
    assert clients.suction_states == []


def test_prepare_waits_for_shooting_pose_to_stabilize(monkeypatch):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    runner = module.TaskRunner(
        clients=clients,
        execution_config=load_execution_config(),
        visual_config=load_visual_servo_config(),
    )

    response = runner.prepare(advanced=True, place_order=[1, 2])

    assert response.success is True
    assert clients.moves[0][3] is True
    assert clients.prepare_requests == [(True, [1, 2])]


def test_calibration_mode_writes_one_final_row_per_target(monkeypatch, tmp_path, capsys):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    runner = module.TaskRunner(
        clients=clients,
        execution_config=replace(load_execution_config(), calibration_mode=True),
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
    assert "当前模式：标定采集模式" in output
    assert "标定 CSV 已覆盖创建" in output
    block_rows = _read_csv_rows(tmp_path / "方块视觉伺服.csv")
    board_rows = _read_csv_rows(tmp_path / "托盘视觉伺服.csv")
    assert [row["事件"] for row in block_rows] == ["伺服成功"]
    assert [row["事件"] for row in board_rows] == ["伺服成功"]
    assert block_rows[0]["高位检测像素X"] == "120.5"
    assert block_rows[0]["实测TCP位置X"] == "1.0"
    assert board_rows[0]["高位检测像素X"] == "320.5"
    assert clients.actual_pose_calls == 2
    assert clients.suction_states == [module.RobotClients.OFF]


def test_formal_mode_uses_suction_without_csv_or_actual_pose(monkeypatch, tmp_path):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    runner = module.TaskRunner(
        clients=clients,
        execution_config=load_execution_config(),
        visual_config=load_visual_servo_config(),
        servo_csv_output_dir=tmp_path,
    )
    runner._align = lambda *_args, **_kwargs: (
        True,
        [1, 2, 200, -180, 0, 90],
        None,
        "成功",
    )

    runner.execute_all(1)

    assert clients.suction_states == [
        module.RobotClients.SUCK,
        module.RobotClients.BLOW,
        module.RobotClients.OFF,
    ]
    assert clients.actual_pose_calls == 0
    assert not (tmp_path / "方块视觉伺服.csv").exists()
    assert not (tmp_path / "托盘视觉伺服.csv").exists()


def test_calibration_and_formal_modes_keep_same_motion_sequence(monkeypatch, tmp_path):
    module = _load_task_runner(monkeypatch)
    formal_clients = _FakeClients()
    calibration_clients = _FakeClients()

    def build_runner(clients, calibration_mode, output_dir):
        runner = module.TaskRunner(
            clients=clients,
            execution_config=replace(
                load_execution_config(),
                calibration_mode=calibration_mode,
            ),
            visual_config=load_visual_servo_config(),
            servo_csv_output_dir=output_dir,
        )
        runner._align = lambda *_args, **_kwargs: (
            True,
            [1, 2, 200, -180, 0, 90],
            None,
            "成功",
        )
        return runner

    build_runner(formal_clients, False, tmp_path / "formal").execute_all(1)
    build_runner(calibration_clients, True, tmp_path / "calibration").execute_all(1)

    assert calibration_clients.moves == formal_clients.moves
    assert calibration_clients.suction_states == [module.RobotClients.OFF]


def test_servo_finish_pose_read_failure_returns_blank_xyz(monkeypatch):
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
    assert result["实测TCP位置Y"] == ""
    assert result["实测TCP位置Z"] == ""


def test_calibration_failure_writes_one_failure_row_and_closes_csv(monkeypatch, tmp_path):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    runner = module.TaskRunner(
        clients=clients,
        execution_config=replace(load_execution_config(), calibration_mode=True),
        visual_config=load_visual_servo_config(),
        servo_csv_output_dir=tmp_path,
    )
    runner._align = lambda *_args, **_kwargs: (False, [0, 0, 200, -180, 0, 90], None, "未识别")

    with pytest.raises(RuntimeError, match="方块视觉伺服失败"):
        runner.execute_all(1)

    block_rows = _read_csv_rows(tmp_path / "方块视觉伺服.csv")
    board_rows = _read_csv_rows(tmp_path / "托盘视觉伺服.csv")
    assert [row["事件"] for row in block_rows] == ["伺服失败"]
    assert board_rows == []
    assert clients.suction_states == [module.RobotClients.OFF]
    assert clients.actual_pose_calls == 1
    assert runner.servo_csv_logger.is_open is False


def test_calibration_mode_refuses_motion_when_suction_off_fails(monkeypatch, tmp_path):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()

    def fail_to_turn_off(state):
        clients.suction_states.append(state)
        raise RuntimeError("吸盘无法关闭")

    clients.set_suction = fail_to_turn_off
    runner = module.TaskRunner(
        clients=clients,
        execution_config=replace(load_execution_config(), calibration_mode=True),
        visual_config=load_visual_servo_config(),
        servo_csv_output_dir=tmp_path,
    )

    with pytest.raises(RuntimeError, match="吸盘无法关闭"):
        runner.execute_all(1)

    assert clients.suction_states == [module.RobotClients.OFF]
    assert clients.moves == []
    assert clients.actual_pose_calls == 0
    assert runner.state is module.TaskState.FAILED
    assert runner.servo_csv_logger.is_open is False


def test_interactive_prepare_retries_after_failed_high_localization(monkeypatch):
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
