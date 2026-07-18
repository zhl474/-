import importlib
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
        )

    def move_arm(self, pose, speed, wait_sec=0.0, wait_until_stable=False):
        self.moves.append((list(pose), speed, wait_sec, wait_until_stable))

    def rotate_tool(self, _angle):
        return None

    def set_suction(self, state):
        self.suction_states.append(state)


def test_pick_alignment_failure_never_descends_or_starts_suction(monkeypatch):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    runner = module.TaskRunner(
        clients=clients,
        execution_config=load_execution_config(),
        visual_config=load_visual_servo_config(),
    )
    runner._align = lambda *_args, **_kwargs: (False, [], None, "未识别")

    with pytest.raises(RuntimeError, match="方块视觉伺服失败"):
        runner.execute_all(1)

    assert runner.state is module.TaskState.FAILED
    assert clients.suction_states == []
    assert len(clients.moves) == 1


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
