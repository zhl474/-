import importlib
import csv
import json
import sys
import types
from dataclasses import replace

import pytest

from competition_lib.config import load_execution_config, load_visual_servo_config


def _camera_to_sucker_offset():
    """现场标定值可由参数中心调整，运动测试始终读取当前配置。"""
    return [
        float(value)
        for value in load_visual_servo_config()["camera_to_sucker_offset_mm"]
    ]


def _execution_config(**changes):
    """测试不依赖现场正在使用的标定/正式模式开关。"""
    base = replace(
        load_execution_config(),
        calibration_mode=False,
        visual_servo_enabled=True,
    )
    return replace(base, **changes)


def _load_task_runner(monkeypatch):
    rospy = types.ModuleType("rospy")
    rospy.loginfo = lambda *_args, **_kwargs: None
    rospy.logerr = lambda *_args, **_kwargs: None
    rospy.logwarn = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "rospy", rospy)

    clients_module = types.ModuleType("competition_lib.ros_clients")

    class RobotClients:
        SUCK = 0
        BLOW = 1
        OFF = 2

    clients_module.RobotClients = RobotClients
    monkeypatch.setitem(sys.modules, "competition_lib.ros_clients", clients_module)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")
    sys.modules.pop("competition_lib.task_runner", None)
    return importlib.import_module("competition_lib.task_runner")


def _read_csv_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file_handle:
        return list(csv.DictReader(file_handle))


def _completed_rotation(module, angle_deg=180.0):
    """构造一个已经到期的摆放旋转记录，供只测试摆放动作的用例使用。"""
    return module.ServoRotationEstimate(
        start_angle_deg=float(angle_deg),
        target_angle_deg=float(angle_deg),
        estimated_duration_sec=0.0,
        command_accepted_at=0.0,
        ready_at=0.0,
    )


class _FakeClients:
    def __init__(self):
        self.moves = []
        self.rotation_angles = []
        self.suction_states = []
        self.prepare_requests = []
        self.actual_pose_calls = 0
        self.block_offset_requests = []
        self.board_offset_requests = []

    def get_task_target(self, _index):
        return types.SimpleNamespace(
            category="T",
            row=1.0,
            col=1.0,
            pick_observation_pose=[0, 0, 200, -180, 0, 90],
            place_observation_pose=[10, 10, 200, -180, 0, 90],
            detected_angle_deg=0.0,
            rotation_delta_deg=0.0,
            pick_surface_z_mm=8.0,
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

    def move_arm(
        self,
        pose,
        speed,
        wait_sec=0.0,
        wait_until_stable=False,
        blend_radius_mm=None,
    ):
        self.moves.append(
            (list(pose), speed, wait_sec, wait_until_stable, blend_radius_mm)
        )

    def rotate_tool(self, angle):
        self.rotation_angles.append(float(angle))
        return None

    def set_suction(self, state):
        self.suction_states.append(state)

    def prepare_task(self, advanced=False, place_order=()):
        self.prepare_requests.append((advanced, list(place_order)))
        return types.SimpleNamespace(success=True, task_count=1, message="成功")

    def detect_block_offset(self, category, high_angle_deg):
        self.block_offset_requests.append((category, high_angle_deg))
        return types.SimpleNamespace(found=True)

    def detect_board_offset(self, row, col):
        self.board_offset_requests.append((row, col))
        return types.SimpleNamespace(found=True)

    def get_actual_pose(self):
        self.actual_pose_calls += 1
        return types.SimpleNamespace(
            success=True,
            tcp_pose=[1, 2, 3, -180, 0, 90],
            camera_pose=[4, 5, 6, -180, 0, 90],
        )


class _TargetTypeFakeClients(_FakeClients):
    """支持按任务序号返回 target_type 的标定测试客户端。"""

    def __init__(self, target_types):
        super().__init__()
        self.target_types = list(target_types)

    def get_task_target(self, index):
        target = super().get_task_target(index)
        target.target_type = self.target_types[index]
        return target


def test_formal_pick_alignment_failure_never_writes_csv_or_reads_actual_pose(
    monkeypatch, tmp_path
):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    runner = module.TaskRunner(
        clients=clients,
        execution_config=_execution_config(),
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


def test_闭环模式调用方块和托盘低位检测(monkeypatch):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    runner = module.TaskRunner(
        clients=clients,
        execution_config=_execution_config(),
        visual_config=load_visual_servo_config(),
    )

    thresholds = []

    def align_once(offset_func, start_pose, _label, error_threshold_px, **_kwargs):
        thresholds.append(error_threshold_px)
        offset_func()
        return True, list(start_pose), None, "成功"

    runner._align = align_once
    target = clients.get_task_target(0)
    target.pick_surface_z_mm = 8.0

    place_rotation = runner._pick(target)
    runner._place(target, place_rotation=place_rotation)

    assert clients.block_offset_requests == [("T", 0.0)]
    assert clients.board_offset_requests == [(1.0, 1.0)]
    assert thresholds == [
        runner.config.block_error_threshold_px,
        runner.config.tray_error_threshold_px,
    ]


def test_pick_uses_surface_height_and_configured_offset(monkeypatch):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    execution_config = _execution_config(pick_surface_offset_mm=-2.5)
    runner = module.TaskRunner(
        clients=clients,
        execution_config=execution_config,
        visual_config=load_visual_servo_config(),
    )
    runner._align = lambda *_args, **_kwargs: (True, [0, 0, 200, -180, 0, 90], None, "成功")

    target = clients.get_task_target(0)
    target.pick_surface_z_mm = 180.0
    runner._pick(target)

    assert clients.moves[2][0][2] == 177.5


@pytest.mark.parametrize("surface_z_mm", [8.0, 10.5])
def test_闭环抓取使用动态预抓取和托盘伺服高度抬升(
    monkeypatch,
    surface_z_mm,
):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    runner = module.TaskRunner(
        clients=clients,
        execution_config=_execution_config(),
        visual_config=load_visual_servo_config(),
    )
    runner._align = lambda *_args, **_kwargs: (
        True,
        [1, 2, 200, -180, 0, 90],
        None,
        "成功",
    )
    target = clients.get_task_target(0)
    target.pick_surface_z_mm = surface_z_mm

    runner._pick(target)

    expected_pick_z_mm = surface_z_mm + runner.config.pick_surface_offset_mm
    offset_x, offset_y = _camera_to_sucker_offset()
    pick_x, pick_y = 1.0 + offset_x, 2.0 + offset_y
    expected_poses = [
        [0, 0, 200, -180, 0, 90],
        [
            pick_x,
            pick_y,
            expected_pick_z_mm + runner.config.pick_approach_clearance_mm,
            -180,
            0,
            90,
        ],
        [pick_x, pick_y, expected_pick_z_mm, -180, 0, 90],
        [pick_x, pick_y, 200.0, -180, 0, 90],
    ]
    for move, expected_pose in zip(clients.moves, expected_poses):
        assert move[0] == pytest.approx(expected_pose)
    assert [move[1] for move in clients.moves] == [
        runner.config.arm_speed,
        runner.config.pick_approach_speed,
        runner.config.pick_speed,
        runner.config.arm_speed,
    ]
    assert [move[3] for move in clients.moves] == [True, True, False, False]
    assert [move[4] for move in clients.moves] == [None, None, None, 5.0]
    assert clients.suction_states == [module.RobotClients.SUCK]


def test_开环抓取直接到动态预抓取位并抬到托盘伺服高度(monkeypatch):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    execution_config = _execution_config(visual_servo_enabled=False)
    runner = module.TaskRunner(
        clients=clients,
        execution_config=execution_config,
        visual_config=load_visual_servo_config(),
    )
    runner._align = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("开环抓取不应调用视觉伺服")
    )
    target = clients.get_task_target(0)
    target.pick_surface_z_mm = 8.0

    runner._pick(target)

    expected_pick_z_mm = target.pick_surface_z_mm + execution_config.pick_surface_offset_mm
    offset_x, offset_y = _camera_to_sucker_offset()
    expected_poses = [
        [
            offset_x,
            offset_y,
            expected_pick_z_mm + execution_config.pick_approach_clearance_mm,
            -180.0,
            0.0,
            90.0,
        ],
        [offset_x, offset_y, expected_pick_z_mm, -180.0, 0.0, 90.0],
        [offset_x, offset_y, 200.0, -180.0, 0.0, 90.0],
    ]
    assert len(clients.moves) == len(expected_poses)
    for move, expected_pose in zip(clients.moves, expected_poses):
        assert move[0] == pytest.approx(expected_pose)
    assert [move[1] for move in clients.moves] == [
        execution_config.pick_approach_speed,
        execution_config.pick_speed,
        execution_config.arm_speed,
    ]
    assert [move[3] for move in clients.moves] == [True, False, False]
    assert [move[4] for move in clients.moves] == [None, None, 5.0]
    assert clients.suction_states == [module.RobotClients.SUCK]
    assert clients.block_offset_requests == []
    assert runner.state is module.TaskState.PICKING


def test_抓后抬升圆滑半径为零时恢复到点停止(monkeypatch):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    runner = module.TaskRunner(
        clients=clients,
        execution_config=_execution_config(
            visual_servo_enabled=False,
            pick_retreat_blend_radius_mm=0.0,
        ),
        visual_config=load_visual_servo_config(),
    )
    target = clients.get_task_target(0)

    runner._pick(target)

    assert clients.moves[-1][4] is None


def test_抓后抬升圆滑半径不小于抬升距离时在运动前拒绝(monkeypatch):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    runner = module.TaskRunner(
        clients=clients,
        execution_config=_execution_config(
            visual_servo_enabled=False,
            pick_retreat_blend_radius_mm=27.0,
        ),
        visual_config=load_visual_servo_config(),
    )
    target = clients.get_task_target(0)

    with pytest.raises(RuntimeError, match="圆滑半径必须小于竖直抬升距离"):
        runner._pick(target)

    assert clients.moves == []
    assert clients.suction_states == []


def test_开环抓取在运动前拒绝无效粗定位位姿(monkeypatch):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    runner = module.TaskRunner(
        clients=clients,
        execution_config=_execution_config(visual_servo_enabled=False),
        visual_config=load_visual_servo_config(),
    )
    target = clients.get_task_target(0)
    target.pick_observation_pose[0] = float("nan")

    with pytest.raises(RuntimeError, match="方块观察位"):
        runner._pick(target)

    assert clients.moves == []
    assert clients.suction_states == []


def test_place_keeps_dynamic_observation_height_and_directly_releases(monkeypatch):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    execution_config = _execution_config()
    runner = module.TaskRunner(
        clients=clients,
        execution_config=execution_config,
        visual_config=load_visual_servo_config(),
    )
    runner._align = lambda *_args, **_kwargs: (True, [11, 12, 234, -180, 0, 90], None, "成功")
    target = clients.get_task_target(0)

    runner._place(target, place_rotation=_completed_rotation(module))

    offset_x, offset_y = _camera_to_sucker_offset()
    assert clients.moves[0][0] == [10, 10, 200, -180, 0, 90]
    assert clients.moves[1][0] == pytest.approx(
        [11.0 + offset_x, 12.0 + offset_y, 234.0, -180.0, 0.0, 90.0]
    )
    assert [move[3] for move in clients.moves] == [True, True]
    assert clients.suction_states == [module.RobotClients.BLOW]
    assert not hasattr(execution_config, "place_high_z")
    assert not hasattr(execution_config, "place_down_z")
    assert not hasattr(execution_config, "place_lift_step_mm")


def test_开环摆放应用xy偏置并保留托盘高度(monkeypatch):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    execution_config = _execution_config(visual_servo_enabled=False)
    runner = module.TaskRunner(
        clients=clients,
        execution_config=execution_config,
        visual_config=load_visual_servo_config(),
    )
    runner._align = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("开环摆放不应调用视觉伺服")
    )

    runner._place(
        clients.get_task_target(0),
        place_rotation=_completed_rotation(module),
    )

    offset_x, offset_y = _camera_to_sucker_offset()
    assert len(clients.moves) == 1
    assert clients.moves[0][0] == pytest.approx(
        [10.0 + offset_x, 10.0 + offset_y, 200.0, -180.0, 0.0, 90.0]
    )
    assert clients.moves[0][3] is True
    assert clients.suction_states == [module.RobotClients.BLOW]
    assert clients.board_offset_requests == []
    assert runner.state is module.TaskState.PLACING


def test_pick_rejects_missing_surface_height_before_moving(monkeypatch):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    target = clients.get_task_target(0)
    target.pick_surface_z_valid = False
    runner = module.TaskRunner(
        clients=clients,
        execution_config=_execution_config(),
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
        ("place_observation_pose", [0, 0, 160.0, -180, 0, 90], "托盘观察位"),
        ("pick_surface_z_mm", float("inf"), "方块抓取表面高度"),
        ("pick_surface_z_mm", -1.0, "最终抓取位"),
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
        execution_config=_execution_config(),
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
        execution_config=_execution_config(),
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
        execution_config=_execution_config(),
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
        execution_config=_execution_config(),
        visual_config=load_visual_servo_config(),
    )

    response = runner.prepare(advanced=True, place_order=[1, 2])

    assert response.success is True
    assert clients.moves[0][3] is True
    assert clients.prepare_requests == [(True, [1, 2])]


def test_prepare_actively_resets_servo_before_shooting(monkeypatch):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    events = []

    def rotate_tool(angle):
        clients.rotation_angles.append(float(angle))
        events.append(("舵机复位", float(angle)))

    def move_arm(pose, speed, **kwargs):
        clients.moves.append((list(pose), speed, 0.0, kwargs.get("wait_until_stable", False), None))
        events.append(("高位拍摄位", list(pose)))

    def prepare_task(advanced=False, place_order=()):
        events.append(("拍摄后规划", bool(advanced)))
        return types.SimpleNamespace(success=True, task_count=1, message="成功")

    clients.rotate_tool = rotate_tool
    clients.move_arm = move_arm
    clients.prepare_task = prepare_task
    monkeypatch.setattr(
        module.time,
        "sleep",
        lambda seconds: events.append(("最坏角差等待", float(seconds))),
    )
    config = _execution_config(initial_motor_angle_deg=180.0)
    runner = module.TaskRunner(
        clients=clients,
        execution_config=config,
        visual_config=load_visual_servo_config(),
    )
    runner.angle_planner.last_angle = 23.0

    runner.prepare()

    assert [event[0] for event in events] == [
        "舵机复位",
        "最坏角差等待",
        "高位拍摄位",
        "拍摄后规划",
    ]
    assert events[0][1] == 180.0
    assert events[1][1] == pytest.approx(180.0 / 270.0)
    assert runner.angle_planner.last_angle == 180.0


def test_calibration_mode_writes_one_final_row_per_target(monkeypatch, tmp_path, capsys):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    runner = module.TaskRunner(
        clients=clients,
        execution_config=_execution_config(calibration_mode=True),
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


def test_calibration_mode_writes_static_rounds_and_zero_error_tcp(monkeypatch, tmp_path):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    runner = module.TaskRunner(
        clients=clients,
        execution_config=_execution_config(
            calibration_mode=True,
            post_success_sample_frames=20,
        ),
        visual_config=load_visual_servo_config(),
        servo_csv_output_dir=tmp_path,
        experiment_session_id="static_test",
    )
    response = types.SimpleNamespace(
        found=True,
        px=321.0,
        py=242.0,
        dx_px=1.0,
        dy_px=2.0,
        detected_angle_deg=3.0,
        score=0.9,
    )

    def align_with_static_events(
        _func,
        pose,
        _label,
        _error_threshold_px,
        event_callback=None,
    ):
        event_callback({
            "事件": "稳定帧",
            "伺服轮次": 1,
            "连续稳定帧序号": 1,
            "像素误差X": 1.0,
            "像素误差Y": 2.0,
            "末端命令X": pose[0],
            "末端命令Y": pose[1],
            "末端命令Z": pose[2],
        })
        for index in range(20):
            event_callback({
                "事件": "成功后静止帧",
                "伺服轮次": 1,
                "静止采样序号": index + 1,
                "像素误差X": 1.0,
                "像素误差Y": 2.0,
                "末端命令X": pose[0],
                "末端命令Y": pose[1],
                "末端命令Z": pose[2],
            })
        return True, [1, 2, 200, -180, 0, 90], response, "成功"

    runner._align = align_with_static_events
    runner.execute_all(1)

    block_rows = _read_csv_rows(tmp_path / "方块视觉伺服.csv")
    round_rows = _read_csv_rows(
        tmp_path / "实验日志" / "static_test" / "方块视觉伺服逐轮.csv"
    )
    assert len(round_rows) == 21
    assert sum(row["事件"] == "成功后静止帧" for row in round_rows) == 20
    assert block_rows[0]["静止采样有效帧数"] == "20"
    assert block_rows[0]["静止采样完整"] == "True"
    # 当前矩阵 [dx,dy] -> [0.2*dy, 0.2*dx]。
    assert float(block_rows[0]["零误差等效TCP位置X"]) == pytest.approx(1.4)
    assert float(block_rows[0]["零误差等效TCP位置Y"]) == pytest.approx(2.2)
    assert block_rows[0]["实测相机位置X"] == "4.0"


def test_标定模式警告并强制开启视觉伺服(
    monkeypatch,
    tmp_path,
):
    module = _load_task_runner(monkeypatch)
    warnings = []
    monkeypatch.setattr(
        module.rospy,
        "logwarn",
        lambda message, *args: warnings.append(message % args if args else message),
    )
    clients = _FakeClients()
    runner = module.TaskRunner(
        clients=clients,
        execution_config=replace(
            _execution_config(),
            calibration_mode=True,
            visual_servo_enabled=False,
        ),
        visual_config=load_visual_servo_config(),
        servo_csv_output_dir=tmp_path,
    )
    align_labels = []
    runner._align = lambda _func, pose, label, _threshold, **_kwargs: align_labels.append(label) or (
        True,
        list(pose),
        None,
        "成功",
    )

    runner.execute_all(1)

    assert runner.visual_servo_enabled is True
    assert len(warnings) == 1
    assert "强制开启视觉伺服" in warnings[0]
    assert align_labels == ["方块视觉伺服", "托盘视觉伺服"]
    assert len(_read_csv_rows(tmp_path / "方块视觉伺服.csv")) == 1
    assert len(_read_csv_rows(tmp_path / "托盘视觉伺服.csv")) == 1
    assert clients.suction_states == [module.RobotClients.OFF]


def test_formal_mode_uses_suction_without_csv_or_actual_pose(monkeypatch, tmp_path):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    runner = module.TaskRunner(
        clients=clients,
        execution_config=_execution_config(),
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


def test_calibration_and_formal_modes_keep_same_motion_points_but_only_formal_blends(
    monkeypatch,
    tmp_path,
):
    module = _load_task_runner(monkeypatch)
    formal_clients = _FakeClients()
    calibration_clients = _FakeClients()

    def build_runner(clients, calibration_mode, output_dir):
        runner = module.TaskRunner(
            clients=clients,
            execution_config=replace(
                _execution_config(),
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

    assert [move[:4] for move in calibration_clients.moves] == [
        move[:4] for move in formal_clients.moves
    ]
    assert formal_clients.moves[3][4] == 5.0
    assert all(move[4] is None for move in calibration_clients.moves)
    assert calibration_clients.suction_states == [module.RobotClients.OFF]


def test_servo_finish_pose_read_failure_returns_blank_xyz(monkeypatch):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    clients.get_actual_pose = lambda: (_ for _ in ()).throw(RuntimeError("控制器离线"))
    runner = module.TaskRunner(
        clients=clients,
        execution_config=_execution_config(),
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
        execution_config=_execution_config(calibration_mode=True),
        visual_config=load_visual_servo_config(),
        servo_csv_output_dir=tmp_path,
    )
    runner._align = lambda *_args, **_kwargs: (False, [0, 0, 200, -180, 0, 90], None, "未识别")

    with pytest.raises(RuntimeError, match="方块视觉伺服失败"):
        runner.execute_all(1)

    block_rows = _read_csv_rows(tmp_path / "方块视觉伺服.csv")
    board_rows = _read_csv_rows(tmp_path / "托盘视觉伺服.csv")
    assert [row["事件"] for row in block_rows] == ["伺服失败"]
    assert block_rows[0]["失败信息"] == "未识别"
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
        execution_config=_execution_config(calibration_mode=True),
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
        execution_config=_execution_config(),
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


def test_high_safety_prepare_failure_prints_summary_without_coarse_prefix(
    monkeypatch,
    capsys,
):
    module = _load_task_runner(monkeypatch)
    message = (
        "高位初步安全检查失败：\n"
        "- 方块 2（T）：预测实际 TCP [-525.100, 80.300, 190.200]，X 越界\n"
        "本轮未打开 Mask 编辑器"
    )

    module.TaskRunner._print_prepare_failure(message)

    output = capsys.readouterr().out
    assert message in output
    assert "粗定位失败:" not in output


def test_completed_state_logs_total_execution_time(monkeypatch):
    module = _load_task_runner(monkeypatch)
    logs = []
    monkeypatch.setattr(module.rospy, "loginfo", lambda message, *args: logs.append(message % args))
    monkeypatch.setattr(module.time, "monotonic", lambda: 125.25)
    runner = module.TaskRunner(
        clients=_FakeClients(),
        execution_config=_execution_config(),
        visual_config=load_visual_servo_config(),
    )
    runner.execution_start_time = 120.0

    runner._set_state(module.TaskState.PICK_COARSE)
    assert logs == []

    runner._set_state(module.TaskState.COMPLETED)

    assert logs == ["任务状态: 完成", "全部方块抓放完成，总耗时: 5.25 秒"]
    assert runner.execution_start_time is None


def test_舵机命令服务返回后才开始预计计时(monkeypatch):
    module = _load_task_runner(monkeypatch)
    events = []

    def rotate_tool(angle):
        events.append(("舵机服务已返回", float(angle)))

    def monotonic():
        events.append(("记录开始时间", 50.0))
        return 50.0

    monkeypatch.setattr(module.time, "monotonic", monotonic)
    planner = module.ServoAnglePlanner(
        _execution_config(timing_debug=False),
        rotate_tool,
    )

    rotation = planner.commit_place_angle(270.0)

    assert events == [
        ("舵机服务已返回", 270.0),
        ("记录开始时间", 50.0),
    ]
    assert rotation.start_angle_deg == 180.0
    assert rotation.target_angle_deg == 270.0
    assert rotation.estimated_duration_sec == pytest.approx(90.0 / 270.0)
    assert rotation.command_accepted_at == 50.0
    assert rotation.ready_at == pytest.approx(50.0 + 90.0 / 270.0)


def test_舵机命令失败时不更新软件角度(monkeypatch):
    module = _load_task_runner(monkeypatch)

    def rotate_failed(_angle):
        raise RuntimeError("串口发送失败")

    planner = module.ServoAnglePlanner(
        _execution_config(timing_debug=False),
        rotate_failed,
    )

    with pytest.raises(RuntimeError, match="串口发送失败"):
        planner.commit_place_angle(270.0)

    assert planner.last_angle == 180.0


@pytest.mark.parametrize(
    ("now", "expected_remaining"),
    [(10.3, 0.2), (10.8, 0.0)],
)
def test_舵机等待只补足未被其它步骤覆盖的时间(
    monkeypatch,
    now,
    expected_remaining,
):
    module = _load_task_runner(monkeypatch)
    runner = module.TaskRunner(
        clients=_FakeClients(),
        execution_config=_execution_config(timing_debug=False),
        visual_config=load_visual_servo_config(),
    )
    rotation = module.ServoRotationEstimate(
        start_angle_deg=180.0,
        target_angle_deg=315.0,
        estimated_duration_sec=0.5,
        command_accepted_at=10.0,
        ready_at=10.5,
    )
    sleep_calls = []
    monkeypatch.setattr(module.time, "monotonic", lambda: now)
    monkeypatch.setattr(module.time, "sleep", sleep_calls.append)

    remaining_sec = runner._wait_for_motor_rotation(rotation, "测试阶段")

    assert remaining_sec == pytest.approx(expected_remaining)
    assert sleep_calls == pytest.approx(
        [expected_remaining] if expected_remaining > 0.0 else []
    )


@pytest.mark.parametrize("visual_servo_enabled", [True, False])
def test_抓取前预旋转统一在下探前检查(
    monkeypatch,
    visual_servo_enabled,
):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    events = []

    original_move_arm = clients.move_arm

    def move_arm(*args, **kwargs):
        original_move_arm(*args, **kwargs)
        events.append(f"机械臂运动{len(clients.moves)}")

    def rotate_tool(angle):
        clients.rotation_angles.append(float(angle))
        events.append(f"舵机旋转{float(angle):.0f}")

    clients.move_arm = move_arm
    clients.rotate_tool = rotate_tool
    runner = module.TaskRunner(
        clients=clients,
        execution_config=_execution_config(
            timing_debug=False,
            visual_servo_enabled=visual_servo_enabled,
        ),
        visual_config=load_visual_servo_config(),
    )
    runner.angle_planner.last_angle = 350.0
    target = clients.get_task_target(0)
    target.pick_surface_z_mm = 8.0
    target.rotation_delta_deg = 20.0

    if visual_servo_enabled:
        runner._align = lambda *_args, **_kwargs: (
            events.append("方块视觉伺服")
            or (True, [0, 0, 200, -180, 0, 90], None, "成功")
        )
    else:
        runner._align = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("开环抓取不应调用视觉伺服")
        )

    def wait_for_rotation(rotation, purpose):
        assert rotation is not None
        events.append(f"检查{purpose}")
        return 0.0

    runner._wait_for_motor_rotation = wait_for_rotation
    runner._pick(target)

    if visual_servo_enabled:
        assert events[:6] == [
            "舵机旋转330",
            "机械臂运动1",
            "方块视觉伺服",
            "机械臂运动2",
            "检查抓取前预旋转",
            "机械臂运动3",
        ]
    else:
        assert events[:4] == [
            "舵机旋转330",
            "机械臂运动1",
            "检查抓取前预旋转",
            "机械臂运动2",
        ]


def test_摆放旋转统一在喷气前检查(monkeypatch):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    events = []

    original_move_arm = clients.move_arm
    original_set_suction = clients.set_suction

    def move_arm(*args, **kwargs):
        original_move_arm(*args, **kwargs)
        events.append(f"机械臂运动{len(clients.moves)}")

    def rotate_tool(angle):
        clients.rotation_angles.append(float(angle))
        events.append(f"舵机旋转{float(angle):.0f}")

    def set_suction(state):
        original_set_suction(state)
        events.append(f"吸盘状态{state}")

    clients.move_arm = move_arm
    clients.rotate_tool = rotate_tool
    clients.set_suction = set_suction
    runner = module.TaskRunner(
        clients=clients,
        execution_config=_execution_config(
            timing_debug=False,
            visual_servo_enabled=False,
        ),
        visual_config=load_visual_servo_config(),
    )
    target = clients.get_task_target(0)
    target.pick_surface_z_mm = 8.0
    target.rotation_delta_deg = 90.0

    def wait_for_rotation(rotation, purpose):
        events.append(f"检查{purpose}")
        if purpose == "抓取前预旋转":
            assert rotation is None
        else:
            assert rotation.target_angle_deg == 270.0
            assert rotation.estimated_duration_sec == pytest.approx(90.0 / 270.0)
        return 0.0

    runner._wait_for_motor_rotation = wait_for_rotation
    place_rotation = runner._pick(target)
    runner._place(target, place_rotation=place_rotation)

    assert clients.moves[2][4] == 5.0
    assert events.index(f"吸盘状态{module.RobotClients.SUCK}") < events.index(
        "机械臂运动3"
    )
    assert events.index("机械臂运动3") < events.index("舵机旋转270")
    assert events.index("舵机旋转270") < events.index("机械臂运动4")
    assert events.index("机械臂运动4") < events.index("检查抓取后摆放旋转")
    assert events.index("检查抓取后摆放旋转") < events.index(
        f"吸盘状态{module.RobotClients.BLOW}"
    )


def test_缺少摆放旋转记录时禁止喷气(monkeypatch):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    runner = module.TaskRunner(
        clients=clients,
        execution_config=_execution_config(
            timing_debug=False,
            visual_servo_enabled=False,
        ),
        visual_config=load_visual_servo_config(),
    )

    with pytest.raises(RuntimeError, match="缺少本次舵机旋转记录"):
        runner._place(clients.get_task_target(0))

    assert module.RobotClients.BLOW not in clients.suction_states


def test_任务步骤计时输出中文标签和耗时(monkeypatch):
    module = _load_task_runner(monkeypatch)
    runner = module.TaskRunner(
        clients=_FakeClients(),
        execution_config=_execution_config(timing_debug=True),
        visual_config=load_visual_servo_config(),
    )
    clock_values = iter([20.0, 20.125])
    logs = []
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock_values))
    monkeypatch.setattr(
        module.rospy,
        "loginfo",
        lambda message, *args: logs.append(message % args),
    )

    result = runner._timed_call("吸盘吸气服务", lambda: "完成")

    assert result == "完成"
    assert logs == ["任务步骤耗时：吸盘吸气服务=125.0 ms"]


def _calibration_runner(module, clients, tmp_path, **changes):
    return module.CalibrationTaskRunner(
        clients=clients,
        execution_config=_execution_config(**changes),
        visual_config=load_visual_servo_config(),
        servo_csv_output_dir=tmp_path,
    )


def test_标定方块下探后退回自身观察高度且不读取全零托盘位(monkeypatch, tmp_path):
    module = _load_task_runner(monkeypatch)
    clients = _FakeClients()
    runner = _calibration_runner(module, clients, tmp_path)
    runner._align = lambda *_args, **_kwargs: (
        True,
        [1, 2, 200, -180, 0, 90],
        None,
        "成功",
    )
    target = clients.get_task_target(0)
    target.target_type = "block"
    target.pick_surface_z_mm = 8.0
    target.place_observation_pose = [0.0] * 6

    runner._pick(target)

    expected_pick_z_mm = target.pick_surface_z_mm + runner.config.pick_surface_offset_mm
    offset_x, offset_y = _camera_to_sucker_offset()
    pick_x, pick_y = 1.0 + offset_x, 2.0 + offset_y
    expected_poses = [
        [0, 0, 200, -180, 0, 90],
        [
            pick_x,
            pick_y,
            expected_pick_z_mm + runner.config.pick_approach_clearance_mm,
            -180,
            0,
            90,
        ],
        [pick_x, pick_y, expected_pick_z_mm, -180, 0, 90],
        [pick_x, pick_y, 200.0, -180, 0, 90],
    ]
    for move, expected_pose in zip(clients.moves, expected_poses):
        assert move[0] == pytest.approx(expected_pose)
    assert [move[3] for move in clients.moves] == [True, True, False, False]
    assert all(move[4] is None for move in clients.moves)
    assert clients.suction_states == []


def test_calibration_runner_dispatches_block_then_tray(monkeypatch, tmp_path):
    module = _load_task_runner(monkeypatch)
    clients = _TargetTypeFakeClients(["block", "block", "tray", "tray"])
    runner = _calibration_runner(module, clients, tmp_path)

    def align_calling_offset(
        offset_func,
        start_pose,
        _label,
        _error_threshold_px,
        **_kwargs,
    ):
        offset_func()
        return True, list(start_pose), None, "成功"

    runner._align = align_calling_offset

    runner.execute_all(4, block_count=2, tray_count=2)

    assert [request[0] for request in clients.block_offset_requests] == ["T", "T"]
    assert clients.board_offset_requests == [(1.0, 1.0), (1.0, 1.0)]
    assert len(clients.moves) == 12
    assert clients.suction_states == [module.RobotClients.OFF]
    block_rows = _read_csv_rows(tmp_path / "方块视觉伺服.csv")
    board_rows = _read_csv_rows(tmp_path / "托盘视觉伺服.csv")
    assert [row["任务序号"] for row in block_rows] == ["1", "2"]
    assert [row["任务序号"] for row in board_rows] == ["1", "2"]
    assert [row["目标类型"] for row in block_rows] == ["方块", "方块"]
    assert [row["目标类型"] for row in board_rows] == ["托盘", "托盘"]


def test_calibration_runner_rejects_unknown_target_type_before_motion(monkeypatch, tmp_path):
    module = _load_task_runner(monkeypatch)
    clients = _TargetTypeFakeClients(["bogus"])
    runner = _calibration_runner(module, clients, tmp_path)

    with pytest.raises(RuntimeError, match="未知目标类型"):
        runner.execute_all(1)

    assert clients.moves == []
    assert clients.block_offset_requests == []
    assert clients.board_offset_requests == []
    assert clients.suction_states == [module.RobotClients.OFF]
    assert runner.state is module.TaskState.FAILED
    assert runner.servo_csv_logger.is_open is False


def test_calibration_runner_metadata_records_planned_counts(monkeypatch, tmp_path):
    module = _load_task_runner(monkeypatch)
    clients = _TargetTypeFakeClients(["block", "tray"])
    runner = module.CalibrationTaskRunner(
        clients=clients,
        execution_config=_execution_config(),
        visual_config=load_visual_servo_config(),
        servo_csv_output_dir=tmp_path,
        experiment_session_id="meta_test",
    )
    runner._align = lambda *_args, **_kwargs: (True, [1, 2, 200, -180, 0, 90], None, "成功")

    runner.execute_all(2, block_count=1, tray_count=1)

    metadata_path = tmp_path / "实验日志" / "meta_test" / "实验元数据.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["计划任务数量"] == 2
    assert metadata["计划方块数量"] == 1
    assert metadata["计划托盘数量"] == 1
    assert metadata["实验状态"] == "完成"


def test_calibration_runner_interactive_retries_and_passes_planned_counts(monkeypatch, tmp_path):
    module = _load_task_runner(monkeypatch)
    runner = _calibration_runner(module, _TargetTypeFakeClients([]), tmp_path)
    responses = iter([
        types.SimpleNamespace(
            success=False,
            task_count=0,
            block_count=0,
            tray_count=0,
            message="托盘格点不足",
        ),
        types.SimpleNamespace(
            success=True,
            task_count=5,
            block_count=2,
            tray_count=3,
            message="标定准备完成",
        ),
    ])
    prepare_calls = []
    runner.prepare = lambda **kwargs: prepare_calls.append(kwargs) or next(responses)
    executed = []
    runner.execute_all = lambda *args, **kwargs: executed.append((args, kwargs))

    # 依次回答：准备失败后重试、识别结果满意、开始标定。
    answers = iter(["", "1", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    runner.run_interactive()

    assert len(prepare_calls) == 2
    assert executed == [( (5,), {"block_count": 2, "tray_count": 3} )]


def test_formal_runner_rejects_calibration_target_type(monkeypatch, tmp_path):
    module = _load_task_runner(monkeypatch)
    clients = _TargetTypeFakeClients(["block"])
    runner = module.TaskRunner(
        clients=clients,
        execution_config=_execution_config(),
        visual_config=load_visual_servo_config(),
        servo_csv_output_dir=tmp_path,
    )

    with pytest.raises(RuntimeError, match="只接受 pick_place"):
        runner.execute_all(1)

    assert clients.moves == []
    assert clients.suction_states == []


def test_abort_token_blocks_followup_action_and_marks_runner_aborted(monkeypatch, tmp_path):
    module = _load_task_runner(monkeypatch)
    token = module.TaskAbortToken()
    state_events = []
    runner = module.TaskRunner(
        clients=_FakeClients(),
        execution_config=_execution_config(),
        visual_config=load_visual_servo_config(),
        servo_csv_output_dir=tmp_path,
        abort_token=token,
        state_callback=state_events.append,
    )
    token.request()

    with pytest.raises(module.TaskAbortedError):
        runner.execute_all(1)

    assert runner.state is module.TaskState.ABORTED
    assert state_events[-1]["state"] == "已中止"


def test_abort_requested_during_blocking_call_prevents_next_command(monkeypatch):
    module = _load_task_runner(monkeypatch)
    token = module.TaskAbortToken()
    runner = module.TaskRunner(
        clients=_FakeClients(),
        execution_config=_execution_config(),
        visual_config=load_visual_servo_config(),
        abort_token=token,
    )
    calls = []

    def blocking_operation():
        calls.append("当前命令")
        token.request()

    with pytest.raises(module.TaskAbortedError):
        runner._timed_call("模拟阻塞命令", blocking_operation)
    if not token.requested:
        calls.append("后续命令")

    assert calls == ["当前命令"]
