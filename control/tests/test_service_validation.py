import importlib.util
import os
import sys
import types

import pytest


class _Response:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _load_controller(monkeypatch):
    rospy = types.ModuleType("rospy")
    rospy.logerr = lambda *_args, **_kwargs: None
    rospy.loginfo = lambda *_args, **_kwargs: None
    rospy.logwarn = lambda *_args, **_kwargs: None
    rospy.is_shutdown = lambda: False
    monkeypatch.setitem(sys.modules, "rospy", rospy)

    akai_fr = types.ModuleType("akai_fr")
    akai_fr.AkaiFr = object
    akai_fr.AkaiElectricSucker = object
    monkeypatch.setitem(sys.modules, "akai_fr", akai_fr)

    control = types.ModuleType("control")
    control.srv = types.ModuleType("control.srv")
    for name in (
        "ClearArmStop",
        "GetActualPose",
        "GetControlStatus",
        "MoveArm",
        "RotateTool",
        "SetSuction",
        "StopArm",
    ):
        setattr(control.srv, name, object)
    for name in (
        "ClearArmStopResponse",
        "GetActualPoseResponse",
        "GetControlStatusResponse",
        "MoveArmResponse",
        "RotateToolResponse",
        "SetSuctionResponse",
        "StopArmResponse",
    ):
        setattr(control.srv, name, _Response)
    monkeypatch.setitem(sys.modules, "control", control)
    monkeypatch.setitem(sys.modules, "control.srv", control.srv)

    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "controller.py"))
    spec = importlib.util.spec_from_file_location("controller_validation", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_controller_reads_the_single_execution_height_without_fallback(monkeypatch, tmp_path):
    module = _load_controller(monkeypatch)
    config_path = tmp_path / "execution.yaml"
    config_path.write_text("motion:\n  minimum_tcp_z_mm: 164.0\n", encoding="utf-8")

    assert module._load_minimum_tcp_z_mm(config_path) == pytest.approx(164.0)

    config_path.write_text("motion:\n  pick_speed: 50\n", encoding="utf-8")
    with pytest.raises(ValueError, match="缺少唯一安全高度字段"):
        module._load_minimum_tcp_z_mm(config_path)


def test_low_pose_is_clamped_but_reports_failure(monkeypatch):
    module = _load_controller(monkeypatch)
    calls = []
    node = object.__new__(module.ControlNode)
    node.minimum_tcp_z_mm = 163.0
    node.arm = types.SimpleNamespace(
        set_speed=lambda _speed: calls.append("set_speed"),
        arm=types.SimpleNamespace(
            MoveL=lambda pose, **_kwargs: (calls.append(("MoveL", pose)), 0)[1]
        ),
    )
    def wait_until_stable(pose, _started_at):
        calls.append(("wait_stable", pose))
        return 0.0, 0.0

    node._wait_until_arm_stable = wait_until_stable
    request = types.SimpleNamespace(
        pose=[0, 0, 100, 0, 0, 0], speed=40, wait_until_stable=True
    )

    response = node.move_arm(request)

    assert response.success is False
    assert calls == [
        "set_speed",
        ("MoveL", [0.0, 0.0, 163.0, 0.0, 0.0, 0.0]),
        ("wait_stable", [0.0, 0.0, 163.0, 0.0, 0.0, 0.0]),
    ]
    assert "请求 Z=100.00 mm 低于安全下限 163.00 mm" in response.message
    assert "已调整到 163.00 mm" in response.message
    assert "本次运动判定失败" in response.message


def test_non_finite_pose_never_calls_robot(monkeypatch):
    module = _load_controller(monkeypatch)
    calls = []
    node = object.__new__(module.ControlNode)
    node.minimum_tcp_z_mm = 163.0
    node.arm = types.SimpleNamespace(
        set_speed=lambda _speed: calls.append("set_speed"),
        arm=types.SimpleNamespace(MoveL=lambda *_args, **_kwargs: calls.append("MoveL")),
    )
    request = types.SimpleNamespace(
        pose=[0, 0, float("nan"), 0, 0, 0], speed=40, wait_until_stable=False
    )

    response = node.move_arm(request)

    assert response.success is False
    assert calls == []


def test_get_actual_pose_returns_tcp_and_camera_pose(monkeypatch):
    module = _load_controller(monkeypatch)
    tcp_pose = [-250.0, 0.0, 200.0, 180.0, 0.0, 90.0]
    camera_pose = [-220.0, 3.0, 250.0, 180.0, 0.0, 90.0]
    node = object.__new__(module.ControlNode)
    node.arm = types.SimpleNamespace(
        arm=types.SimpleNamespace(GetActualTCPPose=lambda: (0, list(tcp_pose))),
        get_camera_pose=lambda: (True, list(camera_pose)),
    )

    response = node.get_actual_pose(None)

    assert response.success is True
    assert response.tcp_pose == tcp_pose
    assert response.camera_pose == camera_pose


def test_get_actual_pose_failure_returns_unsuccessful_response(monkeypatch):
    module = _load_controller(monkeypatch)
    node = object.__new__(module.ControlNode)
    node.arm = types.SimpleNamespace(
        arm=types.SimpleNamespace(GetActualTCPPose=lambda: (0, [0.0] * 6)),
        get_camera_pose=lambda: (False, None),
    )

    response = node.get_actual_pose(None)

    assert response.success is False
    assert "相机光心" in response.message


def test_wait_until_arm_stable_uses_motion_speed_and_pose(monkeypatch):
    module = _load_controller(monkeypatch)

    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    clock = FakeClock()
    monkeypatch.setattr(module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(module.time, "sleep", clock.sleep)

    target_pose = [-250.0, 0.0, 200.0, 180.0, 0.0, 90.0]
    xmlrpc_arm = types.SimpleNamespace(
        GetRobotMotionDone=lambda: (0, 1),
        GetActualTCPCompositeSpeed=lambda: (0, [0.5, 0.2]),
        GetActualTCPPose=lambda: (0, list(target_pose)),
    )
    node = object.__new__(module.ControlNode)
    node.arm = types.SimpleNamespace(arm=xmlrpc_arm)
    node.stable_timeout = 5.0
    node.stable_poll_interval = 0.005
    node.stable_duration = 0.01
    node.linear_speed_threshold = 3.0
    node.angular_speed_threshold = 3.0
    node.position_tolerance = 1.0
    node.orientation_tolerance = 0.5

    node._wait_until_arm_stable(target_pose)

    assert clock.now == 0.01


def test_wait_until_arm_stable_exits_immediately_when_stop_is_latched(monkeypatch):
    module = _load_controller(monkeypatch)
    node = object.__new__(module.ControlNode)
    node.stop_latched = True
    node.arm = types.SimpleNamespace(
        arm=types.SimpleNamespace(
            GetRobotMotionDone=lambda: (_ for _ in ()).throw(
                AssertionError("停止锁生效后不应继续轮询运动")
            )
        )
    )

    with pytest.raises(RuntimeError, match="停止锁"):
        node._wait_until_arm_stable([0, 0, 200, 0, 0, 0])


def test_move_arm_logs_motion_and_stabilization_timing(monkeypatch):
    module = _load_controller(monkeypatch)
    log_calls = []
    module.rospy.loginfo = lambda *args: log_calls.append(args)

    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    clock = FakeClock()
    monkeypatch.setattr(module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(module.time, "sleep", clock.sleep)

    target_pose = [-250.0, 0.0, 200.0, 180.0, 0.0, 90.0]
    motion_states = iter([0, 0, 1, 1, 1, 1])
    speed_states = iter(
        [
            [10.0, 0.2],
            [10.0, 0.2],
            [10.0, 0.2],
            [0.5, 0.2],
            [0.5, 0.2],
            [0.5, 0.2],
        ]
    )
    xmlrpc_arm = types.SimpleNamespace(
        MoveL=lambda *_args, **_kwargs: 0,
        GetRobotMotionDone=lambda: (0, next(motion_states)),
        GetActualTCPCompositeSpeed=lambda: (0, next(speed_states)),
        GetActualTCPPose=lambda: (0, list(target_pose)),
    )
    node = object.__new__(module.ControlNode)
    node.minimum_tcp_z_mm = 163.0
    node.move_arm_timing_debug = True
    node.arm = types.SimpleNamespace(set_speed=lambda _speed: None, arm=xmlrpc_arm)
    node.stable_timeout = 5.0
    node.stable_poll_interval = 0.005
    node.stable_duration = 0.01
    node.linear_speed_threshold = 3.0
    node.angular_speed_threshold = 3.0
    node.position_tolerance = 1.0
    node.orientation_tolerance = 0.5
    request = types.SimpleNamespace(pose=target_pose, speed=40, wait_until_stable=True)

    response = node.move_arm(request)

    assert response.success is True
    assert clock.now == pytest.approx(0.025)
    assert len(log_calls) == 1
    log_format, motion_ms, stabilization_ms, total_ms = log_calls[0]
    assert log_format == "机械臂阶段耗时：运动=%.1f ms，保稳=%.1f ms，总计=%.1f ms"
    assert motion_ms == pytest.approx(10.0)
    assert stabilization_ms == pytest.approx(15.0)
    assert total_ms == pytest.approx(25.0)


def test_move_arm_does_not_log_timing_when_debug_is_disabled(monkeypatch):
    module = _load_controller(monkeypatch)
    log_calls = []
    module.rospy.loginfo = lambda *args: log_calls.append(args)

    node = object.__new__(module.ControlNode)
    node.minimum_tcp_z_mm = 163.0
    node.move_arm_timing_debug = False
    node.arm = types.SimpleNamespace(
        set_speed=lambda _speed: None,
        arm=types.SimpleNamespace(MoveL=lambda *_args, **_kwargs: 0),
    )
    node._wait_until_arm_stable = lambda *_args: (0.01, 0.02)
    request = types.SimpleNamespace(
        pose=[0, 0, 200, 0, 0, 0], speed=40, wait_until_stable=True
    )

    response = node.move_arm(request)

    assert response.success is True
    assert log_calls == []


def test_move_arm_rejects_nonzero_movel_error(monkeypatch):
    module = _load_controller(monkeypatch)
    node = object.__new__(module.ControlNode)
    node.minimum_tcp_z_mm = 163.0
    node.arm = types.SimpleNamespace(
        set_speed=lambda _speed: None,
        arm=types.SimpleNamespace(MoveL=lambda *_args, **_kwargs: 14),
    )
    node._wait_until_arm_stable = lambda _pose: (_ for _ in ()).throw(
        AssertionError("MoveL 失败后不应等待停稳")
    )
    request = types.SimpleNamespace(
        pose=[0, 0, 200, 0, 0, 0], speed=40, wait_until_stable=False
    )

    response = node.move_arm(request)

    assert response.success is False
    assert "14" in response.message


def test_move_arm_skips_stability_wait_when_not_requested(monkeypatch):
    module = _load_controller(monkeypatch)
    calls = []
    log_calls = []
    module.rospy.loginfo = lambda *args: log_calls.append(args)
    node = object.__new__(module.ControlNode)
    node.minimum_tcp_z_mm = 163.0
    node.move_arm_timing_debug = True
    node.arm = types.SimpleNamespace(
        set_speed=lambda _speed: calls.append("set_speed"),
        arm=types.SimpleNamespace(MoveL=lambda _pose, **_kwargs: calls.append("MoveL") or 0),
    )
    node._wait_until_arm_stable = lambda _pose: calls.append("wait_stable")
    request = types.SimpleNamespace(
        pose=[0, 0, 200, 0, 0, 0], speed=40, wait_until_stable=False
    )

    response = node.move_arm(request)

    assert response.success is True
    assert calls == ["set_speed", "MoveL"]
    assert log_calls == []


def test_move_arm_passes_blocking_radius_for_normal_motion(monkeypatch):
    module = _load_controller(monkeypatch)
    move_kwargs = []
    node = object.__new__(module.ControlNode)
    node.minimum_tcp_z_mm = 163.0
    node.arm = types.SimpleNamespace(
        set_speed=lambda _speed: None,
        arm=types.SimpleNamespace(
            MoveL=lambda _pose, **kwargs: move_kwargs.append(kwargs) or 0
        ),
    )
    request = types.SimpleNamespace(
        pose=[0, 0, 200, 0, 0, 0],
        speed=40,
        wait_until_stable=False,
        blend_enabled=False,
        blend_radius_mm=0.0,
    )

    response = node.move_arm(request)

    assert response.success is True
    assert move_kwargs[0]["blendR"] == -1.0
    assert response.message == "机械臂运动完成"


def test_move_arm_submits_nonblocking_blended_motion(monkeypatch):
    module = _load_controller(monkeypatch)
    move_kwargs = []
    node = object.__new__(module.ControlNode)
    node.minimum_tcp_z_mm = 163.0
    node.arm = types.SimpleNamespace(
        set_speed=lambda _speed: None,
        arm=types.SimpleNamespace(
            MoveL=lambda _pose, **kwargs: move_kwargs.append(kwargs) or 0
        ),
    )
    request = types.SimpleNamespace(
        pose=[0, 0, 200, 0, 0, 0],
        speed=40,
        wait_until_stable=False,
        blend_enabled=True,
        blend_radius_mm=5.0,
    )

    response = node.move_arm(request)

    assert response.success is True
    assert move_kwargs[0]["blendR"] == 5.0
    assert response.message == "机械臂圆滑过渡运动已提交"


@pytest.mark.parametrize(
    "invalid_radius",
    [0.0, -1.0, 1000.1, float("inf"), float("nan")],
)
def test_move_arm_rejects_invalid_enabled_blend_radius(monkeypatch, invalid_radius):
    module = _load_controller(monkeypatch)
    calls = []
    node = object.__new__(module.ControlNode)
    node.minimum_tcp_z_mm = 163.0
    node.arm = types.SimpleNamespace(
        set_speed=lambda _speed: calls.append("set_speed"),
        arm=types.SimpleNamespace(MoveL=lambda *_args, **_kwargs: calls.append("MoveL")),
    )
    request = types.SimpleNamespace(
        pose=[0, 0, 200, 0, 0, 0],
        speed=40,
        wait_until_stable=False,
        blend_enabled=True,
        blend_radius_mm=invalid_radius,
    )

    response = node.move_arm(request)

    assert response.success is False
    assert "圆滑半径" in response.message
    assert calls == []


def test_move_arm_rejects_waiting_at_blended_waypoint(monkeypatch):
    module = _load_controller(monkeypatch)
    calls = []
    node = object.__new__(module.ControlNode)
    node.minimum_tcp_z_mm = 163.0
    node.arm = types.SimpleNamespace(
        set_speed=lambda _speed: calls.append("set_speed"),
        arm=types.SimpleNamespace(MoveL=lambda *_args, **_kwargs: calls.append("MoveL")),
    )
    request = types.SimpleNamespace(
        pose=[0, 0, 200, 0, 0, 0],
        speed=40,
        wait_until_stable=True,
        blend_enabled=True,
        blend_radius_mm=5.0,
    )

    response = node.move_arm(request)

    assert response.success is False
    assert "不能同时等待" in response.message
    assert calls == []


@pytest.mark.parametrize(
    ("state", "expected_calls", "expected_message"),
    [
        (0, [("电磁阀", False), ("气泵", True)], "吸盘开始吸气"),
        (1, [("电磁阀", True), ("气泵", True)], "吸盘开始喷气"),
        (2, [("电磁阀", False), ("气泵", False)], "吸盘已关闭"),
    ],
)
def test_set_suction_executes_real_hardware_sequence(
    monkeypatch, state, expected_calls, expected_message
):
    module = _load_controller(monkeypatch)
    calls = []
    node = object.__new__(module.ControlNode)
    node.sucker = types.SimpleNamespace(
        set_solenoid_valve=lambda enabled: calls.append(("电磁阀", enabled)),
        set_pump_motor=lambda enabled: calls.append(("气泵", enabled)),
    )
    request = types.SimpleNamespace(state=state, SUCK=0, BLOW=1, OFF=2)

    response = node.set_suction(request)

    assert response.success is True
    assert response.message == expected_message
    assert calls == expected_calls


def test_set_suction_rejects_unknown_state_without_hardware_action(monkeypatch):
    module = _load_controller(monkeypatch)
    calls = []
    node = object.__new__(module.ControlNode)
    node.sucker = types.SimpleNamespace(
        set_solenoid_valve=lambda enabled: calls.append(("电磁阀", enabled)),
        set_pump_motor=lambda enabled: calls.append(("气泵", enabled)),
    )
    request = types.SimpleNamespace(state=9, SUCK=0, BLOW=1, OFF=2)

    response = node.set_suction(request)

    assert response.success is False
    assert "未知吸盘状态" in response.message
    assert calls == []


def test_stop_arm_latches_before_calling_rpc_and_confirms_stop(monkeypatch):
    module = _load_controller(monkeypatch)
    calls = []

    class FakeRpc:
        def StopMotion(self):
            calls.append("StopMotion")
            assert node.stop_latched is True
            return 0

        def GetRobotMotionDone(self):
            return [0, 1]

        def GetActualTCPCompositeSpeed(self, flag):
            calls.append(("GetActualTCPCompositeSpeed", flag))
            return [0, 0.0, 0.0]

    node = object.__new__(module.ControlNode)
    node._create_stop_rpc = lambda: FakeRpc()

    response = node.stop_arm(None)

    assert response.success is True
    assert response.stop_latched is True
    assert node.stop_latched is True
    assert calls == ["StopMotion", ("GetActualTCPCompositeSpeed", 1)]


def test_stop_arm_failure_keeps_latch_and_warns_physical_estop(monkeypatch):
    module = _load_controller(monkeypatch)
    node = object.__new__(module.ControlNode)
    node._create_stop_rpc = lambda: types.SimpleNamespace(StopMotion=lambda: 9)

    response = node.stop_arm(None)

    assert response.success is False
    assert response.stop_latched is True
    assert node.stop_latched is True
    assert "物理急停" in response.message


def test_stop_arm_repeats_stop_until_racing_move_handler_finishes(monkeypatch):
    module = _load_controller(monkeypatch)
    calls = []

    class FakeRpc:
        def StopMotion(self):
            calls.append("StopMotion")
            return 0

        def GetRobotMotionDone(self):
            return [0, 1]

        def GetActualTCPCompositeSpeed(self, _flag):
            if calls.count("StopMotion") >= 2:
                node.motion_active = False
            return [0, 0.0, 0.0]

    node = object.__new__(module.ControlNode)
    node.motion_active = True
    node.stop_confirm_timeout = 0.2
    node.stable_poll_interval = 0.001
    node._create_stop_rpc = lambda: FakeRpc()

    response = node.stop_arm(None)

    assert response.success is True
    assert calls.count("StopMotion") >= 2


def test_move_that_overlaps_stop_returns_failure_before_next_action(monkeypatch):
    module = _load_controller(monkeypatch)
    node = object.__new__(module.ControlNode)
    node.stop_latched = False
    node.minimum_tcp_z_mm = 163.0
    node.arm = types.SimpleNamespace()
    node.arm.set_speed = lambda _speed: None

    def move_l(*_args, **_kwargs):
        node.stop_latched = True
        return 0

    node.arm.arm = types.SimpleNamespace(MoveL=move_l)
    response = node.move_arm(types.SimpleNamespace(
        pose=[0, 0, 200, 0, 0, 0], speed=20,
        wait_until_stable=False, blend_enabled=False, blend_radius_mm=0.0,
    ))

    assert response.success is False
    assert "停止锁中止" in response.message


def test_stop_latch_rejects_arm_and_servo_but_allows_suction(monkeypatch):
    module = _load_controller(monkeypatch)
    hardware_calls = []
    node = object.__new__(module.ControlNode)
    node.stop_latched = True
    node.minimum_tcp_z_mm = 163.0
    node.arm = types.SimpleNamespace(
        set_speed=lambda _speed: hardware_calls.append("set_speed"),
        arm=types.SimpleNamespace(MoveL=lambda *_args, **_kwargs: hardware_calls.append("MoveL")),
    )
    node.servo_serial = types.SimpleNamespace(
        write=lambda _data: hardware_calls.append("servo")
    )
    node.sucker = types.SimpleNamespace(
        set_solenoid_valve=lambda enabled: hardware_calls.append(("电磁阀", enabled)),
        set_pump_motor=lambda enabled: hardware_calls.append(("气泵", enabled)),
    )

    move_response = node.move_arm(types.SimpleNamespace(
        pose=[0, 0, 200, 0, 0, 0],
        speed=20,
        wait_until_stable=False,
        blend_enabled=False,
        blend_radius_mm=0.0,
    ))
    servo_response = node.rotate_tool(types.SimpleNamespace(angle_deg=180.0))
    suction_response = node.set_suction(
        types.SimpleNamespace(state=2, SUCK=0, BLOW=1, OFF=2)
    )

    assert move_response.success is False
    assert servo_response.success is False
    assert suction_response.success is True
    assert hardware_calls == [("电磁阀", False), ("气泵", False)]


def test_clear_stop_requires_confirmed_stable_motion(monkeypatch):
    module = _load_controller(monkeypatch)
    node = object.__new__(module.ControlNode)
    node.stop_latched = True
    node._query_motion_state = lambda: (False, 4.0, 0.2)

    rejected = node.clear_arm_stop(None)

    assert rejected.success is False
    assert rejected.stop_latched is True

    node._query_motion_state = lambda: (True, 0.0, 0.0)
    accepted = node.clear_arm_stop(None)

    assert accepted.success is True
    assert accepted.stop_latched is False
    assert node.stop_latched is False


def test_clear_stop_rejects_racing_old_move_handler(monkeypatch):
    module = _load_controller(monkeypatch)
    node = object.__new__(module.ControlNode)
    node.stop_latched = True
    node.motion_active = True
    node._query_motion_state = lambda: (_ for _ in ()).throw(
        AssertionError("旧运动线程存在时不应只按瞬时速度解锁")
    )

    response = node.clear_arm_stop(None)

    assert response.success is False
    assert response.stop_latched is True
    assert "仍在执行" in response.message


def test_control_status_reports_last_commanded_states(monkeypatch):
    module = _load_controller(monkeypatch)
    node = object.__new__(module.ControlNode)
    node.stop_latched = True
    node.commanded_suction_state = 0
    node.servo_target_known = True
    node.servo_target_angle_deg = 135.0
    node._query_motion_state = lambda: (True, 0.0, 0.0)

    response = node.get_control_status(None)

    assert response.success is True
    assert response.stop_latched is True
    assert response.motion_state_known is True
    assert response.motion_done is True
    assert response.commanded_suction_state == 0
    assert response.servo_target_known is True
    assert response.servo_target_angle_deg == 135.0


def test_control_status_does_not_report_done_while_move_handler_is_active(monkeypatch):
    module = _load_controller(monkeypatch)
    node = object.__new__(module.ControlNode)
    node.motion_active = True
    node._query_motion_state = lambda: (True, 0.0, 0.0)

    response = node.get_control_status(None)

    assert response.motion_state_known is True
    assert response.motion_done is False
    assert "仍在执行" in response.message
