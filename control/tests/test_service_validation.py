import importlib.util
import os
import sys
import types


class _Response:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _load_controller(monkeypatch):
    rospy = types.ModuleType("rospy")
    rospy.logerr = lambda *_args, **_kwargs: None
    rospy.loginfo = lambda *_args, **_kwargs: None
    rospy.is_shutdown = lambda: False
    monkeypatch.setitem(sys.modules, "rospy", rospy)

    akai_fr = types.ModuleType("akai_fr")
    akai_fr.AkaiFr = object
    akai_fr.AkaiElectricSucker = object
    monkeypatch.setitem(sys.modules, "akai_fr", akai_fr)

    control = types.ModuleType("control")
    control.srv = types.ModuleType("control.srv")
    for name in ("MoveArm", "RotateTool", "SetSuction"):
        setattr(control.srv, name, object)
    for name in ("MoveArmResponse", "RotateToolResponse", "SetSuctionResponse"):
        setattr(control.srv, name, _Response)
    monkeypatch.setitem(sys.modules, "control", control)
    monkeypatch.setitem(sys.modules, "control.srv", control.srv)

    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "controller.py"))
    spec = importlib.util.spec_from_file_location("controller_validation", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_low_pose_is_clamped_and_still_calls_robot(monkeypatch):
    module = _load_controller(monkeypatch)
    calls = []
    node = object.__new__(module.ControlNode)
    node.minimum_z = 165.0
    node.arm = types.SimpleNamespace(
        set_speed=lambda _speed: calls.append("set_speed"),
        arm=types.SimpleNamespace(
            MoveL=lambda pose, **_kwargs: (calls.append(("MoveL", pose)), 0)[1]
        ),
    )
    node._wait_until_arm_stable = lambda pose: calls.append(("wait_stable", pose))
    request = types.SimpleNamespace(
        pose=[0, 0, 100, 0, 0, 0], speed=40, wait_until_stable=True
    )

    response = node.move_arm(request)

    assert response.success is True
    assert calls == [
        "set_speed",
        ("MoveL", [0.0, 0.0, 165.0, 0.0, 0.0, 0.0]),
        ("wait_stable", [0.0, 0.0, 165.0, 0.0, 0.0, 0.0]),
    ]
    assert "已自动调整为 165.00 mm" in response.message


def test_non_finite_pose_never_calls_robot(monkeypatch):
    module = _load_controller(monkeypatch)
    calls = []
    node = object.__new__(module.ControlNode)
    node.minimum_z = 165.0
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


def test_move_arm_rejects_nonzero_movel_error(monkeypatch):
    module = _load_controller(monkeypatch)
    node = object.__new__(module.ControlNode)
    node.minimum_z = 165.0
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
    node = object.__new__(module.ControlNode)
    node.minimum_z = 165.0
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
