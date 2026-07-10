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


def test_invalid_low_pose_never_calls_robot(monkeypatch):
    module = _load_controller(monkeypatch)
    calls = []
    node = object.__new__(module.ControlNode)
    node.minimum_z = 165.0
    node.arm = types.SimpleNamespace(
        set_speed=lambda _speed: calls.append("set_speed"),
        arm=types.SimpleNamespace(MoveL=lambda *_args, **_kwargs: calls.append("MoveL")),
    )
    request = types.SimpleNamespace(pose=[0, 0, 100, 0, 0, 0], speed=40)

    response = node.move_arm(request)

    assert response.success is False
    assert calls == []


def test_non_finite_pose_never_calls_robot(monkeypatch):
    module = _load_controller(monkeypatch)
    calls = []
    node = object.__new__(module.ControlNode)
    node.minimum_z = 165.0
    node.arm = types.SimpleNamespace(
        set_speed=lambda _speed: calls.append("set_speed"),
        arm=types.SimpleNamespace(MoveL=lambda *_args, **_kwargs: calls.append("MoveL")),
    )
    request = types.SimpleNamespace(pose=[0, 0, float("nan"), 0, 0, 0], speed=40)

    response = node.move_arm(request)

    assert response.success is False
    assert calls == []
