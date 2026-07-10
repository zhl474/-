import importlib
import sys
import types


class _ServiceType:
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _load_ros_clients(monkeypatch):
    rospy = types.ModuleType("rospy")
    monkeypatch.setitem(sys.modules, "rospy", rospy)

    control = types.ModuleType("control")
    control.srv = types.ModuleType("control.srv")
    for name in (
        "MoveArm", "MoveArmRequest", "RotateTool", "RotateToolRequest",
        "SetSuction", "SetSuctionRequest",
    ):
        setattr(control.srv, name, _ServiceType)
    monkeypatch.setitem(sys.modules, "control", control)
    monkeypatch.setitem(sys.modules, "control.srv", control.srv)

    image_process = types.ModuleType("image_process")
    image_process.srv = types.ModuleType("image_process.srv")
    for name in (
        "DetectBlockOffset", "DetectBlockOffsetRequest",
        "DetectBoardOffset", "DetectBoardOffsetRequest",
        "GetTaskTarget", "GetTaskTargetRequest",
        "PrepareTask", "PrepareTaskRequest",
    ):
        setattr(image_process.srv, name, _ServiceType)
    monkeypatch.setitem(sys.modules, "image_process", image_process)
    monkeypatch.setitem(sys.modules, "image_process.srv", image_process.srv)

    sys.modules.pop("competition_lib.ros_clients", None)
    return importlib.import_module("competition_lib.ros_clients")


def test_prepare_task_preserves_recoverable_failure_response(monkeypatch):
    module = _load_ros_clients(monkeypatch)
    client = object.__new__(module.RobotClients)
    failure = types.SimpleNamespace(success=False, task_count=0, message="托盘识别失败")
    client._prepare_task = lambda _request: failure

    response = client.prepare_task(advanced=False, place_order=[])

    assert response is failure
