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
        "ClearArmStop", "ClearArmStopRequest",
        "GetActualPose", "GetActualPoseRequest",
        "GetControlStatus", "GetControlStatusRequest",
        "MoveArm", "MoveArmRequest", "RotateTool", "RotateToolRequest",
        "SetSuction", "SetSuctionRequest",
        "StopArm", "StopArmRequest",
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


def test_web_client_uses_finite_service_wait_timeout(monkeypatch):
    module = _load_ros_clients(monkeypatch)
    waits = []
    module.rospy.wait_for_service = lambda name, **kwargs: waits.append((name, kwargs))
    module.rospy.ServiceProxy = lambda *_args, **_kwargs: object()

    module.RobotClients(service_wait_timeout=3.0)

    assert len(waits) == len(module.SERVICE_NAMES)
    assert all(item[1] == {"timeout": 3.0} for item in waits)


def test_move_arm_forwards_stability_wait_flag(monkeypatch):
    module = _load_ros_clients(monkeypatch)
    client = object.__new__(module.RobotClients)
    requests = []
    client._move_arm = lambda request: requests.append(request) or types.SimpleNamespace(success=True)

    client.move_arm([1, 2, 3, 4, 5, 6], 40)
    client.move_arm([1, 2, 3, 4, 5, 6], 40, wait_until_stable=True)

    assert [request.wait_until_stable for request in requests] == [False, True]
    assert [request.blend_enabled for request in requests] == [False, False]
    assert [request.blend_radius_mm for request in requests] == [0.0, 0.0]


def test_move_arm_forwards_blend_radius(monkeypatch):
    module = _load_ros_clients(monkeypatch)
    client = object.__new__(module.RobotClients)
    requests = []
    client._move_arm = lambda request: requests.append(request) or types.SimpleNamespace(success=True)

    client.move_arm([1, 2, 3, 4, 5, 6], 40, blend_radius_mm=5.0)

    assert requests[0].blend_enabled is True
    assert requests[0].blend_radius_mm == 5.0


def test_get_actual_pose_calls_read_only_control_service(monkeypatch):
    module = _load_ros_clients(monkeypatch)
    client = object.__new__(module.RobotClients)
    response = types.SimpleNamespace(success=True, tcp_pose=[1] * 6, camera_pose=[2] * 6)
    requests = []
    client._get_actual_pose = lambda request: requests.append(request) or response

    assert client.get_actual_pose() is response
    assert len(requests) == 1
