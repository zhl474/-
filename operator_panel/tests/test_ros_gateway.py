import sys
import types

import numpy as np

from operator_panel_lib.event_bus import EventBus
from operator_panel_lib.ros_gateway import RosGateway


def test相机预览最多按2FPS编码(monkeypatch):
    bus = EventBus()
    gateway = RosGateway(bus, preview_fps=2.0, jpeg_quality=78)
    encoded_calls = []
    gateway._bridge = types.SimpleNamespace(
        imgmsg_to_cv2=lambda _message, desired_encoding: np.zeros((2, 2, 3), dtype=np.uint8)
    )

    class FakeEncoded:
        def tobytes(self):
            return b"jpeg"

    gateway._cv2 = types.SimpleNamespace(
        IMWRITE_JPEG_QUALITY=1,
        imencode=lambda *_args: encoded_calls.append(_args) or (True, FakeEncoded()),
    )
    times = iter([10.0, 10.2, 10.6])
    monkeypatch.setattr("operator_panel_lib.ros_gateway.time.monotonic", lambda: next(times))

    gateway._on_image(object())
    gateway._on_image(object())
    gateway._on_image(object())

    assert len(encoded_calls) == 2
    assert gateway.image_bytes()[0] == b"jpeg"
    assert len(bus.snapshot("image")) == 2


def test日志级别按ROS等级转换():
    bus = EventBus()
    gateway = RosGateway(bus)
    gateway._on_ros_log(types.SimpleNamespace(
        level=8, name="/control_node", msg="运动异常", file="controller.py", line=12
    ))

    event = bus.snapshot("log")[0]["data"]
    assert event["level"] == "error"
    assert event["source"] == "/control_node"
    assert event["message"] == "运动异常"


def testROS系统图整理节点话题服务并显示相机实测频率(monkeypatch):
    monkeypatch.setenv("ROS_DISTRO", "noetic")
    monkeypatch.setenv("ROS_MASTER_URI", "http://127.0.0.1:11311")

    result = RosGateway._build_ros_system(
        True,
        {"/operator_panel", "/camera_node", "/image_process_node"},
        (
            [["/camera/image_rect", ["/camera_node"]]],
            [["/camera/image_rect", ["/image_process_node", "/operator_panel"]]],
            [["/perception/prepare_task", ["/image_process_node"]]],
        ),
        [["/camera/image_rect", "sensor_msgs/Image"]],
        29.74,
    )

    assert result["distro"] == "noetic"
    assert result["node_count"] == 3
    assert result["topic_count"] == 1
    assert result["service_count"] == 1
    assert result["topics"] == [{
        "name": "/camera/image_rect",
        "type": "sensor_msgs/Image",
        "publishers": ["/camera_node"],
        "subscribers": ["/image_process_node", "/operator_panel"],
        "hz": 29.7,
    }]
    assert result["services"][0]["providers"] == ["/image_process_node"]


def test相机话题频率只统计最近五秒帧(monkeypatch):
    gateway = RosGateway(EventBus())
    gateway._camera_frame_samples.extend([1.0, 4.9, 9.0, 9.5, 10.0])
    monkeypatch.setattr("operator_panel_lib.ros_gateway.time.monotonic", lambda: 10.0)

    assert gateway._camera_hz_locked() == 2.0


def test人工选择ROS服务读取与回答(monkeypatch):
    service_module = types.ModuleType("image_process.srv")

    class GetRequest:
        pass

    class RespondRequest:
        def __init__(self, prompt_id="", choice=""):
            self.prompt_id = prompt_id
            self.choice = choice

    service_module.GetOperatorPrompt = type("GetOperatorPrompt", (), {})
    service_module.GetOperatorPromptRequest = GetRequest
    service_module.RespondOperatorPrompt = type("RespondOperatorPrompt", (), {})
    service_module.RespondOperatorPromptRequest = RespondRequest
    package_module = types.ModuleType("image_process")
    package_module.srv = service_module
    monkeypatch.setitem(sys.modules, "image_process", package_module)
    monkeypatch.setitem(sys.modules, "image_process.srv", service_module)

    calls = []
    prompt_response = types.SimpleNamespace(
        pending=True,
        prompt_id="prompt-1",
        prompt_type="dynamic_board_failure",
        message="速度不一致",
        allow_fixed_yaml=True,
        allow_continue_dynamic=True,
        remaining_seconds=59.2,
    )
    answer_response = types.SimpleNamespace(
        success=True,
        code="accepted",
        message="已接受",
    )

    class FakeRospy:
        @staticmethod
        def wait_for_service(name, timeout):
            calls.append(("wait", name, timeout))

        @staticmethod
        def ServiceProxy(name, _service, persistent=False):
            def invoke(request):
                calls.append(("call", name, request))
                return prompt_response if name.endswith("get_operator_prompt") else answer_response
            return invoke

    gateway = RosGateway(EventBus())
    gateway._rospy = FakeRospy()

    prompt = gateway.get_operator_prompt()
    gateway._operator_prompt = dict(prompt)
    answer = gateway.respond_operator_prompt("prompt-1", "fixed_yaml")

    assert prompt["pending"] is True
    assert prompt["remaining_seconds"] == 59.2
    assert answer["success"] is True
    assert gateway.operator_prompt_snapshot()["pending"] is False
    request = [item[2] for item in calls if item[0] == "call"][-1]
    assert request.prompt_id == "prompt-1"
    assert request.choice == "fixed_yaml"
