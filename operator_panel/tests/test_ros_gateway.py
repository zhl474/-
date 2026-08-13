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
