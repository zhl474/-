"""运行时曝光/增益调整的白名单、clamp 和服务行为测试。

不依赖真实相机：设备句柄用 FakeDevice 顶替，验证
自动曝光联动关闭、范围 clamp、值未变化跳过写入等逻辑。
"""

import importlib.util
import os
import sys
import threading
import types

import numpy as np
import pytest


class _Response:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Range:
    def __init__(self, minimum, maximum, step=1, default=0):
        self.min = minimum
        self.max = maximum
        self.step = step
        self.default_value = default


class FakeDevice:
    """按 camera_node.EXPOSURE_PROPERTY_IDS 里的枚举键模拟 SDK 设备属性。"""

    def __init__(self, module, initial=None):
        self.module = module
        self.bool_values = {
            module.EXPOSURE_PROPERTY_IDS["rgb"]["auto_exposure"][0]: True,
            module.EXPOSURE_PROPERTY_IDS["depth"]["auto_exposure"][0]: True,
        }
        self.int_values = {}
        self.int_ranges = {}
        for sensor, properties in module.EXPOSURE_PROPERTY_IDS.items():
            for key, (prop_id, is_bool) in properties.items():
                if is_bool:
                    continue
                self.int_values[prop_id] = 1
                self.int_ranges[prop_id] = _Range(1, 100)
        for (sensor, key), value in (initial or {}).items():
            prop_id, is_bool = module.EXPOSURE_PROPERTY_IDS[sensor][key]
            if is_bool:
                self.bool_values[prop_id] = value
            else:
                self.int_values[prop_id] = value
        self.bool_writes = []
        self.int_writes = []

    def get_bool_property(self, prop_id):
        return self.bool_values[prop_id]

    def set_bool_property(self, prop_id, value):
        self.bool_values[prop_id] = bool(value)
        self.bool_writes.append((prop_id, bool(value)))

    def get_int_property(self, prop_id):
        return self.int_values[prop_id]

    def set_int_property(self, prop_id, value):
        self.int_values[prop_id] = int(value)
        self.int_writes.append((prop_id, int(value)))

    def get_int_property_range(self, prop_id):
        return self.int_ranges[prop_id]


def _load_camera(monkeypatch):
    rospy = types.ModuleType("rospy")
    rospy.logerr = lambda *_args, **_kwargs: None
    rospy.logwarn = lambda *_args, **_kwargs: None
    rospy.loginfo = lambda *_args, **_kwargs: None
    rospy.get_param = lambda _name, default=None: default
    rospy.Publisher = lambda *_args, **_kwargs: object()
    rospy.Service = lambda *_args, **_kwargs: object()
    rospy.on_shutdown = lambda *_args, **_kwargs: None
    rospy.Time = types.SimpleNamespace(now=lambda: 123.0)
    rospy.Rate = lambda _hz: types.SimpleNamespace(sleep=lambda: None)
    rospy.is_shutdown = lambda: True
    rospy.spin = lambda: None
    rospy.init_node = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "rospy", rospy)

    cv_bridge = types.ModuleType("cv_bridge")
    cv_bridge.CvBridge = object
    monkeypatch.setitem(sys.modules, "cv_bridge", cv_bridge)
    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs.msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs.msg.Image = object
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_msgs)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msgs.msg)

    akai = types.ModuleType("akai")
    akai.DEG = akai.MM = object()
    akai.tf3d = types.SimpleNamespace(
        XYZRPY2TransformMatrix=lambda pose, **_kwargs: np.asarray(pose, dtype=float),
        VectorTransform=lambda _transform, point: np.asarray(point, dtype=float),
    )
    monkeypatch.setitem(sys.modules, "akai", akai)
    akai_fr = types.ModuleType("akai_fr")
    akai_fr.AkaiFr = object
    monkeypatch.setitem(sys.modules, "akai_fr", akai_fr)
    gemini = types.ModuleType("akai_gemini335")
    gemini.AkaiGemini335 = lambda **_kwargs: types.SimpleNamespace(release=lambda: None)
    monkeypatch.setitem(sys.modules, "akai_gemini335", gemini)

    camera = types.ModuleType("camera")
    camera.srv = types.ModuleType("camera.srv")
    for name in (
        "GetSurfaceHeight", "GetStableWorldPoints",
        "GetExposureState", "SetExposureParam",
    ):
        setattr(camera.srv, name, object)
    for name in (
        "GetSurfaceHeightResponse", "GetStableWorldPointsResponse",
        "GetExposureStateResponse", "SetExposureParamResponse",
    ):
        setattr(camera.srv, name, _Response)
    monkeypatch.setitem(sys.modules, "camera", camera)
    monkeypatch.setitem(sys.modules, "camera.srv", camera.srv)

    script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "camera_node.py")
    )
    spec = importlib.util.spec_from_file_location("camera_exposure_control", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_node(module, device):
    node = object.__new__(module.CameraNode)
    node.property_lock = threading.Lock()
    node.exposure_device = device
    return node


def _request(sensor, key, value):
    return types.SimpleNamespace(sensor=sensor, key=key, value=value)


@pytest.fixture
def camera_module(monkeypatch):
    return _load_camera(monkeypatch)


def test请求白名单校验(camera_module):
    ok, error = camera_module.validate_exposure_request("rgb", "exposure")
    assert ok and error == ""
    ok, error = camera_module.validate_exposure_request("depth", "auto_exposure")
    assert ok and error == ""
    for sensor, key in (("rgb", "brightness"), ("color", "exposure"), ("", ""), (None, None)):
        ok, error = camera_module.validate_exposure_request(sensor, key)
        assert not ok and error


def test曝光值clamp到范围(camera_module):
    assert camera_module.clamp_exposure_value(50, 1, 100) == 50
    assert camera_module.clamp_exposure_value(-5, 1, 100) == 1
    assert camera_module.clamp_exposure_value(9999, 1, 100) == 100
    assert camera_module.clamp_exposure_value("42", 1, 100) == 42


def test白名单外的请求直接拒绝不触碰设备(camera_module):
    device = FakeDevice(camera_module)
    node = _make_node(camera_module, device)
    response = node.set_exposure_param(_request("rgb", "brightness", 35))
    assert response.ok is False
    assert device.int_writes == [] and device.bool_writes == []


def test手动写曝光自动关闭自动曝光(camera_module):
    device = FakeDevice(camera_module)
    node = _make_node(camera_module, device)
    response = node.set_exposure_param(_request("rgb", "exposure", 70))
    assert response.ok is True
    assert response.applied == 70
    ae_id = camera_module.EXPOSURE_PROPERTY_IDS["rgb"]["auto_exposure"][0]
    assert (ae_id, False) in device.bool_writes
    assert device.get_int_property(
        camera_module.EXPOSURE_PROPERTY_IDS["rgb"]["exposure"][0]
    ) == 70
    assert "自动关闭自动曝光" in response.message


def test自动曝光已关时不重复关(camera_module):
    device = FakeDevice(camera_module, initial={("rgb", "auto_exposure"): False})
    node = _make_node(camera_module, device)
    response = node.set_exposure_param(_request("rgb", "gain", 16))
    assert response.ok is True
    assert device.bool_writes == []
    assert "自动关闭自动曝光" not in response.message


def test超出范围的值被clamp后写入(camera_module):
    device = FakeDevice(camera_module, initial={("rgb", "auto_exposure"): False})
    node = _make_node(camera_module, device)
    response = node.set_exposure_param(_request("rgb", "exposure", 100000))
    assert response.ok is True
    assert response.applied == 100


def test值未变化时跳过写入(camera_module):
    device = FakeDevice(camera_module, initial={("rgb", "auto_exposure"): False, ("rgb", "exposure"): 70})
    node = _make_node(camera_module, device)
    response = node.set_exposure_param(_request("rgb", "exposure", 70))
    assert response.ok is True
    assert response.applied == 70
    assert device.int_writes == []
    assert "跳过写入" in response.message


def test设置自动曝光开关(camera_module):
    device = FakeDevice(camera_module)
    node = _make_node(camera_module, device)
    response = node.set_exposure_param(_request("depth", "auto_exposure", 0))
    assert response.ok is True
    assert response.applied == 0
    ae_id = camera_module.EXPOSURE_PROPERTY_IDS["depth"]["auto_exposure"][0]
    assert device.get_bool_property(ae_id) is False


def test曝光状态查询返回全部字段(camera_module):
    device = FakeDevice(camera_module)
    node = _make_node(camera_module, device)
    response = node.get_exposure_state(None)
    assert response.ok is True
    assert response.rgb_auto_exposure is True
    assert response.rgb_exposure_min == 1 and response.rgb_exposure_max == 100
    assert response.depth_gain_min == 1 and response.depth_gain_max == 100
