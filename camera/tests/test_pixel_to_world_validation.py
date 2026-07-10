import importlib.util
import os
import sys
import threading
import types

import numpy as np


class _Response:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _load_camera(monkeypatch):
    rospy = types.ModuleType("rospy")
    rospy.logerr = lambda *_args, **_kwargs: None
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
    akai.tf3d = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "akai", akai)
    akai_fr = types.ModuleType("akai_fr")
    akai_fr.AkaiFr = object
    monkeypatch.setitem(sys.modules, "akai_fr", akai_fr)
    gemini = types.ModuleType("akai_gemini335")
    gemini.AkaiGemini335 = object
    monkeypatch.setitem(sys.modules, "akai_gemini335", gemini)

    camera = types.ModuleType("camera")
    camera.srv = types.ModuleType("camera.srv")
    camera.srv.PixelToWorld = object
    camera.srv.PixelToWorldResponse = _Response
    monkeypatch.setitem(sys.modules, "camera", camera)
    monkeypatch.setitem(sys.modules, "camera.srv", camera.srv)

    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "camera_node.py"))
    spec = importlib.util.spec_from_file_location("camera_validation", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_depth_returns_explicit_failure(monkeypatch):
    module = _load_camera(monkeypatch)
    node = object.__new__(module.CameraNode)
    node.lock = threading.Lock()
    node.latest_depth_image = None

    response = node.pixel_to_world(types.SimpleNamespace(x=0, y=0))

    assert response.success is False
    assert "深度图" in response.message


def test_out_of_bounds_pixel_returns_explicit_failure(monkeypatch):
    module = _load_camera(monkeypatch)
    node = object.__new__(module.CameraNode)
    node.lock = threading.Lock()
    node.latest_depth_image = np.ones((10, 10), dtype=np.float32)

    response = node.pixel_to_world(types.SimpleNamespace(x=20, y=0))

    assert response.success is False
    assert "越界" in response.message
