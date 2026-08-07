import importlib.util
import os
import sys
import threading
import time
import types

import numpy as np
import pytest


class _Response:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Message:
    def __init__(self):
        self.header = types.SimpleNamespace(stamp=None, frame_id="")


class _Publisher:
    def __init__(self, *_args, **_kwargs):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _Bridge:
    def cv2_to_imgmsg(self, _image, encoding):
        assert encoding == "bgr8"
        return _Message()


def _load_camera(monkeypatch):
    rospy = types.ModuleType("rospy")
    rospy.logerr = lambda *_args, **_kwargs: None
    rospy.logwarn = lambda *_args, **_kwargs: None
    rospy.loginfo = lambda *_args, **_kwargs: None
    rospy.get_param = lambda _name, default=None: default
    rospy.Publisher = _Publisher
    rospy.Service = lambda *_args, **_kwargs: object()
    rospy.on_shutdown = lambda *_args, **_kwargs: None
    rospy.Time = types.SimpleNamespace(now=lambda: 123.0)
    rospy.Rate = lambda _hz: types.SimpleNamespace(sleep=lambda: None)
    rospy.is_shutdown = lambda: True
    rospy.spin = lambda: None
    rospy.init_node = lambda *_args, **_kwargs: None
    rospy.ROSInterruptException = RuntimeError
    monkeypatch.setitem(sys.modules, "rospy", rospy)

    cv_bridge = types.ModuleType("cv_bridge")
    cv_bridge.CvBridge = _Bridge
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
    camera.srv.GetSurfaceHeight = object
    camera.srv.GetSurfaceHeightResponse = _Response
    camera.srv.GetStableWorldPoints = object
    camera.srv.GetStableWorldPointsResponse = _Response
    monkeypatch.setitem(sys.modules, "camera", camera)
    monkeypatch.setitem(sys.modules, "camera.srv", camera.srv)

    script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "camera_node.py")
    )
    spec = importlib.util.spec_from_file_location("camera_surface_height_validation", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_node(module, depth_image=None, depth_stamp=100.0):
    """构造只包含表面高度查询所需状态的轻量节点。"""
    node = object.__new__(module.CameraNode)
    node.depth_lock = threading.Lock()
    node.depth_condition = threading.Condition(node.depth_lock)
    node.latest_depth_image = depth_image
    node.latest_depth_monotonic = depth_stamp
    node.depth_max_age_sec = 0.5
    node.depth_collectors = []
    node.arm = None
    node.arm_init_lock = threading.Lock()
    node.hand_eye_matrix_path = "/tmp/手眼矩阵.npy"
    node.world_bias_mm = np.zeros(3, dtype=float)
    node.cap = types.SimpleNamespace(
        depth_pixel2cam_point3d=lambda _x, _y, depth_value: [0.0, 0.0, depth_value]
    )
    return node


def _start_stable_request(node, request):
    """在线程中发起阻塞请求，并等到采集器完成登记。"""
    responses = []
    thread = threading.Thread(
        target=lambda: responses.append(node.get_stable_world_points(request))
    )
    thread.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with node.depth_condition:
            if node.depth_collectors:
                return thread, responses
        threading.Event().wait(0.001)
    thread.join(timeout=1.0)
    raise AssertionError("批量深度请求未及时登记采集器")


def test_service_definition_replaces_old_pixel_to_world_service():
    package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    service_path = os.path.join(package_dir, "srv", "GetSurfaceHeight.srv")

    assert not os.path.exists(os.path.join(package_dir, "srv", "PixelToWorld.srv"))
    text = open(service_path, "r", encoding="utf-8").read()
    for field in ("int32 x", "int32 y", "float64 surface_z_mm", "bool success"):
        assert field in text
    stable_text = open(
        os.path.join(package_dir, "srv", "GetStableWorldPoints.srv"),
        "r",
        encoding="utf-8",
    ).read()
    for field in (
        "int32[] x",
        "int32[] y",
        "float64 capture_timeout_sec",
        "float64[] world_xyz",
        "float64[] depth_mad_mm",
    ):
        assert field in stable_text


def test_node_startup_does_not_initialize_arm_or_load_hand_eye(monkeypatch):
    module = _load_camera(monkeypatch)
    arm_calls = []
    monkeypatch.setattr(module, "AkaiFr", lambda: arm_calls.append(True))
    monkeypatch.setattr(
        module.np,
        "load",
        lambda _path: (_ for _ in ()).throw(AssertionError("启动时不应加载手眼矩阵")),
    )

    node = module.CameraNode()

    assert node.arm is None
    assert arm_calls == []
    assert node.depth_max_age_sec == 0.5
    assert node.depth_collectors == []
    assert not hasattr(node, "depth_batch_max_span_sec")
    assert not hasattr(node, "depth_buffer_size")


def test_stable_world_points_use_temporal_median_mad_and_one_camera_pose(monkeypatch):
    module = _load_camera(monkeypatch)
    node = _make_node(module)
    pose_calls = []
    node.arm = types.SimpleNamespace(
        get_camera_pose=lambda: pose_calls.append(True) or (True, [0, 0, 0, 0, 0, 0])
    )
    node.cap = types.SimpleNamespace(
        depth_pixel2cam_point3d=lambda x, y, depth_value: [x, y, depth_value]
    )
    request = types.SimpleNamespace(
        x=[1, 2],
        y=[1, 2],
        frame_count=15,
        min_valid_frames=10,
        capture_timeout_sec=1.0,
    )
    # 请求前发布的旧帧不会进入这次批量结果。
    node._cache_depth_frame(np.full((3, 3), 999.0))
    thread, responses = _start_stable_request(node, request)
    with node.depth_condition:
        registered_monotonic = node.depth_collectors[0]["registered_monotonic"]
    # 即使请求前取得的帧稍后才进入缓存，也不能被本次采集器接收。
    node._cache_depth_frame(
        np.full((3, 3), 888.0),
        captured_monotonic=registered_monotonic - 0.001,
    )
    for depth in range(100, 115):
        node._cache_depth_frame(np.full((3, 3), depth))
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    response = responses[0]
    assert response.success is True
    assert response.point_valid == [True, True]
    assert response.valid_frame_counts == [15, 15]
    assert response.depth_median_mm == [107.0, 107.0]
    assert response.depth_mad_mm == [4.0, 4.0]
    assert response.world_xyz == [1.0, 1.0, 107.0, 2.0, 2.0, 107.0]
    assert len(pose_calls) == 1


def test_stable_world_points_times_out_with_explicit_new_frame_count(monkeypatch):
    module = _load_camera(monkeypatch)
    node = _make_node(module)
    request = types.SimpleNamespace(
        x=[1],
        y=[1],
        frame_count=15,
        min_valid_frames=10,
        capture_timeout_sec=0.05,
    )
    thread, responses = _start_stable_request(node, request)
    for _index in range(4):
        node._cache_depth_frame(np.ones((3, 3)))
    thread.join(timeout=0.5)

    assert not thread.is_alive()
    response = responses[0]
    assert response.success is False
    assert "新深度帧不足：实际 4/15" in response.message
    assert "过期" not in response.message
    assert "时间跨度" not in response.message


def test_stable_world_points_marks_point_invalid_below_minimum_valid_frames(monkeypatch):
    module = _load_camera(monkeypatch)
    node = _make_node(module)
    node.arm = types.SimpleNamespace(
        get_camera_pose=lambda: (True, [0, 0, 0, 0, 0, 0])
    )

    request = types.SimpleNamespace(
        x=[1],
        y=[1],
        frame_count=15,
        min_valid_frames=10,
        capture_timeout_sec=1.0,
    )
    thread, responses = _start_stable_request(node, request)
    for index in range(15):
        depth = 500.0 if index < 9 else 0.0
        node._cache_depth_frame(np.full((3, 3), depth))
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    response = responses[0]
    assert response.success is True
    assert response.point_valid == [False]
    assert response.valid_frame_counts == [9]
    assert response.depth_median_mm == [500.0]
    assert response.world_xyz == [0.0, 0.0, 0.0]


def test_missing_depth_returns_explicit_failure(monkeypatch):
    module = _load_camera(monkeypatch)
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    node = _make_node(module, depth_image=None)

    response = node.get_surface_height(types.SimpleNamespace(x=0, y=0))

    assert response.success is False
    assert response.surface_z_mm == 0.0
    assert "没有可用深度图" in response.message


def test_stale_depth_returns_explicit_failure(monkeypatch):
    module = _load_camera(monkeypatch)
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.6)
    node = _make_node(module, depth_image=np.ones((3, 3)), depth_stamp=100.0)

    response = node.get_surface_height(types.SimpleNamespace(x=1, y=1))

    assert response.success is False
    assert "过期" in response.message
    assert "0.600" in response.message


def test_out_of_bounds_pixel_returns_explicit_failure(monkeypatch):
    module = _load_camera(monkeypatch)
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    node = _make_node(module, depth_image=np.ones((10, 10), dtype=np.float32))

    response = node.get_surface_height(types.SimpleNamespace(x=20, y=0))

    assert response.success is False
    assert "越界" in response.message


def test_neighborhood_without_valid_depth_returns_explicit_failure(monkeypatch):
    module = _load_camera(monkeypatch)
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    depth_image = np.array(
        [[np.nan, 0.0, -1.0], [np.inf, 0.0, -2.0], [np.nan, 0.0, -3.0]],
        dtype=float,
    )
    node = _make_node(module, depth_image=depth_image)

    response = node.get_surface_height(types.SimpleNamespace(x=1, y=1))

    assert response.success is False
    assert "没有有效深度值" in response.message


def test_clipped_3x3_neighborhood_uses_valid_median_and_returns_absolute_z(monkeypatch):
    module = _load_camera(monkeypatch)
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    depth_image = np.array(
        [
            [0.0, 100.0, 999.0],
            [300.0, 200.0, 999.0],
            [999.0, 999.0, 999.0],
        ],
        dtype=float,
    )
    node = _make_node(module, depth_image=depth_image)
    node.arm = types.SimpleNamespace(get_camera_pose=lambda: (True, [1, 2, 3, 0, 0, 0]))
    received = {}
    node.cap = types.SimpleNamespace(
        depth_pixel2cam_point3d=lambda x, y, depth_value: received.update(
            x=x, y=y, depth=depth_value
        )
        or [10.0, 20.0, depth_value]
    )
    node.world_bias_mm = np.array([0.0, 0.0, 2.5])
    module.tf3d.VectorTransform = lambda _transform, point: np.asarray(point) + [1.0, 2.0, 30.0]

    # (0, 0) 使用裁剪后的 2x2 邻域，有效值 [100, 300, 200] 的中位数为 200。
    response = node.get_surface_height(types.SimpleNamespace(x=0, y=0))

    assert response.success is True
    assert received == {"x": 0, "y": 0, "depth": 200.0}
    assert response.surface_z_mm == pytest.approx(232.5)
    assert "成功" in response.message


def test_camera_pose_failure_is_explicit(monkeypatch):
    module = _load_camera(monkeypatch)
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    node = _make_node(module, depth_image=np.ones((3, 3), dtype=float))
    node.arm = types.SimpleNamespace(get_camera_pose=lambda: (False, None))

    response = node.get_surface_height(types.SimpleNamespace(x=1, y=1))

    assert response.success is False
    assert "获取相机位姿失败" in response.message


def test_arm_and_hand_eye_are_initialized_once_across_threads(monkeypatch):
    module = _load_camera(monkeypatch)
    node = _make_node(module)
    load_calls = []
    arm_instances = []

    class _Arm:
        def __init__(self):
            self.matrices = []

        def set_tmat_wrist2camera(self, matrix):
            self.matrices.append(np.asarray(matrix).copy())

    def load_matrix(path):
        load_calls.append(path)
        return np.eye(4)

    def make_arm():
        arm = _Arm()
        arm_instances.append(arm)
        return arm

    monkeypatch.setattr(module.np, "load", load_matrix)
    monkeypatch.setattr(module, "AkaiFr", make_arm)
    results = []
    threads = [threading.Thread(target=lambda: results.append(node._ensure_arm_initialized())) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)

    assert all(not thread.is_alive() for thread in threads)
    assert len(load_calls) == 1
    assert len(arm_instances) == 1
    assert len(arm_instances[0].matrices) == 1
    assert all(result is arm_instances[0] for result in results)


def test_rgb_is_published_when_same_capture_has_no_depth(monkeypatch):
    module = _load_camera(monkeypatch)
    node = object.__new__(module.CameraNode)
    color_image = np.full((2, 2, 3), 7, dtype=np.uint8)
    node.cap = types.SimpleNamespace(read=lambda: (color_image, None))
    node.depth_lock = threading.Lock()
    node.depth_condition = threading.Condition(node.depth_lock)
    node.latest_depth_image = None
    node.latest_depth_monotonic = None
    node.depth_collectors = []
    node.bridge = _Bridge()
    node.image_pub = _Publisher()
    shutdown_values = iter([False, True])
    monkeypatch.setattr(module.rospy, "is_shutdown", lambda: next(shutdown_values))

    node.publish_images()

    assert len(node.image_pub.messages) == 1
    assert node.image_pub.messages[0].header.frame_id == "camera_frame"
    assert node.latest_depth_image is None


def test_new_rgb_without_depth_invalidates_previous_depth_cache(monkeypatch):
    module = _load_camera(monkeypatch)
    node = object.__new__(module.CameraNode)
    color_image = np.full((2, 2, 3), 7, dtype=np.uint8)
    node.cap = types.SimpleNamespace(read=lambda: (color_image, None))
    node.depth_lock = threading.Lock()
    node.depth_condition = threading.Condition(node.depth_lock)
    node.latest_depth_image = np.full((2, 2), 100.0, dtype=float)
    node.latest_depth_monotonic = 100.0
    node.depth_collectors = []
    node.depth_max_age_sec = 0.5
    node.bridge = _Bridge()
    node.image_pub = _Publisher()
    shutdown_values = iter([False, True])
    monkeypatch.setattr(module.rospy, "is_shutdown", lambda: next(shutdown_values))

    node.publish_images()
    response = node.get_surface_height(types.SimpleNamespace(x=0, y=0))

    assert len(node.image_pub.messages) == 1
    assert node.latest_depth_image is None
    assert node.latest_depth_monotonic is None
    assert response.success is False
    assert "没有可用深度图" in response.message


def test_depth_cache_conversion_failure_invalidates_previous_cache(monkeypatch):
    module = _load_camera(monkeypatch)

    class _InvalidDepth:
        def __array__(self, _dtype=None, copy=None):
            del copy
            raise ValueError("深度转换失败")

    node = object.__new__(module.CameraNode)
    color_image = np.full((2, 2, 3), 7, dtype=np.uint8)
    node.cap = types.SimpleNamespace(read=lambda: (color_image, _InvalidDepth()))
    node.depth_lock = threading.Lock()
    node.depth_condition = threading.Condition(node.depth_lock)
    node.latest_depth_image = np.full((2, 2), 100.0, dtype=float)
    node.latest_depth_monotonic = 100.0
    node.depth_collectors = []
    node.bridge = _Bridge()
    node.image_pub = _Publisher()
    shutdown_values = iter([False, True])
    monkeypatch.setattr(module.rospy, "is_shutdown", lambda: next(shutdown_values))

    node.publish_images()

    assert len(node.image_pub.messages) == 1
    assert node.latest_depth_image is None
    assert node.latest_depth_monotonic is None
