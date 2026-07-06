import os
import sys
import types
import importlib.util

import cv2
import numpy as np
import pytest


PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

if "ultralytics" not in sys.modules:
    ultralytics_stub = types.ModuleType("ultralytics")
    ultralytics_stub.YOLO = object
    sys.modules["ultralytics"] = ultralytics_stub

from image_process_lib.block_category import (
    BLOCK_CATEGORY_NAMES,
    category_from_code,
    category_to_code,
)
from image_process_lib.single_block_detector import detect_block_with_high_prior_roi
from image_process_lib.template_config import load_template_geometry
from image_process_lib.template_match.kernels_create import (
    build_angle_values,
    create_base_shape,
    create_rotation_kernels,
    get_template_rect_size,
)
from image_process_lib.template_match.template_match import get_rect


def _legacy_l_yellow():
    base = np.zeros((37 + 5 + 37, 37 + 5 + 37 + 5 + 37), dtype=np.float32)
    base[0:37 + 5, 37 + 5 + 37 + 5:] = 1
    base[37 + 5:, :] = 1
    return base


def _legacy_l_blue():
    base = np.zeros((37 + 5 + 37, 37 + 5 + 37 + 5 + 37), dtype=np.float32)
    base[0:37 + 5, :37] = 1
    base[37 + 5:, :] = 1
    return base


def _legacy_t():
    base = np.zeros((37 + 5 + 37, 37 + 5 + 37 + 5 + 37), dtype=np.float32)
    base[0:37 + 5, 37 + 5:37 + 5 + 37] = 1
    base[37 + 5:, :] = 1
    return base


def _legacy_z_blue():
    base = np.zeros((37 + 5 + 37, 37 + 5 + 37 + 5 + 37), dtype=np.float32)
    base[0:37, :37 + 5 + 37] = 1
    base[37:37 + 5, 37 + 5:37 + 5 + 37] = 1
    base[37 + 5:, 37 + 5:] = 1
    return base


def _legacy_z_green():
    base = np.zeros((37 + 5 + 37, 37 + 5 + 37 + 5 + 37), dtype=np.float32)
    base[0:37, 37 + 5:] = 1
    base[37:37 + 5, 37 + 5:37 + 5 + 37] = 1
    base[37 + 5:, :37 + 5 + 37] = 1
    return base


@pytest.mark.parametrize(
    ("category", "legacy_shape"),
    [
        ("L_yellow", _legacy_l_yellow()),
        ("L_blue", _legacy_l_blue()),
        ("T", _legacy_t()),
        ("z_blue", _legacy_z_blue()),
        ("z_green", _legacy_z_green()),
    ],
)
def test_create_base_shape_matches_legacy_hardcoded_templates(category, legacy_shape):
    np.testing.assert_array_equal(create_base_shape(category, 37, 5), legacy_shape)


@pytest.mark.parametrize(
    ("category", "expected_size"),
    [
        ("line", (4 * 37 + 3 * 5, 37)),
        ("square", (2 * 37 + 5, 2 * 37 + 5)),
        ("L_yellow", (3 * 37 + 2 * 5, 2 * 37 + 5)),
        ("L_blue", (3 * 37 + 2 * 5, 2 * 37 + 5)),
        ("T", (3 * 37 + 2 * 5, 2 * 37 + 5)),
        ("z_blue", (3 * 37 + 2 * 5, 2 * 37 + 5)),
        ("z_green", (3 * 37 + 2 * 5, 2 * 37 + 5)),
    ],
)
def test_template_rect_size_uses_block_and_connector_formula(category, expected_size):
    width, height = get_template_rect_size(category, 37, 5)
    assert (width, height) == expected_size
    assert create_base_shape(category, 37, 5).shape == (height, width)


def test_angle_values_cover_full_period_by_category():
    assert len(build_angle_values("T", angle_step=1)) == 360
    assert len(build_angle_values("z_blue", angle_step=1)) == 180
    assert len(build_angle_values("square", angle_step=1)) == 90


def test_angle_values_use_center_window_and_wrap_zero():
    angles = build_angle_values("T", angle_step=1, angle_center=2, angle_window=5)
    assert angles == [357.0, 358.0, 359.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_angle_values_center_window_reduces_template_count_by_category():
    assert len(build_angle_values("T", angle_step=1, angle_center=37, angle_window=10)) == 21

    z_angles = build_angle_values("z_blue", angle_step=1, angle_center=185, angle_window=10)
    assert len(z_angles) == 21
    assert z_angles[0] == 175.0
    assert z_angles[-1] == 15.0

    line_angles = build_angle_values("line", angle_step=1, angle_center=-5, angle_window=10)
    assert len(line_angles) == 21
    assert line_angles[0] == 165.0
    assert line_angles[-1] == 5.0

    square_angles = build_angle_values("square", angle_step=1, angle_center=95, angle_window=10)
    assert len(square_angles) == 21
    assert square_angles[0] == 85.0
    assert square_angles[-1] == 15.0


def test_rotation_kernels_are_binary_without_edge_weighting():
    kernels, _, angles = create_rotation_kernels(37, 5, "T", angle_values=[0, 45], device="cpu")
    assert angles == [0.0, 45.0]
    assert set(np.unique(kernels.detach().cpu().numpy()).tolist()).issubset({0.0, 1.0})


def test_block_category_code_roundtrip_is_stable():
    assert BLOCK_CATEGORY_NAMES == (
        "L_blue",
        "L_yellow",
        "z_blue",
        "z_green",
        "square",
        "T",
        "line",
    )
    for index, category in enumerate(BLOCK_CATEGORY_NAMES):
        assert category_to_code(category) == index
        assert category_from_code(index) == category
        assert category_from_code(float(index)) == category

    assert category_to_code("LL") == 1
    assert category_to_code("O") == 4
    assert category_to_code("unknown") == -1
    assert category_from_code(-1) == ""
    assert category_from_code("bad") == ""


def test_cropped_search_matches_full_search_center():
    category = "T"
    block_px = 8
    connector_px = 2
    base_shape = create_base_shape(category, block_px, connector_px)
    image = np.zeros((140, 180), dtype=np.uint8)
    top_left_x = 80
    top_left_y = 50
    h, w = base_shape.shape
    image[top_left_y:top_left_y + h, top_left_x:top_left_x + w] = (base_shape * 255).astype(np.uint8)

    debug_full = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    debug_cropped = debug_full.copy()
    full_rect = get_rect(
        image,
        block_px,
        connector_px,
        category,
        debug_full,
        0,
        0,
        angle_values=[0],
    )
    cropped_rect = get_rect(
        image,
        block_px,
        connector_px,
        category,
        debug_cropped,
        0,
        0,
        angle_values=[0],
        search_center=full_rect[0],
        search_radius=15,
    )

    np.testing.assert_allclose(cropped_rect[0], full_rect[0], atol=1.0)
    assert cropped_rect[2] == full_rect[2]


def test_low_prior_roi_template_match_removes_white_background():
    category = "T"
    block_px = 12
    connector_px = 3
    template_angle = 18.0
    high_theta = -template_angle
    target_center = (165, 116)
    image = np.full((240, 320, 3), 255, dtype=np.uint8)

    kernels, _, _ = create_rotation_kernels(
        block_px,
        connector_px,
        category,
        device="cpu",
        angle_values=[template_angle],
    )
    shape_mask = (kernels[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    mask_h, mask_w = shape_mask.shape
    x1 = target_center[0] - mask_w // 2
    y1 = target_center[1] - mask_h // 2
    roi = image[y1:y1 + mask_h, x1:x1 + mask_w]
    roi[shape_mask > 0] = (35, 60, 210)

    result = detect_block_with_high_prior_roi(
        image,
        template_geometry={"block_px": block_px, "connector_px": connector_px},
        category=category,
        high_theta_deg=high_theta,
        angle_window=3,
        angle_step=1,
        roi_expand_px=30,
        white_s_max=45,
        white_v_min=180,
        min_foreground_area=50,
    )

    assert result["found"] is True
    assert result["category"] == category
    assert abs(result["px"] - target_center[0]) <= 2.0
    assert abs(result["py"] - target_center[1]) <= 2.0
    assert abs(result["theta"] - high_theta) <= 1.0


def test_load_template_geometry_reads_active_and_named_profiles(tmp_path):
    config_path = tmp_path / "template_config.yaml"
    config_path.write_text(
        """
template_sizes:
  active_profile: high
  profiles:
    high:
      block_px: 37
      connector_px: 5
    low:
      block_px: 31.4
      connector_px: 4.6
""",
        encoding="utf-8",
    )

    assert load_template_geometry(config_path=str(config_path)) == {
        "block_px": 37,
        "connector_px": 5,
    }
    assert load_template_geometry(profile="low", config_path=str(config_path)) == {
        "block_px": 31,
        "connector_px": 5,
    }


@pytest.mark.parametrize(
    "config_text",
    [
        "template_sizes:\n  active_profile: high\n",
        """
template_sizes:
  active_profile: high
  profiles:
    high:
      block_px: abc
      connector_px: 5
""",
        """
template_sizes:
  active_profile: high
  profiles:
    high:
      block_px: 37
      connector_px: 0
""",
    ],
)
def test_load_template_geometry_rejects_bad_configs(tmp_path, config_text):
    config_path = tmp_path / "template_config.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    with pytest.raises((KeyError, ValueError)):
        load_template_geometry(config_path=str(config_path))


def test_visual_servo_service_contains_template_prior_fields():
    srv_path = os.path.join(
        os.path.dirname(PACKAGE_DIR),
        "image_process",
        "srv",
        "VisualServoOffset.srv",
    )
    text = open(srv_path, "r", encoding="utf-8").read()
    for field in (
        "string template_profile",
        "bool use_angle_prior",
        "float32 angle_center_deg",
        "float32 angle_window_deg",
        "float32 angle_step_deg",
        "bool use_position_prior",
        "float32 search_center_x",
        "float32 search_center_y",
        "float32 search_radius_px",
    ):
        assert field in text


def _install_module_stub(monkeypatch, name):
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_process_module_with_stubs(monkeypatch, module_name):
    rospy = _install_module_stub(monkeypatch, "rospy")
    rospy.Service = object
    rospy.Subscriber = object
    rospy.ServiceProxy = object
    rospy.get_param = lambda _name, default=None: default
    rospy.loginfo = lambda *_args, **_kwargs: None
    rospy.logwarn = lambda *_args, **_kwargs: None
    rospy.logerr = lambda *_args, **_kwargs: None
    rospy.init_node = lambda *_args, **_kwargs: None
    rospy.spin = lambda: None

    sensor_msgs = _install_module_stub(monkeypatch, "sensor_msgs")
    sensor_msgs.msg = _install_module_stub(monkeypatch, "sensor_msgs.msg")
    sensor_msgs.msg.Image = type("Image", (), {})

    cv_bridge = _install_module_stub(monkeypatch, "cv_bridge")
    cv_bridge.CvBridge = type("CvBridge", (), {})

    ultralytics = _install_module_stub(monkeypatch, "ultralytics")
    ultralytics.YOLO = object

    image_process = _install_module_stub(monkeypatch, "image_process")
    image_process.srv = _install_module_stub(monkeypatch, "image_process.srv")
    for name in (
        "GetTargetPos",
        "GetTargetPosResponse",
        "VisualTargetOffset",
        "VisualTargetOffsetResponse",
        "VisualBoardOffset",
        "VisualBoardOffsetResponse",
        "VisualServoOffset",
        "VisualServoOffsetResponse",
    ):
        setattr(image_process.srv, name, type(name, (), {}))

    camera = _install_module_stub(monkeypatch, "camera")
    camera.srv = _install_module_stub(monkeypatch, "camera.srv")
    camera.srv.pixel2world = type("pixel2world", (), {})
    camera.srv.pixel2worldRequest = type("pixel2worldRequest", (), {})

    process_path = os.path.join(PACKAGE_DIR, "scripts", "process.py")
    spec = importlib.util.spec_from_file_location(module_name, process_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_process_module_imports_with_ros_stubs(monkeypatch):
    rospy = _install_module_stub(monkeypatch, "rospy")
    rospy.Service = object
    rospy.Subscriber = object
    rospy.ServiceProxy = object
    rospy.get_param = lambda _name, default=None: default
    rospy.loginfo = lambda *_args, **_kwargs: None
    rospy.logwarn = lambda *_args, **_kwargs: None
    rospy.logerr = lambda *_args, **_kwargs: None
    rospy.init_node = lambda *_args, **_kwargs: None
    rospy.spin = lambda: None

    sensor_msgs = _install_module_stub(monkeypatch, "sensor_msgs")
    sensor_msgs.msg = _install_module_stub(monkeypatch, "sensor_msgs.msg")
    sensor_msgs.msg.Image = type("Image", (), {})

    cv_bridge = _install_module_stub(monkeypatch, "cv_bridge")
    cv_bridge.CvBridge = type("CvBridge", (), {})

    std_msgs = _install_module_stub(monkeypatch, "std_msgs")
    std_msgs.msg = _install_module_stub(monkeypatch, "std_msgs.msg")
    std_msgs.msg.Float32MultiArray = type("Float32MultiArray", (), {})
    std_msgs.msg.MultiArrayDimension = type("MultiArrayDimension", (), {})

    image_process = _install_module_stub(monkeypatch, "image_process")
    image_process.srv = _install_module_stub(monkeypatch, "image_process.srv")
    for name in (
        "GetTargetPos",
        "GetTargetPosResponse",
        "VisualTargetOffset",
        "VisualTargetOffsetResponse",
        "VisualBoardOffset",
        "VisualBoardOffsetResponse",
        "VisualServoOffset",
        "VisualServoOffsetResponse",
    ):
        setattr(image_process.srv, name, type(name, (), {}))

    camera = _install_module_stub(monkeypatch, "camera")
    camera.srv = _install_module_stub(monkeypatch, "camera.srv")
    camera.srv.pixel2world = type("pixel2world", (), {})
    camera.srv.pixel2worldRequest = type("pixel2worldRequest", (), {})

    process_path = os.path.join(PACKAGE_DIR, "scripts", "process.py")
    spec = importlib.util.spec_from_file_location("process_import_smoke", process_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "ImageProcessor")
    assert module.load_template_geometry.__name__ == "load_template_geometry"


def test_process_block_template_options_use_request_angle_prior(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_parse_smoke")
    processor = object.__new__(module.ImageProcessor)
    req = types.SimpleNamespace(
        template_profile="low",
        angle_step_deg=1.0,
        use_angle_prior=True,
        angle_center_deg=37.0,
        angle_window_deg=10.0,
        use_position_prior=False,
        search_center_x=0.0,
        search_center_y=0.0,
        search_radius_px=0.0,
    )

    options = processor.parse_block_template_match_options(req)

    assert options["template_profile"] == "low"
    assert options["angle_step"] == 1.0
    assert options["angle_center"] == 37.0
    assert options["angle_window"] == 10.0
    assert options["search_center"] is None
    assert options["search_radius"] is None
    assert (
        processor.make_block_angle_prior_message(
            options["angle_center"],
            options["angle_window"],
            options["angle_step"],
        )
        == "角度先验: center=37.0, window=10.0, step=1.0"
    )


def test_competition_module_imports_with_service_stubs(monkeypatch):
    rospy = _install_module_stub(monkeypatch, "rospy")
    rospy.init_node = lambda *_args, **_kwargs: None
    rospy.wait_for_service = lambda *_args, **_kwargs: None
    rospy.ServiceProxy = object

    sensor_msgs = _install_module_stub(monkeypatch, "sensor_msgs")
    sensor_msgs.msg = _install_module_stub(monkeypatch, "sensor_msgs.msg")
    sensor_msgs.msg.Image = type("Image", (), {})

    control = _install_module_stub(monkeypatch, "control")
    control.srv = _install_module_stub(monkeypatch, "control.srv")
    for name in ("arm", "armRequest", "motor", "motorRequest", "suck", "suckRequest"):
        setattr(control.srv, name, type(name, (), {}))

    image_process = _install_module_stub(monkeypatch, "image_process")
    image_process.srv = _install_module_stub(monkeypatch, "image_process.srv")
    for name in (
        "GetTargetPos",
        "GetTargetPosRequest",
        "VisualServoOffset",
        "VisualServoOffsetRequest",
    ):
        setattr(image_process.srv, name, type(name, (), {}))

    competition_path = os.path.join(
        os.path.dirname(PACKAGE_DIR),
        "competition",
        "scripts",
        "competition.py",
    )
    spec = importlib.util.spec_from_file_location("competition_import_smoke", competition_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "build_base_pick_list")
    assert len(module.build_base_pick_list()) == 34
    assert module.decode_category_code(2) == "z_blue"
    assert module.decode_category_code(-1) == ""

    parsed = module.parse_cube_location_response([1, 2, 3, 4, 5, 6])
    assert parsed == (1.0, 2.0, 3.0, 4.0, 5.0, "line")

    parsed_old = module.parse_cube_location_response([1, 2, 3, 4, 5], fallback_category="T")
    assert parsed_old == (1.0, 2.0, 3.0, 4.0, 5.0, "T")
