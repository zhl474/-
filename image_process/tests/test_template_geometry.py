import os
import sys
import threading
import types
import importlib.util

import cv2
import numpy as np
import pytest
import yaml


PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

if "ultralytics" not in sys.modules:
    ultralytics_stub = types.ModuleType("ultralytics")
    ultralytics_stub.YOLO = object
    sys.modules["ultralytics"] = ultralytics_stub

from image_process_lib.block_category import (
    BLOCK_CATEGORY_NAMES,
    normalize_category_name,
)
import image_process_lib.block_servo_detector as block_servo_detector_module
from image_process_lib.block_servo_detector import (
    _segment_roi_by_local_rgb_color,
    detect_block_with_high_prior_roi,
)
from image_process_lib.template_config import (
    load_color_segmentation_config,
    load_template_geometry,
)
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


def test_block_category_order_and_legacy_names_are_stable():
    assert BLOCK_CATEGORY_NAMES == (
        "L_blue",
        "L_yellow",
        "z_blue",
        "z_green",
        "square",
        "T",
        "line",
    )
    assert normalize_category_name("LL") == "L_yellow"
    assert normalize_category_name("O") == "square"


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
    debug_output = {}
    debug_rect = get_rect(
        image,
        block_px,
        connector_px,
        category,
        debug_full.copy(),
        0,
        0,
        angle_values=[0],
        debug_output=debug_output,
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

    np.testing.assert_allclose(debug_rect[0], full_rect[0], atol=1.0)
    assert debug_output["best_kernel"].ndim == 2
    assert debug_output["match_center"] == (float(full_rect[0][0]), float(full_rect[0][1]))
    assert "template_top_left" in debug_output
    assert "score" in debug_output
    np.testing.assert_allclose(cropped_rect[0], full_rect[0], atol=1.0)
    assert cropped_rect[2] == full_rect[2]


def _bgr_from_rgb(r_value, g_value, b_value):
    return np.array(
        [
            int(np.clip(round(float(b_value)), 0, 255)),
            int(np.clip(round(float(g_value)), 0, 255)),
            int(np.clip(round(float(r_value)), 0, 255)),
        ],
        dtype=np.uint8,
    )


def test_low_prior_roi_template_match_uses_local_rgb_color_seed():
    category = "T"
    block_px = 12
    connector_px = 3
    template_angle = 18.0
    high_theta = -template_angle
    target_center = (165, 116)
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    color_config = load_color_segmentation_config(category)
    target_bgr = _bgr_from_rgb(*color_config["rgb"])

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
    roi[shape_mask > 0] = target_bgr

    result = detect_block_with_high_prior_roi(
        image,
        template_geometry={"block_px": block_px, "connector_px": connector_px},
        category=category,
        high_theta_deg=high_theta,
        angle_window=3,
        angle_step=1,
        search_radius_px=30,
        fallback_search_radius_px=50,
        boundary_guard_px=3,
        white_s_max=45,
        white_v_min=180,
        min_foreground_area=50,
        timing_enabled=True,
    )

    assert result["found"] is True
    assert result["category"] == category
    assert abs(result["px"] - target_center[0]) <= 2.0
    assert abs(result["py"] - target_center[1]) <= 2.0
    assert abs(result["theta"] - high_theta) <= 1.0
    assert result["debug_panel"] is not None
    assert result["debug_panel"].size > 0
    timing = result["timing"]
    assert timing["状态"] == "成功"
    assert timing["ROI尺寸"] is not None
    assert timing["模板数量"] == 7
    assert timing["模板核尺寸"] is not None
    assert timing["后端"] in ("cpu", "cuda")
    assert timing["匹配模式"] == "快速30像素"
    for stage_name in ("先验ROI", "RGB分割", "模板生成", "张量准备", "卷积选优", "匹配收尾", "检测调试图"):
        assert timing["阶段毫秒"][stage_name] is not None


def test_low_prior_roi_template_match_skips_debug_images_when_disabled():
    category = "T"
    block_px = 12
    connector_px = 3
    template_angle = 18.0
    high_theta = -template_angle
    target_center = (165, 116)
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    color_config = load_color_segmentation_config(category)
    target_bgr = _bgr_from_rgb(*color_config["rgb"])

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
    image[y1:y1 + mask_h, x1:x1 + mask_w][shape_mask > 0] = target_bgr

    result = detect_block_with_high_prior_roi(
        image,
        template_geometry={"block_px": block_px, "connector_px": connector_px},
        category=category,
        high_theta_deg=high_theta,
        angle_window=3,
        angle_step=1,
        search_radius_px=30,
        fallback_search_radius_px=50,
        boundary_guard_px=3,
        min_foreground_area=50,
        debug_enabled=False,
    )

    assert result["found"] is True
    assert abs(result["px"] - target_center[0]) <= 2.0
    assert abs(result["py"] - target_center[1]) <= 2.0
    assert result["debug_image"] is None
    assert result["debug_panel"] is None


def test_low_pick_aligned_template_cache_hit_keeps_result_stable():
    category = "T"
    block_px = 12
    connector_px = 3
    template_angle = 18.0
    high_theta = -template_angle
    target_center = (165, 116)
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    color_config = load_color_segmentation_config(category)
    target_bgr = _bgr_from_rgb(*color_config["rgb"])

    kernels, _, _ = create_rotation_kernels(
        block_px,
        connector_px,
        category,
        device="cpu",
        angle_values=[template_angle],
    )
    shape_mask = (kernels[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    x1 = target_center[0] - shape_mask.shape[1] // 2
    y1 = target_center[1] - shape_mask.shape[0] // 2
    image[y1:y1 + shape_mask.shape[0], x1:x1 + shape_mask.shape[1]][shape_mask > 0] = target_bgr

    block_servo_detector_module._LOW_PICK_TEMPLATE_CACHE.clear()
    common_kwargs = {
        "template_geometry": {"block_px": block_px, "connector_px": connector_px},
        "category": category,
        "high_theta_deg": high_theta,
        "angle_window": 3,
        "angle_step": 1,
        "search_radius_px": 30,
        "fallback_search_radius_px": 50,
        "boundary_guard_px": 3,
        "min_foreground_area": 50,
        "debug_enabled": False,
        "timing_enabled": True,
    }
    first_result = detect_block_with_high_prior_roi(image, **common_kwargs)
    second_result = detect_block_with_high_prior_roi(image, **common_kwargs)

    assert first_result["found"] is True
    assert second_result["found"] is True
    assert abs(first_result["px"] - target_center[0]) <= 2.0
    assert abs(first_result["py"] - target_center[1]) <= 2.0
    assert abs(first_result["theta"] - high_theta) <= 1.0
    np.testing.assert_allclose(
        (second_result["px"], second_result["py"], second_result["theta"]),
        (first_result["px"], first_result["py"], first_result["theta"]),
        atol=1e-6,
    )
    first_timing = first_result["timing"]
    second_timing = second_result["timing"]
    assert first_timing["匹配模式"] == "快速30像素"
    assert first_timing["模板缓存"] == "未命中"
    assert second_timing["模板缓存"] == "命中"
    assert first_timing["阶段毫秒"]["模板生成"] is not None
    assert first_timing["匹配图尺寸"] == (61, 61)


def test_low_prior_roi_timing_marks_template_stages_not_executed_when_foreground_too_small():
    category = "T"
    block_px = 12
    connector_px = 3
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    target_bgr = _bgr_from_rgb(*load_color_segmentation_config(category)["rgb"])

    # 中心 seed 搜索窗口的第一个候选 patch 与目标颜色完全一致，面积仅 49 px。
    image[80:87, 120:127] = target_bgr
    result = detect_block_with_high_prior_roi(
        image,
        template_geometry={"block_px": block_px, "connector_px": connector_px},
        category=category,
        high_theta_deg=0.0,
        angle_window=3,
        angle_step=1,
        search_radius_px=30,
        fallback_search_radius_px=50,
        boundary_guard_px=3,
        min_foreground_area=100,
        debug_enabled=False,
        timing_enabled=True,
    )

    assert result["found"] is False
    timing = result["timing"]
    assert timing["状态"] == "前景面积过小"
    assert timing["ROI尺寸"] is not None
    assert timing["前景面积"] < 100
    assert timing["阶段毫秒"]["先验ROI"] is not None
    assert timing["阶段毫秒"]["RGB分割"] is not None
    for stage_name in ("张量准备", "卷积选优", "匹配收尾"):
        assert timing["阶段毫秒"][stage_name] is None


def test_local_rgb_segment_returns_seed_debug_boxes():
    image = np.full((90, 90, 3), 255, dtype=np.uint8)
    image[25:70, 25:70] = _bgr_from_rgb(0, 0, 0)

    mask, stages = _segment_roi_by_local_rgb_color(image, "T", return_stages=True)

    assert mask.shape == image.shape[:2]
    assert "seed_search_box" in stages
    assert "seed_patch_box" in stages
    assert "local_color" in stages
    assert cv2.countNonZero(mask) > 0


def test_local_rgb_seed_prefers_homogeneous_patch_over_mixed_boundary():
    category = "T"
    config = load_color_segmentation_config(category)
    patch_size = config["seed_patch_size"]
    seed_stride = config["seed_stride"]
    seed_search_half_size = config["seed_search_half_size"]
    prior_rgb = np.array(config["rgb"], dtype=np.float32)
    homogeneous_rgb = np.clip(prior_rgb + np.array([5.0, 0.0, 0.0]), 0, 255)
    low_rgb = np.clip(prior_rgb - 50.0, 0, 255)
    high_rgb = np.clip(prior_rgb + 50.0, 0, 255)

    image = np.full((90, 90, 3), 255, dtype=np.uint8)
    search_start = max(0, int(np.floor(45.0 - seed_search_half_size)))
    boundary_x = search_start + 4 * seed_stride
    homogeneous_x = search_start + 6 * seed_stride
    patch_y = search_start + 4 * seed_stride

    boundary_bgr = image[patch_y:patch_y + patch_size, boundary_x:boundary_x + patch_size]
    split_col = patch_size // 2
    boundary_bgr[:, :split_col] = _bgr_from_rgb(*low_rgb)
    boundary_bgr[:, split_col:] = _bgr_from_rgb(*high_rgb)
    image[patch_y:patch_y + patch_size, homogeneous_x:homogeneous_x + patch_size] = _bgr_from_rgb(*homogeneous_rgb)

    _mask, stages = _segment_roi_by_local_rgb_color(image, category, return_stages=True)

    np.testing.assert_allclose(stages["local_color"], homogeneous_rgb, atol=1.0)
    assert stages["seed_variance_d2"] < 1.0


def _valid_template_size_config():
    return {
        "active_profile": "high",
        "profiles": {
            "high": {
                "block_px": 37,
                "connector_px": 5,
            },
        },
    }


def _valid_color_segmentation_config():
    return {
        "seed_search_half_size": 40,
        "seed_patch_size": 7,
        "seed_stride": 8,
        "local_dist_thresh": 45,
        "categories": {
            "T": {
                "rgb": [10, 20, 30],
            },
        },
    }


def _write_template_config(tmp_path, color_segmentation):
    config_data = {
        "template_sizes": _valid_template_size_config(),
    }
    if color_segmentation is not None:
        config_data["color_segmentation"] = color_segmentation

    config_path = tmp_path / "template_config.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return config_path


def test_load_color_segmentation_config_reads_category_prior(tmp_path):
    config_path = _write_template_config(tmp_path, _valid_color_segmentation_config())

    config = load_color_segmentation_config("T", config_path=str(config_path))

    assert config == {
        "seed_search_half_size": 40,
        "seed_patch_size": 7,
        "seed_stride": 8,
        "local_dist_thresh": 45.0,
        "rgb": (10.0, 20.0, 30.0),
    }


def test_load_color_segmentation_config_rejects_missing_section(tmp_path):
    config_path = _write_template_config(tmp_path, None)

    with pytest.raises(ValueError, match="color_segmentation"):
        load_color_segmentation_config("T", config_path=str(config_path))


@pytest.mark.parametrize(
    "missing_key",
    [
        "seed_search_half_size",
        "seed_patch_size",
        "seed_stride",
        "local_dist_thresh",
    ],
)
def test_load_color_segmentation_config_rejects_missing_global_field(tmp_path, missing_key):
    color_config = _valid_color_segmentation_config()
    del color_config[missing_key]
    config_path = _write_template_config(tmp_path, color_config)

    with pytest.raises(ValueError, match=missing_key):
        load_color_segmentation_config("T", config_path=str(config_path))


@pytest.mark.parametrize(
    ("bad_key", "bad_value"),
    [
        ("seed_search_half_size", 0),
        ("seed_patch_size", -1),
        ("seed_stride", "bad"),
        ("local_dist_thresh", 0),
    ],
)
def test_load_color_segmentation_config_rejects_bad_global_value(tmp_path, bad_key, bad_value):
    color_config = _valid_color_segmentation_config()
    color_config[bad_key] = bad_value
    config_path = _write_template_config(tmp_path, color_config)

    with pytest.raises(ValueError, match=bad_key):
        load_color_segmentation_config("T", config_path=str(config_path))


def test_load_color_segmentation_config_rejects_missing_category(tmp_path):
    color_config = _valid_color_segmentation_config()
    color_config["categories"] = {}
    config_path = _write_template_config(tmp_path, color_config)

    with pytest.raises(KeyError, match="T"):
        load_color_segmentation_config("T", config_path=str(config_path))


def test_load_color_segmentation_config_rejects_missing_rgb_prior(tmp_path):
    color_config = _valid_color_segmentation_config()
    del color_config["categories"]["T"]["rgb"]
    config_path = _write_template_config(tmp_path, color_config)

    with pytest.raises(ValueError, match="rgb"):
        load_color_segmentation_config("T", config_path=str(config_path))


@pytest.mark.parametrize("bad_rgb", [[1, 2], [1, 2, "bad"]])
def test_load_color_segmentation_config_rejects_bad_rgb_prior(tmp_path, bad_rgb):
    color_config = _valid_color_segmentation_config()
    color_config["categories"]["T"]["rgb"] = bad_rgb
    config_path = _write_template_config(tmp_path, color_config)

    with pytest.raises(ValueError, match="rgb"):
        load_color_segmentation_config("T", config_path=str(config_path))


def test_local_rgb_segment_rejects_window_without_patch():
    image = np.full((5, 5, 3), 255, dtype=np.uint8)

    with pytest.raises(ValueError, match="无法枚举 seed patch"):
        _segment_roi_by_local_rgb_color(image, "T")


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


def test_block_servo_service_has_explicit_high_angle_prior():
    srv_path = os.path.join(
        os.path.dirname(PACKAGE_DIR),
        "image_process",
        "srv",
        "DetectBlockOffset.srv",
    )
    text = open(srv_path, "r", encoding="utf-8").read()
    for field in ("string category", "float64 high_angle_deg", "float64 dx_px", "float64 dy_px"):
        assert field in text


def _install_module_stub(monkeypatch, name):
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_process_module_with_stubs(monkeypatch, module_name):
    class ServiceStub:
        def __init__(self, *args, **kwargs):
            self.args = args
            for key, value in kwargs.items():
                setattr(self, key, value)

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
    rospy.Time = types.SimpleNamespace(now=lambda: 100)

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
        "DetectBlockOffset",
        "DetectBlockOffsetResponse",
        "DetectBoardOffset",
        "DetectBoardOffsetResponse",
        "GetTaskTarget",
        "GetTaskTargetResponse",
        "PrepareTask",
        "PrepareTaskResponse",
    ):
        setattr(image_process.srv, name, type(name, (ServiceStub,), {}))

    camera = _install_module_stub(monkeypatch, "camera")
    camera.srv = _install_module_stub(monkeypatch, "camera.srv")
    camera.srv.GetStableWorldPoints = type("GetStableWorldPoints", (ServiceStub,), {})

    process_path = os.path.join(PACKAGE_DIR, "scripts", "process.py")
    spec = importlib.util.spec_from_file_location(module_name, process_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_image_processor_runtime(monkeypatch, image_node_module):
    """屏蔽与本组配置初始化测试无关的 ROS 注册和模型加载。"""
    monkeypatch.setattr(image_node_module, "YOLO", lambda _path: object())
    monkeypatch.setattr(
        image_node_module.rospy,
        "Subscriber",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        image_node_module.rospy,
        "Service",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        image_node_module.rospy,
        "on_shutdown",
        lambda _callback: None,
        raising=False,
    )


def _set_execution_calibration_mode(
    monkeypatch,
    tmp_path,
    image_node_module,
    calibration_mode,
):
    """为初始化测试显式指定标定模式，避免依赖现场运行配置。"""
    with open(image_node_module.EXECUTION_CONFIG_PATH, "r", encoding="utf-8") as config_file:
        execution_data = yaml.safe_load(config_file) or {}
    execution_data["calibration_mode"] = calibration_mode
    execution_path = tmp_path / f"execution_{calibration_mode}.yaml"
    execution_path.write_text(
        yaml.safe_dump(execution_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(image_node_module, "EXECUTION_CONFIG_PATH", str(execution_path))


def test_process_module_imports_with_ros_stubs(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_import_smoke")
    assert hasattr(module, "ImageProcessor")
    assert callable(module.main)


def test_runtime_initialization_never_creates_depth_client(monkeypatch, tmp_path):
    module = _load_process_module_with_stubs(monkeypatch, "process_default_without_depth")
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    _set_execution_calibration_mode(monkeypatch, tmp_path, image_node_module, False)
    monkeypatch.setattr(image_node_module, "YOLO", lambda _path: object())
    subscriber_calls = []
    monkeypatch.setattr(
        image_node_module.rospy,
        "Subscriber",
        lambda *args, **kwargs: subscriber_calls.append((args, kwargs)) or object(),
    )
    monkeypatch.setattr(
        image_node_module.rospy,
        "Service",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        image_node_module.rospy,
        "on_shutdown",
        lambda _callback: None,
        raising=False,
    )
    monkeypatch.setattr(
        image_node_module.rospy,
        "ServiceProxy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("正式模式不应创建深度服务客户端")
        ),
    )

    processor = module.ImageProcessor()

    assert processor.calibration_mode is False
    assert processor.image_topic == "/camera/image_rect"
    assert subscriber_calls[0][0][0] == "/camera/image_rect"
    assert processor.stable_world_points_client is None
    assert processor.high_tcp_localizer is not None
    assert processor.dynamic_board_selection_mode == "shadow"
    assert processor.dynamic_board_library.board_count == 8460
    assert processor.dynamic_board_library.placement_count == 2021


def test_dynamic_execute_initialization_rejects_enabled_visual_servo(
    monkeypatch,
    tmp_path,
):
    module = _load_process_module_with_stubs(
        monkeypatch,
        "process_dynamic_execute_servo_guard",
    )
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    execution_data = yaml.safe_load(
        open(image_node_module.EXECUTION_CONFIG_PATH, "r", encoding="utf-8")
    )
    execution_data["servo"]["enabled"] = True
    execution_path = tmp_path / "execution.yaml"
    execution_path.write_text(
        yaml.safe_dump(execution_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(image_node_module, "EXECUTION_CONFIG_PATH", str(execution_path))
    original_get_param = image_node_module.rospy.get_param
    monkeypatch.setattr(
        image_node_module.rospy,
        "get_param",
        lambda name, default=None: (
            "execute"
            if name == "~dynamic_board_selection_mode"
            else original_get_param(name, default)
        ),
    )
    _stub_image_processor_runtime(monkeypatch, image_node_module)

    with pytest.raises(ValueError, match="动态盘面 execute.*servo.enabled=false"):
        module.ImageProcessor()


def test_formal_launch_files_explicitly_use_rectified_image_topic():
    src_dir = os.path.dirname(PACKAGE_DIR)
    for launch_name in ("competition.launch", "calibration.launch"):
        launch_path = os.path.join(src_dir, "competition", "launch", launch_name)
        text = open(launch_path, "r", encoding="utf-8").read()
        assert '<param name="image_topic" value="/camera/image_rect"/>' in text


@pytest.mark.parametrize(
    ("visual_servo_enabled", "expected_safety_offset"),
    [
        (True, (0.0, 0.0)),
        (False, (-94.1, -13.8)),
    ],
)
def test_image_node_selects_high_tcp_safety_offset_from_servo_mode(
    monkeypatch,
    tmp_path,
    visual_servo_enabled,
    expected_safety_offset,
):
    module = _load_process_module_with_stubs(
        monkeypatch,
        f"process_safety_offset_{visual_servo_enabled}",
    )
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    execution_data = yaml.safe_load(
        open(image_node_module.EXECUTION_CONFIG_PATH, "r", encoding="utf-8")
    )
    execution_data["calibration_mode"] = False
    execution_data["servo"]["enabled"] = visual_servo_enabled
    execution_path = tmp_path / "execution.yaml"
    execution_path.write_text(
        yaml.safe_dump(execution_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(image_node_module, "EXECUTION_CONFIG_PATH", str(execution_path))
    _stub_image_processor_runtime(monkeypatch, image_node_module)
    received = {}

    def make_localizer(**kwargs):
        received.update(kwargs)
        return object()

    monkeypatch.setattr(image_node_module, "HighPixelToTcpLocalizer", make_localizer)

    processor = module.ImageProcessor()

    with open(image_node_module.PERCEPTION_CONFIG_PATH, "r", encoding="utf-8") as config_file:
        localization_config = (yaml.safe_load(config_file) or {})["high_tcp_localization"]

    assert processor.visual_servo_enabled is visual_servo_enabled
    assert processor.high_tcp_safety_xy_offset == expected_safety_offset
    assert received["safety_xy_offset"] == expected_safety_offset
    assert received["tcp_min_xyz"] == (
        localization_config["safe_x_range_mm"][0],
        localization_config["safe_y_range_mm"][0],
        165.0,
    )
    assert received["tcp_max_xyz"] == (
        localization_config["safe_x_range_mm"][1],
        localization_config["safe_y_range_mm"][1],
        None,
    )


def test_calibration_mode_forces_closed_loop_high_tcp_safety_offset(monkeypatch, tmp_path):
    module = _load_process_module_with_stubs(monkeypatch, "process_calibration_safety_offset")
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    execution_data = yaml.safe_load(
        open(image_node_module.EXECUTION_CONFIG_PATH, "r", encoding="utf-8")
    )
    execution_data["servo"]["enabled"] = False
    execution_path = tmp_path / "execution.yaml"
    execution_path.write_text(
        yaml.safe_dump(execution_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(image_node_module, "EXECUTION_CONFIG_PATH", str(execution_path))
    monkeypatch.setattr(
        image_node_module.rospy,
        "get_param",
        lambda name, default=None: True if name == "~calibration_mode" else default,
    )
    monkeypatch.setattr(image_node_module.np, "load", lambda _path: np.eye(4))
    _stub_image_processor_runtime(monkeypatch, image_node_module)
    monkeypatch.setattr(
        image_node_module.rospy,
        "ServiceProxy",
        lambda *_args, **_kwargs: object(),
    )

    processor = module.ImageProcessor()

    assert processor.calibration_mode is True
    assert processor.visual_servo_enabled is True
    assert processor.high_tcp_safety_xy_offset == (0.0, 0.0)


@pytest.mark.parametrize("invalid_enabled", ["false", 0, None])
def test_image_node_rejects_non_boolean_servo_enabled(
    monkeypatch,
    tmp_path,
    invalid_enabled,
):
    module = _load_process_module_with_stubs(
        monkeypatch,
        f"process_invalid_servo_enabled_{invalid_enabled}",
    )
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    execution_data = yaml.safe_load(
        open(image_node_module.EXECUTION_CONFIG_PATH, "r", encoding="utf-8")
    )
    execution_data["servo"]["enabled"] = invalid_enabled
    execution_path = tmp_path / "execution.yaml"
    execution_path.write_text(
        yaml.safe_dump(execution_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(image_node_module, "EXECUTION_CONFIG_PATH", str(execution_path))
    _stub_image_processor_runtime(monkeypatch, image_node_module)

    with pytest.raises(ValueError, match="servo.enabled 必须是 YAML 布尔值"):
        module.ImageProcessor()


@pytest.mark.parametrize(
    "invalid_offset",
    [[-94.1], [-94.1, -13.8, 0.0], [float("nan"), -13.8]],
)
def test_image_node_rejects_invalid_camera_to_sucker_offset(
    monkeypatch,
    tmp_path,
    invalid_offset,
):
    module = _load_process_module_with_stubs(
        monkeypatch,
        f"process_invalid_safety_offset_{len(invalid_offset)}",
    )
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    visual_config_path = tmp_path / "visual_servo.yaml"
    visual_config_path.write_text(
        yaml.safe_dump(
            {"camera_to_sucker_offset_mm": invalid_offset},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        image_node_module,
        "VISUAL_SERVO_CONFIG_PATH",
        str(visual_config_path),
    )
    _stub_image_processor_runtime(monkeypatch, image_node_module)

    with pytest.raises(ValueError, match="camera_to_sucker_offset_mm 必须包含 2 个有限数值"):
        module.ImageProcessor()


def test_calibration_initialization_creates_batch_depth_client_without_waiting(
    monkeypatch,
    tmp_path,
):
    module = _load_process_module_with_stubs(monkeypatch, "process_depth_client_without_wait")
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    execution_data = yaml.safe_load(
        open(image_node_module.EXECUTION_CONFIG_PATH, "r", encoding="utf-8")
    )
    execution_path = tmp_path / "execution.yaml"
    execution_path.write_text(
        yaml.safe_dump(execution_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(image_node_module, "EXECUTION_CONFIG_PATH", str(execution_path))
    monkeypatch.setattr(
        image_node_module.rospy,
        "get_param",
        lambda name, default=None: True if name == "~calibration_mode" else default,
    )
    monkeypatch.setattr(image_node_module.np, "load", lambda _path: np.eye(4))
    monkeypatch.setattr(image_node_module, "YOLO", lambda _path: object())
    monkeypatch.setattr(
        image_node_module.rospy,
        "Subscriber",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        image_node_module.rospy,
        "Service",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        image_node_module.rospy,
        "on_shutdown",
        lambda _callback: None,
        raising=False,
    )
    proxy_calls = []

    class StableWorldPointsClient:
        def wait_for_service(self):
            raise AssertionError("标定模式初始化不应阻塞等待服务")

    stable_world_points_client = StableWorldPointsClient()
    monkeypatch.setattr(
        image_node_module.rospy,
        "ServiceProxy",
        lambda service_name, service_type: (
            proxy_calls.append((service_name, service_type)) or stable_world_points_client
        ),
    )

    processor = module.ImageProcessor()

    assert processor.calibration_mode is True
    assert processor.high_tcp_localizer is None
    assert processor.stable_world_points_client is stable_world_points_client
    assert proxy_calls == [
        ("/camera/stable_world_points", image_node_module.GetStableWorldPoints)
    ]


@pytest.mark.parametrize(
    "private_paths,expected_relative_paths",
    [
        (
            {},
            (
                "image_process/config/block_pixel_to_tcp_calibration.yaml",
                "image_process/config/tray_pixel_to_tcp_calibration.yaml",
            ),
        ),
        (
            {
                "~block_pixel_to_tcp_calibration_path": "custom/方块标定.yaml",
                "~tray_pixel_to_tcp_calibration_path": "custom/托盘标定.yaml",
            },
            ("custom/方块标定.yaml", "custom/托盘标定.yaml"),
        ),
    ],
)
def test_relative_calibration_paths_from_yaml_and_private_params_use_src_dir(
    monkeypatch,
    tmp_path,
    private_paths,
    expected_relative_paths,
):
    module = _load_process_module_with_stubs(
        monkeypatch,
        "process_relative_calibration_paths",
    )
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    _set_execution_calibration_mode(monkeypatch, tmp_path, image_node_module, False)
    original_get_param = image_node_module.rospy.get_param
    monkeypatch.setattr(
        image_node_module.rospy,
        "get_param",
        lambda name, default=None: private_paths.get(name, original_get_param(name, default)),
    )
    monkeypatch.setattr(image_node_module, "YOLO", lambda _path: object())
    monkeypatch.setattr(
        image_node_module.rospy,
        "Subscriber",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        image_node_module.rospy,
        "Service",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        image_node_module.rospy,
        "on_shutdown",
        lambda _callback: None,
        raising=False,
    )
    received = {}

    def make_localizer(**kwargs):
        received.update(kwargs)
        return object()

    monkeypatch.setattr(image_node_module, "HighPixelToTcpLocalizer", make_localizer)

    module.ImageProcessor()

    expected_block, expected_tray = (
        os.path.join(image_node_module.SRC_DIR, relative_path)
        for relative_path in expected_relative_paths
    )
    assert received["block_calibration_path"] == expected_block
    assert received["tray_calibration_path"] == expected_tray


@pytest.mark.parametrize("minimum_tcp_z_mm", [0.0, -1.0])
def test_image_node_rejects_nonpositive_minimum_tcp_z(
    monkeypatch,
    tmp_path,
    minimum_tcp_z_mm,
):
    module = _load_process_module_with_stubs(
        monkeypatch,
        f"process_bad_minimum_z_{minimum_tcp_z_mm}",
    )
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    execution_path = tmp_path / "execution.yaml"
    execution_path.write_text(
        yaml.safe_dump(
            {
                "shooting_pose": [-250.0, 20.0, 380.0, -180.0, 0.0, 90.0],
                "motion": {
                    "minimum_tcp_z_mm": minimum_tcp_z_mm,
                    "pick_surface_offset_mm": 162.0,
                },
                "servo": {},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(image_node_module, "EXECUTION_CONFIG_PATH", str(execution_path))
    monkeypatch.setattr(image_node_module, "YOLO", lambda _path: object())

    with pytest.raises(ValueError, match="minimum_tcp_z_mm 必须是大于 0"):
        module.ImageProcessor()


def _make_timestamp_snapshot_processor(module, timeout_sec=0.05):
    """构造只包含时间戳快照状态的轻量节点实例。"""
    processor = object.__new__(module.ImageProcessor)
    processor.image_lock = threading.Lock()
    processor.image_condition = threading.Condition(processor.image_lock)
    processor.latest_image = None
    processor.latest_image_stamp = None
    processor.fresh_image_timeout_sec = timeout_sec
    return processor


def _publish_timestamped_test_image(processor, image, stamp):
    """模拟收到带发布时间戳的相机图像。"""
    with processor.image_condition:
        processor.latest_image = image
        processor.latest_image_stamp = stamp
        processor.image_condition.notify_all()


def test_timestamp_snapshot_rejects_image_published_before_request(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_timestamp_snapshot_timeout")
    processor = _make_timestamp_snapshot_processor(module, timeout_sec=0.01)
    _publish_timestamped_test_image(processor, np.full((2, 2, 3), 7, dtype=np.uint8), stamp=10)

    assert processor.get_image_snapshot_newer_than(10) is None


def test_timestamp_snapshot_waits_for_image_published_after_request(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_timestamp_snapshot_new_frame")
    processor = _make_timestamp_snapshot_processor(module)
    _publish_timestamped_test_image(processor, np.full((2, 2, 3), 1, dtype=np.uint8), stamp=10)
    wait_started = threading.Event()
    original_wait = processor.image_condition.wait

    def signal_before_wait(timeout=None):
        wait_started.set()
        return original_wait(timeout)

    processor.image_condition.wait = signal_before_wait
    result = []
    worker = threading.Thread(
        target=lambda: result.append(processor.get_image_snapshot_newer_than(10))
    )
    worker.start()
    assert wait_started.wait(timeout=1.0)
    new_image = np.full((2, 2, 3), 2, dtype=np.uint8)
    _publish_timestamped_test_image(processor, new_image, stamp=11)
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert len(result) == 1
    np.testing.assert_array_equal(result[0], new_image)
    assert result[0] is not new_image


def test_task_target_service_rejects_out_of_range_index(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_target_bounds")
    processor = object.__new__(module.ImageProcessor)
    processor.task_targets = []

    response = processor.get_task_target(types.SimpleNamespace(index=0))

    assert response.success is False
    assert "越界" in response.message


def test_task_target_service_returns_high_localization_diagnostics(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_target_diagnostics")
    processor = object.__new__(module.ImageProcessor)
    processor.task_targets = [
        types.SimpleNamespace(
            pick_observation_pose=[1.0] * 6,
            place_observation_pose=[2.0] * 6,
            row=3.0,
            col=4.0,
            category="T",
            detected_angle_deg=5.0,
            rotation_delta_deg=6.0,
            pick_surface_z_mm=7.0,
            pick_surface_z_valid=True,
            pick_high_detected_pixel_xy=(100.5, 200.5),
            pick_high_depth_sample_pixel_xy=(0.0, 0.0),
            pick_high_image_center_xy=(640.0, 360.0),
            pick_high_world_position=(0.0, 0.0, 0.0),
            pick_high_world_position_valid=False,
            pick_rough_localization_source="tcp_calibration",
            pick_depth_valid_frame_count=0,
            pick_depth_median_mm=0.0,
            pick_depth_mad_mm=0.0,
            pick_calibration_target_tcp_z_mm=200.0,
            place_high_detected_pixel_xy=(300.5, 400.5),
            place_high_depth_sample_pixel_xy=(0.0, 0.0),
            place_high_image_center_xy=(640.0, 360.0),
            place_high_world_position=(0.0, 0.0, 0.0),
            place_high_world_position_valid=False,
            place_rough_localization_source="tcp_calibration",
            place_depth_valid_frame_count=0,
            place_depth_median_mm=0.0,
            place_depth_mad_mm=0.0,
            place_calibration_target_tcp_z_mm=193.0,
        )
    ]

    response = processor.get_task_target(types.SimpleNamespace(index=0))

    assert response.success is True
    assert response.pick_high_detected_pixel_xy == [100.5, 200.5]
    assert response.pick_high_world_position == [0.0, 0.0, 0.0]
    assert response.pick_high_world_position_valid is False
    assert response.pick_rough_localization_source == "tcp_calibration"
    assert response.place_high_detected_pixel_xy == [300.5, 400.5]
    assert response.place_high_world_position == [0.0, 0.0, 0.0]
    assert response.place_rough_localization_source == "tcp_calibration"


def test_high_localization_diagnostic_marks_tcp_calibration_without_world_coordinate(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_tcp_calibration_diagnostics")

    diagnostic = module.ImageProcessor.make_high_localization_diagnostic(
        100.6,
        200.4,
        (720, 1280, 3),
        (101.0, 200.0),
    )

    assert diagnostic["high_depth_sample_pixel_xy"] == (101.0, 200.0)
    assert diagnostic["high_world_position_valid"] is False
    assert diagnostic["high_world_position"] == (0.0, 0.0, 0.0)
    assert diagnostic["rough_localization_source"] == "tcp_calibration"


def test_prepare_task_clears_previous_result_when_image_is_missing(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_prepare_no_image")
    processor = object.__new__(module.ImageProcessor)
    processor.task_targets = [object()]
    processor.board_grid_points = object()
    processor.board_grid_image_shape = (1, 1)
    processor.board_grid_image = object()
    processor.calibration_mode = False
    processor.fresh_image_timeout_sec = 0.5
    received_stamps = []
    processor.get_image_snapshot_newer_than = lambda stamp: received_stamps.append(stamp) or None

    response = processor.prepare_task(types.SimpleNamespace(advanced=False, place_order=[]))

    assert response.success is False
    assert processor.task_targets == []
    assert processor.board_grid_points is None
    assert received_stamps == [100]


def test_prepare_task_uses_one_snapshot_and_clears_targets_when_any_point_fails(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_prepare_point_failure")
    processor = object.__new__(module.ImageProcessor)
    processor.task_targets = [object()]
    processor.board_grid_points = object()
    processor.board_grid_image_shape = (1, 1)
    processor.board_grid_image = object()
    processor.calibration_mode = False
    processor.fresh_image_timeout_sec = 0.5
    snapshot = np.zeros((6, 8, 3), dtype=np.uint8)
    processor.get_image_snapshot_newer_than = lambda _stamp: snapshot
    received_images = []
    processor._detect_board_for_task = lambda image: received_images.append(image)
    raw_blocks = [{"category": "T", "px": 1.0, "py": 2.0, "theta": 0.0}]
    processor._detect_blocks_automatic = lambda image: (
        received_images.append(image) or (raw_blocks, image.copy(), {})
    )
    processor._summarize_detected_blocks = lambda blocks: (
        blocks,
        [1, 0, 0, 0, 0, 0, 0],
    )
    processor._load_layout_for_request = lambda _request, _counts: ([object()], "测试布局")
    processor._validate_high_task_safety = lambda *_args: None
    processor._edit_and_rematch_blocks = lambda _image, blocks, _geometry, debug: (
        blocks,
        debug,
    )
    processor._save_high_block_debug_images = lambda *_args: None
    processor._build_observed_blocks_for_task = lambda blocks, _shape: blocks

    def fail_tray_calibration(_layout, _image_shape):
        raise ValueError("托盘预测 TCP 超出安全范围")

    processor._build_placement_targets = fail_tray_calibration

    response = processor.prepare_task(types.SimpleNamespace(advanced=False, place_order=[]))

    assert response.success is False
    assert response.task_count == 0
    assert "超出安全范围" in response.message
    assert processor.task_targets == []
    assert len(received_images) == 2
    assert all(image is snapshot for image in received_images)


def test_block_offset_service_uses_configured_angle_prior(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_angle_prior")
    processor = object.__new__(module.ImageProcessor)
    processor.block_angle_step_deg = 1.0
    processor.block_angle_window_deg = 10.0
    received = {}
    processor.detect_block_visual_offset = lambda **kwargs: received.update(kwargs) or {
        "found": True, "px": 1, "py": 2, "dx_px": 3, "dy_px": 4,
        "theta": 5, "score": 1, "message": "成功",
    }
    response = processor.detect_block_offset_service(types.SimpleNamespace(category="T", high_angle_deg=37.0))

    assert response.found
    assert received["angle_center"] == 37.0
    assert received["angle_window"] == 10.0
    assert received["angle_step"] == 1.0


def test_competition_module_imports_with_service_stubs(monkeypatch):
    rospy = _install_module_stub(monkeypatch, "rospy")
    rospy.init_node = lambda *_args, **_kwargs: None
    rospy.wait_for_service = lambda *_args, **_kwargs: None
    rospy.ServiceProxy = object

    control = _install_module_stub(monkeypatch, "control")
    control.srv = _install_module_stub(monkeypatch, "control.srv")
    for name in (
        "GetActualPose", "GetActualPoseRequest",
        "MoveArm", "MoveArmRequest", "RotateTool", "RotateToolRequest", "SetSuction", "SetSuctionRequest",
    ):
        setattr(control.srv, name, type(name, (), {}))

    image_process = _install_module_stub(monkeypatch, "image_process")
    image_process.srv = _install_module_stub(monkeypatch, "image_process.srv")
    for name in (
        "DetectBlockOffset", "DetectBlockOffsetRequest", "DetectBoardOffset", "DetectBoardOffsetRequest",
        "GetTaskTarget", "GetTaskTargetRequest", "PrepareTask", "PrepareTaskRequest",
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

    assert hasattr(module, "TaskRunner")
    assert callable(module.main)


def test_calibrated_height_uses_predicted_tcp_z_without_depth_client(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_calibrated_height")
    processor = object.__new__(module.ImageProcessor)
    processor.block_observation_height_mm = 192.0
    processor.pick_surface_offset_mm = 162.0
    processor.minimum_tcp_z_mm = 165.0

    surface_z, sample_pixel = processor.resolve_pick_surface_height(12.2, 33.8, 200.5)

    assert surface_z == pytest.approx(8.5)
    assert sample_pixel == (0.0, 0.0)


def test_block_task_detection_uses_calibrated_tcp_pose_and_surface_height(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_block_tcp_pipeline")
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    monkeypatch.setattr(image_node_module, "load_template_geometry", lambda _mode: {})
    monkeypatch.setattr(image_node_module, "save_image_to_path", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        image_node_module,
        "detect_blocks_in_image",
        lambda *_args, **_kwargs: ([{
            "category": "T",
            "px": 500.0,
            "py": 300.0,
            "theta": 12.5,
        }], np.zeros((4, 4, 3), dtype=np.uint8)),
    )
    processor = object.__new__(module.ImageProcessor)
    processor.model = object()
    processor.save_top_surface_mask_vis = False
    processor.high_template_match_debug_path = ""
    processor.block_observation_height_mm = 192.0
    processor.pick_surface_offset_mm = 162.0
    processor.minimum_tcp_z_mm = 165.0
    processor.high_tcp_localizer = types.SimpleNamespace(
        locate_block=lambda pixel: [-250.0, 20.0, 200.0, -180.0, 0.0, 90.0]
    )

    blocks, counts = processor._detect_blocks_for_task(
        np.zeros((720, 1280, 3), dtype=np.uint8)
    )

    assert len(blocks) == 1
    assert sum(counts) == 1
    assert blocks[0].observation_pose == (-250.0, 20.0, 200.0, -180.0, 0.0, 90.0)
    assert blocks[0].pick_surface_z_mm == pytest.approx(8.0)
    assert blocks[0].pick_surface_z_valid is True
    assert blocks[0].high_world_position_valid is False
    assert blocks[0].rough_localization_source == "tcp_calibration"


@pytest.mark.parametrize(
    ("preview_device", "expected_cuda_visible_devices"),
    [("cpu", ""), ("cuda", "7")],
)
def test_high_mask_editor_subprocess_uses_configured_device_and_validated_commit(
    monkeypatch,
    tmp_path,
    preview_device,
    expected_cuda_visible_devices,
):
    module = _load_process_module_with_stubs(monkeypatch, "process_high_mask_editor_success")
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    editor_script = tmp_path / "editor.py"
    editor_script.write_text("# 测试占位脚本\n", encoding="utf-8")
    manifest_paths = []
    monkeypatch.setattr(
        image_node_module,
        "创建高位Mask编辑会话",
        lambda temp_dir, _image, _blocks: manifest_paths.append(
            os.path.join(temp_dir, "session.json")
        ) or manifest_paths[-1],
    )
    expected_mask = np.full((4, 5), 255, dtype=np.uint8)
    monkeypatch.setattr(
        image_node_module,
        "读取已提交高位Mask",
        lambda manifest, blocks: [expected_mask.copy()],
    )
    popen_calls = []

    class CompletedProcess:
        returncode = 0

        @staticmethod
        def poll():
            return 0

    def fake_popen(args, cwd, env):
        popen_calls.append((args, cwd, env))
        return CompletedProcess()

    monkeypatch.setattr(image_node_module.subprocess, "Popen", fake_popen)
    processor = object.__new__(module.ImageProcessor)
    processor.high_mask_manual_editor_enabled = True
    processor.high_mask_editor_script_path = str(editor_script)
    processor.high_mask_editor_preview_device = preview_device
    processor.high_mask_editor_process_lock = threading.Lock()
    processor.active_high_mask_editor_process = None
    block = {
        "category": "T",
        "mask": np.zeros((4, 5), dtype=np.uint8),
        "crop_box": (0, 0, 5, 4),
        "detection_box": (0.0, 0.0, 5.0, 4.0),
    }

    masks = processor._run_high_mask_manual_editor(
        np.zeros((8, 10, 3), dtype=np.uint8),
        [block],
    )

    assert np.array_equal(masks[0], expected_mask)
    assert popen_calls[0][0] == [sys.executable, str(editor_script)]
    assert popen_calls[0][1] == image_node_module.SRC_DIR
    assert popen_calls[0][2]["CUDA_VISIBLE_DEVICES"] == expected_cuda_visible_devices
    assert (
        popen_calls[0][2][image_node_module.预览设备环境变量]
        == preview_device
    )
    assert popen_calls[0][2][image_node_module.会话环境变量] == manifest_paths[0]
    assert processor.active_high_mask_editor_process is None


def test_high_mask_editor_cancel_rejects_current_detection(monkeypatch, tmp_path):
    module = _load_process_module_with_stubs(monkeypatch, "process_high_mask_editor_cancel")
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    monkeypatch.setenv("DISPLAY", ":0")
    editor_script = tmp_path / "editor.py"
    editor_script.write_text("# 测试占位脚本\n", encoding="utf-8")
    monkeypatch.setattr(
        image_node_module,
        "创建高位Mask编辑会话",
        lambda temp_dir, _image, _blocks: os.path.join(temp_dir, "session.json"),
    )

    class CancelledProcess:
        returncode = image_node_module.编辑取消退出码

        @staticmethod
        def poll():
            return image_node_module.编辑取消退出码

    monkeypatch.setattr(
        image_node_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: CancelledProcess(),
    )
    processor = object.__new__(module.ImageProcessor)
    processor.high_mask_manual_editor_enabled = True
    processor.high_mask_editor_script_path = str(editor_script)
    processor.high_mask_editor_preview_device = "cpu"
    processor.high_mask_editor_process_lock = threading.Lock()
    processor.active_high_mask_editor_process = None

    with pytest.raises(RuntimeError, match="用户取消"):
        processor._run_high_mask_manual_editor(
            np.zeros((8, 10, 3), dtype=np.uint8),
            [{
                "category": "T",
                "mask": np.zeros((4, 5), dtype=np.uint8),
                "crop_box": (0, 0, 5, 4),
                "detection_box": (0.0, 0.0, 5.0, 4.0),
            }],
        )


def test_disabled_high_mask_editor_does_not_create_session_or_subprocess(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_high_mask_editor_disabled")
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    monkeypatch.setattr(
        image_node_module,
        "创建高位Mask编辑会话",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("禁用人工编辑时不应创建临时会话")
        ),
    )
    monkeypatch.setattr(
        image_node_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("禁用人工编辑时不应启动子进程")
        ),
    )
    processor = object.__new__(module.ImageProcessor)
    processor.high_mask_manual_editor_enabled = False
    original_mask = np.zeros((4, 5), dtype=np.uint8)
    original_mask[1:3, 2:4] = 255

    masks = processor._run_high_mask_manual_editor(
        np.zeros((8, 10, 3), dtype=np.uint8),
        [{"mask": original_mask}],
    )

    assert np.array_equal(masks[0], original_mask)
    assert masks[0] is not original_mask


def test_prepare_task_rejects_concurrent_request_without_taking_snapshot(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_prepare_task_concurrent")
    processor = object.__new__(module.ImageProcessor)
    processor.prepare_task_lock = threading.Lock()
    processor.prepare_task_lock.acquire()
    processor.get_image_snapshot_newer_than = lambda _stamp: (_ for _ in ()).throw(
        AssertionError("并发请求不应获取新快照")
    )

    try:
        response = processor.prepare_task(types.SimpleNamespace())
    finally:
        processor.prepare_task_lock.release()

    assert response.success is False
    assert response.task_count == 0
    assert "并发" in response.message


def test_shutdown_terminates_active_high_mask_editor_process(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_high_mask_editor_shutdown")

    class RunningProcess:
        def __init__(self):
            self.running = True
            self.terminate_called = False
            self.wait_calls = []

        def poll(self):
            return None if self.running else 0

        def terminate(self):
            self.terminate_called = True
            self.running = False

        def wait(self, timeout):
            self.wait_calls.append(timeout)
            return 0

    process = RunningProcess()
    processor = object.__new__(module.ImageProcessor)
    processor.high_mask_editor_process_lock = threading.Lock()
    processor.active_high_mask_editor_process = process
    recorder_closed = []
    processor.close_debug_video_recorders = lambda: recorder_closed.append(True)

    processor.close_runtime_resources()

    assert process.terminate_called is True
    assert process.wait_calls == [2.0]
    assert recorder_closed == [True]


def test_calibration_manual_editor_failure_stops_before_depth_and_planning(monkeypatch):
    module = _load_process_module_with_stubs(
        monkeypatch,
        "process_high_mask_failure_order_calibration",
    )
    processor = object.__new__(module.ImageProcessor)
    processor.task_targets = [object()]
    processor.board_grid_points = object()
    processor.board_grid_image_shape = (1, 1)
    processor.board_grid_image = np.zeros((1, 1, 3), dtype=np.uint8)
    processor.calibration_mode = True
    processor.get_image_snapshot_newer_than = lambda _stamp: np.zeros(
        (20, 30, 3),
        dtype=np.uint8,
    )
    processor._detect_board_for_task = lambda _image: None
    processor._detect_blocks_raw = lambda _image: (_ for _ in ()).throw(
        RuntimeError("用户取消了本轮高位 Mask 编辑")
    )
    processor.high_tcp_localizer = types.SimpleNamespace(
        locate_block=lambda _pixel: (_ for _ in ()).throw(
            AssertionError("编辑失败后不应执行像素转 TCP")
        )
    )
    processor._build_calibration_targets = lambda *_args: (_ for _ in ()).throw(
        AssertionError("编辑失败后不应执行深度采样")
    )
    processor._load_layout_for_request = lambda *_args: (_ for _ in ()).throw(
        AssertionError("编辑失败后不应进入任务规划")
    )

    response = processor._prepare_task_locked(types.SimpleNamespace())

    assert response.success is False
    assert response.task_count == 0
    assert "用户取消" in response.message
    assert processor.task_targets == []


def test_block_raw_detection_applies_manual_masks_before_return(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_high_mask_editor_order")
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    original_mask = np.zeros((10, 12), dtype=np.uint8)
    edited_mask = np.full((10, 12), 255, dtype=np.uint8)
    raw_block = {
        "category": "T",
        "score": 0.9,
        "px": 1.0,
        "py": 2.0,
        "theta": 3.0,
        "mask": original_mask,
        "crop_box": (0, 0, 12, 10),
        "detection_box": (1.0, 1.0, 11.0, 9.0),
    }
    monkeypatch.setattr(image_node_module, "load_template_geometry", lambda _name: {})
    monkeypatch.setattr(
        image_node_module,
        "detect_blocks_in_image",
        lambda *_args, **_kwargs: ([raw_block], image.copy()),
    )
    call_order = []

    def fake_rematch(_image, blocks, masks, **_kwargs):
        call_order.append("父进程重匹配")
        assert np.array_equal(masks[0], edited_mask)
        updated = dict(blocks[0], px=50.0, py=60.0, theta=70.0, mask=masks[0])
        return [updated], image.copy()

    monkeypatch.setattr(image_node_module, "rematch_blocks_from_masks", fake_rematch)
    monkeypatch.setattr(image_node_module, "save_image_to_path", lambda *_args: True)
    processor = object.__new__(module.ImageProcessor)
    processor.model = object()
    processor.save_top_surface_mask_vis = False
    processor.high_template_match_debug_path = ""
    processor.high_mask_manual_editor_enabled = True
    processor._run_high_mask_manual_editor = lambda _image, _blocks: (
        call_order.append("人工编辑") or [edited_mask]
    )

    blocks, counts = processor._detect_blocks_raw(image)

    assert call_order == ["人工编辑", "父进程重匹配"]
    assert blocks == [{"category": "T", "px": 50.0, "py": 60.0, "theta": 70.0}]
    assert sum(counts) == 1


def test_high_safety_check_collects_all_block_and_tray_violations(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_high_safety_batch")
    processor = object.__new__(module.ImageProcessor)
    blocks = [
        {"category": "T", "px": 10.0, "py": 20.0},
        {"category": "square", "px": 30.0, "py": 40.0},
    ]
    layout = [
        {"index": 5, "row": 1.0, "col": 2.0, "category": "T", "angle_deg": 0.0},
    ]
    processor._placement_specs = lambda _layout: [(layout[0], (50.0, 60.0))]

    assessments = {
        ("block", (10.0, 20.0)): types.SimpleNamespace(
            safe=False,
            safety_tcp_xyz=(-525.1, 80.3, 190.2),
            violated_axes=("X",),
        ),
        ("block", (30.0, 40.0)): types.SimpleNamespace(
            safe=False,
            safety_tcp_xyz=(-180.2, 321.8, 189.7),
            violated_axes=("Y",),
        ),
        ("tray", (50.0, 60.0)): types.SimpleNamespace(
            safe=False,
            safety_tcp_xyz=(-530.0, 320.0, 200.0),
            violated_axes=("X", "Y"),
        ),
    }
    processor.high_tcp_localizer = types.SimpleNamespace(
        assess=lambda subject, pixel: assessments[(subject, tuple(pixel))]
    )

    violations = processor._collect_high_tcp_safety_violations(blocks, layout)

    assert violations == [
        "方块 1（T）：预测实际 TCP [-525.100, 80.300, 190.200]，X 越界",
        "方块 2（square）：预测实际 TCP [-180.200, 321.800, 189.700]，Y 越界",
        "托盘目标 6：预测实际 TCP [-530.000, 320.000, 200.000]，X、Y 越界",
    ]


def test_high_safety_check_continues_after_single_prediction_failure(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_high_safety_predict_error")
    processor = object.__new__(module.ImageProcessor)
    blocks = [
        {"category": "T", "px": 10.0, "py": 20.0},
        {"category": "square", "px": 30.0, "py": 40.0},
    ]
    processor._placement_specs = lambda _layout: []

    def assess(_subject, pixel):
        if tuple(pixel) == (10.0, 20.0):
            raise ValueError("模型输入无效")
        return types.SimpleNamespace(
            safe=False,
            safety_tcp_xyz=(-300.0, 320.0, 200.0),
            violated_axes=("Y",),
        )

    processor.high_tcp_localizer = types.SimpleNamespace(assess=assess)

    violations = processor._collect_high_tcp_safety_violations(blocks, [])

    assert violations == [
        "方块 1（T）：TCP 预测失败：模型输入无效",
        "方块 2（square）：预测实际 TCP [-300.000, 320.000, 200.000]，Y 越界",
    ]


def test_formal_precheck_failure_does_not_open_mask_editor(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_high_safety_precheck")
    processor = object.__new__(module.ImageProcessor)
    processor.task_targets = [object()]
    processor.board_grid_points = None
    processor.board_grid_image_shape = None
    processor.board_grid_image = None
    processor.calibration_mode = False
    processor.fresh_image_timeout_sec = 0.5
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    processor.get_image_snapshot_newer_than = lambda _stamp: image
    processor._detect_board_for_task = lambda _image: None
    raw_blocks = [{"category": "T", "px": 10.0, "py": 20.0, "theta": 0.0}]
    processor._detect_blocks_automatic = lambda _image: (raw_blocks, image.copy(), {})
    layout = [{"index": 0, "row": 1.0, "col": 1.0, "category": "T", "angle_deg": 0.0}]
    processor._load_layout_for_request = lambda _request, _counts: (layout, "测试布局")
    processor._placement_specs = lambda _layout: [(layout[0], (50.0, 60.0))]

    def assess(subject, _pixel):
        if subject == "block":
            return types.SimpleNamespace(
                safe=False,
                safety_tcp_xyz=(-525.1, 80.3, 190.2),
                violated_axes=("X",),
            )
        return types.SimpleNamespace(
            safe=True,
            safety_tcp_xyz=(-300.0, 0.0, 200.0),
            violated_axes=(),
        )

    processor.high_tcp_localizer = types.SimpleNamespace(assess=assess)
    processor._edit_and_rematch_blocks = lambda *_args: (_ for _ in ()).throw(
        AssertionError("初检失败时不应打开 Mask 编辑器")
    )

    response = processor._prepare_task_locked(types.SimpleNamespace(advanced=False))

    assert response.success is False
    assert response.task_count == 0
    assert response.message == (
        "高位初步安全检查失败：\n"
        "- 方块 1（T）：预测实际 TCP [-525.100, 80.300, 190.200]，X 越界\n"
        "本轮未打开 Mask 编辑器"
    )
    assert processor.task_targets == []


def test_formal_final_check_uses_edited_center_and_reports_all(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_high_safety_final")
    processor = object.__new__(module.ImageProcessor)
    processor.task_targets = []
    processor.board_grid_points = None
    processor.board_grid_image_shape = None
    processor.board_grid_image = None
    processor.calibration_mode = False
    processor.fresh_image_timeout_sec = 0.5
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    processor.get_image_snapshot_newer_than = lambda _stamp: image
    processor._detect_board_for_task = lambda _image: None
    raw_blocks = [{"category": "T", "px": 10.0, "py": 20.0, "theta": 0.0}]
    edited_blocks = [{"category": "T", "px": 70.0, "py": 80.0, "theta": 5.0}]
    processor._detect_blocks_automatic = lambda _image: (raw_blocks, image.copy(), {})
    layout = [{"index": 0, "row": 1.0, "col": 1.0, "category": "T", "angle_deg": 0.0}]
    processor._load_layout_for_request = lambda _request, _counts: (layout, "测试布局")
    processor._placement_specs = lambda _layout: [(layout[0], (50.0, 60.0))]
    edit_calls = []
    processor._edit_and_rematch_blocks = lambda *_args: (
        edit_calls.append(True) or (edited_blocks, image.copy())
    )
    processor._save_high_block_debug_images = lambda *_args: None

    def assess(subject, pixel):
        if subject == "block" and tuple(pixel) == (70.0, 80.0):
            return types.SimpleNamespace(
                safe=False,
                safety_tcp_xyz=(-180.2, 321.8, 189.7),
                violated_axes=("Y",),
            )
        return types.SimpleNamespace(
            safe=True,
            safety_tcp_xyz=(-300.0, 0.0, 200.0),
            violated_axes=(),
        )

    processor.high_tcp_localizer = types.SimpleNamespace(assess=assess)
    processor._build_observed_blocks_for_task = lambda *_args: (_ for _ in ()).throw(
        AssertionError("最终复检失败后不应生成正式目标")
    )

    response = processor._prepare_task_locked(types.SimpleNamespace(advanced=False))

    assert edit_calls == [True]
    assert response.success is False
    assert response.message == (
        "高位最终安全检查失败：\n"
        "- 方块 1（T）：预测实际 TCP [-180.200, 321.800, 189.700]，Y 越界\n"
        "请重新识别或重新编辑 Mask"
    )


def test_tray_target_uses_independent_tcp_calibration_and_predicted_z(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_tray_tcp_pipeline")
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    monkeypatch.setattr(
        image_node_module,
        "interpolate_grid_point",
        lambda *_args: (700.0, 300.0),
    )
    processor = object.__new__(module.ImageProcessor)
    processor.board_grid_points = object()
    received_pixels = []
    processor.high_tcp_localizer = types.SimpleNamespace(
        locate_tray=lambda pixel: received_pixels.append(pixel)
        or [-280.0, 40.0, 190.5, -180.0, 0.0, 90.0]
    )
    layout = [{
        "index": 0,
        "row": 1.0,
        "col": 2.0,
        "angle_deg": 0.0,
        "category": "T",
        "cells": ((1, 1), (2, 1), (2, 2), (3, 1)),
    }]

    targets = processor._build_placement_targets(layout, (720, 1280))

    assert received_pixels == [(700.0, 300.0)]
    assert targets[0].observation_pose == (-280.0, 40.0, 190.5, -180.0, 0.0, 90.0)
    assert targets[0].high_world_position_valid is False
    assert targets[0].rough_localization_source == "tcp_calibration"


def test_calibration_targets_use_block_depth_z_and_ignore_tray_depth_z(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_depth_xyz_targets")
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    monkeypatch.setattr(
        image_node_module,
        "interpolate_grid_point",
        lambda *_args: (700.0, 300.0),
    )
    processor = object.__new__(module.ImageProcessor)
    processor.board_grid_points = object()
    processor.depth_rough_localizer = image_node_module.DepthRoughLocalizer(
        [-250.0, 0.0, 380.0, 0.0, 0.0, 0.0],
        np.eye(4),
    )
    processor.block_observation_height_mm = 192.0
    processor.pick_surface_offset_mm = 162.0
    processor.minimum_tcp_z_mm = 165.0
    processor.safe_x_range_mm = (-444.224, -148.17)
    processor.safe_y_range_mm = (-263.279, 315.925)
    processor.block_depth_max_mad_mm = 1.0
    processor.block_plane_max_rmse_mm = 1.0
    processor.tray_tcp_below_block_observation_mm = 7.0
    blocks = [
        {"category": "T", "px": 100.0, "py": 100.0, "theta": 0.0},
        {"category": "T", "px": 200.0, "py": 100.0, "theta": 0.0},
        {"category": "T", "px": 100.0, "py": 200.0, "theta": 0.0},
    ]
    layout = [{
        "index": 0,
        "row": 1.0,
        "col": 1.0,
        "angle_deg": 0.0,
        "category": "T",
    }]
    block_world = [
        [-300.0, 0.0, 8.0],
        [-250.0, 50.0, 9.5],
        [-200.0, -20.0, 8.6],
    ]

    def build_samples(tray_depth_z):
        worlds = [*block_world, [-225.0, 30.0, tray_depth_z]]
        return [
            {
                "pixel": (float(index), float(index)),
                "world": np.asarray(world, dtype=float),
                "valid_count": 15,
                "depth_median_mm": 500.0 + index,
                "depth_mad_mm": 0.2 if index < 3 else 20.0,
            }
            for index, world in enumerate(worlds)
        ]

    processor._query_stable_world_points = lambda _pixels: build_samples(1.0)
    observed_a, targets_a = processor._build_calibration_targets(
        blocks,
        layout,
        (720, 1280),
    )
    processor._query_stable_world_points = lambda _pixels: build_samples(999.0)
    observed_b, targets_b = processor._build_calibration_targets(
        blocks,
        layout,
        (720, 1280),
    )

    assert [item.observation_pose[2] for item in observed_a] == pytest.approx(
        [200.0, 201.5, 200.6]
    )
    assert [item.pick_surface_z_mm for item in observed_a] == pytest.approx([8.0, 9.5, 8.6])
    assert [item.observation_pose for item in observed_a] == [
        item.observation_pose for item in observed_b
    ]
    assert targets_a[0].observation_pose == targets_b[0].observation_pose
    assert targets_a[0].observation_pose[2] == pytest.approx(194.35)
    assert targets_a[0].depth_mad_mm == pytest.approx(20.0)
    assert targets_a[0].rough_localization_source == "stable_depth_xy_block_plane_z"


def _make_depth_limit_processor(image_node_module):
    """构造仅包含深度粗定位限位参数的轻量图像节点。"""
    processor = object.__new__(image_node_module.ImageProcessor)
    processor.depth_rough_localizer = image_node_module.DepthRoughLocalizer(
        [-250.0, 0.0, 380.0, 0.0, 0.0, 0.0],
        np.eye(4),
    )
    processor.block_observation_height_mm = 192.0
    processor.pick_surface_offset_mm = 162.0
    processor.minimum_tcp_z_mm = 165.0
    processor.safe_x_range_mm = (-520.224, -148.17)
    processor.safe_y_range_mm = (-263.279, 315.925)
    processor.block_depth_max_mad_mm = 1.0
    processor.block_plane_max_rmse_mm = 1.0
    processor.tray_tcp_below_block_observation_mm = 7.0
    return processor


def _depth_sample(world, pixel=(0.0, 0.0)):
    return {
        "pixel": tuple(float(value) for value in pixel),
        "world": np.asarray(world, dtype=float),
        "valid_count": 15,
        "depth_median_mm": 500.0,
        "depth_mad_mm": 0.2,
    }


@pytest.mark.parametrize(
    "tcp_xyz",
    [
        [-142.386, 0.0, 200.0],
        [-300.0, 320.0, 200.0],
        [-300.0, 0.0, 164.9],
    ],
)
def test_calibration_pose_limit_reports_full_xyz_pixel_and_bounds(monkeypatch, tcp_xyz):
    module = _load_process_module_with_stubs(monkeypatch, "process_full_depth_limit_error")
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    processor = _make_depth_limit_processor(image_node_module)
    label = "方块 square（block）高位像素 (372.0, 592.0) 深度粗定位"
    pose = [*tcp_xyz, -180.0, 0.0, 90.0]

    with pytest.raises(ValueError) as error:
        processor._validate_calibration_pose(pose, label)

    message = str(error.value)
    assert label in message
    assert f"TCP XYZ {[float(value) for value in tcp_xyz]}" in message
    assert "最小值 [-520.224, -263.279, 165.0]" in message
    assert "最大值 [-148.17, 315.925, None]" in message


def test_block_depth_limit_error_identifies_detected_high_pixel(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_block_depth_limit_pixel")
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    processor = _make_depth_limit_processor(image_node_module)
    processor._placement_specs = lambda _layout: []
    processor._query_stable_world_points = lambda _pixels: [
        _depth_sample([-142.386, 0.0, 8.0], pixel=(372.0, 592.0))
    ]
    block = {"category": "square", "px": 372.0, "py": 592.0, "theta": 0.0}

    with pytest.raises(ValueError) as error:
        processor._build_calibration_targets([block], [], (720, 1280))

    message = str(error.value)
    assert "方块 square（block）高位像素 (372.0, 592.0)" in message
    assert "深度粗定位 TCP XYZ [-142.386, 0.0, 200.0]" in message
    assert "超出安全范围" in message


def test_tray_depth_limit_error_identifies_grid_high_pixel(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_tray_depth_limit_pixel")
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    processor = _make_depth_limit_processor(image_node_module)
    item = {
        "index": 7,
        "row": 1.0,
        "col": 1.0,
        "angle_deg": 0.0,
        "category": "T",
    }
    processor._placement_specs = lambda _layout: [(item, (700.0, 300.0))]
    worlds = [
        [-300.0, 0.0, 8.0],
        [-250.0, 0.0, 8.0],
        [-300.0, 50.0, 8.0],
        [-142.386, 0.0, 5.0],
    ]
    processor._query_stable_world_points = lambda _pixels: [
        _depth_sample(world, pixel=(index, index))
        for index, world in enumerate(worlds)
    ]
    blocks = [
        {"category": "T", "px": float(index), "py": float(index), "theta": 0.0}
        for index in range(3)
    ]

    with pytest.raises(ValueError) as error:
        processor._build_calibration_targets(blocks, [item], (720, 1280))

    message = str(error.value)
    assert "托盘目标 7（tray）高位像素 (700.0, 300.0)" in message
    assert "深度粗定位 TCP XYZ [-142.386, 0.0," in message
    assert "超出安全范围" in message


def test_calibration_final_pick_z_error_identifies_detected_high_pixel(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_pick_z_limit_pixel")
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    processor = _make_depth_limit_processor(image_node_module)
    processor._placement_specs = lambda _layout: []
    processor._query_stable_world_points = lambda _pixels: [
        _depth_sample([-300.0, 0.0, 2.0], pixel=(372.0, 592.0))
    ]
    block = {"category": "square", "px": 372.0, "py": 592.0, "theta": 0.0}

    with pytest.raises(ValueError) as error:
        processor._build_calibration_targets([block], [], (720, 1280))

    message = str(error.value)
    assert "方块 square（block）高位像素 (372.0, 592.0)" in message
    assert "最终抓取 TCP Z=164.000 mm 低于安全下限 165.000 mm" in message


def test_stable_depth_query_forwards_capture_timeout(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_depth_request_timeout")
    processor = object.__new__(module.ImageProcessor)
    processor.calibration_depth_frame_count = 15
    processor.calibration_depth_min_valid_frames = 10
    processor.calibration_depth_capture_timeout_sec = 2.0
    calls = []

    def stable_client(*args):
        calls.append(args)
        return types.SimpleNamespace(
            success=True,
            point_valid=[True],
            world_xyz=[-250.0, 20.0, 8.0],
            valid_frame_counts=[15],
            depth_median_mm=[500.0],
            depth_mad_mm=[0.2],
            message="成功",
        )

    processor.stable_world_points_client = stable_client

    samples = processor._query_stable_world_points([(100.2, 200.8)])

    assert calls == [([100], [201], 15, 10, 2.0)]
    assert samples[0]["valid_count"] == 15
    assert samples[0]["world"] == pytest.approx([-250.0, 20.0, 8.0])


def test_calibration_targets_reject_block_depth_mad_before_motion(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_depth_mad_gate")
    processor = object.__new__(module.ImageProcessor)
    processor.block_depth_max_mad_mm = 1.0
    processor._placement_specs = lambda _layout: []
    processor._query_stable_world_points = lambda _pixels: [{
        "pixel": (100.0, 100.0),
        "world": np.asarray([-300.0, 0.0, 8.0]),
        "valid_count": 15,
        "depth_median_mm": 500.0,
        "depth_mad_mm": 1.01,
    }]

    with pytest.raises(RuntimeError, match="MAD=.*超过阈值"):
        processor._build_calibration_targets(
            [{"category": "T", "px": 100.0, "py": 100.0, "theta": 0.0}],
            [],
            (720, 1280),
        )


def test_calibration_targets_reject_block_plane_rmse_before_motion(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_depth_plane_gate")
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    processor = object.__new__(module.ImageProcessor)
    processor.depth_rough_localizer = image_node_module.DepthRoughLocalizer(
        [-250.0, 0.0, 380.0, 0.0, 0.0, 0.0],
        np.eye(4),
    )
    processor.block_observation_height_mm = 192.0
    processor.pick_surface_offset_mm = 162.0
    processor.minimum_tcp_z_mm = 165.0
    processor.safe_x_range_mm = (-444.224, -148.17)
    processor.safe_y_range_mm = (-263.279, 315.925)
    processor.block_depth_max_mad_mm = 1.0
    processor.block_plane_max_rmse_mm = 1.0
    processor._placement_specs = lambda _layout: []
    worlds = [
        [-300.0, 0.0, 8.0],
        [-250.0, 0.0, 8.0],
        [-300.0, 50.0, 8.0],
        [-250.0, 50.0, 20.0],
    ]
    processor._query_stable_world_points = lambda _pixels: [
        {
            "pixel": (float(index), float(index)),
            "world": np.asarray(world),
            "valid_count": 15,
            "depth_median_mm": 500.0,
            "depth_mad_mm": 0.2,
        }
        for index, world in enumerate(worlds)
    ]
    blocks = [
        {"category": "T", "px": float(index), "py": float(index), "theta": 0.0}
        for index in range(4)
    ]

    with pytest.raises(RuntimeError, match="平面 RMSE=.*超过阈值"):
        processor._build_calibration_targets(blocks, [], (720, 1280))


def test_calibrated_height_rejects_low_final_pick_z_during_high_preparation(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_low_calibrated_pick_z")
    processor = object.__new__(module.ImageProcessor)
    processor.block_observation_height_mm = 192.0
    processor.pick_surface_offset_mm = 162.0
    processor.minimum_tcp_z_mm = 165.0

    with pytest.raises(ValueError, match="最终抓取 TCP Z=.*低于安全下限"):
        processor.resolve_pick_surface_height(12.2, 33.8, 193.0)


def _calibration_prep_processor(module):
    """构造标定准备编排测试用的轻量图像节点。"""
    processor = object.__new__(module.ImageProcessor)
    processor.calibration_mode = True
    processor.fresh_image_timeout_sec = 0.5
    processor.task_targets = [object()]
    processor.board_grid_points = None
    processor.board_grid_image_shape = None
    processor.board_grid_image = None
    processor.block_plane_max_rmse_mm = 1.0
    processor.get_image_snapshot_newer_than = lambda _stamp: np.zeros(
        (6, 8, 3),
        dtype=np.uint8,
    )
    return processor


def _observed_block(px, py, z=200.0):
    from image_process_lib.task_planner import ObservedBlock

    return ObservedBlock(
        category="square",
        observation_pose=[px, py, z, -180.0, 0.0, 90.0],
        detected_angle_deg=0.0,
        pick_surface_z_mm=8.0,
        pick_surface_z_valid=True,
        high_detected_pixel_xy=(px, py),
        high_image_center_xy=(640.0, 360.0),
    )


def _depth_sample_list(count):
    return [
        {
            "pixel": (0.0, 0.0),
            "world": np.asarray([-300.0, 0.0, 8.0]),
            "valid_count": 15,
            "depth_median_mm": 500.0,
            "depth_mad_mm": 0.2,
        }
        for _ in range(count)
    ]


def test_calibration_prep_without_tray_keeps_all_block_targets(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_calibration_prep_no_tray")
    processor = _calibration_prep_processor(module)
    blocks = [
        {"category": "square", "px": float(100 + index * 10), "py": 200.0, "theta": 0.0}
        for index in range(5)
    ]
    processor._detect_blocks_raw = lambda _image: (blocks, [5, 0, 0, 0, 0, 0, 0])
    processor._detect_board_for_task = lambda _image: (_ for _ in ()).throw(
        RuntimeError("托盘格点不足")
    )
    queried_pixels = []
    processor._query_stable_world_points = lambda pixels: (
        queried_pixels.append(list(pixels)) or _depth_sample_list(len(pixels))
    )
    observed = [_observed_block(100.0 + index * 10, 200.0) for index in range(5)]
    processor._build_calibration_block_observed = lambda _blocks, _samples, _shape: observed
    tray_calls = []
    processor._build_calibration_tray_targets = lambda *_args: tray_calls.append(True)

    response = processor._prepare_task_locked(types.SimpleNamespace())

    assert response.success is True
    assert response.task_count == 5
    assert response.block_count == 5
    assert response.tray_count == 0
    assert "未识别到托盘" in response.message
    assert tray_calls == []
    assert len(queried_pixels) == 1
    assert len(queried_pixels[0]) == 5
    assert [target.target_type for target in processor.task_targets] == ["block"] * 5


def test_calibration_prep_with_tray_appends_34_tray_targets(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_calibration_prep_with_tray")
    from image_process_lib.task_planner import PlacementTarget

    image_node_module = sys.modules[module.ImageProcessor.__module__]
    monkeypatch.setattr(
        image_node_module,
        "interpolate_grid_point",
        lambda _grid_points, row, col: (700.0 + row, 300.0 + col),
    )
    processor = _calibration_prep_processor(module)
    blocks = [
        {"category": "square", "px": 100.0, "py": 100.0, "theta": 0.0},
        {"category": "square", "px": 200.0, "py": 100.0, "theta": 0.0},
        {"category": "square", "px": 100.0, "py": 200.0, "theta": 0.0},
    ]
    processor._detect_blocks_raw = lambda _image: (blocks, [3, 0, 0, 0, 0, 0, 0])
    processor._detect_board_for_task = lambda _image: None
    queried_pixels = []
    processor._query_stable_world_points = lambda pixels: (
        queried_pixels.append(list(pixels)) or _depth_sample_list(len(pixels))
    )
    observed = [
        _observed_block(-300.0, 0.0),
        _observed_block(-250.0, 50.0),
        _observed_block(-200.0, -20.0),
    ]
    processor._build_calibration_block_observed = lambda _blocks, _samples, _shape: observed
    tray_placements = [
        PlacementTarget(
            index,
            1.0 + index / 10.0,
            1.0,
            0.0,
            "",
            [-250.0, 20.0, 193.0, -180.0, 0.0, 90.0],
        )
        for index in range(34)
    ]
    tray_received = {}
    processor._build_calibration_tray_targets = lambda specs, samples, plane, shape: (
        tray_received.update({"specs": list(specs), "plane": plane}) or tray_placements
    )

    response = processor._prepare_task_locked(types.SimpleNamespace())

    assert response.success is True
    assert response.task_count == 37
    assert response.block_count == 3
    assert response.tray_count == 34
    assert "行列：" in response.message
    assert len(queried_pixels) == 1
    assert len(queried_pixels[0]) == 3 + 34
    assert tray_received["plane"] is not None
    assert len(tray_received["specs"]) == 34
    target_types = [target.target_type for target in processor.task_targets]
    assert target_types == ["block"] * 3 + ["tray"] * 34
    assert processor.task_targets[3].row == pytest.approx(1.0)
    assert processor.task_targets[3].col == 1.0
    assert processor.task_targets[36].row == pytest.approx(4.3)


def test_calibration_prep_with_tray_rejects_fewer_than_three_blocks(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_calibration_prep_few_blocks")
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    monkeypatch.setattr(
        image_node_module,
        "interpolate_grid_point",
        lambda _grid_points, row, col: (700.0 + row, 300.0 + col),
    )
    processor = _calibration_prep_processor(module)
    blocks = [
        {"category": "square", "px": 100.0, "py": 100.0, "theta": 0.0},
        {"category": "square", "px": 200.0, "py": 100.0, "theta": 0.0},
    ]
    processor._detect_blocks_raw = lambda _image: (blocks, [2, 0, 0, 0, 0, 0, 0])
    processor._detect_board_for_task = lambda _image: None
    processor._query_stable_world_points = lambda pixels: _depth_sample_list(len(pixels))
    observed = [_observed_block(-300.0, 0.0), _observed_block(-250.0, 50.0)]
    processor._build_calibration_block_observed = lambda _blocks, _samples, _shape: observed

    response = processor._prepare_task_locked(types.SimpleNamespace())

    assert response.success is False
    assert "至少需要 3 个不共线方块" in response.message
    assert processor.task_targets == []


def test_calibration_prep_with_tray_rejects_collinear_blocks(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_calibration_prep_collinear")
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    monkeypatch.setattr(
        image_node_module,
        "interpolate_grid_point",
        lambda _grid_points, row, col: (700.0 + row, 300.0 + col),
    )
    processor = _calibration_prep_processor(module)
    blocks = [
        {"category": "square", "px": 100.0, "py": 100.0, "theta": 0.0},
        {"category": "square", "px": 200.0, "py": 100.0, "theta": 0.0},
        {"category": "square", "px": 300.0, "py": 100.0, "theta": 0.0},
    ]
    processor._detect_blocks_raw = lambda _image: (blocks, [3, 0, 0, 0, 0, 0, 0])
    processor._detect_board_for_task = lambda _image: None
    processor._query_stable_world_points = lambda pixels: _depth_sample_list(len(pixels))
    observed = [
        _observed_block(-300.0, 0.0),
        _observed_block(-250.0, 0.0),
        _observed_block(-200.0, 0.0),
    ]
    processor._build_calibration_block_observed = lambda _blocks, _samples, _shape: observed

    response = processor._prepare_task_locked(types.SimpleNamespace())

    assert response.success is False
    assert "不共线" in response.message
    assert processor.task_targets == []


def test_calibration_prep_with_tray_rejects_high_plane_rmse(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_calibration_prep_rmse")
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    monkeypatch.setattr(
        image_node_module,
        "interpolate_grid_point",
        lambda _grid_points, row, col: (700.0 + row, 300.0 + col),
    )
    processor = _calibration_prep_processor(module)
    blocks = [
        {"category": "square", "px": 100.0, "py": 100.0, "theta": 0.0},
        {"category": "square", "px": 200.0, "py": 100.0, "theta": 0.0},
        {"category": "square", "px": 100.0, "py": 200.0, "theta": 0.0},
        {"category": "square", "px": 200.0, "py": 200.0, "theta": 0.0},
    ]
    processor._detect_blocks_raw = lambda _image: (blocks, [4, 0, 0, 0, 0, 0, 0])
    processor._detect_board_for_task = lambda _image: None
    processor._query_stable_world_points = lambda pixels: _depth_sample_list(len(pixels))
    observed = [
        _observed_block(-300.0, 0.0),
        _observed_block(-250.0, 50.0),
        _observed_block(-200.0, -20.0),
        _observed_block(-250.0, -40.0, z=210.0),
    ]
    processor._build_calibration_block_observed = lambda _blocks, _samples, _shape: observed

    response = processor._prepare_task_locked(types.SimpleNamespace())

    assert response.success is False
    assert "平面 RMSE" in response.message
    assert processor.task_targets == []


def test_task_target_service_returns_calibration_target_type(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_calibration_target_type")
    processor = object.__new__(module.ImageProcessor)

    def make_target(target_type):
        return types.SimpleNamespace(
            pick_observation_pose=[0.0] * 6,
            place_observation_pose=[0.0] * 6,
            row=0.0,
            col=0.0,
            category="square",
            detected_angle_deg=0.0,
            rotation_delta_deg=0.0,
            pick_surface_z_mm=0.0,
            pick_surface_z_valid=True,
            pick_high_detected_pixel_xy=(0.0, 0.0),
            pick_high_depth_sample_pixel_xy=(0.0, 0.0),
            pick_high_image_center_xy=(0.0, 0.0),
            pick_high_world_position=(0.0, 0.0, 0.0),
            pick_high_world_position_valid=False,
            pick_rough_localization_source="",
            pick_depth_valid_frame_count=0,
            pick_depth_median_mm=0.0,
            pick_depth_mad_mm=0.0,
            pick_calibration_target_tcp_z_mm=0.0,
            place_high_detected_pixel_xy=(0.0, 0.0),
            place_high_depth_sample_pixel_xy=(0.0, 0.0),
            place_high_image_center_xy=(0.0, 0.0),
            place_high_world_position=(0.0, 0.0, 0.0),
            place_high_world_position_valid=False,
            place_rough_localization_source="",
            place_depth_valid_frame_count=0,
            place_depth_median_mm=0.0,
            place_depth_mad_mm=0.0,
            place_calibration_target_tcp_z_mm=0.0,
            target_type=target_type,
        )

    processor.task_targets = [make_target("block"), make_target("tray")]
    assert processor.get_task_target(types.SimpleNamespace(index=0)).target_type == "block"
    assert processor.get_task_target(types.SimpleNamespace(index=1)).target_type == "tray"

    processor.task_targets = []
    error_response = processor.get_task_target(types.SimpleNamespace(index=0))
    assert error_response.success is False
    assert error_response.target_type == ""


def _make_dynamic_routing_processor(module, mode):
    """构造只验证 shadow/execute/进阶分流的轻量正式节点。"""
    processor = object.__new__(module.ImageProcessor)
    processor.calibration_mode = False
    processor.fresh_image_timeout_sec = 0.5
    processor.dynamic_board_selection_mode = mode
    processor.dynamic_board_failure_prompt_timeout_sec = 60.0
    processor.task_targets = []
    processor.board_grid_points = None
    processor.board_grid_image_shape = None
    processor.board_grid_image = None
    processor.get_image_snapshot_newer_than = lambda _stamp: np.zeros(
        (20, 30, 3), dtype=np.uint8
    )
    processor._detect_board_for_task = lambda _image: None
    raw_blocks = [{"category": "T", "px": 1.0, "py": 2.0, "theta": 3.0}]
    processor._detect_blocks_automatic = lambda _image: (
        raw_blocks,
        np.zeros((20, 30, 3), dtype=np.uint8),
        {},
    )
    processor._load_layout_for_request = lambda *_args: ([{
        "index": 0,
        "row": 1.0,
        "col": 1.0,
        "category": "T",
        "angle_deg": 0.0,
    }], "固定测试布局")
    processor._validate_high_task_safety = lambda *_args: None
    processor._edit_and_rematch_blocks = lambda *_args: (
        raw_blocks,
        np.zeros((20, 30, 3), dtype=np.uint8),
    )
    processor._save_high_block_debug_images = lambda *_args: None
    processor._build_observed_blocks_for_task = lambda *_args: [object()]
    processor._build_placement_targets = lambda *_args: ["固定目标"]
    processor._plan_formal_tasks = lambda *_args: (["固定任务"], "固定规划")
    processor.dynamic_reports = []
    processor._save_dynamic_board_report = lambda *_args, **kwargs: (
        processor.dynamic_reports.append(kwargs)
    )
    decision = types.SimpleNamespace(
        tasks=("动态任务",),
        board_id="v5_g00001_l00",
        decision_fingerprint="a" * 64,
    )
    processor._run_dynamic_board_selection = lambda *_args: {"decision": decision}
    return processor


def test_dynamic_shadow_keeps_fixed_yaml_tasks_and_records_dynamic_result(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_dynamic_shadow_route")
    processor = _make_dynamic_routing_processor(module, "shadow")

    response = processor._prepare_task_locked(types.SimpleNamespace(advanced=False))

    assert response.success is True
    assert processor.task_targets == ["固定任务"]
    assert "V5 shadow 选出" in response.message
    assert processor.dynamic_reports[0]["outcome"] == "shadow成功，执行固定YAML"


def test_dynamic_execute_returns_confirmed_dynamic_tasks_without_building_fixed_targets(
    monkeypatch,
):
    module = _load_process_module_with_stubs(monkeypatch, "process_dynamic_execute_route")
    processor = _make_dynamic_routing_processor(module, "execute")
    processor._build_placement_targets = lambda *_args: (_ for _ in ()).throw(
        AssertionError("execute 成功后不应构造固定目标")
    )
    processor._plan_formal_tasks = lambda *_args: (_ for _ in ()).throw(
        AssertionError("execute 成功后不应进入固定规划")
    )

    response = processor._prepare_task_locked(types.SimpleNamespace(advanced=False))

    assert response.success is True
    assert processor.task_targets == ["动态任务"]
    assert "V5 动态盘面" in response.message
    assert processor.dynamic_reports[0]["outcome"] == "execute成功，执行动态盘面"


def test_advanced_task_bypasses_dynamic_selection_even_in_execute_mode(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_dynamic_advanced_bypass")
    processor = _make_dynamic_routing_processor(module, "execute")
    processor._run_dynamic_board_selection = lambda *_args: (_ for _ in ()).throw(
        AssertionError("进阶任务不应调用动态盘面选择")
    )

    response = processor._prepare_task_locked(types.SimpleNamespace(advanced=True))

    assert response.success is True
    assert processor.task_targets == ["固定任务"]
    assert processor.dynamic_reports == []


@pytest.mark.parametrize(
    ("failure_choice", "expected_success", "expected_tasks"),
    [
        ("fixed_yaml", True, ["固定任务"]),
        ("stop", False, []),
    ],
)
def test_dynamic_execute_failure_terminal_choice_routes_safely(
    monkeypatch,
    failure_choice,
    expected_success,
    expected_tasks,
):
    module = _load_process_module_with_stubs(
        monkeypatch,
        f"process_dynamic_failure_{failure_choice}",
    )
    image_node_module = sys.modules[module.ImageProcessor.__module__]
    processor = _make_dynamic_routing_processor(module, "execute")
    processor._run_dynamic_board_selection = lambda *_args: (_ for _ in ()).throw(
        image_node_module.DynamicBoardSelectionError("注入的动态失败")
    )
    monkeypatch.setattr(
        image_node_module,
        "prompt_dynamic_selection_failure",
        lambda *_args, **_kwargs: failure_choice,
    )

    response = processor._prepare_task_locked(types.SimpleNamespace(advanced=False))

    assert response.success is expected_success
    assert processor.task_targets == expected_tasks
    if failure_choice == "fixed_yaml":
        assert processor.dynamic_reports[-1]["human_failure_choice"] == "fixed_yaml"
    else:
        assert "已停止" in response.message
        assert processor.dynamic_reports[-1]["human_failure_choice"] == "stop"
