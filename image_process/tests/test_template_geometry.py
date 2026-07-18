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
    _build_rectified_roi_transform,
    _transform_angle,
    _transform_point,
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
    create_compact_rotation_kernels,
    create_rotation_kernels,
    get_template_rect_size,
)
from image_process_lib.template_match.template_match import get_rect
from image_process_lib.rough_localization import RoughLocalizer


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


@pytest.mark.parametrize("category", BLOCK_CATEGORY_NAMES)
def test_compact_rotation_kernels_keep_all_template_pixels(category):
    full_kernels, full_size, full_angles = create_rotation_kernels(
        37,
        5,
        category,
        angle_values=[-10, 0, 10],
        device="cpu",
    )
    compact_kernels, compact_hw, compact_angles = create_compact_rotation_kernels(
        37,
        5,
        category,
        angle_values=[-10, 0, 10],
        device="cpu",
    )
    assert compact_angles == full_angles
    assert compact_kernels.shape == (3, 1, compact_hw[0], compact_hw[1])
    assert compact_hw[0] * compact_hw[1] < full_size * full_size
    np.testing.assert_array_equal(
        compact_kernels.sum(dim=(1, 2, 3)).numpy(),
        full_kernels.sum(dim=(1, 2, 3)).numpy(),
    )


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
        roi_expand_px=30,
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
        roi_expand_px=30,
        min_foreground_area=50,
        debug_enabled=False,
    )

    assert result["found"] is True
    assert abs(result["px"] - target_center[0]) <= 2.0
    assert abs(result["py"] - target_center[1]) <= 2.0
    assert result["debug_image"] is None
    assert result["debug_panel"] is None


def test_rectified_low_prior_roi_recovers_center_angle_and_uses_template_cache():
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

    block_servo_detector_module._LOW_PREPARED_TEMPLATE_CACHE.clear()
    common_kwargs = {
        "template_geometry": {"block_px": block_px, "connector_px": connector_px},
        "category": category,
        "high_theta_deg": high_theta,
        "angle_window": 3,
        "angle_step": 1,
        "roi_expand_px": 30,
        "min_foreground_area": 50,
        "debug_enabled": False,
        "timing_enabled": True,
        "rectified_roi_enabled": True,
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
    assert first_timing["匹配模式"] == "转正紧凑核"
    assert first_timing["模板缓存"] == "未命中"
    assert second_timing["模板缓存"] == "命中"
    assert first_timing["阶段毫秒"]["ROI矫正"] is not None
    assert first_timing["匹配图尺寸"][0] < first_timing["ROI尺寸"][0]
    assert first_timing["匹配图尺寸"][1] < first_timing["ROI尺寸"][1]


def test_rectified_roi_inverse_transform_restores_center_and_angle():
    image_center = (160.0, 120.0)
    matrix, inverse_matrix, roi_size = _build_rectified_roi_transform(
        image_center,
        (120.0, 80.0),
        -18.0,
    )
    restored_center = _transform_point(((roi_size[0] - 1) / 2.0, (roi_size[1] - 1) / 2.0), inverse_matrix)
    np.testing.assert_allclose(restored_center, image_center, atol=1e-4)
    assert abs(_transform_angle(0.0, inverse_matrix) + 18.0) <= 1e-4
    assert matrix.shape == (2, 3)


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
        roi_expand_px=30,
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
    for stage_name in ("模板生成", "张量准备", "卷积选优", "匹配收尾"):
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
    camera.srv.PixelToWorld = type("PixelToWorld", (ServiceStub,), {})
    camera.srv.PixelToWorldRequest = type("PixelToWorldRequest", (ServiceStub,), {})

    process_path = os.path.join(PACKAGE_DIR, "scripts", "process.py")
    spec = importlib.util.spec_from_file_location(module_name, process_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_process_module_imports_with_ros_stubs(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_import_smoke")
    assert hasattr(module, "ImageProcessor")
    assert callable(module.main)


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


def test_prepare_task_clears_previous_result_when_image_is_missing(monkeypatch):
    module = _load_process_module_with_stubs(monkeypatch, "process_prepare_no_image")
    processor = object.__new__(module.ImageProcessor)
    processor.task_targets = [object()]
    processor.board_grid_points = object()
    processor.board_grid_image_shape = (1, 1)
    processor.board_grid_image = object()
    processor.get_image_snapshot = lambda: None

    response = processor.prepare_task(types.SimpleNamespace(advanced=False, place_order=[]))

    assert response.success is False
    assert processor.task_targets == []
    assert processor.board_grid_points is None


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
    for name in ("MoveArm", "MoveArmRequest", "RotateTool", "RotateToolRequest", "SetSuction", "SetSuctionRequest"):
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


def test_depth_first_servo_pose_uses_individual_height_offsets_when_depth_valid(monkeypatch):
    localizer = RoughLocalizer(
        shooting_pose=[0, 0, 0, 0, 0, 0],
        wrist_to_camera_mm=np.array([
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
        pixel_to_world_client=lambda _x, _y: types.SimpleNamespace(
            success=True, world_position=[100.0, 200.0, 50.0], message="成功"
        ),
        x_mm_per_pixel=0.5,
        y_mm_per_pixel=0.5,
        fallback_enabled=True,
        warning_func=lambda _message: None,
    )

    block_pose, source, world_position = localizer.locate(
        12.2, 33.8, (720, 1280, 3), "方块测试", 180.0
    )
    board_pose, board_source, board_world_position = localizer.locate(
        12.2, 33.8, (720, 1280, 3), "托盘测试", 210.0
    )

    assert source == "depth"
    assert world_position == [100.0, 200.0, 50.0]
    assert block_pose == [90.0, 180.0, 230.0, 0.0, 0.0, 0.0]
    assert board_source == "depth"
    assert board_world_position == [100.0, 200.0, 50.0]
    assert board_pose == [90.0, 180.0, 260.0, 0.0, 0.0, 0.0]


def test_depth_first_servo_pose_falls_back_when_depth_invalid(monkeypatch):
    warnings = []
    localizer = RoughLocalizer(
        shooting_pose=[100.0, 200.0, 0.0, 0.0, 0.0, 0.0],
        wrist_to_camera_mm=np.eye(4),
        pixel_to_world_client=lambda _x, _y: types.SimpleNamespace(
            success=False, world_position=[0.0, 0.0, 0.0], message="无效深度"
        ),
        x_mm_per_pixel=0.5,
        y_mm_per_pixel=0.25,
        fallback_enabled=True,
        warning_func=warnings.append,
    )

    pose, source, world_position = localizer.locate(150.0, 40.0, (100, 200, 3), "测试", 200.0)

    assert source == "fallback"
    assert world_position == []
    assert pose == [97.5, 225.0, 200.0, 0.0, 0.0, 0.0]
    assert warnings and "回退旧粗估" in warnings[0]


def test_depth_failure_does_not_fallback_when_disabled():
    localizer = RoughLocalizer(
        shooting_pose=[0, 0, 0, 0, 0, 0],
        wrist_to_camera_mm=np.eye(4),
        pixel_to_world_client=lambda _x, _y: types.SimpleNamespace(
            success=False, world_position=[0, 0, 0], message="深度无效"
        ),
        x_mm_per_pixel=0.5,
        y_mm_per_pixel=0.5,
        fallback_enabled=False,
        warning_func=lambda _message: None,
    )
    with pytest.raises(ValueError, match="深度无效"):
        localizer.locate(1, 1, (10, 10, 3), "测试", 200.0)
