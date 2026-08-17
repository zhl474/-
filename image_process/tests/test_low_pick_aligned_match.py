"""低位抓取点对齐紧凑核无 Padding 匹配测试（CPU 可运行）。"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from image_process_lib.block_detection import coreect_LL_location
from image_process_lib.block_servo_detector import detect_block_with_high_prior_roi
from image_process_lib.template_config import load_color_segmentation_config
from image_process_lib.template_match.kernels_create import (
    _template_l_pick_in_canvas,
    build_angle_foreground_metadata,
    build_angle_values,
    create_base_shape,
    create_pick_aligned_kernels,
    embed_in_center,
    get_template_rect_size,
    rotate_image,
)

BLOCK_PX = 12
CONNECTOR_PX = 3
CATEGORIES = ["L_yellow", "L_blue", "T", "z_blue", "z_green", "square", "line"]
ANGLE_CENTER = -20.0
ANGLE_WINDOW = 5.0

# 每个类别取候选列表内的实际角度作为合成绘制角度：
# 周期 360 的 L/T 命中 340，周期 180 的 Z/line 命中 160，周期 90 的 square 命中 70。
# 直接用归一化等效角绘制，避免两次离散旋转的栅格化差异影响断言。
DRAW_THETA = {
    "L_yellow": 340.0,
    "L_blue": 340.0,
    "T": 340.0,
    "z_blue": 160.0,
    "z_green": 160.0,
    "square": 70.0,
    "line": 160.0,
}

LOW_KWARGS = {
    "template_geometry": {"block_px": BLOCK_PX, "connector_px": CONNECTOR_PX},
    "high_theta_deg": -ANGLE_CENTER,
    "angle_window": ANGLE_WINDOW,
    "angle_step": 1.0,
    "search_radius_px": 30,
    "fallback_search_radius_px": 50,
    "boundary_guard_px": 3,
    "kernel_safety_margin_px": 2,
    "legacy_fallback_enabled": True,
    "min_foreground_area": 50,
    "debug_enabled": False,
    "timing_enabled": True,
}


def _canvas_template(category, theta, block_px=BLOCK_PX, connector_px=CONNECTOR_PX):
    """生成指定角度的模板画布二值图，返回 (画布, 边长)。"""
    base = create_base_shape(category, block_px, connector_px)
    height, width = base.shape
    length = int(math.ceil(math.sqrt(width ** 2 + height ** 2)))
    if length % 2 == 0:
        length += 1
    rotated = rotate_image(embed_in_center(base, length), theta)
    return (rotated > 0.5).astype(np.uint8), length


def _expected_pick_in_canvas(category, theta):
    """模板抓取点在画布中的坐标；L 形按 coreect_LL_location 相同规则计算。"""
    binary, length = _canvas_template(category, theta)
    c = length // 2
    if category not in ("L_yellow", "L_blue"):
        return (c, c)
    rect_size = get_template_rect_size(category, BLOCK_PX, CONNECTOR_PX)
    rect = ((c, c), rect_size, -float(theta))
    box = np.intp(cv2.boxPoints(rect))
    return coreect_LL_location(box, binary, rect)


def _make_low_image(category, theta, pick_offset=(0, 0), image_size=(240, 320)):
    """把模板画布画到图像上，使抓取点落在 图像中心+pick_offset；返回 (image, 期望抓取点)。"""
    binary, length = _canvas_template(category, theta)
    pick_canvas = _expected_pick_in_canvas(category, theta)
    center_x, center_y = image_size[1] / 2.0, image_size[0] / 2.0
    pick_x = center_x + pick_offset[0]
    pick_y = center_y + pick_offset[1]
    x1 = int(round(pick_x - pick_canvas[0]))
    y1 = int(round(pick_y - pick_canvas[1]))
    rgb = load_color_segmentation_config(category)["rgb"]
    bgr = (rgb[2], rgb[1], rgb[0])
    image = np.full((image_size[0], image_size[1], 3), 255, dtype=np.uint8)
    overlap_x0 = max(0, x1)
    overlap_y0 = max(0, y1)
    overlap_x1 = min(image_size[1], x1 + length)
    overlap_y1 = min(image_size[0], y1 + length)
    if overlap_x1 > overlap_x0 and overlap_y1 > overlap_y0:
        sub = binary[overlap_y0 - y1:overlap_y1 - y1, overlap_x0 - x1:overlap_x1 - x1]
        image[overlap_y0:overlap_y1, overlap_x0:overlap_x1][sub > 0] = bgr
    return image, (pick_x, pick_y)


@pytest.mark.parametrize("category", CATEGORIES)
def test_pick_aligned_kernels_keep_11_templates_with_black_margin(category):
    angles = build_angle_values(category, angle_step=1.0, angle_center=ANGLE_CENTER, angle_window=ANGLE_WINDOW)
    prepared = create_pick_aligned_kernels(
        category, BLOCK_PX, CONNECTOR_PX, angles, device="cpu", safety_margin_px=2,
    )
    assert len(angles) == 11
    assert prepared["angles"] == [float(angle) for angle in angles]
    kernel_h, kernel_w = prepared["kernel_size"]
    assert kernel_h > 0 and kernel_w > 0
    kernels = prepared["kernels"].numpy()
    assert kernels.shape == (11, 1, kernel_h, kernel_w)
    for index in range(11):
        ys, xs = np.nonzero(kernels[index, 0])
        assert xs.min() >= 2 and ys.min() >= 2
        assert xs.max() < kernel_w - 2 and ys.max() < kernel_h - 2
    for anchor in prepared["rect_center_anchors"]:
        assert 0 <= anchor[0] < kernel_w and 0 <= anchor[1] < kernel_h
    for anchor in prepared["pick_anchors"]:
        assert 0 <= anchor[0] < kernel_w and 0 <= anchor[1] < kernel_h
    pick_anchor = prepared["pick_anchor"]
    assert 0 <= pick_anchor[0] < kernel_w and 0 <= pick_anchor[1] < kernel_h


@pytest.mark.parametrize("category,theta", [
    ("L_blue", 7),
    ("L_blue", 43),
    ("L_blue", 79),
    ("L_yellow", 12),
    ("L_yellow", 88),
    ("L_yellow", 153),
])
def test_l_template_pick_matches_mask_based_coreect(category, theta):
    metadata = build_angle_foreground_metadata(category, BLOCK_PX, CONNECTOR_PX)
    item = metadata[float(theta)]
    rect_size = get_template_rect_size(category, BLOCK_PX, CONNECTOR_PX)
    template_pick_canvas = _template_l_pick_in_canvas(
        item["binary"], item["bbox"], item["anchor"], rect_size, float(theta)
    )
    mask_pick = _expected_pick_in_canvas(category, float(theta))
    assert abs(template_pick_canvas[0] - mask_pick[0]) <= 1.0
    assert abs(template_pick_canvas[1] - mask_pick[1]) <= 1.0


@pytest.mark.parametrize("offset_x,offset_y,expected_mode", [
    (0, 0, "快速30像素"),
    (26, 0, "快速30像素"),
    (-26, 0, "快速30像素"),
    (28, 0, "扩大50像素"),
    (-28, 0, "扩大50像素"),
    (48, 0, "扩大50像素"),
    (-48, 0, "扩大50像素"),
])
def test_low_two_level_search_mode_and_pick_accuracy(offset_x, offset_y, expected_mode):
    category = "T"
    image, expected_pick = _make_low_image(category, DRAW_THETA[category], (offset_x, offset_y))
    result = detect_block_with_high_prior_roi(image, category=category, **LOW_KWARGS)
    assert result["found"] is True
    assert result["timing"]["匹配模式"] == expected_mode
    assert abs(result["px"] - expected_pick[0]) <= 1.0
    assert abs(result["py"] - expected_pick[1]) <= 1.0


@pytest.mark.parametrize("category", CATEGORIES)
def test_low_pick_recovery_all_categories(category):
    image, expected_pick = _make_low_image(category, DRAW_THETA[category], (15, -7))
    result = detect_block_with_high_prior_roi(image, category=category, **LOW_KWARGS)
    assert result["found"] is True
    assert abs(result["px"] - expected_pick[0]) <= 1.0
    assert abs(result["py"] - expected_pick[1]) <= 1.0
    detected_angle = (-result["theta"]) % 360.0
    expected_angle = DRAW_THETA[category] % 360.0
    angle_diff = abs(detected_angle - expected_angle)
    assert min(angle_diff, 360.0 - angle_diff) <= 1.0


def test_low_pick_recovery_with_category_runs_overrides():
    """低位链路按类别 overrides：渲染与匹配都用微调线段时，抓取点照常恢复。"""
    category = "L_blue"
    runs = {
        "x_runs": (BLOCK_PX, CONNECTOR_PX, BLOCK_PX, CONNECTOR_PX, BLOCK_PX + 2),
        "y_runs": (BLOCK_PX, CONNECTOR_PX, BLOCK_PX),
    }
    base = create_base_shape(category, BLOCK_PX, CONNECTOR_PX, template_runs=runs)
    height, width = base.shape
    length = int(math.ceil(math.sqrt(width ** 2 + height ** 2)))
    if length % 2 == 0:
        length += 1
    theta = DRAW_THETA[category]
    binary = (rotate_image(embed_in_center(base, length), theta) > 0.5).astype(np.uint8)
    c = length // 2
    rect = ((c, c), (sum(runs["x_runs"]), sum(runs["y_runs"])), -float(theta))
    pick_canvas = coreect_LL_location(np.intp(cv2.boxPoints(rect)), binary, rect)

    center_x, center_y = 160.0, 120.0
    x1 = int(round(center_x - pick_canvas[0]))
    y1 = int(round(center_y - pick_canvas[1]))
    rgb = load_color_segmentation_config(category)["rgb"]
    bgr = (rgb[2], rgb[1], rgb[0])
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    image[y1:y1 + length, x1:x1 + length][binary > 0] = bgr

    kwargs = dict(LOW_KWARGS)
    kwargs["template_geometry"] = {
        "block_px": BLOCK_PX,
        "connector_px": CONNECTOR_PX,
        "runs": {category: runs},
    }
    result = detect_block_with_high_prior_roi(image, category=category, **kwargs)
    assert result["found"] is True
    assert abs(result["px"] - center_x) <= 1.0
    assert abs(result["py"] - center_y) <= 1.0


@pytest.mark.parametrize("category", ["L_yellow", "L_blue"])
def test_l_shape_pick_at_center_stays_inside_roi(category):
    image, expected_pick = _make_low_image(category, DRAW_THETA[category], (0, 0))
    result = detect_block_with_high_prior_roi(image, category=category, **LOW_KWARGS)
    assert result["found"] is True
    assert result["timing"]["匹配模式"] == "快速30像素"
    assert abs(result["px"] - expected_pick[0]) <= 1.0
    assert abs(result["py"] - expected_pick[1]) <= 1.0


def test_low_level1_conv_output_is_61x61():
    category = "T"
    image, _ = _make_low_image(category, DRAW_THETA[category], (0, 0))
    result = detect_block_with_high_prior_roi(image, category=category, **LOW_KWARGS)
    assert result["found"] is True
    assert result["timing"]["匹配模式"] == "快速30像素"
    assert result["match_debug"]["conv_output_size"] == (61, 61)


def test_low_clipped_roi_near_image_edge_uses_legacy_fallback():
    category = "T"
    # 使用 100x100 小图：相机中心 (50,50)，ROI 左边界 x1 = 50-ax-R < 0，
    # 一级与二级 ROI 都被画面截断，方块本身完整可见，应走慢速兜底。
    image, expected_pick = _make_low_image(category, DRAW_THETA[category], (0, 0), image_size=(100, 100))
    assert expected_pick == (50.0, 50.0)
    result = detect_block_with_high_prior_roi(image, category=category, **LOW_KWARGS)
    assert result["found"] is True
    assert result["timing"]["匹配模式"] == "慢速兜底"
    assert abs(result["px"] - expected_pick[0]) <= 1.0
    assert abs(result["py"] - expected_pick[1]) <= 1.0


def test_low_legacy_fallback_disabled_returns_not_found():
    category = "T"
    image, _ = _make_low_image(category, DRAW_THETA[category], (0, 0), image_size=(100, 100))
    kwargs = dict(LOW_KWARGS, legacy_fallback_enabled=False)
    result = detect_block_with_high_prior_roi(image, category=category, **kwargs)
    assert result["found"] is False
    assert "禁用" in result["message"]


def test_low_fast_path_tolerates_scattered_noise():
    """散点噪声（同色杂点散布全图）不应触发升级，快速路径直接成功。"""
    category = "T"
    rng = np.random.default_rng(7)
    image, expected_pick = _make_low_image(category, DRAW_THETA[category], (4, -3))
    rgb = load_color_segmentation_config(category)["rgb"]
    bgr = (rgb[2], rgb[1], rgb[0])
    speck_mask = rng.random(image.shape[:2]) < 0.0015
    image[speck_mask] = bgr
    result = detect_block_with_high_prior_roi(image, category=category, **LOW_KWARGS)
    assert result["found"] is True
    assert result["timing"]["匹配模式"] == "快速30像素"
    assert result["timing"].get("快速路径失败原因") is None
    assert abs(result["px"] - expected_pick[0]) <= 1.0
    assert abs(result["py"] - expected_pick[1]) <= 1.0


def test_low_detection_never_calls_warp_affine(monkeypatch):
    category = "T"
    image, expected_pick = _make_low_image(category, DRAW_THETA[category], (5, 3))
    detect_block_with_high_prior_roi(image, category=category, **LOW_KWARGS)

    def _forbidden_warp(*_args, **_kwargs):
        raise AssertionError("低位检测流程调用了 cv2.warpAffine")

    monkeypatch.setattr(cv2, "warpAffine", _forbidden_warp)
    result = detect_block_with_high_prior_roi(image, category=category, **LOW_KWARGS)
    assert result["found"] is True
    assert abs(result["px"] - expected_pick[0]) <= 1.0
    assert abs(result["py"] - expected_pick[1]) <= 1.0
