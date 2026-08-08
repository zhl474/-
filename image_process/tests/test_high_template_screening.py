"""高位模板尺寸筛角与无 Padding 匹配方案测试（CPU 可运行）。"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from image_process_lib.template_match.kernels_create import (
    build_angle_foreground_metadata,
    compute_screened_input_padding,
    create_base_shape,
    create_screened_kernels,
    embed_in_center,
    rotate_image,
    select_screen_angles,
)
from image_process_lib.template_match.template_match import get_rect

import image_process_lib.block_scene_detector as detector

BLOCK_PX = 38
CONNECTOR_PX = 5
SMALL_BLOCK_PX = 10
SMALL_CONNECTOR_PX = 2
CATEGORIES = ["L_yellow", "L_blue", "T", "z_blue", "z_green", "square", "line"]

SCREENING_CONFIG = {
    "enabled": True,
    "size_tolerance_px": 4,
    "relaxed_size_tolerance_px": 8,
    "min_candidate_angles": 3,
    "kernel_safety_margin_px": 2,
    "minimum_translation_margin_px": 4,
    "legacy_fallback_enabled": True,
}


def _rotated_template(category, theta, block_px, connector_px):
    """按真实几何生成旋转后的二值模板，返回 (画布二值图, 画布边长, 前景bbox左上角)。"""
    base = create_base_shape(category, block_px, connector_px)
    height, width = base.shape
    length = int(math.ceil(math.sqrt(width ** 2 + height ** 2)))
    if length % 2 == 0:
        length += 1
    canvas = embed_in_center(base, length)
    rotated = rotate_image(canvas, theta)
    binary = (rotated > 0.5).astype(np.uint8)
    ys, xs = np.nonzero(binary)
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    return binary, length, (x1, y1), (x2 - x1, y2 - y1)


def _make_mask(binary, margin):
    """把旋转模板的前景紧边框放入带外扩黑边的 Mask 中，返回 (Mask, 期望旋转中心)。"""
    ys, xs = np.nonzero(binary)
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    bbox = binary[y1:y2, x1:x2]
    mask = np.zeros((bbox.shape[0] + 2 * margin, bbox.shape[1] + 2 * margin), dtype=np.uint8)
    mask[margin:margin + bbox.shape[0], margin:margin + bbox.shape[1]] = bbox
    c = len(binary) // 2
    return mask, (margin - x1 + c, margin - y1 + c)


@pytest.mark.parametrize("category", CATEGORIES)
def test_angle_metadata_matches_binary_foreground(category):
    metadata = build_angle_foreground_metadata(category, BLOCK_PX, CONNECTOR_PX)
    assert len(metadata) >= 90
    for angle, item in metadata.items():
        ys, xs = np.nonzero(item["binary"])
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
        assert (x1, y1, x2, y2) == (0, 0, item["fg_w"], item["fg_h"])
        assert item["bbox"][2] - item["bbox"][0] == item["fg_w"]
        assert item["bbox"][3] - item["bbox"][1] == item["fg_h"]
        assert item["anchor"][0] + item["bbox"][0] == item["anchor"][1] + item["bbox"][1]


@pytest.mark.parametrize("category,theta", [
    ("L_yellow", 13),
    ("L_blue", 47),
    ("T", 91),
    ("z_blue", 33),
    ("z_green", 127),
    ("square", 21),
    ("line", 57),
])
def test_screen_keeps_true_angle_within_tolerance_4(category, theta):
    metadata = build_angle_foreground_metadata(category, BLOCK_PX, CONNECTOR_PX)
    _, _, _, fg_size = _rotated_template(category, float(theta), BLOCK_PX, CONNECTOR_PX)
    candidates, tolerance = select_screen_angles(
        metadata, fg_size[0], fg_size[1], 4, 8, 3
    )
    assert tolerance == 4
    assert float(theta) in candidates


def test_select_screen_angles_relaxes_and_falls_back():
    metadata = {
        0.0: {"fg_w": 100, "fg_h": 50},
        1.0: {"fg_w": 100, "fg_h": 50},
        2.0: {"fg_w": 101, "fg_h": 50},
        3.0: {"fg_w": 200, "fg_h": 200},
    }
    candidates, tolerance = select_screen_angles(metadata, 106, 54, 4, 8, 3)
    assert tolerance == 8
    assert candidates == [0.0, 1.0, 2.0]
    candidates, tolerance = select_screen_angles(metadata, 205, 205, 4, 8, 3)
    assert tolerance is None
    assert len(candidates) < 3


@pytest.mark.parametrize("category", CATEGORIES)
def test_screen_can_relax_to_8_on_real_metadata(category):
    metadata = build_angle_foreground_metadata(category, BLOCK_PX, CONNECTOR_PX)
    found = False
    for item in metadata.values():
        for dx in range(1, 9):
            candidates, tolerance = select_screen_angles(
                metadata, item["fg_w"] + dx, item["fg_h"], 4, 8, 3
            )
            if tolerance == 8:
                assert len(candidates) >= 3
                found = True
                break
        if found:
            break
    assert found


def test_get_rect_falls_back_to_legacy_when_screening_fails():
    mask = np.zeros((56, 56), dtype=np.uint8)
    mask[8:48, 8:48] = 255
    debug = {}
    rect_fast = get_rect(
        mask, SMALL_BLOCK_PX, SMALL_CONNECTOR_PX, "square", None, 0, 0,
        debug_output=debug,
        screening_config=dict(SCREENING_CONFIG),
    )
    assert debug.get("screening_fallback") is True
    assert "screening_fallback_reason" in debug
    rect_legacy = get_rect(
        mask, SMALL_BLOCK_PX, SMALL_CONNECTOR_PX, "square", None, 0, 0
    )
    assert rect_fast == rect_legacy


def test_get_rect_raises_when_fallback_disabled():
    mask = np.zeros((56, 56), dtype=np.uint8)
    mask[8:48, 8:48] = 255
    config = dict(SCREENING_CONFIG, legacy_fallback_enabled=False)
    with pytest.raises(RuntimeError):
        get_rect(
            mask, SMALL_BLOCK_PX, SMALL_CONNECTOR_PX, "square", None, 0, 0,
            screening_config=config,
        )


def test_screening_disabled_uses_legacy_path():
    binary, _, _, _ = _rotated_template("line", 0.0, SMALL_BLOCK_PX, SMALL_CONNECTOR_PX)
    mask, _ = _make_mask(binary, margin=8)
    debug = {}
    config = dict(SCREENING_CONFIG, enabled=False)
    get_rect(
        mask, SMALL_BLOCK_PX, SMALL_CONNECTOR_PX, "line", None, 0, 0,
        debug_output=debug,
        screening_config=config,
    )
    assert "screening_fallback" not in debug
    assert "mask_fg_size" not in debug


@pytest.mark.parametrize("image_h,image_w,kernel_h,kernel_w,margin,expect", [
    (200, 200, 100, 100, 4, {
        "pad_left": 0, "pad_right": 0, "pad_top": 0, "pad_bottom": 0,
        "target_w": 200, "target_h": 200,
    }),
    (100, 100, 120, 120, 4, {
        "pad_left": 14, "pad_right": 14, "pad_top": 14, "pad_bottom": 14,
        "target_w": 128, "target_h": 128,
    }),
    (101, 100, 120, 120, 4, {
        "pad_left": 14, "pad_right": 14, "pad_top": 13, "pad_bottom": 14,
        "target_w": 128, "target_h": 128,
    }),
    (200, 100, 120, 120, 4, {
        "pad_left": 14, "pad_right": 14, "pad_top": 0, "pad_bottom": 0,
        "target_w": 128, "target_h": 200,
    }),
    (100, 200, 120, 120, 4, {
        "pad_left": 0, "pad_right": 0, "pad_top": 14, "pad_bottom": 14,
        "target_w": 200, "target_h": 128,
    }),
])
def test_compute_screened_input_padding_variants(image_h, image_w, kernel_h, kernel_w, margin, expect):
    result = compute_screened_input_padding(image_h, image_w, kernel_h, kernel_w, margin)
    for key, value in expect.items():
        assert result[key] == value


def test_fast_path_pads_when_kernel_larger_than_mask():
    binary, length, (x1, y1), _ = _rotated_template("square", 45.0, SMALL_BLOCK_PX, SMALL_CONNECTOR_PX)
    ys, xs = np.nonzero(binary)
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    mask = binary[y1:y2, x1:x2].copy()
    debug = {}
    rect = get_rect(
        mask, SMALL_BLOCK_PX, SMALL_CONNECTOR_PX, "square", None, 0, 0,
        debug_output=debug,
        screening_config=dict(SCREENING_CONFIG),
    )
    assert any(debug["padding"])
    assert debug.get("screening_fallback") is False
    expected = (length // 2 - x1, length // 2 - y1)
    assert abs(rect[0][0] - expected[0]) <= 1.0
    assert abs(rect[0][1] - expected[1]) <= 1.0


def test_fast_path_asymmetric_padding():
    binary, length, (x1, y1), _ = _rotated_template("line", 0.0, SMALL_BLOCK_PX, SMALL_CONNECTOR_PX)
    ys, xs = np.nonzero(binary)
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    bbox = binary[y1:y2, x1:x2]
    mask = np.zeros((bbox.shape[0] + 1, bbox.shape[1] + 2), dtype=np.uint8)
    mask[:bbox.shape[0], :bbox.shape[1]] = bbox
    debug = {}
    rect = get_rect(
        mask, SMALL_BLOCK_PX, SMALL_CONNECTOR_PX, "line", None, 0, 0,
        debug_output=debug,
        screening_config=dict(SCREENING_CONFIG),
    )
    pad_left, pad_right, pad_top, pad_bottom = debug["padding"]
    assert pad_top != pad_bottom or pad_left != pad_right
    expected = (length // 2 - x1, length // 2 - y1)
    assert abs(rect[0][0] - expected[0]) <= 1.0
    assert abs(rect[0][1] - expected[1]) <= 1.0


@pytest.mark.parametrize("category,theta", [
    ("L_yellow", 67),
    ("L_blue", 143),
    ("T", 200),
    ("z_blue", 33),
    ("z_green", 127),
    ("square", 45),
    ("line", 57),
])
def test_fast_path_recovers_center_with_angle_anchors(category, theta):
    binary, _, _, _ = _rotated_template(category, float(theta), SMALL_BLOCK_PX, SMALL_CONNECTOR_PX)
    mask, expected = _make_mask(binary, margin=10)
    debug = {}
    rect = get_rect(
        mask, SMALL_BLOCK_PX, SMALL_CONNECTOR_PX, category, None, 0, 0,
        debug_output=debug,
        screening_config=dict(SCREENING_CONFIG),
    )
    assert debug.get("screening_fallback") is False
    assert abs(rect[0][0] - expected[0]) <= 1.0
    assert abs(rect[0][1] - expected[1]) <= 1.0
    assert abs(abs(rect[2]) - float(theta)) <= 1.0


def test_fast_matches_legacy_center_angle_and_contour():
    binary, _, _, _ = _rotated_template("T", 30.0, SMALL_BLOCK_PX, SMALL_CONNECTOR_PX)
    mask, _ = _make_mask(binary, margin=12)
    debug_fast = {}
    debug_legacy = {}
    img_fast = np.zeros((80, 120, 3), dtype=np.uint8)
    img_legacy = np.zeros((80, 120, 3), dtype=np.uint8)
    rect_fast = get_rect(
        mask, SMALL_BLOCK_PX, SMALL_CONNECTOR_PX, "T", img_fast, 0, 0,
        debug_output=debug_fast,
        screening_config=dict(SCREENING_CONFIG),
    )
    rect_legacy = get_rect(
        mask, SMALL_BLOCK_PX, SMALL_CONNECTOR_PX, "T", img_legacy, 0, 0,
        debug_output=debug_legacy,
    )
    assert debug_fast.get("screening_fallback") is False
    assert abs(rect_fast[0][0] - rect_legacy[0][0]) <= 1.0
    assert abs(rect_fast[0][1] - rect_legacy[0][1]) <= 1.0
    assert abs(rect_fast[2] - rect_legacy[2]) <= 1.0
    assert np.array_equal(img_fast, img_legacy)


def test_screening_recomputed_per_mask():
    binary0, _, _, _ = _rotated_template("line", 0.0, SMALL_BLOCK_PX, SMALL_CONNECTOR_PX)
    binary90, _, _, _ = _rotated_template("line", 90.0, SMALL_BLOCK_PX, SMALL_CONNECTOR_PX)
    mask0, _ = _make_mask(binary0, margin=8)
    mask90, _ = _make_mask(binary90, margin=8)
    debug0 = {}
    debug90 = {}
    get_rect(
        mask0, SMALL_BLOCK_PX, SMALL_CONNECTOR_PX, "line", None, 0, 0,
        debug_output=debug0,
        screening_config=dict(SCREENING_CONFIG),
    )
    get_rect(
        mask90, SMALL_BLOCK_PX, SMALL_CONNECTOR_PX, "line", None, 0, 0,
        debug_output=debug90,
        screening_config=dict(SCREENING_CONFIG),
    )
    assert debug0["mask_fg_size"] != debug90["mask_fg_size"]


def test_screened_kernels_cached_and_anchors_in_bounds():
    metadata = build_angle_foreground_metadata("square", SMALL_BLOCK_PX, SMALL_CONNECTOR_PX)
    angles = [float(angle) for angle in sorted(metadata)[20:30]]
    prepared = create_screened_kernels(
        "square", SMALL_BLOCK_PX, SMALL_CONNECTOR_PX, angles,
        device="cpu", safety_margin_px=2,
    )
    kernel_h, kernel_w = prepared["kernel_size"]
    for anchor_x, anchor_y in prepared["anchors"]:
        assert 0 <= anchor_x < kernel_w
        assert 0 <= anchor_y < kernel_h
    assert prepared["angles"] == angles
    cached = create_screened_kernels(
        "square", SMALL_BLOCK_PX, SMALL_CONNECTOR_PX, angles,
        device="cpu", safety_margin_px=2,
    )
    assert cached["kernels"] is prepared["kernels"]


def test_match_block_mask_wires_screening_config():
    binary, _, _, _ = _rotated_template("square", 45.0, SMALL_BLOCK_PX, SMALL_CONNECTOR_PX)
    ys, xs = np.nonzero(binary)
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    mask = binary[y1:y2, x1:x2].copy()
    result = detector.match_block_mask(
        mask,
        "square",
        {"block_px": SMALL_BLOCK_PX, "connector_px": SMALL_CONNECTOR_PX},
        (0, 0, mask.shape[1], mask.shape[0]),
        screening_config=dict(SCREENING_CONFIG),
    )
    assert result["screening"]["mask_fg_size"] == (mask.shape[1], mask.shape[0])
    assert result["screening"].get("screening_fallback") is False
