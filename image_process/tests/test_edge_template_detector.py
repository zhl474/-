"""识别 V2 边缘模板匹配器的合成图精度和 L 抓点口径测试。

合成图用与 build_angle_foreground_metadata 完全相同的旋转代码路径
（embed_in_center + rotate_image）渲染，保证期望中心与角度无歧义。
全部用 CPU 和大角度步进（10°），不依赖 GPU 与真实模型。
"""

import math

import cv2
import numpy as np
import pytest

from image_process_lib.block_detection import coreect_LL_location
from image_process_lib.edge_template_detector import (
    EdgeTemplateConfig,
    EdgeTemplateMatcher,
    build_edge_distance_field,
)
from image_process_lib.template_match.kernels_create import (
    ROTATION_TOTAL_ANGLE,
    create_base_shape,
    embed_in_center,
    get_template_rect_size,
    rotate_image,
)


BLOCK_PX = 18
CONNECTOR_PX = 4
ANGLE_STEP = 10.0
CANVAS_H = 240
CANVAS_W = 280
PASTE_Y = 70
PASTE_X = 90


def _render_synthetic_block(category, angle, template_runs=None):
    """按元数据同款旋转路径渲染方块，返回 (BGR图, 方块Mask, 检测框, 期望中心)。"""
    base = create_base_shape(category, BLOCK_PX, CONNECTOR_PX, template_runs=template_runs)
    height, width = base.shape
    length = int(math.ceil(math.sqrt(width ** 2 + height ** 2)))
    if length % 2 == 0:
        length += 1
    square = embed_in_center(base, length)
    rotated = rotate_image(square, angle)
    mask = rotated > 0.5

    image = np.full((CANVAS_H, CANVAS_W, 3), 40, dtype=np.uint8)
    region = image[PASTE_Y:PASTE_Y + length, PASTE_X:PASTE_X + length]
    region[mask] = 210

    ys, xs = np.nonzero(mask)
    box = (
        float(PASTE_X + xs.min() - 4),
        float(PASTE_Y + ys.min() - 4),
        float(PASTE_X + xs.max() + 5),
        float(PASTE_Y + ys.max() + 5),
    )
    expected_center = (float(length // 2 + PASTE_X), float(length // 2 + PASTE_Y))
    return image, rotated, mask, box, expected_center


def _make_matcher():
    config = EdgeTemplateConfig(angle_step_deg=ANGLE_STEP, search_margin_px=20)
    return EdgeTemplateMatcher(
        {"block_px": BLOCK_PX, "connector_px": CONNECTOR_PX},
        config,
        device="cpu",
    )


def _folded_angle_diff(a, b, period):
    diff = abs(a - b) % period
    return min(diff, period - diff)


@pytest.mark.parametrize(
    "category,angle",
    [
        ("T", 0.0),
        ("T", 30.0),
        ("square", 20.0),
        ("line", 40.0),
        ("z_blue", 70.0),
        ("z_green", 50.0),
        ("L_blue", 30.0),
    ],
)
def test_match_recovers_center_and_angle(category, angle):
    image, _, _, box, expected_center = _render_synthetic_block(category, angle)
    matcher = _make_matcher()
    distance_field = build_edge_distance_field(image, matcher.config)
    match = matcher.match_block(distance_field, {
        "category": category,
        "score": 0.9,
        "box": box,
    })

    position_error = math.hypot(
        match["center_px"] - expected_center[0],
        match["center_py"] - expected_center[1],
    )
    assert position_error <= 2.0
    if category not in ("L_yellow", "L_blue"):
        # 非 L 类输出点即旋转中心；L 类输出点为方案A抓点，单独在下方测试。
        assert math.hypot(
            match["px"] - match["center_px"], match["py"] - match["center_py"]
        ) <= 1e-6

    period = float(ROTATION_TOTAL_ANGLE[category])
    expected_theta = (-angle + 180.0) % 360.0 - 180.0
    angle_error = _folded_angle_diff(match["theta"], expected_theta, period)
    assert angle_error <= ANGLE_STEP + 1e-6

    assert match["category"] == category
    assert -180.0 <= match["theta"] < 180.0


@pytest.mark.parametrize("category,angle", [
    ("L_yellow", 0.0),
    ("L_yellow", 30.0),
    ("L_blue", 60.0),
])
def test_l_pick_point_matches_v1_rule(category, angle):
    """L 块 V2 输出必须与 V1 coreect_LL_location 的方案A抓点同口径。"""
    image, rotated, mask, box, _ = _render_synthetic_block(category, angle)
    matcher = _make_matcher()
    distance_field = build_edge_distance_field(image, matcher.config)
    match = matcher.match_block(distance_field, {
        "category": category,
        "score": 0.9,
        "box": box,
    })

    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    biggest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(biggest)
    box_points = np.intp(cv2.boxPoints(rect))
    pick_x, pick_y = coreect_LL_location(box_points, mask_u8, rect)
    v1_global = (
        float(pick_x + PASTE_X),
        float(pick_y + PASTE_Y),
    )
    # V1 用实测矩形 float 角点，V2 用理想模板构造矩形，boxPoints 取整会差 1~2px。
    assert math.hypot(match["px"] - v1_global[0], match["py"] - v1_global[1]) <= 3.0

    expected_theta = (-angle + 180.0) % 360.0 - 180.0
    assert _folded_angle_diff(match["theta"], expected_theta, 360.0) <= ANGLE_STEP + 1e-6


def test_detect_blocks_skips_unknown_category():
    image, _, _, box, expected_center = _render_synthetic_block("T", 0.0)
    matcher = _make_matcher()
    detections = [
        {"category": "T", "score": 0.9, "box": box},
        {"category": "board", "score": 0.9, "box": (0.0, 0.0, 10.0, 10.0)},
    ]
    blocks, debug_image = matcher.detect_blocks(image, detections)
    assert len(blocks) == 1
    assert blocks[0]["category"] == "T"
    assert debug_image is not None
    assert math.hypot(
        blocks[0]["px"] - expected_center[0],
        blocks[0]["py"] - expected_center[1],
    ) <= 2.0


def test_match_honors_category_runs_overrides():
    """按类别 overrides 端到端：渲染和匹配器都用微调线段时，L 抓点与 V1 规则同口径。"""
    runs = {
        "x_runs": (BLOCK_PX, CONNECTOR_PX, BLOCK_PX, CONNECTOR_PX, BLOCK_PX + 3),
        "y_runs": (BLOCK_PX, CONNECTOR_PX, BLOCK_PX),
    }
    image, rotated, mask, box, _ = _render_synthetic_block(
        "L_blue", 20.0, template_runs=runs
    )
    matcher = EdgeTemplateMatcher(
        {
            "block_px": BLOCK_PX,
            "connector_px": CONNECTOR_PX,
            "runs": {"L_blue": runs},
        },
        EdgeTemplateConfig(angle_step_deg=ANGLE_STEP, search_margin_px=20),
        device="cpu",
    )
    distance_field = build_edge_distance_field(image, matcher.config)
    match = matcher.match_block(distance_field, {
        "category": "L_blue",
        "score": 0.9,
        "box": box,
    })

    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(biggest)
    box_points = np.intp(cv2.boxPoints(rect))
    pick_x, pick_y = coreect_LL_location(box_points, mask_u8, rect)
    v1_global = (float(pick_x + PASTE_X), float(pick_y + PASTE_Y))
    assert math.hypot(match["px"] - v1_global[0], match["py"] - v1_global[1]) <= 3.0

    expected_theta = (-20.0 + 180.0) % 360.0 - 180.0
    assert _folded_angle_diff(match["theta"], expected_theta, 360.0) <= ANGLE_STEP + 1e-6


def test_config_from_mapping_validates():
    config = EdgeTemplateConfig.from_mapping({})
    assert config.angle_step_deg == 2.0
    assert config.search_margin_px == 20

    with pytest.raises(ValueError):
        EdgeTemplateConfig.from_mapping({"angle_step_deg": 0})
    with pytest.raises(ValueError):
        EdgeTemplateConfig.from_mapping({"canny_low": 200, "canny_high": 100})
    with pytest.raises(ValueError):
        EdgeTemplateConfig.from_mapping({"gaussian_ksize": 4})
    with pytest.raises(ValueError):
        EdgeTemplateConfig.from_mapping({"search_margin_px": -1})
    with pytest.raises(ValueError):
        EdgeTemplateConfig.from_mapping("bad")


def test_l_kernel_pick_anchors_differ_from_rotation_center():
    """L 类每个角度都应记录独立抓点锚点，非 L 类抓点即旋转中心。"""
    matcher = _make_matcher()
    l_entry = matcher._kernels_for("L_yellow")
    assert len(l_entry["pick_anchors"]) == len(l_entry["anchors"]) == len(l_entry["angles"])
    assert l_entry["pick_anchors"] != l_entry["anchors"]

    t_entry = matcher._kernels_for("T")
    assert t_entry["pick_anchors"] == t_entry["anchors"]

    rect_w, rect_h = get_template_rect_size("L_yellow", BLOCK_PX, CONNECTOR_PX)
    for anchor, pick in zip(l_entry["anchors"], l_entry["pick_anchors"]):
        assert abs(pick[0] - anchor[0]) <= max(rect_w, rect_h)
        assert abs(pick[1] - anchor[1]) <= max(rect_w, rect_h)
