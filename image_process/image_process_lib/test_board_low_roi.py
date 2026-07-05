import os
import sys

import cv2
import numpy as np


PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

from image_process_lib.board_detect import detect_nearest_board_dot_in_roi


def _make_board_image(width=320, height=240):
    """生成浅色托盘背景，黑色圆点用于模拟低位可见格点。"""
    return np.full((height, width, 3), 220, dtype=np.uint8)


def _draw_dot(image, center, radius=5):
    cv2.circle(image, center, radius, (20, 20, 20), -1)


def test_low_board_roi_detects_single_dot():
    image = _make_board_image()
    _draw_dot(image, (165, 122))

    result = detect_nearest_board_dot_in_roi(
        image,
        center_point=(160, 120),
        roi_half_size=60,
    )

    assert result["found"] is True
    np.testing.assert_allclose(result["point"], [165, 122], atol=1.0)
    assert result["count"] == 1


def test_low_board_roi_chooses_candidate_nearest_to_center():
    image = _make_board_image()
    _draw_dot(image, (150, 120))
    _draw_dot(image, (205, 120))
    _draw_dot(image, (160, 134))

    result = detect_nearest_board_dot_in_roi(
        image,
        center_point=(160, 120),
        roi_half_size=80,
    )

    assert result["found"] is True
    np.testing.assert_allclose(result["point"], [150, 120], atol=1.0)
    assert result["count"] == 3


def test_low_board_roi_ignores_dots_outside_roi():
    image = _make_board_image()
    _draw_dot(image, (250, 120))

    result = detect_nearest_board_dot_in_roi(
        image,
        center_point=(160, 120),
        roi_half_size=50,
    )

    assert result["found"] is False
    assert result["message"] == "低位 ROI 内未检测到托盘圆点"
    assert result["count"] == 0


def test_low_board_roi_filters_large_block_interference():
    image = _make_board_image()
    cv2.rectangle(image, (135, 95), (185, 145), (20, 20, 20), -1)
    _draw_dot(image, (205, 120))

    result = detect_nearest_board_dot_in_roi(
        image,
        center_point=(160, 120),
        roi_half_size=80,
        max_area=250,
    )

    assert result["found"] is True
    np.testing.assert_allclose(result["point"], [205, 120], atol=1.0)
    assert result["count"] == 1
