"""吸盘偏移标定策略（SINGLE_CALIBRATION / THREE_CALIBRATION）与三套分区偏移的测试。

覆盖：
  - validate_sucker_offset_config：默认策略、成对校验、THREE 必填左右、非法值报错；
  - resolve_sucker_offset：SINGLE 恒用中间、THREE 分左右与边界、像素缺失回退、
    非法 subject 报错；
  - apply_camera_to_sucker_offset：旧签名兼容、分区应用、Z 与姿态不变；
  - load_visual_servo_config：真实 yaml 三套字段读取与策略注入。
"""

import os
import sys

import numpy as np
import pytest


PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)
COMPETITION_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, "..", "competition"))
if COMPETITION_DIR not in sys.path:
    sys.path.insert(0, COMPETITION_DIR)

from image_process_lib.sucker_offset import (  # noqa: E402
    DEFAULT_IMAGE_WIDTH_PX,
    STRATEGY_SINGLE_CALIBRATION,
    STRATEGY_THREE_CALIBRATION,
    classify_pixel_side,
    resolve_sucker_offset,
    validate_sucker_offset_config,
)
from competition_lib.config import (  # noqa: E402
    DEFAULT_VISUAL_SERVO_CONFIG_PATH,
    load_visual_servo_config,
)
from competition_lib.visual_servo import apply_camera_to_sucker_offset  # noqa: E402

CENTER = [-85.264, -13.905]
LEFT = [-83.0, -12.0]
RIGHT = [-87.0, -15.5]


def _single_config():
    """SINGLE_CALIBRATION 策略配置：左右存在但与中间一致。"""
    return {
        "camera_to_sucker_offset_mm": list(CENTER),
        "sucker_offset_strategy": STRATEGY_SINGLE_CALIBRATION,
        "camera_to_sucker_offset_left_mm": list(CENTER),
        "camera_to_sucker_offset_right_mm": list(CENTER),
    }


def _three_config():
    """THREE_CALIBRATION 策略配置：左右与中间不同。"""
    return {
        "camera_to_sucker_offset_mm": list(CENTER),
        "sucker_offset_strategy": STRATEGY_THREE_CALIBRATION,
        "camera_to_sucker_offset_left_mm": list(LEFT),
        "camera_to_sucker_offset_right_mm": list(RIGHT),
    }


def _minimal_single_config():
    """最简配置：无策略、无左右字段，等价旧版 yaml。"""
    return {"camera_to_sucker_offset_mm": list(CENTER)}


# ---------------------------------------------------------------- validate


def test_validate_defaults_to_single_without_strategy_field():
    normalized = validate_sucker_offset_config(_minimal_single_config())
    assert normalized["strategy"] == STRATEGY_SINGLE_CALIBRATION
    assert normalized["center"] == CENTER
    assert normalized["left"] is None
    assert normalized["right"] is None


def test_validate_single_with_paired_left_right():
    normalized = validate_sucker_offset_config(_single_config())
    assert normalized["strategy"] == STRATEGY_SINGLE_CALIBRATION
    assert normalized["center"] == CENTER
    assert normalized["left"] == CENTER
    assert normalized["right"] == CENTER


def test_validate_three_requires_paired_left_right():
    normalized = validate_sucker_offset_config(_three_config())
    assert normalized["strategy"] == STRATEGY_THREE_CALIBRATION
    assert normalized["left"] == LEFT
    assert normalized["right"] == RIGHT


@pytest.mark.parametrize(
    "invalid_center",
    [[-85.264], [-85.264, -13.905, 0.0], [float("nan"), -13.905],
     [float("inf"), -13.905], "不是数值", None],
)
def test_validate_rejects_invalid_center(invalid_center):
    with pytest.raises(
        ValueError, match="camera_to_sucker_offset_mm 必须包含 2 个有限数值"
    ):
        validate_sucker_offset_config(
            {"camera_to_sucker_offset_mm": invalid_center}
        )


@pytest.mark.parametrize(
    "invalid_strategy",
    ["", "single", "THREE", "SINGLE", 1, None],
)
def test_validate_rejects_invalid_strategy(invalid_strategy):
    config = _minimal_single_config()
    config["sucker_offset_strategy"] = invalid_strategy
    with pytest.raises(ValueError, match="sucker_offset_strategy 只能是"):
        validate_sucker_offset_config(config)


def test_validate_rejects_left_without_right():
    config = _minimal_single_config()
    config["camera_to_sucker_offset_left_mm"] = list(LEFT)
    with pytest.raises(ValueError, match="必须成对配置"):
        validate_sucker_offset_config(config)


def test_validate_rejects_right_without_left():
    config = _minimal_single_config()
    config["camera_to_sucker_offset_right_mm"] = list(RIGHT)
    with pytest.raises(ValueError, match="必须成对配置"):
        validate_sucker_offset_config(config)


@pytest.mark.parametrize(
    "invalid_pair",
    [[-83.0], [-83.0, -12.0, 0.0], [float("nan"), -12.0], "不是数值"],
)
def test_validate_rejects_invalid_left(invalid_pair):
    config = _minimal_single_config()
    config["camera_to_sucker_offset_left_mm"] = invalid_pair
    config["camera_to_sucker_offset_right_mm"] = list(RIGHT)
    with pytest.raises(
        ValueError, match="camera_to_sucker_offset_left_mm 必须包含 2 个有限数值"
    ):
        validate_sucker_offset_config(config)


def test_validate_three_without_left_right_is_rejected():
    with pytest.raises(
        ValueError, match="THREE_CALIBRATION 策略必须同时配置"
    ):
        validate_sucker_offset_config(
            {
                "camera_to_sucker_offset_mm": list(CENTER),
                "sucker_offset_strategy": STRATEGY_THREE_CALIBRATION,
            }
        )


def test_validate_three_with_only_left_is_rejected():
    config = {
        "camera_to_sucker_offset_mm": list(CENTER),
        "sucker_offset_strategy": STRATEGY_THREE_CALIBRATION,
        "camera_to_sucker_offset_left_mm": list(LEFT),
    }
    with pytest.raises(ValueError, match="必须成对配置"):
        validate_sucker_offset_config(config)


def test_validate_rejects_non_dict_config():
    with pytest.raises(ValueError, match="必须是字典"):
        validate_sucker_offset_config("not-a-dict")


# ---------------------------------------------------------------- resolve


def test_resolve_single_strategy_uses_center_for_block_left_and_right():
    config = _single_config()
    assert resolve_sucker_offset(config, "block", [120.0, 220.0]) == (CENTER, "center")
    assert resolve_sucker_offset(config, "block", [900.0, 220.0]) == (CENTER, "center")
    assert resolve_sucker_offset(config, "tray", [640.0, 300.0]) == (CENTER, "center")


def test_resolve_default_strategy_uses_center_for_block():
    config = _minimal_single_config()
    assert resolve_sucker_offset(config, "block", [100.0, 200.0]) == (CENTER, "center")


def test_resolve_three_tray_always_center():
    config = _three_config()
    assert resolve_sucker_offset(config, "tray", [100.0, 200.0]) == (CENTER, "center")
    assert resolve_sucker_offset(config, "tray", [900.0, 200.0]) == (CENTER, "center")


def test_resolve_three_block_splits_by_center_line():
    config = _three_config()
    assert resolve_sucker_offset(config, "block", [100.0, 220.0]) == (LEFT, "left")
    assert resolve_sucker_offset(config, "block", [900.0, 220.0]) == (RIGHT, "right")


def test_resolve_three_boundary_pixels():
    config = _three_config()
    # u < 640 → left；u == 640 → right（硬分界）。
    assert resolve_sucker_offset(config, "block", [639.999, 220.0]) == (LEFT, "left")
    assert resolve_sucker_offset(config, "block", [640.0, 220.0]) == (RIGHT, "right")


def test_resolve_three_honors_custom_image_width():
    config = _three_config()
    assert resolve_sucker_offset(
        config, "block", [399.0, 220.0], image_width_px=800.0
    ) == (LEFT, "left")
    assert resolve_sucker_offset(
        config, "block", [400.0, 220.0], image_width_px=800.0
    ) == (RIGHT, "right")


def test_resolve_three_missing_pixel_falls_back_to_center():
    config = _three_config()
    assert resolve_sucker_offset(config, "block", None) == (CENTER, "center")
    assert resolve_sucker_offset(config, "block", []) == (CENTER, "center")
    assert resolve_sucker_offset(config, "block", [100.0]) == (CENTER, "center")


def test_resolve_three_zero_pixel_marker_falls_back_to_center():
    config = _three_config()
    # (0, 0) 是 TaskTarget 的默认缺失标记，不是真实检测像素。
    assert resolve_sucker_offset(config, "block", [0.0, 0.0]) == (CENTER, "center")


def test_resolve_three_non_finite_pixel_falls_back_to_center():
    config = _three_config()
    assert resolve_sucker_offset(config, "block", [float("nan"), 220.0]) == (
        CENTER,
        "center",
    )
    assert resolve_sucker_offset(config, "block", [float("inf"), 220.0]) == (
        CENTER,
        "center",
    )
    assert resolve_sucker_offset(config, "block", ["不是数值", 220.0]) == (
        CENTER,
        "center",
    )


def test_resolve_three_unpaired_left_right_falls_back_to_center():
    config = {
        "camera_to_sucker_offset_mm": list(CENTER),
        "sucker_offset_strategy": STRATEGY_THREE_CALIBRATION,
    }
    assert resolve_sucker_offset(config, "block", [100.0, 220.0]) == (CENTER, "center")


@pytest.mark.parametrize("invalid_subject", ["", "cube", "BLOCK", None, 1])
def test_resolve_rejects_invalid_subject(invalid_subject):
    with pytest.raises(ValueError, match="subject 只能是 block 或 tray"):
        resolve_sucker_offset(_three_config(), invalid_subject)


def test_resolve_default_image_width_is_1280():
    assert DEFAULT_IMAGE_WIDTH_PX == 1280.0


# ---------------------------------------------------------------- apply


def test_apply_old_signature_uses_center_offset():
    pose = [10.0, 20.0, 300.0, -180.0, 0.0, 90.0]
    result = apply_camera_to_sucker_offset(pose, _three_config())
    assert result == pytest.approx(
        [10.0 + CENTER[0], 20.0 + CENTER[1], 300.0, -180.0, 0.0, 90.0]
    )


def test_apply_tray_uses_center_even_in_three_strategy():
    pose = [10.0, 20.0, 300.0, -180.0, 0.0, 90.0]
    result = apply_camera_to_sucker_offset(
        pose, _three_config(), subject="tray"
    )
    assert result == pytest.approx(
        [10.0 + CENTER[0], 20.0 + CENTER[1], 300.0, -180.0, 0.0, 90.0]
    )


def test_apply_block_uses_zone_offset_in_three_strategy():
    pose = [10.0, 20.0, 300.0, -180.0, 0.0, 90.0]
    left_result = apply_camera_to_sucker_offset(
        pose, _three_config(), subject="block", pixel_xy=[100.0, 220.0]
    )
    assert left_result == pytest.approx(
        [10.0 + LEFT[0], 20.0 + LEFT[1], 300.0, -180.0, 0.0, 90.0]
    )
    right_result = apply_camera_to_sucker_offset(
        pose, _three_config(), subject="block", pixel_xy=[900.0, 220.0]
    )
    assert right_result == pytest.approx(
        [10.0 + RIGHT[0], 20.0 + RIGHT[1], 300.0, -180.0, 0.0, 90.0]
    )


def test_apply_block_uses_center_in_single_strategy():
    pose = [10.0, 20.0, 300.0, -180.0, 0.0, 90.0]
    result = apply_camera_to_sucker_offset(
        pose, _single_config(), subject="block", pixel_xy=[100.0, 220.0]
    )
    assert result == pytest.approx(
        [10.0 + CENTER[0], 20.0 + CENTER[1], 300.0, -180.0, 0.0, 90.0]
    )


def test_apply_keeps_v3_model_none_behavior():
    config = _three_config()
    config["sucker_offset_model"] = {"type": "none"}
    pose = [10.0, 20.0, 300.0, -180.0, 0.0, 90.0]
    result = apply_camera_to_sucker_offset(
        pose, config, subject="block", pixel_xy=[100.0, 220.0]
    )
    assert result == pytest.approx(
        [10.0 + LEFT[0], 20.0 + LEFT[1], 300.0, -180.0, 0.0, 90.0]
    )


def test_apply_rejects_non_six_dimensional_pose():
    with pytest.raises(ValueError, match="相机位姿必须包含 6 个数值"):
        apply_camera_to_sucker_offset([1.0, 2.0], _three_config())


# ------------------------------------------------- load real yaml config


def test_real_visual_servo_yaml_loads_three_offsets_and_strategy():
    """真实 visual_servo.yaml：默认 SINGLE，三套偏移齐全且当前左右=中间。"""
    config = load_visual_servo_config(DEFAULT_VISUAL_SERVO_CONFIG_PATH)
    assert config["sucker_offset_strategy"] == STRATEGY_SINGLE_CALIBRATION
    assert config["camera_to_sucker_offset_left_mm"] == config[
        "camera_to_sucker_offset_mm"
    ]
    assert config["camera_to_sucker_offset_right_mm"] == config[
        "camera_to_sucker_offset_mm"
    ]


def test_real_yaml_resolve_equals_old_behavior(tmp_path):
    """当前 yaml（SINGLE + 左右=中间）下，方块左右像素与旧版一样用中间偏移。"""
    config = load_visual_servo_config(DEFAULT_VISUAL_SERVO_CONFIG_PATH)
    left_offset, left_side = resolve_sucker_offset(
        config, "block", [100.0, 200.0]
    )
    right_offset, right_side = resolve_sucker_offset(
        config, "block", [900.0, 200.0]
    )
    assert left_side == "center"
    assert right_side == "center"
    assert left_offset == config["camera_to_sucker_offset_mm"]
    assert right_offset == config["camera_to_sucker_offset_mm"]


# ------------------------------------------------- classify_pixel_side


def test_classify_pixel_side_left_and_right():
    assert classify_pixel_side([100.0, 220.0]) == "left"
    assert classify_pixel_side([900.0, 220.0]) == "right"


def test_classify_pixel_side_boundary_640():
    assert classify_pixel_side([639.999, 220.0]) == "left"
    assert classify_pixel_side([640.0, 220.0]) == "right"


def test_classify_pixel_side_honors_custom_width():
    assert classify_pixel_side([399.0, 220.0], image_width_px=800.0) == "left"
    assert classify_pixel_side([400.0, 220.0], image_width_px=800.0) == "right"


@pytest.mark.parametrize(
    "invalid_pixel",
    [None, [], [100.0], [float("nan"), 220.0], [float("inf"), 220.0],
     ["不是数值", 220.0], [0.0, 0.0]],
)
def test_classify_pixel_side_invalid_returns_none(invalid_pixel):
    assert classify_pixel_side(invalid_pixel) is None
