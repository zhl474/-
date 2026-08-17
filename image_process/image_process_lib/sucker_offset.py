"""吸盘偏移标定策略与三套分区偏移的选择、校验。

SINGLE_CALIBRATION（默认，等价旧行为）：
    方块与托盘全部使用中间标定 camera_to_sucker_offset_mm。
THREE_CALIBRATION：
    托盘恒用中间标定；方块按高位检测像素 u 相对图像中线分左右：
      u < 中线（默认 640，1280 宽图像）→ camera_to_sucker_offset_left_mm
      u >= 中线                       → camera_to_sucker_offset_right_mm

本模块只依赖 numpy，供 competition 执行端与 image_process 感知端共用，
保证两端的策略选择与校验规则完全一致。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

STRATEGY_SINGLE_CALIBRATION = "SINGLE_CALIBRATION"
STRATEGY_THREE_CALIBRATION = "THREE_CALIBRATION"
_SUPPORTED_STRATEGIES = (STRATEGY_SINGLE_CALIBRATION, STRATEGY_THREE_CALIBRATION)

DEFAULT_IMAGE_WIDTH_PX = 1280.0
_SUBJECTS = ("block", "tray")


def _pair_array(value, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须包含 2 个有限数值") from exc
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须包含 2 个有限数值")
    return array


def _pair_list(value, name: str) -> list:
    return [float(item) for item in _pair_array(value, name)]


def classify_pixel_side(
    pixel_xy: Optional[Sequence[float]],
    image_width_px: float = DEFAULT_IMAGE_WIDTH_PX,
) -> Optional[str]:
    """按高位检测像素的 u 相对图像中线判定左右分区。

    pixel_xy 缺失/非有限/(0, 0)（TaskTarget 默认缺失标记）时返回 None。
    返回 "left"（u < 中线）或 "right"（u >= 中线）。
    偏移分区与模型分区共用本函数，保证两侧判定完全一致。
    """
    if pixel_xy is None:
        return None
    try:
        values = np.asarray(pixel_xy, dtype=float)
    except (TypeError, ValueError):
        return None
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        return None
    if float(values[0]) == 0.0 and float(values[1]) == 0.0:
        return None
    center_x = float(image_width_px) / 2.0
    if float(values[0]) < center_x:
        return "left"
    return "right"


def validate_sucker_offset_config(config) -> dict:
    """校验标定策略与三套偏移，返回规范化结果。

    返回 {"strategy": str, "center": [vx, vy], "left": [...]|None, "right": [...]|None}。
    规则：
      - camera_to_sucker_offset_mm（中间）必填，且为 2 个有限数值；
      - sucker_offset_strategy 缺省为 SINGLE_CALIBRATION，取值必须合法；
      - left/right 可选但必须成对配置，各自为 2 个有限数值；
      - THREE_CALIBRATION 策略下 left/right 必须同时配置（否则运行期无法分侧）。
    """
    if not isinstance(config, dict):
        raise ValueError("visual_servo 配置必须是字典")
    strategy = config.get("sucker_offset_strategy", STRATEGY_SINGLE_CALIBRATION)
    if strategy not in _SUPPORTED_STRATEGIES:
        raise ValueError(
            f"sucker_offset_strategy 只能是 {STRATEGY_SINGLE_CALIBRATION} "
            f"或 {STRATEGY_THREE_CALIBRATION}，当前为 {strategy!r}"
        )
    center = _pair_list(
        config.get("camera_to_sucker_offset_mm"),
        "camera_to_sucker_offset_mm",
    )
    left_raw = config.get("camera_to_sucker_offset_left_mm")
    right_raw = config.get("camera_to_sucker_offset_right_mm")
    if (left_raw is None) != (right_raw is None):
        raise ValueError(
            "camera_to_sucker_offset_left_mm 与 camera_to_sucker_offset_right_mm "
            "必须成对配置"
        )
    left = (
        None
        if left_raw is None
        else _pair_list(left_raw, "camera_to_sucker_offset_left_mm")
    )
    right = (
        None
        if right_raw is None
        else _pair_list(right_raw, "camera_to_sucker_offset_right_mm")
    )
    if strategy == STRATEGY_THREE_CALIBRATION and (left is None or right is None):
        raise ValueError(
            f"{STRATEGY_THREE_CALIBRATION} 策略必须同时配置 "
            "camera_to_sucker_offset_left_mm 与 camera_to_sucker_offset_right_mm"
        )
    return {
        "strategy": strategy,
        "center": center,
        "left": left,
        "right": right,
    }


def resolve_sucker_offset(
    config,
    subject: str,
    pixel_xy: Optional[Sequence[float]] = None,
    image_width_px: float = DEFAULT_IMAGE_WIDTH_PX,
) -> Tuple[list, str]:
    """按策略与目标主体选出实际使用的吸盘偏移。

    config：visual_servo.yaml 解析后的 dict（strategy 键可由调用方按启动值覆盖）。
    subject："block"（方块，THREE 策略下按像素分左右）或 "tray"（托盘，恒用中间）。
    pixel_xy：方块高位检测像素 (u, v)；缺失/非有限/(0, 0)（TaskTarget 默认缺失标记）
              时回退中间偏移。
    image_width_px：检测图像宽（当前相机固定 1280），中线 = 宽 / 2 = 640。

    返回 (offset, side)：offset 为 [vx, vy]（mm）；side 为 "left"/"right"/"center"，
    供日志与 CSV 记录。
    """
    if subject not in _SUBJECTS:
        raise ValueError(
            f"subject 只能是 block 或 tray，当前为 {subject!r}"
        )
    strategy = config.get("sucker_offset_strategy", STRATEGY_SINGLE_CALIBRATION)
    center = _pair_list(
        config.get("camera_to_sucker_offset_mm"),
        "camera_to_sucker_offset_mm",
    )
    if strategy == STRATEGY_SINGLE_CALIBRATION or subject == "tray":
        return center, "center"
    left_raw = config.get("camera_to_sucker_offset_left_mm")
    right_raw = config.get("camera_to_sucker_offset_right_mm")
    if left_raw is None or right_raw is None:
        # 未成对配置时回退中间，等价旧行为（校验层已拦截 THREE 缺左右的情况）。
        return center, "center"
    left = _pair_list(left_raw, "camera_to_sucker_offset_left_mm")
    right = _pair_list(right_raw, "camera_to_sucker_offset_right_mm")
    side = classify_pixel_side(pixel_xy, image_width_px)
    if side is None:
        # 像素缺失/非有限/(0,0) 时回退中间偏移。
        return center, "center"
    if side == "left":
        return left, "left"
    return right, "right"
