"""Line 模板像素尺寸计算。

对应 tools/vision/计算line模板像素尺寸.py，供网页控制台复用同一套公式。
"""

import math

# 游标卡尺测得 line 模板的真实尺寸（单位：毫米）
LINE_SHORT_SIDE_MM = 17.8
LINE_LONG_SIDE_MM = 78.2


def calculate_template_size(
    p1,
    p2,
    short_side_mm=LINE_SHORT_SIDE_MM,
    long_side_mm=LINE_LONG_SIDE_MM,
):
    """由 line 长边像素长度按真实比例推算 block_px 与 connector_px。"""
    if short_side_mm <= 0 or long_side_mm <= 0:
        raise ValueError("真实尺寸必须大于 0")

    long_side_px = math.dist(p1, p2)
    if long_side_px <= 0:
        raise ValueError("P1 和 P2 不能相同，请填写长边两端不同的像素坐标。")

    # line 的短边就是一个方块边长，按真实长短边比例换算。
    block_px = long_side_px * short_side_mm / long_side_mm
    # line 长边由 4 个方块边长和 3 个连接间隙组成。
    connector_px = (long_side_px - 4 * block_px) / 3
    return long_side_px, block_px, connector_px
