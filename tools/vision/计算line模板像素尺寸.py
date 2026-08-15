#!/home/zhl/fr3env/fr3env/bin/python
"""根据 line 模板长边上的两个像素点，计算模板尺寸。

直接修改下方 P1、P2 的坐标后运行：
/home/zhl/fr3env/fr3env/bin/python 计算line模板像素尺寸.py
"""

import math


# ======================== 直接修改这里的测量点 ========================
# 长边一端的像素坐标 (x, y)
P1 = (977,427)
# 长边另一端的像素坐标 (x, y)
P2 = (981,268)

# 游标卡尺测得 line 模板的真实尺寸（单位：毫米）
LINE_SHORT_SIDE_MM = 17.8
LINE_LONG_SIDE_MM = 78.2


def calculate_template_size(p1, p2):
    """由长边像素长度按真实比例推算 block_px 与 connector_px。"""
    long_side_px = math.dist(p1, p2)
    if long_side_px <= 0:
        raise ValueError("P1 和 P2 不能相同，请填写长边两端不同的像素坐标。")

    # line 的短边就是一个方块边长，按真实长短边比例换算。
    block_px = long_side_px * LINE_SHORT_SIDE_MM / LINE_LONG_SIDE_MM
    # line 长边由 4 个方块边长和 3 个连接间隙组成。
    connector_px = (long_side_px - 4 * block_px) / 3
    return long_side_px, block_px, connector_px


if __name__ == "__main__":
    try:
        long_px, block_px, connector_px = calculate_template_size(P1, P2)
    except ValueError as error:
        print(f"输入错误：{error}")
    else:
        print(f"长边像素长度: {long_px:.2f} px")
        print(f"block_px: {block_px:.2f}")
        print(f"connector_px: {connector_px:.2f}")
