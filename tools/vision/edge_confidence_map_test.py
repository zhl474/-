# -*- coding: utf-8 -*-
"""
Edge Confidence Map 实验脚本（独立，不依赖其他运行代码）

实验流程：
1. 原图灰度化
2. 轻微 Gaussian 模糊
3. Canny 边缘（独立输出，供对比）
4. LSD 线段检测（独立于 Canny，直接用灰度图）
5. 删除短线（length < MIN_LENGTH px）
6. 利用 white mask，检查每条线两侧 3~5px：
   一侧白色比例 > WHITE_RATIO_THRESH -> 该线为"白板邻接边"
7. 生成 Edge Confidence Map：
   黑色 0   = 背景
   85       = 普通边
   255      = 白板邻接边

运行方式（改下面 IMAGE_PATH 等参数后直接跑）：
/home/zhl/fr3env/fr3env/bin/python edge_confidence_map_test.py
"""

import cv2
import numpy as np
import os

# ===================== 传参区（改这里） =====================
IMAGE_PATH = "/home/zhl/SingleArmTetris/SingleArmTetris/src/tools/vision/test_image.png"  # 输入图片

# Canny 参数
CANNY_LOW = 50
CANNY_HIGH = 150

# Gaussian 参数
GAUSS_KSIZE = (3, 3)

# LSD 短线过滤
MIN_LENGTH = 15.0  # px，小于此长度的线段直接丢弃

# 白板 mask 参数（HSV 白色阈值）
WHITE_LOWER = np.array([0, 0, 180])    # HSV 下界
WHITE_UPPER = np.array([179, 60, 255])  # HSV 上界

# 线两侧采样
SIDE_OFFSET = 4.0        # 沿法线偏移 3~5px
SIDE_RATIO_THRESH = 0.8  # 一侧白色比例 > 80% -> 白板邻接边

# 输出
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))  # 结果保存到脚本同目录
# ==========================================================


def build_white_mask(bgr):
    """生成白色区域 mask（单通道 0/255）"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    # 轻微形态学去噪
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def sample_side_ratio(line, white_mask, sign):
    """
    沿线段法线方向偏移 SIDE_OFFSET，采样该偏移线上的白色像素比例。
    line: [x1, y1, x2, y2]
    sign: +1 或 -1，表示法线的哪一侧
    返回白色比例 (0~1)，采样失败返回 0
    """
    x1, y1, x2, y2 = line
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    if length < 1e-6:
        return 0.0
    # 单位法向量
    nx, ny = -dy / length, dx / length
    ox, oy = nx * SIDE_OFFSET * sign, ny * SIDE_OFFSET * sign

    h, w = white_mask.shape
    n_samples = max(int(length), 2)
    ts = np.linspace(0, 1, n_samples)
    xs = x1 + ts * dx + ox
    ys = y1 + ts * dy + oy
    xi = np.round(xs).astype(int)
    yi = np.round(ys).astype(int)
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    if not np.any(valid):
        return 0.0
    vals = white_mask[yi[valid], xi[valid]]
    return float(np.count_nonzero(vals)) / len(vals)


def main():
    bgr = cv2.imread(IMAGE_PATH)
    if bgr is None:
        raise FileNotFoundError("读不到图片: %s" % IMAGE_PATH)

    # Step 1 灰度化
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Step 2 轻微 Gaussian
    blur = cv2.GaussianBlur(gray, GAUSS_KSIZE, 0)

    # Step 3 Canny（独立输出对比）
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    # Step 4 LSD
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(gray)[0]  # shape (N,1,4)
    if lines is None:
        lines = np.empty((0, 1, 4), dtype=np.float32)
    print("LSD 原始线段数: %d" % len(lines))

    # white mask
    white_mask = build_white_mask(bgr)

    # Step 5 + 6 + 7 生成 Edge Confidence Map
    conf_map = np.zeros_like(gray)
    n_kept, n_white_edge = 0, 0
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = line
        if np.hypot(x2 - x1, y2 - y1) < MIN_LENGTH:
            continue  # Step 5 删短线
        n_kept += 1

        # Step 6 两侧白色比例
        ratio_p = sample_side_ratio(line, white_mask, +1)
        ratio_m = sample_side_ratio(line, white_mask, -1)
        is_white_edge = max(ratio_p, ratio_m) > SIDE_RATIO_THRESH

        # Step 7 画线：普通边 85，白板邻接边 255
        val = 255 if is_white_edge else 85
        if is_white_edge:
            n_white_edge += 1
        cv2.line(conf_map,
                 (int(round(x1)), int(round(y1))),
                 (int(round(x2)), int(round(y2))),
                 val, 1, cv2.LINE_AA)

    print("过滤后线段数: %d, 其中白板邻接边: %d" % (n_kept, n_white_edge))

    # 保存结果
    os.makedirs(SAVE_DIR, exist_ok=True)
    out_gray = os.path.join(SAVE_DIR, "ecm_gray.png")
    out_edges = os.path.join(SAVE_DIR, "ecm_canny.png")
    out_mask = os.path.join(SAVE_DIR, "ecm_white_mask.png")
    out_map = os.path.join(SAVE_DIR, "ecm_confidence_map.png")
    cv2.imwrite(out_gray, gray)
    cv2.imwrite(out_edges, edges)
    cv2.imwrite(out_mask, white_mask)
    cv2.imwrite(out_map, conf_map)
    print("已保存:")
    print("  灰度图        ->", out_gray)
    print("  Canny 边缘    ->", out_edges)
    print("  white mask    ->", out_mask)
    print("  置信边图      ->", out_map)

    # 拼一张对比图方便肉眼检查：原图 | Canny | white mask | confidence map
    def to3(u8):
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)

    panel = np.hstack([
        bgr,
        cv2.addWeighted(to3(edges), 0.8, bgr, 0.2, 0),
        cv2.addWeighted(to3(white_mask), 0.8, bgr, 0.2, 0),
        to3(conf_map),
    ])
    out_panel = os.path.join(SAVE_DIR, "ecm_panel.png")
    cv2.imwrite(out_panel, panel)
    print("  四联对比图    ->", out_panel)

    cv2.imshow("panel (press any key to exit)", cv2.resize(
        panel, None, fx=0.5, fy=0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
