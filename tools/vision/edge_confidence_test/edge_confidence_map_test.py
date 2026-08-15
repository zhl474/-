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

可视化（核心目的：看线段能否准确划分物体边界，用于识别定位）：
- 叠加在【原图】上：普通边=橙色，白板邻接边=红色，肉眼对比边界贴合程度
- 叠加在【灰度图】上：同样画法，排除颜色干扰看结构
- 四联对比图：原图 | 原图+线段 | 灰度+线段 | 伪彩置信图

运行方式（改下面 IMAGE_PATH 等参数后直接跑）：
/home/zhl/fr3env/fr3env/bin/python edge_confidence_map_test.py
"""

import cv2
import numpy as np
import os

# ===================== 传参区（改这里） =====================
IMAGE_PATH = "/home/zhl/SingleArmTetris/SingleArmTetris/src/tools/vision/1.png"  # 输入图片

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

# 可视化参数
LINE_THICKNESS = 2                 # 叠加线宽
NORMAL_EDGE_COLOR = (0, 200, 255)  # 普通边颜色（BGR，橙色）
WHITE_EDGE_COLOR = (0, 0, 255)     # 白板邻接边颜色（BGR，红色）
SHOW_WINDOW = True                 # False 则只存图不弹窗（无显示器环境）
PANEL_SCALE = 0.5                  # 四联对比图窗口缩放

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


def detect_and_classify(gray, white_mask):
    """
    Step 4~6: LSD 检测 + 短线过滤 + 白板邻接分类。
    返回 [(x1, y1, x2, y2, is_white_edge), ...]
    """
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(gray)[0]  # shape (N,1,4)
    if lines is None:
        lines = np.empty((0, 1, 4), dtype=np.float32)
    print("LSD 原始线段数: %d" % len(lines))

    kept = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = line
        if np.hypot(x2 - x1, y2 - y1) < MIN_LENGTH:
            continue  # Step 5 删短线
        ratio_p = sample_side_ratio(line, white_mask, +1)
        ratio_m = sample_side_ratio(line, white_mask, -1)
        is_white = max(ratio_p, ratio_m) > SIDE_RATIO_THRESH
        kept.append((x1, y1, x2, y2, is_white))
    return kept


def draw_overlay(base_bgr, seg_lines, with_legend=True):
    """
    把分割线段叠加到一张 BGR 图上（原图或灰度转 BGR 都可以）。
    普通边=橙色，白板邻接边=红色，直接看边界贴合程度。
    """
    vis = base_bgr.copy()
    for x1, y1, x2, y2, is_white in seg_lines:
        color = WHITE_EDGE_COLOR if is_white else NORMAL_EDGE_COLOR
        cv2.line(vis,
                 (int(round(x1)), int(round(y1))),
                 (int(round(x2)), int(round(y2))),
                 color, LINE_THICKNESS, cv2.LINE_AA)
    if with_legend:
        h = vis.shape[0]
        cv2.putText(vis, "white-board edge", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE_EDGE_COLOR, 2)
        cv2.putText(vis, "normal edge", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, NORMAL_EDGE_COLOR, 2)
    return vis


def build_confidence_map(gray, seg_lines):
    """
    Step 7: 灰度 Edge Confidence Map
    黑 0 = 背景，85 = 普通边，255 = 白板邻接边
    注意不用 LINE_AA，保证像素值严格是 0/85/255，方便后续模板匹配
    """
    conf_map = np.zeros_like(gray)
    for x1, y1, x2, y2, is_white in seg_lines:
        val = 255 if is_white else 85
        cv2.line(conf_map,
                 (int(round(x1)), int(round(y1))),
                 (int(round(x2)), int(round(y2))),
                 val, 1)
    return conf_map


def colorize_confidence(conf_map):
    """置信图伪彩色：背景黑，普通边绿，白板邻接边红（仅用于肉眼观察）"""
    lut = np.zeros((1, 256, 3), dtype=np.uint8)
    lut[0, 85] = (0, 255, 0)    # 普通边 -> 绿
    lut[0, 255] = (0, 0, 255)   # 白板邻接边 -> 红
    return cv2.LUT(cv2.cvtColor(conf_map, cv2.COLOR_GRAY2BGR), lut)


def main():
    bgr = cv2.imread(IMAGE_PATH)
    if bgr is None:
        raise FileNotFoundError("读不到图片: %s" % IMAGE_PATH)

    # Step 1 灰度化
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # 用于叠加显示

    # Step 2 轻微 Gaussian
    blur = cv2.GaussianBlur(gray, GAUSS_KSIZE, 0)

    # Step 3 Canny（独立输出对比）
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    # white mask
    white_mask = build_white_mask(bgr)

    # Step 4~6 LSD + 过滤 + 分类
    seg_lines = detect_and_classify(gray, white_mask)
    n_white = sum(1 for *_, w in seg_lines if w)
    print("过滤后线段数: %d, 其中白板邻接边: %d" % (len(seg_lines), n_white))

    # Step 7 Edge Confidence Map
    conf_map = build_confidence_map(gray, seg_lines)

    # ============ 可视化：把分割结果画回原图和灰度图 ============
    vis_on_bgr = draw_overlay(bgr, seg_lines)        # 原图 + 分割线
    vis_on_gray = draw_overlay(gray_bgr, seg_lines)  # 灰度 + 分割线
    conf_color = colorize_confidence(conf_map)       # 伪彩置信图

    # 保存结果
    os.makedirs(SAVE_DIR, exist_ok=True)
    outputs = {
        "ecm_gray.png": gray,                    # 灰度图
        "ecm_canny.png": edges,                  # Canny 边缘
        "ecm_white_mask.png": white_mask,        # white mask
        "ecm_confidence_map.png": conf_map,      # 置信边图（灰度 0/85/255）
        "ecm_overlay_bgr.png": vis_on_bgr,       # 原图 + 分割线
        "ecm_overlay_gray.png": vis_on_gray,     # 灰度 + 分割线
        "ecm_confidence_color.png": conf_color,  # 伪彩置信图
    }
    for name, img in outputs.items():
        path = os.path.join(SAVE_DIR, name)
        cv2.imwrite(path, img)
        print("已保存 %-28s -> %s" % (name, path))

    # 四联对比图：原图 | 原图+分割线 | 灰度+分割线 | 伪彩置信图
    panel = np.hstack([bgr, vis_on_bgr, vis_on_gray, conf_color])
    out_panel = os.path.join(SAVE_DIR, "ecm_panel.png")
    cv2.imwrite(out_panel, panel)
    print("  四联对比图    ->", out_panel)

    if SHOW_WINDOW:
        cv2.imshow("panel (press any key to exit)", cv2.resize(
            panel, None, fx=PANEL_SCALE, fy=PANEL_SCALE))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
