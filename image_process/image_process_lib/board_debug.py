"""低位托盘识别调试图与阶段拼图。"""

import os

import cv2
import numpy as np


def _put_chinese_text(image, text, org, color, font_size=22):
    """在调试图上写中文；Pillow 不可用时退回 OpenCV 英文渲染能力。"""
    try:
        from PIL import Image, ImageDraw, ImageFont

        font_candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        font = None
        for font_path in font_candidates:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        if font is None:
            font = ImageFont.load_default()

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(pil_image)
        b, g, r = color
        draw.text(org, text, font=font, fill=(r, g, b))
        image[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(
            image,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )


def _to_bgr(image):
    """把灰度图或透明图统一转成 BGR，便于拼接视频帧。"""
    if image is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def _fit_image_to_cell(image, cell_w, cell_h):
    """等比例缩放图像并放进固定大小单元格，保持视频帧尺寸稳定。"""
    image = _to_bgr(image)
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

    scale = min(float(cell_w) / float(w), float(cell_h) / float(h))
    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    x = (cell_w - resized_w) // 2
    y = (cell_h - resized_h) // 2
    canvas[y:y + resized_h, x:x + resized_w] = resized
    return canvas


def _make_panel_cell(title, image, cell_w=360, cell_h=260, title_h=34):
    """生成带中文标题的托盘调试拼图单元。"""
    cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    cell[:title_h, :] = (35, 35, 35)
    _put_chinese_text(cell, title, (10, 6), (255, 255, 255), font_size=22)
    cell[title_h:, :] = _fit_image_to_cell(image, cell_w, cell_h - title_h)
    cv2.rectangle(cell, (0, 0), (cell_w - 1, cell_h - 1), (80, 80, 80), 1)
    return cell


def _draw_low_board_roi_local_debug(
    roi,
    threshold_img,
    candidates,
    selected_candidate,
    roi_offset,
    selected_candidates=None,
):
    """在 ROI 局部坐标上画候选圆点，便于检查筛选条件。"""
    debug = _to_bgr(roi)
    offset_x, offset_y = roi_offset
    selected_candidates = selected_candidates or []
    for candidate in candidates:
        x, y, width, height = candidate["bbox"]
        local_x = int(round(float(x) - float(offset_x)))
        local_y = int(round(float(y) - float(offset_y)))
        cv2.rectangle(debug, (local_x, local_y), (local_x + int(width), local_y + int(height)), (0, 180, 0), 1)
        px = int(round(float(candidate["px"]) - float(offset_x)))
        py = int(round(float(candidate["py"]) - float(offset_y)))
        cv2.circle(debug, (px, py), 4, (0, 255, 0), 1)

    for candidate in selected_candidates:
        px = int(round(float(candidate["px"]) - float(offset_x)))
        py = int(round(float(candidate["py"]) - float(offset_y)))
        cv2.circle(debug, (px, py), 8, (0, 255, 255), 2)

    if selected_candidate is not None:
        px = int(round(float(selected_candidate["px"]) - float(offset_x)))
        py = int(round(float(selected_candidate["py"]) - float(offset_y)))
        cv2.drawMarker(
            debug,
            (px, py),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=22,
            thickness=2,
        )

    threshold_debug = _to_bgr(threshold_img)
    blended = cv2.addWeighted(debug, 0.65, threshold_debug, 0.35, 0)
    return blended


def make_low_board_debug_panel(
    image,
    roi_bounds,
    roi,
    gray,
    blackhat,
    threshold_img,
    candidates,
    selected_candidate,
    final_debug,
    message="",
    selected_candidates=None,
):
    """把低位托盘伺服关键阶段拼成一帧视频图。"""
    x1, y1, x2, y2 = roi_bounds
    roi_source = image.copy() if image is not None else None
    if roi_source is not None:
        cv2.rectangle(roi_source, (x1, y1), (x2, y2), (0, 255, 255), 2)

    candidates_debug = _draw_low_board_roi_local_debug(
        roi,
        threshold_img,
        candidates,
        selected_candidate,
        (x1, y1),
        selected_candidates=selected_candidates,
    )
    cells = [
        _make_panel_cell("1 原图与中心ROI", roi_source),
        _make_panel_cell("2 裁剪ROI原图", roi),
        _make_panel_cell("3 ROI灰度图", gray),
        _make_panel_cell("4 黑帽增强图", blackhat),
        _make_panel_cell("5 Otsu二值与候选", candidates_debug),
        _make_panel_cell("6 最终选点结果", final_debug),
    ]
    top = np.hstack(cells[:3])
    bottom = np.hstack(cells[3:])
    panel = np.vstack([top, bottom])
    if message:
        _put_chinese_text(panel, message, (12, panel.shape[0] - 30), (0, 255, 255), font_size=22)
    return panel


def draw_low_board_roi_debug(
    image,
    roi_bounds,
    center_point,
    candidates,
    selected_candidate=None,
    selected_candidates=None,
    row=None,
    col=None,
    target_mode_label="",
):
    """画低位托盘 ROI、候选圆点、选中圆点和中文调试信息。"""
    debug_image = image.copy()
    x1, y1, x2, y2 = roi_bounds
    center_x, center_y = center_point
    selected_candidates = selected_candidates or []
    cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.drawMarker(
        debug_image,
        (int(round(center_x)), int(round(center_y))),
        (255, 0, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=2,
    )

    for candidate in candidates:
        x, y, width, height = candidate["bbox"]
        px, py = candidate["px"], candidate["py"]
        cv2.rectangle(debug_image, (int(x), int(y)), (int(x + width), int(y + height)), (0, 180, 0), 1)
        cv2.circle(debug_image, (int(round(px)), int(round(py))), 4, (0, 255, 0), 1)

    for candidate in selected_candidates:
        px, py = candidate["px"], candidate["py"]
        cv2.circle(debug_image, (int(round(px)), int(round(py))), 8, (0, 255, 255), 2)

    dx_px = 0.0
    dy_px = 0.0
    if selected_candidate is not None:
        px, py = selected_candidate["px"], selected_candidate["py"]
        dx_px = float(px - center_x)
        dy_px = float(py - center_y)
        cv2.drawMarker(
            debug_image,
            (int(round(px)), int(round(py))),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )
        cv2.line(
            debug_image,
            (int(round(center_x)), int(round(center_y))),
            (int(round(px)), int(round(py))),
            (255, 0, 0),
            1,
        )

    row_col_text = ""
    if row is not None and col is not None:
        row_col_text = f" 目标行列=({float(row):.2f},{float(col):.2f})"
    mode_text = f" 模式={target_mode_label}" if target_mode_label else ""
    _put_chinese_text(debug_image, f"低位托盘ROI 候选数量={len(candidates)}{row_col_text}{mode_text}", (20, 30), (255, 0, 0))
    _put_chinese_text(debug_image, f"像素误差 dx={dx_px:.1f} dy={dy_px:.1f}", (20, 60), (255, 0, 0))
    return debug_image

