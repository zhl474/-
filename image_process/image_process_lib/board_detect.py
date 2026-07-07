import cv2
import numpy as np
from ultralytics import YOLO


BOARD_MODEL_PATH = "/home/zhl/SingleArmTetris/SingleArmTetris/src/competition/model/best.pt"
BOARD_ROW_COUNT = 14
BOARD_COL_COUNT = 10
BOARD_POINT_COUNT = BOARD_ROW_COUNT * BOARD_COL_COUNT
DEFAULT_DEBUG_PATH = "/home/zhl/桌面/托盘格点粗定位.jpg"
LOW_BOARD_ROI_HALF_SIZE = 120
LOW_BOARD_BLACKHAT_KERNEL_SIZE = 15
LOW_BOARD_MIN_DOT_AREA = 20
LOW_BOARD_MAX_DOT_AREA = 250
LOW_BOARD_MIN_DOT_CIRCULARITY = 0.35
LOW_BOARD_MAX_DOT_ASPECT_RATIO = 1.8
LOW_BOARD_HALF_GRID_TOLERANCE = 1e-3

LOW_BOARD_TARGET_MODE_LABELS = {
    "dot": "单点",
    "horizontal_mid": "左右中点",
    "vertical_mid": "上下中点",
    "cell_center": "四点中心",
}

_board_model = None


class BoardGridDetectionError(RuntimeError):
    """托盘格点识别失败。"""


def _get_board_model():
    """延迟加载托盘模型，避免每次服务调用都重新读取权重。"""
    global _board_model
    if _board_model is None:
        _board_model = YOLO(BOARD_MODEL_PATH)
    return _board_model


def _save_debug_image(image_path, image):
    """保存调试图像，兼容中文路径。"""
    if not image_path or image is None:
        return False
    try:
        dot_index = image_path.rfind(".")
        image_ext = image_path[dot_index:] if dot_index >= 0 else ".jpg"
        ok, encoded_image = cv2.imencode(image_ext, image)
        if not ok:
            return False
        encoded_image.tofile(image_path)
        return True
    except Exception:
        return False


def _red_error(message):
    """用红色输出现场必须处理的托盘识别错误。"""
    print(f"\033[91m{message}\033[0m")


def _make_odd_kernel_size(kernel_size):
    """把形态学核尺寸规整成大于等于 3 的奇数。"""
    kernel_size = int(round(float(kernel_size)))
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def _clip_roi_bounds(image_shape, center_point, roi_half_size):
    """按全图中心点裁剪 ROI 边界，靠近图像边缘时自动截断。"""
    h, w = image_shape[:2]
    cx, cy = center_point
    half_size = max(1, int(round(float(roi_half_size))))
    x1 = max(0, int(round(float(cx))) - half_size)
    y1 = max(0, int(round(float(cy))) - half_size)
    x2 = min(w, int(round(float(cx))) + half_size)
    y2 = min(h, int(round(float(cy))) + half_size)
    return x1, y1, x2, y2


def _put_chinese_text(image, text, org, color, font_size=22):
    """在调试图上写中文；Pillow 不可用时退回 OpenCV 英文渲染能力。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import os

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


def _make_blob_detector():
    """创建托盘格点 blob 检测器，用于检测白色圆点。"""
    params = cv2.SimpleBlobDetector_Params()

    # 二值图里格点颜色
    params.filterByColor = True
    params.blobColor = 0

    # 面积过滤
    params.filterByArea = True
    params.minArea = 25
    params.maxArea = 200

    # 圆度过滤
    params.filterByCircularity = True
    params.minCircularity = 0.3

    # 这两个默认可能会误杀一些不完美圆点，建议关掉
    params.filterByInertia = False
    params.filterByConvexity = False

    # 点间距明显大于 10，这个可以保留
    params.minDistBetweenBlobs = 8

    return cv2.SimpleBlobDetector_create(params)


def _extract_low_board_dot_candidates(
    threshold_img,
    roi_offset,
    roi_center,
    min_area,
    max_area,
    min_circularity,
    max_aspect_ratio,
):
    """从低位 ROI 二值图里提取形状接近圆点的候选连通域。"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        threshold_img,
        8,
    )
    candidates = []
    offset_x, offset_y = roi_offset
    center_x, center_y = roi_center

    for label in range(1, num_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if width <= 1 or height <= 1:
            continue

        aspect_ratio = max(width, height) / float(min(width, height))
        if aspect_ratio > max_aspect_ratio:
            continue

        component_mask = np.zeros_like(threshold_img, dtype=np.uint8)
        component_mask[labels == label] = 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1e-6:
            continue
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        if circularity < min_circularity:
            continue

        local_x, local_y = centroids[label]
        px = float(local_x + offset_x)
        py = float(local_y + offset_y)
        distance = float((local_x - center_x) ** 2 + (local_y - center_y) ** 2)
        candidates.append(
            {
                "px": px,
                "py": py,
                "area": area,
                "bbox": (x + offset_x, y + offset_y, width, height),
                "circularity": circularity,
                "distance": distance,
            }
        )

    return candidates


def _normalize_half_grid_value(value, name):
    """把目标行列规整到整数或 .5；其它小数直接拒绝。"""
    value = float(value)
    normalized = round(value * 2.0) / 2.0
    if abs(value - normalized) > LOW_BOARD_HALF_GRID_TOLERANCE:
        raise ValueError(f"低位托盘目标{name}只支持整数或 .5，当前为 {value}")
    return normalized


def _parse_low_board_target_mode(row, col):
    """根据 row/col 的小数部分判断低位托盘目标模式。"""
    if row is None or col is None:
        return {
            "mode": "dot",
            "label": LOW_BOARD_TARGET_MODE_LABELS["dot"],
            "row": None,
            "col": None,
            "r0": None,
            "c0": None,
            "fr": 0.0,
            "fc": 0.0,
        }

    row = _normalize_half_grid_value(row, "行")
    col = _normalize_half_grid_value(col, "列")
    if row < 1.0 or row > BOARD_ROW_COUNT or col < 1.0 or col > BOARD_COL_COUNT:
        raise ValueError(f"低位托盘目标超出范围: row={row}, col={col}")

    r0 = int(np.floor(row))
    c0 = int(np.floor(col))
    fr = float(row - r0)
    fc = float(col - c0)

    if fr == 0.0 and fc == 0.0:
        mode = "dot"
    elif fr == 0.0 and fc == 0.5:
        mode = "horizontal_mid"
    elif fr == 0.5 and fc == 0.0:
        mode = "vertical_mid"
    elif fr == 0.5 and fc == 0.5:
        mode = "cell_center"
    else:
        raise ValueError(f"低位托盘目标只支持整数或 .5: row={row}, col={col}")

    if mode in ("vertical_mid", "cell_center") and r0 + 1 > BOARD_ROW_COUNT:
        raise ValueError(f"低位托盘目标需要下一行，但 row={row} 已到边界")
    if mode in ("horizontal_mid", "cell_center") and c0 + 1 > BOARD_COL_COUNT:
        raise ValueError(f"低位托盘目标需要下一列，但 col={col} 已到边界")

    return {
        "mode": mode,
        "label": LOW_BOARD_TARGET_MODE_LABELS[mode],
        "row": row,
        "col": col,
        "r0": r0,
        "c0": c0,
        "fr": fr,
        "fc": fc,
    }


def _candidate_center_distance(candidate, center_point):
    """计算候选圆点到图像中心的平方距离。"""
    center_x, center_y = center_point
    return float((candidate["px"] - center_x) ** 2 + (candidate["py"] - center_y) ** 2)


def _pick_nearest_candidate(candidates, center_point):
    """从候选圆点中选离图像中心最近的一个。"""
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: _candidate_center_distance(candidate, center_point))


def _average_candidate_points(candidates):
    """取多个候选圆点中心的平均值，作为虚拟摆放目标点。"""
    points = np.array([[candidate["px"], candidate["py"]] for candidate in candidates], dtype=np.float32)
    return np.mean(points, axis=0)


def _make_virtual_target_candidate(point):
    """把虚拟目标点包装成和候选圆点相同的调试绘制结构。"""
    return {
        "px": float(point[0]),
        "py": float(point[1]),
        "area": 0.0,
        "circularity": 0.0,
        "bbox": (float(point[0]), float(point[1]), 0, 0),
    }


def _format_missing_low_board_neighbors(missing_labels):
    """生成缺邻点的中文错误信息。"""
    if not missing_labels:
        return ""
    return "未找到" + "、".join(missing_labels) + "邻点"


def _select_low_board_target(candidates, center_point, target_info):
    """按目标模式从低位候选圆点里选择真实邻点，并计算最终目标点。"""
    if not candidates:
        return {
            "found": False,
            "point": None,
            "selected_candidate": None,
            "selected_candidates": [],
            "message": "低位 ROI 内未检测到托盘圆点",
        }

    center_x, center_y = center_point
    mode = target_info["mode"]

    if mode == "dot":
        selected = _pick_nearest_candidate(candidates, center_point)
        point = np.array([selected["px"], selected["py"]], dtype=np.float32)
        return {
            "found": True,
            "point": point,
            "selected_candidate": selected,
            "selected_candidates": [selected],
            "message": "低位托盘单点识别成功",
        }

    if mode == "horizontal_mid":
        left = _pick_nearest_candidate([candidate for candidate in candidates if candidate["px"] < center_x], center_point)
        right = _pick_nearest_candidate([candidate for candidate in candidates if candidate["px"] > center_x], center_point)
        missing = []
        if left is None:
            missing.append("左侧")
        if right is None:
            missing.append("右侧")
        if missing:
            return {
                "found": False,
                "point": None,
                "selected_candidate": None,
                "selected_candidates": [candidate for candidate in (left, right) if candidate is not None],
                "message": _format_missing_low_board_neighbors(missing),
            }
        selected_candidates = [left, right]
        point = _average_candidate_points(selected_candidates)
        return {
            "found": True,
            "point": point,
            "selected_candidate": _make_virtual_target_candidate(point),
            "selected_candidates": selected_candidates,
            "message": "低位托盘左右中点识别成功",
        }

    if mode == "vertical_mid":
        top = _pick_nearest_candidate([candidate for candidate in candidates if candidate["py"] < center_y], center_point)
        bottom = _pick_nearest_candidate([candidate for candidate in candidates if candidate["py"] > center_y], center_point)
        missing = []
        if top is None:
            missing.append("上方")
        if bottom is None:
            missing.append("下方")
        if missing:
            return {
                "found": False,
                "point": None,
                "selected_candidate": None,
                "selected_candidates": [candidate for candidate in (top, bottom) if candidate is not None],
                "message": _format_missing_low_board_neighbors(missing),
            }
        selected_candidates = [top, bottom]
        point = _average_candidate_points(selected_candidates)
        return {
            "found": True,
            "point": point,
            "selected_candidate": _make_virtual_target_candidate(point),
            "selected_candidates": selected_candidates,
            "message": "低位托盘上下中点识别成功",
        }

    quadrant_specs = [
        ("左上", lambda candidate: candidate["px"] < center_x and candidate["py"] < center_y),
        ("右上", lambda candidate: candidate["px"] > center_x and candidate["py"] < center_y),
        ("左下", lambda candidate: candidate["px"] < center_x and candidate["py"] > center_y),
        ("右下", lambda candidate: candidate["px"] > center_x and candidate["py"] > center_y),
    ]
    selected_candidates = []
    missing = []
    for label, predicate in quadrant_specs:
        selected = _pick_nearest_candidate([candidate for candidate in candidates if predicate(candidate)], center_point)
        if selected is None:
            missing.append(label)
        else:
            selected_candidates.append(selected)
    if missing:
        return {
            "found": False,
            "point": None,
            "selected_candidate": None,
            "selected_candidates": selected_candidates,
            "message": _format_missing_low_board_neighbors(missing),
        }

    point = _average_candidate_points(selected_candidates)
    return {
        "found": True,
        "point": point,
        "selected_candidate": _make_virtual_target_candidate(point),
        "selected_candidates": selected_candidates,
        "message": "低位托盘四点中心识别成功",
    }


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


def detect_nearest_board_dot_in_roi(
    img,
    center_point,
    roi_half_size=LOW_BOARD_ROI_HALF_SIZE,
    blackhat_kernel_size=LOW_BOARD_BLACKHAT_KERNEL_SIZE,
    min_area=LOW_BOARD_MIN_DOT_AREA,
    max_area=LOW_BOARD_MAX_DOT_AREA,
    min_circularity=LOW_BOARD_MIN_DOT_CIRCULARITY,
    max_aspect_ratio=LOW_BOARD_MAX_DOT_ASPECT_RATIO,
    debug_path=None,
    row=None,
    col=None,
):
    """低位托盘视觉伺服：在中心 ROI 内按 row/col 选择单点或虚拟中点。"""
    if img is None:
        return {
            "found": False,
            "point": None,
            "candidates": [],
            "debug_image": None,
            "message": "没有可用图像",
            "count": 0,
        }

    target_info = _parse_low_board_target_mode(row, col)
    roi_bounds = _clip_roi_bounds(img.shape, center_point, roi_half_size)
    x1, y1, x2, y2 = roi_bounds
    if x2 <= x1 or y2 <= y1:
        debug_image = img.copy()
        _put_chinese_text(debug_image, "低位ROI为空，无法检测托盘圆点", (20, 30), (0, 0, 255))
        _save_debug_image(debug_path, debug_image)
        return {
            "found": False,
            "point": None,
            "candidates": [],
            "debug_image": debug_image,
            "message": "低位 ROI 为空，无法检测托盘圆点",
            "count": 0,
        }

    roi = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    kernel_size = _make_odd_kernel_size(blackhat_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, threshold_img = cv2.threshold(
        blackhat,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    roi_center = (float(center_point[0] - x1), float(center_point[1] - y1))
    candidates = _extract_low_board_dot_candidates(
        threshold_img,
        (x1, y1),
        roi_center,
        float(min_area),
        float(max_area),
        float(min_circularity),
        float(max_aspect_ratio),
    )
    target_selection = _select_low_board_target(candidates, center_point, target_info)
    selected_candidate = target_selection["selected_candidate"]
    selected_candidates = target_selection["selected_candidates"]
    debug_image = draw_low_board_roi_debug(
        img,
        roi_bounds,
        center_point,
        candidates,
        selected_candidate=selected_candidate,
        selected_candidates=selected_candidates,
        row=row,
        col=col,
        target_mode_label=target_info["label"],
    )
    row_col_text = f"目标行列=({float(row):.2f},{float(col):.2f})" if row is not None and col is not None else ""
    if selected_candidate is None:
        selected_text = "未选中目标"
    elif target_info["mode"] == "dot":
        selected_text = f"选中圆点 面积={selected_candidate['area']:.1f} 圆度={selected_candidate['circularity']:.2f}"
    else:
        selected_text = f"选中{target_info['label']} 邻点数={len(selected_candidates)}"
    debug_panel = make_low_board_debug_panel(
        img,
        roi_bounds,
        roi,
        gray,
        blackhat,
        threshold_img,
        candidates,
        selected_candidate,
        debug_image,
        message=f"候选数量={len(candidates)} {selected_text} {row_col_text} 模式={target_info['label']}",
        selected_candidates=selected_candidates,
    )
    _save_debug_image(debug_path, debug_image)

    if not target_selection["found"]:
        return {
            "found": False,
            "point": None,
            "candidates": candidates,
            "debug_image": debug_image,
            "debug_panel": debug_panel,
            "message": target_selection["message"],
            "count": 0,
            "target_mode": target_info["mode"],
            "target_mode_label": target_info["label"],
            "selected_candidates": selected_candidates,
            "blackhat_image": blackhat,
            "threshold_image": threshold_img,
            "roi_bounds": roi_bounds,
        }

    point = np.array(target_selection["point"], dtype=np.float32)
    return {
        "found": True,
        "point": point,
        "candidates": candidates,
        "debug_image": debug_image,
        "message": target_selection["message"],
        "count": len(candidates),
        "selected": selected_candidate,
        "selected_candidates": selected_candidates,
        "target_mode": target_info["mode"],
        "target_mode_label": target_info["label"],
        "debug_panel": debug_panel,
        "blackhat_image": blackhat,
        "threshold_image": threshold_img,
        "roi_bounds": roi_bounds,
    }


def _get_board_crop(img):
    """用 YOLO OBB 找托盘，并透视裁剪到托盘局部图。"""
    model = _get_board_model()
    results = model(img)
    if not results:
        raise BoardGridDetectionError("未检测到托盘")

    result = results[0]
    if result.obb is None or len(result.obb) == 0:
        raise BoardGridDetectionError("未检测到托盘 OBB")

    obb_points = result.obb.xyxyxyxy.cpu().numpy()[0]
    pts = obb_points.reshape(4, 2).astype(np.float32)

    width = int(
        max(
            np.linalg.norm(pts[0] - pts[1]),
            np.linalg.norm(pts[2] - pts[3]),
        )
    )
    height = int(
        max(
            np.linalg.norm(pts[1] - pts[2]),
            np.linalg.norm(pts[3] - pts[0]),
        )
    )
    if width <= 1 or height <= 1:
        raise BoardGridDetectionError("托盘 OBB 尺寸异常")

    dst_pts = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(pts, dst_pts)
    crop_img = cv2.warpPerspective(img, matrix, (width, height))
    return crop_img, matrix


def _detect_grid_keypoints(crop_img):
    """在托盘裁剪图中检测 140 个圆形格点。"""
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    # cv2.imshow("1",blackhat)
    _, threshold_img = cv2.threshold(
        blackhat,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    # cv2.imshow("2",threshold_img)
    detector = _make_blob_detector()
    keypoints = detector.detect(gray)
    debug_image = cv2.drawKeypoints(
        crop_img,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return keypoints, debug_image, threshold_img


def _transform_points_to_origin(points, matrix):
    """把裁剪图格点坐标反变换回原始图像坐标。"""
    matrix_inv = np.linalg.inv(matrix)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    orig_points = cv2.perspectiveTransform(points, matrix_inv)
    return orig_points.reshape(-1, 2)


def _sort_points_to_grid(crop_points, orig_points):
    """把 140 个格点排序成 grid_points[row][col] 的 1-based 表。"""
    points = np.asarray(crop_points, dtype=np.float32)
    if len(points) != BOARD_POINT_COUNT:
        raise BoardGridDetectionError("托盘格点数量不是140个，无法排序")

    mean, eigenvectors = cv2.PCACompute(points, mean=None)
    center = mean[0]
    axis_a = eigenvectors[0]
    axis_b = eigenvectors[1]
    proj_a = np.dot(points - center, axis_a)
    proj_b = np.dot(points - center, axis_b)

    # 14 行方向通常是托盘长轴；用投影范围选择行轴。
    extent_a = float(np.max(proj_a) - np.min(proj_a))
    extent_b = float(np.max(proj_b) - np.min(proj_b))
    if extent_a >= extent_b:
        row_proj, col_proj = proj_a, proj_b
    else:
        row_proj, col_proj = proj_b, proj_a

    sorted_indices = np.argsort(row_proj)
    row_groups = [
        sorted_indices[i * BOARD_COL_COUNT:(i + 1) * BOARD_COL_COUNT]
        for i in range(BOARD_ROW_COUNT)
    ]

    # row=1 约定为画面中更靠下的一行，也就是左下角原点。
    first_row_y = float(np.mean(orig_points[row_groups[0], 1]))
    last_row_y = float(np.mean(orig_points[row_groups[-1], 1]))

    # 图像坐标系 y 越大越靠下。
    # 如果当前第一组在上面，就反转，让最下面那一行排到 row=1。
    if first_row_y < last_row_y:
        row_groups.reverse()

    grid_points = [[None for _ in range(BOARD_COL_COUNT + 1)] for _ in range(BOARD_ROW_COUNT + 1)]
    ordered_rows = []
    for row_indices in row_groups:
        row_indices = np.array(row_indices, dtype=np.int32)
        col_sorted = row_indices[np.argsort(col_proj[row_indices])]

        # col=1 约定为画面中更靠左的一列。
        if float(orig_points[col_sorted[0], 0]) > float(orig_points[col_sorted[-1], 0]):
            col_sorted = col_sorted[::-1]
        ordered_rows.append(col_sorted)

    for row_idx, col_sorted in enumerate(ordered_rows, start=1):
        for col_idx, point_index in enumerate(col_sorted, start=1):
            px, py = orig_points[point_index]
            grid_points[row_idx][col_idx] = np.array([float(px), float(py)], dtype=np.float32)

    return grid_points


def interpolate_grid_point(grid_points, row, col):
    """按 1-based 行列读取托盘点；小数行列用相邻真实格点双线性插值。"""
    row = float(row)
    col = float(col)
    if row < 1.0 or row > BOARD_ROW_COUNT or col < 1.0 or col > BOARD_COL_COUNT:
        raise ValueError(f"托盘目标超出范围: row={row}, col={col}")

    row0 = int(np.floor(row))
    row1 = int(np.ceil(row))
    col0 = int(np.floor(col))
    col1 = int(np.ceil(col))
    row_t = row - row0
    col_t = col - col0

    p00 = grid_points[row0][col0]
    p01 = grid_points[row0][col1]
    p10 = grid_points[row1][col0]
    p11 = grid_points[row1][col1]
    top = p00 * (1.0 - col_t) + p01 * col_t
    bottom = p10 * (1.0 - col_t) + p11 * col_t
    return top * (1.0 - row_t) + bottom * row_t


def draw_grid_debug(image, grid_points, target_point=None, center_point=None):
    """在原图上画出 140 个格点、目标点和相机中心，便于现场确认。"""
    debug_image = image.copy()
    for row in range(1, BOARD_ROW_COUNT + 1):
        for col in range(1, BOARD_COL_COUNT + 1):
            px, py = grid_points[row][col]
            cv2.circle(debug_image, (int(round(px)), int(round(py))), 3, (0, 255, 0), -1)
            if row in (1, BOARD_ROW_COUNT) and col in (1, BOARD_COL_COUNT):
                cv2.putText(
                    debug_image,
                    f"{row},{col}",
                    (int(round(px)) + 4, int(round(py)) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

    if center_point is not None:
        cx, cy = center_point
        cv2.drawMarker(
            debug_image,
            (int(round(cx)), int(round(cy))),
            (255, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )

    if target_point is not None:
        px, py = target_point
        cv2.drawMarker(
            debug_image,
            (int(round(px)), int(round(py))),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )
        if center_point is not None:
            cv2.line(
                debug_image,
                (int(round(center_point[0])), int(round(center_point[1]))),
                (int(round(px)), int(round(py))),
                (255, 0, 0),
                1,
            )
    return debug_image


def board_grid_detect(img, debug_path=DEFAULT_DEBUG_PATH):
    """识别完整托盘 140 个格点，返回 1-based 行列表。"""
    if img is None:
        return {
            "found": False,
            "grid_points": None,
            "debug_image": None,
            "message": "没有可用图像",
            "count": 0,
        }

    try:
        crop_img, matrix = _get_board_crop(img)
        keypoints, crop_debug_image, _ = _detect_grid_keypoints(crop_img)
        point_count = len(keypoints)
        _save_debug_image(debug_path, crop_debug_image)

        if point_count != BOARD_POINT_COUNT:
            message = f"托盘格点识别数量不是140个，请重新识别；当前识别到{point_count}个"
            _red_error(message)
            return {
                "found": False,
                "grid_points": None,
                "debug_image": crop_debug_image,
                "message": message,
                "count": point_count,
            }

        crop_points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        orig_points = _transform_points_to_origin(crop_points, matrix)
        grid_points = _sort_points_to_grid(crop_points, orig_points)
        debug_image = draw_grid_debug(img, grid_points)
        _save_debug_image(debug_path, debug_image)
        return {
            "found": True,
            "grid_points": grid_points,
            "debug_image": debug_image,
            "message": "托盘140个格点识别成功",
            "count": point_count,
        }
    except Exception as exc:
        message = f"托盘格点识别失败: {exc}"
        _red_error(message)
        _save_debug_image(debug_path, img)
        return {
            "found": False,
            "grid_points": None,
            "debug_image": img.copy() if img is not None else None,
            "message": message,
            "count": 0,
        }


def board_detect(img):
    """兼容旧接口：从完整 140 格点表中取四角返回。"""
    result = board_grid_detect(img)
    if not result["found"]:
        raise BoardGridDetectionError(result["message"])

    grid_points = result["grid_points"]

    # 新坐标系：
    # [1,1] 是左下角
    # [14,1] 是左上角
    # [1,BOARD_COL_COUNT] 是右下角
    # [BOARD_ROW_COUNT,BOARD_COL_COUNT] 是右上角
    left_bottom = grid_points[1][1]
    left_top = grid_points[BOARD_ROW_COUNT][1]
    right_bottom = grid_points[1][BOARD_COL_COUNT]
    right_top = grid_points[BOARD_ROW_COUNT][BOARD_COL_COUNT]

    return np.array([left_top, left_bottom, right_top, right_bottom], dtype=np.float32)


if __name__ == "__main__":
    image_path = "/home/zhl/图片/数据集/15_Color.png"
    image = cv2.imread(image_path)
    detect_result = board_grid_detect(image)
    print(detect_result["grid_points"])
