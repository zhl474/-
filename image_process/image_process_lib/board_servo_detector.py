"""低位托盘中心 ROI 圆点与半格目标识别。"""

import cv2
import numpy as np

from image_process_lib.board_debug import draw_low_board_roi_debug, make_low_board_debug_panel


BOARD_ROW_COUNT = 14
BOARD_COL_COUNT = 10
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


def detect_nearest_board_dot_in_roi(
    img,
    center_point,
    roi_half_size=LOW_BOARD_ROI_HALF_SIZE,
    blackhat_kernel_size=LOW_BOARD_BLACKHAT_KERNEL_SIZE,
    min_area=LOW_BOARD_MIN_DOT_AREA,
    max_area=LOW_BOARD_MAX_DOT_AREA,
    min_circularity=LOW_BOARD_MIN_DOT_CIRCULARITY,
    max_aspect_ratio=LOW_BOARD_MAX_DOT_ASPECT_RATIO,
    row=None,
    col=None,
    debug_enabled=True,
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
        debug_image = None
        if debug_enabled:
            debug_image = img.copy()
            _put_chinese_text(debug_image, "低位ROI为空，无法检测托盘圆点", (20, 30), (0, 0, 255))
        return {
            "found": False,
            "point": None,
            "candidates": [],
            "debug_image": debug_image,
            "debug_panel": None,
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
    debug_image = None
    debug_panel = None
    if debug_enabled:
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
