import cv2
import numpy as np

from image_process_lib.block_detection import (
    coreect_LL_location,
    draw_mask_on_full_image,
    get_mask,
)
from image_process_lib.block_category import normalize_category_name
from image_process_lib.template_config import load_color_segmentation_config, load_template_geometry
from image_process_lib.template_match.kernels_create import get_template_rect_size
from image_process_lib.template_match.template_match import get_rect


def undistort_bgr_image(img_bgr, camera_matrix, dist_coeff):
    """Gemini335 已输出可直接使用的图像，这里保留接口但不再做去畸变。"""
    return img_bgr


def _empty_detection(message, debug_image=None, debug_panel=None):
    return {
        "found": False,
        "category": "",
        "px": 0.0,
        "py": 0.0,
        "theta": 0.0,
        "score": 0.0,
        "debug_image": debug_image,
        "debug_panel": debug_panel,
        "message": message,
    }


def _make_debug_image(img_bgr):
    """生成可画调试信息的图像，输入为空时返回空白占位。"""
    if img_bgr is None or img_bgr.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return np.copy(img_bgr)


def _clip_bbox_from_points(points, image_shape, expand_px=0):
    """根据多边形点生成裁剪框，自动限制在图像范围内。"""
    image_h, image_w = image_shape[:2]
    expand_px = int(round(float(expand_px)))
    x1 = max(0, int(np.floor(np.min(points[:, 0]))) - expand_px)
    y1 = max(0, int(np.floor(np.min(points[:, 1]))) - expand_px)
    x2 = min(image_w, int(np.ceil(np.max(points[:, 0]))) + expand_px + 1)
    y2 = min(image_h, int(np.ceil(np.max(points[:, 1]))) + expand_px + 1)
    return x1, y1, x2, y2


def _put_chinese_text(image, text, org, color, font_size=22):
    """在调试拼图上写中文标题；Pillow 不可用时退回 OpenCV。"""
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
    """把灰度或带透明通道的图统一转成 BGR。"""
    if image is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def _fit_image_to_cell(image, cell_w, cell_h):
    """等比例缩放图像并放入固定大小单元格，保证视频帧尺寸稳定。"""
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
    """生成带中文标题的单个调试单元格。"""
    cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    cell[:title_h, :] = (35, 35, 35)
    _put_chinese_text(cell, title, (10, 6), (255, 255, 255), font_size=22)
    cell[title_h:, :] = _fit_image_to_cell(image, cell_w, cell_h - title_h)
    cv2.rectangle(cell, (0, 0), (cell_w - 1, cell_h - 1), (80, 80, 80), 1)
    return cell


def _make_block_debug_panel(
    original_debug,
    roi_seed_debug=None,
    raw_mask=None,
    final_mask=None,
    match_mask_debug=None,
    match_debug=None,
    message="",
):
    """把低位方块视觉伺服关键阶段拼成一帧视频图。"""
    cells = [
        _make_panel_cell("1 原图与先验ROI", original_debug),
        _make_panel_cell("2 ROI与seed取色范围", roi_seed_debug),
        _make_panel_cell("3 RGB原始二值", raw_mask),
        _make_panel_cell("4 最终匹配mask", final_mask),
        _make_panel_cell("5 二值图模板匹配", match_mask_debug),
        _make_panel_cell("6 模板匹配结果", match_debug),
    ]
    top = np.hstack(cells[:3])
    bottom = np.hstack(cells[3:])
    panel = np.vstack([top, bottom])
    if message:
        _put_chinese_text(panel, message, (12, panel.shape[0] - 30), (0, 255, 255), font_size=22)
    return panel


def _segment_roi_by_local_rgb_color(roi_bgr, category, return_stages=False):
    """在低位 ROI 内用局部 RGB 颜色种子分割目标方块。"""
    if roi_bgr is None or roi_bgr.size == 0:
        raise ValueError("低位 ROI 为空，无法进行 RGB 颜色分割")

    color_config = load_color_segmentation_config(category)
    seed_search_half_size = color_config["seed_search_half_size"]
    seed_patch_size = color_config["seed_patch_size"]
    seed_stride = color_config["seed_stride"]
    local_dist_thresh = color_config["local_dist_thresh"]
    r_prior, g_prior, b_prior = color_config["rgb"]

    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    roi_h, roi_w = roi_rgb.shape[:2]
    center_x = roi_w / 2.0
    center_y = roi_h / 2.0
    search_x1 = max(0, int(np.floor(center_x - seed_search_half_size)))
    search_y1 = max(0, int(np.floor(center_y - seed_search_half_size)))
    search_x2 = min(roi_w, int(np.ceil(center_x + seed_search_half_size + 1)))
    search_y2 = min(roi_h, int(np.ceil(center_y + seed_search_half_size + 1)))

    if search_x2 - search_x1 < seed_patch_size or search_y2 - search_y1 < seed_patch_size:
        raise ValueError(
            "低位 ROI 中心搜索窗口过小，无法枚举 seed patch: "
            f"roi={roi_w}x{roi_h}, "
            f"window=({search_x1},{search_y1})-({search_x2},{search_y2}), "
            f"seed_patch_size={seed_patch_size}"
        )

    best_patch = None
    best_d2 = None
    for patch_y in range(search_y1, search_y2 - seed_patch_size + 1, seed_stride):
        for patch_x in range(search_x1, search_x2 - seed_patch_size + 1, seed_stride):
            patch = roi_rgb[patch_y:patch_y + seed_patch_size, patch_x:patch_x + seed_patch_size]
            r_mean, g_mean, b_mean = patch.reshape(-1, 3).mean(axis=0)
            d2 = (
                (r_mean - r_prior) ** 2
                + (g_mean - g_prior) ** 2
                + (b_mean - b_prior) ** 2
            )
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_patch = {
                    "x": patch_x,
                    "y": patch_y,
                    "mean": (float(r_mean), float(g_mean), float(b_mean)),
                }

    if best_patch is None:
        raise ValueError(
            "低位 ROI 中心搜索窗口无法枚举任何 seed patch: "
            f"roi={roi_w}x{roi_h}, "
            f"window=({search_x1},{search_y1})-({search_x2},{search_y2}), "
            f"seed_patch_size={seed_patch_size}, seed_stride={seed_stride}"
        )

    r0, g0, b0 = best_patch["mean"]
    d2_map = (
        (roi_rgb[:, :, 0] - r0) ** 2
        + (roi_rgb[:, :, 1] - g0) ** 2
        + (roi_rgb[:, :, 2] - b0) ** 2
    )
    raw_foreground = (d2_map < local_dist_thresh ** 2).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    morph_foreground = cv2.morphologyEx(raw_foreground, cv2.MORPH_OPEN, kernel)
    morph_foreground = cv2.morphologyEx(morph_foreground, cv2.MORPH_CLOSE, kernel)
    foreground = morph_foreground

    seed_center_x = best_patch["x"] + seed_patch_size / 2.0
    seed_center_y = best_patch["y"] + seed_patch_size / 2.0
    print(
        "低位RGB颜色分割: "
        f"category={category}, "
        f"seed_center=({seed_center_x:.1f}, {seed_center_y:.1f}), "
        f"local_color=[{r0:.1f}, {g0:.1f}, {b0:.1f}], "
        f"local_dist_thresh={local_dist_thresh:.1f}"
    )

    if return_stages:
        return foreground, {
            "raw_mask": raw_foreground,
            "morph_mask": morph_foreground,
            "filled_mask": foreground,
            "seed_center": (seed_center_x, seed_center_y),
            "seed_search_box": (search_x1, search_y1, search_x2, search_y2),
            "seed_patch_box": (
                best_patch["x"],
                best_patch["y"],
                best_patch["x"] + seed_patch_size,
                best_patch["y"] + seed_patch_size,
            ),
            "local_color": (r0, g0, b0),
            "local_dist_thresh": local_dist_thresh,
            "seed_d2": float(best_d2),
        }
    return foreground


def _draw_seed_patch_debug(roi_bgr, mask_stages):
    """在 ROI 原图上标出中心搜索窗口和最终 seed patch。"""
    debug = _make_debug_image(roi_bgr)
    search_box = mask_stages.get("seed_search_box")
    patch_box = mask_stages.get("seed_patch_box")
    seed_center = mask_stages.get("seed_center")
    local_color = mask_stages.get("local_color")
    local_dist_thresh = mask_stages.get("local_dist_thresh")

    if search_box is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in search_box]
        cv2.rectangle(debug, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 1)

    if patch_box is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in patch_box]
        cv2.rectangle(debug, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 2)

    if seed_center is not None:
        cv2.drawMarker(
            debug,
            (int(round(seed_center[0])), int(round(seed_center[1]))),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )

    if local_color is not None:
        r0, g0, b0 = local_color
        text = f"seed RGB=[{r0:.0f},{g0:.0f},{b0:.0f}]"
        _put_chinese_text(debug, text, (8, 8), (0, 255, 255), font_size=20)
    if local_dist_thresh is not None:
        _put_chinese_text(debug, f"阈值={float(local_dist_thresh):.1f}", (8, 36), (0, 255, 255), font_size=20)

    return debug


def _draw_template_match_on_mask(mask, match_debug_output):
    """在模板匹配实际输入的二值图上画出最佳模板轮廓。"""
    debug = _to_bgr(mask)
    if not match_debug_output:
        return debug

    best_kernel = match_debug_output.get("best_kernel")
    template_top_left = match_debug_output.get("template_top_left")
    match_center = match_debug_output.get("match_center")
    angle = match_debug_output.get("angle")
    score = match_debug_output.get("score")
    if best_kernel is None or template_top_left is None:
        return debug

    contours, _ = cv2.findContours(best_kernel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    offset = (
        int(round(float(template_top_left[0]))),
        int(round(float(template_top_left[1]))),
    )
    cv2.drawContours(debug, contours, -1, (0, 255, 0), 1, offset=offset)

    if match_center is not None:
        cv2.drawMarker(
            debug,
            (int(round(float(match_center[0]))), int(round(float(match_center[1])))),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )

    if angle is not None and score is not None:
        _put_chinese_text(debug, f"角度={float(angle):.1f} 分数={float(score):.1f}", (8, 8), (0, 255, 255), font_size=20)

    return debug


def _format_seed_debug_message(category, foreground_area, mask_stages, prefix=""):
    """生成低位方块 debug 面板底部摘要。"""
    local_color = mask_stages.get("local_color", (0.0, 0.0, 0.0))
    local_dist_thresh = float(mask_stages.get("local_dist_thresh", 0.0))
    seed_rgb = [int(round(float(value))) for value in local_color]
    prefix_text = f"{prefix} " if prefix else ""
    return f"{prefix_text}类别={category} 面积={foreground_area} seedRGB={seed_rgb} 阈值={local_dist_thresh:.1f}"



def _draw_prior_roi_debug(debug_image, roi_box, match_point=None, category="", theta=0.0):
    """绘制低位先验 ROI 和模板匹配结果。"""
    box = np.intp(roi_box)
    cv2.drawContours(debug_image, [box], 0, (0, 255, 255), 2)
    if match_point is not None:
        cv2.drawMarker(
            debug_image,
            (int(match_point[0]), int(match_point[1])),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )
    if category:
        cv2.putText(
            debug_image,
            f"{category} theta={float(theta):.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )


def detect_block_with_high_prior_roi(
    img_bgr,
    template_geometry=None,
    template_profile="low",
    category="",
    high_theta_deg=0.0,
    angle_window=10.0,
    angle_step=1.0,
    roi_expand_px=50,
    white_s_max=45,
    white_v_min=180,
    min_foreground_area=200,
):
    """低位方块精定位：使用高位类别和角度生成 ROI 后直接模板匹配。

    低位画面中相机已在方块正上方，不再重新 YOLO 检测类别和框。
    这里用画面中心、高位旋转角和低位模板尺寸估算旋转矩形 ROI；
    ROI 内按局部 RGB 颜色分割后，只在高位角度附近做模板匹配。
    """
    debug_image = _make_debug_image(img_bgr)
    if img_bgr is None or img_bgr.size == 0:
        return _empty_detection("输入图像为空", debug_image)

    category = normalize_category_name(str(category or "").strip())
    if not category:
        return _empty_detection("低位先验 ROI 缺少方块类别", debug_image)

    if template_geometry is None:
        template_geometry = load_template_geometry(template_profile)
    block_px = template_geometry["block_px"]
    connector_px = template_geometry["connector_px"]
    rect_size = get_template_rect_size(category, block_px, connector_px)

    image_h, image_w = img_bgr.shape[:2]
    center = (image_w / 2.0, image_h / 2.0)
    high_theta_deg = float(high_theta_deg)
    roi_expand_px = max(0.0, float(roi_expand_px))
    expanded_size = (
        float(rect_size[0]) + 2.0 * roi_expand_px,
        float(rect_size[1]) + 2.0 * roi_expand_px,
    )
    roi_rect = (center, expanded_size, high_theta_deg)
    roi_box = cv2.boxPoints(roi_rect)
    crop_x1, crop_y1, crop_x2, crop_y2 = _clip_bbox_from_points(roi_box, img_bgr.shape)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return _empty_detection("低位先验 ROI 越界为空", debug_image)

    roi_bgr = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    roi_debug_image = np.copy(debug_image)
    _draw_prior_roi_debug(roi_debug_image, roi_box, category=category, theta=high_theta_deg)
    local_roi_box = roi_box - np.array([crop_x1, crop_y1], dtype=np.float32)
    roi_polygon_mask = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(roi_polygon_mask, np.intp(local_roi_box), 255)

    foreground_mask, mask_stages = _segment_roi_by_local_rgb_color(
        roi_bgr,
        category,
        return_stages=True,
    )
    roi_seed_debug = _draw_seed_patch_debug(roi_bgr, mask_stages)
    foreground_mask = cv2.bitwise_and(foreground_mask, roi_polygon_mask)
    foreground_area = int(cv2.countNonZero(foreground_mask))
    if foreground_area < int(min_foreground_area):
        _draw_prior_roi_debug(debug_image, roi_box, category=category, theta=high_theta_deg)
        debug_panel = _make_block_debug_panel(
            roi_debug_image,
            roi_seed_debug=roi_seed_debug,
            raw_mask=mask_stages["raw_mask"],
            final_mask=foreground_mask,
            match_mask_debug=_draw_template_match_on_mask(foreground_mask, None),
            match_debug=debug_image,
            message=_format_seed_debug_message(
                category,
                foreground_area,
                mask_stages,
                prefix="前景面积过小",
            ),
        )
        return _empty_detection(
            f"低位 ROI 前景面积过小: {foreground_area}",
            debug_image,
            debug_panel=debug_panel,
        )

    # get_rect 内部模板角度为逆时针正；返回的 OpenCV 矩形角度是相反数。
    template_angle_center = -high_theta_deg
    match_debug_output = {}
    rect = get_rect(
        foreground_mask,
        block_px,
        connector_px,
        category,
        debug_image,
        crop_x1,
        crop_y1,
        angle_step=angle_step,
        angle_center=template_angle_center,
        angle_window=angle_window,
        debug_output=match_debug_output,
    )
    match_mask_debug = _draw_template_match_on_mask(foreground_mask, match_debug_output)

    box = cv2.boxPoints(rect)
    box = np.intp(box)
    px = int(rect[0][0]) + crop_x1
    py = int(rect[0][1]) + crop_y1
    if category in ("L_yellow", "L_blue"):
        px, py = coreect_LL_location(box, foreground_mask, rect)
        px += crop_x1
        py += crop_y1

    theta = rect[2]
    if theta < -180:
        theta += 360

    cv2.drawMarker(
        debug_image,
        (int(center[0]), int(center[1])),
        (255, 0, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=2,
    )
    cv2.line(
        debug_image,
        (int(center[0]), int(center[1])),
        (int(px), int(py)),
        (255, 0, 0),
        1,
    )
    _draw_prior_roi_debug(
        debug_image,
        roi_box,
        match_point=(px, py),
        category=category,
        theta=theta,
    )
    debug_panel = _make_block_debug_panel(
        roi_debug_image,
        roi_seed_debug=roi_seed_debug,
        raw_mask=mask_stages["raw_mask"],
        final_mask=foreground_mask,
        match_mask_debug=match_mask_debug,
        match_debug=debug_image,
        message=_format_seed_debug_message(category, foreground_area, mask_stages),
    )

    return {
        "found": True,
        "category": category,
        "px": float(px),
        "py": float(py),
        "theta": float(theta),
        "score": 1.0,
        "debug_image": debug_image,
        "debug_panel": debug_panel,
        "message": "低位先验 ROI 模板匹配成功",
    }


def detect_blocks_in_image(
    img_bgr,
    model,
    template_geometry=None,
    template_profile=None,
    crop_margin=8,
    save_mask_overlay=False,
    angle_step=1,
    angle_center=None,
    angle_window=None,
    angle_values=None,
    search_center=None,
    search_radius=None,
    expected_category="",
):
    """检测当前图像里的所有方块，并返回每个方块的吸取点像素和角度。

    这个函数把原来 process.py 里的 YOLO 检测、上表面分割、模板匹配、
    L 型特殊吸取点修正集中到一处。视觉伺服和原有全场识别都调用它，
    避免后续两套识别逻辑漂移。
    """
    if img_bgr is None or img_bgr.size == 0:
        return [], _empty_detection("输入图像为空")

    if template_geometry is None:
        template_geometry = load_template_geometry(template_profile)
    expected_category = normalize_category_name(expected_category.strip()) if expected_category else ""
    block_px = template_geometry["block_px"]
    connector_px = template_geometry["connector_px"]

    image_h, image_w = img_bgr.shape[:2]
    debug_image = np.copy(img_bgr)
    mask_vis_img = np.copy(img_bgr) if save_mask_overlay else None
    result = model(img_bgr, iou=0.5, conf=0.45)
    blocks = []

    for det in result[0].boxes.data.tolist():
        x1, y1, x2, y2, score, cid = det
        category = normalize_category_name(model.names[int(cid)])
        if category == "board":
            continue
        if expected_category and category != expected_category:
            continue

        crop_x1 = max(0, int(x1) - crop_margin)
        crop_y1 = max(0, int(y1) - crop_margin)
        crop_x2 = min(image_w, int(x2) + crop_margin)
        crop_y2 = min(image_h, int(y2) + crop_margin)
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            continue

        cropped_img = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        try:
            mask, _ = get_mask(cropped_img, category)
            local_search_center = None
            if search_center is not None:
                local_search_center = (
                    float(search_center[0]) - crop_x1,
                    float(search_center[1]) - crop_y1,
                )
            rect = get_rect(
                mask,
                block_px,
                connector_px,
                category,
                debug_image,
                crop_x1,
                crop_y1,
                angle_step=angle_step,
                angle_center=angle_center,
                angle_window=angle_window,
                angle_values=angle_values,
                search_center=local_search_center,
                search_radius=search_radius,
            )
        except Exception as exc:
            print(f"方块 {category} 精定位失败，跳过。原因：{exc}")
            continue

        if save_mask_overlay:
            draw_mask_on_full_image(mask_vis_img, mask, crop_x1, crop_y1)

        box = cv2.boxPoints(rect)
        box = np.intp(box)
        px = int(rect[0][0]) + crop_x1
        py = int(rect[0][1]) + crop_y1
        if category in ("L_yellow", "L_blue"):
            px, py = coreect_LL_location(box, mask, rect)
            px += crop_x1
            py += crop_y1

        theta = rect[2]
        if theta < -180:
            theta += 360

        cv2.circle(debug_image, (int(px), int(py)), 3, (0, 0, 255), 2)
        cv2.rectangle(
            debug_image,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            1,
        )
        cv2.putText(
            debug_image,
            f"{category} ({px:.0f},{py:.0f})",
            (max(0, int(x1)), max(15, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        blocks.append({
            "found": True,
            "category": category,
            "px": float(px),
            "py": float(py),
            "theta": float(theta),
            "score": float(score),
            "debug_image": debug_image,
            "mask_overlay": mask_vis_img,
            "message": "识别成功",
        })

    return blocks, debug_image


def detect_single_block_in_image(
    img_bgr,
    model,
    template_geometry=None,
    template_profile=None,
    crop_margin=8,
    expected_category="",
    angle_step=1,
    angle_center=None,
    angle_window=None,
    angle_values=None,
    search_center=None,
    search_radius=None,
):
    """检测画面中的唯一目标方块。

    最小视觉伺服验证默认环境里只有一个方块；如果画面里有多个方块，
    这里选择置信度最高的匹配结果。expected_category 非空时会过滤类别，
    避免调试时误把其他物体当作目标。
    """
    blocks, debug_image = detect_blocks_in_image(
        img_bgr,
        model,
        template_geometry=template_geometry,
        template_profile=template_profile,
        crop_margin=crop_margin,
        angle_step=angle_step,
        angle_center=angle_center,
        angle_window=angle_window,
        angle_values=angle_values,
        search_center=search_center,
        search_radius=search_radius,
        expected_category=expected_category,
    )

    if not blocks:
        return _empty_detection("没有检测到目标方块", debug_image)

    best = max(blocks, key=lambda item: item["score"])
    best["debug_image"] = debug_image
    return best
