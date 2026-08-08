"""高位类别/角度先验约束下的低位方块识别。"""

import time

import cv2
import numpy as np
import torch

from image_process_lib.block_detection import (
    coreect_LL_location,
    draw_mask_on_full_image,
    get_mask,
)
from image_process_lib.block_category import normalize_category_name
from image_process_lib.template_config import load_color_segmentation_config, load_template_geometry
from image_process_lib.template_match.kernels_create import (
    build_angle_values,
    create_pick_aligned_kernels,
)
from image_process_lib.template_match.template_match import get_rect, match_pick_aligned


_TIMING_STAGE_NAMES = (
    "先验ROI",
    "RGB分割",
    "模板生成",
    "张量准备",
    "卷积选优",
    "匹配收尾",
    "检测调试图",
)


_LOW_PICK_TEMPLATE_CACHE = {}
_LOW_PICK_TEMPLATE_CACHE_CAP = 16


def _create_timing_info(category):
    """创建一次低位方块识别的分段耗时容器。"""
    return {
        "类别": category,
        "状态": "未完成",
        "后端": None,
        "ROI尺寸": None,
        "匹配图尺寸": None,
        "模板数量": None,
        "模板核尺寸": None,
        "匹配模式": "快速像素匹配",
        "模板缓存": "未使用",
        "前景面积": None,
        "阶段毫秒": {stage_name: None for stage_name in _TIMING_STAGE_NAMES},
        "_开始时间": time.perf_counter(),
    }


def _add_timing_stage(timing_info, stage_name, started_at):
    """累加一个 CPU 阶段耗时；关闭诊断时不进行计时。"""
    if timing_info is None:
        return
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    previous_ms = timing_info["阶段毫秒"].get(stage_name)
    timing_info["阶段毫秒"][stage_name] = elapsed_ms + (previous_ms or 0.0)


def _finalize_timing_info(timing_info, status):
    """固化检测状态和总耗时，返回给图像节点输出日志。"""
    if timing_info is None:
        return None
    timing_info["状态"] = status
    timing_info["总计毫秒"] = (time.perf_counter() - timing_info.pop("_开始时间")) * 1000.0
    return timing_info


def _empty_detection(message, debug_image=None, debug_panel=None, timing_info=None):
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
        "timing": timing_info,
    }


def _make_debug_image(img_bgr):
    """生成可画调试信息的图像，输入为空时返回空白占位。"""
    if img_bgr is None or img_bgr.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return np.copy(img_bgr)


def _get_low_pick_templates(block_px, connector_px, category, angles, device, kernel_safety_margin_px):
    """按设备、类别、几何、实际角度元组惰性生成并缓存抓取点对齐公共核。"""
    cache_key = (
        str(device),
        str(category),
        int(block_px),
        int(connector_px),
        tuple(round(float(angle), 6) for angle in angles),
        int(kernel_safety_margin_px),
    )
    cached = _LOW_PICK_TEMPLATE_CACHE.get(cache_key)
    if cached is not None:
        _LOW_PICK_TEMPLATE_CACHE[cache_key] = _LOW_PICK_TEMPLATE_CACHE.pop(cache_key)
        return cached, "命中"
    prepared = create_pick_aligned_kernels(
        category,
        block_px,
        connector_px,
        angles,
        device=device,
        safety_margin_px=kernel_safety_margin_px,
    )
    _LOW_PICK_TEMPLATE_CACHE[cache_key] = prepared
    while len(_LOW_PICK_TEMPLATE_CACHE) > _LOW_PICK_TEMPLATE_CACHE_CAP:
        _LOW_PICK_TEMPLATE_CACHE.pop(next(iter(_LOW_PICK_TEMPLATE_CACHE)))
    return prepared, "未命中"


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


def _segment_roi_by_local_rgb_color(roi_bgr, category, return_stages=False, search_center=None):
    """在低位 ROI 内用局部 RGB 颜色种子分割目标方块。

    search_center 为 ROI 内 seed 搜索中心（如相机中心在 ROI 中的坐标）；
    None 时使用 ROI 几何中心。
    """
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
    if search_center is None:
        center_x = roi_w / 2.0
        center_y = roi_h / 2.0
    else:
        center_x = float(search_center[0])
        center_y = float(search_center[1])
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
    best_score = None
    for patch_y in range(search_y1, search_y2 - seed_patch_size + 1, seed_stride):
        for patch_x in range(search_x1, search_x2 - seed_patch_size + 1, seed_stride):
            patch = roi_rgb[patch_y:patch_y + seed_patch_size, patch_x:patch_x + seed_patch_size]
            r_mean, g_mean, b_mean = patch.reshape(-1, 3).mean(axis=0)
            color_d2 = (
                (r_mean - r_prior) ** 2
                + (g_mean - g_prior) ** 2
                + (b_mean - b_prior) ** 2
            )
            mean_rgb = np.array([r_mean, g_mean, b_mean], dtype=np.float32)
            patch_variance_d2 = float(np.mean(np.sum((patch - mean_rgb) ** 2, axis=2)))
            seed_score = float(color_d2 + patch_variance_d2)
            if best_score is None or seed_score < best_score:
                best_score = seed_score
                best_patch = {
                    "x": patch_x,
                    "y": patch_y,
                    "mean": (float(r_mean), float(g_mean), float(b_mean)),
                    "color_d2": float(color_d2),
                    "variance_d2": patch_variance_d2,
                    "score": seed_score,
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
    morph_foreground = raw_foreground
    # morph_foreground = cv2.morphologyEx(raw_foreground, cv2.MORPH_OPEN, kernel)这个b开闭运算在帮倒忙，把正确的边缘干歪了
    # morph_foreground = cv2.morphologyEx(morph_foreground, cv2.MORPH_CLOSE, kernel)
    foreground = morph_foreground

    seed_center_x = best_patch["x"] + seed_patch_size / 2.0
    seed_center_y = best_patch["y"] + seed_patch_size / 2.0
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
            "seed_color_d2": float(best_patch["color_d2"]),
            "seed_variance_d2": float(best_patch["variance_d2"]),
            "seed_score": float(best_patch["score"]),
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
    seed_variance_d2 = mask_stages.get("seed_variance_d2")

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
    if seed_variance_d2 is not None:
        _put_chinese_text(debug, f"方差D2={float(seed_variance_d2):.0f}", (8, 64), (0, 255, 255), font_size=20)

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
    seed_variance_d2 = float(mask_stages.get("seed_variance_d2", 0.0))
    seed_rgb = [int(round(float(value))) for value in local_color]
    prefix_text = f"{prefix} " if prefix else ""
    return (
        f"{prefix_text}类别={category} 面积={foreground_area} "
        f"seedRGB={seed_rgb} 阈值={local_dist_thresh:.1f} 方差D2={seed_variance_d2:.0f}"
    )



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


def _attempt_legacy_slow(
    img_bgr,
    center_x,
    center_y,
    common_pick_x,
    common_pick_y,
    radius_px,
    kernel_w,
    kernel_h,
    block_px,
    connector_px,
    category,
    high_theta_deg,
    angle_step,
    angle_window,
    min_foreground_area,
    debug_image,
    debug_enabled,
    timing_info,
):
    """慢速兜底：在轴对齐 ROI 上用旧全方形核 + same-padding 匹配，不做任何旋转。"""
    roi_w = kernel_w + 2 * radius_px
    roi_h = kernel_h + 2 * radius_px
    x1 = int(round(center_x - common_pick_x - radius_px))
    y1 = int(round(center_y - common_pick_y - radius_px))
    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(img_bgr.shape[1], x1 + roi_w)
    crop_y2 = min(img_bgr.shape[0], y1 + roi_h)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None, "ROI 越界为空"
    roi_bgr = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    seed_center = (center_x - crop_x1, center_y - crop_y1)
    segmentation_started_at = time.perf_counter() if timing_info is not None else None
    try:
        if debug_enabled:
            foreground_mask, mask_stages = _segment_roi_by_local_rgb_color(
                roi_bgr,
                category,
                return_stages=True,
                search_center=seed_center,
            )
        else:
            foreground_mask = _segment_roi_by_local_rgb_color(
                roi_bgr,
                category,
                search_center=seed_center,
            )
            mask_stages = None
    except ValueError as exc:
        _add_timing_stage(timing_info, "RGB分割", segmentation_started_at)
        return None, f"seed 搜索失败: {exc}"
    _add_timing_stage(timing_info, "RGB分割", segmentation_started_at)
    foreground_area = int(cv2.countNonZero(foreground_mask))
    if foreground_area < int(min_foreground_area):
        return None, f"低位 ROI 前景面积过小: {foreground_area}"
    match_debug_output = {}
    rect = get_rect(
        foreground_mask,
        block_px,
        connector_px,
        category,
        debug_image if debug_enabled else None,
        crop_x1,
        crop_y1,
        angle_step=angle_step,
        angle_center=-float(high_theta_deg),
        angle_window=angle_window,
        debug_output=match_debug_output,
        timing_output=timing_info,
    )
    box = np.intp(cv2.boxPoints(rect))
    local_px, local_py = float(rect[0][0]), float(rect[0][1])
    if category in ("L_yellow", "L_blue"):
        local_px, local_py = coreect_LL_location(box, foreground_mask, rect)
    theta = float(rect[2])
    if theta < -180:
        theta += 360
    return (
        {
            "crop_x1": crop_x1,
            "crop_y1": crop_y1,
            "local_px": local_px,
            "local_py": local_py,
            "theta": theta,
            "roi_bgr": roi_bgr,
            "foreground_mask": foreground_mask,
            "mask_stages": mask_stages,
            "match_debug_output": match_debug_output,
            "roi_box": (x1, y1, roi_w, roi_h),
        },
        None,
    )


def detect_block_with_high_prior_roi(
    img_bgr,
    template_geometry=None,
    template_profile="low",
    category="",
    high_theta_deg=0.0,
    angle_window=10.0,
    angle_step=1.0,
    search_radius_px=30,
    fallback_search_radius_px=50,
    boundary_guard_px=3,
    kernel_safety_margin_px=2,
    legacy_fallback_enabled=True,
    white_s_max=45,
    white_v_min=180,
    min_foreground_area=200,
    debug_enabled=True,
    timing_enabled=False,
):
    """低位方块精定位：取消转正，在轴对齐 ROI 上用抓取点对齐紧凑核无 padding 匹配。

    第一级搜索半径 search_radius_px，卷积输出固定为 2*R+1 的方形；
    结果触边、前景触边、ROI 被画面截断或分割/面积检查失败时扩大为
    fallback_search_radius_px；仍异常才走旧全方形核 same-padding 慢速兜底。
    每帧流程不调用任何仿射变换，坐标只经过整数裁剪偏移与模板锚点相加。
    """
    timing_info = _create_timing_info("") if timing_enabled else None
    debug_started_at = time.perf_counter() if timing_info is not None else None
    debug_image = _make_debug_image(img_bgr) if debug_enabled else None
    _add_timing_stage(timing_info, "检测调试图", debug_started_at)
    if img_bgr is None or img_bgr.size == 0:
        return _empty_detection(
            "输入图像为空",
            debug_image,
            timing_info=_finalize_timing_info(timing_info, "输入图像为空"),
        )

    category = normalize_category_name(str(category or "").strip())
    if timing_info is not None:
        timing_info["类别"] = category
    if not category:
        return _empty_detection(
            "低位先验 ROI 缺少方块类别",
            debug_image,
            timing_info=_finalize_timing_info(timing_info, "缺少类别"),
        )

    if template_geometry is None:
        template_geometry = load_template_geometry(template_profile)
    block_px = template_geometry["block_px"]
    connector_px = template_geometry["connector_px"]
    image_h, image_w = img_bgr.shape[:2]
    center_x, center_y = image_w / 2.0, image_h / 2.0
    high_theta_deg = float(high_theta_deg)
    search_radius_px = max(0, int(round(float(search_radius_px))))
    fallback_search_radius_px = max(
        search_radius_px, int(round(float(fallback_search_radius_px)))
    )
    boundary_guard_px = max(0, int(round(float(boundary_guard_px))))
    min_foreground_area = int(min_foreground_area)

    roi_started_at = time.perf_counter() if timing_info is not None else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    angles = build_angle_values(
        category,
        angle_step=angle_step,
        angle_center=-high_theta_deg,
        angle_window=angle_window,
    )
    template_started_at = time.perf_counter() if timing_info is not None else None
    prepared, cache_status = _get_low_pick_templates(
        block_px, connector_px, category, angles, device, kernel_safety_margin_px,
    )
    if timing_info is not None:
        timing_info["模板缓存"] = cache_status
        if cache_status == "未命中":
            if device == "cuda":
                torch.cuda.synchronize()
            timing_info["阶段毫秒"]["模板生成"] = (
                time.perf_counter() - template_started_at
            ) * 1000.0
        else:
            timing_info["阶段毫秒"]["模板生成"] = 0.0
    kernels = prepared["kernels"]
    kernel_h, kernel_w = prepared["kernel_size"]
    rect_center_anchors = prepared["rect_center_anchors"]
    pick_anchors = prepared["pick_anchors"]
    common_pick_x, common_pick_y = prepared["pick_anchor"]
    _add_timing_stage(timing_info, "先验ROI", roi_started_at)

    def _roi_box(radius_px):
        """按抓取点锚点计算轴对齐 ROI，返回裁剪范围、理论 ROI 与是否被画面截断。"""
        roi_w = kernel_w + 2 * radius_px
        roi_h = kernel_h + 2 * radius_px
        x1 = int(round(center_x - common_pick_x - radius_px))
        y1 = int(round(center_y - common_pick_y - radius_px))
        x2, y2 = x1 + roi_w, y1 + roi_h
        clipped = x1 < 0 or y1 < 0 or x2 > image_w or y2 > image_h
        crop_x1 = max(0, x1)
        crop_y1 = max(0, y1)
        crop_x2 = min(image_w, x2)
        crop_y2 = min(image_h, y2)
        return (crop_x1, crop_y1, crop_x2, crop_y2), (x1, y1, roi_w, roi_h), clipped

    last_state = {"roi_size": None, "foreground_area": None}

    def _attempt_fast(radius_px):
        """一级/二级快速匹配；返回 (成功信息 dict, 失败原因)。"""
        (crop_x1, crop_y1, crop_x2, crop_y2), (roi_x1, roi_y1, roi_w, roi_h), clipped = _roi_box(
            radius_px
        )
        last_state["roi_size"] = (roi_w, roi_h)
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None, "ROI 越界为空"
        if clipped:
            return None, "ROI 被画面边界截断"
        roi_bgr = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        segmentation_started_at = time.perf_counter() if timing_info is not None else None
        try:
            if debug_enabled:
                foreground_mask, mask_stages = _segment_roi_by_local_rgb_color(
                    roi_bgr,
                    category,
                    return_stages=True,
                    search_center=(common_pick_x + radius_px, common_pick_y + radius_px),
                )
            else:
                foreground_mask = _segment_roi_by_local_rgb_color(
                    roi_bgr,
                    category,
                    search_center=(common_pick_x + radius_px, common_pick_y + radius_px),
                )
                mask_stages = None
        except ValueError as exc:
            _add_timing_stage(timing_info, "RGB分割", segmentation_started_at)
            return None, f"seed 搜索失败: {exc}"
        _add_timing_stage(timing_info, "RGB分割", segmentation_started_at)
        foreground_area = int(cv2.countNonZero(foreground_mask))
        last_state["foreground_area"] = foreground_area
        if foreground_area < min_foreground_area:
            return None, f"低位 ROI 前景面积过小: {foreground_area}"
        match_debug_output = {}
        result = match_pick_aligned(
            foreground_mask,
            kernels,
            (kernel_h, kernel_w),
            prepared["angles"],
            rect_center_anchors,
            pick_anchors,
            debug_output=match_debug_output,
            timing_output=timing_info,
        )
        out_x, out_y = result["out_position"]
        output_size = 2 * radius_px
        # 只有一级检查结果是否贴近输出边界；二级本身即设计最大搜索范围，
        # 一级因触边升级后，±fallback_search_radius_px 内的结果直接接受。
        # 分割前景的散点噪声不参与判定：模板匹配对散点噪声不敏感，
        # 目标是否仍在搜索范围内由结果位置与 ROI 截断两个检查保证。
        if radius_px == search_radius_px and (
            out_x <= boundary_guard_px
            or out_y <= boundary_guard_px
            or out_x >= output_size - boundary_guard_px
            or out_y >= output_size - boundary_guard_px
        ):
            return None, "匹配结果贴近卷积输出边界"
        return {
            "result": result,
            "match_debug_output": match_debug_output,
            "roi_bgr": roi_bgr,
            "foreground_mask": foreground_mask,
            "mask_stages": mask_stages,
            "crop_x1": crop_x1,
            "crop_y1": crop_y1,
            "roi_box": (roi_x1, roi_y1, roi_w, roi_h),
        }, None

    matched = None
    match_mode = ""
    fast_fail_reason = None
    for radius_px, mode_name in (
        (search_radius_px, f"快速{search_radius_px}像素"),
        (fallback_search_radius_px, f"扩大{fallback_search_radius_px}像素"),
    ):
        if radius_px <= 0:
            continue
        attempt, fail_reason = _attempt_fast(radius_px)
        if attempt is not None:
            matched = attempt
            match_mode = mode_name
            break
        if fail_reason is not None:
            fast_fail_reason = fail_reason

    legacy_info = None
    if matched is None and legacy_fallback_enabled:
        match_mode = "慢速兜底"
        legacy_info, legacy_fail_reason = _attempt_legacy_slow(
            img_bgr,
            center_x,
            center_y,
            common_pick_x,
            common_pick_y,
            fallback_search_radius_px,
            kernel_w,
            kernel_h,
            block_px,
            connector_px,
            category,
            high_theta_deg,
            angle_step,
            angle_window,
            min_foreground_area,
            debug_image,
            debug_enabled,
            timing_info,
        )
        if legacy_info is None:
            fast_fail_reason = legacy_fail_reason or "慢速兜底失败"
    elif matched is None:
        fast_fail_reason = "慢速兜底已禁用"

    if matched is None and legacy_info is None:
        status = "失败"
        if "前景面积过小" in fast_fail_reason:
            status = "前景面积过小"
        if timing_info is not None:
            timing_info["ROI尺寸"] = last_state["roi_size"]
            timing_info["前景面积"] = last_state["foreground_area"]
        return _empty_detection(
            fast_fail_reason,
            debug_image,
            timing_info=_finalize_timing_info(timing_info, status),
        )

    # 只有快速路径未在第一级直接成功时，才记录失败原因，便于诊断兜底触发。
    if fast_fail_reason is not None and timing_info is not None:
        timing_info["快速路径失败原因"] = fast_fail_reason

    postprocess_started_at = time.perf_counter() if timing_info is not None else None
    if matched is not None:
        result = matched["result"]
        pick_x, pick_y = result["pick_point"]
        px = float(matched["crop_x1"] + pick_x)
        py = float(matched["crop_y1"] + pick_y)
        theta = -result["angle"]
        if theta < -180:
            theta += 360
        roi_bgr = matched["roi_bgr"]
        foreground_mask = matched["foreground_mask"]
        mask_stages = matched["mask_stages"]
        match_debug_output = matched["match_debug_output"]
        roi_box = matched["roi_box"]
        crop_x1 = matched["crop_x1"]
        crop_y1 = matched["crop_y1"]
    else:
        px = float(legacy_info["crop_x1"] + legacy_info["local_px"])
        py = float(legacy_info["crop_y1"] + legacy_info["local_py"])
        theta = float(legacy_info["theta"])
        roi_bgr = legacy_info["roi_bgr"]
        foreground_mask = legacy_info["foreground_mask"]
        mask_stages = legacy_info["mask_stages"]
        match_debug_output = legacy_info["match_debug_output"]
        roi_box = legacy_info["roi_box"]
        crop_x1 = legacy_info["crop_x1"]
        crop_y1 = legacy_info["crop_y1"]
    foreground_area = int(cv2.countNonZero(foreground_mask))
    if timing_info is not None:
        timing_info["匹配模式"] = match_mode
        timing_info["ROI尺寸"] = (int(roi_bgr.shape[1]), int(roi_bgr.shape[0]))
        timing_info["前景面积"] = foreground_area
    _add_timing_stage(timing_info, "匹配收尾", postprocess_started_at)

    debug_panel = None
    debug_started_at = time.perf_counter() if timing_info is not None else None
    if debug_enabled:
        roi_seed_debug = _draw_seed_patch_debug(roi_bgr, mask_stages)
        match_mask_debug = _draw_template_match_on_mask(foreground_mask, match_debug_output)
        cv2.drawMarker(
            debug_image,
            (int(center_x), int(center_y)),
            (255, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )
        cv2.line(
            debug_image,
            (int(center_x), int(center_y)),
            (int(px), int(py)),
            (255, 0, 0),
            1,
        )
        roi_x1, roi_y1, roi_w, roi_h = roi_box
        roi_box_points = cv2.boxPoints(
            ((roi_x1 + roi_w / 2.0, roi_y1 + roi_h / 2.0), (roi_w, roi_h), 0.0)
        )
        _draw_prior_roi_debug(
            debug_image,
            roi_box_points,
            match_point=(px, py),
            category=category,
            theta=theta,
        )
        debug_panel = _make_block_debug_panel(
            debug_image,
            roi_seed_debug=roi_seed_debug,
            raw_mask=mask_stages["raw_mask"],
            final_mask=foreground_mask,
            match_mask_debug=match_mask_debug,
            match_debug=debug_image,
            # 以图像中心为零点显示视觉伺服使用的像素误差，便于逐帧核对修正方向。
            message=(
                f"{_format_seed_debug_message(category, foreground_area, mask_stages)} | "
                f"模式={match_mode} | "
                f"像素误差：px={px - center_x:+.1f}，py={py - center_y:+.1f}"
            ),
        )
    _add_timing_stage(timing_info, "检测调试图", debug_started_at)

    success_message = f"低位先验 ROI 模板匹配成功（{match_mode}）"
    if legacy_info is not None and fast_fail_reason is not None:
        success_message += f"；快速路径失败: {fast_fail_reason}"
    return {
        "found": True,
        "category": category,
        "px": float(px),
        "py": float(py),
        "theta": float(theta),
        "score": 1.0,
        "debug_image": debug_image,
        "debug_panel": debug_panel,
        "match_debug": match_debug_output,
        "message": success_message,
        "timing": _finalize_timing_info(timing_info, "成功"),
    }
