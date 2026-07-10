"""高位 YOLO、上表面分割和模板匹配方块识别。"""

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

