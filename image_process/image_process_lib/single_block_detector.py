import cv2
import numpy as np

from image_process_lib.block_detection import (
    coreect_LL_location,
    draw_mask_on_full_image,
    get_mask,
)
from image_process_lib.template_config import get_template_size, load_template_sizes
from image_process_lib.template_match.template_match import get_rect


# 进阶任务动态库和旧代码里可能出现旧类别名，这里统一转成当前模型类别名。
CATEGORY_NAME_MAP = {
    "LR": "L_blue",
    "LL": "L_yellow",
    "ZL": "z_blue",
    "ZR": "z_green",
    "O": "square",
    "suqare": "square",
    "Line": "line",
}


def normalize_category_name(category):
    """统一方块类别名，避免旧模型类别和新模型类别混用。"""
    return CATEGORY_NAME_MAP.get(category, category)


def undistort_bgr_image(img_bgr, camera_matrix, dist_coeff):
    """按当前相机内参去畸变，返回后续检测统一使用的图像。"""
    h, w = img_bgr.shape[:2]
    new_camera_mtx, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeff,
        (w, h),
        1,
        (w, h),
    )
    return cv2.undistort(img_bgr, camera_matrix, dist_coeff, None, new_camera_mtx)


def _empty_detection(message, debug_image=None):
    return {
        "found": False,
        "category": "",
        "px": 0.0,
        "py": 0.0,
        "theta": 0.0,
        "score": 0.0,
        "debug_image": debug_image,
        "message": message,
    }


def detect_blocks_in_image(
    img_bgr,
    model,
    template_sizes=None,
    crop_margin=8,
    save_mask_overlay=False,
):
    """检测当前图像里的所有方块，并返回每个方块的吸取点像素和角度。

    这个函数把原来 process.py 里的 YOLO 检测、上表面分割、模板匹配、
    L 型特殊吸取点修正集中到一处。视觉伺服和原有全场识别都调用它，
    避免后续两套识别逻辑漂移。
    """
    if img_bgr is None or img_bgr.size == 0:
        return [], _empty_detection("输入图像为空")

    if template_sizes is None:
        template_sizes = load_template_sizes()

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

        crop_x1 = max(0, int(x1) - crop_margin)
        crop_y1 = max(0, int(y1) - crop_margin)
        crop_x2 = min(image_w, int(x2) + crop_margin)
        crop_y2 = min(image_h, int(y2) + crop_margin)
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            continue

        cropped_img = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        try:
            mask, _ = get_mask(cropped_img, category)
            template_w, template_h = get_template_size(category, template_sizes)
            rect = get_rect(mask, template_w, template_h, category, debug_image, crop_x1, crop_y1)
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
    template_sizes=None,
    crop_margin=8,
    expected_category="",
):
    """检测画面中的唯一目标方块。

    最小视觉伺服验证默认环境里只有一个方块；如果画面里有多个方块，
    这里选择置信度最高的匹配结果。expected_category 非空时会过滤类别，
    避免调试时误把其他物体当作目标。
    """
    blocks, debug_image = detect_blocks_in_image(
        img_bgr,
        model,
        template_sizes=template_sizes,
        crop_margin=crop_margin,
    )
    if expected_category:
        expected_category = normalize_category_name(expected_category)
        blocks = [block for block in blocks if block["category"] == expected_category]

    if not blocks:
        return _empty_detection("没有检测到目标方块", debug_image)

    best = max(blocks, key=lambda item: item["score"])
    best["debug_image"] = debug_image
    return best
