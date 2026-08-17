"""测试软投票 L 型抓点：读图 → YOLO → 分割 → 模板匹配 → 可视化。

L型：绿圈 = 软投票成功，蓝圈 = 兜底扫描。
其他方块：红圈。
"""

import os
import sys

import cv2
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import image_process_lib.block_detection as block_detection
from image_process_lib.block_detection import get_mask
from image_process_lib.block_category import normalize_category_name
from image_process_lib.template_config import load_template_geometry
from image_process_lib.template_match.template_match import get_rect

DEFAULT_IMAGE = "/home/zhl/SingleArmTetris/SingleArmTetris/src/tools/vision/7.png"
COMPETITION_DIR = os.path.join(SRC_DIR, "..", "competition")
DEFAULT_MODEL_PATH = os.path.join(COMPETITION_DIR, "model", "best5.14.pt")

HIGH_SCREENING_CONFIG = None


def match_block_mask(mask, category, template_geometry, crop_box,
                     debug_image=None, detection_box=None):
    crop_x1, crop_y1, crop_x2, crop_y2 = (int(v) for v in crop_box)
    expected_shape = (crop_y2 - crop_y1, crop_x2 - crop_x1)
    if mask is None or mask.ndim != 2 or tuple(mask.shape) != expected_shape:
        raise ValueError(f"Mask 尺寸错误")
    if not np.any(mask > 0):
        raise ValueError("Mask 为空")

    rect = get_rect(
        mask, template_geometry["block_px"], template_geometry["connector_px"],
        category, debug_image, crop_x1, crop_y1,
        debug_output={}, screening_config=HIGH_SCREENING_CONFIG,
    )

    box = np.intp(cv2.boxPoints(rect))
    local_px = int(rect[0][0])
    local_py = int(rect[0][1])
    if category in ("L_yellow", "L_blue"):
        local_px, local_py = block_detection.coreect_LL_location(box, mask, rect)
    px = local_px + crop_x1
    py = local_py + crop_y1
    theta = float(rect[2])
    if theta < -180:
        theta += 360

    if debug_image is not None:
        if category in ("L_yellow", "L_blue"):
            if block_detection.last_ll_method == "soft_vote":
                color = (0, 255, 0)   # 绿圈
            else:
                color = (255, 0, 0)   # 蓝圈
        else:
            color = (0, 0, 255)       # 红圈
        cv2.circle(debug_image, (int(px), int(py)), 3, color, 2)
        label_x = crop_x1 if detection_box is None else max(0, int(detection_box[0]))
        label_y = crop_y1 if detection_box is None else max(15, int(detection_box[1]) - 6)
        cv2.putText(debug_image, f"{category} ({px:.0f},{py:.0f})",
                    (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255), 1, cv2.LINE_AA)

    return {"px": float(px), "py": float(py), "theta": theta, "rect": rect}


def detect_blocks(img_bgr, model, template_geometry):
    image_h, image_w = img_bgr.shape[:2]
    debug_image = np.copy(img_bgr)
    result = model(img_bgr, iou=0.5, conf=0.45)
    blocks = []
    for det in result[0].boxes.data.tolist():
        x1, y1, x2, y2, score, cid = det
        category = normalize_category_name(model.names[int(cid)])
        if category == "board":
            continue
        crop_x1 = max(0, int(x1) - 8)
        crop_y1 = max(0, int(y1) - 8)
        crop_x2 = min(image_w, int(x2) + 8)
        crop_y2 = min(image_h, int(y2) + 8)
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            continue
        cropped_img = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        try:
            mask, _ = get_mask(cropped_img, category)
            match = match_block_mask(mask, category, template_geometry,
                                     (crop_x1, crop_y1, crop_x2, crop_y2),
                                     debug_image=debug_image,
                                     detection_box=(x1, y1, x2, y2))
        except Exception as exc:
            print(f"方块 {category} 精定位失败，跳过。原因：{exc}")
            continue
        blocks.append({"found": True, "category": category,
                        "px": match["px"], "py": match["py"], "theta": match["theta"],
                        "score": float(score), "mask": mask.copy(),
                        "crop_box": (crop_x1, crop_y1, crop_x2, crop_y2),
                        "detection_box": (float(x1), float(y1), float(x2), float(y2)),
                        "rect": match["rect"], "debug_image": debug_image})
    return blocks, debug_image


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    model_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL_PATH
    print(f"图像: {image_path}")
    print(f"模型: {model_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        sys.exit(1)
    from ultralytics import YOLO
    model = YOLO(model_path, task="detect")
    template_geometry = load_template_geometry("high")
    blocks, debug_image = detect_blocks(img, model, template_geometry)
    print(f"\n检测到 {len(blocks)} 个方块:")
    for i, b in enumerate(blocks, 1):
        print(f"  {i}. {b['category']:12s}  px={b['px']:.1f}  py={b['py']:.1f}  "
              f"theta={b['theta']:.1f}  score={b['score']:.3f}")
    out_path = os.path.join(os.path.dirname(image_path), "test_result.png")
    cv2.imwrite(out_path, debug_image)
    print(f"\n调试图已保存: {out_path}")


if __name__ == "__main__":
    main()