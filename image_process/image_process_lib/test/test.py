"""测试方块检测：读图 → YOLO → 分割 → 模板匹配 → 可视化。

流程完全走 block_scene_detector.detect_blocks_in_image，和实际一致。
test/ 下的 block_detection.py 会替换 image_process_lib 里的旧版本。
"""

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# image_process_lib 的上级目录（即 image_process 包目录）
PACKAGE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
# image_process 的上级目录（即项目根目录）
ROOT_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, ".."))

# 确保 image_process 所在的父目录在 sys.path 中，这样 image_process_lib 可以被找到
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 把 test/ 下的 block_detection 注入到 image_process_lib 命名空间
# 这样 block_scene_detector import 时拿到的是本地改过的版本
import block_detection
sys.modules["image_process_lib.block_detection"] = block_detection

from image_process_lib.template_config import load_template_geometry
from image_process_lib.block_scene_detector import detect_blocks_in_image

import cv2

DEFAULT_IMAGE = "/home/zhl/SingleArmTetris/SingleArmTetris/src/333.jpg"
COMPETITION_DIR = os.path.join(ROOT_DIR, "..", "competition")
DEFAULT_MODEL_PATH = os.path.join(COMPETITION_DIR, "model", "best5.14.pt")


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

    blocks, debug_image = detect_blocks_in_image(
        img, model,
        template_geometry=template_geometry,
    )

    print(f"\n检测到 {len(blocks)} 个方块:")
    for i, b in enumerate(blocks, 1):
        print(f"  {i}. {b['category']:12s}  px={b['px']:.1f}  py={b['py']:.1f}  "
              f"theta={b['theta']:.1f}  score={b['score']:.3f}")

    out_path = os.path.join(os.path.dirname(image_path), "test_result.png")
    cv2.imwrite(out_path, debug_image)
    print(f"\n调试图已保存: {out_path}")


if __name__ == "__main__":
    main()
