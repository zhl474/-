"""YOLO 预览画框与类别摘要的纯函数测试。"""

import numpy as np

from image_process_lib.block_scene_detector import (
    build_yolo_preview_summary,
    draw_yolo_detections,
)


def test画框标注不改变原图并输出同尺寸图():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    detections = [
        {"category": "z_blue", "score": 0.87, "box": (10.4, 20.6, 110.2, 120.9)},
        {"category": "square", "score": 0.52, "box": (150.0, 30.0, 250.0, 130.0)},
    ]

    annotated = draw_yolo_detections(image, detections)

    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)
    assert np.array_equal(image, np.zeros_like(image))  # 原图未被修改


def test空检测列表返回原图拷贝():
    image = np.full((60, 80, 3), 7, dtype=np.uint8)

    annotated = draw_yolo_detections(image, [])

    assert np.array_equal(annotated, image)
    assert annotated is not image


def test摘要按类别统计并保持标准类别顺序():
    detections = [
        {"category": "square", "score": 0.9, "box": (0, 0, 1, 1)},
        {"category": "z_blue", "score": 0.9, "box": (0, 0, 1, 1)},
        {"category": "square", "score": 0.8, "box": (0, 0, 1, 1)},
        {"category": "L_blue", "score": 0.7, "box": (0, 0, 1, 1)},
    ]

    summary = build_yolo_preview_summary(detections)

    assert summary == "识别到 4 个: L_bluex1, z_bluex1, squarex2"


def test未知类别排在标准类别之后():
    detections = [
        {"category": "mystery", "score": 0.9, "box": (0, 0, 1, 1)},
        {"category": "line", "score": 0.9, "box": (0, 0, 1, 1)},
    ]

    summary = build_yolo_preview_summary(detections)

    assert summary == "识别到 2 个: linex1, mysteryx1"


def test无检测时返回未识别提示():
    assert build_yolo_preview_summary([]) == "未识别到方块"
