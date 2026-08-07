"""高位人工 Mask 返回后父进程权威重匹配测试。"""

import sys
import types
from pathlib import Path

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

# 该测试只验证模板匹配编排，不加载实际 YOLO 模型。
if "ultralytics" not in sys.modules:
    ultralytics_stub = types.ModuleType("ultralytics")
    ultralytics_stub.YOLO = object
    sys.modules["ultralytics"] = ultralytics_stub

import image_process_lib.block_scene_detector as detector


def test_rematch_blocks_recomputes_pose_and_keeps_session_metadata(monkeypatch):
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    block = {
        "category": "line",
        "score": 0.91,
        "px": 0.0,
        "py": 0.0,
        "theta": 0.0,
        "crop_box": (20, 10, 70, 50),
        "detection_box": (25.0, 15.0, 65.0, 45.0),
        "mask": np.zeros((40, 50), dtype=np.uint8),
    }
    edited_mask = np.zeros((40, 50), dtype=np.uint8)
    edited_mask[8:30, 6:42] = 255

    monkeypatch.setattr(
        detector,
        "get_rect",
        lambda *_args, **_kwargs: ((17.0, 13.0), (30.0, 10.0), -12.0),
    )
    rematched, debug_image = detector.rematch_blocks_from_masks(
        image,
        [block],
        [edited_mask],
        template_geometry={"block_px": 10, "connector_px": 2},
        save_mask_overlay=True,
    )

    assert rematched[0]["px"] == 37.0
    assert rematched[0]["py"] == 23.0
    assert rematched[0]["theta"] == -12.0
    assert rematched[0]["score"] == 0.91
    assert rematched[0]["crop_box"] == block["crop_box"]
    assert np.array_equal(rematched[0]["mask"], edited_mask)
    assert rematched[0]["debug_image"] is debug_image
    assert rematched[0]["mask_overlay"] is not None


def test_l_block_pick_point_is_recomputed_from_edited_mask(monkeypatch):
    mask = np.zeros((30, 40), dtype=np.uint8)
    mask[4:25, 5:35] = 255
    monkeypatch.setattr(
        detector,
        "get_rect",
        lambda *_args, **_kwargs: ((20.0, 15.0), (30.0, 20.0), -5.0),
    )
    monkeypatch.setattr(detector, "coreect_LL_location", lambda *_args: (7, 9))

    result = detector.match_block_mask(
        mask,
        "L_blue",
        {"block_px": 10, "connector_px": 2},
        (100, 200, 140, 230),
    )

    assert result["px"] == 107.0
    assert result["py"] == 209.0
    assert result["theta"] == -5.0
