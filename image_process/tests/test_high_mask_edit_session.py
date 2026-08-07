"""高位 Mask 人工编辑临时会话的事务和校验测试。"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from image_process_lib.high_mask_edit_session import (
    会话协议版本,
    创建高位Mask编辑会话,
    提交高位Mask编辑结果,
    读取已提交高位Mask,
    读取高位Mask编辑会话,
    高位Mask会话错误,
)


def _make_image_and_blocks():
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    first_mask = np.zeros((20, 30), dtype=np.uint8)
    first_mask[4:16, 5:25] = 255
    second_mask = np.zeros((25, 30), dtype=np.uint8)
    second_mask[3:20, 6:24] = 255
    blocks = [
        {
            "category": "T",
            "score": 0.9,
            "detection_box": (12.5, 11.0, 35.5, 27.0),
            "crop_box": (10, 10, 40, 30),
            "mask": first_mask,
        },
        {
            "category": "z_green",
            "score": 0.8,
            "detection_box": (53.0, 22.0, 76.0, 40.0),
            "crop_box": (50, 20, 80, 45),
            "mask": second_mask,
        },
    ]
    return image, blocks


def test_session_round_trip_and_atomic_commit(tmp_path):
    image, blocks = _make_image_and_blocks()
    blocks[0].update({
        "px": 25.0,
        "py": 20.0,
        "theta": -12.0,
        "rect": ((15.0, 10.0), (30.0, 20.0), -12.0),
    })
    manifest_path = 创建高位Mask编辑会话(tmp_path, image, blocks)

    manifest, loaded_image, loaded_masks = 读取高位Mask编辑会话(manifest_path)
    assert manifest["version"] == 会话协议版本
    assert np.array_equal(loaded_image, image)
    assert np.array_equal(loaded_masks[0], blocks[0]["mask"])
    assert manifest["blocks"][0]["preview_match"] == {
        "local_center": [15.0, 10.0],
        "rect_size": [30.0, 20.0],
        "rect_theta": -12.0,
        "px": 25.0,
        "py": 20.0,
        "theta": -12.0,
    }
    assert not (tmp_path / "commit.json").exists()

    edited_masks = [mask.copy() for mask in loaded_masks]
    edited_masks[0][0:3, 0:3] = 255
    提交高位Mask编辑结果(manifest_path, edited_masks)
    committed_masks = 读取已提交高位Mask(manifest_path, blocks)

    assert np.array_equal(committed_masks[0], edited_masks[0])
    assert np.array_equal(committed_masks[1], edited_masks[1])


def test_session_rejects_non_binary_and_wrong_shape_masks(tmp_path):
    image, blocks = _make_image_and_blocks()
    blocks[0]["mask"][0, 0] = 127
    with pytest.raises(高位Mask会话错误, match="严格二值"):
        创建高位Mask编辑会话(tmp_path / "bad_binary", image, blocks)

    image, blocks = _make_image_and_blocks()
    blocks[0]["mask"] = np.zeros((3, 4), dtype=np.uint8)
    with pytest.raises(高位Mask会话错误, match="尺寸错误"):
        创建高位Mask编辑会话(tmp_path / "bad_shape", image, blocks)


def test_parent_rejects_stale_session_id_and_category(tmp_path):
    image, blocks = _make_image_and_blocks()
    manifest_path = 创建高位Mask编辑会话(tmp_path, image, blocks)
    _manifest, _image, masks = 读取高位Mask编辑会话(manifest_path)
    commit_path = 提交高位Mask编辑结果(manifest_path, masks)

    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    commit["session_id"] = "旧会话"
    commit_path.write_text(json.dumps(commit, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(高位Mask会话错误, match="session_id"):
        读取已提交高位Mask(manifest_path, blocks)

    提交高位Mask编辑结果(manifest_path, masks)
    wrong_blocks = [dict(block) for block in blocks]
    wrong_blocks[1]["category"] = "square"
    with pytest.raises(高位Mask会话错误, match="类别"):
        读取已提交高位Mask(manifest_path, wrong_blocks)


def test_parent_rejects_manifest_crop_box_changed_after_session_created(tmp_path):
    image, blocks = _make_image_and_blocks()
    manifest_path = 创建高位Mask编辑会话(tmp_path, image, blocks)
    _manifest, _image, masks = 读取高位Mask编辑会话(manifest_path)
    提交高位Mask编辑结果(manifest_path, masks)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["blocks"][0]["crop_box"] = [9, 10, 39, 30]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(高位Mask会话错误, match="裁剪框与当前检测不一致"):
        读取已提交高位Mask(manifest_path, blocks)


def test_parent_rejects_missing_commit_and_modified_mask_file(tmp_path):
    image, blocks = _make_image_and_blocks()
    manifest_path = 创建高位Mask编辑会话(tmp_path, image, blocks)
    with pytest.raises(高位Mask会话错误, match="commit.json"):
        读取已提交高位Mask(manifest_path, blocks)

    _manifest, _image, masks = 读取高位Mask编辑会话(manifest_path)
    commit_path = 提交高位Mask编辑结果(manifest_path, masks)
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    mask_path = tmp_path / commit["blocks"][0]["mask_path"]
    invalid = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    invalid[0, 0] = 123
    assert cv2.imwrite(str(mask_path), invalid)
    with pytest.raises(高位Mask会话错误, match="严格二值"):
        读取已提交高位Mask(manifest_path, blocks)
