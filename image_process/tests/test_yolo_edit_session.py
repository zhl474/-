"""YOLO 检测框人工修正临时会话的事务和校验测试。"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from image_process_lib.block_category import BLOCK_CATEGORY_NAMES
from image_process_lib.yolo_edit_session import (
    会话协议版本,
    创建YOLO编辑会话,
    提交YOLO编辑结果,
    读取已提交YOLO检测,
    读取YOLO编辑会话,
    YOLO编辑会话错误,
)


def _make_image_and_detections():
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    detections = [
        {
            "category": "T",
            "score": 0.92,
            "box": (12.0, 11.0, 36.0, 40.0),
        },
        {
            "category": "z_green",
            "score": 0.81,
            "box": (53.0, 22.0, 77.0, 44.0),
        },
    ]
    return image, detections


def test_session_round_trip_and_atomic_commit(tmp_path):
    image, detections = _make_image_and_detections()
    manifest_path = 创建YOLO编辑会话(tmp_path, image, detections)

    manifest, loaded_image = 读取YOLO编辑会话(manifest_path)
    assert manifest["version"] == 会话协议版本
    assert manifest["classes"] == list(BLOCK_CATEGORY_NAMES)
    assert np.array_equal(loaded_image, image)
    assert [record["category"] for record in manifest["detections"]] == ["T", "z_green"]
    assert manifest["detections"][0]["source"] == "yolo"
    assert not (tmp_path / "commit.json").exists()

    corrected = [
        {"category": "T", "score": 0.92, "box": (12.0, 11.0, 36.0, 40.0), "source": "yolo"},
        {"category": "L_blue", "score": 1.0, "box": (60.0, 10.0, 90.0, 50.0), "source": "manual"},
    ]
    提交YOLO编辑结果(manifest_path, corrected)
    committed = 读取已提交YOLO检测(manifest_path)

    assert len(committed) == 2
    assert committed[0] == {
        "category": "T",
        "score": 0.92,
        "box": (12.0, 11.0, 36.0, 40.0),
        "source": "yolo",
    }
    assert committed[1]["category"] == "L_blue"
    assert committed[1]["source"] == "manual"


def test_commit_allows_empty_detection_list(tmp_path):
    """用户删光全部框是合法提交，由下游按没有识别到方块报错。"""
    image, detections = _make_image_and_detections()
    manifest_path = 创建YOLO编辑会话(tmp_path, image, detections)
    提交YOLO编辑结果(manifest_path, [])
    assert 读取已提交YOLO检测(manifest_path) == []


def test_session_rejects_bad_category_and_boxes(tmp_path):
    image, _detections = _make_image_and_detections()

    bad_category = [{"category": "board", "score": 0.9, "box": (10, 10, 40, 40)}]
    with pytest.raises(YOLO编辑会话错误, match="非法"):
        创建YOLO编辑会话(tmp_path / "bad_category", image, bad_category)

    bad_box = [{"category": "T", "score": 0.9, "box": (10, 10, 40, 10)}]
    with pytest.raises(YOLO编辑会话错误, match="尺寸过小"):
        创建YOLO编辑会话(tmp_path / "bad_size", image, bad_box)

    out_of_range = [{"category": "T", "score": 0.9, "box": (90.0, 10.0, 140.0, 50.0)}]
    with pytest.raises(YOLO编辑会话错误, match="超出原图范围"):
        创建YOLO编辑会话(tmp_path / "bad_range", image, out_of_range)

    non_finite = [{"category": "T", "score": 0.9, "box": (float("nan"), 10.0, 40.0, 50.0)}]
    with pytest.raises(YOLO编辑会话错误, match="非有限数值"):
        创建YOLO编辑会话(tmp_path / "bad_nan", image, non_finite)


def test_parent_rejects_stale_session_id_and_tampered_commit(tmp_path):
    image, detections = _make_image_and_detections()
    manifest_path = 创建YOLO编辑会话(tmp_path, image, detections)
    _manifest, _image = 读取YOLO编辑会话(manifest_path)
    commit_path = 提交YOLO编辑结果(manifest_path, detections)

    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    commit["session_id"] = "旧会话"
    commit_path.write_text(json.dumps(commit, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(YOLO编辑会话错误, match="session_id"):
        读取已提交YOLO检测(manifest_path)

    提交YOLO编辑结果(manifest_path, detections)
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    commit["detections"][0]["category"] = "board"
    commit_path.write_text(json.dumps(commit, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(YOLO编辑会话错误, match="非法"):
        读取已提交YOLO检测(manifest_path)

    提交YOLO编辑结果(manifest_path, detections)
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    commit["detections"][0]["index"] = 5
    commit_path.write_text(json.dumps(commit, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(YOLO编辑会话错误, match="序号错误"):
        读取已提交YOLO检测(manifest_path)


def test_parent_rejects_missing_commit(tmp_path):
    image, detections = _make_image_and_detections()
    manifest_path = 创建YOLO编辑会话(tmp_path, image, detections)
    with pytest.raises(YOLO编辑会话错误, match="commit.json"):
        读取已提交YOLO检测(manifest_path)


def test_commit_rejects_unknown_source_and_bad_score(tmp_path):
    image, detections = _make_image_and_detections()
    manifest_path = 创建YOLO编辑会话(tmp_path, image, detections)

    bad_source = [{"category": "T", "score": 0.9, "box": (10, 10, 40, 40), "source": "other"}]
    with pytest.raises(YOLO编辑会话错误, match="来源非法"):
        提交YOLO编辑结果(manifest_path, bad_source)

    bad_score = [{"category": "T", "score": float("nan"), "box": (10, 10, 40, 40), "source": "yolo"}]
    with pytest.raises(YOLO编辑会话错误, match="分数"):
        提交YOLO编辑结果(manifest_path, bad_score)
