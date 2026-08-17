"""V2 角度锁定编辑会话协议的往返与严格校验测试。"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from image_process_lib.angle_lock_edit_session import (
    编辑取消退出码,
    角度锁定会话环境变量,
    角度锁定编辑会话错误,
    创建角度锁定编辑会话,
    提交角度锁定结果,
    读取已提交角度锁定,
    读取角度锁定编辑会话,
)


def _示例图像():
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    image[20:40, 30:50] = 200
    return image


def _示例方块():
    return [{
        "category": "T",
        "score": 0.9,
        "detection_box": (28.0, 18.0, 52.0, 42.0),
        "theta": -30.0,
        "px": 40.0,
        "py": 30.0,
        "center_px": 40.0,
        "center_py": 30.0,
    }]


def _示例配置():
    return {"angle_step_deg": 2.0, "canny_low": 50, "canny_high": 150}


def _示例几何():
    return {"block_px": 18, "connector_px": 4}


def test_会话创建与读取往返(tmp_path):
    manifest_path = 创建角度锁定编辑会话(
        tmp_path, _示例图像(), _示例方块(), _示例几何(), _示例配置()
    )
    assert manifest_path.name == "session.json"
    manifest, image = 读取角度锁定编辑会话(manifest_path)
    assert manifest["blocks"][0]["category"] == "T"
    assert manifest["blocks"][0]["index"] == 1
    assert manifest["v2_config"]["angle_step_deg"] == 2.0
    assert image.shape == (60, 80, 3)


def test_提交与读取锁定往返(tmp_path):
    manifest_path = 创建角度锁定编辑会话(
        tmp_path, _示例图像(), _示例方块(), _示例几何(), _示例配置()
    )
    提交角度锁定结果(manifest_path, {1: 15.0})
    assert 读取已提交角度锁定(manifest_path) == {1: 15.0}

    # 空提交合法：用户看了一眼没改。
    提交角度锁定结果(manifest_path, {})
    assert 读取已提交角度锁定(manifest_path) == {}


def test_非法方块字段被拒绝(tmp_path):
    for bad_block in (
        [{**_示例方块()[0], "category": "board"}],
        [{**_示例方块()[0], "theta": 180.0}],
        [{**_示例方块()[0], "detection_box": (1.0, 2.0, 3.0)}],
        [{**_示例方块()[0], "px": float("nan")}],
    ):
        with pytest.raises(角度锁定编辑会话错误):
            创建角度锁定编辑会话(
                tmp_path / "bad", _示例图像(), bad_block, _示例几何(), _示例配置()
            )
    with pytest.raises(角度锁定编辑会话错误):
        创建角度锁定编辑会话(
            tmp_path / "bad", _示例图像(), [], _示例几何(), _示例配置()
        )


def test_非法锁定提交被拒绝(tmp_path):
    manifest_path = 创建角度锁定编辑会话(
        tmp_path, _示例图像(), _示例方块(), _示例几何(), _示例配置()
    )
    with pytest.raises(角度锁定编辑会话错误):
        提交角度锁定结果(manifest_path, {2: 10.0})  # 序号不在本轮方块里
    with pytest.raises(角度锁定编辑会话错误):
        提交角度锁定结果(manifest_path, {1: 200.0})  # 角度越界
    with pytest.raises(角度锁定编辑会话错误):
        提交角度锁定结果(manifest_path, {1: "abc"})


def test_会话身份不一致被拒绝(tmp_path):
    manifest_path = 创建角度锁定编辑会话(
        tmp_path, _示例图像(), _示例方块(), _示例几何(), _示例配置()
    )
    提交角度锁定结果(manifest_path, {1: 10.0})
    commit_path = Path(manifest_path).parent / "commit.json"
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    commit["session_id"] = "tampered"
    commit_path.write_text(json.dumps(commit), encoding="utf-8")
    with pytest.raises(角度锁定编辑会话错误):
        读取已提交角度锁定(manifest_path)


def test_缺失提交与损坏会话被拒绝(tmp_path):
    manifest_path = 创建角度锁定编辑会话(
        tmp_path, _示例图像(), _示例方块(), _示例几何(), _示例配置()
    )
    with pytest.raises(角度锁定编辑会话错误):
        读取已提交角度锁定(manifest_path)  # 没有 commit.json
    session_path = Path(manifest_path)
    session_path.write_text("not json", encoding="utf-8")
    with pytest.raises(角度锁定编辑会话错误):
        读取角度锁定编辑会话(manifest_path)


def test_常量约定():
    assert 角度锁定会话环境变量 == "SINGLE_ARM_TETRIS_ANGLE_LOCK_SESSION"
    assert 编辑取消退出码 == 2
