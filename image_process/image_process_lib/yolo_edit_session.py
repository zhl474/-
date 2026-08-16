"""YOLO 检测框人工修正子进程使用的临时文件会话协议。

与 high_mask_edit_session.py 同构：父进程把本轮高位原图和 YOLO 原始
检测框写入独立临时目录，GUI 子进程在无 ROS 环境中编辑（删框/画框/
改类别），用 commit.json 原子提交，父进程只认可通过严格校验的提交。
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from image_process_lib.block_category import BLOCK_CATEGORY_NAMES


会话协议版本 = 1
YOLO会话环境变量 = "SINGLE_ARM_TETRIS_YOLO_EDIT_SESSION"
编辑取消退出码 = 2
# 修正框在原图内的最小边长（像素），过小说明是误操作。
最小框边长 = 16.0


class YOLO编辑会话错误(RuntimeError):
    """会话缺失、损坏或与本轮识别不一致。"""


def _写入JSON原子文件(path: Path, data: dict) -> None:
    """先写临时文件再原子替换，避免父进程读到半份提交。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(str(temp_path), str(path))
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _读取JSON(path: Path) -> dict:
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, ValueError) as exc:
        raise YOLO编辑会话错误(f"无法读取会话 JSON：{path}，原因：{exc}") from exc
    if not isinstance(data, dict):
        raise YOLO编辑会话错误(f"会话 JSON 顶层必须是字典：{path}")
    return data


def _会话内路径(session_dir: Path, relative_path: str) -> Path:
    """解析清单路径，并禁止通过相对路径越出临时会话目录。"""
    session_dir = Path(session_dir).resolve()
    path = (session_dir / str(relative_path)).resolve()
    if path != session_dir and session_dir not in path.parents:
        raise YOLO编辑会话错误(f"会话文件路径越界：{relative_path}")
    return path


def _写入PNG(path: Path, image: np.ndarray, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise YOLO编辑会话错误(f"{name}写入失败：{path}")


def _检查检测框(box, image_shape=None, name="检测框") -> tuple:
    """校验单个 xyxy 框：四个有限数值、x2>x1、y2>y1、尺寸和在图内可选。"""
    try:
        result = tuple(float(value) for value in box)
    except (TypeError, ValueError) as exc:
        raise YOLO编辑会话错误(f"{name} 必须包含四个数值") from exc
    if len(result) != 4:
        raise YOLO编辑会话错误(f"{name} 必须包含四个数值")
    if not np.all(np.isfinite(np.asarray(result, dtype=float))):
        raise YOLO编辑会话错误(f"{name} 包含非有限数值")
    x1, y1, x2, y2 = result
    if x2 - x1 < 最小框边长 or y2 - y1 < 最小框边长:
        raise YOLO编辑会话错误(
            f"{name} 尺寸过小：边长必须不小于 {最小框边长} 像素，实际 {result}"
        )
    if image_shape is not None:
        image_h, image_w = int(image_shape[0]), int(image_shape[1])
        if x1 < 0.0 or y1 < 0.0 or x2 > float(image_w) or y2 > float(image_h):
            raise YOLO编辑会话错误(
                f"{name} 超出原图范围：{result}，原图 {image_w}x{image_h}"
            )
    return result


def _检查类别(category, name="方块类别") -> str:
    category = str(category).strip()
    if category not in BLOCK_CATEGORY_NAMES:
        raise YOLO编辑会话错误(
            f"{name} 非法：{category}，必须是 {list(BLOCK_CATEGORY_NAMES)}"
        )
    return category


def 创建YOLO编辑会话(
    session_dir: Path,
    image_bgr: np.ndarray,
    detections: Sequence[dict],
) -> Path:
    """把本轮固定高位图和 YOLO 原始检测框写入独立临时目录。"""
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    if image_bgr is None or image_bgr.size == 0 or image_bgr.ndim != 3:
        raise YOLO编辑会话错误("高位会话原图为空或格式错误")

    image_path = session_dir / "scene.png"
    _写入PNG(image_path, image_bgr, "高位原图")
    image_shape = list(image_bgr.shape)
    records = []
    for offset, detection in enumerate(detections, start=1):
        category = _检查类别(detection.get("category"), f"第 {offset} 个检测类别")
        box = _检查检测框(
            detection.get("box", ()),
            image_shape,
            f"第 {offset} 个检测框",
        )
        try:
            score = float(detection.get("score", 0.0))
        except (TypeError, ValueError) as exc:
            raise YOLO编辑会话错误(f"第 {offset} 个检测分数必须是数值") from exc
        if not np.isfinite(score):
            raise YOLO编辑会话错误(f"第 {offset} 个检测分数不是有限数值")
        records.append({
            "index": offset,
            "category": category,
            "score": score,
            "box": list(box),
            "source": "yolo",
        })

    manifest = {
        "version": 会话协议版本,
        "session_id": uuid.uuid4().hex,
        "image_path": image_path.name,
        "image_shape": image_shape,
        "classes": list(BLOCK_CATEGORY_NAMES),
        "detections": records,
    }
    manifest_path = session_dir / "session.json"
    _写入JSON原子文件(manifest_path, manifest)
    return manifest_path


def 读取YOLO编辑会话(manifest_path: Path):
    """供 GUI 子进程读取并严格校验一轮编辑输入。"""
    manifest_path = Path(manifest_path).resolve()
    session_dir = manifest_path.parent
    manifest = _读取JSON(manifest_path)
    if manifest.get("version") != 会话协议版本:
        raise YOLO编辑会话错误(f"不支持的会话版本：{manifest.get('version')}")
    if not str(manifest.get("session_id", "")).strip():
        raise YOLO编辑会话错误("会话缺少 session_id")
    records = manifest.get("detections")
    if not isinstance(records, list):
        raise YOLO编辑会话错误("会话没有检测框记录")

    image_path = _会话内路径(session_dir, manifest.get("image_path", ""))
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None or image_bgr.size == 0:
        raise YOLO编辑会话错误(f"高位原图读取失败：{image_path}")
    if list(image_bgr.shape) != list(manifest.get("image_shape", [])):
        raise YOLO编辑会话错误("高位原图尺寸与会话清单不一致")

    classes = manifest.get("classes")
    if classes != list(BLOCK_CATEGORY_NAMES):
        raise YOLO编辑会话错误("会话类别列表与当前代码不一致")

    for expected_index, record in enumerate(records, start=1):
        if not isinstance(record, dict) or record.get("index") != expected_index:
            raise YOLO编辑会话错误("检测框序号不连续或记录格式错误")
        _检查类别(record.get("category"), "会话检测类别")
        _检查检测框(record.get("box", ()), image_bgr.shape, "会话检测框")
    return manifest, image_bgr


def 提交YOLO编辑结果(manifest_path: Path, detections: Sequence[dict]) -> Path:
    """GUI 子进程最后写提交标记；父进程只认可完整提交。"""
    manifest, _image_bgr = 读取YOLO编辑会话(manifest_path)
    session_dir = Path(manifest_path).resolve().parent
    commit_records = []
    for offset, detection in enumerate(detections, start=1):
        category = _检查类别(detection.get("category"), f"第 {offset} 个提交类别")
        box = _检查检测框(
            detection.get("box", ()),
            manifest["image_shape"],
            f"第 {offset} 个提交框",
        )
        source = str(detection.get("source", "manual"))
        if source not in ("yolo", "manual"):
            raise YOLO编辑会话错误(f"第 {offset} 个提交来源非法：{source}")
        try:
            score = float(detection.get("score", 1.0))
        except (TypeError, ValueError) as exc:
            raise YOLO编辑会话错误(f"第 {offset} 个提交分数必须是数值") from exc
        if not np.isfinite(score):
            raise YOLO编辑会话错误(f"第 {offset} 个提交分数不是有限数值")
        commit_records.append({
            "index": offset,
            "category": category,
            "box": list(box),
            "source": source,
            "score": score,
        })

    commit = {
        "version": 会话协议版本,
        "session_id": manifest["session_id"],
        "detections": commit_records,
    }
    commit_path = session_dir / "commit.json"
    _写入JSON原子文件(commit_path, commit)
    return commit_path


def 读取已提交YOLO检测(manifest_path: Path) -> list[dict]:
    """ROS 父进程校验提交身份后返回修正的检测框列表。

    允许提交为空列表（用户删光了全部框，由下游按没有识别到方块报错）；
    不允许提交数量与本轮 YOLO 数量强绑定，因为修正本身可增可删。
    """
    manifest, _image_bgr = 读取YOLO编辑会话(manifest_path)
    session_dir = Path(manifest_path).resolve().parent
    commit = _读取JSON(session_dir / "commit.json")
    if commit.get("version") != 会话协议版本:
        raise YOLO编辑会话错误("提交版本与会话协议不一致")
    if commit.get("session_id") != manifest.get("session_id"):
        raise YOLO编辑会话错误("提交 session_id 与本轮会话不一致")
    commit_records = commit.get("detections")
    if not isinstance(commit_records, list):
        raise YOLO编辑会话错误("提交检测框记录格式错误")

    result = []
    for expected_index, record in enumerate(commit_records, start=1):
        if not isinstance(record, dict) or record.get("index") != expected_index:
            raise YOLO编辑会话错误("提交检测框序号错误")
        category = _检查类别(record.get("category"), "提交检测类别")
        box = _检查检测框(
            record.get("box", ()),
            manifest["image_shape"],
            "提交检测框",
        )
        try:
            score = float(record.get("score", 1.0))
        except (TypeError, ValueError) as exc:
            raise YOLO编辑会话错误("提交检测分数必须是数值") from exc
        if not np.isfinite(score):
            raise YOLO编辑会话错误("提交检测分数不是有限数值")
        result.append({
            "category": category,
            "score": score,
            "box": box,
            "source": str(record.get("source", "manual")),
        })
    return result
