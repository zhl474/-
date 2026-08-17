"""V2 角度锁定人工编辑子进程使用的临时文件会话协议。

与 yolo_edit_session.py 同构：父进程把本轮高位原图、V2 识别结果和
模板几何写入独立临时目录，GUI 子进程在无 ROS 环境中逐块锁定角度
（锁定后父进程按锁定角度重匹配位置），用 commit.json 原子提交，
父进程只认可通过严格校验的提交。
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
角度锁定会话环境变量 = "SINGLE_ARM_TETRIS_ANGLE_LOCK_SESSION"
编辑取消退出码 = 2


class 角度锁定编辑会话错误(RuntimeError):
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
        raise 角度锁定编辑会话错误(f"无法读取会话 JSON：{path}，原因：{exc}") from exc
    if not isinstance(data, dict):
        raise 角度锁定编辑会话错误(f"会话 JSON 顶层必须是字典：{path}")
    return data


def _会话内路径(session_dir: Path, relative_path: str) -> Path:
    """解析清单路径，并禁止通过相对路径越出临时会话目录。"""
    session_dir = Path(session_dir).resolve()
    path = (session_dir / str(relative_path)).resolve()
    if path != session_dir and session_dir not in path.parents:
        raise 角度锁定编辑会话错误(f"会话文件路径越界：{relative_path}")
    return path


def _检查类别(category, name="方块类别") -> str:
    category = str(category).strip()
    if category not in BLOCK_CATEGORY_NAMES:
        raise 角度锁定编辑会话错误(
            f"{name} 非法：{category}，必须是 {list(BLOCK_CATEGORY_NAMES)}"
        )
    return category


def _检查角度(theta, name="角度") -> float:
    try:
        value = float(theta)
    except (TypeError, ValueError) as exc:
        raise 角度锁定编辑会话错误(f"{name} 必须是数值") from exc
    if not np.isfinite(value) or value < -180.0 or value >= 180.0:
        raise 角度锁定编辑会话错误(f"{name} 必须在 [-180, 180) 内，实际 {theta}")
    return value


def _检查数值(value, name) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise 角度锁定编辑会话错误(f"{name} 必须是数值") from exc
    if not np.isfinite(number):
        raise 角度锁定编辑会话错误(f"{name} 不是有限数值")
    return number


def _检查几何(template_geometry) -> dict:
    if not isinstance(template_geometry, dict):
        raise 角度锁定编辑会话错误("template_geometry 必须是字典")
    result = dict(template_geometry)
    try:
        result["block_px"] = int(template_geometry["block_px"])
        result["connector_px"] = int(template_geometry["connector_px"])
    except (KeyError, TypeError, ValueError) as exc:
        raise 角度锁定编辑会话错误(
            "template_geometry 必须包含正整数 block_px/connector_px"
        ) from exc
    if result["block_px"] <= 0 or result["connector_px"] <= 0:
        raise 角度锁定编辑会话错误("block_px/connector_px 必须为正数")
    return result


def 创建角度锁定编辑会话(
    session_dir: Path,
    image_bgr: np.ndarray,
    blocks: Sequence[dict],
    template_geometry: dict,
    v2_config: dict,
) -> Path:
    """把本轮高位原图、V2 识别结果和模板几何写入独立临时目录。

    v2_config 是 EdgeTemplateConfig 的完整字段字典；GUI 预览与父进程
    重匹配必须用同一份配置，距离场和角度步进才一致。
    """
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    if image_bgr is None or image_bgr.size == 0 or image_bgr.ndim != 3:
        raise 角度锁定编辑会话错误("角度锁定会话原图为空或格式错误")
    if not blocks:
        raise 角度锁定编辑会话错误("本轮没有可编辑的方块")
    if not isinstance(v2_config, dict):
        raise 角度锁定编辑会话错误("v2_config 必须是字典")

    image_path = session_dir / "scene.png"
    if not cv2.imwrite(str(image_path), image_bgr):
        raise 角度锁定编辑会话错误(f"高位原图写入失败：{image_path}")
    image_shape = list(image_bgr.shape)

    records = []
    for offset, block in enumerate(blocks, start=1):
        category = _检查类别(block.get("category"), f"第 {offset} 个方块类别")
        theta = _检查角度(block.get("theta"), f"第 {offset} 个方块角度")
        box = tuple(
            _检查数值(value, f"第 {offset} 个方块检测框坐标")
            for value in block.get("detection_box", ())
        )
        if len(box) != 4:
            raise 角度锁定编辑会话错误(f"第 {offset} 个方块检测框必须包含四个数值")
        records.append({
            "index": offset,
            "category": category,
            "score": _检查数值(block.get("score", 1.0), f"第 {offset} 个方块分数"),
            "box": list(box),
            "theta": theta,
            "px": _检查数值(block.get("px"), f"第 {offset} 个方块输出点X"),
            "py": _检查数值(block.get("py"), f"第 {offset} 个方块输出点Y"),
            "center_px": _检查数值(block.get("center_px"), f"第 {offset} 个方块中心X"),
            "center_py": _检查数值(block.get("center_py"), f"第 {offset} 个方块中心Y"),
        })

    manifest = {
        "version": 会话协议版本,
        "session_id": uuid.uuid4().hex,
        "image_path": image_path.name,
        "image_shape": image_shape,
        "classes": list(BLOCK_CATEGORY_NAMES),
        "template_geometry": _检查几何(template_geometry),
        "v2_config": dict(v2_config),
        "blocks": records,
    }
    manifest_path = session_dir / "session.json"
    _写入JSON原子文件(manifest_path, manifest)
    return manifest_path


def 读取角度锁定编辑会话(manifest_path: Path):
    """供 GUI 子进程读取并严格校验一轮编辑输入。"""
    manifest_path = Path(manifest_path).resolve()
    session_dir = manifest_path.parent
    manifest = _读取JSON(manifest_path)
    if manifest.get("version") != 会话协议版本:
        raise 角度锁定编辑会话错误(f"不支持的会话版本：{manifest.get('version')}")
    if not str(manifest.get("session_id", "")).strip():
        raise 角度锁定编辑会话错误("会话缺少 session_id")
    records = manifest.get("blocks")
    if not isinstance(records, list) or not records:
        raise 角度锁定编辑会话错误("会话没有方块记录")

    image_path = _会话内路径(session_dir, manifest.get("image_path", ""))
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None or image_bgr.size == 0:
        raise 角度锁定编辑会话错误(f"高位原图读取失败：{image_path}")
    if list(image_bgr.shape) != list(manifest.get("image_shape", [])):
        raise 角度锁定编辑会话错误("高位原图尺寸与会话清单不一致")

    _检查几何(manifest.get("template_geometry"))
    if not isinstance(manifest.get("v2_config"), dict):
        raise 角度锁定编辑会话错误("会话缺少 v2_config 字典")
    for expected_index, record in enumerate(records, start=1):
        if not isinstance(record, dict) or record.get("index") != expected_index:
            raise 角度锁定编辑会话错误("方块序号不连续或记录格式错误")
        _检查类别(record.get("category"), "会话方块类别")
        _检查角度(record.get("theta"), "会话方块角度")
        if len(tuple(record.get("box", ()))) != 4:
            raise 角度锁定编辑会话错误("会话方块检测框格式错误")
    return manifest, image_bgr


def 提交角度锁定结果(manifest_path: Path, angle_locks: dict) -> Path:
    """GUI 子进程最后写提交标记；父进程只认可完整提交。

    angle_locks: {方块序号(1 起): 正式 theta}，只包含被人工锁定的方块；
    允许为空（用户看了一眼没改），父进程按原结果继续。
    """
    manifest, _image_bgr = 读取角度锁定编辑会话(manifest_path)
    session_dir = Path(manifest_path).resolve().parent
    known_indexes = {record["index"] for record in manifest["blocks"]}
    commit_records = []
    for raw_index, raw_theta in dict(angle_locks or {}).items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise 角度锁定编辑会话错误(f"锁定序号必须是整数：{raw_index}") from exc
        if index not in known_indexes:
            raise 角度锁定编辑会话错误(f"锁定序号不在本轮方块里：{index}")
        commit_records.append({
            "index": index,
            "theta": _检查角度(raw_theta, f"第 {index} 个锁定角度"),
        })

    commit = {
        "version": 会话协议版本,
        "session_id": manifest["session_id"],
        "angle_locks": commit_records,
    }
    commit_path = session_dir / "commit.json"
    _写入JSON原子文件(commit_path, commit)
    return commit_path


def 读取已提交角度锁定(manifest_path: Path) -> dict:
    """ROS 父进程校验提交身份后返回 {方块序号(1 起): 正式 theta}。"""
    manifest, _image_bgr = 读取角度锁定编辑会话(manifest_path)
    session_dir = Path(manifest_path).resolve().parent
    commit = _读取JSON(session_dir / "commit.json")
    if commit.get("version") != 会话协议版本:
        raise 角度锁定编辑会话错误("提交版本与会话协议不一致")
    if commit.get("session_id") != manifest.get("session_id"):
        raise 角度锁定编辑会话错误("提交 session_id 与本轮会话不一致")
    commit_records = commit.get("angle_locks")
    if not isinstance(commit_records, list):
        raise 角度锁定编辑会话错误("提交锁定记录格式错误")

    known_indexes = {record["index"] for record in manifest["blocks"]}
    result = {}
    for record in commit_records:
        if not isinstance(record, dict):
            raise 角度锁定编辑会话错误("提交锁定记录格式错误")
        index = int(record.get("index", -1))
        if index not in known_indexes:
            raise 角度锁定编辑会话错误(f"锁定序号不在本轮方块里：{index}")
        if index in result:
            raise 角度锁定编辑会话错误(f"锁定序号重复：{index}")
        result[index] = _检查角度(record.get("theta"), f"第 {index} 个锁定角度")
    return result
