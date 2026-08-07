"""高位 Mask 人工编辑子进程使用的临时文件会话协议。"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


会话协议版本 = 2
会话环境变量 = "SINGLE_ARM_TETRIS_HIGH_MASK_SESSION"
预览设备环境变量 = "SINGLE_ARM_TETRIS_HIGH_MASK_DEVICE"
编辑取消退出码 = 2


class 高位Mask会话错误(RuntimeError):
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
        raise 高位Mask会话错误(f"无法读取会话 JSON：{path}，原因：{exc}") from exc
    if not isinstance(data, dict):
        raise 高位Mask会话错误(f"会话 JSON 顶层必须是字典：{path}")
    return data


def _检查二值Mask(mask: np.ndarray, expected_shape=None) -> np.ndarray:
    if mask is None or mask.size == 0 or mask.ndim != 2:
        raise 高位Mask会话错误("Mask 必须是非空单通道图像")
    if expected_shape is not None and tuple(mask.shape) != tuple(expected_shape):
        raise 高位Mask会话错误(
            f"Mask 尺寸错误：期望 {tuple(expected_shape)}，实际 {tuple(mask.shape)}"
        )
    values = set(int(value) for value in np.unique(mask))
    if not values.issubset({0, 255}):
        raise 高位Mask会话错误(f"Mask 不是严格二值图，包含像素值：{sorted(values)}")
    return mask.astype(np.uint8, copy=True)


def _检查四元素序列(values: Iterable, name: str, cast_type=float) -> tuple:
    try:
        result = tuple(cast_type(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise 高位Mask会话错误(f"{name} 必须包含四个数值") from exc
    if len(result) != 4:
        raise 高位Mask会话错误(f"{name} 必须包含四个数值")
    if not np.all(np.isfinite(np.asarray(result, dtype=float))):
        raise 高位Mask会话错误(f"{name} 包含非有限数值")
    return result


def _会话内路径(session_dir: Path, relative_path: str) -> Path:
    """解析清单路径，并禁止通过相对路径越出临时会话目录。"""
    session_dir = Path(session_dir).resolve()
    path = (session_dir / str(relative_path)).resolve()
    if path != session_dir and session_dir not in path.parents:
        raise 高位Mask会话错误(f"会话文件路径越界：{relative_path}")
    return path


def _写入PNG(path: Path, image: np.ndarray, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise 高位Mask会话错误(f"{name}写入失败：{path}")


def _构建初始预览匹配(block: dict):
    """把父进程已经算好的矩形只用于子进程首屏预览。

    该数据不会从子进程返回，更不会作为最终定位结果。
    """
    rect = block.get("rect")
    if rect is None:
        return None
    try:
        center = tuple(float(value) for value in rect[0])
        size = tuple(float(value) for value in rect[1])
        rect_theta = float(rect[2])
        px = float(block["px"])
        py = float(block["py"])
        theta = float(block["theta"])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise 高位Mask会话错误("方块初始模板预览数据格式错误") from exc
    values = np.asarray([*center, *size, rect_theta, px, py, theta], dtype=float)
    if len(center) != 2 or len(size) != 2 or not np.all(np.isfinite(values)):
        raise 高位Mask会话错误("方块初始模板预览数据无效")
    return {
        "local_center": list(center),
        "rect_size": list(size),
        "rect_theta": rect_theta,
        "px": px,
        "py": py,
        "theta": theta,
    }


def 创建高位Mask编辑会话(
    session_dir: Path,
    image_bgr: np.ndarray,
    blocks: Sequence[dict],
) -> Path:
    """把本轮固定高位图和各实例原始 Mask 写入独立临时目录。"""
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    if image_bgr is None or image_bgr.size == 0 or image_bgr.ndim != 3:
        raise 高位Mask会话错误("高位会话原图为空或格式错误")
    if not blocks:
        raise 高位Mask会话错误("高位会话没有任何方块")

    image_path = session_dir / "scene.png"
    _写入PNG(image_path, image_bgr, "高位原图")
    image_h, image_w = image_bgr.shape[:2]
    records = []
    for offset, block in enumerate(blocks, start=1):
        category = str(block.get("category", "")).strip()
        if not category:
            raise 高位Mask会话错误(f"第 {offset} 个方块缺少类别")
        detection_box = _检查四元素序列(
            block.get("detection_box", ()),
            f"第 {offset} 个检测框",
            float,
        )
        crop_box = _检查四元素序列(
            block.get("crop_box", ()),
            f"第 {offset} 个裁剪框",
            int,
        )
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        if not (0 <= crop_x1 < crop_x2 <= image_w and 0 <= crop_y1 < crop_y2 <= image_h):
            raise 高位Mask会话错误(f"第 {offset} 个裁剪框超出原图：{crop_box}")
        expected_shape = (crop_y2 - crop_y1, crop_x2 - crop_x1)
        mask = _检查二值Mask(block.get("mask"), expected_shape)
        mask_relative_path = f"original/mask_{offset:03d}.png"
        _写入PNG(session_dir / mask_relative_path, mask, "原始 Mask")
        record = {
            "index": offset,
            "category": category,
            "detection_score": float(block.get("score", 0.0)),
            "detection_box": list(detection_box),
            "crop_box": list(crop_box),
            "mask_path": mask_relative_path,
        }
        preview_match = _构建初始预览匹配(block)
        if preview_match is not None:
            record["preview_match"] = preview_match
        records.append(record)

    manifest = {
        "version": 会话协议版本,
        "session_id": uuid.uuid4().hex,
        "image_path": image_path.name,
        "image_shape": list(image_bgr.shape),
        "blocks": records,
    }
    manifest_path = session_dir / "session.json"
    _写入JSON原子文件(manifest_path, manifest)
    return manifest_path


def 读取高位Mask编辑会话(manifest_path: Path):
    """供 GUI 子进程读取并严格校验一轮编辑输入。"""
    manifest_path = Path(manifest_path).resolve()
    session_dir = manifest_path.parent
    manifest = _读取JSON(manifest_path)
    if manifest.get("version") != 会话协议版本:
        raise 高位Mask会话错误(f"不支持的会话版本：{manifest.get('version')}")
    if not str(manifest.get("session_id", "")).strip():
        raise 高位Mask会话错误("会话缺少 session_id")
    records = manifest.get("blocks")
    if not isinstance(records, list) or not records:
        raise 高位Mask会话错误("会话没有方块记录")

    image_path = _会话内路径(session_dir, manifest.get("image_path", ""))
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None or image_bgr.size == 0:
        raise 高位Mask会话错误(f"高位原图读取失败：{image_path}")
    if list(image_bgr.shape) != list(manifest.get("image_shape", [])):
        raise 高位Mask会话错误("高位原图尺寸与会话清单不一致")

    masks = []
    for expected_index, record in enumerate(records, start=1):
        if not isinstance(record, dict) or record.get("index") != expected_index:
            raise 高位Mask会话错误("方块序号不连续或记录格式错误")
        crop_box = _检查四元素序列(record.get("crop_box", ()), "裁剪框", int)
        expected_shape = (crop_box[3] - crop_box[1], crop_box[2] - crop_box[0])
        mask_path = _会话内路径(session_dir, record.get("mask_path", ""))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        masks.append(_检查二值Mask(mask, expected_shape))
    return manifest, image_bgr, masks


def 提交高位Mask编辑结果(manifest_path: Path, masks: Sequence[np.ndarray]) -> Path:
    """GUI 子进程最后写提交标记；父进程只认可完整提交。"""
    manifest, _image_bgr, original_masks = 读取高位Mask编辑会话(manifest_path)
    records = manifest["blocks"]
    if len(masks) != len(records):
        raise 高位Mask会话错误(
            f"提交 Mask 数量错误：期望 {len(records)}，实际 {len(masks)}"
        )
    session_dir = Path(manifest_path).resolve().parent
    commit_records = []
    for record, mask, original_mask in zip(records, masks, original_masks):
        checked_mask = _检查二值Mask(mask, original_mask.shape)
        relative_path = f"result/mask_{int(record['index']):03d}.png"
        _写入PNG(session_dir / relative_path, checked_mask, "编辑后 Mask")
        commit_records.append({
            "index": int(record["index"]),
            "category": str(record["category"]),
            "mask_path": relative_path,
        })

    commit = {
        "version": 会话协议版本,
        "session_id": manifest["session_id"],
        "blocks": commit_records,
    }
    commit_path = session_dir / "commit.json"
    _写入JSON原子文件(commit_path, commit)
    return commit_path


def 读取已提交高位Mask(
    manifest_path: Path,
    expected_blocks: Sequence[dict],
) -> list[np.ndarray]:
    """ROS 父进程校验提交身份、实例顺序、类别和 Mask 尺寸。"""
    manifest, _image_bgr, original_masks = 读取高位Mask编辑会话(manifest_path)
    if len(expected_blocks) != len(manifest["blocks"]):
        raise 高位Mask会话错误("当前检测方块数量与会话不一致")
    session_dir = Path(manifest_path).resolve().parent
    commit = _读取JSON(session_dir / "commit.json")
    if commit.get("version") != 会话协议版本:
        raise 高位Mask会话错误("提交版本与会话协议不一致")
    if commit.get("session_id") != manifest.get("session_id"):
        raise 高位Mask会话错误("提交 session_id 与本轮会话不一致")
    commit_records = commit.get("blocks")
    if not isinstance(commit_records, list) or len(commit_records) != len(expected_blocks):
        raise 高位Mask会话错误("提交方块数量与当前检测不一致")

    result_masks = []
    for index, (manifest_record, commit_record, expected_block, original_mask) in enumerate(
        zip(manifest["blocks"], commit_records, expected_blocks, original_masks),
        start=1,
    ):
        if not isinstance(commit_record, dict) or commit_record.get("index") != index:
            raise 高位Mask会话错误("提交方块序号错误")
        expected_category = str(expected_block.get("category", ""))
        if (
            str(manifest_record.get("category")) != expected_category
            or str(commit_record.get("category")) != expected_category
        ):
            raise 高位Mask会话错误(f"第 {index} 个方块类别与当前检测不一致")
        expected_crop_box = _检查四元素序列(
            expected_block.get("crop_box", ()),
            f"第 {index} 个当前裁剪框",
            int,
        )
        manifest_crop_box = _检查四元素序列(
            manifest_record.get("crop_box", ()),
            f"第 {index} 个会话裁剪框",
            int,
        )
        if manifest_crop_box != expected_crop_box:
            raise 高位Mask会话错误(f"第 {index} 个方块裁剪框与当前检测不一致")
        expected_mask = _检查二值Mask(expected_block.get("mask"))
        crop_shape = (
            expected_crop_box[3] - expected_crop_box[1],
            expected_crop_box[2] - expected_crop_box[0],
        )
        if tuple(expected_mask.shape) != crop_shape:
            raise 高位Mask会话错误(
                f"第 {index} 个当前 Mask 尺寸与裁剪框不一致"
            )
        if tuple(original_mask.shape) != tuple(expected_mask.shape):
            raise 高位Mask会话错误(f"第 {index} 个会话 Mask 尺寸与当前检测不一致")
        mask_path = _会话内路径(session_dir, commit_record.get("mask_path", ""))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        result_masks.append(_检查二值Mask(mask, expected_mask.shape))
    return result_masks
