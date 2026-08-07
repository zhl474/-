"""高位 Mask 手动编辑 Demo 的纯数据处理核心。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np


支持的图片后缀 = {".jpg", ".jpeg", ".png"}


def 查找唯一输入图片(input_dir: Path) -> Path:
    """在输入目录中查找唯一一张 JPG 或 PNG 图片。"""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"输入目录不存在：{input_dir}")

    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in 支持的图片后缀
    )
    if not image_paths:
        raise ValueError(
            f"输入目录中没有图片，请放入一张 jpg、jpeg 或 png：{input_dir}"
        )
    if len(image_paths) > 1:
        names = "、".join(path.name for path in image_paths)
        raise ValueError(
            f"输入目录中只能放一张图片，当前找到 {len(image_paths)} 张：{names}"
        )
    return image_paths[0]


def 转为二值mask(mask: np.ndarray) -> np.ndarray:
    """把任意非零前景转换成单通道 0/255 uint8 Mask。"""
    if mask is None or mask.size == 0:
        raise ValueError("Mask 为空")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.ndim != 2:
        raise ValueError("Mask 必须是二维单通道图像")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def 在mask上画圆(
    mask: np.ndarray,
    center_xy: Tuple[int, int],
    radius: int,
    value: int,
) -> None:
    """在 Mask 上原地绘制圆形画笔，OpenCV 会自动裁剪越界部分。"""
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("可编辑 Mask 必须是 uint8 单通道图像")
    radius = int(radius)
    if radius <= 0:
        raise ValueError("画笔半径必须为正整数")
    if value not in (0, 255):
        raise ValueError("画笔值只能是 0 或 255")

    x, y = int(center_xy[0]), int(center_xy[1])
    cv2.circle(mask, (x, y), radius, int(value), thickness=-1, lineType=cv2.LINE_8)


def 计算适配显示尺寸(
    image_shape: Iterable[int],
    max_width: int,
    max_height: int,
    max_scale: float = 1.0,
) -> Tuple[int, int]:
    """按最大窗口尺寸等比例缩放，返回（显示宽，显示高）。"""
    shape = tuple(image_shape)
    if len(shape) < 2:
        raise ValueError("图像尺寸无效")
    image_h, image_w = int(shape[0]), int(shape[1])
    if image_w <= 0 or image_h <= 0:
        raise ValueError("图像宽高必须为正数")
    if max_width <= 0 or max_height <= 0 or max_scale <= 0:
        raise ValueError("显示尺寸和最大缩放倍数必须为正数")

    scale = min(
        float(max_width) / image_w,
        float(max_height) / image_h,
        float(max_scale),
    )
    display_w = max(1, int(round(image_w * scale)))
    display_h = max(1, int(round(image_h * scale)))
    return display_w, display_h


def 显示坐标转图像坐标(
    display_xy: Tuple[int, int],
    image_shape: Iterable[int],
    display_shape: Iterable[int],
) -> Tuple[int, int]:
    """把缩放窗口中的鼠标坐标映射回原图坐标，并裁剪到有效范围。"""
    image_values = tuple(image_shape)
    display_values = tuple(display_shape)
    image_h, image_w = int(image_values[0]), int(image_values[1])
    display_h, display_w = int(display_values[0]), int(display_values[1])
    if min(image_h, image_w, display_h, display_w) <= 0:
        raise ValueError("图像尺寸或显示尺寸无效")

    display_x, display_y = float(display_xy[0]), float(display_xy[1])
    image_x = int(np.floor(display_x * image_w / display_w))
    image_y = int(np.floor(display_y * image_h / display_h))
    image_x = int(np.clip(image_x, 0, image_w - 1))
    image_y = int(np.clip(image_y, 0, image_h - 1))
    return image_x, image_y


def 叠加mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    """在 BGR 图像上半透明叠加二值 Mask。"""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("底图为空")
    binary_mask = 转为二值mask(mask)
    if binary_mask.shape != image_bgr.shape[:2]:
        raise ValueError("底图与 Mask 尺寸不一致")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("Mask 透明度必须在 0 到 1 之间")

    output = image_bgr.copy()
    foreground = binary_mask > 0
    if not np.any(foreground):
        return output
    colored = output.copy()
    colored[foreground] = np.asarray(color, dtype=np.uint8)
    blended = cv2.addWeighted(output, 1.0 - alpha, colored, alpha, 0.0)
    output[foreground] = blended[foreground]
    return output


def 保存二值mask(path: Path, mask: np.ndarray) -> Path:
    """把 Mask 按无损单通道 PNG 保存，并返回最终路径。"""
    path = Path(path)
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    binary_mask = 转为二值mask(mask)
    if not cv2.imwrite(str(path), binary_mask):
        raise RuntimeError(f"Mask 保存失败：{path}")
    return path


@dataclass
class Mask编辑历史:
    """保存当前 Mask、原始 Seg 和按笔撤销历史。"""

    original_mask: np.ndarray
    current_mask: np.ndarray | None = None
    max_undo_count: int = 30

    def __post_init__(self) -> None:
        self.original_mask = 转为二值mask(self.original_mask)
        if self.current_mask is None:
            self.current_mask = self.original_mask.copy()
        else:
            self.current_mask = 转为二值mask(self.current_mask)
        if self.current_mask.shape != self.original_mask.shape:
            raise ValueError("当前 Mask 与原始 Mask 尺寸不一致")
        self.max_undo_count = max(1, int(self.max_undo_count))
        self._undo_stack: list[np.ndarray] = []
        self._redo_stack: list[np.ndarray] = []

    def 开始一笔(self) -> None:
        """在一次连续鼠标绘制开始前保存快照。"""
        self._undo_stack.append(self.current_mask.copy())
        if len(self._undo_stack) > self.max_undo_count:
            self._undo_stack.pop(0)
        # 撤销后开始新的绘制分支时，旧的重做历史不再有效。
        self._redo_stack.clear()

    def 撤销(self) -> bool:
        """撤销上一笔；没有历史时返回 False。"""
        if not self._undo_stack:
            return False
        self._redo_stack.append(self.current_mask.copy())
        if len(self._redo_stack) > self.max_undo_count:
            self._redo_stack.pop(0)
        self.current_mask = self._undo_stack.pop()
        return True

    def 重做(self) -> bool:
        """重新应用最近一次被撤销的 Mask 状态。"""
        if not self._redo_stack:
            return False
        self._undo_stack.append(self.current_mask.copy())
        if len(self._undo_stack) > self.max_undo_count:
            self._undo_stack.pop(0)
        self.current_mask = self._redo_stack.pop()
        return True

    def 恢复原始(self) -> bool:
        """恢复模型最初产生的 Seg Mask。"""
        if np.array_equal(self.current_mask, self.original_mask):
            return False
        self.开始一笔()
        self.current_mask = self.original_mask.copy()
        return True
