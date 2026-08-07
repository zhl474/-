#!/usr/bin/env python3
"""独立高位 Mask 手动编辑 Demo，不连接相机、ROS 或比赛流程。"""

from __future__ import annotations

import json
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk


# ============================ 可直接修改的参数 ============================

当前目录 = Path(__file__).resolve().parent
源码根目录 = 当前目录.parent.parent
输入目录 = 当前目录 / "input"
输出根目录 = 当前目录 / "output"

检测模型路径 = 源码根目录 / "competition" / "model" / "best5.14.pt"
分割模型路径 = 源码根目录 / "competition" / "model" / "best_seg.pt"

检测置信度 = 0.45
检测IOU阈值 = 0.50
分割置信度 = 0.25
编辑ROI外扩像素 = 30
最小有效MASK面积 = 10

初始画笔半径 = 1
最小画笔半径 = 1
最大画笔半径 = 80
画笔调整步长 = 2
最大撤销步数 = 30

MASK叠加透明度 = 0.35
模板轮廓线宽 = 1
总览最大宽度 = 1600
总览最大高度 = 900
编辑窗口最大宽度 = 1400
编辑窗口最大高度 = 900
编辑窗口最大放大倍数 = 4.0

# ========================================================================


# 让独立脚本能够只读复用项目里的类别、模板几何与模板匹配实现。
图像处理包目录 = 源码根目录 / "image_process"
if str(图像处理包目录) not in sys.path:
    sys.path.insert(0, str(图像处理包目录))

from image_process_lib.block_category import normalize_category_name  # noqa: E402
from image_process_lib.template_config import load_template_geometry  # noqa: E402
from image_process_lib.template_match.kernels_create import (  # noqa: E402
    ROTATION_TOTAL_ANGLE,
    create_rotation_kernels,
)
from image_process_lib.template_match.template_match import get_rect  # noqa: E402

from mask_editor_core import (  # noqa: E402
    Mask编辑历史,
    保存二值mask,
    在mask上画圆,
    叠加mask,
    显示坐标转图像坐标,
    查找唯一输入图片,
    计算适配显示尺寸,
    转为二值mask,
)


# OpenCV Qt 后端用窗口名称查找鼠标回调句柄；中文名称在部分版本中会导致
# namedWindow 已调用但 setMouseCallback 找不到窗口，因此内部标识统一使用 ASCII。
总览窗口名称 = "High Mask Editor - Overview - Ctrl+Click to edit"
编辑窗口名称 = "High Mask Editor - Paint"


def 是总览提交键(key: int) -> bool:
    """总览中 Enter 和 Q 都表示提交。"""
    return int(key) in (ord("q"), ord("Q"), 10, 13)


@dataclass
class 模板匹配结果:
    """保存一个局部 Mask 的模板匹配结果。"""

    local_px: float
    local_py: float
    global_px: float
    global_py: float
    theta: float
    score: float
    rect: tuple
    best_kernel: np.ndarray
    template_top_left: Tuple[float, float]

    def 转字典(self) -> dict:
        return {
            "局部中心": [self.local_px, self.local_py],
            "全图中心": [self.global_px, self.global_py],
            "角度": self.theta,
            "分数": self.score,
        }


@dataclass
class 方块编辑项:
    """保存检测、分割、编辑和模板匹配所需的全部局部数据。"""

    index: int
    category: str
    detection_score: float
    detection_box: Tuple[float, float, float, float]
    crop_box: Tuple[int, int, int, int]
    roi_bgr: np.ndarray
    original_mask: np.ndarray
    edited_mask: np.ndarray
    original_match: 模板匹配结果
    current_match: 模板匹配结果
    changed: bool = False


class 界面绘字器:
    """使用 OpenCV 内置字体绘制纯英文界面，避免系统中文字体兼容问题。"""

    def 绘制(self, image_bgr, text, org, color, font_size=22):
        output = image_bgr.copy()
        scale = max(0.3, float(font_size) / 30.0)
        position = (int(org[0]), int(org[1]) + int(font_size))
        # 先画黑色描边，保证浅色背景上的小字号仍清晰可见。
        cv2.putText(
            output,
            str(text),
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            str(text),
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            tuple(int(value) for value in color),
            1,
            cv2.LINE_AA,
        )
        return output


class 高位模板匹配器:
    """复用当前项目模板匹配，并按类别缓存旋转模板。"""

    def __init__(self, device: Optional[str] = None) -> None:
        geometry = load_template_geometry("high")
        self.block_px = int(geometry["block_px"])
        self.connector_px = int(geometry["connector_px"])
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = str(device).strip().lower()
        if device not in ("cpu", "cuda"):
            raise ValueError(f"模板匹配设备只支持 cpu 或 cuda，当前为：{device}")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA 模板匹配，但当前进程没有可用 CUDA")
        self.device = device
        self._prepared_cache: Dict[str, dict] = {}

    def _获取模板(self, category: str) -> dict:
        prepared = self._prepared_cache.get(category)
        if prepared is not None:
            return prepared
        kernels, kernel_size, angles = create_rotation_kernels(
            self.block_px,
            self.connector_px,
            category,
            device=self.device,
            angle_step=1.0,
        )
        prepared = {
            "kernels": kernels,
            "kernel_size": kernel_size,
            "angles": angles,
        }
        self._prepared_cache[category] = prepared
        return prepared

    def 根据父进程结果构建预览(
        self,
        preview_match: dict,
        category: str,
        crop_box: Tuple[int, int, int, int],
    ) -> 模板匹配结果:
        """使用父进程已算好的角度快速生成首屏黄色轮廓。

        这里只生成单个角度的模板，不做 CPU 全角度卷积。用户打开
        局部编辑器或修改 Mask 后，仍会调用 `匹配()` 重新计算。
        """
        try:
            local_px, local_py = (
                float(value) for value in preview_match["local_center"]
            )
            rect_size = tuple(float(value) for value in preview_match["rect_size"])
            rect_theta = float(preview_match["rect_theta"])
            global_px = float(preview_match["px"])
            global_py = float(preview_match["py"])
            theta = float(preview_match["theta"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("初始模板预览数据格式错误") from exc
        if len(rect_size) != 2:
            raise ValueError("初始模板矩形尺寸错误")

        # get_rect 返回的矩形角度与模板旋转角度符号相反。
        kernels, _kernel_size, _angles = create_rotation_kernels(
            self.block_px,
            self.connector_px,
            category,
            device=self.device,
            angle_values=[-rect_theta],
        )
        best_kernel = kernels[0, 0].detach().cpu().numpy().copy()
        template_top_left = (
            local_px - best_kernel.shape[1] // 2,
            local_py - best_kernel.shape[0] // 2,
        )
        crop_x1, crop_y1, _, _ = crop_box
        return 模板匹配结果(
            local_px=local_px,
            local_py=local_py,
            global_px=global_px if np.isfinite(global_px) else local_px + crop_x1,
            global_py=global_py if np.isfinite(global_py) else local_py + crop_y1,
            theta=theta,
            score=0.0,
            rect=((local_px, local_py), rect_size, rect_theta),
            best_kernel=best_kernel,
            template_top_left=template_top_left,
        )

    def 匹配(
        self,
        mask: np.ndarray,
        category: str,
        crop_box: Tuple[int, int, int, int],
    ) -> 模板匹配结果:
        binary_mask = 转为二值mask(mask)
        area = int(cv2.countNonZero(binary_mask))
        if area < int(最小有效MASK面积):
            raise ValueError(
                f"Mask 前景面积只有 {area} 像素，小于最低要求 {最小有效MASK面积}"
            )

        debug_output: dict = {}
        rect = get_rect(
            binary_mask,
            self.block_px,
            self.connector_px,
            category,
            None,
            0,
            0,
            angle_step=1.0,
            debug_output=debug_output,
            prepared_templates=self._获取模板(category),
        )
        crop_x1, crop_y1, _, _ = crop_box
        local_px, local_py = float(rect[0][0]), float(rect[0][1])
        theta = float(rect[2])
        if theta < -180.0:
            theta += 360.0
        return 模板匹配结果(
            local_px=local_px,
            local_py=local_py,
            global_px=local_px + crop_x1,
            global_py=local_py + crop_y1,
            theta=theta,
            score=float(debug_output.get("score", 0.0)),
            rect=rect,
            best_kernel=debug_output["best_kernel"].copy(),
            template_top_left=tuple(debug_output["template_top_left"]),
        )


def _类别名称(model, class_id: int) -> str:
    names = model.names
    if isinstance(names, dict):
        return str(names[int(class_id)])
    return str(names[int(class_id)])


def _分割上表面(seg_model, roi_bgr: np.ndarray) -> np.ndarray:
    """从一个检测 ROI 中选择面积最大的 top_surface Mask。"""
    results = seg_model(
        roi_bgr,
        conf=float(分割置信度),
        verbose=False,
        retina_masks=True,
    )
    if not results:
        raise RuntimeError("分割模型没有返回结果")
    result = results[0]
    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        raise RuntimeError("分割模型没有检测到上表面 Mask")

    top_surface_ids = {
        int(class_id)
        for class_id, class_name in (
            seg_model.names.items()
            if isinstance(seg_model.names, dict)
            else enumerate(seg_model.names)
        )
        if str(class_name) == "top_surface"
    }
    selected_indices = list(range(len(result.masks.data)))
    if top_surface_ids and result.boxes is not None and result.boxes.cls is not None:
        cls_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        selected_indices = [
            index
            for index, class_id in enumerate(cls_ids)
            if int(class_id) in top_surface_ids
        ]
    if not selected_indices:
        raise RuntimeError("分割结果中没有 top_surface 类别")

    best_mask = None
    best_area = -1
    for index in selected_indices:
        candidate = result.masks.data[index].detach().cpu().numpy()
        candidate = (candidate > 0.5).astype(np.uint8)
        area = int(candidate.sum())
        if area > best_area:
            best_mask = candidate
            best_area = area
    if best_mask is None or best_area <= 0:
        raise RuntimeError("分割模型返回了空 Mask")

    roi_h, roi_w = roi_bgr.shape[:2]
    if best_mask.shape != (roi_h, roi_w):
        best_mask = cv2.resize(
            best_mask,
            (roi_w, roi_h),
            interpolation=cv2.INTER_NEAREST,
        )
    return (best_mask * 255).astype(np.uint8)


def 检测并分割方块(
    image_bgr: np.ndarray,
    detector,
    seg_model,
    matcher: 高位模板匹配器,
) -> list[方块编辑项]:
    """对一张本地高位图片执行检测、分割和初始模板匹配。"""
    image_h, image_w = image_bgr.shape[:2]
    results = detector(
        image_bgr,
        iou=float(检测IOU阈值),
        conf=float(检测置信度),
        verbose=False,
    )
    detections = []
    if results and results[0].boxes is not None:
        for det in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = det
            category = normalize_category_name(_类别名称(detector, int(class_id)))
            if category == "board" or category not in ROTATION_TOTAL_ANGLE:
                continue
            detections.append((x1, y1, x2, y2, score, category))

    # 按从上到下、从左到右排序，让同一张图上的序号稳定易找。
    detections.sort(
        key=lambda item: (
            round((float(item[1]) + float(item[3])) / 2.0, 1),
            round((float(item[0]) + float(item[2])) / 2.0, 1),
        )
    )

    blocks: list[方块编辑项] = []
    for det in detections:
        x1, y1, x2, y2, score, category = det
        crop_x1 = max(0, int(np.floor(x1)) - int(编辑ROI外扩像素))
        crop_y1 = max(0, int(np.floor(y1)) - int(编辑ROI外扩像素))
        crop_x2 = min(image_w, int(np.ceil(x2)) + int(编辑ROI外扩像素))
        crop_y2 = min(image_h, int(np.ceil(y2)) + int(编辑ROI外扩像素))
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            continue

        roi_bgr = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)
        try:
            mask = _分割上表面(seg_model, roi_bgr)
            match = matcher.匹配(mask, category, crop_box)
        except Exception as exc:
            print(f"\033[91m跳过类别 {category}：{exc}\033[0m")
            continue

        index = len(blocks) + 1
        blocks.append(
            方块编辑项(
                index=index,
                category=category,
                detection_score=float(score),
                detection_box=(float(x1), float(y1), float(x2), float(y2)),
                crop_box=crop_box,
                roi_bgr=roi_bgr,
                original_mask=mask.copy(),
                edited_mask=mask.copy(),
                original_match=match,
                current_match=match,
            )
        )
    if not blocks:
        raise RuntimeError("没有得到可编辑的方块 Mask，请检查图片和模型输出")
    return blocks


def _绘制模板轮廓(
    image_bgr: np.ndarray,
    match: Optional[模板匹配结果],
    offset_xy: Tuple[int, int] = (0, 0),
    draw_center: bool = True,
) -> np.ndarray:
    """绘制实际最佳俄罗斯方块模板轮廓和匹配中心。"""
    output = image_bgr.copy()
    if match is None:
        return output
    contours, _ = cv2.findContours(
        转为二值mask(match.best_kernel),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    offset_x = int(round(float(offset_xy[0]) + match.template_top_left[0]))
    offset_y = int(round(float(offset_xy[1]) + match.template_top_left[1]))
    cv2.drawContours(
        output,
        contours,
        -1,
        (0, 255, 255),
        int(模板轮廓线宽),
        offset=(offset_x, offset_y),
    )
    if draw_center:
        cv2.drawMarker(
            output,
            (
                int(round(float(offset_xy[0]) + match.local_px)),
                int(round(float(offset_xy[1]) + match.local_py)),
            ),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=1,
        )
    return output


def _构建总览图(
    image_bgr: np.ndarray,
    blocks: list[方块编辑项],
    text_drawer: 界面绘字器,
    show_masks: bool = True,
) -> np.ndarray:
    overview = image_bgr.copy()
    for block in blocks:
        crop_x1, crop_y1, crop_x2, crop_y2 = block.crop_box
        if show_masks:
            roi = overview[crop_y1:crop_y2, crop_x1:crop_x2]
            roi[:] = 叠加mask(
                roi,
                block.edited_mask,
                color=(0, 255, 0),
                alpha=MASK叠加透明度,
            )

        overview = _绘制模板轮廓(
            overview,
            block.current_match,
            offset_xy=(crop_x1, crop_y1),
            draw_center=False,
        )
    overview = text_drawer.绘制(
        overview,
        "Ctrl+Left Click: edit | Wheel: zoom | T: masks on/off | Enter/Q: save",
        (12, 4),
        (0, 0, 255),
        font_size=17,
    )
    return overview


def _查找点击目标(blocks: list[方块编辑项], point_xy: Tuple[int, int]):
    x, y = point_xy
    candidates = []
    for block in blocks:
        x1, y1, x2, y2 = block.detection_box
        if x1 <= x <= x2 and y1 <= y <= y2:
            area = max(1.0, (x2 - x1) * (y2 - y1))
            candidates.append((area, block))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


class TkMask编辑器:
    """使用自控 Tk Canvas 分离拖动、缩放和 Mask 画笔事件。"""

    CTRL_MASK = 0x0004

    def __init__(self, block: 方块编辑项, matcher: 高位模板匹配器):
        self.block = block
        self.matcher = matcher
        self.history = Mask编辑历史(
            block.original_mask,
            current_mask=block.edited_mask,
            max_undo_count=最大撤销步数,
        )
        self.match: Optional[模板匹配结果] = block.current_match
        self.error = ""
        self.brush_radius = int(初始画笔半径)
        self.show_template = True
        self.drawing = False
        self.paint_value = 255
        self.last_image_point: Optional[Tuple[int, int]] = None
        self.accepted = False

        roi_h, roi_w = block.roi_bgr.shape[:2]
        display_w, display_h = 计算适配显示尺寸(
            block.roi_bgr.shape,
            编辑窗口最大宽度,
            编辑窗口最大高度,
            max_scale=编辑窗口最大放大倍数,
        )
        self.fit_scale = min(display_w / roi_w, display_h / roi_h)
        self.scale = self.fit_scale
        self.max_scale = max(self.fit_scale, self.fit_scale * 12.0)

        self.root = tk.Tk()
        self.root.title(编辑窗口名称)
        self.root.configure(background="#202020")

        self.status_label = tk.Label(
            self.root,
            anchor="w",
            background="#202020",
            foreground="#ffff00",
            font=("Sans", 10),
        )
        self.status_label.pack(fill="x", padx=6, pady=(4, 2))

        self.canvas = tk.Canvas(
            self.root,
            width=display_w,
            height=display_h,
            background="#202020",
            highlightthickness=0,
            cursor="fleur",
        )
        self.canvas.pack(fill="both", expand=True)

        self.help_label = tk.Label(
            self.root,
            text=(
                "Drag: pan | Wheel: zoom | Ctrl+LMB: draw | Ctrl+RMB: erase | "
                "Ctrl+Wheel/[ ]: brush | Z: undo | Y: redo | R: reset | 0: fit | "
                "T: template on/off | Enter: accept | Esc: cancel"
            ),
            anchor="w",
            background="#202020",
            foreground="#ffffff",
            font=("Sans", 9),
        )
        self.help_label.pack(fill="x", padx=6, pady=(2, 4))

        self.photo_image = None
        self.image_item = None
        self._绑定事件()
        self.root.protocol("WM_DELETE_WINDOW", self.取消)
        self.root.after(0, self.canvas.focus_set)
        # 总览中的当前匹配已经可用，打开局部窗口时不重复做全角度卷积。
        # 只有用户真正改变 Mask 、撤销、重做或恢复后才重新匹配。
        self._刷新画布()

    def _绑定事件(self) -> None:
        for button in (1, 3):
            self.canvas.bind(f"<ButtonPress-{button}>", self._鼠标按下)
            self.canvas.bind(f"<B{button}-Motion>", self._鼠标移动)
            self.canvas.bind(f"<ButtonRelease-{button}>", self._鼠标松开)
        self.canvas.bind("<MouseWheel>", self._滚轮)
        self.canvas.bind("<Button-4>", self._滚轮)
        self.canvas.bind("<Button-5>", self._滚轮)
        self.root.bind("<KeyPress-z>", self._撤销)
        self.root.bind("<KeyPress-Z>", self._撤销)
        self.root.bind("<KeyPress-y>", self._重做)
        self.root.bind("<KeyPress-Y>", self._重做)
        self.root.bind("<KeyPress-r>", self._恢复)
        self.root.bind("<KeyPress-R>", self._恢复)
        self.root.bind("<KeyPress-bracketleft>", self._减小画笔)
        self.root.bind("<KeyPress-bracketright>", self._增大画笔)
        self.root.bind("<KeyPress-0>", self._适配窗口)
        self.root.bind("<KeyPress-t>", self._切换模板轮廓)
        self.root.bind("<KeyPress-T>", self._切换模板轮廓)
        self.root.bind("<Return>", self.接受)
        self.root.bind("<Escape>", self.取消)

    @staticmethod
    def _滚轮方向(event) -> int:
        if getattr(event, "num", None) == 4:
            return 1
        if getattr(event, "num", None) == 5:
            return -1
        return 1 if getattr(event, "delta", 0) > 0 else -1

    def _事件图像坐标(self, event) -> Tuple[int, int]:
        canvas_x = float(self.canvas.canvasx(event.x))
        canvas_y = float(self.canvas.canvasy(event.y))
        image_h, image_w = self.history.current_mask.shape
        image_x = int(np.clip(np.floor(canvas_x / self.scale), 0, image_w - 1))
        image_y = int(np.clip(np.floor(canvas_y / self.scale), 0, image_h - 1))
        return image_x, image_y

    def _绘制到mask(self, image_point: Tuple[int, int]) -> None:
        if self.last_image_point is not None:
            cv2.line(
                self.history.current_mask,
                self.last_image_point,
                image_point,
                int(self.paint_value),
                thickness=max(1, int(self.brush_radius) * 2),
                lineType=cv2.LINE_8,
            )
        在mask上画圆(
            self.history.current_mask,
            image_point,
            self.brush_radius,
            self.paint_value,
        )
        self.last_image_point = image_point

    def _鼠标按下(self, event):
        if event.state & self.CTRL_MASK:
            self.history.开始一笔()
            self.drawing = True
            self.paint_value = 255 if event.num == 1 else 0
            self.last_image_point = None
            self.canvas.configure(cursor="crosshair")
            self._绘制到mask(self._事件图像坐标(event))
            self._刷新画布()
        else:
            self.drawing = False
            self.canvas.configure(cursor="fleur")
            self.canvas.scan_mark(event.x, event.y)
        return "break"

    def _鼠标移动(self, event):
        if self.drawing:
            self._绘制到mask(self._事件图像坐标(event))
            self._刷新画布()
        else:
            self.canvas.scan_dragto(event.x, event.y, gain=1)
        return "break"

    def _鼠标松开(self, event):
        if self.drawing:
            self._绘制到mask(self._事件图像坐标(event))
            self.drawing = False
            self.last_image_point = None
            self.canvas.configure(cursor="fleur")
            self._重新匹配并刷新()
        return "break"

    def _滚轮(self, event):
        direction = self._滚轮方向(event)
        if event.state & self.CTRL_MASK:
            self.brush_radius = int(
                np.clip(
                    self.brush_radius + direction * int(画笔调整步长),
                    int(最小画笔半径),
                    int(最大画笔半径),
                )
            )
            self._更新状态文字()
            return "break"

        image_x = float(self.canvas.canvasx(event.x)) / self.scale
        image_y = float(self.canvas.canvasy(event.y)) / self.scale
        zoom_factor = 1.2 if direction > 0 else 1.0 / 1.2
        new_scale = float(np.clip(
            self.scale * zoom_factor,
            self.fit_scale,
            self.max_scale,
        ))
        if np.isclose(new_scale, self.scale):
            return "break"

        self.scale = new_scale
        target_left = image_x * self.scale - event.x
        target_top = image_y * self.scale - event.y
        self._刷新画布(view_left=target_left, view_top=target_top)
        return "break"

    def _更新状态文字(self) -> None:
        if self.match is None:
            text = f"Invalid mask: {self.error} | Brush {self.brush_radius}px"
            self.status_label.configure(text=text, foreground="#ff5050")
        else:
            text = (
                f"Center({self.match.global_px:.1f},{self.match.global_py:.1f}) | "
                f"Angle {self.match.theta:.1f} deg | Score {self.match.score:.1f} | "
                f"Brush {self.brush_radius}px | Zoom {self.scale / self.fit_scale:.2f}x | "
                f"Template {'ON' if self.show_template else 'OFF'}"
            )
            self.status_label.configure(text=text, foreground="#ffff00")

    def _刷新画布(self, view_left=None, view_top=None) -> None:
        if view_left is None:
            view_left = float(self.canvas.canvasx(0))
        if view_top is None:
            view_top = float(self.canvas.canvasy(0))

        display = 叠加mask(
            self.block.roi_bgr,
            self.history.current_mask,
            color=(0, 255, 0),
            alpha=MASK叠加透明度,
        )
        if self.show_template:
            display = _绘制模板轮廓(display, self.match)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        content_w = max(1, int(round(rgb.shape[1] * self.scale)))
        content_h = max(1, int(round(rgb.shape[0] * self.scale)))
        pil_image = Image.fromarray(rgb).resize(
            (content_w, content_h),
            resample=Image.NEAREST,
        )
        self.photo_image = ImageTk.PhotoImage(pil_image, master=self.root)
        if self.image_item is None:
            self.image_item = self.canvas.create_image(
                0,
                0,
                image=self.photo_image,
                anchor="nw",
            )
        else:
            self.canvas.itemconfigure(self.image_item, image=self.photo_image)
        self.canvas.configure(scrollregion=(0, 0, content_w, content_h))
        self.root.update_idletasks()
        self.canvas.xview_moveto(max(0.0, float(view_left)) / content_w)
        self.canvas.yview_moveto(max(0.0, float(view_top)) / content_h)
        self._更新状态文字()

    def _重新匹配并刷新(self) -> None:
        try:
            self.match = self.matcher.匹配(
                self.history.current_mask,
                self.block.category,
                self.block.crop_box,
            )
            self.error = ""
        except Exception as exc:
            self.match = None
            self.error = str(exc)
        self._刷新画布()

    def _撤销(self, _event=None):
        if self.history.撤销():
            self._重新匹配并刷新()
        return "break"

    def _重做(self, _event=None):
        if self.history.重做():
            self._重新匹配并刷新()
        return "break"

    def _恢复(self, _event=None):
        if self.history.恢复原始():
            self._重新匹配并刷新()
        return "break"

    def _调整画笔(self, direction: int):
        self.brush_radius = int(np.clip(
            self.brush_radius + direction * int(画笔调整步长),
            int(最小画笔半径),
            int(最大画笔半径),
        ))
        self._更新状态文字()
        return "break"

    def _减小画笔(self, _event=None):
        return self._调整画笔(-1)

    def _增大画笔(self, _event=None):
        return self._调整画笔(1)

    def _适配窗口(self, _event=None):
        self.scale = self.fit_scale
        self._刷新画布(view_left=0.0, view_top=0.0)
        return "break"

    def _切换模板轮廓(self, _event=None):
        """只切换局部编辑器中的模板轮廓显示，不改变任何识别数据。"""
        self.show_template = not self.show_template
        self._刷新画布()
        return "break"

    def 接受(self, _event=None):
        if self.match is None:
            print(f"\033[91m当前 Mask 不能接受：{self.error}\033[0m")
            self._更新状态文字()
            return "break"
        self.block.edited_mask = 转为二值mask(self.history.current_mask)
        self.block.current_match = self.match
        self.block.changed = not np.array_equal(
            self.block.edited_mask,
            self.block.original_mask,
        )
        self.accepted = True
        self.root.destroy()
        return "break"

    def 取消(self, _event=None):
        self.accepted = False
        self.root.destroy()
        return "break"

    def 运行(self) -> bool:
        self.root.mainloop()
        return self.accepted


def 编辑一个方块(
    block: 方块编辑项,
    matcher: 高位模板匹配器,
    _text_drawer: 界面绘字器,
) -> bool:
    """用自控 Tk 画布打开编辑器，彻底分离浏览与绘制事件。"""
    return TkMask编辑器(block, matcher).运行()


def _保存对比图(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"图片保存失败：{path}")


def 保存一个方块结果(block: 方块编辑项, output_dir: Path) -> None:
    """保存一个已经接受过的目标的原始/编辑 Mask 和前后叠加图。"""
    prefix = f"block_{block.index:02d}_{block.category}"
    保存二值mask(output_dir / f"{prefix}_original_mask.png", block.original_mask)
    保存二值mask(output_dir / f"{prefix}_edited_mask.png", block.edited_mask)

    before = 叠加mask(
        block.roi_bgr,
        block.original_mask,
        alpha=MASK叠加透明度,
    )
    before = _绘制模板轮廓(before, block.original_match)
    after = 叠加mask(
        block.roi_bgr,
        block.edited_mask,
        alpha=MASK叠加透明度,
    )
    after = _绘制模板轮廓(after, block.current_match)
    _保存对比图(output_dir / f"{prefix}_before.png", before)
    _保存对比图(output_dir / f"{prefix}_after.png", after)


def 保存全部结果(
    image_path: Path,
    image_bgr: np.ndarray,
    blocks: list[方块编辑项],
    output_dir: Path,
    text_drawer: 界面绘字器,
) -> None:
    """保存最终总览和包含全部检测项的 JSON。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    overview = _构建总览图(image_bgr, blocks, text_drawer)
    _保存对比图(output_dir / "overview_final.png", overview)

    result = {
        "输入图片": str(image_path),
        "方块数量": len(blocks),
        "方块": [],
    }
    for block in blocks:
        result["方块"].append(
            {
                "序号": block.index,
                "类别": block.category,
                "检测分数": block.detection_score,
                "检测框": list(block.detection_box),
                "编辑ROI": list(block.crop_box),
                "是否修改": block.changed,
                "修改前": block.original_match.转字典(),
                "修改后": block.current_match.转字典(),
            }
        )
    with (output_dir / "result.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)


def 运行交互总览(
    image_path: Path,
    image_bgr: np.ndarray,
    blocks: list[方块编辑项],
    matcher: 高位模板匹配器,
    output_dir: Path,
    text_drawer: 界面绘字器,
) -> bool:
    """显示总览；Q 提交整轮编辑，Esc 或关闭窗口取消整轮。"""
    state = {
        "selected": None,
        "display_shape": image_bgr.shape[:2],
        "show_masks": True,
        "needs_render": True,
    }
    committed = False

    def 总览鼠标回调(event, x, y, _flags, _param):
        # 普通左键保留给 OpenCV 的拖动查看；只有 Ctrl+左键进入编辑。
        if event != cv2.EVENT_LBUTTONDOWN or not (_flags & cv2.EVENT_FLAG_CTRLKEY):
            return
        point = 显示坐标转图像坐标(
            (x, y),
            image_bgr.shape,
            state["display_shape"],
        )
        state["selected"] = _查找点击目标(blocks, point)

    def 总览窗口仍可见():
        try:
            return cv2.getWindowProperty(总览窗口名称, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    cv2.namedWindow(
        总览窗口名称,
        cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL,
    )
    cv2.setMouseCallback(总览窗口名称, 总览鼠标回调)
    # 部分 OpenCV Qt 版本在第一次 imshow 之前会把可见性报告为 0，
    # 因此只在至少显示过一帧后才用该属性判断用户是否关闭窗口。
    已显示过一帧 = False
    try:
        while True:
            if 已显示过一帧 and not 总览窗口仍可见():
                break
            if state["needs_render"]:
                native_overview = _构建总览图(
                    image_bgr,
                    blocks,
                    text_drawer,
                    show_masks=state["show_masks"],
                )
                display_w, display_h = 计算适配显示尺寸(
                    native_overview.shape,
                    总览最大宽度,
                    总览最大高度,
                    max_scale=1.0,
                )
                display = cv2.resize(
                    native_overview,
                    (display_w, display_h),
                    interpolation=cv2.INTER_AREA,
                )
                state["display_shape"] = display.shape[:2]
                cv2.imshow(总览窗口名称, display)
                state["needs_render"] = False
                已显示过一帧 = True

            selected = state["selected"]
            if selected is not None:
                state["selected"] = None
                print(f"开始编辑序号 {selected.index}，类别 {selected.category}")
                # Tk 编辑器运行期间关闭 Qt 总览，避免两个 GUI 事件循环相互抢占。
                cv2.destroyWindow(总览窗口名称)
                cv2.waitKey(1)
                accepted = 编辑一个方块(selected, matcher, text_drawer)
                if accepted:
                    保存一个方块结果(selected, output_dir)
                    print(f"已接受并保存序号 {selected.index} 的 Mask")
                else:
                    print(f"已取消序号 {selected.index} 的本次编辑")
                # 编辑窗口关闭后，重新绑定总览窗口回调。
                cv2.namedWindow(
                    总览窗口名称,
                    cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL,
                )
                cv2.setMouseCallback(总览窗口名称, 总览鼠标回调)
                已显示过一帧 = False
                state["needs_render"] = True

            key = cv2.waitKeyEx(20)
            if 是总览提交键(key):
                committed = True
                break
            if key == 27:
                break
            if key in (ord("t"), ord("T")):
                state["show_masks"] = not state["show_masks"]
                state["needs_render"] = True
                status = "ON" if state["show_masks"] else "OFF"
                print(f"总览绿色 Mask 显示：{status}")
    finally:
        cv2.destroyAllWindows()

    if committed:
        保存全部结果(
            image_path,
            image_bgr,
            blocks,
            output_dir,
            text_drawer,
        )
    return committed


def main() -> None:
    # 只有独立照片 Demo 才加载 YOLO；比赛会话子进程不会加载任何检测模型。
    from ultralytics import YOLO

    image_path = 查找唯一输入图片(输入目录)
    if not 检测模型路径.is_file():
        raise FileNotFoundError(f"检测模型不存在：{检测模型路径}")
    if not 分割模型路径.is_file():
        raise FileNotFoundError(f"分割模型不存在：{分割模型路径}")

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None or image_bgr.size == 0:
        raise RuntimeError(f"图片读取失败：{image_path}")

    print(f"读取图片：{image_path}")
    print(f"加载检测模型：{检测模型路径}")
    detector = YOLO(str(检测模型路径))
    print(f"加载分割模型：{分割模型路径}")
    seg_model = YOLO(str(分割模型路径), task="segment")
    matcher = 高位模板匹配器()
    text_drawer = 界面绘字器()

    print("开始检测、分割和初始模板匹配……")
    blocks = 检测并分割方块(image_bgr, detector, seg_model, matcher)
    print(f"得到 {len(blocks)} 个可编辑方块；请在总览窗口中点击目标。")

    output_dir = 输出根目录 / image_path.stem
    committed = 运行交互总览(
        image_path,
        image_bgr,
        blocks,
        matcher,
        output_dir,
        text_drawer,
    )
    if committed:
        print(f"最终结果已保存：{output_dir}")
    else:
        print("已取消本轮编辑，未提交最终结果。")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\033[91m运行失败：{exc}\033[0m")
        raise SystemExit(1)
