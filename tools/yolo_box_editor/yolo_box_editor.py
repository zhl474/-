#!/usr/bin/env python3
"""YOLO 检测框人工修正编辑器：删框 / 画框 / 改类别。

两种运行方式：
1. 会话模式：ROS 父进程通过 SINGLE_ARM_TETRIS_YOLO_EDIT_SESSION 环境变量
   指向 session.json，本脚本不加载 YOLO 或 ROS，只做框级编辑并原子提交。
2. 独立演示模式：直接 python 运行，按代码开头的参数加载模型和图像，
   用于脱离 ROS 单独调试交互。

纯 OpenCV 单窗口，交互与高位 Mask 编辑器（Ctrl 画笔）和常规标注软件一致：
  Ctrl+拖动 画新框（当前类别，任何位置，包括已有框内部）
  点击框   选中并显示 8 个句柄
  框内拖动 平移该框；拖动句柄调大调小
  1-7      选类别；有选中框时改为把该框改成这个类别
  D/Delete 删除选中框
  Enter/Q  提交；Esc 或关窗取消本轮修正
  H        开关按键帮助
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
SRC_DIR = DEMO_DIR.parent.parent
IMAGE_PROCESS_DIR = SRC_DIR / "image_process"
for _path in (DEMO_DIR, IMAGE_PROCESS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from image_process_lib.yolo_edit_session import (  # noqa: E402
    YOLO会话环境变量,
    YOLO编辑会话错误,
    提交YOLO编辑结果,
    读取YOLO编辑会话,
    编辑取消退出码,
    最小框边长,
)

# ===== 独立演示模式参数（直接 python 运行时使用，改这里传参）=====
演示图像路径 = "/home/zhl/桌面/高位测试图.jpg"
检测模型路径 = str(SRC_DIR / "competition" / "model" / "best5.14.pt")
置信度阈值 = 0.45

窗口名 = "YOLO Box Correction - Enter:OK Esc:cancel"
显示最大宽 = 1600
显示最大高 = 900
状态栏高 = 34
帮助行高 = 24
句柄半径 = 4
句柄命中半径 = 9
类别颜色表 = {
    "L_blue": (255, 64, 0),
    "L_yellow": (0, 210, 230),
    "z_blue": (170, 120, 60),
    "z_green": (0, 180, 0),
    "square": (0, 140, 255),
    "T": (255, 0, 255),
    "line": (180, 180, 180),
}
帮助文本 = [
    "Ctrl+Drag: draw new box (current class)",
    "Click box: select | Drag inside box: move",
    "Drag white handles: resize selected box",
    "1-7: set class / change selected box class",
    "D or Delete: delete selected box",
    "Enter / Q: commit    Esc / close window: cancel",
    "H: toggle this help",
]


class YOLO框编辑器:
    """纯 OpenCV 的检测框级编辑器，返回修正后的检测列表或 None（取消）。"""

    def __init__(self, image_bgr, classes, detections):
        self.image_bgr = image_bgr
        self.classes = [str(name) for name in classes]
        self.detections = []
        for record in detections:
            self.detections.append({
                "category": str(record["category"]),
                "score": float(record.get("score", 1.0)),
                "box": tuple(float(value) for value in record["box"]),
                "source": str(record.get("source", "yolo")),
            })
        self.image_h, self.image_w = image_bgr.shape[:2]
        self.scale = min(
            显示最大宽 / self.image_w,
            (显示最大高 - 状态栏高) / self.image_h,
        )
        if self.scale <= 0 or not np.isfinite(self.scale):
            self.scale = 1.0
        self.selected = None
        self.current_class = self.classes[0] if self.classes else ""
        self.show_help = True
        # 拖动状态：None / ("draw", 起点) / ("move", 框序号, 原框, 起点鼠标位)
        self.drag = None
        self._显示图 = None
        self._鼠标显示坐标 = None

    # ---- 坐标与命中 ----

    def _图像坐标(self, x, y):
        return int(x / self.scale), int(y / self.scale)

    def _命中框(self, px, py):
        """返回包含该图像坐标的最小框序号（同现有总览取最小框的规则）。"""
        best_index = None
        best_area = None
        for index, detection in enumerate(self.detections):
            x1, y1, x2, y2 = detection["box"]
            if x1 <= px <= x2 and y1 <= py <= y2:
                area = (x2 - x1) * (y2 - y1)
                if best_area is None or area < best_area:
                    best_index = index
                    best_area = area
        return best_index

    # ---- 鼠标回调 ----

    def _鼠标回调(self, event, x, y, flags, param):
        px, py = self._图像坐标(x, y)
        self._鼠标显示坐标 = (x, y)
        ctrl_pressed = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
        if event == cv2.EVENT_LBUTTONDOWN:
            if ctrl_pressed:
                # Ctrl+拖动永远画新框（即使起点落在已有框内），
                # 与高位 Mask 编辑器的 Ctrl 画笔习惯一致。
                self.drag = ("draw", (px, py))
                return
            handle = self._命中句柄(x, y)
            if handle is not None:
                index, name = handle
                self.drag = ("resize", index, self.detections[index]["box"], name)
                return
            hit = self._命中框(px, py)
            self.selected = hit
            if hit is None:
                self.drag = None
            else:
                self.drag = ("move", hit, self.detections[hit]["box"], (px, py))
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None:
            kind = self.drag[0]
            if kind == "draw":
                self._预览画框 = (self.drag[1], (px, py))
            elif kind == "move":
                _kind, index, original_box, start = self.drag
                ox1, oy1, ox2, oy2 = original_box
                self.detections[index]["box"] = self._限制框在图内(
                    ox1 + px - start[0],
                    oy1 + py - start[1],
                    ox2 + px - start[0],
                    oy2 + py - start[1],
                )
            else:  # resize
                _kind, index, original_box, name = self.drag
                self.detections[index]["box"] = self._应用句柄调整(
                    original_box, name, px, py
                )
        elif event == cv2.EVENT_LBUTTONUP and self.drag is not None:
            kind = self.drag[0]
            if kind == "draw":
                x1, y1 = self.drag[1]
                box = self._规范框(x1, y1, px, py)
                if self._框可用(box):
                    self.detections.append({
                        "category": self.current_class,
                        "score": 1.0,
                        "box": box,
                        "source": "manual",
                    })
                    self.selected = len(self.detections) - 1
                self._预览画框 = None
            else:
                # 移动/缩放后被图像边界截断到过小，回退到原框。
                index, original_box = self.drag[1], self.drag[2]
                if not self._框可用(self.detections[index]["box"]):
                    self.detections[index]["box"] = original_box
            self.drag = None

    def _句柄位置(self, box):
        """选中框的 8 个调整句柄：四角 + 四边中点（图像坐标）。"""
        x1, y1, x2, y2 = box
        xm = (x1 + x2) / 2.0
        ym = (y1 + y2) / 2.0
        return {
            "左上": (x1, y1), "右上": (x2, y1), "左下": (x1, y2), "右下": (x2, y2),
            "上": (xm, y1), "下": (xm, y2), "左": (x1, ym), "右": (x2, ym),
        }

    def _命中句柄(self, x, y):
        """在显示坐标里命中选中框的句柄；返回 (框序号, 句柄名) 或 None。"""
        if self.selected is None or self.selected >= len(self.detections):
            return None
        box = self.detections[self.selected]["box"]
        for name, (hx, hy) in self._句柄位置(box).items():
            dx = hx * self.scale - x
            dy = hy * self.scale - y
            if dx * dx + dy * dy <= 句柄命中半径 ** 2:
                return (self.selected, name)
        return None

    def _应用句柄调整(self, box, name, px, py):
        """拖动句柄时只改动该句柄控制的边，支持拖过对边自动翻转。"""
        x1, y1, x2, y2 = box
        if "左" in name:
            x1 = px
        if "右" in name:
            x2 = px
        if "上" in name:
            y1 = py
        if "下" in name:
            y2 = py
        return self._限制框在图内(*self._规范框(x1, y1, x2, y2))

    def _限制框在图内(self, x1, y1, x2, y2):
        x1 = min(max(x1, 0.0), float(self.image_w))
        y1 = min(max(y1, 0.0), float(self.image_h))
        x2 = min(max(x2, 0.0), float(self.image_w))
        y2 = min(max(y2, 0.0), float(self.image_h))
        return (x1, y1, x2, y2)

    def _规范框(self, x1, y1, x2, y2):
        return (
            float(min(x1, x2)),
            float(min(y1, y2)),
            float(max(x1, x2)),
            float(max(y1, y2)),
        )

    def _框可用(self, box):
        x1, y1, x2, y2 = box
        return (x2 - x1) >= 最小框边长 and (y2 - y1) >= 最小框边长

    # ---- 绘制 ----

    def _渲染(self):
        display = cv2.resize(
            self.image_bgr,
            (max(1, int(self.image_w * self.scale)), max(1, int(self.image_h * self.scale))),
            interpolation=cv2.INTER_AREA if self.scale < 1.0 else cv2.INTER_LINEAR,
        )
        canvas = np.zeros((display.shape[0] + 状态栏高, display.shape[1], 3), dtype=np.uint8)
        canvas[: display.shape[0]] = display
        self._显示图 = canvas

        for index, detection in enumerate(self.detections):
            self._画框(canvas, detection, selected=(index == self.selected))
        if self.selected is not None and self.selected < len(self.detections):
            self._画句柄(canvas, self.detections[self.selected]["box"])
        preview = getattr(self, "_预览画框", None)
        if preview is not None and self.drag is not None and self.drag[0] == "draw":
            (sx1, sy1), (sx2, sy2) = preview
            p1 = self._显示坐标(sx1, sy1)
            p2 = self._显示坐标(sx2, sy2)
            color = 类别颜色表.get(self.current_class, (255, 255, 255))
            cv2.rectangle(canvas, p1, p2, color, 2)

        self._画状态栏(canvas)
        if self.show_help:
            self._画帮助(canvas)
        self._画类别图例(canvas)
        return canvas

    def _显示坐标(self, px, py):
        return int(px * self.scale), int(py * self.scale)

    def _画框(self, canvas, detection, selected):
        x1, y1, x2, y2 = detection["box"]
        p1 = self._显示坐标(x1, y1)
        p2 = self._显示坐标(x2, y2)
        color = 类别颜色表.get(detection["category"], (255, 255, 255))
        thickness = 3 if selected else 2
        if selected:
            cv2.rectangle(canvas, p1, p2, (255, 255, 255), thickness + 2)
        cv2.rectangle(canvas, p1, p2, color, thickness)
        label = f"{self._类别序号(detection['category'])}:{detection['category']}"
        if detection.get("source", "yolo") == "yolo":
            label += f" {detection['score']:.2f}"
        else:
            label += " *"
        tx, ty = p1[0], max(14, p1[1] - 5)
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def _画句柄(self, canvas, box):
        """给选中框画 8 个白色句柄，提示可以直接拖动调大调小。"""
        for hx, hy in self._句柄位置(box).values():
            cx, cy = self._显示坐标(hx, hy)
            cv2.circle(canvas, (cx, cy), 句柄半径 + 1, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 句柄半径, (255, 255, 255), -1, cv2.LINE_AA)

    def _画状态栏(self, canvas):
        y = canvas.shape[0] - 状态栏高 // 2 + 6
        text = (
            f"Class: {self._类别序号(self.current_class)}:{self.current_class}"
            f" | Boxes: {len(self.detections)} | "
            "Ctrl+Drag:draw Drag:move Handles:resize D:del Enter:OK Esc:cancel H:help"
        )
        cv2.rectangle(canvas, (0, canvas.shape[0] - 状态栏高), (canvas.shape[1], canvas.shape[0]), (30, 30, 30), -1)
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

    def _类别序号(self, category):
        """类别对应的数字键序号，画框时直接看着图例按。"""
        try:
            return self.classes.index(category) + 1
        except ValueError:
            return 0

    def _画类别图例(self, canvas):
        """右上角常驻类别图例：序号 + 色块 + 类名。"""
        line_h = 22
        pad = 6
        width = 168
        x0 = canvas.shape[1] - width - pad
        y0 = pad
        height = line_h * len(self.classes) + 2 * pad
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
        for index, name in enumerate(self.classes):
            color = 类别颜色表.get(name, (255, 255, 255))
            ty = y0 + pad + 14 + index * line_h
            cv2.rectangle(canvas, (x0 + 6, ty - 11), (x0 + 20, ty + 3), color, -1)
            text = f"{index + 1} {name}"
            cv2.putText(canvas, text, (x0 + 26, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, (x0 + 26, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    def _画帮助(self, canvas):
        for line_index, line in enumerate(帮助文本):
            ty = 22 + line_index * 帮助行高
            cv2.putText(canvas, line, (12, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, line, (12, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)

    # ---- 主循环 ----

    def 运行(self):
        """返回修正后的检测列表；取消时返回 None。"""
        cv2.namedWindow(窗口名, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(窗口名, self._鼠标回调)
        committed = False
        try:
            while True:
                canvas = self._渲染()
                cv2.imshow(窗口名, canvas)
                key = cv2.waitKeyEx(20)
                if key == -1:
                    if cv2.getWindowProperty(窗口名, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    continue
                key_char = chr(key & 0xFF) if 0 <= (key & 0xFF) < 128 else ""
                if key == 27:  # Esc
                    break
                if key in (13, 10) or key_char in ("q", "Q"):
                    committed = True
                    break
                if key_char in ("h", "H"):
                    self.show_help = not self.show_help
                    continue
                if key_char in ("d", "D") or key == 65535:  # Delete
                    if self.selected is not None:
                        self.detections.pop(self.selected)
                        self.selected = None
                    continue
                digit = self._数字键类别(key_char)
                if digit is not None:
                    if self.selected is not None:
                        self.detections[self.selected]["category"] = digit
                    else:
                        self.current_class = digit
        finally:
            cv2.destroyWindow(窗口名)
        if not committed:
            return None
        return [
            {
                "category": detection["category"],
                "score": detection["score"],
                "box": tuple(float(value) for value in detection["box"]),
                "source": detection.get("source", "manual"),
            }
            for detection in self.detections
        ]

    def _数字键类别(self, key_char):
        if not key_char.isdigit():
            return None
        index = int(key_char) - 1
        if 0 <= index < len(self.classes):
            return self.classes[index]
        return None


def 会话模式(manifest_path: Path) -> int:
    print("YOLO 检测框人工修正子进程已启动，正在读取会话……", flush=True)
    manifest, image_bgr = 读取YOLO编辑会话(manifest_path)
    editor = YOLO框编辑器(image_bgr, manifest["classes"], manifest["detections"])
    print("按 Enter/Q 提交，Esc 取消本轮识别", flush=True)
    detections = editor.运行()
    if detections is None:
        print("已取消本轮 YOLO 检测框修正。")
        return 编辑取消退出码
    提交YOLO编辑结果(manifest_path, detections)
    print("YOLO 检测框修正结果已提交给 ROS 父进程。")
    return 0


def 独立演示模式() -> int:
    print(f"独立演示模式：加载图像 {演示图像路径}")
    image_bgr = cv2.imread(演示图像路径, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"演示图像读取失败：{演示图像路径}")
    print(f"加载检测模型 {检测模型路径}（首次运行需要几秒）……")
    from ultralytics import YOLO  # 懒加载，会话模式不需要。

    model = YOLO(检测模型路径)
    result = model(image_bgr, iou=0.5, conf=置信度阈值)
    detections = []
    for det in result[0].boxes.data.tolist():
        x1, y1, x2, y2, score, cid = det
        category = str(model.names[int(cid)])
        if category == "board":
            continue
        detections.append({
            "category": category,
            "score": float(score),
            "box": (float(x1), float(y1), float(x2), float(y2)),
            "source": "yolo",
        })
    print(f"YOLO 检测到 {len(detections)} 个方块框，正在打开编辑窗口……")
    from image_process_lib.block_category import (
        BLOCK_CATEGORY_NAMES,
        normalize_category_name,
    )

    for detection in detections:
        detection["category"] = normalize_category_name(detection["category"])
    editor = YOLO框编辑器(image_bgr, list(BLOCK_CATEGORY_NAMES), detections)
    detections = editor.运行()
    if detections is None:
        print("已取消。")
        return 编辑取消退出码
    print("修正结果：")
    for detection in detections:
        box = detection["box"]
        print(
            f"  {detection['category']:<10} score={detection['score']:.2f} "
            f"box=({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}) "
            f"source={detection['source']}"
        )
    return 0


def main() -> int:
    session_value = os.environ.get(YOLO会话环境变量, "").strip()
    if session_value:
        return 会话模式(Path(session_value).resolve())
    return 独立演示模式()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except YOLO编辑会话错误 as exc:
        print(f"\033[91mYOLO 修正会话错误：{exc}\033[0m")
        raise SystemExit(1)
    except Exception as exc:
        print(f"\033[91mYOLO 检测框修正子进程失败：{exc}\033[0m")
        raise SystemExit(1)
