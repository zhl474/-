#!/usr/bin/env python3
"""V2 角度锁定人工编辑器：点选方块锁角度，实时预览重匹配结果。

只有会话模式：ROS 父进程通过 SINGLE_ARM_TETRIS_ANGLE_LOCK_SESSION
环境变量指向 session.json，本脚本不加载 YOLO 或 ROS。父进程把本轮
V2 识别结果和完整 v2 配置写进会话，预览用的受限匹配与父进程提交后
的重匹配走完全相同的代码路径（EdgeTemplateMatcher.match_block）。

纯 OpenCV 单窗口，交互与 YOLO 框修正器一致：
  点击方块   选中（最近的中心点）
  A / D      角度 -/+ 一个角度步进
  Z / C      角度 -/+ 10 度
  T          在终端输入精确角度（正式口径 [-180,180)）
  X          清除该方块的锁定，回到本轮原角度
  Enter      提交锁定（父进程按锁定角度重匹配位置）
  Esc/关窗   放弃修改，按本轮原识别继续

锁定后的方块轮廓画成橙色，未锁定的保持绿色。
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

from image_process_lib.angle_lock_edit_session import (  # noqa: E402
    编辑取消退出码,
    角度锁定会话环境变量,
    角度锁定编辑会话错误,
    提交角度锁定结果,
    读取角度锁定编辑会话,
)
from image_process_lib.edge_template_detector import (  # noqa: E402
    EdgeTemplateConfig,
    EdgeTemplateMatcher,
    build_edge_distance_field,
)
from image_process_lib.template_match.kernels_create import (  # noqa: E402
    ROTATION_TOTAL_ANGLE,
)

窗口名 = "V2 Angle Lock - Enter:OK Esc:skip"
显示最大宽 = 1600
显示最大高 = 900
状态栏高 = 34
选中命中半径 = 60.0
角度粗调步进 = 10.0


def 规范化角度(theta):
    """折到 [-180, 180)。"""
    value = (float(theta) + 180.0) % 360.0 - 180.0
    return 0.0 if abs(value) < 1e-9 else value


class V2角度锁定编辑器:
    """纯 OpenCV 的角度锁定编辑器，返回 {方块序号: theta} 或 None（放弃）。"""

    def __init__(self, image_bgr, manifest):
        self.image_bgr = image_bgr
        self.blocks = manifest["blocks"]
        self.config = EdgeTemplateConfig.from_mapping(manifest["v2_config"])
        # 预览匹配与父进程重匹配同一代码路径；CPU 即可，不占 GPU。
        self.matcher = EdgeTemplateMatcher(
            manifest["template_geometry"],
            self.config,
            device="cpu",
        )
        self.distance_field = build_edge_distance_field(image_bgr, self.config)
        self.image_h, self.image_w = image_bgr.shape[:2]
        self.scale = min(
            显示最大宽 / self.image_w,
            (显示最大高 - 状态栏高) / self.image_h,
        )
        if self.scale <= 0 or not np.isfinite(self.scale):
            self.scale = 1.0
        self.selected = None
        self.locks = {}
        self.previews = {}

    # ---- 角度与预览 ----

    def _当前角度(self, index):
        if index in self.locks:
            return self.locks[index]
        return float(self.blocks[index - 1]["theta"])

    def _调整角度(self, delta):
        if self.selected is None:
            return
        index = self.selected
        self.locks[index] = 规范化角度(self._当前角度(index) + delta)
        self._刷新预览(index)

    def _输入角度(self):
        if self.selected is None:
            return
        index = self.selected
        try:
            raw = input(f"输入第 {index} 个方块的目标角度（当前 {self._当前角度(index):.1f}）：")
            theta = 规范化角度(float(raw.strip()))
        except ValueError:
            print("没有输入有效角度，保持不变。")
            return
        self.locks[index] = theta
        self._刷新预览(index)

    def _刷新预览(self, index):
        """按当前锁定角度做一次受限匹配，实时看到位置重算结果。"""
        block = self.blocks[index - 1]
        category = block["category"]
        period = float(ROTATION_TOTAL_ANGLE[category])
        template_center = ((-self.locks[index]) % 360.0) % period
        detection = {
            "category": category,
            "score": float(block.get("score", 1.0)),
            "box": tuple(float(value) for value in block["box"]),
        }
        try:
            self.previews[index] = self.matcher.match_block(
                self.distance_field,
                detection,
                angle_center=template_center,
                angle_window=self.config.angle_step_deg / 2.0,
            )
        except Exception as exc:
            print(f"预览匹配失败（提交时同样会报错）：{exc}")
            self.previews.pop(index, None)

    def _清除锁定(self):
        if self.selected is None:
            return
        self.locks.pop(self.selected, None)
        self.previews.pop(self.selected, None)

    # ---- 鼠标与坐标 ----

    def _图像坐标(self, x, y):
        return int(x / self.scale), int(y / self.scale)

    def _命中方块(self, px, py):
        best_index = None
        best_distance = None
        for block in self.blocks:
            dx = float(block["center_px"]) - px
            dy = float(block["center_py"]) - py
            distance = dx * dx + dy * dy
            if best_distance is None or distance < best_distance:
                best_index = int(block["index"])
                best_distance = distance
        if best_distance is not None and best_distance <= 选中命中半径 ** 2:
            return best_index
        return None

    def _鼠标回调(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            px, py = self._图像坐标(x, y)
            self.selected = self._命中方块(px, py)

    # ---- 绘制 ----

    def _显示坐标(self, px, py):
        return int(px * self.scale), int(py * self.scale)

    def _最近网格角(self, category, theta):
        period = float(ROTATION_TOTAL_ANGLE[category])
        center = ((-float(theta)) % 360.0) % period
        metadata = self.matcher._metadata_for(category)
        return min(metadata, key=lambda a: min(abs(a - center) % period, period - abs(a - center) % period))

    def _渲染(self):
        display = cv2.resize(
            self.image_bgr,
            (max(1, int(self.image_w * self.scale)), max(1, int(self.image_h * self.scale))),
            interpolation=cv2.INTER_AREA if self.scale < 1.0 else cv2.INTER_LINEAR,
        )
        canvas = np.zeros((display.shape[0] + 状态栏高, display.shape[1], 3), dtype=np.uint8)
        canvas[: display.shape[0]] = display

        for block in self.blocks:
            index = int(block["index"])
            preview = self.previews.get(index)
            if preview is not None:
                center = (preview["center_px"], preview["center_py"])
                pick = (preview["px"], preview["py"])
                template_angle = preview["angle"]
                theta_text = preview["theta"]
            else:
                center = (float(block["center_px"]), float(block["center_py"]))
                pick = (float(block["px"]), float(block["py"]))
                template_angle = self._最近网格角(block["category"], block["theta"])
                theta_text = float(block["theta"])
            locked = index in self.previews
            self._画方块(
                canvas,
                block["category"],
                template_angle,
                center,
                pick,
                locked=locked,
                selected=(index == self.selected),
                index=index,
                theta_text=theta_text,
            )

        self._画状态栏(canvas)
        return canvas

    def _画方块(self, canvas, category, template_angle, center, pick, locked, selected, index, theta_text):
        item = self.matcher._metadata_for(category)[template_angle]
        contours, _ = cv2.findContours(item["binary"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        origin = self._显示坐标(center[0] - item["anchor"][0], center[1] - item["anchor"][1])
        contour_color = (0, 200, 255) if locked else (0, 255, 0)
        thickness = 2 if selected else 1
        cv2.drawContours(canvas, contours, -1, contour_color, thickness, offset=origin)

        center_disp = self._显示坐标(*center)
        pick_disp = self._显示坐标(*pick)
        cv2.drawMarker(canvas, center_disp, (255, 0, 0), cv2.MARKER_CROSS, 10, 1)
        cv2.circle(canvas, pick_disp, 3, (0, 0, 255), 2)
        if selected:
            cv2.circle(canvas, center_disp, 14, (255, 255, 255), 2)

        badge = " *" if locked else ""
        label = f"{index}:{category} {theta_text:.1f}deg{badge}"
        label_x = max(2, min(center_disp[0] - 40, canvas.shape[1] - 170))
        label_y = max(14, center_disp[1] - 18)
        cv2.putText(canvas, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        color = (0, 200, 255) if locked else (255, 255, 255)
        cv2.putText(canvas, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def _画状态栏(self, canvas):
        y = canvas.shape[0] - 状态栏高 // 2 + 6
        if self.selected is not None:
            info = (
                f"Sel {self.selected}:{self.blocks[self.selected - 1]['category']}"
                f" theta={self._当前角度(self.selected):.1f}"
                f" locked={int(len(self.locks))}"
            )
        else:
            info = f"No sel | locked={int(len(self.locks))}"
        text = (
            f"{info} | Click:sel A/D:+-step Z/C:+-10 T:type X:clear"
            " Enter:OK Esc:skip"
        )
        cv2.rectangle(canvas, (0, canvas.shape[0] - 状态栏高), (canvas.shape[1], canvas.shape[0]), (30, 30, 30), -1)
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

    # ---- 主循环 ----

    def 运行(self):
        """返回 {方块序号(1 起): theta}；放弃时返回 None。"""
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
                if key == 27:  # Esc：放弃修改，按原识别继续。
                    break
                if key in (13, 10):
                    committed = True
                    break
                if key_char in ("a", "A"):
                    self._调整角度(-self.config.angle_step_deg)
                elif key_char in ("d", "D"):
                    self._调整角度(self.config.angle_step_deg)
                elif key_char in ("z", "Z"):
                    self._调整角度(-角度粗调步进)
                elif key_char in ("c", "C"):
                    self._调整角度(角度粗调步进)
                elif key_char in ("t", "T"):
                    self._输入角度()
                elif key_char in ("x", "X"):
                    self._清除锁定()
        finally:
            cv2.destroyWindow(窗口名)
        if not committed:
            return None
        return dict(self.locks)


def 会话模式(manifest_path: Path) -> int:
    print("V2 角度锁定编辑子进程已启动，正在读取会话……", flush=True)
    manifest, image_bgr = 读取角度锁定编辑会话(manifest_path)
    editor = V2角度锁定编辑器(image_bgr, manifest)
    print(
        f"共 {len(editor.blocks)} 个方块；点击选中，A/D 微调，Z/C 粗调，"
        "T 输入精确角度；Enter 提交，Esc 放弃修改继续",
        flush=True,
    )
    locks = editor.运行()
    if locks is None:
        print("已放弃修改，本轮按原识别结果继续。")
        return 编辑取消退出码
    提交角度锁定结果(manifest_path, locks)
    print(f"已提交 {len(locks)} 个角度锁定，父进程将按锁定角度重匹配。")
    return 0


def main() -> int:
    session_value = os.environ.get(角度锁定会话环境变量, "").strip()
    if not session_value:
        print(
            "缺少会话环境变量，本脚本只支持由识别节点拉起的会话模式；"
            f"请设置 {角度锁定会话环境变量} 指向 session.json。"
        )
        return 1
    return 会话模式(Path(session_value).resolve())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except 角度锁定编辑会话错误 as exc:
        print(f"\033[91mV2 角度锁定会话错误：{exc}\033[0m")
        raise SystemExit(1)
    except Exception as exc:
        print(f"\033[91mV2 角度锁定编辑子进程失败：{exc}\033[0m")
        raise SystemExit(1)
