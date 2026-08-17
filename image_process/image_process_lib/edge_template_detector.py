"""识别 V2：Canny 距离场 + 全角度居中线模板核的高位方块精定位。

与 V1（YOLO-Seg 上表面分割 + 模板匹配）完全平行，不含分割、不筛角度、
无先验。移植自 tools/vision/edge_template_test/run_pure_canny.py 的
批量验证逻辑，算法保持一致：

  全图 Canny 距离场(cap) + 每类别全角度居中 1px 轮廓线核
  -> 一次 F.conv2d 归一化 -> 两层 argmax 选角度和平移。

输出 px/py/theta 与 V1 同口径，可直接复用部署的像素转 TCP 标定：
  - 非 L 类 px/py = 模板旋转中心（与 V1 match_block_mask 一致）；
  - L_yellow/L_blue px/py = 方案A抓点，用理想模板复刻
    coreect_LL_location 的实测规则（_template_l_pick_in_canvas）；
  - theta = (-模板角 + 180) % 360 - 180，规范到 [-180, 180)，
    与正式流程 get_rect 输出等价（低位伺服角度先验口径一致）。

无 ROS 依赖；调用方需保证串行访问（prepare_task 已有锁）。
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from image_process_lib.block_category import normalize_category_name
from image_process_lib.template_match.kernels_create import (
    ROTATION_TOTAL_ANGLE,
    _template_l_pick_in_canvas,
    build_angle_foreground_metadata,
    resolve_category_runs,
    template_rect_size_from_runs,
)


_L_CATEGORIES = ("L_yellow", "L_blue")


@dataclass(frozen=True)
class EdgeTemplateConfig:
    """V2 边缘模板匹配参数，默认值与 run_pure_canny.py 验证时一致。"""

    angle_step_deg: float = 2.0
    canny_low: float = 50.0
    canny_high: float = 150.0
    gaussian_ksize: int = 3
    distance_cap_px: float = 20.0
    crop_margin_px: int = 8
    kernel_margin_px: int = 2
    search_margin_px: int = 20

    @classmethod
    def from_mapping(cls, mapping):
        """从 perception.yaml 的 block_recognition.v2 字典构建并校验。"""
        if mapping is None:
            mapping = {}
        if not isinstance(mapping, dict):
            raise ValueError("block_recognition.v2 必须是字典")

        def float_param(name, default, minimum=None, exclusive=False):
            value = mapping.get(name, default)
            if isinstance(value, bool):
                raise ValueError(f"block_recognition.v2.{name} 必须是数值")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"block_recognition.v2.{name} 必须是数值") from exc
            if not np.isfinite(number):
                raise ValueError(f"block_recognition.v2.{name} 必须是有限数值")
            if exclusive and number <= minimum:
                raise ValueError(f"block_recognition.v2.{name} 必须大于 {minimum}")
            if not exclusive and minimum is not None and number < minimum:
                raise ValueError(f"block_recognition.v2.{name} 不能小于 {minimum}")
            return number

        def int_param(name, default, minimum=0):
            value = mapping.get(name, default)
            if isinstance(value, bool):
                raise ValueError(f"block_recognition.v2.{name} 必须是整数")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"block_recognition.v2.{name} 必须是整数") from exc
            if not np.isfinite(number) or not number.is_integer() or number < minimum:
                raise ValueError(f"block_recognition.v2.{name} 必须是不小于 {minimum} 的整数")
            return int(number)

        config = cls(
            angle_step_deg=float_param("angle_step_deg", cls.angle_step_deg, 0.0, exclusive=True),
            canny_low=float_param("canny_low", cls.canny_low, 0.0),
            canny_high=float_param("canny_high", cls.canny_high, 0.0, exclusive=True),
            gaussian_ksize=int_param("gaussian_ksize", cls.gaussian_ksize, minimum=1),
            distance_cap_px=float_param(
                "distance_cap_px", cls.distance_cap_px, 0.0, exclusive=True
            ),
            crop_margin_px=int_param("crop_margin_px", cls.crop_margin_px),
            kernel_margin_px=int_param("kernel_margin_px", cls.kernel_margin_px),
            search_margin_px=int_param("search_margin_px", cls.search_margin_px),
        )
        if config.canny_high <= config.canny_low:
            raise ValueError("block_recognition.v2.canny_high 必须大于 canny_low")
        if config.gaussian_ksize % 2 == 0:
            raise ValueError("block_recognition.v2.gaussian_ksize 必须是奇数")
        return config


def build_edge_distance_field(img_bgr, config):
    """构建边缘距离价值场：边缘像素最高，随距离线性衰减到 0。"""
    gray = cv2.GaussianBlur(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
        (config.gaussian_ksize, config.gaussian_ksize),
        0,
    )
    edges = cv2.Canny(gray, config.canny_low, config.canny_high)
    distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    return (
        config.distance_cap_px - np.minimum(distance, config.distance_cap_px)
    ).astype(np.float32)


class EdgeTemplateMatcher:
    """Canny 边缘模板匹配器；核按类别懒构建并缓存，线程不安全。"""

    def __init__(self, template_geometry, config=None, device=None):
        if not isinstance(template_geometry, dict):
            raise ValueError("template_geometry 必须是字典")
        self.block_px = int(template_geometry["block_px"])
        self.connector_px = int(template_geometry["connector_px"])
        if self.block_px <= 0 or self.connector_px <= 0:
            raise ValueError("block_px 和 connector_px 必须为正数")
        # 保留整个几何配置，按类别解析 overrides 线段；缺 runs 键时回落理想展开。
        self._template_geometry = template_geometry
        self.config = config if config is not None else EdgeTemplateConfig()
        self.device = (
            str(device)
            if device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._angle_metadata = {}
        self._kernel_cache = {}

    def _metadata_for(self, category):
        if category not in self._angle_metadata:
            self._angle_metadata[category] = build_angle_foreground_metadata(
                category,
                self.block_px,
                self.connector_px,
                self.config.angle_step_deg,
                template_runs=resolve_category_runs(category, self._template_geometry),
            )
        return self._angle_metadata[category]

    def _angles_for(self, category):
        """该类别全角度网格（模板角，已归一到 [0, 周期)）。"""
        return sorted(self._metadata_for(category))

    def _select_angles_in_window(self, category, angle_center, angle_window):
        """从全角度网格选出距 angle_center 折叠距离不超过窗口的子集。

        窗口太窄没盖住任何网格角时退回最近的一个网格角（锁定语义）。
        """
        period = float(ROTATION_TOTAL_ANGLE[category])
        center = float(angle_center) % period
        window = float(angle_window)
        if not np.isfinite(window) or window < 0.0:
            raise ValueError("angle_window 必须是不小于 0 的有限数值")
        all_angles = self._angles_for(category)

        def folded_distance(angle):
            folded = abs(angle - center) % period
            return min(folded, period - folded)

        selected = [
            angle for angle in all_angles if folded_distance(angle) <= window + 1e-9
        ]
        if not selected:
            selected = [min(all_angles, key=folded_distance)]
        return selected

    def _kernels_for(self, category):
        return self._kernels_for_angles(category, self._angles_for(category))

    def _kernels_for_angles(self, category, angles):
        """构建给定角度列表的居中线核；锚点记录旋转中心和抓点两套。"""
        key = (str(category), tuple(float(angle) for angle in angles))
        cached = self._kernel_cache.get(key)
        if cached is not None:
            return cached
        metadata = self._metadata_for(category)
        angles = [float(angle) for angle in angles]
        binary_list = [metadata[angle]["binary"] for angle in angles]
        margin = self.config.kernel_margin_px
        max_h = max(binary.shape[0] for binary in binary_list) + 2 * margin
        max_w = max(binary.shape[1] for binary in binary_list) + 2 * margin
        kernels = np.zeros((len(angles), 1, max_h, max_w), dtype=np.float32)
        anchors = []
        pick_anchors = []
        runs = resolve_category_runs(category, self._template_geometry)
        rect_size = template_rect_size_from_runs(runs["x_runs"], runs["y_runs"])
        is_l = category in _L_CATEGORIES
        for index, (angle, binary) in enumerate(zip(angles, binary_list)):
            height, width = binary.shape
            offset_x = (max_w - width) // 2 + margin
            offset_y = (max_h - height) // 2 + margin
            canvas = np.zeros((max_h, max_w), dtype=np.uint8)
            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            # 核必须画在画布正中：钉死左上角会使卷积搜索网格覆盖不到正确位置。
            cv2.drawContours(canvas, contours, -1, 255, 1, offset=(offset_x, offset_y))
            kernels[index, 0] = (canvas > 0).astype(np.float32)
            item = metadata[angle]
            anchor = (
                offset_x + item["anchor"][0],
                offset_y + item["anchor"][1],
            )
            anchors.append(anchor)
            if is_l:
                # 与 V1 coreect_LL_location 相同的方案A规则，用理想模板代替实测 Mask。
                pick_canvas = _template_l_pick_in_canvas(
                    binary,
                    item["bbox"],
                    item["anchor"],
                    rect_size,
                    angle,
                )
                pick = (
                    offset_x + pick_canvas[0] - item["bbox"][0],
                    offset_y + pick_canvas[1] - item["bbox"][1],
                )
            else:
                pick = anchor
            pick_anchors.append(pick)
        kernel_tensor = torch.from_numpy(kernels)
        if self.device == "cuda":
            kernel_tensor = kernel_tensor.to(torch.float16)
        entry = {
            "kernels": kernel_tensor.to(self.device),
            "size": (max_h, max_w),
            "angles": angles,
            "anchors": anchors,
            "pick_anchors": pick_anchors,
        }
        self._kernel_cache[key] = entry
        return entry

    def match_block(self, distance_field, detection, angle_center=None, angle_window=None):
        """对单个 YOLO 检测框做边缘模板匹配，返回 V1 口径结果。

        angle_center/angle_window（模板角，度）同时提供时只在折叠距离
        不超过窗口的网格角里搜索，供人工锁定角度后的重匹配使用。
        """
        category = normalize_category_name(detection["category"])
        if category not in ROTATION_TOTAL_ANGLE:
            raise ValueError(f"未知方块类别: {category}")
        x1, y1, x2, y2 = detection["box"]
        margin = self.config.crop_margin_px
        crop_x1 = max(0, int(x1) - margin)
        crop_y1 = max(0, int(y1) - margin)
        crop_x2 = min(distance_field.shape[1], int(x2) + margin)
        crop_y2 = min(distance_field.shape[0], int(y2) + margin)
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            raise ValueError("检测框裁剪后为空")

        roi = distance_field[crop_y1:crop_y2, crop_x1:crop_x2]
        if (angle_center is None) != (angle_window is None):
            raise ValueError("angle_center 与 angle_window 必须同时提供")
        if angle_center is None:
            entry = self._kernels_for(category)
        else:
            selected_angles = self._select_angles_in_window(
                category,
                angle_center,
                angle_window,
            )
            entry = self._kernels_for_angles(category, selected_angles)
        line_kernels = entry["kernels"]
        kernel_size = entry["size"]
        angles = entry["angles"]
        search_margin = self.config.search_margin_px
        padded_h = max(roi.shape[0], kernel_size[0] + 2 * search_margin)
        padded_w = max(roi.shape[1], kernel_size[1] + 2 * search_margin)
        pad_top = (padded_h - roi.shape[0]) // 2
        pad_left = (padded_w - roi.shape[1]) // 2
        padded = np.zeros((padded_h, padded_w), dtype=np.float32)
        padded[pad_top:pad_top + roi.shape[0], pad_left:pad_left + roi.shape[1]] = roi
        roi_tensor = torch.from_numpy(padded)[None, None]
        if self.device == "cuda":
            roi_tensor = roi_tensor.to(torch.float16)
        roi_tensor = roi_tensor.to(self.device)

        # 除以每核线像素数：分数 = 模板所有边缘点的距离场均值，越高越贴合。
        line_counts = line_kernels.sum(dim=(1, 2, 3)).clamp(min=1.0)
        with torch.no_grad():
            response = F.conv2d(roi_tensor, line_kernels)[0] / line_counts[:, None, None]
        best_per_angle = response.amax(dim=(1, 2))
        angle_index = int(torch.argmax(best_per_angle))
        best_map = response[angle_index]
        flat_index = int(torch.argmax(best_map))
        best_y, best_x = np.unravel_index(flat_index, best_map.shape)

        pick_x, pick_y = entry["pick_anchors"][angle_index]
        anchor_x, anchor_y = entry["anchors"][angle_index]
        angle = float(angles[angle_index])
        center_px = float(best_x + anchor_x - pad_left) + crop_x1
        center_py = float(best_y + anchor_y - pad_top) + crop_y1
        px = float(best_x + pick_x - pad_left) + crop_x1
        py = float(best_y + pick_y - pad_top) + crop_y1
        theta = (-angle + 180.0) % 360.0 - 180.0
        if abs(theta) < 1e-9:
            theta = 0.0
        return {
            "category": category,
            "px": px,
            "py": py,
            "center_px": center_px,
            "center_py": center_py,
            "theta": float(theta),
            "angle": angle,
            "match_score": float(best_map[best_y, best_x]),
            "score": float(detection.get("score", 1.0)),
            "crop_box": (crop_x1, crop_y1, crop_x2, crop_y2),
            "detection_box": tuple(float(value) for value in detection["box"]),
            "n_angles": len(angles),
        }

    def _draw_block_annotation(self, debug_image, match):
        """把单个方块匹配结果画到调试图：模板轮廓+中心十字+输出点+标签。"""
        px = match["px"]
        py = match["py"]
        # 匹配角度的模板轮廓画在旋转中心处，与 V1 get_rect 的绿色轮廓同风格。
        item = self._metadata_for(match["category"])[match["angle"]]
        template_contours, _ = cv2.findContours(
            item["binary"],
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        template_origin = (
            int(round(match["center_px"] - item["anchor"][0])),
            int(round(match["center_py"] - item["anchor"][1])),
        )
        contour_color = (0, 200, 255) if match.get("angle_locked") else (0, 255, 0)
        cv2.drawContours(
            debug_image,
            template_contours,
            -1,
            contour_color,
            1,
            offset=template_origin,
        )
        cv2.drawMarker(
            debug_image,
            (int(match["center_px"]), int(match["center_py"])),
            (255, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
            thickness=1,
        )
        cv2.circle(debug_image, (int(px), int(py)), 3, (0, 0, 255), 2)
        label_x = max(0, int(match["detection_box"][0]))
        label_y = max(15, int(match["detection_box"][1]) - 6)
        cv2.putText(
            debug_image,
            f"{match['category']} ({px:.0f},{py:.0f})",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    def detect_blocks(self, img_bgr, detections):
        """对给定检测框逐个边缘匹配，返回方块列表和调试图。"""
        if img_bgr is None or img_bgr.size == 0:
            return [], None
        distance_field = build_edge_distance_field(img_bgr, self.config)
        debug_image = np.copy(img_bgr)
        blocks = []
        for detection in detections:
            try:
                match = self.match_block(distance_field, detection)
            except Exception as exc:
                print(f"方块 {detection.get('category')} V2 边缘匹配失败，跳过。原因：{exc}")
                continue
            self._draw_block_annotation(debug_image, match)
            match["found"] = True
            match["debug_image"] = debug_image
            match["message"] = "V2 边缘模板匹配成功"
            blocks.append(match)
        return blocks, debug_image

    def rematch_blocks_with_angle_locks(self, img_bgr, blocks, angle_locks):
        """按人工锁定的正式 theta 重匹配对应方块，其余保持原结果。

        angle_locks: {方块下标(0 起): 正式 theta(度，[-180, 180))}。
        锁定后在最近网格角 ±(角度步进/2) 内只做位置搜索，px/py 与 L 抓点
        随重匹配结果一起更新；输出 theta 是最近网格角的等价正式角度，
        与人工输入最多差半个角度步进。未锁定的方块原样保留。
        """
        locks = {}
        for raw_index, raw_theta in dict(angle_locks or {}).items():
            index = int(raw_index)
            if not 0 <= index < len(blocks):
                raise ValueError(
                    f"角度锁定下标越界：{index}（本轮共 {len(blocks)} 个方块）"
                )
            theta = float(raw_theta)
            if not np.isfinite(theta):
                raise ValueError(f"第 {index + 1} 个方块锁定角度不是有限数值")
            locks[index] = theta

        results = [dict(block) for block in blocks]
        if not locks:
            debug_image = np.copy(img_bgr)
            for block in results:
                self._draw_block_annotation(debug_image, block)
                block["debug_image"] = debug_image
            return results, debug_image

        distance_field = build_edge_distance_field(img_bgr, self.config)
        for index, theta in locks.items():
            block = results[index]
            category = block["category"]
            period = float(ROTATION_TOTAL_ANGLE[category])
            template_center = ((-theta) % 360.0) % period
            detection = {
                "category": category,
                "score": block.get("score", 1.0),
                "box": tuple(block["detection_box"]),
            }
            rematched = self.match_block(
                distance_field,
                detection,
                angle_center=template_center,
                angle_window=self.config.angle_step_deg / 2.0,
            )
            rematched["found"] = True
            rematched["message"] = "V2 边缘模板匹配成功（人工锁定角度重匹配）"
            rematched["angle_locked"] = True
            results[index] = rematched
            print(
                f"方块 {category} 锁定角度 {theta:.1f}° 重匹配："
                f"({rematched['px']:.1f},{rematched['py']:.1f}) "
                f"theta={rematched['theta']:.1f}°"
            )

        debug_image = np.copy(img_bgr)
        for block in results:
            self._draw_block_annotation(debug_image, block)
            block["debug_image"] = debug_image
        return results, debug_image
