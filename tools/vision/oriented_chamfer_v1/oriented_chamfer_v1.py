# -*- coding: utf-8 -*-
"""
V1 顶面轮廓配准核心库：YOLO 类别 + 粗 ROI + Oriented Chamfer + 中心先验。

这是一个【离线工具库】，不接入正式识别流程。正式流程仍使用
image_process/image_process_lib/block_scene_detector.py 等原文件。

它只复用比赛里的三样既有资产（全部只读）：
1. competition/model/best5.14.pt          YOLO 检测类别/粗框
2. competition/config/template_config.yaml 模板像素尺寸（high/low profile）
3. image_process_lib.template_match.kernels_create.create_base_shape
   与正式流程完全相同的方块顶面二值形状

匹配目标不是纹理，而是 create_base_shape 生成形状的【外轮廓】。
对 ROI Canny 图做 distance transform，搜索 (x, y, theta)，同时加入：
  - Canny 边缘方向 vs 模板轮廓切线方向的一致性
  - YOLO bbox 中心的软先验（只惩罚，不锁死）

运行示例：
  PYTHONPATH=image_process python3 tools/vision/oriented_chamfer_v1/run_v1_test.py \
      --image tools/vision/7.png --output-dir /tmp/v1_out
"""

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# 允许直接以工具目录运行时也能 import 正式流程里的几何定义。
try:
    from image_process_lib.block_category import normalize_category_name
    from image_process_lib.template_config import load_template_geometry
    from image_process_lib.template_match.kernels_create import (
        ROTATION_TOTAL_ANGLE,
        create_base_shape,
    )
except ImportError:  # 独立运行时，手工把 image_process 包目录加进 sys.path。
    _THIS_FILE = os.path.abspath(__file__)
    _SRC_DIR = os.path.abspath(os.path.join(
        _THIS_FILE, "..", "..", "..", "..", "image_process"))
    if os.path.isdir(_SRC_DIR):
        import sys
        if _SRC_DIR not in sys.path:
            sys.path.insert(0, _SRC_DIR)
        from image_process_lib.block_category import normalize_category_name  # noqa: E402
        from image_process_lib.template_config import load_template_geometry  # noqa: E402
        from image_process_lib.template_match.kernels_create import (  # noqa: E402
            ROTATION_TOTAL_ANGLE,
            create_base_shape,
        )
    else:
        raise


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DETECTION_MODEL_PATH = os.path.join(SRC_DIR, "competition", "model", "best5.14.pt")
TEMPLATE_CONFIG_PATH = os.path.join(SRC_DIR, "competition", "config", "template_config.yaml")

SUPPORTED_CATEGORIES = set(ROTATION_TOTAL_ANGLE.keys())


@dataclass
class ChamferV1Config:
    """V1 全部可调参数。数值都是像素 / 度，便于现场直接改。"""

    # ---- ROI / Canny ----
    roi_context_margin: float = 6.0     # 保证模板旋转后仍在 ROI 内的额外像素
    canny_low: float = 50.0
    canny_high: float = 150.0
    gaussian_ksize: int = 3

    # ---- 模板轮廓 ----
    contour_sample_step: float = 1.5    # 轮廓点按弧长重采样间隔，1.5 px 足够
    distance_cap: float = 20.0          # Chamfer 距离截断，抗远处背景边

    # ---- 方向一致性 ----
    orientation_enabled: bool = True
    orientation_weight: float = 0.8     # 0 时退化为普通 Chamfer + 中心先验
    orientation_radius: float = 4.0     # 只对 4px 内的 Canny 边缘算方向误差

    # ---- 中心软先验：lambda * ((x-cx)^2 + (y-cy)^2) ----
    center_prior_weight: float = 0.005

    # ---- 第一阶段粗搜 ----
    coarse_angle_step: float = 2.0
    coarse_xy_step: float = 1.0
    coarse_search_radius: float = 14.0  # YOLO 中心附近 +/- 14 px
    coarse_top_k: int = 5               # 保留 K 个粗搜候选进入精搜

    # ---- 第二阶段精搜 ----
    fine_angle_step: float = 0.2
    fine_angle_window: float = 2.5      # 粗搜中心附近 +/- 2.5 deg
    fine_xy_step: float = 0.5
    fine_xy_radius: float = 3.0

    # ---- YOLO ----
    detection_conf: float = 0.45
    detection_iou: float = 0.5

    # ---- 模板几何 ----
    template_profile: str = "high"


@dataclass
class TemplateContourModel:
    """已知类别的顶面外轮廓模板。"""

    category: str
    period: float
    base_shape: np.ndarray
    points: np.ndarray                 # (N, 2)，已相对模板中心
    tangents: np.ndarray               # (N,)，0..180 deg
    radius: float
    center: Tuple[float, float]
    rect_size: Tuple[int, int]

    def rotated(self, angle_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回旋转后的轮廓点 (rx, ry) 与切线方向（0..180 deg）。

        旋转公式与 cv2.getRotationMatrix2D(..., angle, 1.0) 完全一致，
        因此这里的 angle_deg 就是正式流程 kernels_create.rotate_image 的模板角。
        """
        a = math.radians(float(angle_deg))
        ca, sa = math.cos(a), math.sin(a)
        rx = self.points[:, 0] * ca + self.points[:, 1] * sa
        ry = -self.points[:, 0] * sa + self.points[:, 1] * ca
        tvx = self.tangents[:, 0] * ca + self.tangents[:, 1] * sa
        tvy = -self.tangents[:, 0] * sa + self.tangents[:, 1] * ca
        tangent_angle = np.degrees(np.arctan2(tvy, tvx)) % 180.0
        return rx, ry, tangent_angle


class EdgeObservation:
    """ROI 的 Canny 边缘 + 距离场 + 最近边缘方向场。"""

    def __init__(self, bgr_roi: np.ndarray, config: ChamferV1Config):
        if bgr_roi is None or bgr_roi.ndim != 3 or bgr_roi.shape[0] < 3 or bgr_roi.shape[1] < 3:
            raise ValueError("ROI 图像为空或过小")

        self.bgr = bgr_roi
        self.height, self.width = bgr_roi.shape[:2]
        self.gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
        ksize = int(config.gaussian_ksize)
        if ksize % 2 == 0:
            ksize += 1
        self.blur = cv2.GaussianBlur(self.gray, (ksize, ksize), 0)
        self.edges = cv2.Canny(
            self.blur,
            config.canny_low,
            config.canny_high,
        )
        edge_count = int(np.count_nonzero(self.edges))
        if edge_count <= 0:
            raise RuntimeError("ROI 内 Canny 没有检测到任何边缘")

        # 0 表示 Canny 边缘像素；distance transform 后每个像素是到最近边缘的距离。
        self.dist, self.labels = cv2.distanceTransformWithLabels(
            255 - self.edges,
            cv2.DIST_L2,
            3,
            cv2.DIST_LABEL_PIXEL,
        )

        # 边缘方向：Sobel 梯度是法向，+90° 得切线；边缘无方向，所以折到 0..180。
        gx = cv2.Sobel(self.blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(self.blur, cv2.CV_32F, 0, 1, ksize=3)
        grad_angle = np.degrees(np.arctan2(gy, gx))
        self.edge_tangent = (grad_angle + 90.0) % 180.0

        # DIST_LABEL_PIXEL 的 labels 是 1-based 的边缘像素序号（按 y,x 扫描）。
        # 用它把每个非边缘像素映射到最近边缘像素的切线方向。
        edge_orientations = self.edge_tangent[self.edges > 0].ravel()
        self.nearest_tangent = edge_orientations[np.maximum(self.labels - 1, 0)]

    def sample_tangent(self, qx: np.ndarray, qy: np.ndarray) -> np.ndarray:
        """查询像素位置的边缘方向。

        落在边缘像素上时用该像素自己的 Sobel 方向；否则用距离变换 label
        给出的最近边缘方向。前者避免 OpenCV 在相邻等距边缘上标签不唯一。
        """
        on_edge = self.edges[qy, qx] > 0
        return np.where(on_edge, self.edge_tangent[qy, qx], self.nearest_tangent[qy, qx])


def _angle_diff_180(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """无向直线夹角，0..90 deg。"""
    d = np.abs(a - b) % 180.0
    return np.minimum(d, 180.0 - d)


def _normalize_formal_theta(pose_angle_deg: float) -> float:
    """把模板生成角转换成正式流程 match_block_mask 输出的 theta。

    正式流程 get_rect 里 rect angle = -模板角，match_block_mask 只处理
    theta < -180 时加 360。这里统一规范到 [-180, 180)。
    """
    value = (-float(pose_angle_deg) + 180.0) % 360.0 - 180.0
    if abs(value) < 1e-9:
        return 0.0
    return float(value)


def _resample_contour(contour: np.ndarray, sample_step: float) -> Tuple[np.ndarray, np.ndarray]:
    """按弧长重采样闭合轮廓，返回 (点 Nx2, 单位切向量 Nx2)。"""
    contour = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(contour) <= 3:
        raise ValueError("模板轮廓点过少")

    # findContours 的闭合轮廓首尾重复，去掉最后一点。
    if np.allclose(contour[0], contour[-1], atol=1e-9):
        contour = contour[:-1]
    if len(contour) <= 3:
        raise ValueError("模板轮廓点过少")

    diffs = np.roll(contour, -1, axis=0) - contour
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    perimeter = float(np.sum(seg_len))
    if perimeter <= 1e-6:
        raise ValueError("模板轮廓周长过小")

    n_points = max(8, int(round(perimeter / float(sample_step))))
    cumulative = np.concatenate(([0.0], np.cumsum(seg_len)))
    sample_u = np.linspace(0.0, perimeter, n_points, endpoint=False)
    points = np.empty((n_points, 2), dtype=np.float64)
    for axis in (0, 1):
        # np.interp(period=perimeter) 让闭合轮廓可以循环插值。
        points[:, axis] = np.interp(
            sample_u,
            cumulative,
            np.append(contour[:, axis], contour[0, axis]),
            period=perimeter,
        )

    prev = np.roll(points, 1, axis=0)
    nxt = np.roll(points, -1, axis=0)
    tangents = nxt - prev
    norm = np.hypot(tangents[:, 0], tangents[:, 1])
    if np.any(norm <= 1e-9):
        norm = np.maximum(norm, 1e-9)
    tangents = tangents / norm[:, None]
    return points, tangents


def build_template_contour_model(
    category: str,
    block_px: int,
    connector_px: int,
    sample_step: float = 1.5,
) -> TemplateContourModel:
    """用正式 kernels_create.create_base_shape 生成同类别的顶面外轮廓。"""
    category = normalize_category_name(category)
    if category not in ROTATION_TOTAL_ANGLE:
        raise ValueError(f"未知方块类别: {category}")

    base_shape = create_base_shape(category, block_px, connector_px)
    binary = (base_shape > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        raise ValueError(f"类别 {category} 无法生成模板轮廓")

    contour = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    points, tangents = _resample_contour(contour, sample_step)

    height, width = base_shape.shape
    center = ((width - 1.0) / 2.0, (height - 1.0) / 2.0)
    points = points - np.asarray(center, dtype=np.float64)[None, :]
    radius = float(np.max(np.hypot(points[:, 0], points[:, 1])))
    rect_size = (int(width), int(height))

    return TemplateContourModel(
        category=category,
        period=float(ROTATION_TOTAL_ANGLE[category]),
        base_shape=base_shape,
        points=points,
        tangents=tangents,
        radius=radius,
        center=center,
        rect_size=rect_size,
    )


def load_geometry(profile: Optional[str] = None) -> Dict[str, int]:
    """读取正式 template_config.yaml 的模板几何参数。"""
    geometry = load_template_geometry(profile or "high", config_path=TEMPLATE_CONFIG_PATH)
    return {
        "block_px": int(geometry["block_px"]),
        "connector_px": int(geometry["connector_px"]),
    }


def _make_roi_crop(
    image_shape: Tuple[int, int],
    detection_box: Sequence[float],
    template_radius: float,
    config: ChamferV1Config,
) -> Tuple[int, int, int, int]:
    """在 YOLO bbox 基础上扩大 ROI。

    以 bbox 中心为基准，保证“模板最大半径 + 粗搜半径 + 少量背景”都在 ROI 内；
    如果 bbox 本身就很大，则沿用 bbox + 常规 margin，不缩小。
    """
    image_h, image_w = image_shape[:2]
    x1, y1, x2, y2 = (float(v) for v in detection_box)
    box_cx = (x1 + x2) / 2.0
    box_cy = (y1 + y2) / 2.0

    required_half = (
        template_radius
        + config.coarse_search_radius
        + config.roi_context_margin
        + 2.0
    )
    # 至少把整个 bbox 包住，防止极小的误检框。
    bbox_required_half = max(
        (x2 - x1) / 2.0 + config.roi_context_margin,
        (y2 - y1) / 2.0 + config.roi_context_margin,
    )
    half = max(required_half, bbox_required_half)

    crop_x1 = int(math.floor(box_cx - half))
    crop_y1 = int(math.floor(box_cy - half))
    crop_x2 = int(math.ceil(box_cx + half))
    crop_y2 = int(math.ceil(box_cy + half))

    # 靠图像边缘时尽量给足，再统一 clamp。
    if crop_x1 < 0:
        crop_x2 += -crop_x1
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y2 += -crop_y1
        crop_y1 = 0
    if crop_x2 > image_w:
        crop_x1 -= crop_x2 - image_w
        crop_x2 = image_w
    if crop_y2 > image_h:
        crop_y1 -= crop_y2 - image_h
        crop_y2 = image_h
    crop_x1 = max(0, min(crop_x1, image_w - 1))
    crop_y1 = max(0, min(crop_y1, image_h - 1))
    crop_x2 = max(crop_x1 + 1, min(crop_x2, image_w))
    crop_y2 = max(crop_y1 + 1, min(crop_y2, image_h))
    return crop_x1, crop_y1, crop_x2, crop_y2


def _angle_range(period: float, step: float) -> np.ndarray:
    count = int(math.floor(period / float(step)))
    if count <= 0:
        count = 1
    return np.arange(count, dtype=np.float64) * float(step)


def _score_candidates(
    observation: EdgeObservation,
    model: TemplateContourModel,
    angle_deg: float,
    center_xs: np.ndarray,
    center_ys: np.ndarray,
    prior_x: float,
    prior_y: float,
    config: ChamferV1Config,
) -> Dict[str, np.ndarray]:
    """对同一角度的一批中心位置做向量化打分。

    center_xs/center_ys 是一维数组，内部 meshgrid 后返回的每个 score 数组
    形状均为 (len(center_ys), len(center_xs))，与 np.meshgrid(xs, ys) 一致。
    """
    rx, ry, tangent_angle = model.rotated(angle_deg)
    X, Y = np.meshgrid(center_xs, center_ys, indexing="xy")

    # (ny, nx, n_points)
    qx = np.round(X[:, :, None] + rx[None, None, :]).astype(np.int32)
    qy = np.round(Y[:, :, None] + ry[None, None, :]).astype(np.int32)

    # 所有中心均已通过 valid_bounds 检查，正常情况下不会越界；这里仍做保护。
    np.clip(qx, 0, observation.width - 1, out=qx)
    np.clip(qy, 0, observation.height - 1, out=qy)

    dist = observation.dist[qy, qx]                    # (ny, nx, n_points)
    dist_cost = np.minimum(dist, config.distance_cap).mean(axis=2)

    if config.orientation_enabled and config.orientation_weight > 0:
        edge_ori = observation.sample_tangent(qx, qy)
        ori_err = _angle_diff_180(tangent_angle[None, None, :], edge_ori)
        falloff = np.maximum(0.0, 1.0 - dist / max(config.orientation_radius, 1e-9))
        denom = falloff.sum(axis=2)
        ori_cost = np.divide(
            (falloff * ori_err).sum(axis=2),
            denom,
            out=np.zeros_like(denom),
            where=denom > 1e-9,
        ) / 90.0
    else:
        ori_cost = np.zeros_like(dist_cost)

    center_cost = config.center_prior_weight * (
        (X - prior_x) ** 2 + (Y - prior_y) ** 2
    )
    total = dist_cost + config.orientation_weight * ori_cost + center_cost
    return {
        "total": total,
        "dist_cost": dist_cost,
        "ori_cost": ori_cost,
        "center_cost": center_cost,
        "coverage_2px": (dist <= 2.0).mean(axis=2),
        "mean_dist_raw": dist.mean(axis=2),
    }


def _valid_center_bounds(
    model: TemplateContourModel,
    angle_deg: float,
    observation: EdgeObservation,
) -> Tuple[float, float, float, float]:
    """该角度下，保证全部模板轮廓点都落在 ROI 内的中心范围。"""
    rx, ry, _ = model.rotated(angle_deg)
    min_x = math.ceil(float(-rx.max()))
    max_x = math.floor(float(observation.width - 1 - rx.max()))
    min_y = math.ceil(float(-ry.min()))
    max_y = math.floor(float(observation.height - 1 - ry.max()))
    # 用 min/max 对称修正，避免浮点误差导致 min>max。
    min_x = min(min_x, max_x)
    min_y = min(min_y, max_y)
    return min_x, max_x, min_y, max_y


def _center_grid(
    prior_x: float,
    prior_y: float,
    radius: float,
    step: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成与先验中心和有效边界取交集的中心网格。"""
    x_start = max(min_x, prior_x - radius)
    x_end = min(max_x, prior_x + radius)
    y_start = max(min_y, prior_y - radius)
    y_end = min(max_y, prior_y + radius)
    if x_end < x_start or y_end < y_start:
        return np.array([]), np.array([])
    xs = np.arange(x_start, x_end + 1e-9, step, dtype=np.float64)
    ys = np.arange(y_start, y_end + 1e-9, step, dtype=np.float64)
    if step > 0:
        xs = xs[xs <= x_end + 1e-9]
        ys = ys[ys <= y_end + 1e-9]
    return xs, ys


def _find_best_in_grid(scores: Dict[str, np.ndarray], xs: np.ndarray, ys: np.ndarray):
    total = scores["total"]
    idx = int(np.argmin(total))
    iy, ix = np.unravel_index(idx, total.shape)
    return {
        "total": float(total[iy, ix]),
        "angle": None,
        "x": float(xs[ix]),
        "y": float(ys[iy]),
        "dist_cost": float(scores["dist_cost"][iy, ix]),
        "ori_cost": float(scores["ori_cost"][iy, ix]),
        "center_cost": float(scores["center_cost"][iy, ix]),
        "coverage_2px": float(scores["coverage_2px"][iy, ix]),
        "mean_dist_raw": float(scores["mean_dist_raw"][iy, ix]),
    }


def match_block_v1(
    img_bgr: np.ndarray,
    category: str,
    detection_box: Sequence[float],
    config: Optional[ChamferV1Config] = None,
    geometry: Optional[Dict[str, int]] = None,
    debug: bool = False,
) -> Dict:
    """对单个 YOLO 检测框做 V1 顶面轮廓配准。

    返回完整图坐标 px/py、模板角 pose_angle_deg、正式流程等价 theta、
    分数和搜索耗时。ROI 内的局部坐标放在 roi_* 字段里。
    """
    started = time.perf_counter()
    config = config or ChamferV1Config()
    geometry = geometry or load_geometry(config.template_profile)
    category = normalize_category_name(category)

    model = build_template_contour_model(
        category,
        geometry["block_px"],
        geometry["connector_px"],
        config.contour_sample_step,
    )

    image_h, image_w = img_bgr.shape[:2]
    x1, y1, x2, y2 = _make_roi_crop(
        (image_h, image_w),
        detection_box,
        model.radius,
        config,
    )
    roi_bgr = img_bgr[y1:y2, x1:x2]
    observation = EdgeObservation(roi_bgr, config)

    prior_local_x = ((float(detection_box[0]) + float(detection_box[2])) / 2.0) - x1
    prior_local_y = ((float(detection_box[1]) + float(detection_box[3])) / 2.0) - y1

    # ---------- 第一阶段：粗搜 ----------
    coarse_started = time.perf_counter()
    coarse_angles = _angle_range(model.period, config.coarse_angle_step)
    coarse_candidates: List[Dict] = []
    for angle in coarse_angles:
        min_x, max_x, min_y, max_y = _valid_center_bounds(model, angle, observation)
        xs, ys = _center_grid(
            prior_local_x,
            prior_local_y,
            config.coarse_search_radius,
            config.coarse_xy_step,
            min_x,
            max_x,
            min_y,
            max_y,
        )
        if len(xs) == 0 or len(ys) == 0:
            continue
        scores = _score_candidates(
            observation,
            model,
            angle,
            xs,
            ys,
            prior_local_x,
            prior_local_y,
            config,
        )
        best = _find_best_in_grid(scores, xs, ys)
        best["angle"] = float(angle)
        coarse_candidates.append(best)

    if not coarse_candidates:
        raise RuntimeError("粗搜没有产生任何有效候选，请检查 ROI、搜索半径和模板尺寸")

    coarse_candidates.sort(key=lambda item: item["total"])
    coarse_top = coarse_candidates[: max(1, int(config.coarse_top_k))]
    coarse_elapsed_ms = (time.perf_counter() - coarse_started) * 1000.0

    # ---------- 第二阶段：精搜 ----------
    fine_started = time.perf_counter()
    fine_candidates: List[Dict] = []
    for seed in coarse_top:
        seed_angle = float(seed["angle"])
        angle_start = seed_angle - config.fine_angle_window
        angle_end = seed_angle + config.fine_angle_window
        fine_angle_count = int(math.floor(
            (angle_end - angle_start) / config.fine_angle_step
        )) + 1
        fine_angles = (
            angle_start + np.arange(fine_angle_count, dtype=np.float64) * config.fine_angle_step
        )

        for angle in fine_angles:
            # 角度允许越过周期边界，但需折叠回合法范围，并避免重复越界角。
            wrapped = float(angle) % model.period
            min_x, max_x, min_y, max_y = _valid_center_bounds(model, wrapped, observation)
            xs, ys = _center_grid(
                seed["x"],
                seed["y"],
                config.fine_xy_radius,
                config.fine_xy_step,
                min_x,
                max_x,
                min_y,
                max_y,
            )
            if len(xs) == 0 or len(ys) == 0:
                continue
            scores = _score_candidates(
                observation,
                model,
                wrapped,
                xs,
                ys,
                prior_local_x,
                prior_local_y,
                config,
            )
            best = _find_best_in_grid(scores, xs, ys)
            best["angle"] = wrapped
            fine_candidates.append(best)

    # 精搜失败时回退粗搜最优。
    if fine_candidates:
        best = min(fine_candidates, key=lambda item: item["total"])
    else:
        best = coarse_top[0]
    fine_elapsed_ms = (time.perf_counter() - fine_started) * 1000.0

    pose_angle = float(best["angle"])
    local_px = float(best["x"]) + x1
    local_py = float(best["y"]) + y1
    result = {
        "found": True,
        "category": category,
        "px": local_px,
        "py": local_py,
        "pose_angle_deg": pose_angle,
        "theta": _normalize_formal_theta(pose_angle),
        "score": float(best["total"]),
        "score_components": {
            "dist_cost": float(best["dist_cost"]),
            "ori_cost": float(best["ori_cost"]),
            "center_cost": float(best["center_cost"]),
        },
        "coverage_2px": float(best["coverage_2px"]),
        "mean_dist_raw": float(best["mean_dist_raw"]),
        "detection_box": tuple(float(v) for v in detection_box),
        "roi_box": (int(x1), int(y1), int(x2), int(y2)),
        "roi_local": {
            "x": float(best["x"]),
            "y": float(best["y"]),
            "prior_x": float(prior_local_x),
            "prior_y": float(prior_local_y),
        },
        "rect_size": model.rect_size,
        "template_radius": float(model.radius),
        "coarse_candidates": coarse_top,
        "timing_ms": {
            "total": (time.perf_counter() - started) * 1000.0,
            "coarse": coarse_elapsed_ms,
            "fine": fine_elapsed_ms,
        },
        "model": model,
        "observation": observation,
    }

    if debug:
        result["debug"] = _make_debug_payload(img_bgr, result)
    return result


def _make_debug_payload(img_bgr: np.ndarray, result: Dict) -> Dict:
    model: TemplateContourModel = result["model"]
    observation: EdgeObservation = result["observation"]
    x1, y1, x2, y2 = result["roi_box"]

    full_vis = img_bgr.copy()
    _draw_pose(full_vis, model, result["px"], result["py"], result["pose_angle_deg"], (0, 200, 0), 2)
    px = int(round(result["px"]))
    py = int(round(result["py"]))
    cv2.circle(full_vis, (px, py), 3, (0, 0, 255), -1)
    bx1, by1, bx2, by2 = result["detection_box"]
    cv2.rectangle(
        full_vis,
        (int(round(bx1)), int(round(by1))),
        (int(round(bx2)), int(round(by2))),
        (255, 200, 0),
        1,
    )

    roi_edges = cv2.cvtColor(observation.edges, cv2.COLOR_GRAY2BGR)
    local_px = result["roi_local"]["x"]
    local_py = result["roi_local"]["y"]
    _draw_pose(roi_edges, model, local_px, local_py, result["pose_angle_deg"], (0, 200, 0), 1)
    prior_x = result["roi_local"]["prior_x"]
    prior_y = result["roi_local"]["prior_y"]
    cv2.circle(roi_edges, (int(round(prior_x)), int(round(prior_y))), 4, (255, 200, 0), 1)
    cv2.line(
        roi_edges,
        (int(round(prior_x)), int(round(prior_y))),
        (int(round(local_px)), int(round(local_py))),
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    return {
        "full_vis": full_vis,
        "roi_edges": roi_edges,
    }


def _draw_pose(
    image: np.ndarray,
    model: TemplateContourModel,
    px: float,
    py: float,
    angle_deg: float,
    color,
    thickness: int,
):
    rx, ry, _ = model.rotated(angle_deg)
    pts = np.stack((rx + px, ry + py), axis=1).astype(np.int32)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def detect_blocks_yolo_v1(
    img_bgr: np.ndarray,
    model,
    config: Optional[ChamferV1Config] = None,
    expected_category: str = "",
) -> List[Dict]:
    """只读复用 best5.14.pt，返回与正式 detect_blocks_yolo 相似的检测框列表。

    这里刻意不 import block_scene_detector，因为那个模块会连带 import
    TensorRT 依赖；离线 V1 工具只需要 YOLO 检测和类别。
    """
    if img_bgr is None or img_bgr.size == 0:
        return []
    config = config or ChamferV1Config()
    expected_category = normalize_category_name(expected_category.strip()) if expected_category else ""
    detections = []
    results = model(
        img_bgr,
        iou=config.detection_iou,
        conf=config.detection_conf,
        verbose=False,
    )
    for det in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, cid = det
        category = normalize_category_name(model.names[int(cid)])
        if category not in SUPPORTED_CATEGORIES:
            continue
        if expected_category and category != expected_category:
            continue
        detections.append({
            "category": category,
            "score": float(score),
            "box": (float(x1), float(y1), float(x2), float(y2)),
        })
    return detections
