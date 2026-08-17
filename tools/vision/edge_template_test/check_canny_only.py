# -*- coding: utf-8 -*-
"""诊断：纯 Canny 重叠（变体A，不用距离场）为什么角度差几度。

回答三个问题：
1. 目标函数冤不冤：V1 位姿的重叠率 vs A 位姿的重叠率，谁高？
   V1 高 → A 连自己的最优都没找到（搜索/筛角问题）
   A 高  → 重叠这个目标函数本身就偏好错误位姿（目标函数问题）
2. 用户的肉眼观察：在 A 自己的位置上 ±12° 细扫 0.25°，是否存在
   "再转几度就更好"的更高峰？峰在哪？
3. 角度步进 2°→1° 重跑一遍 A，统计能不能救。
"""
import json
import os
import sys
import time

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
SRC = "/home/zhl/SingleArmTetris/SingleArmTetris/src"
THIS = os.path.dirname(os.path.abspath(__file__))
for p in (SRC, os.path.join(SRC, "image_process"),
          os.path.join(SRC, "tools/vision/oriented_chamfer_v1"), THIS):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np

from image_process_lib.template_match.kernels_create import (
    ROTATION_TOTAL_ANGLE,
    build_angle_foreground_metadata,
    create_base_shape,
    embed_in_center,
    rotate_image,
)
from image_process_lib.template_config import load_template_geometry
import oriented_chamfer_v1 as v1
import edge_conv_test as ect  # 复用它的 run_variant 做 1° 步重跑

# ---------------- 参数区 ----------------
SWEEP_DEG = 12.0     # 在目标角度附近细扫范围
SWEEP_STEP = 0.25
SCREEN_TOL_PX = 6
FINE_ANGLE_STEP = 1.0   # 问题3的加密步进
PAD = 200            # 全图边缘图外补零，防模板出界
# ---------------------------------------

img = cv2.imread(os.path.join(SRC, "tools/vision/7.png"))
geo = load_template_geometry("high")
bpp, cpp = geo["block_px"], geo["connector_px"]

# 全图 Canny（诊断统一在同一个边缘图上评估，避免 ROI 边界差异）
gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
edges = cv2.Canny(gray, 50, 150)
edges_pad = np.zeros((img.shape[0] + 2 * PAD, img.shape[1] + 2 * PAD), dtype=np.float32)
edges_pad[PAD:PAD + img.shape[0], PAD:PAD + img.shape[1]] = (edges > 0)

conv = json.load(open(os.path.join(THIS, "output", "edge_conv_results.json")))
v1refs = [r for r in json.load(open(
    os.path.join(SRC, "tools/vision/oriented_chamfer_v1/output/v1_results.json"))
)["results"] if r["found"]]

base_canvas = {}
line_cache = {}


def line_kernel(cat, angle):
    """任意角度的 1px 外轮廓线模板 + 旋转中心锚点。"""
    key = (cat, round(float(angle), 3))
    if key in line_cache:
        return line_cache[key]
    if cat not in base_canvas:
        shape = create_base_shape(cat, bpp, cpp)
        h, w = shape.shape
        length = int(np.ceil(np.sqrt(w * w + h * h)))
        if length % 2 == 0:
            length += 1
        base_canvas[cat] = (embed_in_center(shape, length), length // 2)
    canvas, center = base_canvas[cat]
    rotated = rotate_image(canvas, float(angle))
    binary = (rotated > 0.5).astype(np.uint8)
    ys, xs = np.nonzero(binary)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    tight = binary[y1:y2, x1:x2]
    line = np.zeros_like(tight)
    cnts, _ = cv2.findContours(tight, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(line, cnts, -1, 1, 1)
    out = (line, (center - int(x1), center - int(y1)))
    line_cache[key] = out
    return out


def overlap_fraction(cat, px, py, angle):
    kernel, (ax, ay) = line_kernel(cat, angle)
    h, w = kernel.shape
    tlx = int(round(px - ax)) + PAD
    tly = int(round(py - ay)) + PAD
    region = edges_pad[tly:tly + h, tlx:tlx + w]
    return float((region * kernel).sum() / kernel.sum())


def angle_diff(a, b, period):
    d = abs(a - b) % period
    return min(d, period - d)


meta2 = {}   # 2° 步 metadata（复现变体A的筛角集合）
meta1 = {}   # 1° 步 metadata（问题3重跑）
print(f"{'case':<7s}{'cat':<10s}{'A角':>7s}{'V1角':>7s}{'fracA':>7s}{'fracV1':>7s}"
      f"{'A位细扫峰':>9s}{'峰角差A':>8s}{'V1角∈筛角':>9s}  判定")
verd = {"objective": 0, "search": 0, "both": 0, "ok": 0}
for i, (r, vr) in enumerate(zip(conv, v1refs)):
    cat = r["category"]
    a = r["A"]
    if a.get("d_pos", 0) <= 2:
        verd["ok"] += 1
        continue
    period = float(ROTATION_TOTAL_ANGLE[cat])
    if cat not in meta2:
        meta2[cat] = build_angle_foreground_metadata(cat, bpp, cpp, 2.0)
    x1, y1, x2, y2 = vr["detection_box"]
    bw, bh = int(x2 - x1), int(y2 - y1)
    cand = [ang for ang in sorted(meta2[cat])
            if abs(meta2[cat][ang]["fg_w"] - bw) <= SCREEN_TOL_PX
            and abs(meta2[cat][ang]["fg_h"] - bh) <= SCREEN_TOL_PX]
    v1_ang_in = any(angle_diff(ang, vr["pose_angle_deg"], period) < 1.5 for ang in cand)

    fracA = overlap_fraction(cat, a["px"], a["py"], a["angle"])
    fracV1 = overlap_fraction(cat, vr["px"], vr["py"], vr["pose_angle_deg"])
    # 在 A 自己的位置上细扫
    offs = np.arange(-SWEEP_DEG, SWEEP_DEG + 1e-6, SWEEP_STEP)
    sweeps = [(float(o), overlap_fraction(cat, a["px"], a["py"], a["angle"] + o)) for o in offs]
    best_off, best_frac = max(sweeps, key=lambda t: t[1])

    if fracV1 > fracA + 0.01:
        v = "search" if v1_ang_in else "search(筛角丢了V1角)"
    elif best_frac > fracA + 0.01:
        v = "objective(细扫更高)" if abs(best_off) > 0.5 else "objective"
    else:
        v = "objective(A确是局部最优)"
    verd["search" if v.startswith("search") else "objective"] += 1
    print(f"[{i:02d}]   {cat:<9s}{a['angle']:7.1f}{vr['pose_angle_deg']:7.1f}"
          f"{fracA:7.3f}{fracV1:7.3f}{best_frac:9.3f}{best_off:8.1f}{'是' if v1_ang_in else '否':>8s}  {v}")

print(f"\n判定统计: {verd}")
print("(search=重叠函数下V1位姿更优但A没找到；objective=重叠函数本身就偏好错误位姿)\n")

# ---------------- 问题3：A 用 1° 步重跑 ----------------
from ultralytics import YOLO

yolo = YOLO(v1.DETECTION_MODEL_PATH)
yolo(img, verbose=False)
dets = v1.detect_blocks_yolo_v1(img, yolo, v1.ChamferV1Config())
d_pos, d_ang, ms = [], [], []
print("A@1°步进 重跑（筛角容差不变）...")
for det, vr in zip(dets, v1refs):
    cat = det["category"]
    x1, y1, x2, y2 = det["box"]
    cx1, cy1 = max(0, int(x1) - ect.CROP_MARGIN), max(0, int(y1) - ect.CROP_MARGIN)
    cx2 = min(img.shape[1], int(x2) + ect.CROP_MARGIN)
    cy2 = min(img.shape[0], int(y2) + ect.CROP_MARGIN)
    roi = img[cy1:cy2, cx1:cx2]
    g = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges_roi = (cv2.Canny(g, 50, 150) > 0).astype(np.float32)
    if cat not in meta1:
        meta1[cat] = build_angle_foreground_metadata(cat, bpp, cpp, FINE_ANGLE_STEP)
    angles = sorted(meta1[cat])
    out = ect.run_variant("A", edges_roi, meta1[cat], angles, (cx1, cy1),
                          (int(x2 - x1), int(y2 - y1)), SCREEN_TOL_PX)
    period = float(ROTATION_TOTAL_ANGLE[cat])
    d_pos.append(float(np.hypot(out["px"] - vr["px"], out["py"] - vr["py"])))
    d_ang.append(float(angle_diff(out["angle"], vr["pose_angle_deg"], period)))
    ms.append(out["ms"])
d_pos_s = sorted(d_pos)
n = len(d_pos)
print(f"A@1°: 中位Δpx={d_pos_s[n//2]:.2f} <2px:{sum(x<2 for x in d_pos)/n:.0%} "
      f"<5px:{sum(x<5 for x in d_pos)/n:.0%} 中位Δ角={sorted(d_ang)[n//2]:.2f}° "
      f"均耗时={np.mean(ms):.1f}ms  (对比 A@2°: 中位0.82px <5px:77% 中位角1.1° 9.6ms)")
