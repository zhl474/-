# -*- coding: utf-8 -*-
"""按相机方向区分顶-白/顶-侧 的边缘模板匹配（模仿 v2 几何意图，不用 LSD）。

原理：方块侧面只在朝向相机（图像中心）的方位可见。模板轮廓每个段的
外法线 n 与“块中心→图像中心”方向 d̂ 的点积：
    n·d̂ > 阈值  → 该段朝相机，真实边是顶-侧过渡（模糊）且附近 h·d̂ 处
                  存在自己方块的侧面外缘（假边）→ 低权重 w_side
    否则        → 顶面直接对白板，边清晰 → 高权重 w_white
权重直接烧进卷积核（逐段不同值的线核），conv 自动算加权分。

先输出失败块偏移方向的诊断（偏移·d̂ > 0 = 模板被拉向中心 = 锁到侧面外缘）。
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
import torch
import torch.nn.functional as F

from image_process_lib.template_match.kernels_create import (
    ROTATION_TOTAL_ANGLE,
    build_angle_foreground_metadata,
)
from image_process_lib.template_config import load_template_geometry
import oriented_chamfer_v1 as v1
import edge_conv_test as ect

# ---------------- 参数区（改这里） ----------------
IMAGE_PATH = os.path.join(SRC, "tools/vision/7.png")
V1_REF_JSON = os.path.join(SRC, "tools/vision/oriented_chamfer_v1/output/v1_results.json")
OUTPUT_DIR = os.path.join(THIS, "output")

CROP_MARGIN = 8
DT_CAP = 20.0
ANGLE_STEP = 2.0
SCREEN_TOL_PX = 6
SEARCH_MARGIN = 4
KERNEL_MARGIN = 2

# 朝相机判据：外法线与 d̂ 点积超过此值 → 顶-侧段
SIDE_DOT_THRESH = 0.3
# 档位：(cap, w_white, w_side)。失败块偏移4.7~7.9px，压cap可清零错位段残值
CAP_SCAN = [(20.0, 1.0, 1.0), (8.0, 1.0, 1.0), (6.0, 1.0, 1.0),
            (5.0, 1.0, 1.0), (4.0, 1.0, 1.0), (6.0, 1.0, 0.3)]
PROBLEM_CASES = [1, 4, 7, 9, 14, 17, 20, 24]
# ------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
meta = {}


def seg_normals_outward(binary):
    """轮廓折线段列表 [(p1,p2,外法线), ...]，法线已验证指向形状外部。"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    segs = []
    for cnt in contours:
        pts = cnt[:, 0, :].astype(np.float64)
        for i in range(len(pts) - 1):
            p1, p2 = pts[i], pts[i + 1]
            d = p2 - p1
            L = np.hypot(*d)
            if L < 1e-6:
                continue
            n = np.array([d[1], -d[0]]) / L     # 候选法线
            mid = (p1 + p2) / 2.0
            # 外侧应不在形状内：取 2px 外测试点
            test = np.round(mid + 2.0 * n).astype(int)
            inside = (0 <= test[0] < binary.shape[1] and 0 <= test[1] < binary.shape[0]
                      and binary[test[1], test[0]] > 0)
            if inside:
                n = -n
            segs.append((p1, p2, n))
    return segs


def build_side_aware_kernels(category, angles, d_hat, w_white, w_side):
    """逐段加权线核：段权 = f(外法线·d̂)。返回核张量、尺寸、锚点、权重和。"""
    binary_list = [meta[category][a]["binary"] for a in angles]
    max_h = max(b.shape[0] for b in binary_list) + 2 * KERNEL_MARGIN
    max_w = max(b.shape[1] for b in binary_list) + 2 * KERNEL_MARGIN
    kernels = np.zeros((len(angles), 1, max_h, max_w), dtype=np.float32)
    anchors, wsums = [], []
    for i, (angle, binary) in enumerate(zip(angles, binary_list)):
        canvas = kernels[i, 0]
        total = 0.0
        for p1, p2, n in seg_normals_outward(binary):
            w = w_side if float(n @ d_hat) > SIDE_DOT_THRESH else w_white
            cv2.line(canvas, (int(round(p1[0])) + KERNEL_MARGIN, int(round(p1[1])) + KERNEL_MARGIN),
                     (int(round(p2[0])) + KERNEL_MARGIN, int(round(p2[1])) + KERNEL_MARGIN),
                     float(w), 1)
            total += w * float(np.hypot(*(p2 - p1)))
        ax, ay = meta[category][angle]["anchor"]
        anchors.append((ax + KERNEL_MARGIN, ay + KERNEL_MARGIN))
        wsums.append(max(total, 1.0))
    t = torch.from_numpy(kernels)
    if DEVICE == "cuda":
        t = t.to(torch.float16)
    return t.to(DEVICE), (max_h, max_w), anchors, np.array(wsums, dtype=np.float32)


def conv_argmax(value_roi, kernels, ksize, anchors, angles, wsums, crop_xy):
    max_h, max_w = ksize
    target_h = max(value_roi.shape[0], max_h + 2 * SEARCH_MARGIN)
    target_w = max(value_roi.shape[1], max_w + 2 * SEARCH_MARGIN)
    pad_top = (target_h - value_roi.shape[0]) // 2
    pad_left = (target_w - value_roi.shape[1]) // 2
    padded = np.zeros((target_h, target_w), dtype=np.float32)
    padded[pad_top:pad_top + value_roi.shape[0],
           pad_left:pad_left + value_roi.shape[1]] = value_roi
    t = torch.from_numpy(padded)[None, None]
    if DEVICE == "cuda":
        t = t.to(torch.float16)
    t = t.to(DEVICE)
    ws = torch.from_numpy(wsums).to(t.device)[:, None, None]
    with torch.no_grad():
        fmap = F.conv2d(t, kernels)[0] / ws
    best = None
    for i, angle in enumerate(angles):
        idx = int(torch.argmax(fmap[i]))
        iy, ix = np.unravel_index(idx, fmap[i].shape)
        s = float(fmap[i, iy, ix])
        if best is None or s > best[0]:
            ax, ay = anchors[i]
            best = (s, float(angle),
                    float(ix + ax - pad_left) + crop_xy[0],
                    float(iy + ay - pad_top) + crop_xy[1])
    _, angle, px, py = best
    return {"angle": angle, "px": px, "py": py, "score": best[0]}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img = cv2.imread(IMAGE_PATH)
    img_h, img_w = img.shape[:2]
    center = np.array([img_w / 2.0, img_h / 2.0])
    geo = load_template_geometry("high")
    bpp, cpp = geo["block_px"], geo["connector_px"]

    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    dt_full = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    V_by_cap = {cap: (cap - np.minimum(dt_full, cap)).astype(np.float32)
                for cap, _, _ in CAP_SCAN}

    for cat in set(ROTATION_TOTAL_ANGLE):
        meta[cat] = build_angle_foreground_metadata(cat, bpp, cpp, ANGLE_STEP)

    from ultralytics import YOLO
    yolo = YOLO(v1.DETECTION_MODEL_PATH)
    yolo(img, verbose=False)
    dets = v1.detect_blocks_yolo_v1(img, yolo, v1.ChamferV1Config())
    v1refs = [r for r in json.load(open(V1_REF_JSON))["results"] if r["found"]]

    # ---------- 诊断：cap20基线失败块的偏移方向 ----------
    print("== 诊断：均匀基线(cap20)在8个失败块上的偏移方向 ==")
    V = V_by_cap[20.0]
    for i in PROBLEM_CASES:
        det, vr = dets[i], v1refs[i]
        cat = det["category"]
        x1, y1, x2, y2 = det["box"]
        cx1, cy1 = max(0, int(x1) - CROP_MARGIN), max(0, int(y1) - CROP_MARGIN)
        angles = sorted(meta[cat])
        bw, bh = int(x2 - x1), int(y2 - y1)
        cand = [a for a in angles
                if abs(meta[cat][a]["fg_w"] - bw) <= SCREEN_TOL_PX
                and abs(meta[cat][a]["fg_h"] - bh) <= SCREEN_TOL_PX] or angles
        d_hat = center - np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        d_hat = d_hat / np.linalg.norm(d_hat)
        kernels, ksize, anchors, wsums = build_side_aware_kernels(cat, cand, d_hat, 1.0, 1.0)
        roi = V[cy1:cy1 + (int(y2) + CROP_MARGIN - cy1), cx1:cx1 + (int(x2) + CROP_MARGIN - cx1)]
        o = conv_argmax(roi, kernels, ksize, anchors, cand, wsums, (cx1, cy1))
        off = np.array([o["px"] - vr["px"], o["py"] - vr["py"]])
        proj = off @ d_hat
        ang = np.degrees(np.arccos(np.clip(proj / np.linalg.norm(off), -1, 1))) \
            if np.linalg.norm(off) > 1e-6 else 0.0
        print(f"[{i:02d}] {cat:<9s} |off|={np.linalg.norm(off):5.1f}px  off·d̂={proj:6.1f}"
              f"  与朝中心向夹角={ang:5.1f}°  {'✓朝中心' if proj > 0.5*np.linalg.norm(off) else ''}")

    # ---------- 全量实验：cap × 权重扫描 ----------
    names = [f"cap{int(c)}w{ww:g}_{ws:g}" for c, ww, ws in CAP_SCAN]
    rows_out, per_variant = [], {n: [] for n in names}
    for det, vr in zip(dets, v1refs):
        cat = det["category"]
        x1, y1, x2, y2 = det["box"]
        cx1, cy1 = max(0, int(x1) - CROP_MARGIN), max(0, int(y1) - CROP_MARGIN)
        period = float(ROTATION_TOTAL_ANGLE[cat])
        angles = sorted(meta[cat])
        bw, bh = int(x2 - x1), int(y2 - y1)
        cand = [a for a in angles
                if abs(meta[cat][a]["fg_w"] - bw) <= SCREEN_TOL_PX
                and abs(meta[cat][a]["fg_h"] - bh) <= SCREEN_TOL_PX] or angles
        d_hat = center - np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        d_hat = d_hat / np.linalg.norm(d_hat)

        row = {"category": cat, "v1": (vr["px"], vr["py"], vr["pose_angle_deg"])}
        for (cap, ww, ws), n in zip(CAP_SCAN, names):
            V = V_by_cap[cap]
            roi = V[cy1:cy1 + (int(y2) + CROP_MARGIN - cy1), cx1:cx1 + (int(x2) + CROP_MARGIN - cx1)]
            t0 = time.perf_counter()
            kernels, ksize, anchors, wsums = build_side_aware_kernels(cat, cand, d_hat, ww, ws)
            o = conv_argmax(roi, kernels, ksize, anchors, cand, wsums, (cx1, cy1))
            o["ms"] = (time.perf_counter() - t0) * 1000.0
            o["d_pos"] = float(np.hypot(o["px"] - vr["px"], o["py"] - vr["py"]))
            d = abs(o["angle"] - vr["pose_angle_deg"]) % period
            o["d_angle"] = float(min(d, period - d))
            per_variant[n].append(o)
            row[n] = o
        rows_out.append(row)

    print("\n===== 汇总（vs V1；w白_侧 = (顶-白权重, 顶-侧权重)） =====")
    print(f"{'档位':<10s} {'中位Δpx':>8s} {'<2px':>6s} {'<5px':>6s} {'中位Δ角':>8s} {'<2°':>6s} {'均耗时':>8s}")
    for n in names:
        d = sorted(o["d_pos"] for o in per_variant[n])
        da = sorted(o["d_angle"] for o in per_variant[n])
        k = len(d)
        print(f"{n:<10s} {d[k//2]:8.2f} {sum(x<2 for x in d)/k:5.0%} "
              f"{sum(x<5 for x in d)/k:5.0%} {da[k//2]:8.2f} {sum(x<2 for x in da)/k:5.0%} "
              f"{np.mean([o['ms'] for o in per_variant[n]]):7.1f}ms")

    print("\n8个失败块明细：")
    for i in PROBLEM_CASES:
        r = rows_out[i]
        print(f"[{i:02d}] {r['category']:<9s} " + "  ".join(
            f"{n}:{r[n]['d_pos']:.1f}" for n in names))

    with open(os.path.join(OUTPUT_DIR, "side_aware_results.json"), "w", encoding="utf-8") as f:
        json.dump(rows_out, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 已存 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
