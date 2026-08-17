# -*- coding: utf-8 -*-
"""修正搜索网格后的全角度实验。

bug：build_edge_kernels 把各角度紧模板钉在公共画布左上角，锚点随角度变化；
画布大时卷积输出网格短，各角度可达中心区间 = [anchor_j, anchor_j+out-1]，
不覆盖方块中心 → 全角度实验实际只搜了"锚点碰巧罩住方块"的角度。

修正：加大平移余量 MARGIN，使所有角度的可达区间交集覆盖方块中心±15px，
并显式校验 V1 中心落在每个角度的网格内（reachability check）。
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
import seg_canny_test as sct
from ultralytics import YOLO

# ---------------- 参数区 ----------------
IMAGE_PATH = os.path.join(SRC, "tools/vision/7.png")
V1_REF_JSON = os.path.join(SRC, "tools/vision/oriented_chamfer_v1/output/v1_results.json")
SEG_MODEL_PATH = os.path.join(SRC, "competition", "model", "best_seg.pt")
OUTPUT_DIR = os.path.join(THIS, "output")
MARGIN = 20              # 核已居中放置，锚点散布小，20px 余量足够
W_CANNY, W_SEG = 3.0, 1.0
USE_SEG = False          # 修正网格后裸Canny即可工作；True可再叠加seg
CENTERED_KERNELS = True  # 各角度紧模板居中放画布，锚点散布→最小
# ----------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_line_kernels_centered(category, angles, thickness, device):
    """居中版线核：紧模板放公共画布中心，锚点=居中偏移+metadata锚点。"""
    binary_list = [sct.meta[category][a]["binary"] for a in angles]
    max_h = max(b.shape[0] for b in binary_list) + 2 * ect.KERNEL_MARGIN
    max_w = max(b.shape[1] for b in binary_list) + 2 * ect.KERNEL_MARGIN
    kernels = np.zeros((len(angles), 1, max_h, max_w), dtype=np.float32)
    anchors = []
    for i, (angle, binary) in enumerate(zip(angles, binary_list)):
        h, w = binary.shape
        ox = (max_w - w) // 2
        oy = (max_h - h) // 2
        canvas = np.zeros((max_h, max_w), dtype=np.uint8)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, cnts, -1, 255, thickness, offset=(ox + ect.KERNEL_MARGIN,
                                                                  oy + ect.KERNEL_MARGIN))
        kernels[i, 0] = (canvas > 0).astype(np.float32)
        ax, ay = sct.meta[category][angle]["anchor"]
        anchors.append((ox + ect.KERNEL_MARGIN + ax, oy + ect.KERNEL_MARGIN + ay))
    t = torch.from_numpy(kernels)
    if device == "cuda":
        t = t.to(torch.float16)
    return t.to(device), (max_h, max_w), anchors


def build_fill_kernels_centered(category, angles, device):
    """居中版实心核，画布与锚点与居中线核一致。"""
    binary_list = [sct.meta[category][a]["binary"] for a in angles]
    max_h = max(b.shape[0] for b in binary_list) + 2 * ect.KERNEL_MARGIN
    max_w = max(b.shape[1] for b in binary_list) + 2 * ect.KERNEL_MARGIN
    kernels = np.zeros((len(angles), 1, max_h, max_w), dtype=np.float32)
    anchors, counts = [], []
    for i, (angle, binary) in enumerate(zip(angles, binary_list)):
        h, w = binary.shape
        ox = (max_w - w) // 2
        oy = (max_h - h) // 2
        kernels[i, 0, oy + ect.KERNEL_MARGIN:oy + ect.KERNEL_MARGIN + h,
                ox + ect.KERNEL_MARGIN:ox + ect.KERNEL_MARGIN + w] = binary.astype(np.float32)
        ax, ay = sct.meta[category][angle]["anchor"]
        anchors.append((ox + ect.KERNEL_MARGIN + ax, oy + ect.KERNEL_MARGIN + ay))
        counts.append(float(binary.sum()))
    t = torch.from_numpy(kernels)
    if device == "cuda":
        t = t.to(torch.float16)
    return t.to(device), (max_h, max_w), anchors, np.array(counts, dtype=np.float32)


def main():
    img = cv2.imread(IMAGE_PATH)
    geo = load_template_geometry("high")
    bpp, cpp = geo["block_px"], geo["connector_px"]
    for cat in set(ROTATION_TOTAL_ANGLE):
        sct.meta[cat] = build_angle_foreground_metadata(cat, bpp, cpp, sct.ANGLE_STEP)
    meta = sct.meta
    models = {c: v1.build_template_contour_model(c, bpp, cpp, 1.5)
              for c in set(ROTATION_TOTAL_ANGLE)}
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges_full = cv2.Canny(gray, 50, 150)
    V_plain = (sct.DT_CAP - np.minimum(
        cv2.distanceTransform(255 - edges_full, cv2.DIST_L2, 3), sct.DT_CAP)).astype(np.float32)

    yolo = YOLO(v1.DETECTION_MODEL_PATH)
    yolo(img, verbose=False)
    seg_model = YOLO(SEG_MODEL_PATH, task="segment")
    seg_model(img[:64, :64], verbose=False)
    dets = v1.detect_blocks_yolo_v1(img, yolo, v1.ChamferV1Config())
    v1refs = [r for r in json.load(open(V1_REF_JSON))["results"] if r["found"]]

    rows, seg_ms, match_ms, reach_fail = [], [], [], 0
    for i, (det, vr) in enumerate(zip(dets, v1refs)):
        cat = det["category"]
        x1, y1, x2, y2 = det["box"]
        cx1, cy1 = max(0, int(x1) - sct.CROP_MARGIN), max(0, int(y1) - sct.CROP_MARGIN)
        cx2 = min(img.shape[1], int(x2) + sct.CROP_MARGIN)
        cy2 = min(img.shape[0], int(y2) + sct.CROP_MARGIN)
        crop = img[cy1:cy2, cx1:cx2]

        if USE_SEG:
            t0 = time.perf_counter()
            seg = sct.get_seg_mask(seg_model, crop)
            seg_ms.append((time.perf_counter() - t0) * 1000.0)
            seg_f = (seg > 0).astype(np.float32)
            keep = (edges_full[cy1:cy2, cx1:cx2] > 0) & (cv2.dilate(
                seg, np.ones((2 * sct.SEG_FILTER_DILATE_PX + 1,) * 2, np.uint8)) > 0)
        else:
            seg_f = np.zeros(crop.shape[:2], dtype=np.float32)
            keep = edges_full[cy1:cy2, cx1:cx2] > 0
        A_e = (sct.DT_CAP - np.minimum(cv2.distanceTransform(
            255 - keep.astype(np.uint8) * 255, cv2.DIST_L2, 3), sct.DT_CAP)).astype(np.float32)

        angles = sorted(meta[cat])
        t0 = time.perf_counter()
        if CENTERED_KERNELS:
            line_k, ksize, anchors = build_line_kernels_centered(cat, angles, 1, DEVICE)
            fill_k, _, _, fill_counts = build_fill_kernels_centered(cat, angles, DEVICE)
        else:
            line_k, ksize, anchors = ect.build_edge_kernels(meta[cat], angles, 1, DEVICE)
            fill_k, _, _, fill_counts = sct.build_filled_kernels(cat, angles, DEVICE)
        th = max(A_e.shape[0], ksize[0] + 2 * MARGIN)
        tw = max(A_e.shape[1], ksize[1] + 2 * MARGIN)
        pt_, pl = (th - A_e.shape[0]) // 2, (tw - A_e.shape[1]) // 2
        E = np.zeros((th, tw), dtype=np.float32)
        S = np.zeros((th, tw), dtype=np.float32)
        E[pt_:pt_ + A_e.shape[0], pl:pl + A_e.shape[1]] = A_e
        S[pt_:pt_ + A_e.shape[0], pl:pl + A_e.shape[1]] = seg_f
        lc = line_k.sum(dim=(1, 2, 3)).clamp(min=1.0)
        fc = torch.from_numpy(fill_counts).to(DEVICE)[:, None, None].clamp(min=1.0)
        with torch.no_grad():
            f_line = F.conv2d(sct.to_t(E), line_k)[0] / lc[:, None, None] / sct.DT_CAP
            f_fill = F.conv2d(sct.to_t(S), fill_k)[0] / fc
        total = W_CANNY * f_line + W_SEG * f_fill

        # 可达性校验：V1中心必须落在每个角度网格内
        period = float(ROTATION_TOTAL_ANGLE[cat])
        j_true = min(range(len(angles)),
                     key=lambda j: min(abs(angles[j] - vr["pose_angle_deg"]) % period,
                                       period - abs(angles[j] - vr["pose_angle_deg"]) % period))
        ax, ay = anchors[j_true]
        ix_v1 = vr["px"] - cx1 + pl - ax
        iy_v1 = vr["py"] - cy1 + pt_ - ay
        ok = 0 <= ix_v1 < total.shape[2] and 0 <= iy_v1 < total.shape[1]
        if not ok:
            reach_fail += 1

        bt = total.amax(dim=(1, 2))
        j_best = int(torch.argmax(bt))
        idx = int(torch.argmax(total[j_best]))
        iy, ix = np.unravel_index(idx, total[j_best].shape)
        ax, ay = anchors[j_best]
        o = {"angle": angles[j_best],
             "px": float(ix + ax - pl) + cx1, "py": float(iy + ay - pt_) + cy1,
             "line": float(f_line[j_best, iy, ix]), "fill": float(f_fill[j_best, iy, ix])}
        o["d_pos"] = float(np.hypot(o["px"] - vr["px"], o["py"] - vr["py"]))
        d = abs(o["angle"] - vr["pose_angle_deg"]) % period
        o["d_angle"] = float(min(d, period - d))
        match_ms.append((time.perf_counter() - t0) * 1000.0)
        rows.append({"index": i, "category": cat, "v1": (vr["px"], vr["py"], vr["pose_angle_deg"]),
                     "best": o, "v1_reachable": ok})

    d = sorted(r["best"]["d_pos"] for r in rows)
    da = sorted(r["best"]["d_angle"] for r in rows)
    n = len(d)
    print(f"修正后全角度({'seg+canny组合' if USE_SEG else '裸Canny'}): 中位Δpx={d[n//2]:.2f} "
          f"<2px:{sum(x<2 for x in d)/n:.0%} <5px:{sum(x<5 for x in d)/n:.0%} "
          f"中位Δ角={da[n//2]:.2f}° <2°:{sum(x<2 for x in da)/n:.0%}")
    print(f"V1位姿不可达块数: {reach_fail}；匹配耗时均值 {np.mean(match_ms):.0f}ms"
          + (f"，seg {np.mean(seg_ms):.1f}ms" if USE_SEG else ""))
    print("\n>5px 的块：")
    for r in rows:
        if r["best"]["d_pos"] > 5:
            print(f"  [{r['index']:02d}] {r['category']:<9s} d={r['best']['d_pos']:5.1f} "
                  f"da={r['best']['d_angle']:6.1f}° line={r['best']['line']:.2f} "
                  f"fill={r['best']['fill']:.2f} V1可达={r['v1_reachable']}")

    with open(os.path.join(OUTPUT_DIR, "fullangle_fixed_results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
