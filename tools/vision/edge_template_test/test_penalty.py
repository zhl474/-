# -*- coding: utf-8 -*-
"""变体D实验：conv chamfer（C）+ 逐点归一化 + 中心先验罚分，看能否救回8个失败块。

不再走 match_template 的裸 argmax，直接 F.conv2d 后自己加罚分取最优，
conv 本体与 match_template 完全相同，只是选优后处理不同。
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

# ---------------- 参数区 ----------------
IMAGE_PATH = os.path.join(SRC, "tools/vision/7.png")
V1_REF_JSON = os.path.join(SRC, "tools/vision/oriented_chamfer_v1/output/v1_results.json")
CROP_MARGIN = 8
CANNY_LOW, CANNY_HIGH = 50.0, 150.0
ANGLE_STEP = 2.0
DT_CAP = 20.0
KERNEL_MARGIN = 2
SEARCH_MARGIN = 4          # 输入补边（模板平移余量）
SCREEN_TOL_PX = 6
CENTER_LAMBDA = 0.005      # 与 V1 center_prior_weight 一致
VARIANTS = ("D_c", "D_a")  # D_c: chamfer+罚分;  D_a: 重叠+罚分
# ---------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_line_kernels(metadata, angles, device):
    binary_list = [metadata[a]["binary"] for a in angles]
    max_h = max(b.shape[0] for b in binary_list) + 2 * KERNEL_MARGIN
    max_w = max(b.shape[1] for b in binary_list) + 2 * KERNEL_MARGIN
    kernels = np.zeros((len(angles), 1, max_h, max_w), dtype=np.float32)
    counts = np.zeros(len(angles), dtype=np.float32)
    anchors = []
    for i, (angle, binary) in enumerate(zip(angles, binary_list)):
        canvas = np.zeros((max_h, max_w), dtype=np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, contours, -1, 255, 1,
                         offset=(KERNEL_MARGIN, KERNEL_MARGIN))
        kernels[i, 0] = (canvas > 0).astype(np.float32)
        counts[i] = float(kernels[i, 0].sum())
        ax, ay = metadata[angle]["anchor"]
        anchors.append((ax + KERNEL_MARGIN, ay + KERNEL_MARGIN))
    t = torch.from_numpy(kernels)
    if device == "cuda":
        t = t.to(torch.float16)
    return t.to(device), (max_h, max_w), anchors, counts


def run_variant_D(mode, image_data, metadata, angles, crop_xy, box_wh, box_center):
    used = angles
    bw, bh = box_wh
    cand = [a for a in angles
            if abs(metadata[a]["fg_w"] - bw) <= SCREEN_TOL_PX
            and abs(metadata[a]["fg_h"] - bh) <= SCREEN_TOL_PX]
    if len(cand) >= 3:
        used = cand

    t0 = time.perf_counter()
    kernels, (max_h, max_w), anchors, counts = build_line_kernels(metadata, used, DEVICE)
    target_h = max(image_data.shape[0], max_h + 2 * SEARCH_MARGIN)
    target_w = max(image_data.shape[1], max_w + 2 * SEARCH_MARGIN)
    pad_top = (target_h - image_data.shape[0]) // 2
    pad_left = (target_w - image_data.shape[1]) // 2
    padded = np.zeros((target_h, target_w), dtype=np.float32)
    padded[pad_top:pad_top + image_data.shape[0],
           pad_left:pad_left + image_data.shape[1]] = image_data

    img_t = torch.from_numpy(padded)[None, None]
    if DEVICE == "cuda":
        img_t = img_t.to(torch.float16)
    img_t = img_t.to(DEVICE)
    with torch.no_grad():
        fmap = F.conv2d(img_t, kernels)[0]          # (N, H', W') 无padding，位置=核左上角
    out_h, out_w = fmap.shape[-2:]
    # 归一化：每个角度除以自身线像素数 → 均值语义（chamfer: 平均距离；overlap: 覆盖率）
    fmap = fmap / torch.from_numpy(counts).to(fmap.device)[:, None, None]
    if mode == "D_c":
        fmap = (DT_CAP / 1.0) - fmap                # 回到“距离越小越好”的负分
        sign = -1.0
    else:
        sign = 1.0

    # 中心先验：对每个输出位置算锚点中心相对 bbox 中心的偏移
    ys = torch.arange(out_h, device=fmap.device, dtype=fmap.dtype)[:, None]
    xs = torch.arange(out_w, device=fmap.device, dtype=fmap.dtype)[None, :]
    best = None
    for i, angle in enumerate(used):
        ax, ay = anchors[i]
        cx = xs + pad_left - (crop_xy[0] - crop_xy[0]) - 0  # 核左上角在 ROI 坐标
        cy = ys
        off_x = (xs + ax - pad_left) - box_center[0]
        off_y = (ys + ay - pad_top) - box_center[1]
        score = sign * fmap[i] - CENTER_LAMBDA * (off_x ** 2 + off_y ** 2)
        idx = int(torch.argmax(score))
        iy, ix = np.unravel_index(idx, score.shape)
        cand_score = float(score[iy, ix])
        if best is None or cand_score > best[0]:
            # 核左上角(ix,iy)(padded坐标) + 锚点 - 补边 = ROI坐标，再 + crop 原点
            best = (cand_score, float(angle),
                    float(ix + ax - pad_left) + crop_xy[0],
                    float(iy + ay - pad_top) + crop_xy[1])
    elapsed = (time.perf_counter() - t0) * 1000.0
    _, angle, px, py = best
    return {"angle": angle, "px": px, "py": py, "n_angles": len(used), "ms": elapsed}


def angle_diff(angle, ref, period):
    d = (angle - ref) % period
    return min(d, period - d)


def main():
    img = cv2.imread(IMAGE_PATH)
    geo = load_template_geometry("high")
    bpp, cpp = geo["block_px"], geo["connector_px"]

    from ultralytics import YOLO
    yolo = YOLO(v1.DETECTION_MODEL_PATH)
    yolo(img, verbose=False)
    dets = v1.detect_blocks_yolo_v1(img, yolo, v1.ChamferV1Config())
    v1_refs = [r for r in json.load(open(V1_REF_JSON))["results"] if r["found"]]

    meta_cache = {}
    rows = []
    for det in dets:
        cat = det["category"]
        x1, y1, x2, y2 = det["box"]
        cx1, cy1 = max(0, int(x1) - CROP_MARGIN), max(0, int(y1) - CROP_MARGIN)
        cx2 = min(img.shape[1], int(x2) + CROP_MARGIN)
        cy2 = min(img.shape[0], int(y2) + CROP_MARGIN)
        roi = img[cy1:cy2, cx1:cx2]
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
        edges_f = (edges > 0).astype(np.float32)
        dt_img = (DT_CAP - np.minimum(
            cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3), DT_CAP)).astype(np.float32)

        if cat not in meta_cache:
            meta_cache[cat] = build_angle_foreground_metadata(cat, bpp, cpp, ANGLE_STEP)
        metadata = meta_cache[cat]
        angles = sorted(metadata.keys())
        period = float(ROTATION_TOTAL_ANGLE[cat])

        ref = None
        for r in v1_refs:
            if r["category"] != cat:
                continue
            bx = (r["detection_box"][0] + r["detection_box"][2]) / 2
            by = (r["detection_box"][1] + r["detection_box"][3]) / 2
            if abs(bx - (x1 + x2) / 2) < 5 and abs(by - (y1 + y2) / 2) < 5:
                ref = r
                break
        if ref is None:
            continue

        bc = (((x1 + x2) / 2 - cx1), ((y1 + y2) / 2 - cy1))
        row = {"category": cat}
        for name in VARIANTS:
            data = dt_img if name == "D_c" else edges_f
            out = run_variant_D(name, data, metadata, angles, (cx1, cy1),
                                (int(x2 - x1), int(y2 - y1)), bc)
            out["d_pos"] = float(np.hypot(out["px"] - ref["px"], out["py"] - ref["py"]))
            out["d_angle"] = float(angle_diff(out["angle"], ref["pose_angle_deg"], period))
            row[name] = out
        rows.append(row)
        print(f"{cat:<10s} " + "  ".join(
            f"{n}: d={row[n]['d_pos']:.1f}px da={row[n]['d_angle']:.1f}deg {row[n]['ms']:.0f}ms"
            for n in VARIANTS))

    print("\n===== D 变体汇总（vs V1） =====")
    for name in VARIANTS:
        d = sorted(r[name]["d_pos"] for r in rows)
        da = sorted(r[name]["d_angle"] for r in rows)
        n = len(rows)
        print(f"{name}: 中位Δpx={d[n//2]:.2f} 均值={np.mean(d):.2f} "
              f"<2px:{sum(x<2 for x in d)/n:.0%} <5px:{sum(x<5 for x in d)/n:.0%} "
              f"中位Δ角={da[n//2]:.2f} <2°:{sum(x<2 for x in da)/n:.0%} "
              f"均耗时={np.mean([r[name]['ms'] for r in rows]):.1f}ms")

    with open(os.path.join(THIS, "output", "penalty_results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
