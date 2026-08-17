# -*- coding: utf-8 -*-
"""纯 Canny + 边缘模板全角度匹配：多图批量运行 + 计时 + 出图。

架构（无 seg、无筛角、无先验）：
  全图Canny距离场(cap20) + 全角度居中线模板核 → F.conv2d 归一化 argmax
核按类别预构建缓存；计时项：YOLO / 核预构建 / 逐块匹配。
7.png 若存在 V1 参照 JSON 则附偏差统计。
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
from ultralytics import YOLO

# ---------------- 参数区（改这里） ----------------
IMAGES = [
    os.path.join(SRC, "tools/vision/4.png"),
    os.path.join(SRC, "tools/vision/7.png"),
]
OUTPUT_DIR = os.path.join(THIS, "output")
V1_REF_JSON = os.path.join(SRC, "tools/vision/oriented_chamfer_v1/output/v1_results.json")

ANGLE_STEP = 2.0
CROP_MARGIN = 8
DT_CAP = 20.0
MARGIN = 20
KERNEL_MARGIN = 2
DETECTION_CONF = 0.45
# -------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
meta = {}
kernel_cache = {}


def build_line_kernels_centered(category, angles):
    binary_list = [meta[category][a]["binary"] for a in angles]
    max_h = max(b.shape[0] for b in binary_list) + 2 * KERNEL_MARGIN
    max_w = max(b.shape[1] for b in binary_list) + 2 * KERNEL_MARGIN
    kernels = np.zeros((len(angles), 1, max_h, max_w), dtype=np.float32)
    anchors = []
    for i, (angle, binary) in enumerate(zip(angles, binary_list)):
        h, w = binary.shape
        ox, oy = (max_w - w) // 2, (max_h - h) // 2
        canvas = np.zeros((max_h, max_w), dtype=np.uint8)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, cnts, -1, 255, 1,
                         offset=(ox + KERNEL_MARGIN, oy + KERNEL_MARGIN))
        kernels[i, 0] = (canvas > 0).astype(np.float32)
        ax, ay = meta[category][angle]["anchor"]
        anchors.append((ox + KERNEL_MARGIN + ax, oy + KERNEL_MARGIN + ay))
    t = torch.from_numpy(kernels)
    if DEVICE == "cuda":
        t = t.to(torch.float16)
    return t.to(DEVICE), (max_h, max_w), anchors


def match_block(V, det, cat):
    x1, y1, x2, y2 = det["box"]
    cx1, cy1 = max(0, int(x1) - CROP_MARGIN), max(0, int(y1) - CROP_MARGIN)
    cx2, cy2 = min(V.shape[1], int(x2) + CROP_MARGIN), min(V.shape[0], int(y2) + CROP_MARGIN)
    roi = V[cy1:cy2, cx1:cx2]
    if cat not in kernel_cache:
        kernel_cache[cat] = build_line_kernels_centered(cat, sorted(meta[cat]))
    line_k, ksize, anchors = kernel_cache[cat]
    angles = sorted(meta[cat])
    th = max(roi.shape[0], ksize[0] + 2 * MARGIN)
    tw = max(roi.shape[1], ksize[1] + 2 * MARGIN)
    pt_, pl = (th - roi.shape[0]) // 2, (tw - roi.shape[1]) // 2
    padded = np.zeros((th, tw), dtype=np.float32)
    padded[pt_:pt_ + roi.shape[0], pl:pl + roi.shape[1]] = roi
    t = torch.from_numpy(padded)[None, None]
    if DEVICE == "cuda":
        t = t.to(torch.float16)
    t = t.to(DEVICE)
    lc = line_k.sum(dim=(1, 2, 3)).clamp(min=1.0)
    with torch.no_grad():
        fmap = F.conv2d(t, line_k)[0] / lc[:, None, None]
    bt = fmap.amax(dim=(1, 2))
    j = int(torch.argmax(bt))
    idx = int(torch.argmax(fmap[j]))
    iy, ix = np.unravel_index(idx, fmap[j].shape)
    ax, ay = anchors[j]
    return {"angle": float(angles[j]),
            "px": float(ix + ax - pl) + cx1, "py": float(iy + ay - pt_) + cy1,
            "score": float(fmap[j, iy, ix]),
            "roi": (cx1, cy1, cx2, cy2), "n_angles": len(angles)}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    geo = load_template_geometry("high")
    bpp, cpp = geo["block_px"], geo["connector_px"]
    models = {c: v1.build_template_contour_model(c, bpp, cpp, 1.5)
              for c in set(ROTATION_TOTAL_ANGLE)}

    t0 = time.perf_counter()
    for cat in set(ROTATION_TOTAL_ANGLE):
        meta[cat] = build_angle_foreground_metadata(cat, bpp, cpp, ANGLE_STEP)
    meta_ms = (time.perf_counter() - t0) * 1000.0

    yolo = YOLO(v1.DETECTION_MODEL_PATH)
    yolo(cv2.imread(IMAGES[0])[:64, :64], verbose=False)  # warmup

    report = {"angle_metadata_ms": meta_ms, "images": []}
    for img_path in IMAGES:
        img = cv2.imread(img_path)
        if img is None:
            print(f"!! 读不到 {img_path}")
            continue
        stem = os.path.splitext(os.path.basename(img_path))[0]
        gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        edges = cv2.Canny(gray, 50, 150)
        V = (DT_CAP - np.minimum(
            cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3), DT_CAP)).astype(np.float32)

        t_y0 = time.perf_counter()
        dets = v1.detect_blocks_yolo_v1(img, yolo, v1.ChamferV1Config())
        yolo_ms = (time.perf_counter() - t_y0) * 1000.0

        # V1 参照（若有）
        v1refs = None
        if os.path.exists(V1_REF_JSON):
            with open(V1_REF_JSON, encoding="utf-8") as f:
                ref = json.load(f)
            if os.path.basename(ref.get("image", "")) == os.path.basename(img_path):
                v1refs = [r for r in ref["results"] if r["found"]]

        blocks, match_ms_list = [], []
        t_b0 = time.perf_counter()
        for i, det in enumerate(dets):
            cat = det["category"]
            tm = time.perf_counter()
            o = match_block(V, det, cat)
            o["ms"] = (time.perf_counter() - tm) * 1000.0
            match_ms_list.append(o["ms"])
            d_pos = d_angle = None
            if v1refs is not None and i < len(v1refs):
                vr = v1refs[i]
                if vr["category"] == cat:
                    d_pos = float(np.hypot(o["px"] - vr["px"], o["py"] - vr["py"]))
                    period = float(ROTATION_TOTAL_ANGLE[cat])
                    d = abs(o["angle"] - vr["pose_angle_deg"]) % period
                    d_angle = float(min(d, period - d))
            o.update({"index": i, "category": cat, "d_pos": d_pos, "d_angle": d_angle})
            blocks.append(o)

            # 出图：Canny 底图 + 绿轮廓 + 红中心
            cx1, cy1, cx2, cy2 = o["roi"]
            base = cv2.cvtColor(edges[cy1:cy2, cx1:cx2], cv2.COLOR_GRAY2BGR)
            rx, ry, _ = models[cat].rotated(o["angle"])
            pts = np.stack((rx + o["px"] - cx1, ry + o["py"] - cy1), axis=1).astype(np.int32)
            cv2.polylines(base, [pts], True, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(base, (int(o["px"]) - cx1, int(o["py"]) - cy1), 3, (0, 0, 255), 1)
            note = (f"a={o['angle']:.0f} d={d_pos:.1f}px da={d_angle:.1f}"
                    if d_pos is not None else f"a={o['angle']:.0f} score={o['score']:.2f}")
            cv2.putText(base, note, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{stem}_{i:02d}_{cat}.png"), base)
        total_ms = (time.perf_counter() - t_b0) * 1000.0

        line = (f"{stem}.png: {len(blocks)}块 | YOLO {yolo_ms:.0f}ms | "
                f"匹配总 {total_ms:.0f}ms (均{np.mean(match_ms_list):.1f}ms/块, "
                f"最大{max(match_ms_list):.1f}ms) | 含出图")
        if v1refs is not None:
            dp = [b["d_pos"] for b in blocks if b["d_pos"] is not None]
            da = [b["d_angle"] for b in blocks if b["d_angle"] is not None]
            n = len(dp)
            line += (f" | vs V1: 中位{sorted(dp)[n//2]:.2f}px "
                     f"<2px {sum(x<2 for x in dp)/n:.0%} <5px {sum(x<5 for x in dp)/n:.0%} "
                     f"角<2° {sum(x<2 for x in da)/len(da):.0%}")
        print(line)
        report["images"].append({"image": img_path, "yolo_ms": yolo_ms,
                                 "match_total_ms": total_ms,
                                 "match_mean_ms": float(np.mean(match_ms_list)),
                                 "blocks": blocks})

    with open(os.path.join(OUTPUT_DIR, "pure_canny_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"角度metadata预构建: {meta_ms:.0f}ms（一次性，7类）")
    print(f"图 + JSON 已存 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
