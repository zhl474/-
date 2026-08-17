# -*- coding: utf-8 -*-
"""顶-白邻接加权扫描：Canny+边缘模板匹配主框架不变，只按白板邻接给边缘分组加权。

假说（用户）：均匀加权时，模板像素与边缘像素无法完美对准，打分会倾向
"折中骑线"拿更大重合；给清晰的顶-白外边加大权重、模糊的顶-侧边小权重，
迫使模板严格贴住硬边，消灭折中位姿。

做法：整图 Canny -> 白板 mask 膨胀 reach px -> 分成 e_white / e_rest 两组
-> 各自距离场 -> V = max(w_white * A(dt_w), w_rest * A(dt_r))，A(x)=cap-min(x,cap)
-> 与变体C完全相同的 conv 匹配（同 ROI / 筛角 / 归一化）。
"""
import json
import os
import sys
import time

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
SRC = "/home/zhl/SingleArmTetris/SingleArmTetris/src"
CONF_DIR = os.path.join(SRC, "tools/vision/edge_confidence_test")
THIS = os.path.dirname(os.path.abspath(__file__))
for p in (SRC, os.path.join(SRC, "image_process"),
          os.path.join(SRC, "tools/vision/oriented_chamfer_v1"), CONF_DIR, THIS):
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
import edge_pipeline_test as ept   # 只用它的 build_white_mask 参数

# ---------------- 参数区（改这里） ----------------
IMAGE_PATH = os.path.join(SRC, "tools/vision/7.png")
V1_REF_JSON = os.path.join(SRC, "tools/vision/oriented_chamfer_v1/output/v1_results.json")
OUTPUT_DIR = os.path.join(THIS, "output")

CROP_MARGIN = 8
DT_CAP = 20.0
ANGLE_STEP = 2.0
SCREEN_TOL_PX = 6
SEARCH_MARGIN = 4
WHITE_REACH_PX = 3        # 白板 mask 膨胀半径：边缘 3px 内有白板算“顶-白邻接”

# (顶-白权重, 顶-侧权重) 扫描档位；含 (1,1) 等价 C 基线做同轮对照
WEIGHT_PAIRS = [(1.0, 1.0), (3.0, 1.0), (3.0, 0.5), (3.0, 0.3), (5.0, 0.3), (1.0, 0.3)]
PROBLEM_CASES = [1, 4, 7, 9, 14, 17, 20, 24]
# ------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def conv_argmax(value_roi, kernels, ksize, anchors, angles, crop_xy):
    """与变体C同构：归一化 conv 选优（高价值=好），返回全图中心+角度。"""
    max_h, max_w = ksize
    target_h = max(value_roi.shape[0], max_h + 2 * SEARCH_MARGIN)
    target_w = max(value_roi.shape[1], max_w + 2 * SEARCH_MARGIN)
    pad_top = (target_h - value_roi.shape[0]) // 2
    pad_left = (target_w - value_roi.shape[1]) // 2
    padded = np.zeros((target_h, target_w), dtype=np.float32)
    padded[pad_top:pad_top + value_roi.shape[0],
           pad_left:pad_left + value_roi.shape[1]] = value_roi
    t = torch.from_numpy(padded)[None, None]
    counts = kernels.sum(dim=(1, 2, 3)).clamp(min=1.0)
    if DEVICE == "cuda":
        t = t.to(torch.float16)
    t = t.to(DEVICE)
    with torch.no_grad():
        fmap = F.conv2d(t, kernels)[0] / counts[:, None, None]
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


def run_match(V_full, meta, cat, det, crop_xy):
    x1, y1, x2, y2 = det["box"]
    bw, bh = int(x2 - x1), int(y2 - y1)
    angles = sorted(meta[cat])
    cand = [a for a in angles
            if abs(meta[cat][a]["fg_w"] - bw) <= SCREEN_TOL_PX
            and abs(meta[cat][a]["fg_h"] - bh) <= SCREEN_TOL_PX]
    if len(cand) < 3:
        cand = angles
    cx1, cy1 = crop_xy
    roi = V_full[cy1:cy1 + (int(y2) + CROP_MARGIN - cy1),
                 cx1:cx1 + (int(x2) + CROP_MARGIN - cx1)]
    kernels, ksize, anchors = ect.build_edge_kernels(meta[cat], cand, 1, DEVICE)
    return conv_argmax(roi, kernels, ksize, anchors, cand, crop_xy)


def angle_diff(a, b, period):
    d = abs(a - b) % period
    return min(d, period - d)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img = cv2.imread(IMAGE_PATH)
    geo = load_template_geometry("high")
    bpp, cpp = geo["block_px"], geo["connector_px"]

    # 两组边缘的距离场只算一次
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges_f = cv2.Canny(gray, 50, 150) > 0
    white_mask = ept.build_white_mask(img)
    white_dil = cv2.dilate(white_mask, np.ones((2 * WHITE_REACH_PX + 1,) * 2, np.uint8)) > 0
    e_white = (edges_f & white_dil).astype(np.uint8) * 255
    e_rest = (edges_f & ~white_dil).astype(np.uint8) * 255
    dt_w = cv2.distanceTransform(255 - e_white, cv2.DIST_L2, 3)
    dt_r = cv2.distanceTransform(255 - e_rest, cv2.DIST_L2, 3)
    A_w = (DT_CAP - np.minimum(dt_w, DT_CAP)).astype(np.float32)
    A_r = (DT_CAP - np.minimum(dt_r, DT_CAP)).astype(np.float32)
    n_w = int((edges_f & white_dil).sum())
    print(f"Canny {int(edges_f.sum())} px：顶-白邻接 {n_w}（{n_w/edges_f.sum():.0%}），"
          f"其余 {int(edges_f.sum())-n_w}")

    meta = {}
    for cat in set(ROTATION_TOTAL_ANGLE):
        meta[cat] = build_angle_foreground_metadata(cat, bpp, cpp, ANGLE_STEP)

    from ultralytics import YOLO
    yolo = YOLO(v1.DETECTION_MODEL_PATH)
    yolo(img, verbose=False)
    dets = v1.detect_blocks_yolo_v1(img, yolo, v1.ChamferV1Config())
    v1refs = [r for r in json.load(open(V1_REF_JSON))["results"] if r["found"]]

    names = [f"w{ww:g}_{wr:g}" for ww, wr in WEIGHT_PAIRS]
    rows_out = []
    per_variant = {n: [] for n in names}
    for det, vr in zip(dets, v1refs):
        cat = det["category"]
        x1, y1, x2, y2 = det["box"]
        cx1 = max(0, int(x1) - CROP_MARGIN)
        cy1 = max(0, int(y1) - CROP_MARGIN)
        period = float(ROTATION_TOTAL_ANGLE[cat])

        row = {"category": cat, "v1": (vr["px"], vr["py"], vr["pose_angle_deg"])}
        for (ww, wr), n in zip(WEIGHT_PAIRS, names):
            t0 = time.perf_counter()
            V = np.maximum(ww * A_w, wr * A_r)
            o = run_match(V, meta, cat, det, (cx1, cy1))
            o["ms"] = (time.perf_counter() - t0) * 1000.0
            o["d_pos"] = float(np.hypot(o["px"] - vr["px"], o["py"] - vr["py"]))
            o["d_angle"] = float(angle_diff(o["angle"], vr["pose_angle_deg"], period))
            per_variant[n].append(o)
            row[n] = o
        rows_out.append(row)

    print("\n===== 汇总（vs V1 参照；w白_侧 = (顶-白权重, 顶-侧权重)） =====")
    print(f"{'档位':<10s} {'中位Δpx':>8s} {'<2px':>6s} {'<5px':>6s} {'中位Δ角':>8s} {'<2°':>6s} {'均耗时':>8s}")
    for n in names:
        d = sorted(o["d_pos"] for o in per_variant[n])
        da = sorted(o["d_angle"] for o in per_variant[n])
        m = np.mean([o["ms"] for o in per_variant[n]])
        k = len(d)
        print(f"{n:<10s} {d[k//2]:8.2f} {sum(x<2 for x in d)/k:5.0%} "
              f"{sum(x<5 for x in d)/k:5.0%} {da[k//2]:8.2f} {sum(x<2 for x in da)/k:5.0%} {m:7.1f}ms")

    print("\n上一轮8个失败块（L/z）明细：")
    for i in PROBLEM_CASES:
        r = rows_out[i]
        print(f"[{i:02d}] {r['category']:<9s} " + "  ".join(
            f"{n}:{r[n]['d_pos']:.1f}" for n in names))

    with open(os.path.join(OUTPUT_DIR, "topwhite_scan_results.json"), "w", encoding="utf-8") as f:
        json.dump(rows_out, f, ensure_ascii=False, indent=2)

    # 可视化：失败块上 C 基线(w1_1) vs 最优档，各一张不遮挡
    best_name = min(names, key=lambda n: np.mean([o["d_pos"] for o in per_variant[n]]))
    print(f"\n平均Δpx最优档: {best_name}")
    models = {}

    def draw_pose(vis, cat, px, py, ang, color):
        if cat not in models:
            models[cat] = v1.build_template_contour_model(cat, bpp, cpp, 1.5)
        rx, ry, _ = models[cat].rotated(ang)
        pts = np.stack((rx + px, ry + py), axis=1).astype(np.int32)
        cv2.polylines(vis, [pts], True, color, 1, cv2.LINE_AA)

    for i in PROBLEM_CASES:
        r = rows_out[i]
        cat = r["category"]
        vr = v1refs[i]
        x1, y1, x2, y2 = (int(v) for v in vr["detection_box"])
        cx1, cy1 = max(0, x1 - CROP_MARGIN), max(0, y1 - CROP_MARGIN)
        cx2 = min(img.shape[1], x2 + CROP_MARGIN)
        cy2 = min(img.shape[0], y2 + CROP_MARGIN)
        crop = img[cy1:cy2, cx1:cx2]
        base = cv2.cvtColor(cv2.Canny(cv2.GaussianBlur(
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (3, 3), 0), 50, 150), cv2.COLOR_GRAY2BGR)
        # 顶-白边缘染红显示（底图上直接标出高权重边在哪）
        white_crop = white_dil[cy1:cy2, cx1:cx2]
        edge_crop = (cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
                                                (3, 3), 0), 50, 150) > 0)
        base[edge_crop & white_crop] = (0, 0, 255)

        for tag, pose, note in (
            ("V1", r["v1"], "V1参照"),
            ("base", (r["w1_1"]["px"], r["w1_1"]["py"], r["w1_1"]["angle"]),
             f"均匀 d={r['w1_1']['d_pos']:.1f}px"),
            ("best", (r[best_name]["px"], r[best_name]["py"], r[best_name]["angle"]),
             f"{best_name} d={r[best_name]['d_pos']:.1f}px"),
        ):
            one = base.copy()
            draw_pose(one, cat, pose[0] - cx1, pose[1] - cy1, pose[2], (0, 255, 0))
            cv2.circle(one, (int(pose[0]) - cx1, int(pose[1]) - cy1), 3, (0, 255, 0), 1)
            cv2.putText(one, note, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"tw{i:02d}_{cat}_{tag}.png"), one)
    print(f"图与JSON已存 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
