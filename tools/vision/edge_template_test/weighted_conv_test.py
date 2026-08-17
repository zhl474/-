# -*- coding: utf-8 -*-
"""加权边缘置信 conv 匹配实验：给 顶-白(TOP_BACKGROUND) 边界更大权重，对比均匀 Canny。

复用 edge_confidence_test/edge_pipeline_test.py 的四类分类（整图跑一次），
把每类 LSD 线段光栅化后按类算距离场，合成加权价值图：
    V(p) = max_c [ w_c * (CAP - min(DT_c(p), CAP)) ]
然后走与 edge_template_test 变体C 完全相同的 conv 匹配（同 ROI、同筛角、同归一化）。

对比：
  C  基线：均匀 Canny 距离场（无类别权重）
  E1 v2默认权重：TOP_BG=3.0 TOP_SIDE=3.0 SIDE_BG=0.1 UNKNOWN=0.7
  E2 顶白专项：TOP_BG=3.0 其余=1.0（隔离"顶-白加权"本身的贡献）
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
import edge_pipeline_test as ept   # 分类流水线（只 import，不跑 main）

# ---------------- 参数区（改这里） ----------------
IMAGE_PATH = os.path.join(SRC, "tools/vision/7.png")
V1_REF_JSON = os.path.join(SRC, "tools/vision/oriented_chamfer_v1/output/v1_results.json")
OUTPUT_DIR = os.path.join(THIS, "output")

CROP_MARGIN = 8
DT_CAP = 20.0
ANGLE_STEP = 2.0
SCREEN_TOL_PX = 6
SEARCH_MARGIN = 4

WEIGHT_SETS = {
    "E1": {"TOP_BACKGROUND": 3.0, "TOP_SIDE": 3.0, "SIDE_BACKGROUND": 0.1, "UNKNOWN": 0.7},
    "E2": {"TOP_BACKGROUND": 3.0, "TOP_SIDE": 1.0, "SIDE_BACKGROUND": 1.0, "UNKNOWN": 1.0},
}

# E3：不换边缘来源（保留完整 Canny），只按像素级白板邻接分两组加权。
# 公平检验“顶-白边界加权”本身，排除 LSD 稀疏化的干扰。
W_WHITE_E3 = 3.0          # 白板邻接 Canny 边缘的权重（顶-白边界的近似）
E3_WHITE_REACH_PX = 3     # 白板 mask 膨胀半径：边缘 3px 内有白板算“邻接”
PROBLEM_CASES = [1, 4, 7, 9, 14, 17, 20, 24]   # 上一轮 A/C 的失败块，单独出图
# ------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def classify_full_image(bgr):
    """整图跑一遍 edge_pipeline_test 的分类，返回带 .cls/.fragments 的线段列表。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, ept.GAUSS_KSIZE, 0)
    lines = [ln for ln in ept.lsd_detect(blur) if ln.length >= ept.MIN_LINE_LENGTH]
    bonus_pool = list(lines)
    if ept.DIR_FILTER_ENABLED:
        lines = ept.direction_filter(lines)
    lines = ept.merge_collinear(lines)
    white_mask = ept.build_white_mask(bgr)
    for ln in lines:
        ln.white_pos = ept.side_white_ratio(ln, white_mask, +1)
        ln.white_neg = ept.side_white_ratio(ln, white_mask, -1)
    image_center = np.array([gray.shape[1] / 2.0, gray.shape[0] / 2.0])
    geo_pairs, cands, _ = ept.find_side_pair_candidates(lines, bonus_pool=bonus_pool)
    for c in cands:
        ept.camera_direction_check(c, image_center, None)
    ept.assign_pairs(cands)
    ept.classify_rest(lines)
    return lines


def build_value_maps(bgr, shape, weight_set):
    """按类光栅化 LSD 线段 → 各类距离场 → 加权合成 V（越大越好，≤ w*CAP）。"""
    per_class = {}
    for ln in lines_g:
        mask = per_class.setdefault(ln.cls, np.zeros(shape, dtype=np.uint8))
        for f1, f2 in ln.fragments:
            cv2.line(mask, ept.pt(f1), ept.pt(f2), 255, 1)
    maps = {}
    for cls, mask in per_class.items():
        w = weight_set.get(cls, 1.0)
        if w <= 0 or not np.any(mask):
            continue
        dt = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
        maps[cls] = w * (DT_CAP - np.minimum(dt, DT_CAP)).astype(np.float32)
    if not maps:
        return np.zeros(shape, dtype=np.float32)
    return np.maximum.reduce(list(maps.values()))


def conv_argmax(value_roi, kernels, ksize, anchors, angles, crop_xy):
    """与变体C同构的归一化 conv 选优（高价值=好），返回全图中心+角度。"""
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


def run_weighted(V_full, meta, cat, det, box_wh, crop_xy):
    angles = sorted(meta[cat])
    bw, bh = box_wh
    cand = [a for a in angles
            if abs(meta[cat][a]["fg_w"] - bw) <= SCREEN_TOL_PX
            and abs(meta[cat][a]["fg_h"] - bh) <= SCREEN_TOL_PX]
    if len(cand) < 3:
        cand = angles
    x1, y1, x2, y2 = det["box"]
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

    print("整图四类分类（LSD + 白板 + 侧面边对）...")
    t0 = time.perf_counter()
    global lines_g
    lines_g = classify_full_image(img)
    cls_count = {}
    for ln in lines_g:
        cls_count[ln.cls] = cls_count.get(ln.cls, 0) + 1
    print(f"  {time.perf_counter()-t0:.1f}s, 共 {len(lines_g)} 条: {cls_count}")

    value_maps = {}
    for name, ws in WEIGHT_SETS.items():
        value_maps[name] = build_value_maps(img, img.shape[:2], ws)

    # E3 价值图：完整 Canny 分白板邻接/非邻接两组距离场
    gray_f = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges_f = cv2.Canny(gray_f, 50, 150) > 0
    white_mask = ept.build_white_mask(img)
    white_dil = cv2.dilate(white_mask, np.ones((2 * E3_WHITE_REACH_PX + 1,) * 2, np.uint8)) > 0
    e_white = (edges_f & white_dil).astype(np.uint8) * 255
    e_rest = (edges_f & ~white_dil).astype(np.uint8) * 255
    dt_w = cv2.distanceTransform(255 - e_white, cv2.DIST_L2, 3)
    dt_r = cv2.distanceTransform(255 - e_rest, cv2.DIST_L2, 3)
    value_maps["E3"] = np.maximum(
        W_WHITE_E3 * (DT_CAP - np.minimum(dt_w, DT_CAP)),
        (DT_CAP - np.minimum(dt_r, DT_CAP)),
    ).astype(np.float32)
    n_w = int((edges_f & white_dil).sum())
    print(f"E3: Canny {int(edges_f.sum())} px，白板邻接 {n_w} px ({n_w/edges_f.sum():.0%})")

    meta = {}
    for cat in set(ROTATION_TOTAL_ANGLE):
        meta[cat] = build_angle_foreground_metadata(cat, bpp, cpp, ANGLE_STEP)

    from ultralytics import YOLO
    yolo = YOLO(v1.DETECTION_MODEL_PATH)
    yolo(img, verbose=False)
    dets = v1.detect_blocks_yolo_v1(img, yolo, v1.ChamferV1Config())
    v1refs = [r for r in json.load(open(V1_REF_JSON))["results"] if r["found"]]

    # C 基线同轮重跑（同参照、同筛角），保证对比公平
    all_names = ("C", *WEIGHT_SETS, "E3")
    results = {name: [] for name in all_names}
    for det, vr in zip(dets, v1refs):
        cat = det["category"]
        x1, y1, x2, y2 = det["box"]
        cx1 = max(0, int(x1) - CROP_MARGIN)
        cy1 = max(0, int(y1) - CROP_MARGIN)
        cx2 = min(img.shape[1], int(x2) + CROP_MARGIN)
        cy2 = min(img.shape[0], int(y2) + CROP_MARGIN)
        roi = img[cy1:cy2, cx1:cx2]
        g = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        e = cv2.Canny(g, 50, 150)
        c_img = (DT_CAP - np.minimum(
            cv2.distanceTransform(255 - e, cv2.DIST_L2, 3), DT_CAP)).astype(np.float32)

        period = float(ROTATION_TOTAL_ANGLE[cat])
        row = {"category": cat, "v1": (vr["px"], vr["py"], vr["pose_angle_deg"])}
        out = ect.run_variant("C", c_img, meta[cat], sorted(meta[cat]), (cx1, cy1),
                              (int(x2 - x1), int(y2 - y1)), SCREEN_TOL_PX)
        out["d_pos"] = float(np.hypot(out["px"] - vr["px"], out["py"] - vr["py"]))
        out["d_angle"] = float(angle_diff(out["angle"], vr["pose_angle_deg"], period))
        results["C"].append(out)

        for name in WEIGHT_SETS:
            t1 = time.perf_counter()
            o = run_weighted(value_maps[name], meta, cat, det,
                             (int(x2 - x1), int(y2 - y1)), (cx1, cy1))
            o["ms"] = (time.perf_counter() - t1) * 1000.0
            o["d_pos"] = float(np.hypot(o["px"] - vr["px"], o["py"] - vr["py"]))
            o["d_angle"] = float(angle_diff(o["angle"], vr["pose_angle_deg"], period))
            results[name].append(o)
        t1 = time.perf_counter()
        o3 = run_weighted(value_maps["E3"], meta, cat, det,
                          (int(x2 - x1), int(y2 - y1)), (cx1, cy1))
        o3["ms"] = (time.perf_counter() - t1) * 1000.0
        o3["d_pos"] = float(np.hypot(o3["px"] - vr["px"], o3["py"] - vr["py"]))
        o3["d_angle"] = float(angle_diff(o3["angle"], vr["pose_angle_deg"], period))
        results["E3"].append(o3)
        row["C"] = results["C"][-1]
        for name in all_names[1:]:
            row[name] = results[name][-1]
        rows_out.append(row)
        print(f"{cat:<10s} " + "  ".join(
            f"{n}: d={row[n]['d_pos']:.1f}px da={row[n]['d_angle']:.1f}°"
            for n in all_names))

    print("\n===== 汇总（vs V1 参照） =====")
    print(f"{'变体':<4s} {'中位Δpx':>8s} {'<2px':>6s} {'<5px':>6s} {'中位Δ角':>8s} {'<2°':>6s}")
    for name in all_names:
        d = sorted(r["d_pos"] for r in results[name])
        da = sorted(r["d_angle"] for r in results[name])
        n = len(d)
        print(f"{name:<4s} {d[n//2]:8.2f} {sum(x<2 for x in d)/n:5.0%} "
              f"{sum(x<5 for x in d)/n:5.0%} {da[n//2]:8.2f} {sum(x<2 for x in da)/n:5.0%}")

    print("\n上一轮8个失败块（L/z）明细：")
    for i in PROBLEM_CASES:
        r = rows_out[i]
        print(f"[{i:02d}] {r['category']:<9s} " + "  ".join(
            f"{n}: d={r[n]['d_pos']:.1f}" for n in all_names))

    with open(os.path.join(OUTPUT_DIR, "weighted_results.json"), "w", encoding="utf-8") as f:
        json.dump(rows_out, f, ensure_ascii=False, indent=2)

    # 失败块可视化：分类着色图 + 各方法位姿（每方法单独一张，不互相遮挡）
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

        cls_vis = (crop * 0.55).astype(np.uint8)
        colors = {"TOP_BACKGROUND": (0, 255, 0), "TOP_SIDE": (255, 0, 0),
                  "SIDE_BACKGROUND": (0, 0, 255), "UNKNOWN": (160, 160, 160)}
        for ln in lines_g:
            for f1, f2 in ln.fragments:
                p1 = (int(f1[0]) - cx1, int(f1[1]) - cy1)
                p2 = (int(f2[0]) - cx1, int(f2[1]) - cy1)
                if (0 <= p1[0] < cls_vis.shape[1] and 0 <= p1[1] < cls_vis.shape[0]) or \
                   (0 <= p2[0] < cls_vis.shape[1] and 0 <= p2[1] < cls_vis.shape[0]):
                    cv2.line(cls_vis, p1, p2, colors[ln.cls], 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"w{i:02d}_{cat}_cls.png"), cls_vis)

        for tag, pose, color, note in (
            ("V1", r["v1"], (0, 255, 255), "V1参照"),
            ("C", (r["C"]["px"], r["C"]["py"], r["C"]["angle"]), (0, 255, 0),
             f"C d={r['C']['d_pos']:.1f}px"),
            ("E3", (r["E3"]["px"], r["E3"]["py"], r["E3"]["angle"]), (0, 255, 0),
             f"E3(Canny白邻接x{W_WHITE_E3}) d={r['E3']['d_pos']:.1f}px"),
            ("E2", (r["E2"]["px"], r["E2"]["py"], r["E2"]["angle"]), (0, 255, 0),
             f"E2(顶白3.0) d={r['E2']['d_pos']:.1f}px"),
            ("E1", (r["E1"]["px"], r["E1"]["py"], r["E1"]["angle"]), (0, 255, 0),
             f"E1(v2权重) d={r['E1']['d_pos']:.1f}px"),
        ):
            one = cv2.cvtColor(cv2.Canny(cv2.GaussianBlur(
                cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (3, 3), 0), 50, 150), cv2.COLOR_GRAY2BGR)
            draw_pose(one, cat, pose[0] - cx1, pose[1] - cy1, pose[2], color)
            cv2.circle(one, (int(pose[0]) - cx1, int(pose[1]) - cy1), 3, color, 1)
            cv2.putText(one, note, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"w{i:02d}_{cat}_{tag}.png"), one)
    print(f"\n图与JSON已存 {OUTPUT_DIR}")


rows_out = []

if __name__ == "__main__":
    main()
