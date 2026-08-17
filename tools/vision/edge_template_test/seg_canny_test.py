# -*- coding: utf-8 -*-
"""YOLO→YOLO-seg 筛选 Canny → 叠加模板匹配 实验（离线）。

用户方案：
  seg 不准没关系，它只需要圈出"自己方块"——用 seg mask 筛掉邻块/背景的 Canny 边缘，
  剩下的 Canny（边界准）以大权重参与匹配，seg 区域本身（稳但糙）以小权重参与，
  两者叠加后做边缘模板匹配。

变体：
  S1_full : 只用 seg 筛过的 Canny，线模板，全角度（检验"全角度失败=邻块干扰"的前提）
  S2_full : S1 + seg 区域重叠项（w_canny=3, w_seg=1），全角度（完整方案）
  S2_b6   : S2 + bbox±6px 筛角（现有基线筛角）
  S3_seg6 : S2 + seg前景宽高±6px 筛角（正式同源的正确筛角）
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
IMAGE_PATH = os.path.join(SRC, "tools/vision/7.png")
V1_REF_JSON = os.path.join(SRC, "tools/vision/oriented_chamfer_v1/output/v1_results.json")
SEG_MODEL_PATH = os.path.join(SRC, "competition", "model", "best_seg.pt")
OUTPUT_DIR = os.path.join(THIS, "output")

CROP_MARGIN = 8
DT_CAP = 20.0
ANGLE_STEP = 2.0
SEG_CONF = 0.25
SEG_FILTER_DILATE_PX = 4   # seg mask 膨胀半径：Canny 保留在此范围内
W_CANNY = 3.0              # Canny 边缘分权重（大）
W_SEG = 1.0                # seg 区域重叠分权重（小）
SCREEN_TOL_PX = 6
MARGIN = 10                # 平移余量
# -------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
meta = {}


def get_seg_mask(seg_model, crop_bgr):
    """与正式 _get_mask_by_yolo_seg 同逻辑：top_surface 类里取最大 mask。"""
    results = seg_model(crop_bgr, conf=SEG_CONF, verbose=False, retina_masks=True)
    result = results[0]
    if result.masks is None or len(result.masks.data) == 0:
        raise RuntimeError("seg 无 mask")
    top_ids = [int(k) for k, name in result.names.items() if name == "top_surface"]
    idxs = list(range(len(result.masks.data)))
    if top_ids and result.boxes is not None and result.boxes.cls is not None:
        cls = result.boxes.cls.detach().cpu().numpy().astype(int)
        idxs = [i for i, c in enumerate(cls) if c in top_ids]
    best, best_area = None, -1
    for i in idxs:
        m = (result.masks.data[i].detach().cpu().numpy() > 0.5).astype(np.uint8)
        if m.sum() > best_area:
            best_area, best = int(m.sum()), m
    if best is None or best_area <= 0:
        raise RuntimeError("seg mask 为空")
    if best.shape != crop_bgr.shape[:2]:
        best = cv2.resize(best, (crop_bgr.shape[1], crop_bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    return best


def build_filled_kernels(category, angles, device):
    """实心模板核（与线核同画布布局，anchor 相同），用于 seg 区域重叠项。"""
    binary_list = [meta[category][a]["binary"] for a in angles]
    max_h = max(b.shape[0] for b in binary_list) + 2 * ect.KERNEL_MARGIN
    max_w = max(b.shape[1] for b in binary_list) + 2 * ect.KERNEL_MARGIN
    kernels = np.zeros((len(angles), 1, max_h, max_w), dtype=np.float32)
    anchors, counts = [], []
    for i, binary in enumerate(binary_list):
        h, w = binary.shape
        kernels[i, 0, ect.KERNEL_MARGIN:ect.KERNEL_MARGIN + h,
                ect.KERNEL_MARGIN:ect.KERNEL_MARGIN + w] = binary.astype(np.float32)
        ax, ay = meta[category][angles[i]]["anchor"]
        anchors.append((ax + ect.KERNEL_MARGIN, ay + ect.KERNEL_MARGIN))
        counts.append(float(binary.sum()))
    t = torch.from_numpy(kernels)
    if device == "cuda":
        t = t.to(torch.float16)
    return t.to(device), (max_h, max_w), anchors, np.array(counts, dtype=np.float32)


def to_t(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None, None]
    if DEVICE == "cuda":
        t = t.to(torch.float16)
    return t.to(DEVICE)


def match_combined(roi_h, roi_w, edge_val, seg_f, angles, crop_xy, w_c, w_s):
    """edge_val: 筛后Canny价值图(cap-dt)；seg_f: seg 0/1。返回最优位姿。"""
    line_k, ksize, anchors = ect.build_edge_kernels(meta[CAT], angles, 1, DEVICE)
    fill_k, ksize2, anchors2, fill_counts = build_filled_kernels(CAT, angles, DEVICE)

    def pad(a):
        th = max(a.shape[0], ksize[0] + 2 * MARGIN)
        tw = max(a.shape[1], ksize[1] + 2 * MARGIN)
        pt_, pl = (th - a.shape[0]) // 2, (tw - a.shape[1]) // 2
        out = np.zeros((th, tw), dtype=np.float32)
        out[pt_:pt_ + a.shape[0], pl:pl + a.shape[1]] = a
        return out, pt_, pl

    E, pt_e, pl_e = pad(edge_val)
    S, pt_s, pl_s = pad(seg_f)
    lc = line_k.sum(dim=(1, 2, 3)).clamp(min=1.0)
    fc = torch.from_numpy(fill_counts).to(DEVICE)[:, None, None].clamp(min=1.0)
    with torch.no_grad():
        f_line = F.conv2d(to_t(E), line_k)[0] / lc[:, None, None] / DT_CAP   # 0..1
        f_fill = F.conv2d(to_t(S), fill_k)[0] / fc / 1.0                      # 重叠率 0..1
    total = w_c * f_line + w_s * f_fill
    best = None
    for i, ang in enumerate(angles):
        idx = int(torch.argmax(total[i]))
        iy, ix = np.unravel_index(idx, total[i].shape)
        s = float(total[i, iy, ix])
        if best is None or s > best[0]:
            ax, ay = anchors[i]
            best = (s, float(ang), float(ix + ax - pl_e) + crop_xy[0],
                    float(iy + ay - pt_e) + crop_xy[1])
    return {"angle": best[1], "px": best[2], "py": best[3], "score": best[0]}


def main():
    global CAT
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img = cv2.imread(IMAGE_PATH)
    geo = load_template_geometry("high")
    bpp, cpp = geo["block_px"], geo["connector_px"]
    for cat in set(ROTATION_TOTAL_ANGLE):
        meta[cat] = build_angle_foreground_metadata(cat, bpp, cpp, ANGLE_STEP)
    models = {c: v1.build_template_contour_model(c, bpp, cpp, 1.5)
              for c in set(ROTATION_TOTAL_ANGLE)}

    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges_full = cv2.Canny(gray, 50, 150)

    yolo = YOLO(v1.DETECTION_MODEL_PATH)
    yolo(img, verbose=False)
    seg_model = YOLO(SEG_MODEL_PATH, task="segment")
    seg_model(img[:64, :64], verbose=False)  # warmup
    dets = v1.detect_blocks_yolo_v1(img, yolo, v1.ChamferV1Config())
    v1refs = [r for r in json.load(open(V1_REF_JSON))["results"] if r["found"]]

    variants = ["S1_full", "S2_full", "S2_b6", "S3_seg6"]
    rows = []
    per = {n: [] for n in variants}
    seg_time = []
    for i, (det, vr) in enumerate(zip(dets, v1refs)):
        CAT = det["category"]
        x1, y1, x2, y2 = det["box"]
        cx1, cy1 = max(0, int(x1) - CROP_MARGIN), max(0, int(y1) - CROP_MARGIN)
        cx2 = min(img.shape[1], int(x2) + CROP_MARGIN)
        cy2 = min(img.shape[0], int(y2) + CROP_MARGIN)
        crop = img[cy1:cy2, cx1:cx2]

        t0 = time.perf_counter()
        try:
            seg = get_seg_mask(seg_model, crop)
        except Exception as exc:
            print(f"[{i:02d}] {CAT} seg 失败: {exc}")
            seg = np.zeros(crop.shape[:2], dtype=np.uint8)
        seg_time.append((time.perf_counter() - t0) * 1000.0)
        seg_f = (seg > 0).astype(np.float32)

        # seg 筛选 Canny：只保留 seg 膨胀范围内的边缘
        edges_roi = edges_full[cy1:cy2, cx1:cx2] > 0
        keep = edges_roi & (cv2.dilate(
            seg, np.ones((2 * SEG_FILTER_DILATE_PX + 1,) * 2, np.uint8)) > 0)
        e_f = keep.astype(np.uint8) * 255
        A_e = (DT_CAP - np.minimum(
            cv2.distanceTransform(255 - e_f, cv2.DIST_L2, 3), DT_CAP)).astype(np.float32)

        period = float(ROTATION_TOTAL_ANGLE[CAT])
        angles = sorted(meta[CAT])
        # seg 前景宽高（正式同源筛角输入）
        ys, xs = np.nonzero(seg)
        seg_w = int(xs.max()) - int(xs.min()) + 1
        seg_h = int(ys.max()) - int(ys.min()) + 1
        bw, bh = int(x2 - x1), int(y2 - y1)

        cand_sets = {
            "S1_full": angles,
            "S2_full": angles,
            "S2_b6": [a for a in angles
                      if abs(meta[CAT][a]["fg_w"] - bw) <= SCREEN_TOL_PX
                      and abs(meta[CAT][a]["fg_h"] - bh) <= SCREEN_TOL_PX] or angles,
            "S3_seg6": [a for a in angles
                        if abs(meta[CAT][a]["fg_w"] - seg_w) <= SCREEN_TOL_PX
                        and abs(meta[CAT][a]["fg_h"] - seg_h) <= SCREEN_TOL_PX] or angles,
        }

        row = {"index": i, "category": CAT,
               "v1": (vr["px"], vr["py"], vr["pose_angle_deg"]), "seg_fg": [seg_w, seg_h]}
        for name in variants:
            o = match_combined(crop.shape[0], crop.shape[1], A_e, seg_f,
                               cand_sets[name], (cx1, cy1),
                               W_CANNY, 0.0 if name == "S1_full" else W_SEG)
            o["d_pos"] = float(np.hypot(o["px"] - vr["px"], o["py"] - vr["py"]))
            d = abs(o["angle"] - vr["pose_angle_deg"]) % period
            o["d_angle"] = float(min(d, period - d))
            o["n_angles"] = len(cand_sets[name])
            per[name].append(o)
            row[name] = o
        rows.append(row)

    print(f"\n===== 汇总（vs V1；seg平均耗时 {np.mean(seg_time):.1f}ms/块）=====")
    print(f"{'变体':<10s} {'中位Δpx':>8s} {'<2px':>6s} {'<5px':>6s} {'中位Δ角':>8s} {'<2°':>6s}")
    for name in variants:
        d = sorted(o["d_pos"] for o in per[name])
        da = sorted(o["d_angle"] for o in per[name])
        k = len(d)
        print(f"{name:<10s} {d[k//2]:8.2f} {sum(x<2 for x in d)/k:5.0%} "
              f"{sum(x<5 for x in d)/k:5.0%} {da[k//2]:8.2f} {sum(x<2 for x in da)/k:5.0%}")

    print("\n原8个失败块：")
    for i in (1, 4, 7, 9, 14, 17, 20, 24):
        r = rows[i]
        print(f"[{i:02d}] {r['category']:<9s} " + "  ".join(
            f"{n}:{r[n]['d_pos']:.1f}" for n in variants))

    with open(os.path.join(OUTPUT_DIR, "seg_canny_results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # ---- 出图：每块 _seg(筛后边缘可视化) + _V1 + _S1full + _S2full + _S3 ----
    for i, (det, vr) in enumerate(zip(dets, v1refs)):
        r = rows[i]
        cat = r["category"]
        x1, y1, x2, y2 = (int(v) for v in vr["detection_box"])
        cx1, cy1 = max(0, x1 - CROP_MARGIN), max(0, y1 - CROP_MARGIN)
        cx2 = min(img.shape[1], x2 + CROP_MARGIN)
        cy2 = min(img.shape[0], y2 + CROP_MARGIN)
        crop = img[cy1:cy2, cx1:cx2]
        segvis = (crop * 0.45).astype(np.uint8)
        ys, xs = np.nonzero(get_seg_mask(seg_model, crop))
        if len(ys):
            ov = segvis.copy()
            ov[ys, xs] = (0, 80, 0)
            segvis = cv2.addWeighted(segvis, 0.5, ov, 0.5, 0)
        keep = (edges_full[cy1:cy2, cx1:cx2] > 0) & (cv2.dilate(
            get_seg_mask(seg_model, crop),
            np.ones((2 * SEG_FILTER_DILATE_PX + 1,) * 2, np.uint8)) > 0)
        segvis[keep] = (255, 255, 255)
        cv2.putText(segvis, "seg(green)+filtered Canny(white)", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"sc{i:02d}_{cat}_seg.png"), segvis)

        base = cv2.cvtColor((~keep).astype(np.uint8) * 0 + (keep.astype(np.uint8) * 255),
                            cv2.COLOR_GRAY2BGR) if False else segvis.copy()
        for tag, key in (("V1", "v1"), ("S1full", "S1_full"), ("S2full", "S2_full"),
                         ("S3", "S3_seg6")):
            pose = r[key] if key == "v1" else (r[key]["px"], r[key]["py"], r[key]["angle"])
            one = base.copy()
            rx, ry, _ = models[cat].rotated(pose[2])
            pts = np.stack((rx + pose[0] - cx1, ry + pose[1] - cy1), axis=1).astype(np.int32)
            cv2.polylines(one, [pts], True, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(one, (int(pose[0]) - cx1, int(pose[1]) - cy1), 3, (0, 255, 255), 1)
            note = "V1参照" if key == "v1" else \
                f"{key} d={r[key]['d_pos']:.1f}px da={r[key]['d_angle']:.1f}°"
            cv2.putText(one, note, (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"sc{i:02d}_{cat}_{tag}.png"), one)
    print(f"\n35块×5图 + JSON 已存 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
