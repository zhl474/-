# -*- coding: utf-8 -*-
"""可视化：全角度理论测试 与 预言机筛角 的逐块对比图。

每块输出三张独立图（不互相遮挡）：
  fa{NN}_{cat}_V1.png     V1 参照位姿
  fa{NN}_{cat}_full.png   全角度（无筛角无先验）conv 匹配结果
  fa{NN}_{cat}_oracle.png 预言机筛角（用真实前景宽高筛角）结果
底图 = ROI Canny 白线；绿线=该图位姿；标题行印偏差。
同时保存 fa_results.json。
"""
import json
import os
import sys

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

# ---------------- 参数区 ----------------
IMAGE_PATH = os.path.join(SRC, "tools/vision/7.png")
V1_REF_JSON = os.path.join(SRC, "tools/vision/oriented_chamfer_v1/output/v1_results.json")
OUTPUT_DIR = os.path.join(THIS, "output")
CROP_MARGIN = 8
DT_CAP = 20.0
ANGLE_STEP = 2.0
SCREEN_TOL_PX = 6      # 预言机筛角容差（输入=真实前景宽高）
MARGIN = 10            # 全角度平移余量
# ----------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def conv_best(roi, kernels, ksize, anchors, angles, crop_xy):
    max_h, max_w = ksize
    th = max(roi.shape[0], max_h + 2 * MARGIN)
    tw = max(roi.shape[1], max_w + 2 * MARGIN)
    pt_, pl = (th - roi.shape[0]) // 2, (tw - roi.shape[1]) // 2
    padded = np.zeros((th, tw), dtype=np.float32)
    padded[pt_:pt_ + roi.shape[0], pl:pl + roi.shape[1]] = roi
    t = torch.from_numpy(padded)[None, None]
    if DEVICE == "cuda":
        t = t.to(torch.float16)
    t = t.to(DEVICE)
    counts = kernels.sum(dim=(1, 2, 3)).clamp(min=1.0)
    with torch.no_grad():
        fmap = F.conv2d(t, kernels)[0] / counts[:, None, None]
    best = None
    for j, ang in enumerate(angles):
        idx = int(torch.argmax(fmap[j]))
        iy, ix = np.unravel_index(idx, fmap[j].shape)
        s = float(fmap[j, iy, ix])
        if best is None or s > best[0]:
            ax, ay = anchors[j]
            best = (s, float(ang), float(ix + ax - pl) + crop_xy[0],
                    float(iy + ay - pt_) + crop_xy[1])
    return {"angle": best[1], "px": best[2], "py": best[3], "score": best[0]}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img = cv2.imread(IMAGE_PATH)
    geo = load_template_geometry("high")
    bpp, cpp = geo["block_px"], geo["connector_px"]
    meta = {c: build_angle_foreground_metadata(c, bpp, cpp, ANGLE_STEP)
            for c in set(ROTATION_TOTAL_ANGLE)}
    models = {c: v1.build_template_contour_model(c, bpp, cpp, 1.5)
              for c in set(ROTATION_TOTAL_ANGLE)}

    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    V = (DT_CAP - np.minimum(
        cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3), DT_CAP)).astype(np.float32)

    from ultralytics import YOLO
    yolo = YOLO(v1.DETECTION_MODEL_PATH)
    yolo(img, verbose=False)
    dets = v1.detect_blocks_yolo_v1(img, yolo, v1.ChamferV1Config())
    v1refs = [r for r in json.load(open(V1_REF_JSON))["results"] if r["found"]]

    def draw_pose(vis, cat, px, py, ang):
        rx, ry, _ = models[cat].rotated(ang)
        pts = np.stack((rx + px, ry + py), axis=1).astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(vis, (int(px), int(py)), 3, (0, 255, 0), 1)

    def true_fg(cat, vr):
        rx, ry, _ = models[cat].rotated(vr["pose_angle_deg"])
        xs, ys = rx + vr["px"], ry + vr["py"]
        return float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)

    rows = []
    for i, (det, vr) in enumerate(zip(dets, v1refs)):
        cat = det["category"]
        x1, y1, x2, y2 = det["box"]
        cx1, cy1 = max(0, int(x1) - CROP_MARGIN), max(0, int(y1) - CROP_MARGIN)
        cx2 = min(img.shape[1], int(x2) + CROP_MARGIN)
        cy2 = min(img.shape[0], int(y2) + CROP_MARGIN)
        roi = V[cy1:cy2, cx1:cx2]
        period = float(ROTATION_TOTAL_ANGLE[cat])
        angles = sorted(meta[cat])

        # 全角度
        kernels, ksize, anchors = ect.build_edge_kernels(meta[cat], angles, 1, DEVICE)
        o_full = conv_best(roi, kernels, ksize, anchors, angles, (cx1, cy1))

        # 预言机筛角：真实前景宽高
        fw, fh = true_fg(cat, vr)
        cand = [a for a in angles
                if abs(meta[cat][a]["fg_w"] - fw) <= SCREEN_TOL_PX
                and abs(meta[cat][a]["fg_h"] - fh) <= SCREEN_TOL_PX]
        if len(cand) < 3:
            cand = angles
        kernels2, ksize2, anchors2 = ect.build_edge_kernels(meta[cat], cand, 1, DEVICE)
        o_ora = conv_best(roi, kernels2, ksize2, anchors2, cand, (cx1, cy1))

        for o in (o_full, o_ora):
            o["d_pos"] = float(np.hypot(o["px"] - vr["px"], o["py"] - vr["py"]))
            d = abs(o["angle"] - vr["pose_angle_deg"]) % period
            o["d_angle"] = float(min(d, period - d))

        rows.append({"index": i, "category": cat,
                     "v1": (vr["px"], vr["py"], vr["pose_angle_deg"]),
                     "full": o_full, "oracle": o_ora})

        # 出图：每方法独立一张
        crop = img[cy1:cy2, cx1:cx2]
        base = cv2.cvtColor(cv2.Canny(cv2.GaussianBlur(
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (3, 3), 0), 50, 150),
            cv2.COLOR_GRAY2BGR)
        for tag, pose, note in (
            ("V1", rows[-1]["v1"], "V1参照"),
            ("full", (o_full["px"], o_full["py"], o_full["angle"]),
             f"全角度 d={o_full['d_pos']:.1f}px da={o_full['d_angle']:.1f}°"),
            ("oracle", (o_ora["px"], o_ora["py"], o_ora["angle"]),
             f"预言机筛角({len(cand)}角) d={o_ora['d_pos']:.1f}px da={o_ora['d_angle']:.1f}°"),
        ):
            one = base.copy()
            draw_pose(one, cat, pose[0] - cx1, pose[1] - cy1, pose[2])
            cv2.putText(one, note, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"fa{i:02d}_{cat}_{tag}.png"), one)

    with open(os.path.join(OUTPUT_DIR, "fa_results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    for name, key in (("全角度", "full"), ("预言机筛角", "oracle")):
        d = sorted(r[key]["d_pos"] for r in rows)
        da = sorted(r[key]["d_angle"] for r in rows)
        n = len(d)
        print(f"{name}: 中位Δpx={d[n//2]:.2f} <2px:{sum(x<2 for x in d)/n:.0%} "
              f"<5px:{sum(x<5 for x in d)/n:.0%} 中位Δ角={da[n//2]:.2f}°")
    print(f"\n35块×3张图 + fa_results.json 已存 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
