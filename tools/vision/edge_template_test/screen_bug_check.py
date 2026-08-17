# -*- coding: utf-8 -*-
"""定位筛角丢真角度的错因：bbox宽高 是否可以顶替 顶面前景宽高。

1. 量化：8个失败块 + 10个好块上，YOLO bbox 与 V1位姿下模板轮廓紧边框（=真实顶面前景）
   的宽高差、中心偏移方向。差 >6px 即筛角必然丢角。
2. 预言机测试：筛角改用真实前景宽高（作弊输入），conv 匹配其余不变，
   看失败块是否复活 → 区分“筛角是唯一错因”还是“还有侧缘位置锁死”。
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

# ---------------- 参数区 ----------------
IMAGE_PATH = os.path.join(SRC, "tools/vision/7.png")
V1_REF_JSON = os.path.join(SRC, "tools/vision/oriented_chamfer_v1/output/v1_results.json")
CROP_MARGIN = 8
DT_CAP = 20.0
ANGLE_STEP = 2.0
SCREEN_TOL_PX = 6
SEARCH_MARGIN = 4
PROBLEM_CASES = [1, 4, 7, 9, 14, 17, 20, 24]
GOOD_CASES = [0, 2, 3, 5, 8, 10, 11, 19, 22, 25]
# ----------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def true_fg(cat, px, py, angle):
    """V1位姿下模板轮廓的紧边框（宽,高,中心），即真实顶面前景的观测。"""
    m = models[cat]
    rx, ry, _ = m.rotated(angle)
    xs, ys = rx + px, ry + py
    return (float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1),
            (float((xs.max() + xs.min()) / 2), float((ys.max() + ys.min()) / 2)))


def conv_best(V_roi, kernels, ksize, anchors, angles, crop_xy):
    max_h, max_w = ksize
    th = max(V_roi.shape[0], max_h + 2 * SEARCH_MARGIN)
    tw = max(V_roi.shape[1], max_w + 2 * SEARCH_MARGIN)
    pt_, pl = (th - V_roi.shape[0]) // 2, (tw - V_roi.shape[1]) // 2
    padded = np.zeros((th, tw), dtype=np.float32)
    padded[pt_:pt_ + V_roi.shape[0], pl:pl + V_roi.shape[1]] = V_roi
    t = torch.from_numpy(padded)[None, None]
    if DEVICE == "cuda":
        t = t.to(torch.float16)
    t = t.to(DEVICE)
    counts = kernels.sum(dim=(1, 2, 3)).clamp(min=1.0)
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
                    float(ix + ax - pl) + crop_xy[0], float(iy + ay - pt_) + crop_xy[1])
    return {"angle": best[1], "px": best[2], "py": best[3]}


def main():
    img = cv2.imread(IMAGE_PATH)
    img_h, img_w = img.shape[:2]
    center = np.array([img_w / 2.0, img_h / 2.0])
    geo = load_template_geometry("high")
    bpp, cpp = geo["block_px"], geo["connector_px"]

    global models
    models = {}
    for cat in set(ROTATION_TOTAL_ANGLE):
        models[cat] = v1.build_template_contour_model(cat, bpp, cpp, 1.5)
    meta = {cat: build_angle_foreground_metadata(cat, bpp, cpp, ANGLE_STEP)
            for cat in set(ROTATION_TOTAL_ANGLE)}

    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    V = (DT_CAP - np.minimum(
        cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3), DT_CAP)).astype(np.float32)

    from ultralytics import YOLO
    yolo = YOLO(v1.DETECTION_MODEL_PATH)
    yolo(img, verbose=False)
    dets = v1.detect_blocks_yolo_v1(img, yolo, v1.ChamferV1Config())
    v1refs = [r for r in json.load(open(V1_REF_JSON))["results"] if r["found"]]

    print("== 1) bbox vs 真实顶面前景 的宽高差（>6px 即筛角丢真角）==")
    print(f"{'case':<7s}{'类别':<10s}{'Δw':>6s}{'Δh':>6s}  bbox中心-前景中心·d̂   判定")
    inflation = {}
    for i in PROBLEM_CASES + GOOD_CASES:
        det, vr = dets[i], v1refs[i]
        cat = det["category"]
        x1, y1, x2, y2 = det["box"]
        bw, bh = float(x2 - x1), float(y2 - y1)
        fw, fh, fc = true_fg(cat, vr["px"], vr["py"], vr["pose_angle_deg"])
        dw, dh = bw - fw, bh - fh
        bc = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        d_hat = center - fc
        d_hat /= np.linalg.norm(d_hat)
        proj = float((bc - np.array(fc)) @ d_hat)
        inflation[i] = (dw, dh, proj)
        lost = abs(dw) > SCREEN_TOL_PX or abs(dh) > SCREEN_TOL_PX
        tag = "问题块" if i in PROBLEM_CASES else "好块"
        print(f"[{i:02d}]   {cat:<9s}{dw:6.1f}{dh:6.1f}   {proj:8.1f}px"
              f"   {tag} {'!!会丢真角' if lost else 'ok'}")

    print("\n== 2) 预言机筛角（用真实前景宽高筛，其余不变）==")
    for i in PROBLEM_CASES:
        det, vr = dets[i], v1refs[i]
        cat = det["category"]
        x1, y1, x2, y2 = det["box"]
        cx1, cy1 = max(0, int(x1) - CROP_MARGIN), max(0, int(y1) - CROP_MARGIN)
        period = float(ROTATION_TOTAL_ANGLE[cat])
        fw, fh, fc = true_fg(cat, vr["px"], vr["py"], vr["pose_angle_deg"])
        angles = sorted(meta[cat])
        cand = [a for a in angles
                if abs(meta[cat][a]["fg_w"] - fw) <= SCREEN_TOL_PX
                and abs(meta[cat][a]["fg_h"] - fh) <= SCREEN_TOL_PX]
        roi = V[cy1:cy1 + (int(y2) + CROP_MARGIN - cy1), cx1:cx1 + (int(x2) + CROP_MARGIN - cx1)]
        kernels, ksize, anchors = ect.build_edge_kernels(meta[cat], cand, 1, DEVICE)
        o = conv_best(roi, kernels, ksize, anchors, cand, (cx1, cy1))
        dp = float(np.hypot(o["px"] - vr["px"], o["py"] - vr["py"]))
        d = abs(o["angle"] - vr["pose_angle_deg"]) % period
        da = float(min(d, period - d))
        n_ok = sum(1 for a in cand if min(abs(a - vr["pose_angle_deg"]) % period,
                                          period - abs(a - vr["pose_angle_deg"]) % period) < 1.5)
        print(f"[{i:02d}] {cat:<9s} 候选{len(cand):3d}角(真角在内:{'是' if n_ok else '否'})  "
              f"→ d={dp:5.1f}px  da={da:4.1f}°  {'✓复活' if dp < 2 else '✗仍偏(侧缘锁死实锤)' if da < 1.5 else '✗角度仍错'}")


if __name__ == "__main__":
    main()
