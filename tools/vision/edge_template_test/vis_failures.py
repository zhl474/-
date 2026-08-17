# -*- coding: utf-8 -*-
"""把 A/C 变体与 V1 参照分歧最大的块画出来肉眼裁决。"""
import json
import os
import sys

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
SRC = "/home/zhl/SingleArmTetris/SingleArmTetris/src"
for p in (SRC, os.path.join(SRC, "image_process"),
          os.path.join(SRC, "tools/vision/oriented_chamfer_v1"),
          os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np

# ---------------- 参数区 ----------------
CASES = None          # None = 全部35块；或传下标列表只画指定块，如 [1, 4, 7]
CROP_MARGIN = 8
# ----------------------------------------

import oriented_chamfer_v1 as v1
from image_process_lib.template_config import load_template_geometry

rows = json.load(open(os.path.join(os.path.dirname(__file__), "output", "edge_conv_results.json")))
if CASES is None:
    CASES = list(range(len(rows)))
with open(os.path.join(SRC, "tools/vision/oriented_chamfer_v1/output/v1_results.json")) as f:
    v1rows = [x for x in json.load(f)["results"] if x["found"]]
img = cv2.imread(os.path.join(SRC, "tools/vision/7.png"))
geometry = load_template_geometry("high")
outdir = os.path.join(os.path.dirname(__file__), "output")
models = {}


def draw_pose(vis, cat, px, py, angle, color):
    if cat not in models:
        models[cat] = v1.build_template_contour_model(
            cat, geometry["block_px"], geometry["connector_px"], 1.5)
    m = models[cat]
    rx, ry, _ = m.rotated(angle)
    pts = np.stack((rx + px, ry + py), axis=1).astype(np.int32)
    cv2.polylines(vis, [pts], True, color, 1, cv2.LINE_AA)


for i in CASES:
    r = rows[i]
    cat = r["category"]
    vr = v1rows[i]
    b = vr["detection_box"]
    x1, y1, x2, y2 = (int(v) for v in b)
    cx1, cy1 = max(0, x1 - CROP_MARGIN), max(0, y1 - CROP_MARGIN)
    cx2 = min(img.shape[1], x2 + CROP_MARGIN)
    cy2 = min(img.shape[0], y2 + CROP_MARGIN)
    roi = img[cy1:cy2, cx1:cx2]

    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 三个方法各出一张独立图，避免轮廓互相遮挡
    a = r["A"]
    c = r["C"]
    for tag, color, pose, note in (
        ("V1", (0, 255, 255), (r["v1_px"], r["v1_py"], r["v1_angle"]), "参照(V1全流程)"),
        ("A", (0, 255, 0), (a["px"], a["py"], a["angle"]), f"A 边缘x Canny  d={a['d_pos']:.1f}px"),
        ("C", (0, 0, 255), (c["px"], c["py"], c["angle"]), f"C 边缘x 距离场  d={c['d_pos']:.1f}px"),
    ):
        one = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        px, py, ang = pose
        draw_pose(one, cat, px - cx1, py - cy1, ang, color)
        cv2.circle(one, (int(px) - cx1, int(py) - cy1), 3, color, 1)
        cv2.putText(one, note, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (255, 255, 255), 1, cv2.LINE_AA)
        out = os.path.join(outdir, f"case{i:02d}_{cat}_{tag}.png")
        cv2.imwrite(out, one)

    print(f"case{i:02d} {cat:<9s} dA={a['d_pos']:.1f} dC={c['d_pos']:.1f}")
