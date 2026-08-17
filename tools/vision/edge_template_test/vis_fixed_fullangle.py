# -*- coding: utf-8 -*-
"""居中核+全角度修正版的逐块结果图：fn{NN}_{cat}_fixed.png（黄线=匹配位姿，底图=Canny）。"""
import json, os, sys
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
SRC = "/home/zhl/SingleArmTetris/SingleArmTetris/src"
THIS = os.path.dirname(os.path.abspath(__file__))
for p in (SRC, os.path.join(SRC, "image_process"), os.path.join(SRC, "tools/vision/oriented_chamfer_v1"), THIS):
    if p not in sys.path: sys.path.insert(0, p)
import cv2, numpy as np
from image_process_lib.template_match.kernels_create import ROTATION_TOTAL_ANGLE
from image_process_lib.template_config import load_template_geometry
import oriented_chamfer_v1 as v1
import fullangle_fixed_test as fft

img = cv2.imread(fft.IMAGE_PATH)
geo = load_template_geometry("high")
models = {c: v1.build_template_contour_model(c, geo["block_px"], geo["connector_px"], 1.5)
          for c in set(ROTATION_TOTAL_ANGLE)}
rows = json.load(open(os.path.join(THIS, "output", "fullangle_fixed_results.json")))
gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
edges = cv2.Canny(gray, 50, 150)
for r in rows:
    i, cat = r["index"], r["category"]
    vr = r["v1"]
    x1, y1, x2, y2 = [int(v) for v in
                      (vr[0] - 60, vr[1] - 60, vr[0] + 60, vr[1] + 60)]
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(img.shape[1], x2), min(img.shape[0], y2)
    base = cv2.cvtColor(edges[cy1:cy2, cx1:cx2], cv2.COLOR_GRAY2BGR)
    o = r["best"]
    rx, ry, _ = models[cat].rotated(o["angle"])
    pts = np.stack((rx + o["px"] - cx1, ry + o["py"] - cy1), axis=1).astype(np.int32)
    cv2.polylines(base, [pts], True, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.circle(base, (int(o["px"]) - cx1, int(o["py"]) - cy1), 3, (0, 0, 255), 1)
    cv2.putText(base, f"FULLANGLE a={o['angle']:.0f} d={o['d_pos']:.1f}px da={o['d_angle']:.1f}deg",
                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(THIS, "output", f"fn{i:02d}_{cat}_fixed.png"), base)
print("35张 fn*_fixed.png 已存 output/")
