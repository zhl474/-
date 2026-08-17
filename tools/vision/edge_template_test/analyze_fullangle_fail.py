# -*- coding: utf-8 -*-
"""解剖 seg+canny+全角度 匹配的失败模式。

对每个 S2_full 偏差>5px 的块：
  A. 错误位姿（全角度argmax选出）的分数分解：贴边项 / seg重叠项 / 总分
  B. 真角度（离V1角最近的网格角）下的最优位置分数分解
  C. 判定：错误总分 > 真角度总分 → 打分函数被什么勾走（看哪一项高）
        错误总分 < 真角度总分 → 搜索/网格问题
并出图：错误位姿 vs 真角度位姿（各一张，底图=seg绿+筛后Canny白）。
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
import seg_canny_test as sct
from ultralytics import YOLO

# ---------------- 参数区 ----------------
IMAGE_PATH = os.path.join(SRC, "tools/vision/7.png")
V1_REF_JSON = os.path.join(SRC, "tools/vision/oriented_chamfer_v1/output/v1_results.json")
SEG_MODEL_PATH = os.path.join(SRC, "competition", "model", "best_seg.pt")
OUTPUT_DIR = os.path.join(THIS, "output")
BAD_THRESH_PX = 5.0
# ----------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

    yolo = YOLO(v1.DETECTION_MODEL_PATH)
    yolo(img, verbose=False)
    seg_model = YOLO(SEG_MODEL_PATH, task="segment")
    seg_model(img[:64, :64], verbose=False)
    dets = v1.detect_blocks_yolo_v1(img, yolo, v1.ChamferV1Config())
    v1refs = [r for r in json.load(open(V1_REF_JSON))["results"] if r["found"]]

    print(f"{'case':<6s}{'类别':<10s}{'错误角':>7s}{'真角':>7s}{'d':>6s}"
          f"  {'错误位姿: 贴边/重叠/总分':>26s}  {'真角最优: 贴边/重叠/总分':>26s}  判定")
    n_obj, n_search = 0, 0
    for i, (det, vr) in enumerate(zip(dets, v1refs)):
        cat = det["category"]
        prev = json.load(open(os.path.join(OUTPUT_DIR, "seg_canny_results.json")))[i]
        if prev["S2_full"]["d_pos"] <= BAD_THRESH_PX:
            continue
        x1, y1, x2, y2 = det["box"]
        cx1, cy1 = max(0, int(x1) - sct.CROP_MARGIN), max(0, int(y1) - sct.CROP_MARGIN)
        cx2 = min(img.shape[1], int(x2) + sct.CROP_MARGIN)
        cy2 = min(img.shape[0], int(y2) + sct.CROP_MARGIN)
        crop = img[cy1:cy2, cx1:cx2]

        seg = sct.get_seg_mask(seg_model, crop)
        seg_f = (seg > 0).astype(np.float32)
        edges_roi = edges_full[cy1:cy2, cx1:cx2] > 0
        keep = edges_roi & (cv2.dilate(
            seg, np.ones((2 * sct.SEG_FILTER_DILATE_PX + 1,) * 2, np.uint8)) > 0)
        e_f = keep.astype(np.uint8) * 255
        A_e = (sct.DT_CAP - np.minimum(
            cv2.distanceTransform(255 - e_f, cv2.DIST_L2, 3), sct.DT_CAP)).astype(np.float32)

        period = float(ROTATION_TOTAL_ANGLE[cat])
        angles = sorted(meta[cat])

        line_k, ksize, anchors = ect.build_edge_kernels(meta[cat], angles, 1, DEVICE)
        fill_k, _, _, fill_counts = sct.build_filled_kernels(cat, angles, DEVICE)

        th = max(A_e.shape[0], ksize[0] + 2 * sct.MARGIN)
        tw = max(A_e.shape[1], ksize[1] + 2 * sct.MARGIN)
        pt_, pl = (th - A_e.shape[0]) // 2, (tw - A_e.shape[1]) // 2
        Ep = np.zeros((th, tw), dtype=np.float32)
        Sp = np.zeros((th, tw), dtype=np.float32)
        Ep[pt_:pt_ + A_e.shape[0], pl:pl + A_e.shape[1]] = A_e
        Sp[pt_:pt_ + A_e.shape[0], pl:pl + A_e.shape[1]] = seg_f
        lc = line_k.sum(dim=(1, 2, 3)).clamp(min=1.0).to(DEVICE)
        fc = torch.from_numpy(fill_counts).to(DEVICE)[:, None, None].clamp(min=1.0)
        with torch.no_grad():
            f_line = F.conv2d(sct.to_t(Ep), line_k)[0] / lc[:, None, None] / sct.DT_CAP
            f_fill = F.conv2d(sct.to_t(Sp), fill_k)[0] / fc
        total = sct.W_CANNY * f_line + sct.W_SEG * f_fill

        # 全角度最优（错误位姿）
        best = None
        for j, ang in enumerate(angles):
            idx = int(torch.argmax(total[j]))
            iy, ix = np.unravel_index(idx, total[j].shape)
            s = float(total[j, iy, ix])
            if best is None or s > best[0]:
                ax, ay = anchors[j]
                best = (s, j, float(ix + ax - pl) + cx1, float(iy + ay - pt_) + cy1,
                        float(f_line[j, iy, ix]), float(f_fill[j, iy, ix]))
        # 真角度索引
        j_true = min(range(len(angles)),
                     key=lambda j: min(abs(angles[j] - vr["pose_angle_deg"]) % period,
                                       period - abs(angles[j] - vr["pose_angle_deg"]) % period))
        idx = int(torch.argmax(total[j_true]))
        iy, ix = np.unravel_index(idx, total[j_true].shape)
        s_true = float(total[j_true, iy, ix])
        ax, ay = anchors[j_true]
        true_pose = (float(ix + ax - pl) + cx1, float(iy + ay - pt_) + cy1, angles[j_true],
                     float(f_line[j_true, iy, ix]), float(f_fill[j_true, iy, ix]))

        verdict = "打分偏好错误位姿" if best[0] > s_true else "搜索/网格问题"
        if verdict.startswith("打分"):
            n_obj += 1
        else:
            n_search += 1
        print(f"[{i:02d}] {cat:<9s}{angles[best[1]]:7.0f}{angles[j_true]:7.0f}"
              f"{prev['S2_full']['d_pos']:6.1f}  "
              f"{best[4]:5.2f}/{best[5]:5.2f}/{best[0]:6.2f}  "
              f"{true_pose[3]:5.2f}/{true_pose[4]:5.2f}/{s_true:6.2f}  {verdict}"
              f"  真角下重叠率={true_pose[4]:.2f}")

        # 出图
        segvis = (crop * 0.45).astype(np.uint8)
        m = seg > 0
        ov = segvis.copy()
        ov[m] = (0, 80, 0)
        segvis = cv2.addWeighted(segvis, 0.5, ov, 0.5, 0)
        segvis[keep] = (255, 255, 255)

        for tag, pose, note in (
            ("wrong", (best[2], best[3], angles[best[1]]),
             f"WRONG a={angles[best[1]]:.0f} line={best[4]:.2f} fill={best[5]:.2f}"),
            ("true", (true_pose[0], true_pose[1], true_pose[2]),
             f"TRUE a={true_pose[2]:.0f} line={true_pose[3]:.2f} fill={true_pose[4]:.2f}"),
            ("v1", (vr["px"], vr["py"], vr["pose_angle_deg"]), "V1 ref"),
        ):
            one = segvis.copy()
            rx, ry, _ = models[cat].rotated(pose[2])
            pts = np.stack((rx + pose[0] - cx1, ry + pose[1] - cy1), axis=1).astype(np.int32)
            cv2.polylines(one, [pts], True, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(one, (int(pose[0]) - cx1, int(pose[1]) - cy1), 3, (0, 0, 255), 1)
            cv2.putText(one, note, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"an{i:02d}_{cat}_{tag}.png"), one)

    print(f"\n判定统计: 打分偏好错误位姿 {n_obj} / 搜索问题 {n_search}")
    print(f"图: {OUTPUT_DIR}/an{{NN}}_{{cat}}_{{wrong|true|v1}}.png")


if __name__ == "__main__":
    main()
