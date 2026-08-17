# -*- coding: utf-8 -*-
"""
边缘模板 + 现有卷积匹配的可行性实验（离线，不动正式代码）。

思路：ROI 只取 YOLO bbox + crop_margin（与正式裁剪一致），
模板不用实心 mask 而用外轮廓线条，直接调正式
template_match.match_template 的 F.conv2d 选优。

三个变体：
  A  1px 线模板 × Canny 二值图      （原始思路：线线重叠计数）
  B  3px 粗线模板 × Canny 二值图    （A 的平滑版，缓解尖刺）
  C  1px 线模板 × 距离变换图        （conv 版 chamfer，理论等价 V1 距离项）

参照真值 = oriented_chamfer_v1 在同图上的 v1_results.json。
"""
import os
import sys
import time

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

# ---------------- 参数区（直接改这里） ----------------
IMAGE_PATH = "/home/zhl/SingleArmTetris/SingleArmTetris/src/tools/vision/7.png"
V1_REF_JSON = "/home/zhl/SingleArmTetris/SingleArmTetris/src/tools/vision/oriented_chamfer_v1/output/v1_results.json"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

CROP_MARGIN = 8          # 与正式 image_node 的 crop_margin 一致
CANNY_LOW = 50.0
CANNY_HIGH = 150.0
GAUSS_K = 3
ANGLE_STEP = 2.0         # 与 V1 粗搜一致
LINE_THICK_A = 1         # 变体A/B 模板线宽
LINE_THICK_B = 3
DT_CAP = 20.0            # 变体C 距离截断，与 V1 distance_cap 一致
KERNEL_MARGIN = 2        # 核画布四周留边，防止粗线被裁
SCREEN_TOL_PX = 6        # bbox 宽高筛角度容差；None=不筛
DETECTION_CONF = 0.45
DETECTION_IOU = 0.5
TEMPLATE_PROFILE = "high"
# ------------------------------------------------------

SRC = "/home/zhl/SingleArmTetris/SingleArmTetris/src"
for p in (SRC, os.path.join(SRC, "image_process")):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np
import torch

from image_process_lib.template_match.kernels_create import (
    ROTATION_TOTAL_ANGLE,
    build_angle_foreground_metadata,
)
from image_process_lib.template_match.template_match import match_template
from image_process_lib.template_config import load_template_geometry

sys.path.insert(0, os.path.join(SRC, "tools/vision/oriented_chamfer_v1"))
import oriented_chamfer_v1 as v1  # noqa: E402

from ultralytics import YOLO  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_edge_kernels(metadata, angles, thickness, device):
    """把每个角度的实心紧边框模板转成外轮廓线核，统一放进公共画布。

    返回 kernels (N,1,H,W)、kernel_size、anchors（旋转中心相对公共画布左上角）。
    语义与 create_screened_kernels 一致，只是模板内容从实心换成轮廓线。
    """
    binary_list = [metadata[a]["binary"] for a in angles]
    max_h = max(b.shape[0] for b in binary_list) + 2 * KERNEL_MARGIN
    max_w = max(b.shape[1] for b in binary_list) + 2 * KERNEL_MARGIN
    kernels = np.zeros((len(angles), 1, max_h, max_w), dtype=np.float32)
    anchors = []
    for i, (angle, binary) in enumerate(zip(angles, binary_list)):
        canvas = np.zeros((max_h, max_w), dtype=np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, contours, -1, 255, thickness,
                         offset=(KERNEL_MARGIN, KERNEL_MARGIN))
        kernels[i, 0] = (canvas > 0).astype(np.float32)
        ax, ay = metadata[angle]["anchor"]
        anchors.append((ax + KERNEL_MARGIN, ay + KERNEL_MARGIN))
    tensor = torch.from_numpy(kernels)
    if device == "cuda":
        tensor = tensor.to(torch.float16)
    return tensor.to(device), (max_h, max_w), anchors


def make_image_tensor(data2d, device):
    """(H,W) numpy → (1,1,H,W) tensor。不缩放，argmax 语义不变。"""
    t = torch.from_numpy(np.ascontiguousarray(data2d, dtype=np.float32))[None, None]
    if device == "cuda":
        t = t.to(torch.float16)
    return t.to(device)


def run_variant(name, image_data, metadata, angles, crop_xy, box_wh, screen_tol):
    """跑一个变体：筛角度 → 建核 → 补边 → conv 选优 → 返回全图中心与角度。"""
    used_angles = angles
    if screen_tol is not None:
        bw, bh = box_wh
        cand = [a for a in angles
                if abs(metadata[a]["fg_w"] - bw) <= screen_tol
                and abs(metadata[a]["fg_h"] - bh) <= screen_tol]
        if len(cand) >= 3:
            used_angles = cand
    t0 = time.perf_counter()
    thickness = LINE_THICK_B if name == "B" else LINE_THICK_A
    kernels, ksize, anchors = build_edge_kernels(metadata, used_angles, thickness, DEVICE)
    # 与正式 compute_screened_input_padding 同思路：补零边保证核放得下 + 平移余量。
    max_h, max_w = ksize
    margin = 4
    target_h = max(image_data.shape[0], max_h + 2 * margin)
    target_w = max(image_data.shape[1], max_w + 2 * margin)
    pad_top = (target_h - image_data.shape[0]) // 2
    pad_left = (target_w - image_data.shape[1]) // 2
    padded = np.zeros((target_h, target_w), dtype=np.float32)
    padded[pad_top:pad_top + image_data.shape[0], pad_left:pad_left + image_data.shape[1]] = image_data
    image_tensor = make_image_tensor(padded, DEVICE)
    _, pos = match_template(image_tensor, kernels, ksize, used_angles, anchors=anchors,
                            use_same_padding=False)
    elapsed = (time.perf_counter() - t0) * 1000.0
    ax, ay = pos["anchor"]
    cx = crop_xy[0] + (pos["x"] - pad_left) + ax
    cy = crop_xy[1] + (pos["y"] - pad_top) + ay
    return {
        "angle": float(pos["angle"]), "px": float(cx), "py": float(cy),
        "score": float(pos["score"]), "n_angles": len(used_angles), "ms": elapsed,
    }


def angle_diff(angle, ref, period):
    d = (angle - ref) % period
    return min(d, period - d)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img = cv2.imread(IMAGE_PATH)
    geometry = load_template_geometry(TEMPLATE_PROFILE)
    block_px, connector_px = geometry["block_px"], geometry["connector_px"]

    yolo = YOLO(v1.DETECTION_MODEL_PATH)
    yolo(img, verbose=False)
    config = v1.ChamferV1Config()
    dets = v1.detect_blocks_yolo_v1(img, yolo, config)
    print(f"YOLO: {len(dets)} 个目标   device={DEVICE}")

    with open(V1_REF_JSON, encoding="utf-8") as f:
        v1_refs = [r for r in __import__("json").load(f)["results"] if r["found"]]

    model_cache = {}
    rows = []
    for det in dets:
        cat = det["category"]
        x1, y1, x2, y2 = det["box"]
        cx1 = max(0, int(x1) - CROP_MARGIN)
        cy1 = max(0, int(y1) - CROP_MARGIN)
        cx2 = min(img.shape[1], int(x2) + CROP_MARGIN)
        cy2 = min(img.shape[0], int(y2) + CROP_MARGIN)
        roi = img[cy1:cy2, cx1:cx2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (GAUSS_K, GAUSS_K), 0)
        edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
        edges_f = (edges > 0).astype(np.float32)
        # 变体C：match_template 取 argmax，所以用 (cap - 距离)，最大化它 = 最小化线上距离和
        dt = (DT_CAP - np.minimum(
            cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3), DT_CAP
        )).astype(np.float32)

        if cat not in model_cache:
            model_cache[cat] = build_angle_foreground_metadata(
                cat, block_px, connector_px, ANGLE_STEP)
        metadata = model_cache[cat]
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
            print(f"!! {cat} 找不到 V1 参照，跳过")
            continue

        row = {"category": cat, "v1_px": ref["px"], "v1_py": ref["py"],
               "v1_angle": ref["pose_angle_deg"]}
        for name, data, tol in (
            ("A", edges_f, SCREEN_TOL_PX),
            ("B", edges_f, SCREEN_TOL_PX),
            ("C", dt, SCREEN_TOL_PX),
            ("A_full", edges_f, None),   # 不筛角度，检验 bbox 筛角是否误伤
        ):
            try:
                out = run_variant(name, data, metadata, angles, (cx1, cy1),
                                  (int(x2 - x1), int(y2 - y1)), tol)
                out["d_pos"] = float(np.hypot(out["px"] - ref["px"], out["py"] - ref["py"]))
                out["d_angle"] = float(angle_diff(out["angle"], ref["pose_angle_deg"], period))
                row[name] = out
            except Exception as exc:
                row[name] = {"error": str(exc)}
        rows.append(row)
        parts = []
        for name in ("A", "B", "C", "A_full"):
            if "error" in row[name]:
                parts.append(f"{name}:ERR")
            else:
                parts.append(f"{name}: d={row[name]['d_pos']:.1f}px "
                             f"da={row[name]['d_angle']:.1f}deg "
                             f"({row[name]['n_angles']}ang {row[name]['ms']:.0f}ms)")
        print(f"{cat:<10s} " + "  ".join(parts))

    # ---------------- 汇总 ----------------
    print("\n===== 汇总（相对 V1 参照的偏差） =====")
    print(f"{'变体':<4s} {'中位Δpx':>8s} {'均值Δpx':>8s} {'<2px':>6s} {'<5px':>6s} "
          f"{'中位Δ角':>8s} {'<2°':>6s} {'均耗时':>8s} {'平均角度数':>8s}")
    for name in ("A", "B", "C", "A_full"):
        ok = [r[name] for r in rows if "error" not in r.get(name, {"error": 1})]
        if not ok:
            print(f"{name:<4s} 全部失败")
            continue
        d = sorted(o["d_pos"] for o in ok)
        da = sorted(o["d_angle"] for o in ok)
        n = len(ok)
        print(f"{name:<4s} {d[n//2]:8.2f} {np.mean(d):8.2f} "
              f"{sum(x < 2 for x in d)/n:5.0%} {sum(x < 5 for x in d)/n:5.0%} "
              f"{da[n//2]:8.2f} {sum(x < 2 for x in da)/n:5.0%} "
              f"{np.mean([o['ms'] for o in ok]):7.1f}ms "
              f"{np.mean([o['n_angles'] for o in ok]):8.1f}")

    import json
    with open(os.path.join(OUTPUT_DIR, "edge_conv_results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nJSON: {os.path.join(OUTPUT_DIR, 'edge_conv_results.json')}")


if __name__ == "__main__":
    main()
