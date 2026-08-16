# -*- coding: utf-8 -*-
"""
V1 离线测试入口（不接入正式识别流程）。

默认读 competition/model/best5.14.pt 和
competition/config/template_config.yaml，但不会 import image_node /
block_detection 等带 TensorRT / ROS 依赖的正式运行代码。

用法示例：
  cd /home/zhl/SingleArmTetris/SingleArmTetris/src
  PYTHONPATH=image_process python3 tools/vision/oriented_chamfer_v1/run_v1_test.py \
      --image tools/vision/7.png \
      --output-dir /tmp/oriented_chamfer_v1

  # 只测一张图里的 T
  PYTHONPATH=image_process python3 tools/vision/oriented_chamfer_v1/run_v1_test.py \
      --image tools/vision/7.png --category T

  # 手动指定一个框，跳过 YOLO
  PYTHONPATH=image_process python3 tools/vision/oriented_chamfer_v1/run_v1_test.py \
      --image tools/vision/7.png --category T --box 969 27 1101 134

  # 对比普通 Chamfer（关方向项）
  ... --orientation-weight 0
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

_THIS_FILE = os.path.abspath(__file__)
_THIS_DIR = os.path.dirname(_THIS_FILE)
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
_IMAGE_PROCESS_DIR = os.path.join(_SRC_DIR, "image_process")
if _IMAGE_PROCESS_DIR not in sys.path:
    sys.path.insert(0, _IMAGE_PROCESS_DIR)

from oriented_chamfer_v1 import (  # noqa: E402
    DETECTION_MODEL_PATH,
    ChamferV1Config,
    detect_blocks_yolo_v1,
    load_geometry,
    match_block_v1,
)

# 防止只读 /home 下 ultralytics 写 settings 报错；比赛机可写，这里 setdefault 不覆盖。
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

from ultralytics import YOLO  # noqa: E402


def _build_config(args) -> ChamferV1Config:
    config = ChamferV1Config()
    config.roi_context_margin = args.roi_context_margin
    config.canny_low = args.canny_low
    config.canny_high = args.canny_high
    config.orientation_enabled = args.orientation_weight > 0
    config.orientation_weight = args.orientation_weight
    config.orientation_radius = args.orientation_radius
    config.center_prior_weight = args.center_prior_weight
    config.coarse_angle_step = args.coarse_angle_step
    config.coarse_xy_step = args.coarse_xy_step
    config.coarse_search_radius = args.coarse_search_radius
    config.coarse_top_k = args.coarse_top_k
    config.fine_angle_step = args.fine_angle_step
    config.fine_angle_window = args.fine_angle_window
    config.fine_xy_step = args.fine_xy_step
    config.fine_xy_radius = args.fine_xy_radius
    config.detection_conf = args.detection_conf
    config.detection_iou = args.detection_iou
    config.template_profile = args.profile
    return config


def main():
    parser = argparse.ArgumentParser(description="V1 Oriented Chamfer 顶面配准离线测试")
    parser.add_argument("--image", required=True, help="输入高位原图")
    parser.add_argument("--category", default="", help="只测该类别；默认全部 YOLO 检测")
    parser.add_argument("--box", nargs=4, type=float, default=None, help="手动指定框 x1 y1 x2 y2，跳过 YOLO")
    parser.add_argument("--max-blocks", type=int, default=0, help="最多测几个框，0=全部")
    parser.add_argument("--profile", default="high", choices=["high", "low"], help="模板像素尺寸 profile")
    parser.add_argument("--output-dir", default=os.path.join(_THIS_DIR, "output"), help="调试图/JSON 输出目录")

    parser.add_argument("--detection-conf", type=float, default=0.45)
    parser.add_argument("--detection-iou", type=float, default=0.5)
    parser.add_argument("--roi-context-margin", type=float, default=6.0)
    parser.add_argument("--canny-low", type=float, default=50.0)
    parser.add_argument("--canny-high", type=float, default=150.0)

    parser.add_argument("--orientation-weight", type=float, default=0.8, help="0 时关闭方向一致性项")
    parser.add_argument("--orientation-radius", type=float, default=4.0)
    parser.add_argument("--center-prior-weight", type=float, default=0.005)

    parser.add_argument("--coarse-angle-step", type=float, default=2.0)
    parser.add_argument("--coarse-xy-step", type=float, default=1.0)
    parser.add_argument("--coarse-search-radius", type=float, default=14.0)
    parser.add_argument("--coarse-top-k", type=int, default=5)
    parser.add_argument("--fine-angle-step", type=float, default=0.2)
    parser.add_argument("--fine-angle-window", type=float, default=2.5)
    parser.add_argument("--fine-xy-step", type=float, default=0.5)
    parser.add_argument("--fine-xy-radius", type=float, default=3.0)

    args = parser.parse_args()
    image_path = os.path.abspath(args.image)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"读不到图片: {image_path}")

    config = _build_config(args)
    geometry = load_geometry(config.template_profile)
    print(f"模板几何 {config.template_profile}: {geometry}")
    print(f"输入图片: {image_path}  {img_bgr.shape[1]}x{img_bgr.shape[0]}")
    print(f"Canny: {config.canny_low:.0f}/{config.canny_high:.0f}   "
          f"orientation_weight={config.orientation_weight}   "
          f"center_prior_weight={config.center_prior_weight}")

    if args.box is not None:
        category = args.category.strip()
        if not category:
            raise ValueError("手动 --box 时必须同时给 --category")
        x1, y1, x2, y2 = args.box
        if x2 <= x1 or y2 <= y1:
            raise ValueError("box 必须满足 x2 > x1 且 y2 > y1")
        detections = [{"category": category, "score": 1.0, "box": tuple(args.box)}]
        print(f"手动框模式: {category} {tuple(round(v, 1) for v in args.box)}")
    else:
        detection_model = YOLO(DETECTION_MODEL_PATH)
        detections = detect_blocks_yolo_v1(img_bgr, detection_model, config, args.category)
        print(f"YOLO 检测到 {len(detections)} 个目标")
        for index, det in enumerate(detections, start=1):
            x1, y1, x2, y2 = det["box"]
            print(f"  [{index:02d}] {det['category']:<10s} conf={det['score']:.3f}  "
                  f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

    if args.max_blocks and args.max_blocks > 0:
        detections = detections[: args.max_blocks]

    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    annotated_full = img_bgr.copy()
    global_started = time.perf_counter()

    for index, det in enumerate(detections, start=1):
        category = det["category"]
        box = det["box"]
        print(f"\n=== [{index:02d}/{len(detections):02d}] {category} "
              f"box=({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}) ===")
        try:
            result = match_block_v1(
                img_bgr,
                category,
                box,
                config=config,
                geometry=geometry,
                debug=True,
            )
        except Exception as exc:
            print(f"  !! 匹配失败: {exc}")
            results.append({
                "index": index,
                "category": category,
                "found": False,
                "error": str(exc),
                "detection_box": tuple(float(v) for v in box),
            })
            continue

        comp = result["score_components"]
        print(
            f"  px={result['px']:.2f}  py={result['py']:.2f}  "
            f"pose_angle={result['pose_angle_deg']:.2f}  theta={result['theta']:.2f}"
        )
        print(
            f"  score={result['score']:.3f}  "
            f"dist={comp['dist_cost']:.3f}  "
            f"ori={comp['ori_cost']:.3f}  "
            f"center={comp['center_cost']:.3f}  "
            f"cov2px={result['coverage_2px']:.3f}"
        )
        print(
            f"  roi={result['roi_box']}  "
            f"coarse={result['timing_ms']['coarse']:.0f}ms  "
            f"fine={result['timing_ms']['fine']:.0f}ms"
        )

        # 保存每块调试图。
        debug = result["debug"]
        full_vis = debug["full_vis"]
        roi_vis = debug["roi_edges"]
        prefix = f"{index:02d}_{category}_{result['px']:.0f}_{result['py']:.0f}"
        cv2.imwrite(os.path.join(args.output_dir, prefix + "_roi_edges.png"), roi_vis)
        # 全图局部调试仅裁 ROI 附近，避免每块保存一次大图。
        x1, y1, x2, y2 = result["roi_box"]
        cv2.imwrite(
            os.path.join(args.output_dir, prefix + "_full_crop.png"),
            full_vis[max(0, y1 - 15):min(img_bgr.shape[0], y2 + 15),
                     max(0, x1 - 15):min(img_bgr.shape[1], x2 + 15)],
        )

        # 汇总图叠加当前结果。
        _draw_result_on_full(annotated_full, result)

        serializable = {
            "index": index,
            "category": result["category"],
            "found": bool(result["found"]),
            "px": float(result["px"]),
            "py": float(result["py"]),
            "pose_angle_deg": float(result["pose_angle_deg"]),
            "theta": float(result["theta"]),
            "score": float(result["score"]),
            "score_components": {k: float(v) for k, v in comp.items()},
            "coverage_2px": float(result["coverage_2px"]),
            "mean_dist_raw": float(result["mean_dist_raw"]),
            "detection_box": result["detection_box"],
            "roi_box": result["roi_box"],
            "roi_local": {k: float(v) for k, v in result["roi_local"].items()},
            "rect_size": result["rect_size"],
            "timing_ms": {k: float(v) for k, v in result["timing_ms"].items()},
            "coarse_candidates": result["coarse_candidates"],
        }
        results.append(serializable)

    total_ms = (time.perf_counter() - global_started) * 1000.0
    cv2.imwrite(os.path.join(args.output_dir, "summary_full.png"), annotated_full)

    manifest = {
        "image": image_path,
        "image_size": [int(img_bgr.shape[1]), int(img_bgr.shape[0])],
        "config": {
            "template_profile": config.template_profile,
            "canny_low": config.canny_low,
            "canny_high": config.canny_high,
            "orientation_weight": config.orientation_weight,
            "orientation_radius": config.orientation_radius,
            "center_prior_weight": config.center_prior_weight,
            "coarse_angle_step": config.coarse_angle_step,
            "coarse_xy_step": config.coarse_xy_step,
            "coarse_search_radius": config.coarse_search_radius,
            "coarse_top_k": config.coarse_top_k,
            "fine_angle_step": config.fine_angle_step,
            "fine_angle_window": config.fine_angle_window,
            "fine_xy_step": config.fine_xy_step,
            "fine_xy_radius": config.fine_xy_radius,
            "detection_conf": config.detection_conf,
            "detection_iou": config.detection_iou,
        },
        "total_ms": float(total_ms),
        "results": results,
    }
    manifest_path = os.path.join(args.output_dir, "v1_results.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n完成 {len(results)} 个目标，总耗时 {total_ms:.0f} ms")
    print(f"JSON: {manifest_path}")
    print(f"汇总图: {os.path.join(args.output_dir, 'summary_full.png')}")
    print(f"每块调试图: {args.output_dir}")


def _draw_result_on_full(image, result):
    """把单个 V1 结果画到汇总全图上（只调试图，不影响结果序列化）。"""
    model = result["model"]
    rx, ry, _ = model.rotated(result["pose_angle_deg"])
    pts = np.stack((rx + result["px"], ry + result["py"]), axis=1).astype(np.int32)
    cv2.polylines(image, [pts], isClosed=True, color=(0, 220, 0), thickness=1, lineType=cv2.LINE_AA)
    px = int(round(result["px"]))
    py = int(round(result["py"]))
    cv2.circle(image, (px, py), 2, (0, 0, 255), -1)
    x1, y1, x2, y2 = result["roi_box"]
    cv2.rectangle(image, (x1, y1), (x2, y2), (200, 120, 0), 1)
    cv2.putText(
        image,
        f"{result['category']} a={result['pose_angle_deg']:.1f}",
        (x1, max(12, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )


if __name__ == "__main__":
    main()
