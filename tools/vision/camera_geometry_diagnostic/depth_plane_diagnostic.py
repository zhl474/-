#!/home/zhl/fr3env/fr3env/bin/python
"""Gemini 335 对齐深度的相机坐标平面性诊断。

本脚本独占相机但不移动机械臂。每个距离单独运行一次，并修改下方距离标签。
"""

import csv
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parents[2]
TOOLS_VISION_DIR = SRC_ROOT / "tools" / "vision"
for path in (SCRIPT_DIR, TOOLS_VISION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diagnostic_core import (  # noqa: E402
    detect_asymmetric_circles,
    draw_circle_order,
    fit_plane_svd,
    make_board_material_masks,
    plane_quadratic_cross_validation,
    save_image,
    to_builtin,
    validate_image_size,
)


# ==================== 运行参数（直接修改本文件）====================
CAMERA_CONFIG_PATH = SRC_ROOT / "camera" / "config" / "新相机参数.yaml"
OUTPUT_ROOT = Path.home() / "桌面" / "相机几何诊断" / "深度平面"
EXPECTED_IMAGE_SIZE = (1280, 720)  # (宽, 高)
PATTERN_SIZE = (4, 7)

# 每个距离分别运行；可用 high_330mm / near_170mm / control_500mm。
TEST_LABEL = "high_330mm"
APPROX_DISTANCE_MM = 330.0
WARMUP_FRAMES = 20
CAPTURE_FRAMES = 30
MIN_VALID_FRAMES_PER_PIXEL = 20
SAMPLE_STRIDE_PX = 5
MAX_BLACK_SAMPLE_POINTS = 18000
MAX_WHITE_SAMPLE_POINTS = 6000
WHITE_RADIUS_RATIO = 0.28
BOARD_EDGE_MARGIN_PX = 8
RANDOM_SEED = 42
CV_FOLDS = 5

MIN_BLACK_VALID_RATIO = 0.70
MAX_MATERIAL_RESIDUAL_DIFFERENCE_MM = 1.0
PROJECT_MAX_PLANE_RMSE_MM = 1.0
PROJECT_MAX_PLANE_P95_MM = 2.0
CURVE_IMPROVEMENT_WARNING_RATIO = 0.20


def _configure_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _capture_frames(camera):
    for index in range(max(0, int(WARMUP_FRAMES))):
        color, depth = camera.read()
        if color is None or depth is None:
            print(f"预热帧 {index + 1}/{WARMUP_FRAMES} 不完整。")
    colors = []
    depths = []
    attempts = 0
    max_attempts = max(100, int(CAPTURE_FRAMES) * 5)
    while len(depths) < int(CAPTURE_FRAMES) and attempts < max_attempts:
        attempts += 1
        color, depth = camera.read()
        if color is None or depth is None:
            continue
        validate_image_size(color, EXPECTED_IMAGE_SIZE)
        validate_image_size(depth, EXPECTED_IMAGE_SIZE)
        if color.ndim != 3 or color.shape[2] != 3 or depth.ndim != 2:
            raise RuntimeError(f"帧格式异常：彩色={color.shape}，深度={depth.shape}")
        colors.append(color.copy())
        depths.append(depth.copy())
        print(f"已采集深度帧 {len(depths)}/{CAPTURE_FRAMES}")
    if len(depths) != int(CAPTURE_FRAMES):
        raise RuntimeError(f"深度帧不足：实际 {len(depths)}/{CAPTURE_FRAMES}")
    return colors, np.stack(depths, axis=0)


def _temporal_median(depth_stack):
    values = np.asarray(depth_stack, dtype=np.float32)
    valid = np.isfinite(values) & (values > 0.0)
    masked = np.where(valid, values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(masked, axis=0)
    valid_counts = np.sum(valid, axis=0)
    required = int(MIN_VALID_FRAMES_PER_PIXEL)
    if required <= 0 or required > values.shape[0]:
        raise ValueError(
            f"MIN_VALID_FRAMES_PER_PIXEL 必须在 1 到 {values.shape[0]} 之间"
        )
    median[valid_counts < required] = np.nan
    return median, valid_counts


def _valid_ratio(mask, median_depth):
    pixel_count = int(np.count_nonzero(mask))
    if pixel_count == 0:
        return 0.0
    valid = mask & np.isfinite(median_depth) & (median_depth > 0.0)
    return float(np.count_nonzero(valid) / pixel_count)


def _sample_pixels(mask, median_depth, stride, maximum, seed):
    valid = mask & np.isfinite(median_depth) & (median_depth > 0.0)
    ys, xs = np.nonzero(valid)
    grid = (xs % max(1, int(stride)) == 0) & (ys % max(1, int(stride)) == 0)
    xs = xs[grid]
    ys = ys[grid]
    if len(xs) > int(maximum):
        rng = np.random.default_rng(int(seed))
        chosen = rng.choice(len(xs), size=int(maximum), replace=False)
        xs = xs[chosen]
        ys = ys[chosen]
    return np.column_stack([xs, ys]).astype(np.int32)


def _deproject(camera, pixels, median_depth, minimum_points=30):
    points = []
    records = []
    for x, y in np.asarray(pixels, dtype=int):
        depth = float(median_depth[y, x])
        try:
            point = np.asarray(
                camera.depth_pixel2cam_point3d(int(x), int(y), depth_value=depth),
                dtype=float,
            ).reshape(-1)
        except Exception:  # noqa: BLE001
            continue
        if point.size != 3 or not np.all(np.isfinite(point)):
            continue
        points.append(point)
        records.append([int(x), int(y), depth, float(point[0]), float(point[1]), float(point[2])])
    if len(points) < int(minimum_points):
        raise RuntimeError(
            f"有效相机三维点只有 {len(points)} 个，少于要求 {minimum_points} 个"
        )
    return np.asarray(points, dtype=float).reshape(-1, 3), records


def _write_point_csv(path, records, residuals, material):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "材料区域", "像素X", "像素Y", "深度中位数毫米",
                "相机X毫米", "相机Y毫米", "相机Z毫米", "相对黑底拟合平面残差毫米",
            ]
        )
        for record, residual in zip(records, residuals):
            writer.writerow([material, *record, float(residual)])


def _plot_residual_heatmap(output_path, reference_color, black_records, black_residuals):
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    height, width = reference_color.shape[:2]
    heatmap = np.full((height, width), np.nan, dtype=float)
    for record, residual in zip(black_records, black_residuals):
        heatmap[int(record[1]), int(record[0])] = float(residual)
    limit = max(0.1, float(np.nanpercentile(np.abs(heatmap), 99)))
    figure, axis = plt.subplots(figsize=(13, 7))
    axis.imshow(cv2.cvtColor(reference_color, cv2.COLOR_BGR2RGB), alpha=0.28)
    plot = axis.imshow(heatmap, cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest")
    axis.set_title("黑底区域相对最佳拟合平面的有符号残差")
    axis.set_xlabel("像素 u")
    axis.set_ylabel("像素 v")
    figure.colorbar(plot, ax=axis, label="残差（毫米）")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_radial_residual(output_path, records, residuals):
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    pixels = np.asarray([[record[0], record[1]] for record in records], dtype=float)
    center = np.array([EXPECTED_IMAGE_SIZE[0] / 2.0, EXPECTED_IMAGE_SIZE[1] / 2.0])
    scale = np.linalg.norm(center)
    radius = np.linalg.norm(pixels - center, axis=1) / scale
    residuals = np.asarray(residuals, dtype=float)
    bin_edges = np.linspace(0.0, max(1e-6, float(np.max(radius))), 16)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_medians = []
    valid_centers = []
    for left, right, center_value in zip(bin_edges[:-1], bin_edges[1:], bin_centers):
        selected = (radius >= left) & (radius < right)
        if np.any(selected):
            valid_centers.append(center_value)
            bin_medians.append(float(np.median(residuals[selected])))
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.scatter(radius, residuals, s=4, alpha=0.16, label="采样点")
    axis.plot(valid_centers, bin_medians, "o-", color="black", linewidth=2, label="分箱中位数")
    axis.axhline(0.0, color="tab:red", linewidth=1)
    axis.set_title("平面残差随图像半径变化")
    axis.set_xlabel("归一化图像半径")
    axis.set_ylabel("有符号残差（毫米）")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plane_report(plane):
    return {
        "质心毫米": plane.centroid.tolist(),
        "法向量": plane.normal.tolist(),
        "RMSE毫米": float(plane.rmse),
        "P95绝对残差毫米": float(plane.p95),
        "最大绝对残差毫米": float(plane.max_abs),
    }


def main():
    """采集同一静止平板的深度并分析材料与空间残差。"""
    from akai_gemini335 import AkaiGemini335
    from pyorbbecsdk import Config

    run_time = datetime.now(timezone.utc).astimezone()
    output_dir = OUTPUT_ROOT / f"{run_time.strftime('%Y%m%d-%H%M%S')}_{TEST_LABEL}"
    output_dir.mkdir(parents=True, exist_ok=False)
    camera = None
    report = {
        "schema_version": 1,
        "诊断时间": run_time.isoformat(timespec="seconds"),
        "测试标签": TEST_LABEL,
        "估计光心到板面距离毫米": float(APPROX_DISTANCE_MM),
        "图像尺寸": list(EXPECTED_IMAGE_SIZE),
        "采集帧数": int(CAPTURE_FRAMES),
        "单像素最少有效帧数": int(MIN_VALID_FRAMES_PER_PIXEL),
        "相机配置路径": str(CAMERA_CONFIG_PATH),
        "相机配置": yaml.safe_load(CAMERA_CONFIG_PATH.read_text(encoding="utf-8")) or {},
        "警告": [],
    }
    try:
        print("正在独占连接 Gemini335；请先停止 camera_node，并保持相机和板静止。")
        camera = AkaiGemini335(yaml_path=str(CAMERA_CONFIG_PATH))
        camera.config = Config()  # 修复库 bug：release() 需要该属性
        colors, depth_stack = _capture_frames(camera)
        reference_color = colors[-1]
        np.save(output_dir / "原始对齐深度帧.npy", depth_stack)
        save_image(output_dir / "参考彩色图.png", reference_color)
        median_depth, valid_counts = _temporal_median(depth_stack)
        np.save(output_dir / "深度时间中位数.npy", median_depth)
        np.save(output_dir / "逐像素有效帧数.npy", valid_counts)

        detection = detect_asymmetric_circles(reference_color, PATTERN_SIZE)
        save_image(output_dir / "圆点板检测.png", draw_circle_order(reference_color, detection, PATTERN_SIZE))
        if not detection.found:
            raise RuntimeError(f"参考彩色图未检测到完整圆点板：{detection.message}")
        board_mask, white_mask, black_mask, nearest_spacing = make_board_material_masks(
            reference_color.shape[:2],
            detection.centers,
            white_radius_ratio=WHITE_RADIUS_RATIO,
            edge_margin_px=BOARD_EDGE_MARGIN_PX,
        )
        mask_vis = reference_color.copy()
        mask_vis[board_mask] = (0.65 * mask_vis[board_mask] + 0.35 * np.array([255, 0, 0])).astype(np.uint8)
        mask_vis[black_mask] = (0.55 * mask_vis[black_mask] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
        mask_vis[white_mask] = (0.45 * mask_vis[white_mask] + 0.55 * np.array([0, 255, 255])).astype(np.uint8)
        save_image(output_dir / "材料区域掩码.png", mask_vis)

        valid_ratios = {
            "板内整体": _valid_ratio(board_mask, median_depth),
            "白圆区域": _valid_ratio(white_mask, median_depth),
            "黑底区域": _valid_ratio(black_mask, median_depth),
        }
        black_pixels = _sample_pixels(
            black_mask,
            median_depth,
            SAMPLE_STRIDE_PX,
            MAX_BLACK_SAMPLE_POINTS,
            RANDOM_SEED,
        )
        white_pixels = _sample_pixels(
            white_mask,
            median_depth,
            SAMPLE_STRIDE_PX,
            MAX_WHITE_SAMPLE_POINTS,
            RANDOM_SEED + 1,
        )
        black_points, black_records = _deproject(camera, black_pixels, median_depth)
        white_points, white_records = _deproject(
            camera,
            white_pixels,
            median_depth,
            minimum_points=0,
        )
        plane = fit_plane_svd(black_points)
        black_residuals = (black_points - plane.centroid) @ plane.normal
        white_residuals = (
            (white_points - plane.centroid) @ plane.normal
            if len(white_points)
            else np.asarray([], dtype=float)
        )
        material_difference = (
            abs(float(np.median(white_residuals)) - float(np.median(black_residuals)))
            if len(white_residuals) >= 30
            else None
        )
        quadratic = plane_quadratic_cross_validation(
            black_points,
            folds=CV_FOLDS,
            seed=RANDOM_SEED,
            max_points=MAX_BLACK_SAMPLE_POINTS,
        )
        _write_point_csv(output_dir / "黑底采样三维点.csv", black_records, black_residuals, "黑底")
        _write_point_csv(output_dir / "白圆采样三维点.csv", white_records, white_residuals, "白圆")
        _plot_residual_heatmap(
            output_dir / "黑底平面残差热力图.png",
            reference_color,
            black_records,
            black_residuals,
        )
        _plot_radial_residual(
            output_dir / "平面残差随图像半径变化.png",
            black_records,
            black_residuals,
        )

        material_confounded = (
            valid_ratios["黑底区域"] < float(MIN_BLACK_VALID_RATIO)
            or material_difference is None
            or material_difference > float(MAX_MATERIAL_RESIDUAL_DIFFERENCE_MM)
        )
        project_pass = (
            plane.rmse <= float(PROJECT_MAX_PLANE_RMSE_MM)
            and plane.p95 <= float(PROJECT_MAX_PLANE_P95_MM)
        )
        curve_signal = quadratic["quadratic_improvement_ratio"] >= float(
            CURVE_IMPROVEMENT_WARNING_RATIO
        )
        if valid_ratios["黑底区域"] < float(MIN_BLACK_VALID_RATIO):
            report["警告"].append(
                f"黑底有效率 {valid_ratios['黑底区域']:.1%} 低于 {MIN_BLACK_VALID_RATIO:.1%}"
            )
        if material_difference is None:
            report["警告"].append(
                f"白圆有效相机点只有 {len(white_points)} 个，无法可靠比较白圆/黑底材料残差"
            )
        elif material_difference > float(MAX_MATERIAL_RESIDUAL_DIFFERENCE_MM):
            report["警告"].append(
                f"白圆/黑底中位残差差 {material_difference:.3f}mm 超过 "
                f"{MAX_MATERIAL_RESIDUAL_DIFFERENCE_MM:.3f}mm"
            )
        if curve_signal:
            report["警告"].append(
                f"二次曲面留出误差相对平面改善 {quadratic['quadratic_improvement_ratio']:.1%}，"
                "存在平滑曲率信号"
            )
        if material_confounded:
            conclusion = "材料混杂或有效深度不足，无法确认深度曲率"
        elif curve_signal:
            conclusion = "相机坐标深度存在系统性曲率信号；彩色 K/D 不能修复该问题"
        elif project_pass:
            conclusion = "该距离下黑底相机坐标平面性满足当前项目门禁"
        else:
            conclusion = "未见明确二次曲率优势，但平面残差未满足当前项目门禁"
        if float(APPROX_DISTANCE_MM) < 300.0:
            report["警告"].append("本次距离低于 Gemini335 建议工作距离下限，只能作为极近距离对照")

        report.update(
            {
                "圆点检测分支": detection.branch,
                "圆心最近邻中位距离像素": float(nearest_spacing),
                "材料有效率": valid_ratios,
                "采样点数量": {"黑底": int(len(black_points)), "白圆": int(len(white_points))},
                "黑底平面": _plane_report(plane),
                "白圆相对黑底平面": {
                    "中位有符号残差毫米": (
                        float(np.median(white_residuals)) if len(white_residuals) else None
                    ),
                    "P95绝对残差毫米": (
                        float(np.percentile(np.abs(white_residuals), 95))
                        if len(white_residuals)
                        else None
                    ),
                },
                "白圆黑底中位残差差毫米": material_difference,
                "平面与二次曲面留出比较": quadratic,
                "材料混杂": bool(material_confounded),
                "二次曲率信号": bool(curve_signal),
                "当前项目平面门禁通过": bool(project_pass),
                "门禁": {
                    "黑底最小有效率": float(MIN_BLACK_VALID_RATIO),
                    "材料最大残差差毫米": float(MAX_MATERIAL_RESIDUAL_DIFFERENCE_MM),
                    "平面最大RMSE毫米": float(PROJECT_MAX_PLANE_RMSE_MM),
                    "平面最大P95毫米": float(PROJECT_MAX_PLANE_P95_MM),
                    "曲面改善警告比例": float(CURVE_IMPROVEMENT_WARNING_RATIO),
                },
                "结论": conclusion,
                "使用限制": "二次曲面仅用于诊断，本工具不会生成深度修正表",
            }
        )
        print(f"黑底平面 RMSE={plane.rmse:.4f}mm，P95={plane.p95:.4f}mm")
        print(
            "二次曲面留出改善="
            f"{quadratic['quadratic_improvement_ratio']:.1%}，材料残差差="
            + ("不可用" if material_difference is None else f"{material_difference:.4f}mm")
        )
        print(f"结论：{conclusion}")
    except Exception as exc:
        report["致命错误"] = str(exc)
        raise
    finally:
        if camera is not None:
            camera.release()
        report_path = output_dir / "深度平面诊断报告.json"
        report_path.write_text(
            json.dumps(to_builtin(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"诊断报告：{report_path}")


if __name__ == "__main__":
    main()
