#!/home/zhl/fr3env/fr3env/bin/python
# -*- coding: utf-8 -*-
"""方块和托盘高位像素到 TCP XYZ 的精简标定工具。

直接修改本文件开头的参数后运行，不需要命令行参数。
工具只检查 TCP 点是否共面，并比较像素到 TCP XYZ 的映射模型。
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2

# 避免无图形环境下 Matplotlib 写入用户配置目录。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pixel_to_tcp_calibration_matplotlib")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import yaml


# ----------------------------- 直接运行配置 -----------------------------
SRC_DIR = Path(__file__).resolve().parents[2]
RESULT_ROOT = SRC_DIR / "tools" / "vision" / "像素-tcp标定结果与数据分析"

BLOCK_INPUT_CSV = Path("/home/zhl/桌面/标定数据/方块视觉伺服.csv")
TRAY_INPUT_CSV = Path("/home/zhl/桌面/标定数据/托盘视觉伺服.csv")
# 2026-08-17 批次 z_blue 高位漏检 2 个，方块只有 33 个；高位识别恢复后回到 35。
# 2026-08-17 深夜手工挑样本重拟合：只保留队友肉眼确认过的 15 条（panel-180621
# 任务序号 1,3,4,5,9,11,13,16,19,20,21,23,29,34,35），疑似/未确认的 20 条已剔除；
# 上限放宽到 49 以便后续可混入 194117 批次样本。
BLOCK_EXPECTED_SUCCESS_RANGE = (15, 49)
TRAY_EXPECTED_SUCCESS_RANGE = (34, 34)

# TCP 点云的最小主轴/次小主轴不超过 1% 时认为近似共面。
PLANARITY_RATIO_TOL = 0.01
CV_FOLDS = 5
RANDOM_SEED = 42
SIMPLE_MODEL_SLACK = 0.05
CV_REPEAT_SEED_COUNT = 200
STATIC_SAMPLE_COMPLETE_RATIO = 0.8
PIXEL_COVERAGE_GAP_WARNING_RATIO = 0.25

# 这两项默认不设人为工程阈值；需要时可改成毫米数值。
MAX_SELECTED_CV_RMSE_MM: Optional[float] = None
MAX_SELECTED_CV_P95_MM: Optional[float] = None

# ----------------------------- 左右分侧标定 -----------------------------
# True 时方块标定额外按高位检测像素 u 分左右两套模型输出：
#   block_pixel_to_tcp_calibration_left.yaml  （u < SIDE_SPLIT_U_PX）
#   block_pixel_to_tcp_calibration_right.yaml （u >= SIDE_SPLIT_U_PX）
# 单文件 block_pixel_to_tcp_calibration.yaml 照常输出（SINGLE 策略与回退用）。
# 每侧成功样本数少于 MIN_SAMPLES_PER_SIDE 时该侧不写文件并判定整个方块标定失败。
SPLIT_BLOCK_SIDES = True
SIDE_SPLIT_U_PX = 640.0
MIN_SAMPLES_PER_SIDE = 5


COL_CATEGORY = "方块类别"
COL_EVENT = "事件"
COL_PIXEL = ["高位检测像素X", "高位检测像素Y"]
COL_TCP = ["实测TCP位置X", "实测TCP位置Y", "实测TCP位置Z"]
COL_TCP_XY = COL_TCP[:2]
COL_TARGET_Z = "标定目标TCP位置Z"
COL_WORLD_Z = "高位世界坐标Z"
COL_DEPTH_VALID_COUNT = "深度有效帧数"
COL_DEPTH_MAD = "深度MAD毫米"
COL_SOURCE = "粗定位来源"
COL_SESSION_ID = "实验批次ID"
COL_TASK_INDEX = "任务序号"
COL_TARGET_TYPE = "目标类型"
COL_HIGH_ANGLE = "高位检测角度deg"
COL_COMMAND_XY = ["最终命令TCP位置X", "最终命令TCP位置Y"]
COL_ZERO_XY = ["零误差等效TCP位置X", "零误差等效TCP位置Y"]
COL_ACTUAL_MINUS_COMMAND = ["实测减命令TCP位置X", "实测减命令TCP位置Y"]
COL_FINAL_PIXEL_ERROR = ["最终像素误差X", "最终像素误差Y"]
COL_STATIC_MEAN = ["静止像素误差均值X", "静止像素误差均值Y"]
COL_STATIC_STD = ["静止像素误差标准差X", "静止像素误差标准差Y"]
COL_STATIC_VALID = "静止采样有效帧数"
COL_STATIC_REQUESTED = "静止采样请求帧数"
COL_STATIC_COMPLETE = "静止采样完整"
REQUIRED_COLUMNS = [COL_CATEGORY, COL_EVENT, *COL_PIXEL, *COL_TCP]
SUCCESS_EVENT = "伺服成功"
FAILURE_EVENT = "伺服失败"
MODEL_NAMES = ("affine", "homography", "poly2", "poly3")
_MATPLOTLIB_CONFIGURED = False


@dataclass(frozen=True)
class CalibrationJob:
    """一份标定 CSV 的直接运行配置。"""

    label: str
    subject: str
    input_csv: Path
    expected_success_range: Tuple[int, int]
    output_dir: Path
    calibration_filename: str


CALIBRATION_JOBS = (
    CalibrationJob(
        label="方块",
        subject="block",
        input_csv=BLOCK_INPUT_CSV,
        expected_success_range=BLOCK_EXPECTED_SUCCESS_RANGE,
        output_dir=RESULT_ROOT / "方块",
        calibration_filename="block_pixel_to_tcp_calibration.yaml",
    ),
    CalibrationJob(
        label="托盘",
        subject="tray",
        input_csv=TRAY_INPUT_CSV,
        expected_success_range=TRAY_EXPECTED_SUCCESS_RANGE,
        output_dir=RESULT_ROOT / "托盘",
        calibration_filename="tray_pixel_to_tcp_calibration.yaml",
    ),
)


@dataclass
class PlaneFit:
    """TCP 点云的正交平面拟合结果。"""

    centroid: np.ndarray
    normal: np.ndarray
    basis: np.ndarray
    signed_distances: np.ndarray
    singular_values: np.ndarray
    rmse: float
    mae: float
    p95_abs: float
    max_abs: float
    sigma3_over_sigma2: float


def configure_matplotlib() -> None:
    """配置中文字体，保证标定图表中文可读。"""
    global _MATPLOTLIB_CONFIGURED
    if _MATPLOTLIB_CONFIGURED:
        return
    font_paths = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    )
    font_names: List[str] = []
    for font_path in font_paths:
        if not font_path.is_file():
            continue
        try:
            font_manager.fontManager.addfont(str(font_path))
            font_names.append(font_manager.FontProperties(fname=str(font_path)).get_name())
        except (OSError, RuntimeError):
            continue
    plt.rcParams["font.sans-serif"] = font_names + [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    _MATPLOTLIB_CONFIGURED = True


def read_csv_auto(path: Path) -> Tuple[pd.DataFrame, str]:
    """自动尝试项目中常用的中文 CSV 编码。"""
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding), encoding
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{encoding}: {exc}")
    raise RuntimeError("CSV 读取失败：\n" + "\n".join(errors))


def fit_plane_svd(points: np.ndarray) -> PlaneFit:
    """使用 SVD 正交拟合三维平面。"""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("TCP 平面拟合至少需要 3 个有效三维点")
    if not np.isfinite(points).all():
        raise ValueError("TCP 平面拟合数据包含 NaN 或无穷值")

    centroid = points.mean(axis=0)
    centered = points - centroid
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1].astype(float)
    normal /= np.linalg.norm(normal)
    dominant_axis = int(np.argmax(np.abs(normal)))
    if normal[dominant_axis] < 0:
        normal = -normal
    basis = vt[:2].astype(float)
    signed = centered @ normal
    absolute = np.abs(signed)
    denominator = max(float(singular_values[-2]), np.finfo(float).eps)
    return PlaneFit(
        centroid=centroid,
        normal=normal,
        basis=basis,
        signed_distances=signed,
        singular_values=singular_values,
        rmse=float(np.sqrt(np.mean(signed**2))),
        mae=float(np.mean(absolute)),
        p95_abs=float(np.percentile(absolute, 95)),
        max_abs=float(np.max(absolute)),
        sigma3_over_sigma2=float(singular_values[-1] / denominator),
    )


def plane_to_report(fit: PlaneFit) -> Dict[str, Any]:
    """把平面拟合结果转成可写入 JSON 的中文报告。"""
    return {
        "质心_mm": fit.centroid.tolist(),
        "单位法向量": fit.normal.tolist(),
        "正交RMSE_mm": fit.rmse,
        "正交MAE_mm": fit.mae,
        "正交P95_mm": fit.p95_abs,
        "最大点面距离_mm": fit.max_abs,
        "奇异值": fit.singular_values.tolist(),
        "sigma3_over_sigma2": fit.sigma3_over_sigma2,
        "平面判定阈值": PLANARITY_RATIO_TOL,
        "平面判定通过": fit.sigma3_over_sigma2 <= PLANARITY_RATIO_TOL,
    }


def normalize_uv(uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对像素坐标标准化，降低多项式拟合的数值条件数。"""
    mean = uv.mean(axis=0)
    scale = uv.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (uv - mean) / scale, mean, scale


def polynomial_features(uv_normalized: np.ndarray, degree: int) -> Tuple[np.ndarray, List[str]]:
    """生成总次数不超过 degree 的二维多项式特征。"""
    u = uv_normalized[:, 0]
    v = uv_normalized[:, 1]
    columns: List[np.ndarray] = []
    names: List[str] = []
    for total_degree in range(degree + 1):
        for u_power in range(total_degree, -1, -1):
            v_power = total_degree - u_power
            columns.append((u**u_power) * (v**v_power))
            if u_power == 0 and v_power == 0:
                names.append("1")
                continue
            parts = []
            if u_power:
                parts.append("u" if u_power == 1 else f"u^{u_power}")
            if v_power:
                parts.append("v" if v_power == 1 else f"v^{v_power}")
            names.append("*".join(parts))
    return np.column_stack(columns), names


def fit_polynomial_mapping(uv: np.ndarray, xyz: np.ndarray, degree: int) -> Dict[str, Any]:
    """拟合直接输出 TCP XYZ 的多项式模型。"""
    normalized, mean, scale = normalize_uv(uv)
    design, feature_names = polynomial_features(normalized, degree)
    if len(uv) < design.shape[1]:
        raise ValueError(f"{degree} 次多项式至少需要 {design.shape[1]} 个训练点")
    coefficients, _, rank, _ = np.linalg.lstsq(design, xyz, rcond=None)
    if rank < design.shape[1]:
        raise ValueError(f"{degree} 次多项式设计矩阵秩不足")
    return {
        "kind": "polynomial",
        "degree": degree,
        "uv_mean": mean,
        "uv_scale": scale,
        "coef": coefficients,
        "feature_names": feature_names,
    }


def predict_polynomial_mapping(model: Mapping[str, Any], uv: np.ndarray) -> np.ndarray:
    """使用多项式模型预测 TCP XYZ。"""
    normalized = (uv - model["uv_mean"]) / model["uv_scale"]
    design, _ = polynomial_features(normalized, int(model["degree"]))
    return design @ model["coef"]


def fit_homography_mapping(uv: np.ndarray, xyz: np.ndarray) -> Dict[str, Any]:
    """把 TCP 投影到最佳拟合平面，再拟合像素到平面坐标的单应。"""
    if len(uv) < 4:
        raise ValueError("单应模型至少需要 4 个训练点")
    plane = fit_plane_svd(xyz)
    plane_xy = (xyz - plane.centroid) @ plane.basis.T
    matrix, _ = cv2.findHomography(
        uv.astype(np.float64),
        plane_xy.astype(np.float64),
        method=0,
    )
    if matrix is None or not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-12:
        raise RuntimeError("单应矩阵拟合失败")
    matrix = matrix / matrix[2, 2]
    return {
        "kind": "homography",
        "H": matrix,
        "plane_centroid": plane.centroid,
        "plane_basis": plane.basis,
        "plane_normal": plane.normal,
    }


def predict_homography_mapping(model: Mapping[str, Any], uv: np.ndarray) -> np.ndarray:
    """使用单应模型预测 TCP XYZ。"""
    plane_xy = cv2.perspectiveTransform(
        uv.astype(np.float64).reshape(-1, 1, 2),
        model["H"],
    ).reshape(-1, 2)
    return model["plane_centroid"] + plane_xy @ model["plane_basis"]


def fit_mapping(model_name: str, uv: np.ndarray, xyz: np.ndarray) -> Dict[str, Any]:
    """按名称拟合一种候选映射模型。"""
    if model_name == "affine":
        return fit_polynomial_mapping(uv, xyz, degree=1)
    if model_name == "homography":
        return fit_homography_mapping(uv, xyz)
    if model_name == "poly2":
        return fit_polynomial_mapping(uv, xyz, degree=2)
    if model_name == "poly3":
        return fit_polynomial_mapping(uv, xyz, degree=3)
    raise KeyError(f"未知映射模型: {model_name}")


def predict_mapping(model_name: str, model: Mapping[str, Any], uv: np.ndarray) -> np.ndarray:
    """使用指定候选模型预测 TCP XYZ。"""
    if model_name in {"affine", "poly2", "poly3"}:
        return predict_polynomial_mapping(model, uv)
    if model_name == "homography":
        return predict_homography_mapping(model, uv)
    raise KeyError(f"未知映射模型: {model_name}")


def make_kfold_indices(sample_count: int, folds: int, seed: int) -> List[np.ndarray]:
    """生成可重复的 K 折交叉验证索引。"""
    if sample_count < 5:
        raise ValueError("映射交叉验证至少需要 5 个有效点")
    rng = np.random.default_rng(seed)
    indices = np.arange(sample_count)
    rng.shuffle(indices)
    return [chunk for chunk in np.array_split(indices, min(folds, sample_count)) if len(chunk)]


def cross_validate_mapping(
    model_name: str,
    uv: np.ndarray,
    xyz: np.ndarray,
    folds: int,
    seed: int,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """返回一种模型的 K 折统计与逐点 OOF 预测。"""
    fold_indices = make_kfold_indices(len(uv), folds, seed)
    predictions = np.full_like(xyz, np.nan, dtype=float)
    all_indices = np.arange(len(uv))
    for test_indices in fold_indices:
        train_mask = np.ones(len(uv), dtype=bool)
        train_mask[test_indices] = False
        train_indices = all_indices[train_mask]
        model = fit_mapping(model_name, uv[train_indices], xyz[train_indices])
        predictions[test_indices] = predict_mapping(model_name, model, uv[test_indices])

    if not np.isfinite(predictions).all():
        raise RuntimeError(f"{model_name} 的 OOF 预测包含无效数值")
    residual = predictions - xyz
    euclidean = np.linalg.norm(residual, axis=1)
    axis_rmse = np.sqrt(np.mean(residual**2, axis=0))
    summary = {
        "模型": model_name,
        "样本数": int(len(uv)),
        "折数": int(len(fold_indices)),
        "CV三维RMSE": float(np.sqrt(np.mean(euclidean**2))),
        "CV三维MAE": float(np.mean(euclidean)),
        "CV三维P95": float(np.percentile(euclidean, 95)),
        "CV三维最大误差": float(np.max(euclidean)),
        "CV_X_RMSE": float(axis_rmse[0]),
        "CV_Y_RMSE": float(axis_rmse[1]),
        "CV_Z_RMSE": float(axis_rmse[2]),
    }
    return summary, predictions


def choose_simplest_near_best_model(summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """在近最优误差内优先选择结构更简单的模型。"""
    rmse_by_model = {str(row["模型"]): float(row["CV三维RMSE"]) for row in summaries}
    best_model = min(rmse_by_model, key=rmse_by_model.get)
    best_rmse = rmse_by_model[best_model]
    limit = best_rmse * (1.0 + SIMPLE_MODEL_SLACK)
    selected = next(name for name in MODEL_NAMES if rmse_by_model.get(name, np.inf) <= limit)
    return {
        "CV最优模型": best_model,
        "CV最优RMSE_mm": best_rmse,
        "近最优容差比例": SIMPLE_MODEL_SLACK,
        "近最优RMSE上限_mm": limit,
        "推荐最简模型": selected,
    }


def to_builtin(value: Any) -> Any:
    """递归转换 Numpy 类型，便于写入 JSON 和 YAML。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def pixel_convex_hull(uv: np.ndarray) -> np.ndarray:
    """计算标定像素覆盖凸包，并拒绝共线数据。"""
    unique_points = np.unique(np.asarray(uv, dtype=float), axis=0)
    if len(unique_points) < 3:
        raise ValueError("标定至少需要 3 个不同的像素点")
    hull = cv2.convexHull(unique_points.astype(np.float32)).reshape(-1, 2)
    if len(hull) < 3 or abs(cv2.contourArea(hull.astype(np.float32))) <= 1e-9:
        raise ValueError("标定像素点共线，无法建立二维映射")
    return hull


def build_calibration_document(
    job: CalibrationJob,
    uv: np.ndarray,
    model_name: str,
    model: Mapping[str, Any],
    cv_summary: Mapping[str, Any],
    training_rmse: float,
) -> Dict[str, Any]:
    """生成与正式定位加载器兼容的标定 YAML 内容。"""
    hull = pixel_convex_hull(uv)
    return {
        "schema_version": 1,
        "calibration_type": "pixel_to_tcp_position",
        "description": "高位检测像素坐标到伺服成功实测 TCP 位置的离线标定结果",
        "input": {
            "coordinate": "high_detection_pixel_xy",
            "columns": list(COL_PIXEL),
            "axes": ["u", "v"],
            "unit": "pixel",
        },
        "output": {
            "coordinate": "tcp_position_xyz",
            "columns": list(COL_TCP),
            "axes": ["x", "y", "z"],
            "unit": "mm",
            "orientation_included": False,
        },
        "model": {"name": model_name, "parameters": to_builtin(dict(model))},
        "metrics": {
            "sample_count": int(len(uv)),
            "cross_validation": {
                "folds": int(cv_summary["折数"]),
                "three_dimensional_rmse_mm": float(cv_summary["CV三维RMSE"]),
                "three_dimensional_mae_mm": float(cv_summary["CV三维MAE"]),
                "three_dimensional_p95_mm": float(cv_summary["CV三维P95"]),
                "three_dimensional_max_error_mm": float(cv_summary["CV三维最大误差"]),
                "axis_rmse_mm": [
                    float(cv_summary["CV_X_RMSE"]),
                    float(cv_summary["CV_Y_RMSE"]),
                    float(cv_summary["CV_Z_RMSE"]),
                ],
            },
            "full_training_three_dimensional_rmse_mm": float(training_rmse),
        },
        "coverage": {
            "sample_count": int(len(uv)),
            "pixel_range": {
                "u": [float(np.min(uv[:, 0])), float(np.max(uv[:, 0]))],
                "v": [float(np.min(uv[:, 1])), float(np.max(uv[:, 1]))],
            },
            "pixel_convex_hull": hull.tolist(),
            "extrapolation_policy": "diagnostic_only",
        },
        "usage_note": (
            "仅适用于高位相机的高位检测像素。输出仅含 TCP XYZ；"
            "调用方须自行提供或沿用 TCP 姿态 R/P/YAW。"
        ),
        "calibration_subject": job.subject,
    }


def plot_plane(points: np.ndarray, fit: PlaneFit, label: str, output_path: Path) -> None:
    """绘制 TCP 点云与最佳拟合平面。"""
    fig = plt.figure(figsize=(8, 7))
    axis = fig.add_subplot(111, projection="3d")
    axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=34)
    plane_xy = (points - fit.centroid) @ fit.basis.T
    first_min, second_min = plane_xy.min(axis=0)
    first_max, second_max = plane_xy.max(axis=0)
    first_grid, second_grid = np.meshgrid(
        np.linspace(first_min, first_max, 15),
        np.linspace(second_min, second_max, 15),
    )
    surface = (
        fit.centroid[None, None, :]
        + first_grid[..., None] * fit.basis[0]
        + second_grid[..., None] * fit.basis[1]
    )
    axis.plot_surface(surface[..., 0], surface[..., 1], surface[..., 2], alpha=0.25)
    axis.set_xlabel("TCP X（毫米）")
    axis.set_ylabel("TCP Y（毫米）")
    axis.set_zlabel("TCP Z（毫米）")
    axis.set_title(
        f"{label} TCP 平面拟合\n"
        f"RMSE={fit.rmse:.4f} mm，最大距离={fit.max_abs:.4f} mm，"
        f"σ3/σ2={fit.sigma3_over_sigma2:.3e}"
    )
    spans = np.ptp(points, axis=0)
    spans[spans < 1e-9] = 1.0
    axis.set_box_aspect(spans)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_model_comparison(summaries: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    """绘制候选映射模型的交叉验证误差对比。"""
    table = pd.DataFrame(summaries).sort_values("CV三维RMSE")
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(table["模型"], table["CV三维RMSE"])
    axis.set_xlabel("映射模型")
    axis.set_ylabel("5折交叉验证三维 RMSE（毫米）")
    axis.set_title("高位像素到 TCP XYZ 映射模型对比")
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    """写入中文 JSON 检查报告。"""
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(to_builtin(dict(data)), file_handle, ensure_ascii=False, indent=2)


def remove_stale_calibration(path: Path) -> None:
    """检查失败时移除上次结果，避免误将旧 YAML 当作本次产物。"""
    if path.is_file():
        path.unlink()


def fit_xy_mapping(model_name: str, uv: np.ndarray, tcp_xy: np.ndarray) -> Dict[str, Any]:
    """拟合 schema v2 的像素到 TCP XY 模型。"""
    if model_name == "affine":
        return fit_polynomial_mapping(uv, tcp_xy, degree=1)
    if model_name == "poly2":
        return fit_polynomial_mapping(uv, tcp_xy, degree=2)
    if model_name == "poly3":
        return fit_polynomial_mapping(uv, tcp_xy, degree=3)
    if model_name == "homography":
        if len(uv) < 4:
            raise ValueError("二维单应模型至少需要 4 个训练点")
        matrix, _ = cv2.findHomography(
            uv.astype(np.float64),
            tcp_xy.astype(np.float64),
            method=0,
        )
        if matrix is None or not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-12:
            raise RuntimeError("二维单应矩阵拟合失败")
        return {"kind": "homography", "H": matrix / matrix[2, 2]}
    raise KeyError(f"未知 XY 映射模型: {model_name}")


def predict_xy_mapping(model_name: str, model: Mapping[str, Any], uv: np.ndarray) -> np.ndarray:
    """使用 schema v2 候选模型预测 TCP XY。"""
    if model_name in {"affine", "poly2", "poly3"}:
        return predict_polynomial_mapping(model, uv)
    if model_name == "homography":
        return cv2.perspectiveTransform(
            uv.astype(np.float64).reshape(-1, 1, 2),
            np.asarray(model["H"], dtype=float),
        ).reshape(-1, 2)
    raise KeyError(f"未知 XY 映射模型: {model_name}")


def cross_validate_xy_mapping(model_name, uv, tcp_xy, folds, seed):
    """对像素到 TCP XY 模型执行 K 折交叉验证。"""
    fold_indices = make_kfold_indices(len(uv), folds, seed)
    predictions = np.full_like(tcp_xy, np.nan, dtype=float)
    all_indices = np.arange(len(uv))
    for test_indices in fold_indices:
        train_mask = np.ones(len(uv), dtype=bool)
        train_mask[test_indices] = False
        train_indices = all_indices[train_mask]
        model = fit_xy_mapping(model_name, uv[train_indices], tcp_xy[train_indices])
        predictions[test_indices] = predict_xy_mapping(model_name, model, uv[test_indices])
    if not np.isfinite(predictions).all():
        raise RuntimeError(f"{model_name} 的 XY OOF 预测包含无效数值")
    residual = predictions - tcp_xy
    distance = np.linalg.norm(residual, axis=1)
    axis_rmse = np.sqrt(np.mean(residual**2, axis=0))
    return {
        "模型": model_name,
        "样本数": int(len(uv)),
        "折数": int(len(fold_indices)),
        "CV二维RMSE": float(np.sqrt(np.mean(distance**2))),
        "CV二维MAE": float(np.mean(distance)),
        "CV二维P95": float(np.percentile(distance, 95)),
        "CV二维最大误差": float(np.max(distance)),
        "CV_X_RMSE": float(axis_rmse[0]),
        "CV_Y_RMSE": float(axis_rmse[1]),
    }, predictions


def choose_xy_model(summaries):
    """在二维近最优误差内选择最简单的映射模型。"""
    rmse_by_model = {row["模型"]: float(row["CV二维RMSE"]) for row in summaries}
    best_name = min(rmse_by_model, key=rmse_by_model.get)
    best_rmse = rmse_by_model[best_name]
    limit = best_rmse * (1.0 + SIMPLE_MODEL_SLACK)
    selected = next(name for name in MODEL_NAMES if rmse_by_model.get(name, np.inf) <= limit)
    return {
        "CV最优模型": best_name,
        "CV最优RMSE_mm": best_rmse,
        "近最优容差比例": SIMPLE_MODEL_SLACK,
        "推荐最简模型": selected,
    }


def fit_vertical_z_plane(tcp_xy, target_z):
    """使用外部深度锚定的目标 Z 拟合 z=a*x+b*y+c。"""
    xy = np.asarray(tcp_xy, dtype=float)
    z = np.asarray(target_z, dtype=float)
    design = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    coefficients, _residual_sum, rank, _singular = np.linalg.lstsq(design, z, rcond=None)
    if rank < 3:
        raise ValueError("方块实测 TCP XY 共线，无法拟合 Z 平面")
    residual = design @ coefficients - z
    return coefficients, residual, float(np.sqrt(np.mean(residual**2)))


def build_v2_document(
    job,
    uv,
    model_name,
    model,
    cv_summary,
    training_rmse,
    z_coefficients,
    generation_id,
    z_source,
    tray_offset_mm=None,
    z_plane_rmse_mm=None,
    zone=None,
):
    """生成 XY 模型与 Z 平面解耦的 schema v2 标定文档。

    zone 为 "left"/"right" 时写入分区元数据，供左右分侧标定留档。
    """
    document = {
        "schema_version": 2,
        "generation_id": generation_id,
        "calibration_type": "pixel_to_tcp_position",
        "calibration_subject": job.subject,
        "description": "高位像素到 TCP XY 模型与独立 TCP Z 平面标定结果",
        "input": {
            "coordinate": "high_detection_pixel_xy",
            "columns": list(COL_PIXEL),
            "axes": ["u", "v"],
            "unit": "pixel",
        },
        "output": {
            "coordinate": "tcp_position_xyz",
            "columns": list(COL_TCP),
            "axes": ["x", "y", "z"],
            "unit": "mm",
            "orientation_included": False,
        },
        "xy_model": {"name": model_name, "parameters": to_builtin(dict(model))},
        "z_plane": {
            "equation": "z = a*x + b*y + c",
            "coefficients": to_builtin(np.asarray(z_coefficients, dtype=float)),
            "source": z_source,
        },
        "metrics": {
            "sample_count": int(len(uv)),
            "xy_cross_validation": to_builtin(dict(cv_summary)),
            "xy_full_training_rmse_mm": float(training_rmse),
        },
        "coverage": {
            "sample_count": int(len(uv)),
            "pixel_range": {
                "u": [float(np.min(uv[:, 0])), float(np.max(uv[:, 0]))],
                "v": [float(np.min(uv[:, 1])), float(np.max(uv[:, 1]))],
            },
            "pixel_convex_hull": pixel_convex_hull(uv).tolist(),
            "extrapolation_policy": "diagnostic_only",
        },
        "usage_note": "正式运行仅使用离线 XY 模型和独立 Z 平面，不查询深度相机。",
    }
    if zone is not None:
        document["zone"] = str(zone)
        document["usage_note"] = (
            f"左右分侧标定（{zone}侧，u 相对中线 {SIDE_SPLIT_U_PX:g} 分侧）；"
            "正式运行仅使用离线 XY 模型和独立 Z 平面，不查询深度相机。"
        )
    if z_plane_rmse_mm is not None:
        document["metrics"]["external_depth_z_plane_rmse_mm"] = float(z_plane_rmse_mm)
    if tray_offset_mm is not None:
        document["z_plane"]["derived_from"] = "block_observation_tcp_plane"
        document["z_plane"]["tray_tcp_below_block_observation_mm"] = float(tray_offset_mm)
    return document


def analyze_job(job: CalibrationJob) -> bool:
    """分析一份方块或托盘 CSV，通过质量门禁后才写标定 YAML。"""
    configure_matplotlib()
    job.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = job.output_dir / "标定检查报告.json"
    calibration_path = job.output_dir / job.calibration_filename
    report: Dict[str, Any] = {
        "标定对象": job.label,
        "输入CSV": str(job.input_csv),
        "期望成功数量": list(job.expected_success_range),
        "质量门禁通过": False,
        "问题": [],
    }
    problems: List[str] = report["问题"]

    try:
        dataframe, encoding = read_csv_auto(job.input_csv)
        report["CSV编码"] = encoding
        report["CSV总行数"] = int(len(dataframe))
        missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
        if missing:
            raise ValueError(f"CSV 缺少列: {missing}")

        dataframe = dataframe.loc[:, REQUIRED_COLUMNS].copy()
        dataframe[COL_CATEGORY] = dataframe[COL_CATEGORY].fillna("").astype(str).str.strip()
        dataframe[COL_EVENT] = dataframe[COL_EVENT].fillna("").astype(str).str.strip()
        invalid_events = sorted(set(dataframe[COL_EVENT]) - {SUCCESS_EVENT, FAILURE_EVENT})
        if invalid_events:
            problems.append(f"存在非法事件值: {invalid_events}")

        failure_rows = dataframe[dataframe[COL_EVENT] == FAILURE_EVENT]
        success_rows = dataframe[dataframe[COL_EVENT] == SUCCESS_EVENT].copy()
        report["伺服成功数量"] = int(len(success_rows))
        report["伺服失败数量"] = int(len(failure_rows))
        report["失败行号"] = [int(index) + 2 for index in failure_rows.index]
        if len(failure_rows):
            problems.append(f"存在 {len(failure_rows)} 条伺服失败记录")
        expected_min, expected_max = job.expected_success_range
        if not expected_min <= len(success_rows) <= expected_max:
            problems.append(
                f"伺服成功数量为 {len(success_rows)}，"
                f"可接受范围 [{expected_min}, {expected_max}]"
            )

        blank_category = success_rows[COL_CATEGORY] == ""
        if blank_category.any():
            rows = [int(index) + 2 for index in success_rows.index[blank_category]]
            problems.append(f"成功记录的方块类别为空，行号: {rows}")

        for column in [*COL_PIXEL, *COL_TCP]:
            success_rows[column] = pd.to_numeric(success_rows[column], errors="coerce")
        numeric = success_rows.loc[:, [*COL_PIXEL, *COL_TCP]].to_numpy(dtype=float)
        finite_mask = np.isfinite(numeric).all(axis=1)
        if not finite_mask.all():
            invalid_indices = success_rows.index[~finite_mask]
            problems.append(
                "成功记录的像素或 TCP 字段存在空值/非数值，行号: "
                f"{[int(index) + 2 for index in invalid_indices]}"
            )
        valid_rows = success_rows.loc[finite_mask].copy()
        report["有效成功点数量"] = int(len(valid_rows))
        if len(valid_rows) < 5:
            raise ValueError("有效成功点少于 5，无法完成映射交叉验证")

        uv = valid_rows.loc[:, COL_PIXEL].to_numpy(dtype=float)
        xyz = valid_rows.loc[:, COL_TCP].to_numpy(dtype=float)
        hull = pixel_convex_hull(uv)
        report["像素覆盖凸包顶点数"] = int(len(hull))

        plane = fit_plane_svd(xyz)
        plane_report = plane_to_report(plane)
        report["TCP平面检查"] = plane_report
        plot_plane(xyz, plane, job.label, job.output_dir / "TCP平面拟合.png")
        if not plane_report["平面判定通过"]:
            problems.append(
                "TCP 点云未通过平面判定: "
                f"σ3/σ2={plane.sigma3_over_sigma2:.6g} > {PLANARITY_RATIO_TOL}"
            )

        summaries: List[Dict[str, Any]] = []
        predictions_by_model: Dict[str, np.ndarray] = {}
        model_errors: Dict[str, str] = {}
        for model_name in MODEL_NAMES:
            try:
                summary, predictions = cross_validate_mapping(
                    model_name,
                    uv,
                    xyz,
                    CV_FOLDS,
                    RANDOM_SEED,
                )
                summaries.append(summary)
                predictions_by_model[model_name] = predictions
            except Exception as exc:  # noqa: BLE001
                model_errors[model_name] = str(exc)
        report["映射模型交叉验证"] = summaries
        report["映射模型失败原因"] = model_errors
        if not summaries:
            raise ValueError("所有候选映射模型都无法完成交叉验证")

        selection = choose_simplest_near_best_model(summaries)
        report["模型选择"] = selection
        selected_name = selection["推荐最简模型"]
        selected_summary = next(row for row in summaries if row["模型"] == selected_name)
        if (
            MAX_SELECTED_CV_RMSE_MM is not None
            and selected_summary["CV三维RMSE"] > MAX_SELECTED_CV_RMSE_MM
        ):
            problems.append(
                f"推荐模型 CV RMSE={selected_summary['CV三维RMSE']:.6g} mm "
                f"超过阈值 {MAX_SELECTED_CV_RMSE_MM} mm"
            )
        if (
            MAX_SELECTED_CV_P95_MM is not None
            and selected_summary["CV三维P95"] > MAX_SELECTED_CV_P95_MM
        ):
            problems.append(
                f"推荐模型 CV P95={selected_summary['CV三维P95']:.6g} mm "
                f"超过阈值 {MAX_SELECTED_CV_P95_MM} mm"
            )

        plot_model_comparison(summaries, job.output_dir / "映射模型交叉验证对比.png")
        error_table = valid_rows.loc[:, [COL_CATEGORY, *COL_PIXEL, *COL_TCP]].copy()
        error_table.insert(0, "CSV行号", [int(index) + 2 for index in valid_rows.index])
        for model_name, predictions in predictions_by_model.items():
            error_table[f"{model_name}_OOF三维误差_mm"] = np.linalg.norm(
                predictions - xyz,
                axis=1,
            )
        error_table.to_csv(
            job.output_dir / "逐点OOF误差.csv",
            index=False,
            encoding="utf-8-sig",
        )

        selected_model = fit_mapping(selected_name, uv, xyz)
        training_predictions = predict_mapping(selected_name, selected_model, uv)
        training_rmse = float(
            np.sqrt(np.mean(np.sum((training_predictions - xyz) ** 2, axis=1)))
        )
        report["全数据训练三维RMSE_mm"] = training_rmse

        if problems:
            remove_stale_calibration(calibration_path)
        else:
            document = build_calibration_document(
                job,
                uv,
                selected_name,
                selected_model,
                selected_summary,
                training_rmse,
            )
            with calibration_path.open("w", encoding="utf-8") as file_handle:
                yaml.safe_dump(document, file_handle, allow_unicode=True, sort_keys=False)
            report["质量门禁通过"] = True
            report["生成标定YAML"] = str(calibration_path)
    except Exception as exc:  # noqa: BLE001
        problems.append(str(exc))
        remove_stale_calibration(calibration_path)

    report["生成标定YAML"] = report.get("生成标定YAML", None)
    write_json(report_path, report)
    if report["质量门禁通过"]:
        print(f"[{job.label}] 标定通过：{calibration_path}")
        return True
    print(f"[{job.label}] 标定未通过，请查看：{report_path}", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    return False


def _read_v2_success_rows(job):
    """读取一份新标定 CSV，并返回成功行和基础检查报告。"""
    dataframe, encoding = read_csv_auto(job.input_csv)
    required = [
        COL_CATEGORY,
        COL_EVENT,
        *COL_PIXEL,
        *COL_TCP,
        COL_TARGET_Z,
        COL_WORLD_Z,
        COL_DEPTH_VALID_COUNT,
        COL_DEPTH_MAD,
        COL_SOURCE,
    ]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{job.label} CSV 缺少列: {missing}")
    # 保留 schema v3 的实验诊断列；核心门禁仍只依赖 required 中的旧字段。
    dataframe = dataframe.copy()
    dataframe[COL_EVENT] = dataframe[COL_EVENT].fillna("").astype(str).str.strip()
    dataframe[COL_CATEGORY] = dataframe[COL_CATEGORY].fillna("").astype(str).str.strip()
    dataframe[COL_SOURCE] = dataframe[COL_SOURCE].fillna("").astype(str).str.strip()
    if COL_TARGET_TYPE in dataframe.columns:
        dataframe[COL_TARGET_TYPE] = (
            dataframe[COL_TARGET_TYPE].fillna("").astype(str).str.strip()
        )
    invalid_events = sorted(set(dataframe[COL_EVENT]) - {SUCCESS_EVENT, FAILURE_EVENT})
    if invalid_events:
        raise ValueError(f"{job.label} CSV 存在非法事件值: {invalid_events}")
    failures = dataframe[dataframe[COL_EVENT] == FAILURE_EVENT]
    successes = dataframe[dataframe[COL_EVENT] == SUCCESS_EVENT].copy()
    if len(failures):
        raise ValueError(f"{job.label}存在 {len(failures)} 条伺服失败记录")
    expected_min, expected_max = job.expected_success_range
    if not expected_min <= len(successes) <= expected_max:
        raise ValueError(
            f"{job.label}伺服成功数量为 {len(successes)}，"
            f"可接受范围 [{expected_min}, {expected_max}]"
        )
    # 新版托盘日志使用目标类型和托盘行列标识目标，不再填写方块类别。
    if job.subject == "block" and (successes[COL_CATEGORY] == "").any():
        raise ValueError(f"{job.label}成功记录存在空方块类别")
    if COL_TARGET_TYPE in successes.columns:
        expected_target_type = "方块" if job.subject == "block" else "托盘"
        invalid_target_types = sorted(
            set(successes[COL_TARGET_TYPE]) - {expected_target_type}
        )
        if invalid_target_types:
            raise ValueError(
                f"{job.label}成功记录的目标类型应全部为{expected_target_type}，"
                f"实际异常值: {invalid_target_types}"
            )
    numeric_columns = [
        *COL_PIXEL,
        *COL_TCP,
        COL_TARGET_Z,
        COL_WORLD_Z,
        COL_DEPTH_VALID_COUNT,
        COL_DEPTH_MAD,
    ]
    for column in numeric_columns:
        successes[column] = pd.to_numeric(successes[column], errors="coerce")
    if not np.isfinite(successes[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{job.label}成功记录存在空值或非有限数值")
    pixel_convex_hull(successes[COL_PIXEL].to_numpy(dtype=float))
    return successes, {
        "标定对象": job.label,
        "输入CSV": str(job.input_csv),
        "CSV编码": encoding,
        "伺服成功数量": int(len(successes)),
        "伺服失败数量": 0,
        "质量门禁通过": False,
        "问题": [],
    }


def _fit_v2_xy_job(job, rows, report, side=None):
    """完成一个主体（或左右分区中的一侧）的 XY 模型比较、选择和全量拟合。"""
    side_label = "" if side is None else f"{side}侧"
    uv = rows[COL_PIXEL].to_numpy(dtype=float)
    tcp_xy = rows[COL_TCP_XY].to_numpy(dtype=float)
    summaries = []
    predictions = {}
    failures = {}
    for model_name in MODEL_NAMES:
        try:
            summary, prediction = cross_validate_xy_mapping(
                model_name,
                uv,
                tcp_xy,
                CV_FOLDS,
                RANDOM_SEED,
            )
            summaries.append(summary)
            predictions[model_name] = prediction
        except Exception as exc:  # noqa: BLE001
            failures[model_name] = str(exc)
    if not summaries:
        raise ValueError(f"{job.label}{side_label}所有 XY 候选模型均无法完成交叉验证")
    selection = choose_xy_model(summaries)
    selected_name = selection["推荐最简模型"]
    selected_summary = next(row for row in summaries if row["模型"] == selected_name)
    selected_model = fit_xy_mapping(selected_name, uv, tcp_xy)
    training_prediction = predict_xy_mapping(selected_name, selected_model, uv)
    training_rmse = float(
        np.sqrt(np.mean(np.sum((training_prediction - tcp_xy) ** 2, axis=1)))
    )
    if side is None:
        report["XY映射模型交叉验证"] = summaries
        report["XY映射模型失败原因"] = failures
        report["XY模型选择"] = selection
        report["XY全数据训练RMSE_mm"] = training_rmse
    else:
        report[f"{side}侧XY映射模型交叉验证"] = summaries
        report[f"{side}侧XY映射模型失败原因"] = failures
        report[f"{side}侧XY模型选择"] = selection
        report[f"{side}侧XY全数据训练RMSE_mm"] = training_rmse

    error_table = rows[[COL_CATEGORY, *COL_PIXEL, *COL_TCP_XY]].copy()
    for model_name, prediction in predictions.items():
        error_table[f"{model_name}_OOF二维误差_mm"] = np.linalg.norm(
            prediction - tcp_xy,
            axis=1,
        )
    error_table.to_csv(
        job.output_dir / f"逐点OOF误差{side_label}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return uv, selected_name, selected_model, selected_summary, training_rmse


def _split_block_sides(rows, side_u_px: float = SIDE_SPLIT_U_PX):
    """按高位检测像素 u 把方块成功行分成左右两份（u < 阈值 → left）。"""
    u = rows[COL_PIXEL[0]].to_numpy(dtype=float)
    left_rows = rows.loc[u < side_u_px].copy()
    right_rows = rows.loc[u >= side_u_px].copy()
    return left_rows, right_rows


def _side_calibration_filename(job, side: str):
    """左右分侧标定输出文件名：block_pixel_to_tcp_calibration_left/right.yaml。"""
    calibration_path = Path(job.calibration_filename)
    return f"{calibration_path.stem}_{side}{calibration_path.suffix}"


def _numeric_matrix(rows, columns):
    """读取一组可选实验列；任一空值时返回 None。"""
    if any(column not in rows.columns for column in columns):
        return None
    numeric = rows.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return numeric if np.isfinite(numeric).all() else None


def _axis_gap_report(values):
    """报告单轴最大无样本区间，用于识别两簇随机 K 折盲区。"""
    ordered = np.unique(np.asarray(values, dtype=float))
    if len(ordered) < 2:
        return {"最小值": float(ordered[0]), "最大值": float(ordered[0]), "最大间隔": 0.0, "间隔占跨度比例": 0.0}
    gaps = np.diff(ordered)
    gap_index = int(np.argmax(gaps))
    span = float(ordered[-1] - ordered[0])
    ratio = float(gaps[gap_index] / span) if span > 0.0 else 0.0
    return {
        "最小值": float(ordered[0]),
        "最大值": float(ordered[-1]),
        "最大间隔": float(gaps[gap_index]),
        "最大间隔起点": float(ordered[gap_index]),
        "最大间隔终点": float(ordered[gap_index + 1]),
        "间隔占跨度比例": ratio,
        "超过警告阈值": ratio > PIXEL_COVERAGE_GAP_WARNING_RATIO,
    }


def _evaluate_xy_labels(uv, labels):
    summaries = []
    predictions = {}
    failures = {}
    for model_name in MODEL_NAMES:
        try:
            summary, prediction = cross_validate_xy_mapping(
                model_name, uv, labels, CV_FOLDS, RANDOM_SEED
            )
            summaries.append(summary)
            predictions[model_name] = prediction
        except Exception as exc:  # noqa: BLE001
            failures[model_name] = str(exc)
    return {
        "模型统计": summaries,
        "模型选择": choose_xy_model(summaries) if summaries else None,
        "失败原因": failures,
    }, predictions


def _repeat_seed_stability(uv, labels):
    """重复随机折分，只用于判断选择稳定性，不替代部署用固定 seed。"""
    scores = {name: [] for name in MODEL_NAMES}
    selected_counts = {name: 0 for name in MODEL_NAMES}
    failed_seeds = []
    for seed in range(CV_REPEAT_SEED_COUNT):
        summaries = []
        try:
            for model_name in MODEL_NAMES:
                summary, _prediction = cross_validate_xy_mapping(
                    model_name, uv, labels, CV_FOLDS, seed
                )
                summaries.append(summary)
                scores[model_name].append(float(summary["CV二维RMSE"]))
            selected_counts[choose_xy_model(summaries)["推荐最简模型"]] += 1
        except Exception:  # noqa: BLE001
            failed_seeds.append(seed)
    statistics = []
    for model_name in MODEL_NAMES:
        values = np.asarray(scores[model_name], dtype=float)
        statistics.append(
            {
                "模型": model_name,
                "有效随机种子数": int(len(values)),
                "RMSE均值_mm": float(np.mean(values)) if len(values) else None,
                "RMSE标准差_mm": float(np.std(values)) if len(values) else None,
                "推荐次数": int(selected_counts[model_name]),
            }
        )
    return {"重复次数": CV_REPEAT_SEED_COUNT, "模型统计": statistics, "失败种子": failed_seeds}


def _plot_vector_panel(axis, uv, vectors, title, unit, amplification):
    magnitude = np.linalg.norm(vectors, axis=1)
    plot = axis.scatter(uv[:, 0], uv[:, 1], c=magnitude, cmap="viridis", s=26)
    axis.quiver(
        uv[:, 0],
        uv[:, 1],
        vectors[:, 0] * amplification,
        vectors[:, 1] * amplification,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="tab:red",
        width=0.003,
    )
    axis.invert_yaxis()
    axis.set_xlabel("高位像素 u")
    axis.set_ylabel("高位像素 v")
    axis.set_title(f"{title}\n箭头放大 {amplification:g} 倍")
    plt.colorbar(plot, ax=axis, label=f"幅值（{unit}）")


def _plot_experiment_vectors(job, uv, actual_labels, actual_predictions, rows):
    """绘制四类空间矢量，缺失的实验字段以说明文字代替。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    homography_prediction = actual_predictions.get("homography")
    if homography_prediction is not None:
        _plot_vector_panel(
            axes[0, 0], uv, homography_prediction - actual_labels,
            "单应 OOF 残差", "mm", 45.0,
        )
    else:
        axes[0, 0].text(0.5, 0.5, "无单应 OOF 预测", ha="center", va="center")

    optional_vectors = (
        (COL_FINAL_PIXEL_ERROR, "最终低位像素残差", "px", 12.0),
        (COL_ACTUAL_MINUS_COMMAND, "实测减命令 TCP", "mm", 45.0),
        (COL_STATIC_MEAN, "终止残差均值", "px", 12.0),
    )
    for axis, (columns, title, unit, amplification) in zip(axes.flat[1:], optional_vectors):
        vectors = _numeric_matrix(rows, columns)
        if vectors is None:
            axis.text(0.5, 0.5, f"缺少{title}字段", ha="center", va="center")
            axis.set_axis_off()
            continue
        _plot_vector_panel(axis, uv, vectors, title, unit, amplification)
    fig.suptitle(f"{job.label}实验空间矢量诊断")
    fig.tight_layout()
    fig.savefig(job.output_dir / "实验空间矢量诊断.png", dpi=180)
    plt.close(fig)


def _plot_label_comparison(job, label_reports):
    rows = []
    for label_name, report in label_reports.items():
        for model in report.get("模型统计", []):
            rows.append(
                {"标签": label_name, "模型": model["模型"], "CV二维RMSE": model["CV二维RMSE"]}
            )
    if not rows:
        return
    table = pd.DataFrame(rows)
    pivot = table.pivot(index="模型", columns="标签", values="CV二维RMSE").reindex(MODEL_NAMES)
    axis = pivot.plot(kind="bar", figsize=(10, 6))
    axis.set_ylabel("5折交叉验证二维 RMSE（毫米）")
    axis.set_xlabel("映射模型")
    axis.set_title(f"{job.label}不同 TCP 标签的模型对比")
    axis.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(job.output_dir / "不同标签模型对比.png", dpi=180)
    plt.close()


def _plot_seed_stability(job, stability):
    table = pd.DataFrame(stability["模型统计"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(table["模型"], table["推荐次数"])
    axes[0].set_title("200组随机折分的推荐次数")
    axes[0].set_ylabel("次数")
    axes[1].bar(table["模型"], table["RMSE均值_mm"], yerr=table["RMSE标准差_mm"])
    axes[1].set_title("随机折分 RMSE 均值与标准差")
    axes[1].set_ylabel("毫米")
    for axis in axes:
        axis.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{job.label}模型选择稳定性")
    fig.tight_layout()
    fig.savefig(job.output_dir / "CV随机种子稳定性.png", dpi=180)
    plt.close(fig)


def _plot_terminal_jitter(job, rows):
    means = _numeric_matrix(rows, COL_STATIC_MEAN)
    standard_deviations = _numeric_matrix(rows, COL_STATIC_STD)
    if means is None or standard_deviations is None:
        return
    indices = np.arange(1, len(rows) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for axis_index, axis_name in enumerate(("X", "Y")):
        axes[axis_index].errorbar(
            indices,
            means[:, axis_index],
            yerr=standard_deviations[:, axis_index],
            fmt="o",
            capsize=2,
        )
        axes[axis_index].axhline(0.0, color="black", linewidth=1)
        axes[axis_index].set_ylabel(f"像素误差 {axis_name}（px）")
        axes[axis_index].grid(alpha=0.3)
    axes[-1].set_xlabel("成功目标序号")
    fig.suptitle(f"{job.label}成功后静止检测均值与标准差")
    fig.tight_layout()
    fig.savefig(job.output_dir / "终止残差与抖动.png", dpi=180)
    plt.close(fig)


def _analyze_experiment_log(job, rows, report):
    """分析 schema v3 实验字段；任何诊断失败都不得改变标定门禁。"""
    required = [
        COL_SESSION_ID,
        COL_TASK_INDEX,
        *COL_COMMAND_XY,
        *COL_FINAL_PIXEL_ERROR,
        *COL_STATIC_MEAN,
        *COL_STATIC_STD,
        COL_STATIC_VALID,
        COL_STATIC_REQUESTED,
        COL_STATIC_COMPLETE,
    ]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        report["实验日志诊断"] = {"可用": False, "原因": f"缺少实验诊断列: {missing}"}
        return
    try:
        uv = rows[COL_PIXEL].to_numpy(dtype=float)
        actual_labels = rows[COL_TCP_XY].to_numpy(dtype=float)
        session_ids = rows[COL_SESSION_ID].fillna("").astype(str).str.strip().unique().tolist()
        if len(session_ids) != 1 or not session_ids[0]:
            raise ValueError(f"实验批次 ID 不唯一或为空: {session_ids}")
        numeric_columns = [
            COL_TASK_INDEX,
            *COL_COMMAND_XY,
            *COL_FINAL_PIXEL_ERROR,
            *COL_STATIC_MEAN,
            *COL_STATIC_STD,
            COL_STATIC_VALID,
            COL_STATIC_REQUESTED,
        ]
        numeric = rows.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
        diagnostic_rows = rows.copy()
        diagnostic_rows.loc[:, numeric_columns] = numeric

        complete_text = diagnostic_rows[COL_STATIC_COMPLETE].astype(str).str.lower()
        static_complete = complete_text.isin({"true", "1", "是"})
        requested_samples = numeric[COL_STATIC_REQUESTED]
        expected_complete = np.ceil(STATIC_SAMPLE_COMPLETE_RATIO * requested_samples)
        expected_complete_flags = (
            (requested_samples > 0) & (numeric[COL_STATIC_VALID] >= expected_complete)
        )
        completeness_consistent = bool(
            np.array_equal(
                static_complete.to_numpy(),
                expected_complete_flags.to_numpy(),
            )
        )

        archive_dir = (
            job.input_csv.parent
            if job.input_csv.parent.name == session_ids[0]
            else job.input_csv.parent / "实验日志" / session_ids[0]
        )
        round_filename = (
            "方块视觉伺服逐轮.csv" if job.subject == "block" else "托盘视觉伺服逐轮.csv"
        )
        round_path = archive_dir / round_filename
        round_report = {"路径": str(round_path), "存在": round_path.is_file()}
        if round_path.is_file():
            round_table, round_encoding = read_csv_auto(round_path)
            round_report.update(
                {
                    "编码": round_encoding,
                    "总行数": int(len(round_table)),
                    "事件数量": round_table.get("事件", pd.Series(dtype=str)).value_counts().to_dict(),
                }
            )

        label_reports = {}
        actual_report, actual_predictions = _evaluate_xy_labels(uv, actual_labels)
        label_reports["实测TCP"] = actual_report
        command_labels = _numeric_matrix(diagnostic_rows, COL_COMMAND_XY)
        if command_labels is not None:
            label_reports["最终命令TCP"], _unused = _evaluate_xy_labels(uv, command_labels)
        zero_labels = _numeric_matrix(diagnostic_rows, COL_ZERO_XY)
        if bool(static_complete.all()) and zero_labels is not None:
            label_reports["零误差等效TCP"], _unused = _evaluate_xy_labels(uv, zero_labels)

        homography_prediction = actual_predictions.get("homography")
        category_report = {}
        if homography_prediction is not None:
            distances = np.linalg.norm(homography_prediction - actual_labels, axis=1)
            categories = diagnostic_rows[COL_CATEGORY].astype(str)
            for category in sorted(categories.unique()):
                values = distances[categories == category]
                category_report[category] = {
                    "数量": int(len(values)),
                    "单应OOF_RMS_mm": float(np.sqrt(np.mean(values**2))),
                }
            is_l = categories.isin({"L_yellow", "L_blue"}).to_numpy()
            category_report["L与非L汇总"] = {
                "L_RMS_mm": float(np.sqrt(np.mean(distances[is_l] ** 2))) if is_l.any() else None,
                "非L_RMS_mm": float(np.sqrt(np.mean(distances[~is_l] ** 2))) if (~is_l).any() else None,
            }
            if COL_HIGH_ANGLE in diagnostic_rows.columns:
                angles = pd.to_numeric(
                    diagnostic_rows[COL_HIGH_ANGLE], errors="coerce"
                ).to_numpy(dtype=float)
                finite_angles = np.isfinite(angles)
                angle_groups = {}
                if finite_angles.any():
                    rounded = np.round(angles[finite_angles] / 15.0) * 15.0
                    finite_distances = distances[finite_angles]
                    for angle in sorted(np.unique(rounded)):
                        values = finite_distances[rounded == angle]
                        angle_groups[f"{angle:g}deg"] = {
                            "数量": int(len(values)),
                            "单应OOF_RMS_mm": float(np.sqrt(np.mean(values**2))),
                        }
                category_report["按15度角度分组"] = angle_groups

        stability = _repeat_seed_stability(uv, actual_labels)
        coverage = {
            "u轴": _axis_gap_report(uv[:, 0]),
            "v轴": _axis_gap_report(uv[:, 1]),
            "警告阈值比例": PIXEL_COVERAGE_GAP_WARNING_RATIO,
        }
        warnings = []
        for axis_name in ("u轴", "v轴"):
            if coverage[axis_name]["超过警告阈值"]:
                warnings.append(f"{axis_name}存在超过总跨度25%的无样本区间")
        if not bool(static_complete.all()):
            warnings.append("存在静止采样不足80%的目标，未比较零误差等效TCP模型")
        if not completeness_consistent:
            warnings.append("静止采样完整标记与有效帧数不一致")

        diagnostic_columns = [
            COL_CATEGORY,
            COL_TASK_INDEX,
            *COL_PIXEL,
            *COL_TCP_XY,
            *COL_COMMAND_XY,
            *COL_ZERO_XY,
            *COL_FINAL_PIXEL_ERROR,
            *COL_STATIC_MEAN,
            *COL_STATIC_STD,
            COL_STATIC_VALID,
            COL_STATIC_REQUESTED,
            COL_STATIC_COMPLETE,
        ]
        available_columns = [column for column in diagnostic_columns if column in diagnostic_rows]
        diagnostic_rows.loc[:, available_columns].to_csv(
            job.output_dir / "逐目标终止诊断.csv", index=False, encoding="utf-8-sig"
        )
        _plot_experiment_vectors(job, uv, actual_labels, actual_predictions, diagnostic_rows)
        _plot_label_comparison(job, label_reports)
        _plot_seed_stability(job, stability)
        _plot_terminal_jitter(job, diagnostic_rows)
        report["实验日志诊断"] = {
            "可用": True,
            "实验批次ID": session_ids[0],
            "逐轮日志": round_report,
            "静止采样完整目标数": int(static_complete.sum()),
            "静止采样目标总数": int(len(static_complete)),
            "像素覆盖": coverage,
            "不同标签模型比较": label_reports,
            "随机折分稳定性": stability,
            "类别与L形统计": category_report,
            "警告": warnings,
        }
    except Exception as exc:  # noqa: BLE001
        report["实验日志诊断"] = {"可用": False, "原因": str(exc)}


def analyze_calibration_pair(block_job, tray_job) -> bool:
    """成对生成 schema v2 方块/托盘标定，任一失败时两份结果均不保留。"""
    configure_matplotlib()
    jobs = (block_job, tray_job)
    for job in jobs:
        job.output_dir.mkdir(parents=True, exist_ok=True)
        remove_stale_calibration(job.output_dir / job.calibration_filename)

    reports = {
        job.subject: {
            "标定对象": job.label,
            "输入CSV": str(job.input_csv),
            "质量门禁通过": False,
            "问题": [],
        }
        for job in jobs
    }
    try:
        with (SRC_DIR / "image_process" / "config" / "perception.yaml").open(
            "r", encoding="utf-8"
        ) as file_handle:
            perception = yaml.safe_load(file_handle) or {}
        depth_config = perception.get("calibration_depth", {})
        min_valid_frames = int(depth_config.get("min_valid_frames", 10))
        block_max_mad_mm = float(depth_config.get("block_max_mad_mm", 1.0))
        block_plane_max_rmse_mm = float(depth_config.get("block_plane_max_rmse_mm", 1.0))
        tray_offset_mm = float(
            depth_config.get("tray_tcp_below_block_observation_mm", 7.0)
        )
        observation_height_mm = float(
            perception.get("pick_height", {}).get("block_observation_height_mm", 192.0)
        )
        fixed_z = perception.get("high_tcp_localization", {}).get("fixed_tcp_z", {})
        fixed_z_enabled = bool(fixed_z.get("enabled", False))
        fixed_block_z_mm = (
            float(fixed_z["block_observation_z_mm"]) if fixed_z_enabled else None
        )
        fixed_tray_z_mm = float(fixed_z["tray_z_mm"]) if fixed_z_enabled else None

        block_rows, reports["block"] = _read_v2_success_rows(block_job)
        tray_rows, reports["tray"] = _read_v2_success_rows(tray_job)
        if not (block_rows[COL_SOURCE] == "stable_depth_xyz").all():
            raise ValueError("方块 CSV 粗定位来源必须全部为 stable_depth_xyz")
        expected_tray_source = (
            "stable_depth_xy_fixed_z" if fixed_z_enabled else "stable_depth_xy_block_plane_z"
        )
        if not (tray_rows[COL_SOURCE] == expected_tray_source).all():
            expected_text = (
                "stable_depth_xy_fixed_z（固定 Z 模式采集）"
                if fixed_z_enabled
                else "stable_depth_xy_block_plane_z（深度平面模式采集）"
            )
            raise ValueError(f"托盘 CSV 粗定位来源必须全部为 {expected_text}")
        if (block_rows[COL_DEPTH_VALID_COUNT] < min_valid_frames).any():
            raise ValueError("方块 CSV 存在有效深度帧数不足的记录")
        if (tray_rows[COL_DEPTH_VALID_COUNT] < min_valid_frames).any():
            raise ValueError("托盘 CSV 存在有效深度帧数不足的记录")
        maximum_block_mad = float(block_rows[COL_DEPTH_MAD].max())
        reports["block"]["最大深度MAD_mm"] = maximum_block_mad
        if maximum_block_mad > block_max_mad_mm:
            raise ValueError(
                f"方块最大深度 MAD={maximum_block_mad:.6g} mm "
                f"超过阈值 {block_max_mad_mm} mm"
            )
        block_target_z = block_rows[COL_TARGET_Z].to_numpy(dtype=float)
        if fixed_z_enabled:
            # 固定 Z 模式：采集时目标 Z 全部是常数，验证与当前配置一致；
            # 深度表面 Z 与固定值的偏差仅作诊断对照，不再参与门禁。
            if not np.allclose(
                block_target_z,
                fixed_block_z_mm,
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(
                    "固定 Z 模式下方块标定目标 TCP Z 必须全部等于配置的 "
                    f"fixed_tcp_z.block_observation_z_mm={fixed_block_z_mm:.3f} mm"
                )
            if not np.allclose(
                tray_rows[COL_TARGET_Z].to_numpy(dtype=float),
                fixed_tray_z_mm,
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(
                    "固定 Z 模式下托盘标定目标 TCP Z 必须全部等于配置的 "
                    f"fixed_tcp_z.tray_z_mm={fixed_tray_z_mm:.3f} mm"
                )
            depth_bias = float(
                np.mean(
                    block_rows[COL_WORLD_Z].to_numpy(dtype=float)
                    - (fixed_block_z_mm - observation_height_mm)
                )
            )
            reports["block"]["固定Z深度诊断"] = {
                "深度表面Z减固定表面Z均值_mm": depth_bias,
                "说明": "正值表示深度测得表面偏高，该偏差不参与标定生成",
            }
        else:
            block_depth_target_z = (
                block_rows[COL_WORLD_Z].to_numpy(dtype=float) + observation_height_mm
            )
            if not np.allclose(
                block_target_z,
                block_depth_target_z,
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(
                    "方块标定目标 TCP Z 必须严格等于深度表面世界 Z 加 "
                    f"{observation_height_mm:.3f} mm"
                )

        block_xy = block_rows[COL_TCP_XY].to_numpy(dtype=float)
        if fixed_z_enabled:
            # 固定 Z 模式：z_plane 直接写常数，不拟合（与正式运行链路一致）。
            block_z_coefficients = np.asarray([0.0, 0.0, fixed_block_z_mm], dtype=float)
            block_z_rmse = 0.0
            reports["block"]["固定Z平面"] = {
                "系数_a_b_c": block_z_coefficients.tolist(),
                "来源": "fixed_constant_block_observation_z",
            }
            tray_z_coefficients = np.asarray([0.0, 0.0, fixed_tray_z_mm], dtype=float)
            reports["tray"]["固定Z平面"] = {
                "系数_a_b_c": tray_z_coefficients.tolist(),
                "来源": "fixed_constant_tray_z",
            }
        else:
            block_z_coefficients, block_z_residuals, block_z_rmse = fit_vertical_z_plane(
                block_xy,
                block_target_z,
            )
            reports["block"]["外部深度Z平面"] = {
                "系数_a_b_c": block_z_coefficients.tolist(),
                "RMSE_mm": block_z_rmse,
                "最大绝对残差_mm": float(np.max(np.abs(block_z_residuals))),
                "门禁_mm": block_plane_max_rmse_mm,
            }
            if block_z_rmse > block_plane_max_rmse_mm:
                raise ValueError(
                    f"方块外部深度 Z 平面 RMSE={block_z_rmse:.6g} mm "
                    f"超过阈值 {block_plane_max_rmse_mm} mm"
                )
            tray_z_coefficients = block_z_coefficients.copy()
            tray_z_coefficients[2] -= tray_offset_mm
            reports["tray"]["Z平面来源"] = {
                "来源": "方块观察TCP平面",
                "向下控制偏移_mm": tray_offset_mm,
                "系数_a_b_c": tray_z_coefficients.tolist(),
            }

        fitted = {}
        for job, rows in ((block_job, block_rows), (tray_job, tray_rows)):
            fitted[job.subject] = _fit_v2_xy_job(job, rows, reports[job.subject])
            _analyze_experiment_log(job, rows, reports[job.subject])

        generation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        block_z_source = (
            "fixed_constant_block_observation_z"
            if fixed_z_enabled
            else "external_depth_block_surface_plus_observation_height"
        )
        documents = {}
        for job in jobs:
            uv, name, model, summary, training_rmse = fitted[job.subject]
            if job.subject == "block":
                z_coefficients = block_z_coefficients
                z_source = block_z_source
                offset = None
                plane_rmse = None if fixed_z_enabled else block_z_rmse
            else:
                z_coefficients = tray_z_coefficients
                z_source = (
                    "fixed_constant_tray_z"
                    if fixed_z_enabled
                    else "derived_from_block_observation_tcp_plane"
                )
                offset = None if fixed_z_enabled else tray_offset_mm
                plane_rmse = None
            documents[job.subject] = build_v2_document(
                job,
                uv,
                name,
                model,
                summary,
                training_rmse,
                z_coefficients,
                generation_id,
                z_source,
                tray_offset_mm=offset,
                z_plane_rmse_mm=plane_rmse,
            )

        # 左右分侧模型：同一批次、同一 Z 平面/常数，仅 XY 模型按侧独立拟合。
        side_documents = {}
        if SPLIT_BLOCK_SIDES:
            left_rows, right_rows = _split_block_sides(block_rows)
            reports["block"]["左右分区"] = {
                "分侧阈值u": SIDE_SPLIT_U_PX,
                "左侧样本数": int(len(left_rows)),
                "右侧样本数": int(len(right_rows)),
                "最少样本数门禁": MIN_SAMPLES_PER_SIDE,
            }
            for side, side_rows in (("left", left_rows), ("right", right_rows)):
                if len(side_rows) < MIN_SAMPLES_PER_SIDE:
                    raise ValueError(
                        f"方块{side}侧成功样本数 {len(side_rows)} 少于 "
                        f"最少样本数 {MIN_SAMPLES_PER_SIDE}，左右分侧标定失败"
                    )
                side_uv, side_name, side_model, side_summary, side_rmse = (
                    _fit_v2_xy_job(
                        block_job,
                        side_rows,
                        reports["block"],
                        side=side,
                    )
                )
                side_documents[side] = build_v2_document(
                    block_job,
                    side_uv,
                    side_name,
                    side_model,
                    side_summary,
                    side_rmse,
                    block_z_coefficients,
                    generation_id,
                    block_z_source,
                    z_plane_rmse_mm=None if fixed_z_enabled else block_z_rmse,
                    zone=side,
                )

        staged_paths = {}
        for job in jobs:
            calibration_path = job.output_dir / job.calibration_filename
            staged_path = calibration_path.with_suffix(calibration_path.suffix + ".tmp")
            with staged_path.open("w", encoding="utf-8") as file_handle:
                yaml.safe_dump(
                    documents[job.subject],
                    file_handle,
                    allow_unicode=True,
                    sort_keys=False,
                )
            staged_paths[job.subject] = staged_path
        side_staged = []
        for side, document in side_documents.items():
            target = block_job.output_dir / _side_calibration_filename(block_job, side)
            staged = target.with_suffix(target.suffix + ".tmp")
            with staged.open("w", encoding="utf-8") as file_handle:
                yaml.safe_dump(
                    document,
                    file_handle,
                    allow_unicode=True,
                    sort_keys=False,
                )
            side_staged.append((target, staged))

        # 全部内容均成功序列化后才替换候选文件；任何异常都会在下方统一清理。
        for job in jobs:
            calibration_path = job.output_dir / job.calibration_filename
            os.replace(staged_paths[job.subject], calibration_path)
            reports[job.subject]["质量门禁通过"] = True
            reports[job.subject]["生成标定YAML"] = str(calibration_path)
            reports[job.subject]["生成批次"] = generation_id
        for target, staged in side_staged:
            os.replace(staged, target)
        if side_staged:
            reports["block"]["左右分区"]["生成YAML"] = [
                str(target) for target, _staged in side_staged
            ]
    except Exception as exc:  # noqa: BLE001
        for job in jobs:
            remove_stale_calibration(job.output_dir / job.calibration_filename)
            remove_stale_calibration(
                (job.output_dir / job.calibration_filename).with_suffix(
                    Path(job.calibration_filename).suffix + ".tmp"
                )
            )
            for side in ("left", "right"):
                side_path = job.output_dir / _side_calibration_filename(job, side)
                remove_stale_calibration(side_path)
                remove_stale_calibration(
                    side_path.with_suffix(side_path.suffix + ".tmp")
                )
            reports[job.subject]["质量门禁通过"] = False
            reports[job.subject].pop("生成标定YAML", None)
            reports[job.subject].pop("生成批次", None)
            reports[job.subject].setdefault("问题", []).append(str(exc))

    all_passed = all(reports[job.subject]["质量门禁通过"] for job in jobs)
    for job in jobs:
        write_json(job.output_dir / "标定检查报告.json", reports[job.subject])
        state = "通过" if reports[job.subject]["质量门禁通过"] else "未通过"
        print(f"[{job.label}] 成对标定{state}：{job.output_dir}")
    return all_passed


def main() -> None:
    """成对分析方块和托盘 CSV，并生成同批次 schema v2 标定。"""
    if not analyze_calibration_pair(CALIBRATION_JOBS[0], CALIBRATION_JOBS[1]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
