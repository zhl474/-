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

BLOCK_INPUT_CSV = Path("/home/zhl/桌面/logs/方块视觉伺服.csv")
TRAY_INPUT_CSV = Path("/home/zhl/桌面/logs/托盘视觉伺服.csv")
BLOCK_EXPECTED_SUCCESS_COUNT = 34
TRAY_EXPECTED_SUCCESS_COUNT = 34

# TCP 点云的最小主轴/次小主轴不超过 1% 时认为近似共面。
PLANARITY_RATIO_TOL = 0.01
CV_FOLDS = 5
RANDOM_SEED = 42
SIMPLE_MODEL_SLACK = 0.05

# 这两项默认不设人为工程阈值；需要时可改成毫米数值。
MAX_SELECTED_CV_RMSE_MM: Optional[float] = None
MAX_SELECTED_CV_P95_MM: Optional[float] = None


COL_CATEGORY = "方块类别"
COL_EVENT = "事件"
COL_PIXEL = ["高位检测像素X", "高位检测像素Y"]
COL_TCP = ["实测TCP位置X", "实测TCP位置Y", "实测TCP位置Z"]
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
    expected_success_count: int
    output_dir: Path
    calibration_filename: str


CALIBRATION_JOBS = (
    CalibrationJob(
        label="方块",
        subject="block",
        input_csv=BLOCK_INPUT_CSV,
        expected_success_count=BLOCK_EXPECTED_SUCCESS_COUNT,
        output_dir=RESULT_ROOT / "方块",
        calibration_filename="block_pixel_to_tcp_calibration.yaml",
    ),
    CalibrationJob(
        label="托盘",
        subject="tray",
        input_csv=TRAY_INPUT_CSV,
        expected_success_count=TRAY_EXPECTED_SUCCESS_COUNT,
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


def analyze_job(job: CalibrationJob) -> bool:
    """分析一份方块或托盘 CSV，通过质量门禁后才写标定 YAML。"""
    configure_matplotlib()
    job.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = job.output_dir / "标定检查报告.json"
    calibration_path = job.output_dir / job.calibration_filename
    report: Dict[str, Any] = {
        "标定对象": job.label,
        "输入CSV": str(job.input_csv),
        "期望成功数量": job.expected_success_count,
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
        if len(success_rows) != job.expected_success_count:
            problems.append(
                f"伺服成功数量为 {len(success_rows)}，"
                f"期望 {job.expected_success_count}"
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


def main() -> None:
    """按顺序分析方块和托盘 CSV，任一失败时返回非零状态。"""
    configure_matplotlib()
    results = [analyze_job(job) for job in CALIBRATION_JOBS]
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
