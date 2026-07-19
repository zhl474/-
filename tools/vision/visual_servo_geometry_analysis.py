#!/home/zhl/fr3env/fr3env/bin/python
# -*- coding: utf-8 -*-
"""
视觉伺服粗定位几何分析

功能：
1. 从“伺服成功”行提取每个方块的实测 TCP XYZ，使用 SVD 正交拟合平面。
2. 按方块提取唯一的高位世界 XYZ，拟合平面，并比较两个平面的法向夹角。
3. 比较像素 (u, v) -> TCP (X, Y, Z) 的 affine / homography / poly2 / poly3 映射，
   使用 K 折交叉验证评估泛化误差，而不是只看训练误差。

脚本只做机械化统计和输出，不替代工程阈值设定。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
# 当前执行环境的 ~/.config/matplotlib 不可写。将字体缓存放入临时目录，避免每次运行均产生警告。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/visual_servo_geometry_matplotlib")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import yaml


# ----------------------------- 列名配置 -----------------------------
COL_RUN = "运行编号"
COL_TASK = "任务序号"
COL_CATEGORY = "方块类别"
COL_TRAY_ROW = "托盘行"
COL_TRAY_COL = "托盘列"
COL_EVENT = "事件"

COL_PIXEL = ["高位检测像素X", "高位检测像素Y"]
COL_WORLD = ["高位世界坐标X", "高位世界坐标Y", "高位世界坐标Z"]
COL_TCP = ["实测TCP位置X", "实测TCP位置Y", "实测TCP位置Z"]

SUCCESS_EVENT = "伺服成功"
TCP_CALIBRATION_FILENAME = "像素到TCP标定结果.yaml"
TCP_CALIBRATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AnalysisSpec:
    """方块与托盘共用的几何分析对象配置。"""

    subject_label: str
    sample_label: str
    output_prefix: str = ""
    calibration_filename: str = TCP_CALIBRATION_FILENAME
    calibration_subject: Optional[str] = None
    extra_unique_columns: Tuple[str, ...] = ()
    tray_grid_diagnostic: bool = False


BLOCK_ANALYSIS_SPEC = AnalysisSpec(
    subject_label="方块",
    sample_label="方块",
)

TRAY_ANALYSIS_SPEC = AnalysisSpec(
    subject_label="托盘",
    sample_label="托盘目标",
    output_prefix="托盘",
    calibration_filename="托盘像素到TCP标定结果.yaml",
    calibration_subject="tray",
    extra_unique_columns=(COL_TRAY_ROW, COL_TRAY_COL),
    tray_grid_diagnostic=True,
)


# ----------------------------- 直接运行配置 -----------------------------
# 这是测试/标定脚本的默认配置。通常只需修改这里，随后直接运行本文件即可。
DEFAULT_INPUT_CSV = "/home/zhl/桌面/logs/方块视觉伺服.csv"
DEFAULT_OUTPUT_DIR = "visual_servo_geometry_results"
DEFAULT_EXPECTED_COUNT = 35
DEFAULT_PLANE_RMSE_TOL: Optional[float] = None
DEFAULT_PLANE_MAX_TOL: Optional[float] = None
# 无量纲形状判据：最小主轴/次小主轴不超过 1%，可视为近似平面。
# 绝对误差阈值必须结合项目单位与装配公差确定，因此默认不强行设置。
DEFAULT_PLANARITY_RATIO_TOL: Optional[float] = 0.01
# 两法向夹角不超过 1° 时判为平行；若需更严格的工程公差，请直接修改该值。
DEFAULT_PARALLEL_ANGLE_TOL_DEG: Optional[float] = 1.0
# 如两个坐标系轴方向不一致，填写 world->TCP 的 3×3 旋转矩阵；否则保持 None。
DEFAULT_WORLD_TO_TCP_ROTATION: Optional[str] = None
DEFAULT_UNIQUE_VALUE_TOL = 1e-9
DEFAULT_CV_FOLDS = 5
DEFAULT_RANDOM_SEED = 42
DEFAULT_SIMPLE_MODEL_SLACK = 0.05
DEFAULT_SKIP_PLOTS = False


@dataclass
class PlaneFit:
    centroid: np.ndarray
    normal: np.ndarray
    basis: np.ndarray
    d: float
    signed_distances: np.ndarray
    singular_values: np.ndarray
    rmse: float
    mae: float
    max_abs: float
    p95_abs: float
    sigma3_over_sigma2: float
    normal_variance_ratio: float

    def to_jsonable(self) -> Dict[str, Any]:
        result = asdict(self)
        for key, value in list(result.items()):
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                result[key] = value.item()
        return result


# ----------------------------- 基础工具 -----------------------------
def read_csv_auto(path: Path) -> Tuple[pd.DataFrame, str]:
    """自动尝试常见中文 CSV 编码。"""
    errors: List[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding), encoding
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{encoding}: {exc}")
    raise RuntimeError("CSV 读取失败：\n" + "\n".join(errors))


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"CSV 缺少列：{missing}")


def normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def to_numeric_inplace(df: pd.DataFrame, columns: Sequence[str]) -> None:
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def finite_rows(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    arr = df.loc[:, columns].to_numpy(dtype=float)
    return pd.Series(np.isfinite(arr).all(axis=1), index=df.index)


def build_key_columns(df: pd.DataFrame) -> List[str]:
    """优先使用运行编号+任务序号；运行编号不存在时退化为任务序号。"""
    if COL_RUN in df.columns:
        return [COL_RUN, COL_TASK]
    return [COL_TASK]


def key_to_text(key: Any) -> str:
    if isinstance(key, tuple):
        return " / ".join(map(str, key))
    return str(key)


# ----------------------------- 样本级数据整理 -----------------------------
def prepare_analysis_rows(df: pd.DataFrame, spec: AnalysisSpec) -> pd.DataFrame:
    """保留有效样本记录，清理事件文本和数值列。"""
    require_columns(
        df,
        [
            COL_TASK,
            COL_CATEGORY,
            COL_EVENT,
            *COL_PIXEL,
            *COL_WORLD,
            *COL_TCP,
            *spec.extra_unique_columns,
        ],
    )

    out = df.copy()
    out[COL_EVENT] = normalize_text(out[COL_EVENT])
    out[COL_CATEGORY] = normalize_text(out[COL_CATEGORY])
    out = out[out[COL_CATEGORY] != ""].copy()

    to_numeric_inplace(
        out,
        [COL_TASK, *COL_PIXEL, *COL_WORLD, *COL_TCP, *spec.extra_unique_columns],
    )
    return out


def extract_success_rows(
    block_df: pd.DataFrame,
    key_cols: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    每个键必须恰好对应一行“伺服成功”。

    返回：
    - success_rows：每个成功方块一行
    - success_validation：每个任务的成功行数量检查
    """
    all_keys = block_df.loc[:, list(key_cols)].drop_duplicates()
    success = block_df[block_df[COL_EVENT] == SUCCESS_EVENT].copy()

    counts = (
        success.groupby(list(key_cols), dropna=False)
        .size()
        .rename("伺服成功行数")
        .reset_index()
    )
    validation = all_keys.merge(counts, on=list(key_cols), how="left")
    validation["伺服成功行数"] = validation["伺服成功行数"].fillna(0).astype(int)
    validation["成功行唯一"] = validation["伺服成功行数"] == 1

    bad = validation[~validation["成功行唯一"]]
    if not bad.empty:
        preview = bad.head(20).to_dict(orient="records")
        raise ValueError(
            "存在任务的‘伺服成功’行数量不等于 1。请先修正日志。\n"
            f"前 20 个异常键：{preview}"
        )

    # 由于已经验证唯一，这里不会静默丢弃重复项。
    success = success.sort_values(list(key_cols)).reset_index(drop=True)
    return success, validation


def extract_unique_task_values(
    block_df: pd.DataFrame,
    key_cols: Sequence[str],
    value_cols: Sequence[str],
    consistency_tol: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    从同一方块的多条事件记录中提取“唯一值”。

    数值采用中位数作为代表值，同时输出 max-min 跨事件波动；
    任何列的波动超过 consistency_tol 都会被标记。
    """
    rows: List[Dict[str, Any]] = []
    checks: List[Dict[str, Any]] = []

    grouped = block_df.groupby(list(key_cols), dropna=False, sort=True)
    for key, group in grouped:
        if len(key_cols) == 1:
            key = (key,)
        key_dict = dict(zip(key_cols, key))

        row: Dict[str, Any] = dict(key_dict)
        check: Dict[str, Any] = dict(key_dict)
        all_ok = True

        # 类别也保留为方块级元数据。
        categories = normalize_text(group[COL_CATEGORY])
        categories = categories[categories != ""].unique().tolist()
        row[COL_CATEGORY] = categories[0] if len(categories) == 1 else "|".join(categories)
        check["方块类别唯一"] = len(categories) == 1
        all_ok &= len(categories) == 1

        for col in value_cols:
            values = pd.to_numeric(group[col], errors="coerce").dropna().to_numpy(dtype=float)
            if values.size == 0:
                row[col] = np.nan
                spread = np.nan
                ok = False
            else:
                row[col] = float(np.median(values))
                spread = float(np.max(values) - np.min(values))
                ok = spread <= consistency_tol
            check[f"{col}_跨事件极差"] = spread
            check[f"{col}_唯一"] = ok
            all_ok &= ok

        check["全部唯一"] = bool(all_ok)
        rows.append(row)
        checks.append(check)

    return pd.DataFrame(rows), pd.DataFrame(checks)


# ----------------------------- 平面拟合 -----------------------------
def fit_plane_svd(points: np.ndarray) -> PlaneFit:
    """使用总最小二乘/SVD，最小化点到平面的正交距离。"""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points 必须为 N×3，当前形状：{points.shape}")
    if points.shape[0] < 3:
        raise ValueError("平面拟合至少需要 3 个有效点")
    if not np.isfinite(points).all():
        raise ValueError("平面拟合输入包含 NaN/Inf")

    centroid = points.mean(axis=0)
    centered = points - centroid
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

    normal = vt[-1].astype(float)
    normal /= np.linalg.norm(normal)
    # 固定法向符号，保证不同运行输出稳定；不影响平面本身。
    dominant_axis = int(np.argmax(np.abs(normal)))
    if normal[dominant_axis] < 0:
        normal = -normal

    # 平面内正交基。若法向翻转，basis 不需要跟着翻转。
    basis = vt[:2].astype(float)
    d = -float(np.dot(normal, centroid))
    signed = centered @ normal
    abs_dist = np.abs(signed)

    rmse = float(np.sqrt(np.mean(signed**2)))
    mae = float(np.mean(abs_dist))
    max_abs = float(np.max(abs_dist))
    p95_abs = float(np.percentile(abs_dist, 95))

    eps = np.finfo(float).eps
    sigma3_over_sigma2 = float(singular_values[-1] / max(singular_values[-2], eps))
    variance = singular_values**2
    normal_variance_ratio = float(variance[-1] / max(np.sum(variance), eps))

    return PlaneFit(
        centroid=centroid,
        normal=normal,
        basis=basis,
        d=d,
        signed_distances=signed,
        singular_values=singular_values,
        rmse=rmse,
        mae=mae,
        max_abs=max_abs,
        p95_abs=p95_abs,
        sigma3_over_sigma2=sigma3_over_sigma2,
        normal_variance_ratio=normal_variance_ratio,
    )


def plane_judgement(
    fit: PlaneFit,
    rmse_tol: Optional[float],
    max_tol: Optional[float],
    ratio_tol: Optional[float],
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    if rmse_tol is not None:
        checks["RMSE通过"] = fit.rmse <= rmse_tol
    if max_tol is not None:
        checks["最大距离通过"] = fit.max_abs <= max_tol
    if ratio_tol is not None:
        checks["奇异值比通过"] = fit.sigma3_over_sigma2 <= ratio_tol

    if checks:
        checks["平面判定通过"] = all(bool(v) for v in checks.values())
    else:
        checks["平面判定通过"] = None
    return checks


def parse_rotation_matrix(text: Optional[str]) -> Optional[np.ndarray]:
    """解析 9 个逗号分隔数字，表示 world 坐标系到 TCP 坐标系的旋转矩阵。"""
    if text is None:
        return None
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(values) != 9:
        raise ValueError("--world-to-tcp-rotation 必须提供 9 个逗号分隔数字")
    rotation = np.asarray(values, dtype=float).reshape(3, 3)
    should_be_identity = rotation.T @ rotation
    if not np.allclose(should_be_identity, np.eye(3), atol=1e-3):
        raise ValueError("提供的旋转矩阵不满足 R^T R ≈ I")
    if np.linalg.det(rotation) < 0.0:
        raise ValueError("提供的矩阵行列式小于 0，不是合法旋转矩阵")
    return rotation


def normal_angle_deg(
    tcp_normal: np.ndarray,
    world_normal: np.ndarray,
    world_to_tcp_rotation: Optional[np.ndarray],
) -> Tuple[float, np.ndarray]:
    transformed = np.asarray(world_normal, dtype=float)
    if world_to_tcp_rotation is not None:
        transformed = world_to_tcp_rotation @ transformed
    transformed /= np.linalg.norm(transformed)

    # 法向 n 和 -n 表示同一平面，因此使用绝对点积，角度范围为 [0, 90°]。
    cosine = float(np.clip(abs(np.dot(tcp_normal, transformed)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine))), transformed


# ----------------------------- 映射模型 -----------------------------
def normalize_uv(uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = uv.mean(axis=0)
    scale = uv.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (uv - mean) / scale, mean, scale


def polynomial_features(uv_normalized: np.ndarray, degree: int) -> Tuple[np.ndarray, List[str]]:
    """生成总次数不超过 degree 的二维多项式特征，包含常数项。"""
    u = uv_normalized[:, 0]
    v = uv_normalized[:, 1]
    features: List[np.ndarray] = []
    names: List[str] = []
    for total_degree in range(degree + 1):
        for u_power in range(total_degree, -1, -1):
            v_power = total_degree - u_power
            features.append((u**u_power) * (v**v_power))
            if u_power == 0 and v_power == 0:
                names.append("1")
            else:
                parts = []
                if u_power:
                    parts.append("u" if u_power == 1 else f"u^{u_power}")
                if v_power:
                    parts.append("v" if v_power == 1 else f"v^{v_power}")
                names.append("*".join(parts))
    return np.column_stack(features), names


def fit_polynomial_mapping(uv: np.ndarray, xyz: np.ndarray, degree: int) -> Dict[str, Any]:
    uvn, mean, scale = normalize_uv(uv)
    design, feature_names = polynomial_features(uvn, degree)
    coef, _, _, _ = np.linalg.lstsq(design, xyz, rcond=None)
    return {
        "kind": "polynomial",
        "degree": degree,
        "uv_mean": mean,
        "uv_scale": scale,
        "coef": coef,
        "feature_names": feature_names,
    }


def predict_polynomial_mapping(model: Dict[str, Any], uv: np.ndarray) -> np.ndarray:
    uvn = (uv - model["uv_mean"]) / model["uv_scale"]
    design, _ = polynomial_features(uvn, int(model["degree"]))
    return design @ model["coef"]


def fit_homography_mapping(uv: np.ndarray, xyz: np.ndarray) -> Dict[str, Any]:
    """
    先把 TCP 点投影到其最佳拟合平面的二维基坐标，再拟合像素到平面坐标的单应性。
    """
    if uv.shape[0] < 4:
        raise ValueError("单应性拟合至少需要 4 个点")

    plane = fit_plane_svd(xyz)
    plane_xy = (xyz - plane.centroid) @ plane.basis.T
    h, _ = cv2.findHomography(
        uv.astype(np.float64),
        plane_xy.astype(np.float64),
        method=0,
    )
    if h is None or not np.isfinite(h).all():
        raise RuntimeError("cv2.findHomography 拟合失败")
    h = h / h[2, 2]
    return {
        "kind": "homography",
        "H": h,
        "plane_centroid": plane.centroid,
        "plane_basis": plane.basis,
        "plane_normal": plane.normal,
    }


def predict_homography_mapping(model: Dict[str, Any], uv: np.ndarray) -> np.ndarray:
    uv_cv = uv.astype(np.float64).reshape(-1, 1, 2)
    plane_xy = cv2.perspectiveTransform(uv_cv, model["H"]).reshape(-1, 2)
    return model["plane_centroid"] + plane_xy @ model["plane_basis"]


def fit_mapping(model_name: str, uv: np.ndarray, xyz: np.ndarray) -> Dict[str, Any]:
    if model_name == "affine":
        return fit_polynomial_mapping(uv, xyz, degree=1)
    if model_name == "poly2":
        return fit_polynomial_mapping(uv, xyz, degree=2)
    if model_name == "poly3":
        return fit_polynomial_mapping(uv, xyz, degree=3)
    if model_name == "homography":
        return fit_homography_mapping(uv, xyz)
    raise KeyError(f"未知模型：{model_name}")


def predict_mapping(model_name: str, model: Dict[str, Any], uv: np.ndarray) -> np.ndarray:
    if model_name in {"affine", "poly2", "poly3"}:
        return predict_polynomial_mapping(model, uv)
    if model_name == "homography":
        return predict_homography_mapping(model, uv)
    raise KeyError(f"未知模型：{model_name}")


def make_kfold_indices(n: int, folds: int, seed: int) -> List[np.ndarray]:
    if n < 5:
        raise ValueError("映射交叉验证至少建议 5 个有效点")
    folds = max(2, min(folds, n))
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    return [chunk for chunk in np.array_split(indices, folds) if chunk.size > 0]


def cross_validate_mapping(
    model_name: str,
    uv: np.ndarray,
    xyz: np.ndarray,
    folds: int,
    seed: int,
) -> Tuple[Dict[str, Any], np.ndarray]:
    fold_indices = make_kfold_indices(len(uv), folds, seed)
    oof_pred = np.full_like(xyz, np.nan, dtype=float)

    all_indices = np.arange(len(uv))
    for test_idx in fold_indices:
        train_mask = np.ones(len(uv), dtype=bool)
        train_mask[test_idx] = False
        train_idx = all_indices[train_mask]

        if model_name == "homography" and len(train_idx) < 4:
            raise ValueError("训练折中的点数不足以拟合单应性")

        model = fit_mapping(model_name, uv[train_idx], xyz[train_idx])
        oof_pred[test_idx] = predict_mapping(model_name, model, uv[test_idx])

    residual = oof_pred - xyz
    euclidean = np.linalg.norm(residual, axis=1)
    axis_rmse = np.sqrt(np.mean(residual**2, axis=0))

    summary: Dict[str, Any] = {
        "模型": model_name,
        "样本数": len(uv),
        "折数": len(fold_indices),
        "CV三维RMSE": float(np.sqrt(np.mean(euclidean**2))),
        "CV三维MAE": float(np.mean(euclidean)),
        "CV三维中位数": float(np.median(euclidean)),
        "CV三维P95": float(np.percentile(euclidean, 95)),
        "CV三维最大误差": float(np.max(euclidean)),
        "CV_X_RMSE": float(axis_rmse[0]),
        "CV_Y_RMSE": float(axis_rmse[1]),
        "CV_Z_RMSE": float(axis_rmse[2]),
    }
    return summary, oof_pred


def choose_simplest_near_best_model(
    cv_summary: pd.DataFrame,
    slack: float,
) -> Dict[str, Any]:
    """
    在 CV RMSE 不超过最优值 (1+slack) 的模型中，选择最简单模型。
    避免 poly3 仅凭微小误差优势被自动选中。
    """
    complexity_order = ["affine", "homography", "poly2", "poly3"]
    rmse_by_model = dict(zip(cv_summary["模型"], cv_summary["CV三维RMSE"]))
    best_model = min(rmse_by_model, key=rmse_by_model.get)
    best_rmse = float(rmse_by_model[best_model])
    limit = best_rmse * (1.0 + slack)

    selected = next(name for name in complexity_order if rmse_by_model.get(name, np.inf) <= limit)
    interpretation = {
        "affine": "仿射模型已足够；在当前覆盖范围内不需要更复杂非线性。",
        "homography": "单应模型更合适；这是平面成像常见的投影关系，不属于任意复杂非线性。",
        "poly2": "二次模型在交叉验证中有稳定收益；存在仿射/单应性未解释的非线性。",
        "poly3": "三次模型才达到近最优；需警惕样本量不足、外推失真或未建模系统误差。",
    }[selected]

    return {
        "CV最优模型": best_model,
        "CV最优RMSE": best_rmse,
        "近最优容差比例": slack,
        "近最优上限": limit,
        "推荐最简模型": selected,
        "机械判读": interpretation,
    }


def model_to_jsonable(model: Dict[str, Any]) -> Dict[str, Any]:
    """将 Numpy 模型参数转换为 JSON/YAML 可序列化的 Python 基础类型。"""
    def to_builtin(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {key: to_builtin(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [to_builtin(item) for item in value]
        return value

    return to_builtin(model)


def pixel_convex_hull(uv: np.ndarray) -> np.ndarray:
    """计算标定样本的像素凸包，拒绝所有点共线的退化覆盖区域。"""
    unique_points = np.unique(np.asarray(uv, dtype=float), axis=0)
    if unique_points.shape[0] < 3:
        raise ValueError("像素到TCP标定至少需要 3 个不同的像素点")
    # OpenCV 的 convexHull 仅接受 float32 或 int32。
    hull = cv2.convexHull(unique_points.astype(np.float32)).reshape(-1, 2)
    if hull.shape[0] < 3 or abs(cv2.contourArea(hull.astype(np.float32))) <= 1e-9:
        raise ValueError("标定像素点共线，无法建立二维像素到TCP标定")
    return hull


def build_pixel_to_tcp_calibration(
    uv: np.ndarray,
    selected_model_name: str,
    selected_model: Dict[str, Any],
    selected_cv_summary: Mapping[str, Any],
    calibration_subject: Optional[str] = None,
) -> Dict[str, Any]:
    """生成供运行代码加载的像素到 TCP XYZ 标定文件内容。"""
    hull = pixel_convex_hull(uv)
    uv = np.asarray(uv, dtype=float)
    parameters = model_to_jsonable(selected_model)
    cv = model_to_jsonable(dict(selected_cv_summary))
    calibration = {
        "schema_version": TCP_CALIBRATION_SCHEMA_VERSION,
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
        "model": {
            "name": selected_model_name,
            "parameters": parameters,
        },
        "metrics": {
            "sample_count": int(len(uv)),
            "cross_validation": {
                "folds": int(cv["折数"]),
                "three_dimensional_rmse_mm": float(cv["CV三维RMSE"]),
                "three_dimensional_mae_mm": float(cv["CV三维MAE"]),
                "three_dimensional_p95_mm": float(cv["CV三维P95"]),
                "three_dimensional_max_error_mm": float(cv["CV三维最大误差"]),
                "axis_rmse_mm": [
                    float(cv["CV_X_RMSE"]),
                    float(cv["CV_Y_RMSE"]),
                    float(cv["CV_Z_RMSE"]),
                ],
            },
            "full_training_three_dimensional_rmse_mm": float(cv["全数据训练RMSE"]),
        },
        "coverage": {
            "sample_count": int(len(uv)),
            "pixel_range": {
                "u": [float(np.min(uv[:, 0])), float(np.max(uv[:, 0]))],
                "v": [float(np.min(uv[:, 1])), float(np.max(uv[:, 1]))],
            },
            "pixel_convex_hull": hull.tolist(),
            # 样本凸包只描述本次采样覆盖，不代表正式任务的最大抓取范围。
            "extrapolation_policy": "diagnostic_only",
        },
        "usage_note": (
            "仅适用于高位相机的高位检测像素。输出仅含 TCP XYZ；"
            "调用方须自行提供或沿用 TCP 姿态 R/P/YAW。"
        ),
    }
    if calibration_subject:
        calibration["calibration_subject"] = str(calibration_subject)
    return calibration


def write_pixel_to_tcp_calibration(
    output_dir: Path,
    uv: np.ndarray,
    selected_model_name: str,
    selected_model: Dict[str, Any],
    selected_cv_summary: Mapping[str, Any],
    filename: str = TCP_CALIBRATION_FILENAME,
    calibration_subject: Optional[str] = None,
) -> Path:
    """写出可直接由 competition_lib 加载的 YAML 标定结果。"""
    calibration = build_pixel_to_tcp_calibration(
        uv,
        selected_model_name,
        selected_model,
        selected_cv_summary,
        calibration_subject=calibration_subject,
    )
    output_path = output_dir / filename
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(calibration, file, allow_unicode=True, sort_keys=False)
    return output_path


# ----------------------------- 绘图 -----------------------------
def configure_matplotlib() -> None:
    """显式注册系统中的中文字体，避免字体名称不匹配时退回到 DejaVu Sans。"""
    # 本机的字体文件实际注册名常为 “Noto Sans CJK JP”，而非 “... SC”。
    # 直接按文件注册可避免不同 Linux 发行版中的字体别名差异。
    font_paths = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ]
    font_names: List[str] = []
    for font_path in font_paths:
        if not font_path.is_file():
            continue
        try:
            font_manager.fontManager.addfont(str(font_path))
            font_names.append(font_manager.FontProperties(fname=str(font_path)).get_name())
        except (OSError, RuntimeError):
            # 尝试下一个候选字体；若均不可用，仍可正常生成英文图表。
            continue

    if not font_names:
        print(
            "[WARN] 未找到可用中文字体，图表中的中文可能无法显示。"
            "请安装 Noto Sans CJK 或 Droid Sans Fallback。",
            file=sys.stderr,
        )

    # 保留常见字体名作为非本机环境的兜底，但不把 DejaVu 放在中文字体之前。
    plt.rcParams["font.sans-serif"] = font_names + [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Microsoft YaHei",
        "SimHei",
        "Droid Sans Fallback",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def set_axes_equal_3d(ax: Any, points: np.ndarray) -> None:
    spans = np.ptp(points, axis=0)
    spans[spans < 1e-9] = 1.0
    ax.set_box_aspect(spans)


def plot_plane_fit(points: np.ndarray, fit: PlaneFit, title: str, output_path: Path) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=32)

    # 在平面基坐标覆盖范围上画拟合平面。
    plane_xy = (points - fit.centroid) @ fit.basis.T
    s_min, t_min = plane_xy.min(axis=0)
    s_max, t_max = plane_xy.max(axis=0)
    ss, tt = np.meshgrid(np.linspace(s_min, s_max, 15), np.linspace(t_min, t_max, 15))
    surface = (
        fit.centroid[None, None, :]
        + ss[..., None] * fit.basis[0][None, None, :]
        + tt[..., None] * fit.basis[1][None, None, :]
    )
    ax.plot_surface(surface[..., 0], surface[..., 1], surface[..., 2], alpha=0.25)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        f"{title}\n正交RMSE={fit.rmse:.6g}, 最大距离={fit.max_abs:.6g}, "
        f"σ3/σ2={fit.sigma3_over_sigma2:.3e}"
    )
    set_axes_equal_3d(ax, points)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_mapping_cv(cv_summary: pd.DataFrame, output_path: Path) -> None:
    ordered = cv_summary.sort_values("CV三维RMSE")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ordered["模型"], ordered["CV三维RMSE"])
    ax.set_ylabel("K折交叉验证三维 RMSE")
    ax.set_title("像素到 TCP 映射模型泛化误差")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_mapping_oof_errors(
    oof_table: pd.DataFrame,
    model_names: Sequence[str],
    output_path: Path,
    subject_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(oof_table))
    for model_name in model_names:
        ax.plot(x, oof_table[f"{model_name}_OOF三维误差"], marker="o", markersize=3, label=model_name)
    ax.set_xlabel(f"{subject_label}样本索引")
    ax.set_ylabel("OOF 三维欧氏误差")
    ax.set_title(f"{subject_label}各映射模型逐点交叉验证误差")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_tray_grid_oof_errors(
    grid_oof_table: pd.DataFrame,
    selected_model_name: str,
    output_path: Path,
) -> None:
    """按托盘行列绘制自动选中模型的 OOF 三维误差，定位标定薄弱格点。"""
    error_col = f"{selected_model_name}_OOF三维误差"
    rows = pd.to_numeric(grid_oof_table[COL_TRAY_ROW], errors="coerce")
    cols = pd.to_numeric(grid_oof_table[COL_TRAY_COL], errors="coerce")
    errors = pd.to_numeric(grid_oof_table[error_col], errors="coerce")
    valid = np.isfinite(rows) & np.isfinite(cols) & np.isfinite(errors)
    if not valid.any():
        raise ValueError("托盘格点 OOF 误差图缺少有效的行、列或误差数据")

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        cols[valid],
        rows[valid],
        c=errors[valid],
        cmap="YlOrRd",
        s=86,
        edgecolors="black",
        linewidths=0.5,
    )
    for row, col, error in zip(rows[valid], cols[valid], errors[valid]):
        ax.annotate(f"{error:.2f}", (col, row), xytext=(4, 4), textcoords="offset points", fontsize=8)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("OOF 三维误差（毫米）")
    ax.set_xlabel("托盘列")
    ax.set_ylabel("托盘行")
    ax.set_title(f"托盘格点 OOF 误差（{selected_model_name}）")
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


# ----------------------------- 主流程 -----------------------------
def run_geometry_analysis(args: argparse.Namespace, spec: AnalysisSpec = BLOCK_ANALYSIS_SPEC) -> None:
    """执行方块或托盘的共用几何分析、诊断输出与可调用标定导出。"""
    input_path = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def output_path(filename: str) -> Path:
        return output_dir / f"{spec.output_prefix}{filename}"

    df, encoding = read_csv_auto(input_path)
    block_df = prepare_analysis_rows(df, spec)
    key_cols = build_key_columns(block_df)

    success_rows, success_validation = extract_success_rows(block_df, key_cols)

    # 成功行的 TCP 点
    success_rows = success_rows[finite_rows(success_rows, COL_TCP)].copy()
    if args.expected_count is not None and len(success_rows) != args.expected_count:
        print(
            f"[WARN] 有效成功 TCP 点数量={len(success_rows)}，与 --expected-count={args.expected_count} 不一致。",
            file=sys.stderr,
        )

    tcp_points = success_rows[COL_TCP].to_numpy(dtype=float)
    tcp_plane = fit_plane_svd(tcp_points)
    tcp_judge = plane_judgement(
        tcp_plane,
        args.plane_rmse_tol,
        args.plane_max_tol,
        args.planarity_ratio_tol,
    )

    # 每个样本的高位像素、世界坐标和对象特有字段必须在各事件行间一致。
    high_task, high_consistency = extract_unique_task_values(
        block_df,
        key_cols,
        [*COL_PIXEL, *COL_WORLD, *spec.extra_unique_columns],
        consistency_tol=args.unique_value_tol,
    )
    inconsistent = high_consistency[~high_consistency["全部唯一"]]
    if not inconsistent.empty:
        inconsistent.to_csv(output_path("高位字段不唯一_异常任务.csv"), index=False, encoding="utf-8-sig")
        raise ValueError(
            f"部分{spec.sample_label}的高位像素/世界坐标在事件行之间并不唯一。"
            f"详情已写入：{output_path('高位字段不唯一_异常任务.csv').name}"
        )

    high_valid = high_task[finite_rows(high_task, COL_WORLD)].copy()
    world_points = high_valid[COL_WORLD].to_numpy(dtype=float)
    world_plane = fit_plane_svd(world_points)
    world_judge = plane_judgement(
        world_plane,
        args.plane_rmse_tol,
        args.plane_max_tol,
        args.planarity_ratio_tol,
    )

    rotation = parse_rotation_matrix(args.world_to_tcp_rotation)
    angle_deg, world_normal_in_tcp = normal_angle_deg(
        tcp_plane.normal,
        world_plane.normal,
        rotation,
    )
    if args.parallel_angle_tol_deg is None:
        parallel_pass: Optional[bool] = None
    else:
        parallel_pass = angle_deg <= args.parallel_angle_tol_deg

    # 像素 -> TCP：按任务键连接高位字段和成功 TCP。
    success_min = success_rows.loc[:, [*key_cols, COL_CATEGORY, *COL_TCP]].copy()
    mapping_df = high_task.merge(
        success_min,
        on=[*key_cols, COL_CATEGORY],
        how="inner",
        validate="one_to_one",
    )
    mapping_df = mapping_df[
        finite_rows(mapping_df, [*COL_PIXEL, *COL_TCP])
    ].sort_values(list(key_cols)).reset_index(drop=True)

    uv = mapping_df[COL_PIXEL].to_numpy(dtype=float)
    xyz = mapping_df[COL_TCP].to_numpy(dtype=float)
    if len(mapping_df) < 10:
        print(
            f"[WARN] 映射有效样本只有 {len(mapping_df)} 个，复杂模型的交叉验证可信度有限。",
            file=sys.stderr,
        )

    model_names = ["affine", "homography", "poly2", "poly3"]
    cv_rows: List[Dict[str, Any]] = []
    oof_predictions: Dict[str, np.ndarray] = {}
    final_models: Dict[str, Dict[str, Any]] = {}

    for model_name in model_names:
        summary, oof_pred = cross_validate_mapping(
            model_name,
            uv,
            xyz,
            folds=args.cv_folds,
            seed=args.random_seed,
        )
        cv_rows.append(summary)
        oof_predictions[model_name] = oof_pred

        final_model = fit_mapping(model_name, uv, xyz)
        full_pred = predict_mapping(model_name, final_model, uv)
        full_error = np.linalg.norm(full_pred - xyz, axis=1)
        summary["全数据训练RMSE"] = float(np.sqrt(np.mean(full_error**2)))
        final_models[model_name] = final_model

    cv_summary = pd.DataFrame(cv_rows)
    model_choice = choose_simplest_near_best_model(
        cv_summary,
        slack=args.simple_model_slack,
    )
    selected_model_name = str(model_choice["推荐最简模型"])
    selected_cv_summary = cv_summary.loc[
        cv_summary["模型"] == selected_model_name
    ].iloc[0].to_dict()
    calibration_path = write_pixel_to_tcp_calibration(
        output_dir,
        uv,
        selected_model_name,
        final_models[selected_model_name],
        selected_cv_summary,
        filename=spec.calibration_filename,
        calibration_subject=spec.calibration_subject,
    )

    # 输出逐点 OOF 预测，便于快速定位离群样本，而不靠人工逐行看原日志。
    metadata_cols = list(dict.fromkeys([*key_cols, COL_CATEGORY, *spec.extra_unique_columns]))
    oof_table = mapping_df.loc[:, [*metadata_cols, *COL_PIXEL, *COL_TCP]].copy()
    for model_name, pred in oof_predictions.items():
        oof_table[f"{model_name}_预测TCP_X"] = pred[:, 0]
        oof_table[f"{model_name}_预测TCP_Y"] = pred[:, 1]
        oof_table[f"{model_name}_预测TCP_Z"] = pred[:, 2]
        oof_table[f"{model_name}_OOF三维误差"] = np.linalg.norm(pred - xyz, axis=1)

    # 输出平面逐点距离。
    tcp_point_table = success_rows.loc[:, [*metadata_cols, *COL_TCP]].copy()
    tcp_point_table["到TCP拟合平面有符号距离"] = tcp_plane.signed_distances
    tcp_point_table["到TCP拟合平面绝对距离"] = np.abs(tcp_plane.signed_distances)

    world_point_table = high_valid.loc[:, [*metadata_cols, *COL_WORLD]].copy()
    world_point_table["到高位世界拟合平面有符号距离"] = world_plane.signed_distances
    world_point_table["到高位世界拟合平面绝对距离"] = np.abs(world_plane.signed_distances)

    plane_summary = pd.DataFrame(
        [
            {
                "平面": "成功TCP平面",
                "点数": len(tcp_points),
                "法向X": tcp_plane.normal[0],
                "法向Y": tcp_plane.normal[1],
                "法向Z": tcp_plane.normal[2],
                "平面常数d": tcp_plane.d,
                "正交RMSE": tcp_plane.rmse,
                "正交MAE": tcp_plane.mae,
                "P95绝对距离": tcp_plane.p95_abs,
                "最大绝对距离": tcp_plane.max_abs,
                "sigma3_over_sigma2": tcp_plane.sigma3_over_sigma2,
                "法向方差占比": tcp_plane.normal_variance_ratio,
                **tcp_judge,
            },
            {
                "平面": "高位世界坐标平面",
                "点数": len(world_points),
                "法向X": world_plane.normal[0],
                "法向Y": world_plane.normal[1],
                "法向Z": world_plane.normal[2],
                "平面常数d": world_plane.d,
                "正交RMSE": world_plane.rmse,
                "正交MAE": world_plane.mae,
                "P95绝对距离": world_plane.p95_abs,
                "最大绝对距离": world_plane.max_abs,
                "sigma3_over_sigma2": world_plane.sigma3_over_sigma2,
                "法向方差占比": world_plane.normal_variance_ratio,
                **world_judge,
            },
        ]
    )

    parallel_summary = {
        "高位世界法向是否已旋转到TCP坐标系": rotation is not None,
        "高位世界法向_用于比较": world_normal_in_tcp.tolist(),
        "TCP法向": tcp_plane.normal.tolist(),
        "两平面锐角_度": angle_deg,
        "平行角阈值_度": args.parallel_angle_tol_deg,
        "平行判定通过": parallel_pass,
        "注意": (
            "已使用 world->TCP 旋转矩阵比较法向。"
            if rotation is not None
            else "未提供坐标系旋转；该角度只有在两组XYZ使用相同轴方向时才有物理意义。"
        ),
    }

    # 文件输出
    success_validation.to_csv(output_path("成功行唯一性检查.csv"), index=False, encoding="utf-8-sig")
    high_consistency.to_csv(output_path("高位字段唯一性检查.csv"), index=False, encoding="utf-8-sig")
    tcp_point_table.to_csv(output_path("TCP平面逐点距离.csv"), index=False, encoding="utf-8-sig")
    world_point_table.to_csv(output_path("高位世界平面逐点距离.csv"), index=False, encoding="utf-8-sig")
    plane_summary.to_csv(output_path("平面拟合汇总.csv"), index=False, encoding="utf-8-sig")
    cv_summary.sort_values("CV三维RMSE").to_csv(
        output_path("像素到TCP映射_交叉验证汇总.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    oof_table.to_csv(output_path("像素到TCP映射_OOF逐点预测.csv"), index=False, encoding="utf-8-sig")
    if spec.tray_grid_diagnostic:
        selected_error_col = f"{selected_model_name}_OOF三维误差"
        grid_columns = [*metadata_cols, *COL_PIXEL, *COL_TCP, selected_error_col]
        oof_table.loc[:, grid_columns].to_csv(
            output_path("格点OOF误差.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    json_report = {
        "input": {
            "path": str(input_path),
            "encoding": encoding,
            "key_columns": key_cols,
            "raw_rows": len(df),
            "analysis_rows": len(block_df),
            # 兼容已有方块分析报告的字段名；托盘报告同样表示分析样本行数。
            "block_rows": len(block_df),
            "success_points": len(tcp_points),
            "mapping_points": len(mapping_df),
        },
        "analysis_subject": spec.calibration_subject or "block",
        "thresholds": {
            "plane_rmse_tol": args.plane_rmse_tol,
            "plane_max_tol": args.plane_max_tol,
            "planarity_ratio_tol": args.planarity_ratio_tol,
            "parallel_angle_tol_deg": args.parallel_angle_tol_deg,
            "unique_value_tol": args.unique_value_tol,
        },
        "tcp_plane": {**tcp_plane.to_jsonable(), "judgement": tcp_judge},
        "world_plane": {**world_plane.to_jsonable(), "judgement": world_judge},
        "parallelism": parallel_summary,
        "mapping_model_choice": model_choice,
        "pixel_to_tcp_calibration": {
            "path": str(calibration_path),
            "selected_model": selected_model_name,
        },
        "mapping_models": {
            name: model_to_jsonable(model) for name, model in final_models.items()
        },
    }
    with output_path("分析报告.json").open("w", encoding="utf-8") as f:
        json.dump(json_report, f, ensure_ascii=False, indent=2)

    if not args.skip_plots:
        configure_matplotlib()
        plot_plane_fit(
            tcp_points,
            tcp_plane,
            f"{spec.subject_label}伺服成功 TCP 点平面拟合",
            output_path("TCP平面拟合.png"),
        )
        plot_plane_fit(
            world_points,
            world_plane,
            f"{spec.subject_label}高位世界坐标点平面拟合",
            output_path("高位世界平面拟合.png"),
        )
        plot_mapping_cv(cv_summary, output_path("像素到TCP映射_CV误差.png"))
        plot_mapping_oof_errors(
            oof_table,
            model_names,
            output_path("像素到TCP映射_逐点OOF误差.png"),
            spec.subject_label,
        )
        if spec.tray_grid_diagnostic:
            plot_tray_grid_oof_errors(
                oof_table,
                selected_model_name,
                output_path("格点OOF误差.png"),
            )

    print("=" * 72)
    print(f"输入文件：{input_path}")
    print(f"输出目录：{output_dir}")
    print(f"{spec.subject_label}成功 TCP 点数：{len(tcp_points)}")
    print(f"高位世界点数：{len(world_points)}")
    print(f"映射样本数：{len(mapping_df)}")
    print(f"像素到TCP标定文件：{calibration_path}")
    print("-" * 72)
    print(
        "TCP 平面："
        f"RMSE={tcp_plane.rmse:.6g}, max={tcp_plane.max_abs:.6g}, "
        f"sigma3/sigma2={tcp_plane.sigma3_over_sigma2:.3e}, 判定={tcp_judge['平面判定通过']}"
    )
    print(
        "世界平面："
        f"RMSE={world_plane.rmse:.6g}, max={world_plane.max_abs:.6g}, "
        f"sigma3/sigma2={world_plane.sigma3_over_sigma2:.3e}, 判定={world_judge['平面判定通过']}"
    )
    print(
        f"两平面法向锐角={angle_deg:.6g}°，平行判定={parallel_pass}；"
        + parallel_summary["注意"]
    )
    print("-" * 72)
    print(cv_summary.sort_values("CV三维RMSE").to_string(index=False))
    print("-" * 72)
    print("映射模型机械选择：")
    print(json.dumps(model_choice, ensure_ascii=False, indent=2))
    print("=" * 72)


def analyze(args: argparse.Namespace) -> None:
    """兼容原方块标定入口，内部使用共用分析核心。"""
    run_geometry_analysis(args, BLOCK_ANALYSIS_SPEC)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="视觉伺服粗定位：平面拟合、平行性和像素到TCP映射比较",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_csv", default=DEFAULT_INPUT_CSV, help="输入 CSV 路径")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--expected-count", type=int, default=DEFAULT_EXPECTED_COUNT, help="期望的伺服成功方块数量；不一致只警告")

    # 不替用户擅自规定“平面”的物理公差。只有显式传入才给布尔结论。
    parser.add_argument("--plane-rmse-tol", type=float, default=DEFAULT_PLANE_RMSE_TOL, help="平面正交 RMSE 允许值")
    parser.add_argument("--plane-max-tol", type=float, default=DEFAULT_PLANE_MAX_TOL, help="点到平面的最大绝对距离允许值")
    parser.add_argument("--planarity-ratio-tol", type=float, default=DEFAULT_PLANARITY_RATIO_TOL, help="奇异值比 sigma3/sigma2 允许值")
    parser.add_argument("--parallel-angle-tol-deg", type=float, default=DEFAULT_PARALLEL_ANGLE_TOL_DEG, help="平行判定允许的最大法向锐角")

    parser.add_argument(
        "--world-to-tcp-rotation",
        default=DEFAULT_WORLD_TO_TCP_ROTATION,
        help=(
            "高位世界坐标系到TCP坐标系的3x3旋转矩阵，按行给9个逗号分隔数。"
            "若不提供，法向夹角只在两坐标系轴方向一致时有效"
        ),
    )
    parser.add_argument("--unique-value-tol", type=float, default=DEFAULT_UNIQUE_VALUE_TOL, help="同一方块跨事件字段一致性容差")
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS, help="映射模型 K 折交叉验证折数")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help="交叉验证随机种子")
    parser.add_argument(
        "--simple-model-slack",
        type=float,
        default=DEFAULT_SIMPLE_MODEL_SLACK,
        help="在最优CV RMSE的该比例范围内，优先选择更简单模型",
    )
    parser.add_argument("--skip-plots", action="store_true", default=DEFAULT_SKIP_PLOTS, help="不生成 PNG 图")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
