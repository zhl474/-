"""高位相机像素到机械臂 TCP 位置的离线标定结果加载与预测。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


CALIBRATION_SCHEMA_VERSION = 1
CALIBRATION_TYPE = "pixel_to_tcp_position"
_DENOMINATOR_EPS = 1e-12


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} 必须是字典")
    return value


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数值数组") from exc
    shape_matches = array.ndim == len(shape) and all(
        expected == -1 or actual == expected
        for actual, expected in zip(array.shape, shape)
    )
    if not shape_matches or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须是形状为 {shape} 的有限数值数组")
    return array


def _polynomial_feature_names(degree: int) -> list[str]:
    names: list[str] = []
    for total_degree in range(degree + 1):
        for u_power in range(total_degree, -1, -1):
            v_power = total_degree - u_power
            if u_power == 0 and v_power == 0:
                names.append("1")
                continue
            parts = []
            if u_power:
                parts.append("u" if u_power == 1 else f"u^{u_power}")
            if v_power:
                parts.append("v" if v_power == 1 else f"v^{v_power}")
            names.append("*".join(parts))
    return names


def _polynomial_features(uv_normalized: np.ndarray, degree: int) -> np.ndarray:
    u, v = uv_normalized
    features = []
    for total_degree in range(degree + 1):
        for u_power in range(total_degree, -1, -1):
            v_power = total_degree - u_power
            features.append((u**u_power) * (v**v_power))
    return np.asarray(features, dtype=float)


def _point_in_convex_hull(pixel_xy: np.ndarray, hull: np.ndarray, tolerance: float) -> bool:
    """用凸包边的叉积判断像素点是否在标定覆盖范围内（含边界）。"""
    edges = np.roll(hull, -1, axis=0) - hull
    offsets = pixel_xy - hull
    cross_products = edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
    return bool(
        np.all(cross_products >= -tolerance)
        or np.all(cross_products <= tolerance)
    )


@dataclass(frozen=True)
class PixelToTcpCalibration:
    """已校验的像素到 TCP XYZ 标定模型。"""

    model_name: str
    parameters: Mapping[str, Any]
    pixel_convex_hull: np.ndarray
    metadata: Mapping[str, Any]

    def is_pixel_within_coverage(self, pixel_xy: Sequence[float], tolerance_px: float = 1e-6) -> bool:
        """判断输入像素是否处于采集样本形成的凸包内。"""
        pixel = _finite_array(pixel_xy, (2,), "pixel_xy")
        try:
            tolerance = float(tolerance_px)
        except (TypeError, ValueError) as exc:
            raise ValueError("tolerance_px 必须是非负有限数值") from exc
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance_px 必须是非负有限数值")
        return _point_in_convex_hull(pixel, self.pixel_convex_hull, tolerance)

    def predict(self, pixel_xy: Sequence[float], *, allow_extrapolation: bool = False) -> np.ndarray:
        """将一组高位检测像素 [X, Y] 转换为 TCP 位置 [X, Y, Z]，单位为毫米。"""
        pixel = _finite_array(pixel_xy, (2,), "pixel_xy")
        if not allow_extrapolation and not self.is_pixel_within_coverage(pixel):
            raise ValueError(
                "输入像素位于标定样本覆盖凸包之外，默认拒绝外推；"
                "确认安全后可传入 allow_extrapolation=True"
            )

        if self.model_name in {"affine", "poly2", "poly3"}:
            prediction = self._predict_polynomial(pixel)
        elif self.model_name == "homography":
            prediction = self._predict_homography(pixel)
        else:  # load 函数已校验，这里保留防御性分支。
            raise ValueError(f"不支持的标定模型：{self.model_name}")

        if not np.all(np.isfinite(prediction)):
            raise ValueError("标定预测结果包含非有限数值")
        return prediction

    def _predict_polynomial(self, pixel: np.ndarray) -> np.ndarray:
        degree = int(self.parameters["degree"])
        mean = _finite_array(self.parameters["uv_mean"], (2,), "model.parameters.uv_mean")
        scale = _finite_array(self.parameters["uv_scale"], (2,), "model.parameters.uv_scale")
        if np.any(scale <= _DENOMINATOR_EPS):
            raise ValueError("model.parameters.uv_scale 必须大于零")
        feature_count = len(_polynomial_feature_names(degree))
        coefficient = _finite_array(
            self.parameters["coef"],
            (feature_count, 3),
            "model.parameters.coef",
        )
        normalized = (pixel - mean) / scale
        return _polynomial_features(normalized, degree) @ coefficient

    def _predict_homography(self, pixel: np.ndarray) -> np.ndarray:
        matrix = _finite_array(self.parameters["H"], (3, 3), "model.parameters.H")
        centroid = _finite_array(
            self.parameters["plane_centroid"],
            (3,),
            "model.parameters.plane_centroid",
        )
        basis = _finite_array(
            self.parameters["plane_basis"],
            (2, 3),
            "model.parameters.plane_basis",
        )
        homogeneous = matrix @ np.array([pixel[0], pixel[1], 1.0], dtype=float)
        if abs(homogeneous[2]) <= _DENOMINATOR_EPS:
            raise ValueError("单应标定的齐次分母接近零，无法预测 TCP 位置")
        plane_xy = homogeneous[:2] / homogeneous[2]
        return centroid + plane_xy @ basis


def _validate_model(name: Any, parameters: Mapping[str, Any]) -> str:
    if not isinstance(name, str) or name not in {"affine", "homography", "poly2", "poly3"}:
        raise ValueError(f"model.name 不支持：{name}")

    if name == "homography":
        _finite_array(parameters.get("H"), (3, 3), "model.parameters.H")
        _finite_array(parameters.get("plane_centroid"), (3,), "model.parameters.plane_centroid")
        _finite_array(parameters.get("plane_basis"), (2, 3), "model.parameters.plane_basis")
        return str(name)

    expected_degree = {"affine": 1, "poly2": 2, "poly3": 3}[str(name)]
    try:
        degree = int(parameters.get("degree"))
    except (TypeError, ValueError) as exc:
        raise ValueError("model.parameters.degree 必须是整数") from exc
    if degree != expected_degree:
        raise ValueError(f"{name} 的 degree 必须为 {expected_degree}")
    _finite_array(parameters.get("uv_mean"), (2,), "model.parameters.uv_mean")
    scale = _finite_array(parameters.get("uv_scale"), (2,), "model.parameters.uv_scale")
    if np.any(scale <= _DENOMINATOR_EPS):
        raise ValueError("model.parameters.uv_scale 必须大于零")
    feature_names = parameters.get("feature_names")
    if feature_names != _polynomial_feature_names(degree):
        raise ValueError("model.parameters.feature_names 与多项式次数不匹配")
    _finite_array(
        parameters.get("coef"),
        (len(feature_names), 3),
        "model.parameters.coef",
    )
    return str(name)


def load_pixel_to_tcp_calibration(path: str | Path) -> PixelToTcpCalibration:
    """读取并校验 `像素到TCP标定结果.yaml`，返回可直接预测的标定对象。"""
    calibration_path = Path(path).expanduser()
    try:
        with calibration_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except OSError as exc:
        raise ValueError(f"无法读取 TCP 标定文件：{calibration_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"TCP 标定文件不是合法 YAML：{calibration_path}") from exc

    document = _require_mapping(data, "标定文件根节点")
    if document.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError(f"仅支持 schema_version={CALIBRATION_SCHEMA_VERSION} 的 TCP 标定文件")
    if document.get("calibration_type") != CALIBRATION_TYPE:
        raise ValueError(f"calibration_type 必须为 {CALIBRATION_TYPE}")

    input_info = _require_mapping(document.get("input"), "input")
    output_info = _require_mapping(document.get("output"), "output")
    if input_info.get("coordinate") != "high_detection_pixel_xy" or input_info.get("unit") != "pixel":
        raise ValueError("input 必须描述 high_detection_pixel_xy，单位必须为 pixel")
    if output_info.get("coordinate") != "tcp_position_xyz" or output_info.get("unit") != "mm":
        raise ValueError("output 必须描述 tcp_position_xyz，单位必须为 mm")

    model = _require_mapping(document.get("model"), "model")
    parameters = _require_mapping(model.get("parameters"), "model.parameters")
    model_name = _validate_model(model.get("name"), parameters)

    coverage = _require_mapping(document.get("coverage"), "coverage")
    hull = _finite_array(coverage.get("pixel_convex_hull"), (-1, 2), "coverage.pixel_convex_hull")
    if hull.shape[0] < 3:
        raise ValueError("coverage.pixel_convex_hull 至少需要 3 个顶点")
    # 由凸包首尾边计算面积，拒绝退化成直线的标定覆盖区域。
    area = abs(0.5 * np.sum(hull[:, 0] * np.roll(hull[:, 1], -1) - hull[:, 1] * np.roll(hull[:, 0], -1)))
    if area <= _DENOMINATOR_EPS:
        raise ValueError("coverage.pixel_convex_hull 退化为直线，不能用于二维像素标定")

    return PixelToTcpCalibration(
        model_name=model_name,
        parameters=dict(parameters),
        pixel_convex_hull=hull,
        metadata=dict(document),
    )
