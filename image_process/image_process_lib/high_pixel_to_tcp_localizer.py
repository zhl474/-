"""用方块和托盘的独立标定，将高位检测像素转换为安全观察位姿。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import yaml

from image_process_lib.pixel_to_tcp_calibration import (
    PixelToTcpCalibration,
    load_pixel_to_tcp_calibration,
)
from image_process_lib.sucker_offset import classify_pixel_side


PACKAGE_DIR = Path(__file__).resolve().parents[1]
EXECUTION_CONFIG_PATH = PACKAGE_DIR.parent / "competition" / "config" / "execution.yaml"
DEFAULT_TCP_MIN_XY = (-444.224, -263.279)
DEFAULT_TCP_MAX_XYZ = (-148.17, 315.925, None)
_SUBJECT_LABELS = {"block": "方块", "tray": "托盘"}
_AXIS_NAMES = ("X", "Y", "Z")


def _load_default_tcp_min_xyz() -> tuple[float, float, float]:
    """读取默认 TCP 安全边界的 Z，避免独立定位器保留另一份最低高度。"""
    try:
        with EXECUTION_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            execution_config = yaml.safe_load(config_file) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(
            f"无法读取唯一 TCP 高度配置 {EXECUTION_CONFIG_PATH}: {exc}"
        ) from exc
    motion_config = execution_config.get("motion")
    if not isinstance(motion_config, dict) or "minimum_tcp_z_mm" not in motion_config:
        raise ValueError(
            "execution.yaml 缺少唯一安全高度字段 motion.minimum_tcp_z_mm"
        )
    raw_minimum_tcp_z_mm = motion_config["minimum_tcp_z_mm"]
    if isinstance(raw_minimum_tcp_z_mm, bool):
        raise ValueError("motion.minimum_tcp_z_mm 必须是大于 0 的有限数值")
    try:
        minimum_tcp_z_mm = float(raw_minimum_tcp_z_mm)
    except (TypeError, ValueError) as exc:
        raise ValueError("motion.minimum_tcp_z_mm 必须是大于 0 的有限数值") from exc
    if not np.isfinite(minimum_tcp_z_mm) or minimum_tcp_z_mm <= 0.0:
        raise ValueError("motion.minimum_tcp_z_mm 必须是大于 0 的有限数值")
    return (*DEFAULT_TCP_MIN_XY, minimum_tcp_z_mm)


@dataclass(frozen=True)
class HighTcpSafetyAssessment:
    """一次高位像素定位及其实际待执行 TCP 的结构化安全评估。"""

    subject: str
    pixel_xy: tuple[float, float]
    predicted_tcp_xyz: tuple[float, float, float]
    safety_tcp_xyz: tuple[float, float, float]
    safety_min_xyz: tuple[float, float, float]
    safety_max_xyz: tuple[float, float, float]
    violated_axes: tuple[str, ...]

    @property
    def safe(self) -> bool:
        """没有任何坐标轴越界时视为安全。"""
        return not self.violated_axes


def _finite_vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须包含 {size} 个数值") from exc
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须包含 {size} 个有限数值")
    return array


def _validate_safety_bounds(
    tcp_min_xyz: Sequence[float],
    tcp_max_xyz: Sequence[Optional[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """校验 TCP 安全边界；最大 Z 为 None 时表示不设置上限。"""
    minimum = _finite_vector(tcp_min_xyz, 3, "tcp_min_xyz")
    try:
        maximum_values = list(tcp_max_xyz)
    except TypeError as exc:
        raise ValueError("tcp_max_xyz 必须包含 3 个数值，Z 可为 None") from exc
    if len(maximum_values) != 3:
        raise ValueError("tcp_max_xyz 必须包含 3 个数值，Z 可为 None")
    if maximum_values[0] is None or maximum_values[1] is None:
        raise ValueError("tcp_max_xyz 的 X、Y 上限不能为 None")
    if maximum_values[2] is None:
        maximum_values[2] = np.inf
    try:
        maximum = np.asarray(maximum_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("tcp_max_xyz 必须包含有效数值，Z 可为 None") from exc
    if maximum.shape != (3,) or np.any(np.isnan(maximum)) or np.any(np.isneginf(maximum)):
        raise ValueError("tcp_max_xyz 必须包含有效上限，Z 可为 None")
    if not np.all(np.isfinite(maximum[:2])):
        raise ValueError("tcp_max_xyz 的 X、Y 上限必须是有限数值")
    if np.any(minimum > maximum):
        raise ValueError("tcp_min_xyz 不能大于对应的 tcp_max_xyz")
    return minimum, maximum


def _validate_fixed_tcp_z(
    fixed_tcp_z_mm: Optional[Mapping[str, float]],
    minimum_tcp_z_mm: float,
) -> Optional[dict[str, float]]:
    """校验固定 TCP Z 配置；None 表示继续使用标定 yaml 的 z_plane。"""
    if fixed_tcp_z_mm is None:
        return None
    if not isinstance(fixed_tcp_z_mm, Mapping):
        raise ValueError("fixed_tcp_z_mm 必须是含 block 和 tray 键的字典")
    fixed: dict[str, float] = {}
    for subject in ("block", "tray"):
        label = _SUBJECT_LABELS[subject]
        if subject not in fixed_tcp_z_mm:
            raise ValueError(
                "固定 TCP Z 必须同时提供 block 和 tray 两个常数，"
                f"缺少 {label}（{subject}）"
            )
        raw_value = fixed_tcp_z_mm[subject]
        if isinstance(raw_value, bool):
            raise ValueError(f"{label}固定 TCP Z 必须是有限数值")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}固定 TCP Z 必须是有限数值") from exc
        if not np.isfinite(value):
            raise ValueError(f"{label}固定 TCP Z 必须是有限数值")
        if value < minimum_tcp_z_mm:
            raise ValueError(
                f"{label}固定 TCP Z={value:.3f} mm 低于安全下限 "
                f"{minimum_tcp_z_mm:.3f} mm"
            )
        fixed[subject] = value
    return fixed


class HighPixelToTcpLocalizer:
    """加载方块和托盘标定，并生成带拍摄姿态 R/P/YAW 的观察位姿。"""

    def __init__(
        self,
        block_calibration_path: str | Path,
        tray_calibration_path: str | Path,
        shooting_pose: Sequence[float],
        tcp_min_xyz: Optional[Sequence[float]] = None,
        tcp_max_xyz: Sequence[Optional[float]] = DEFAULT_TCP_MAX_XYZ,
        safety_xy_offset: Sequence[float] = (0.0, 0.0),
        fixed_tcp_z_mm: Optional[Mapping[str, float]] = None,
        block_calibration_left_path: Optional[str | Path] = None,
        block_calibration_right_path: Optional[str | Path] = None,
        split_models: bool = False,
    ) -> None:
        shooting = _finite_vector(shooting_pose, 6, "高位拍摄位姿")
        self._shooting_rpy = shooting[3:6].copy()
        if tcp_min_xyz is None:
            tcp_min_xyz = _load_default_tcp_min_xyz()
        self._tcp_min_xyz, self._tcp_max_xyz = _validate_safety_bounds(
            tcp_min_xyz,
            tcp_max_xyz,
        )
        # 工作台为平面的固定 Z 模式：不传时保持标定 yaml z_plane 的原始行为。
        self._fixed_tcp_z_mm = _validate_fixed_tcp_z(
            fixed_tcp_z_mm,
            float(self._tcp_min_xyz[2]),
        )
        # 该偏移只用于检查实际待执行 TCP，不改变标定模型输出和后续任务位姿。
        self._safety_xy_offset = _finite_vector(
            safety_xy_offset,
            2,
            "safety_xy_offset",
        )
        if (block_calibration_left_path is None) != (
            block_calibration_right_path is None
        ):
            raise ValueError(
                "左右方块标定必须成对配置：block_calibration_left_path 与 "
                "block_calibration_right_path 要么同时提供，要么都不提供"
            )
        if split_models and (
            block_calibration_left_path is None
            or block_calibration_right_path is None
        ):
            raise ValueError(
                "分侧模型模式（split_models=True）必须同时提供左右方块标定文件"
            )
        self._split_models = bool(split_models)
        self._calibrations: dict[str, PixelToTcpCalibration] = {
            "block": self._load_subject_calibration(
                block_calibration_path,
                "block",
            ),
            "tray": self._load_subject_calibration(
                tray_calibration_path,
                "tray",
            ),
        }
        if block_calibration_left_path is not None:
            self._calibrations["block_left"] = self._load_subject_calibration(
                block_calibration_left_path,
                "block",
            )
            self._calibrations["block_right"] = self._load_subject_calibration(
                block_calibration_right_path,
                "block",
            )
        self._validate_calibration_consistency()

    def _validate_calibration_consistency(self) -> None:
        """全部已加载标定必须同 schema 版本，v2 时同 generation_id 批次。"""
        loaded = list(self._calibrations.values())
        schema_versions = {calibration.schema_version for calibration in loaded}
        if len(schema_versions) != 1:
            raise ValueError(
                "已加载标定的 schema 版本不一致（方块/左右方块/托盘），禁止混用"
            )
        if self._calibrations["block"].schema_version == 2:
            generations = {
                calibration.metadata.get("generation_id")
                for calibration in loaded
            }
            if len(generations) != 1:
                raise ValueError(
                    "已加载标定的 schema v2 批次不一致（方块/左右方块/托盘"
                    "必须来自同一次标定生成）"
                )

    @staticmethod
    def _load_subject_calibration(
        path: str | Path,
        subject: str,
    ) -> PixelToTcpCalibration:
        label = _SUBJECT_LABELS[subject]
        try:
            return load_pixel_to_tcp_calibration(path, expected_subject=subject)
        except ValueError as exc:
            raise ValueError(f"{label}像素到 TCP 标定加载失败（{path}）：{exc}") from exc

    def _subject_calibration(
        self,
        subject: str,
        pixel_xy: Sequence[float],
    ) -> PixelToTcpCalibration:
        if not isinstance(subject, str) or subject not in ("block", "tray"):
            raise ValueError(
                f"主体 {subject} 的高位像素 {pixel_xy!r} 无法定位："
                "主体仅支持 block 或 tray"
            )
        if subject == "block" and self._split_models:
            side = classify_pixel_side(pixel_xy)
            if (
                side is not None
                and "block_left" in self._calibrations
                and "block_right" in self._calibrations
            ):
                return self._calibrations[f"block_{side}"]
            # 像素缺失/无效（含 (0,0) 标记）时回退单模型。
            return self._calibrations["block"]
        return self._calibrations[subject]

    @property
    def fixed_tcp_z_mm(self) -> Optional[dict[str, float]]:
        """当前固定 TCP Z 常数；None 表示使用标定 yaml 的 z_plane。"""
        return None if self._fixed_tcp_z_mm is None else dict(self._fixed_tcp_z_mm)

    def update_dynamic_bounds(
        self,
        minimum_tcp_z_mm: Optional[float] = None,
        safety_xy_offset: Optional[Sequence[float]] = None,
        shooting_pose: Optional[Sequence[float]] = None,
    ) -> None:
        """轮间热更新来自 execution.yaml / visual_servo.yaml 的动态边界。

        标定 yaml 与 perception.yaml 的固定 Z 常数不在此刷新，仍随节点重启生效。
        """
        if minimum_tcp_z_mm is not None:
            if (
                isinstance(minimum_tcp_z_mm, bool)
                or not np.isfinite(minimum_tcp_z_mm)
                or minimum_tcp_z_mm <= 0.0
            ):
                raise ValueError("minimum_tcp_z_mm 必须是大于 0 的有限数值")
            self._tcp_min_xyz[2] = float(minimum_tcp_z_mm)
        if safety_xy_offset is not None:
            self._safety_xy_offset = _finite_vector(
                safety_xy_offset,
                2,
                "safety_xy_offset",
            )
        if shooting_pose is not None:
            shooting = _finite_vector(shooting_pose, 6, "高位拍摄位姿")
            self._shooting_rpy = shooting[3:6].copy()

    def calibration_summary(self, subject: str) -> dict:
        """只读暴露标定元信息，供定位全景导出留档。

        分侧模型模式下方块返回 {"模式": "左右分区", "左": {...}, "右": {...}}，
        顶层同时保留单模型摘要键（兼容既有调用与固定 Z 对照日志）。
        """
        calibration = self._subject_calibration(subject, (0.0, 0.0))
        summary = self._summary_of(calibration, subject)
        if (
            subject == "block"
            and self._split_models
            and "block_left" in self._calibrations
        ):
            summary["模式"] = "左右分区"
            summary["左"] = self._summary_of(
                self._calibrations["block_left"],
                "block",
            )
            summary["右"] = self._summary_of(
                self._calibrations["block_right"],
                "block",
            )
        return summary

    def _summary_of(
        self,
        calibration: PixelToTcpCalibration,
        subject: str,
    ) -> dict:
        document = calibration.metadata
        z_plane = calibration.z_plane_coefficients
        coverage = document.get("coverage")
        return {
            "schema_version": int(calibration.schema_version),
            "实验批次": document.get("generation_id"),
            "XY模型": calibration.model_name,
            "Z平面系数a_b_c": (
                None if z_plane is None else [float(value) for value in z_plane]
            ),
            # 固定 Z 模式下上方的 z_plane 系数只是留档对照，实际 Z 以该常数为准。
            "TCP_Z来源": (
                "calibration_z_plane"
                if self._fixed_tcp_z_mm is None
                else f"fixed_constant={self._fixed_tcp_z_mm[subject]:.3f}"
            ),
            "凸包顶点数": int(len(calibration.pixel_convex_hull)),
            "标定样本数": (
                None if not isinstance(coverage, dict) else coverage.get("sample_count")
            ),
        }

    def is_pixel_within_coverage(
        self,
        subject: str,
        pixel_xy: Sequence[float],
        tolerance_px: float = 0.0,
    ) -> bool:
        """判断像素是否在标定采集凸包内，仅作诊断不限制预测。"""
        calibration = self._subject_calibration(subject, pixel_xy)
        return calibration.is_pixel_within_coverage(pixel_xy, tolerance_px=tolerance_px)

    def predict_tcp_xyz(self, subject: str, pixel_xy: Sequence[float]) -> np.ndarray:
        """预测并校验 TCP XYZ；标定样本凸包不作为正式抓取范围。"""
        assessment = self.assess(subject, pixel_xy)
        tcp_xyz = np.asarray(assessment.predicted_tcp_xyz, dtype=float)
        if assessment.safe:
            return tcp_xyz.copy()

        label = _SUBJECT_LABELS[subject]
        maximum_text = [
            float(value) if np.isfinite(value) else None
            for value in self._tcp_max_xyz
        ]
        raise ValueError(
            f"{label}（{subject}）高位像素 {pixel_xy!r} 预测 TCP XYZ "
            f"{tcp_xyz.tolist()} 超出安全范围：最小值 {self._tcp_min_xyz.tolist()}，"
            f"最大值 {maximum_text}"
        )

    def assess(
        self,
        subject: str,
        pixel_xy: Sequence[float],
        safety_xy_offset: Optional[Sequence[float]] = None,
    ) -> HighTcpSafetyAssessment:
        """预测目标 TCP 并返回非抛出式安全结果；标定预测错误仍由调用方处理。

        safety_xy_offset 为 None 时使用构造/热更新时的全局偏移（旧调用行为不变）；
        传入时用该偏移做安全校验，供开环模式按目标分区（方块左右/托盘中间）使用
        不同的实际执行偏移。
        """
        calibration = self._subject_calibration(subject, pixel_xy)
        label = _SUBJECT_LABELS[subject]
        try:
            tcp_xyz = calibration.predict(pixel_xy)
        except ValueError as exc:
            raise ValueError(
                f"{label}（{subject}）高位像素 {pixel_xy!r} 的 TCP 标定预测失败：{exc}"
            ) from exc
        # 固定 Z 模式只在 XY 保留标定预测，Z 用平面常数覆盖（安全校验同样使用覆盖值）。
        if self._fixed_tcp_z_mm is not None:
            tcp_xyz[2] = self._fixed_tcp_z_mm[subject]

        safety_tcp_xyz = tcp_xyz.copy()
        if safety_xy_offset is None:
            safety_tcp_xyz[:2] += self._safety_xy_offset
        else:
            safety_tcp_xyz[:2] += _finite_vector(
                safety_xy_offset,
                2,
                "safety_xy_offset",
            )
        # 场地变动后 XY 安全范围已停用：XY 只保留有限性校验，范围拦截仅剩 Z 下限。
        violated_axes = tuple(
            axis_name
            for axis_name, value, minimum, maximum in zip(
                _AXIS_NAMES,
                safety_tcp_xyz,
                self._tcp_min_xyz,
                self._tcp_max_xyz,
            )
            if not np.isfinite(value)
            or (axis_name == "Z" and (value < minimum or value > maximum))
        )
        return HighTcpSafetyAssessment(
            subject=subject,
            pixel_xy=(float(pixel_xy[0]), float(pixel_xy[1])),
            predicted_tcp_xyz=tuple(float(value) for value in tcp_xyz),
            safety_tcp_xyz=tuple(float(value) for value in safety_tcp_xyz),
            safety_min_xyz=tuple(float(value) for value in self._tcp_min_xyz),
            safety_max_xyz=tuple(float(value) for value in self._tcp_max_xyz),
            violated_axes=violated_axes,
        )

    def locate(self, subject: str, pixel_xy: Sequence[float]) -> list[float]:
        """按主体生成 [X, Y, Z, R, P, YAW] 高位粗定位观察位姿。"""
        tcp_xyz = self.predict_tcp_xyz(subject, pixel_xy)
        return [*tcp_xyz.tolist(), *self._shooting_rpy.tolist()]

    def locate_block(self, pixel_xy: Sequence[float]) -> list[float]:
        """生成方块观察位姿。"""
        return self.locate("block", pixel_xy)

    def locate_tray(self, pixel_xy: Sequence[float]) -> list[float]:
        """生成托盘观察位姿。"""
        return self.locate("tray", pixel_xy)
