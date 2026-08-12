"""机械臂 MoveL 运动路程—时间预测。

本模块只负责将非负路程（mm）转换为预计运动时间（s），
不判断机械臂限位、目标可达性、路径安全性或舅机时间。
"""

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Sequence, Union

import numpy as np


DEFAULT_CALIBRATION_PATH = (
    Path(__file__).resolve().parent.parent
    / "config"
    / "arm_motion_time_calibration.json"
)


@dataclass(frozen=True)
class ArmMotionTimeModel:
    """基于实测表的连续路程—时间模型。"""

    distances_mm: np.ndarray
    times_seconds: np.ndarray
    short_distance_exponent: float
    long_distance_fit_start_mm: float
    long_distance_slope_s_per_mm: float
    motion_type: str
    move_speed_percent: float
    acceleration_percent: float
    source_path: Path

    def predict_seconds(self, distance_mm: float) -> float:
        """预测单个非负路程的运动时间，单位为秒。"""
        distance = _validate_scalar_distance(distance_mm)
        return float(self._predict_array(np.asarray([distance], dtype=float))[0])

    def predict_array_seconds(
        self,
        distances_mm: Union[Sequence[float], np.ndarray],
    ) -> np.ndarray:
        """批量预测路程数组，返回与输入形状相同的秒数组。"""
        distances = np.asarray(distances_mm, dtype=float)
        if distances.ndim == 0:
            raise ValueError("批量接口需要至少一维的路程数组")
        if not np.all(np.isfinite(distances)):
            raise ValueError("路程数组必须全部为有限数值")
        if np.any(distances < 0.0):
            raise ValueError("路程不能为负数")
        return self._predict_array(distances)

    def _predict_array(self, distances: np.ndarray) -> np.ndarray:
        """对已经校验的路程数组执行分段计算。"""
        first_distance = float(self.distances_mm[0])
        first_time = float(self.times_seconds[0])
        last_distance = float(self.distances_mm[-1])
        last_time = float(self.times_seconds[-1])

        # np.interp 负责标定范围内的连续分段线性插值。
        predicted = np.interp(distances, self.distances_mm, self.times_seconds)

        short_mask = distances < first_distance
        if np.any(short_mask):
            # 短距离按加减速主导的幂函数连续连接 (0, 0) 与首个实测点。
            predicted[short_mask] = first_time * np.power(
                distances[short_mask] / first_distance,
                self.short_distance_exponent,
            )

        long_mask = distances > last_distance
        if np.any(long_mask):
            # 长距离从最后一个实测点开始按尾段回归斜率连续外推。
            predicted[long_mask] = last_time + self.long_distance_slope_s_per_mm * (
                distances[long_mask] - last_distance
            )
        return predicted


def _validate_scalar_distance(distance_mm: float) -> float:
    """校验单个路程输入。"""
    if isinstance(distance_mm, bool):
        raise ValueError("路程必须是有限非负数")
    try:
        distance = float(distance_mm)
    except (TypeError, ValueError) as error:
        raise ValueError("路程必须是有限非负数") from error
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("路程必须是有限非负数")
    return distance


def _linear_regression_slope(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """计算最小二乘线性回归斜率。"""
    centered_x = x_values - float(np.mean(x_values))
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 0.0:
        raise ValueError("长距离回归至少需要两个不同路程点")
    centered_y = y_values - float(np.mean(y_values))
    return float(np.dot(centered_x, centered_y) / denominator)


def load_arm_motion_time_model(
    calibration_path: Union[str, Path] = DEFAULT_CALIBRATION_PATH,
) -> ArmMotionTimeModel:
    """从 JSON 标定文件加载并校验路程—时间模型。"""
    path = Path(calibration_path).expanduser().resolve()
    try:
        with path.open("r", encoding="utf-8") as calibration_file:
            document = json.load(calibration_file)
    except FileNotFoundError as error:
        raise ValueError(f"机械臂路程时间标定文件不存在：{path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取机械臂路程时间标定：{path}") from error

    if document.get("schema_version") != 1:
        raise ValueError("机械臂路程时间标定 schema_version 必须为 1")
    if document.get("time_metric") != "move_call_median_s":
        raise ValueError("当前模型只支持 move_call_median_s 时间指标")
    if document.get("short_distance_method") != "sqrt_to_origin":
        raise ValueError("当前模型只支持 sqrt_to_origin 短距离方法")
    if document.get("long_distance_method") != "anchored_linear_regression":
        raise ValueError(
            "当前模型只支持 anchored_linear_regression 长距离方法"
        )

    table = document.get("distance_time_table")
    if not isinstance(table, list) or len(table) < 2:
        raise ValueError("distance_time_table 至少需要两个标定点")
    try:
        distances = np.asarray([row["distance_mm"] for row in table], dtype=float)
        times = np.asarray([row["time_seconds"] for row in table], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("distance_time_table 数据格式无效") from error

    if not np.all(np.isfinite(distances)) or not np.all(np.isfinite(times)):
        raise ValueError("标定路程和时间必须全部为有限数值")
    if np.any(distances <= 0.0) or np.any(times <= 0.0):
        raise ValueError("标定路程和时间必须全部大于 0")
    if np.any(np.diff(distances) <= 0.0):
        raise ValueError("标定路程必须严格递增")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("标定时间必须单调不减")

    try:
        short_exponent = float(document["short_distance_exponent"])
        tail_start = float(document["long_distance_fit_start_mm"])
        move_speed = float(document["move_speed_percent"])
        acceleration = float(document["acceleration_percent"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("标定模型参数缺失或格式无效") from error
    if not 0.0 < short_exponent <= 1.0:
        raise ValueError("short_distance_exponent 必须在 (0, 1] 范围内")
    tail_mask = distances >= tail_start
    if int(np.count_nonzero(tail_mask)) < 2:
        raise ValueError("long_distance_fit_start_mm 之后至少需要两个标定点")
    tail_slope = _linear_regression_slope(distances[tail_mask], times[tail_mask])
    if not math.isfinite(tail_slope) or tail_slope <= 0.0:
        raise ValueError("长距离回归斜率必须大于 0")

    # 防止外部修改冻结模型中的标定数组。
    distances.setflags(write=False)
    times.setflags(write=False)
    return ArmMotionTimeModel(
        distances_mm=distances,
        times_seconds=times,
        short_distance_exponent=short_exponent,
        long_distance_fit_start_mm=tail_start,
        long_distance_slope_s_per_mm=tail_slope,
        motion_type=str(document.get("motion_type", "")),
        move_speed_percent=move_speed,
        acceleration_percent=acceleration,
        source_path=path,
    )


@lru_cache(maxsize=1)
def get_default_arm_motion_time_model() -> ArmMotionTimeModel:
    """加载并缓存项目默认标定模型。"""
    return load_arm_motion_time_model(DEFAULT_CALIBRATION_PATH)


def predict_arm_motion_time(distance_mm: float) -> float:
    """使用默认标定表预测单个路程的 MoveL 运动时间。"""
    return get_default_arm_motion_time_model().predict_seconds(distance_mm)


def predict_arm_motion_times(
    distances_mm: Union[Sequence[float], np.ndarray],
) -> np.ndarray:
    """使用默认标定表批量预测 MoveL 运动时间。"""
    return get_default_arm_motion_time_model().predict_array_seconds(distances_mm)
