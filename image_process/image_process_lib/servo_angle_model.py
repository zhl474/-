"""舵机绝对角边界规划的无硬件副作用模型。"""

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class ServoAngleTransition:
    """一次抓放所需的抓取角、摆放角与可选抓前预旋转。"""

    start_angle_deg: float
    pick_angle_deg: float
    place_angle_deg: float
    pre_pick_target_angle_deg: Optional[float]

    @property
    def pre_pick_rotation_deg(self) -> float:
        if self.pre_pick_target_angle_deg is None:
            return 0.0
        return abs(self.pre_pick_target_angle_deg - self.start_angle_deg)

    @property
    def loaded_rotation_deg(self) -> float:
        return abs(self.place_angle_deg - self.pick_angle_deg)


def _finite_number(value, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label}必须是有限数值")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是有限数值") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label}必须是有限数值")
    return number


def plan_servo_angle_transition(
    current_angle_deg: float,
    rotation_delta_deg: float,
    lower_margin_deg: float,
    upper_margin_deg: float,
) -> ServoAngleTransition:
    """复现真实执行器的 0°/360° 越界预旋转规则。"""
    current = _finite_number(current_angle_deg, "当前舵机角度")
    delta = _finite_number(rotation_delta_deg, "相对旋转角度")
    lower = _finite_number(lower_margin_deg, "舵机下侧安全角")
    upper = _finite_number(upper_margin_deg, "舵机上侧安全角")
    if not 0.0 <= current <= 360.0:
        raise ValueError("当前舵机角度必须位于 [0, 360]")
    if not 0.0 <= lower < upper <= 360.0:
        raise ValueError("舵机安全角边界无效")

    pick_angle = current
    pre_pick_target = None
    direct_place_angle = current + delta
    if direct_place_angle > 360.0:
        pick_angle = upper - delta
        pre_pick_target = pick_angle
    elif direct_place_angle < 0.0:
        pick_angle = lower - delta
        pre_pick_target = pick_angle
    place_angle = pick_angle + delta
    if not 0.0 <= pick_angle <= 360.0 or not 0.0 <= place_angle <= 360.0:
        raise ValueError(
            "相对旋转量超出舵机边界规划能力："
            f"抓取角={pick_angle:.3f}°，摆放角={place_angle:.3f}°"
        )
    return ServoAngleTransition(
        start_angle_deg=current,
        pick_angle_deg=pick_angle,
        place_angle_deg=place_angle,
        pre_pick_target_angle_deg=pre_pick_target,
    )


def worst_case_servo_reset_seconds(
    initial_angle_deg: float,
    velocity_deg_per_sec: float,
) -> float:
    """无反馈复位时，从任一 0°～360° 位置到初始角的最坏等待。"""
    initial = _finite_number(initial_angle_deg, "初始舵机角度")
    velocity = _finite_number(velocity_deg_per_sec, "舵机速度")
    if not 0.0 <= initial <= 360.0:
        raise ValueError("初始舵机角度必须位于 [0, 360]")
    if velocity <= 0.0:
        raise ValueError("舵机速度必须大于 0")
    return max(initial, 360.0 - initial) / velocity
