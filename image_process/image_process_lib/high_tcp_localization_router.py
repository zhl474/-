"""高位定位链路分流门面：servo（现行伺服标定）/ direct（九点标定）/ shadow。

servo  = 现行 HighPixelToTcpLocalizer（伺服数据拟合标定，行为与历史完全一致）
direct = 九点标定版（tools/vision/nine_point_calibration.py 生成的标定文件，
         方块按 u<640/u>=640 分左右模型，托盘单模型）
shadow = 正式执行 servo 结果不变，同一像素并算 direct，日志输出两分支偏差

模式由 mode_provider 每次调用时读取（image_node 注入 rosparam 覆盖逻辑），
切换无需重启节点。本模块不依赖 numpy / rospy，便于独立单元测试。
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

HIGH_TCP_SERVO = "servo"
HIGH_TCP_DIRECT = "direct"
HIGH_TCP_SHADOW = "shadow"
HIGH_TCP_LOCALIZATION_MODES = (HIGH_TCP_SERVO, HIGH_TCP_DIRECT, HIGH_TCP_SHADOW)

_SUBJECT_LABELS = {"block": "方块", "tray": "托盘"}
# 托盘一轮任务约 140 个格点：攒满即输出一条汇总，避免逐点刷屏。
_TRAY_SHADOW_FLUSH_COUNT = 140


def _default_warn(message: str, *args) -> None:
    formatted = message % args if args else message
    print(f"[高位定位分流] {formatted}")


class HighTcpLocalizationRouter:
    """与 HighPixelToTcpLocalizer 同接口的分流派发门面。

    构造时同时持有 servo 与 direct 两套 localizer（direct 允许为 None：
    标定文件缺失时只保留 servo，切 direct/shadow 会告警并回退 servo）。
    """

    def __init__(
        self,
        servo_localizer,
        direct_localizer,
        mode_provider: Callable[[], str],
        warn: Optional[Callable[..., None]] = None,
    ) -> None:
        if servo_localizer is None:
            raise ValueError("servo localizer 不能为 None")
        self._servo = servo_localizer
        self._direct = direct_localizer
        self._mode_provider = mode_provider
        self._warn = warn or _default_warn
        self._tray_shadow_deltas: list[tuple[float, float]] = []
        self._direct_missing_warned = False

    @property
    def servo_localizer(self):
        return self._servo

    @property
    def direct_localizer(self):
        return self._direct

    def _resolve_mode(self) -> str:
        try:
            value = str(self._mode_provider()).strip().lower()
        except Exception as exc:  # 模式读取失败不能阻断定位链路。
            self._warn("读取高位定位链路模式失败（%s），本次按 servo 处理", exc)
            return HIGH_TCP_SERVO
        if value not in HIGH_TCP_LOCALIZATION_MODES:
            self._warn("未知高位定位链路模式 %r，本次按 servo 处理", value)
            return HIGH_TCP_SERVO
        return value

    def _active_localizer(self):
        """按当前模式选择实际执行分支；direct 实例缺失时回退 servo。"""
        if self._resolve_mode() != HIGH_TCP_DIRECT:
            return self._servo
        if self._direct is not None:
            return self._direct
        if not self._direct_missing_warned:
            self._warn(
                "direct 九点标定实例未构建（标定文件缺失或 mode 启动时为 servo），"
                "本次按 servo 处理"
            )
            self._direct_missing_warned = True
        return self._servo

    def _flush_tray_shadow(self) -> None:
        if not self._tray_shadow_deltas:
            return
        count = len(self._tray_shadow_deltas)
        sorted_xy = sorted(delta[0] for delta in self._tray_shadow_deltas)
        sorted_z = sorted(delta[1] for delta in self._tray_shadow_deltas)
        median = sorted_xy[count // 2]
        self._warn(
            "九点标定shadow：托盘 %d 点 ΔXY 中位=%.2fmm 最大=%.2fmm；ΔZ 中位=%.2fmm 最大=%.2fmm",
            count,
            median,
            sorted_xy[-1],
            sorted_z[count // 2],
            max(abs(value) for value in sorted_z),
        )
        self._tray_shadow_deltas = []

    def _shadow_compare(self, subject: str, pixel_xy, servo_xyz) -> None:
        """shadow 模式：direct 分支并算同一点，输出与 servo 结果的偏差。"""
        if self._direct is None:
            return
        label = _SUBJECT_LABELS.get(subject, str(subject))
        try:
            direct_xyz = self._direct.predict_tcp_xyz(subject, pixel_xy)
        except Exception as exc:
            self._warn(
                "九点标定shadow：%s 像素%s direct 分支预测失败：%s",
                label,
                tuple(float(value) for value in pixel_xy),
                exc,
            )
            return
        dx = float(direct_xyz[0]) - float(servo_xyz[0])
        dy = float(direct_xyz[1]) - float(servo_xyz[1])
        dz = float(direct_xyz[2]) - float(servo_xyz[2])
        if subject == "block":
            # 方块间出现视为一轮托盘格点结束的边界，先冲刷汇总。
            self._flush_tray_shadow()
            self._warn(
                "九点标定shadow：%s 像素(%.1f, %.1f) ΔXY=%.2fmm ΔZ=%.2fmm "
                "servo=[%.2f, %.2f, %.2f] direct=[%.2f, %.2f, %.2f]",
                label,
                float(pixel_xy[0]),
                float(pixel_xy[1]),
                math.hypot(dx, dy),
                dz,
                float(servo_xyz[0]),
                float(servo_xyz[1]),
                float(servo_xyz[2]),
                float(direct_xyz[0]),
                float(direct_xyz[1]),
                float(direct_xyz[2]),
            )
            return
        self._tray_shadow_deltas.append((math.hypot(dx, dy), dz))
        if len(self._tray_shadow_deltas) >= _TRAY_SHADOW_FLUSH_COUNT:
            self._flush_tray_shadow()

    # ------------------------- 转发 HighPixelToTcpLocalizer 接口 -------------------------

    @property
    def fixed_tcp_z_mm(self):
        """两分支构建时共用同一 fixed_tcp_z 配置，取 servo 侧即可。"""
        return self._servo.fixed_tcp_z_mm

    def update_dynamic_bounds(
        self,
        minimum_tcp_z_mm: Optional[float] = None,
        safety_xy_offset: Optional[Sequence[float]] = None,
        shooting_pose: Optional[Sequence[float]] = None,
    ) -> None:
        self._servo.update_dynamic_bounds(
            minimum_tcp_z_mm=minimum_tcp_z_mm,
            safety_xy_offset=safety_xy_offset,
            shooting_pose=shooting_pose,
        )
        if self._direct is not None:
            self._direct.update_dynamic_bounds(
                minimum_tcp_z_mm=minimum_tcp_z_mm,
                safety_xy_offset=safety_xy_offset,
                shooting_pose=shooting_pose,
            )

    def predict_tcp_xyz(self, subject: str, pixel_xy):
        if self._resolve_mode() == HIGH_TCP_SHADOW:
            tcp_xyz = self._servo.predict_tcp_xyz(subject, pixel_xy)
            self._shadow_compare(subject, pixel_xy, tcp_xyz)
            return tcp_xyz
        return self._active_localizer().predict_tcp_xyz(subject, pixel_xy)

    def locate(self, subject: str, pixel_xy) -> list[float]:
        if self._resolve_mode() == HIGH_TCP_SHADOW:
            pose = self._servo.locate(subject, pixel_xy)
            self._shadow_compare(subject, pixel_xy, pose[:3])
            return pose
        return self._active_localizer().locate(subject, pixel_xy)

    def locate_block(self, pixel_xy) -> list[float]:
        return self.locate("block", pixel_xy)

    def locate_tray(self, pixel_xy) -> list[float]:
        return self.locate("tray", pixel_xy)

    def assess(self, subject: str, pixel_xy, safety_xy_offset=None):
        if self._resolve_mode() == HIGH_TCP_SHADOW:
            assessment = self._servo.assess(
                subject,
                pixel_xy,
                safety_xy_offset=safety_xy_offset,
            )
            self._shadow_compare(subject, pixel_xy, assessment.predicted_tcp_xyz)
            return assessment
        return self._active_localizer().assess(
            subject,
            pixel_xy,
            safety_xy_offset=safety_xy_offset,
        )

    def is_pixel_within_coverage(
        self,
        subject: str,
        pixel_xy,
        tolerance_px: float = 0.0,
    ) -> bool:
        return self._active_localizer().is_pixel_within_coverage(
            subject,
            pixel_xy,
            tolerance_px=tolerance_px,
        )

    def calibration_summary(self, subject: str) -> dict:
        """返回实际执行分支的标定摘要，另一分支存在时附在附加键里留档。"""
        active = self._active_localizer()
        summary = dict(active.calibration_summary(subject))
        summary["生效链路"] = (
            HIGH_TCP_DIRECT if active is self._direct else HIGH_TCP_SERVO
        )
        if active is not self._servo:
            summary["伺服标定版"] = self._servo.calibration_summary(subject)
        if self._direct is not None and active is not self._direct:
            summary["九点标定版"] = self._direct.calibration_summary(subject)
        return summary
