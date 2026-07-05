#!/home/zhl/fr3env/fr3env/bin/python
import os

import numpy as np
import yaml


class ServoEccentricCompensator:
    """舵机转轴偏心补偿器，补偿量单位固定为毫米。"""

    def __init__(self, config_path, debug=True):
        self.config_path = config_path
        self.debug = debug
        self.enabled = False
        self.periodic = True
        self.table = []
        self.angles = np.array([], dtype=np.float64)
        self.compensations = np.zeros((0, 3), dtype=np.float64)

        self._load_config(config_path)

    def _load_config(self, config_path):
        if not config_path or not os.path.exists(config_path):
            print(f"[吸盘偏心补偿] 配置文件不存在，已关闭补偿: {config_path}")
            return

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self.enabled = bool(data.get("enabled", False))
        self.periodic = bool(data.get("periodic", True))
        interpolation = data.get("interpolation", "linear")
        unit = data.get("unit", "mm")
        angle_unit = data.get("angle_unit", "deg")

        if interpolation != "linear":
            raise ValueError("舵机偏心补偿只支持 linear 插值")
        if unit != "mm":
            raise ValueError("舵机偏心补偿配置的 unit 必须是 mm")
        if angle_unit != "deg":
            raise ValueError("舵机偏心补偿配置的 angle_unit 必须是 deg")

        rows = data.get("table", []) or []
        self.table = self._parse_table(rows)
        if self.table:
            self.angles = np.array([item[0] for item in self.table], dtype=np.float64)
            self.compensations = np.array([item[1] for item in self.table], dtype=np.float64)

        if self.enabled and len(self.table) == 0:
            raise ValueError("舵机偏心补偿已启用，但 table 为空")

        print(
            f"[吸盘偏心补偿] enabled={self.enabled}, "
            f"table_size={len(self.table)}, config={config_path}"
        )

    def _parse_table(self, rows):
        parsed = []
        for row in rows:
            theta = self._normalize_angle(float(row["theta_deg"]))
            comp = np.array(
                [
                    float(row.get("dx_mm", 0.0)),
                    float(row.get("dy_mm", 0.0)),
                    float(row.get("dz_mm", 0.0)),
                ],
                dtype=np.float64,
            )
            parsed.append((theta, comp))

        parsed.sort(key=lambda item: item[0])
        for idx in range(1, len(parsed)):
            if np.isclose(parsed[idx - 1][0], parsed[idx][0], atol=1e-9):
                raise ValueError(f"舵机偏心补偿表存在重复角度: {parsed[idx][0]}")
        return parsed

    def _normalize_angle(self, theta_deg):
        theta = float(theta_deg) % 360.0
        if np.isclose(theta, 360.0, atol=1e-9):
            return 0.0
        return theta

    def get_compensation(self, theta_deg):
        if not self.enabled or len(self.table) == 0:
            return np.zeros(3, dtype=np.float64)

        theta = self._normalize_angle(theta_deg)
        exact_idx = np.where(np.isclose(self.angles, theta, atol=1e-9))[0]
        if len(exact_idx) > 0:
            return self.compensations[exact_idx[0]].copy()

        if len(self.table) == 1:
            return self.compensations[0].copy()

        insert_idx = int(np.searchsorted(self.angles, theta))
        if 0 < insert_idx < len(self.angles):
            return self._interpolate(
                theta,
                self.angles[insert_idx - 1],
                self.compensations[insert_idx - 1],
                self.angles[insert_idx],
                self.compensations[insert_idx],
            )

        if self.periodic:
            lower_angle = self.angles[-1]
            lower_comp = self.compensations[-1]
            upper_angle = self.angles[0] + 360.0
            upper_comp = self.compensations[0]
            theta_for_interp = theta if theta >= lower_angle else theta + 360.0
            return self._interpolate(
                theta_for_interp,
                lower_angle,
                lower_comp,
                upper_angle,
                upper_comp,
            )

        # 非周期模式下，表外角度按最近端点补偿，避免外推产生意外运动。
        if theta < self.angles[0]:
            return self.compensations[0].copy()
        return self.compensations[-1].copy()

    def _interpolate(self, theta, lower_angle, lower_comp, upper_angle, upper_comp):
        span = upper_angle - lower_angle
        if np.isclose(span, 0.0, atol=1e-9):
            return lower_comp.copy()
        ratio = (theta - lower_angle) / span
        return lower_comp + ratio * (upper_comp - lower_comp)

    def apply(self, p_nominal, theta_deg):
        p_cmd = np.array(p_nominal, dtype=np.float64).copy()
        compensation = self.get_compensation(theta_deg)
        if p_cmd.shape[0] < 3:
            raise ValueError("机械臂目标点至少需要包含 x, y, z 三个元素")

        p_before = p_cmd.copy()
        p_cmd[:3] = p_cmd[:3] + compensation

        if self.debug:
            print(
                "[吸盘偏心补偿] "
                f"theta={float(theta_deg):.3f} deg, "
                f"补偿(mm)={compensation.tolist()}, "
                f"补偿前={p_before.tolist()}, "
                f"补偿后={p_cmd.tolist()}"
            )

        return p_cmd
