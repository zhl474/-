"""只读汇总正式模式定位 Z 的完整计算链路，供网页端展示。

数值全部由当前 perception.yaml、execution.yaml 和两个像素标定 yaml 实时代入
公式算出，调参后无需改本文件；标定模式（深度链路）不在计算范围内，仅备注。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PACKAGE_DIR.parent

DEFAULT_PERCEPTION_PATH = SRC_DIR / "image_process" / "config" / "perception.yaml"
DEFAULT_EXECUTION_PATH = SRC_DIR / "competition" / "config" / "execution.yaml"
DEFAULT_BLOCK_CALIBRATION_PATH = (
    SRC_DIR / "image_process" / "config" / "block_pixel_to_tcp_calibration.yaml"
)
DEFAULT_TRAY_CALIBRATION_PATH = (
    SRC_DIR / "image_process" / "config" / "tray_pixel_to_tcp_calibration.yaml"
)

_REQUIRED_MOTION_KEYS = (
    "pick_surface_offset_mm",
    "pick_approach_clearance_mm",
    "place_descent_offset_mm",
    "pick_rotate_safe_lift_mm",
    "pick_retreat_blend_radius_mm",
    "minimum_tcp_z_mm",
)


def _load_yaml(path: Path, name: str) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file_handle:
            data = yaml.safe_load(file_handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"无法读取{name}（{path}）：{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{name} 必须是字典：{path}")
    return data


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是有限数值") from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{name} 必须是有限数值")
    return number


def _round(value: float) -> float:
    return round(float(value), 3)


def _src_path(value: Any, fallback: Path) -> Path:
    """与正式节点一致：相对路径按 src 目录解析。"""
    raw_path = str(value or fallback)
    return (
        Path(raw_path).expanduser()
        if Path(raw_path).is_absolute()
        else SRC_DIR / raw_path
    )


def _load_z_plane(path: Path) -> Optional[Dict[str, Any]]:
    """容错读取标定 yaml 的 z_plane；文件缺失或 schema v1 时返回 None。"""
    try:
        document = _load_yaml(path, "像素标定")
    except ValueError:
        return None
    z_plane = document.get("z_plane")
    if not isinstance(z_plane, dict):
        return None
    try:
        coefficients = [float(value) for value in z_plane.get("coefficients", [])]
    except (TypeError, ValueError):
        return None
    if len(coefficients) != 3:
        return None
    return {
        "coefficients": coefficients,
        "generation_id": document.get("generation_id"),
        "source": z_plane.get("source"),
    }


def _plane_range(coefficients, x_range, y_range) -> Tuple[float, float]:
    """线性平面在安全 XY 矩形上的取值范围（最值在四个角点）。"""
    a, b, c = coefficients
    corners = [
        a * x + b * y + c
        for x in x_range
        for y in y_range
    ]
    return min(corners), max(corners)


def _scalar_or_range(values: List[float]) -> Dict[str, float]:
    return {"min": _round(min(values)), "max": _round(max(values))}


def build_localization_z_chain(
    perception_path: Path = DEFAULT_PERCEPTION_PATH,
    execution_path: Path = DEFAULT_EXECUTION_PATH,
    block_path: Optional[Path] = None,
    tray_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """读取三份配置并把当前值代入定位 Z 公式链，返回网页展示结构。"""
    perception = _load_yaml(perception_path, "perception.yaml")
    execution = _load_yaml(execution_path, "execution.yaml")

    motion = execution.get("motion", {})
    missing = [key for key in _REQUIRED_MOTION_KEYS if key not in motion]
    if missing:
        raise ValueError("execution.yaml motion 缺少配置项: " + ", ".join(missing))
    offset = _finite(motion["pick_surface_offset_mm"], "pick_surface_offset_mm")
    clearance = _finite(motion["pick_approach_clearance_mm"], "pick_approach_clearance_mm")
    descent = _finite(motion["place_descent_offset_mm"], "place_descent_offset_mm")
    lift = _finite(motion["pick_rotate_safe_lift_mm"], "pick_rotate_safe_lift_mm")
    blend = _finite(motion["pick_retreat_blend_radius_mm"], "pick_retreat_blend_radius_mm")
    minimum = _finite(motion["minimum_tcp_z_mm"], "minimum_tcp_z_mm")

    localization = perception.get("high_tcp_localization", {})
    pick_height = perception.get("pick_height", {})
    observation_height = _finite(
        pick_height.get("block_observation_height_mm", 192.0),
        "block_observation_height_mm",
    )
    x_range = tuple(
        _finite(value, "safe_x_range_mm") for value in localization.get(
            "safe_x_range_mm", [-444.224, -148.17]
        )
    )
    y_range = tuple(
        _finite(value, "safe_y_range_mm") for value in localization.get(
            "safe_y_range_mm", [-263.279, 315.925]
        )
    )

    fixed_config = localization.get("fixed_tcp_z", {})
    fixed_enabled = fixed_config.get("enabled", False)
    fixed_block = fixed_tray = None
    if fixed_enabled:
        if "block_observation_z_mm" not in fixed_config or "tray_z_mm" not in fixed_config:
            raise ValueError(
                "fixed_tcp_z.enabled=true 但缺少 block_observation_z_mm 或 tray_z_mm"
            )
        fixed_block = _finite(
            fixed_config["block_observation_z_mm"], "fixed_tcp_z.block_observation_z_mm"
        )
        fixed_tray = _finite(fixed_config["tray_z_mm"], "fixed_tcp_z.tray_z_mm")

    calibration_config = perception.get("calibration", {})
    block_calibration_path = block_path or _src_path(
        calibration_config.get("block_pixel_to_tcp"), DEFAULT_BLOCK_CALIBRATION_PATH
    )
    tray_calibration_path = tray_path or _src_path(
        calibration_config.get("tray_pixel_to_tcp"), DEFAULT_TRAY_CALIBRATION_PATH
    )
    z_planes = {
        "block": _load_z_plane(block_calibration_path),
        "tray": _load_z_plane(tray_calibration_path),
    }

    constants: List[Dict[str, Any]] = [
        {
            "symbol": "B",
            "label": "方块观察固定 Z",
            "value": None if fixed_block is None else _round(fixed_block),
            "source": "perception.yaml high_tcp_localization.fixed_tcp_z.block_observation_z_mm",
        },
        {
            "symbol": "T",
            "label": "托盘固定 Z",
            "value": None if fixed_tray is None else _round(fixed_tray),
            "source": "perception.yaml high_tcp_localization.fixed_tcp_z.tray_z_mm",
        },
        {
            "symbol": "h",
            "label": "方块观察高度",
            "value": _round(observation_height),
            "source": "perception.yaml pick_height.block_observation_height_mm",
        },
        {
            "symbol": "o",
            "label": "抓取表面偏移",
            "value": _round(offset),
            "source": "execution.yaml motion.pick_surface_offset_mm",
        },
        {
            "symbol": "c",
            "label": "预抓取间隙",
            "value": _round(clearance),
            "source": "execution.yaml motion.pick_approach_clearance_mm",
        },
        {
            "symbol": "d",
            "label": "摆放下探距离",
            "value": _round(descent),
            "source": "execution.yaml motion.place_descent_offset_mm",
        },
        {
            "symbol": "L",
            "label": "旋转安全抬升",
            "value": _round(lift),
            "source": "execution.yaml motion.pick_rotate_safe_lift_mm",
        },
        {
            "symbol": "z_min",
            "label": "TCP 安全下限",
            "value": _round(minimum),
            "source": "execution.yaml motion.minimum_tcp_z_mm",
        },
    ]

    if fixed_enabled:
        rows = _fixed_mode_rows(fixed_block, fixed_tray, observation_height, offset,
                                clearance, descent, lift)
        # 固定 Z 同时作用于标定模式：深度只提供粗定位 XY，高度与正式模式完全一致。
        rows += [
            _row("标定·方块观察位", "Z_obs = B（深度只供 XY）", "B", _round(fixed_block)),
            _row("标定·最终抓取 Z", "Z_pick = B − (h − o)", f"B − ({observation_height:g} − {offset:g})",
                 _round(fixed_block - (observation_height - offset))),
            _row("标定·托盘 Z", "Z_place = T（深度只供 XY）", "T", _round(fixed_tray)),
        ]
        checks = _fixed_mode_checks(
            fixed_block, fixed_tray, observation_height, offset, minimum, blend
        )
        mode = "fixed_constant"
        mode_label = "固定常数（正式与标定模式一致，z_plane 已禁用）"
    else:
        rows = _plane_mode_rows(
            z_planes, x_range, y_range, observation_height, offset, clearance, descent, lift
        )
        checks = _plane_mode_checks(
            z_planes, x_range, y_range, observation_height, offset, minimum, blend
        )
        mode = "calibration_z_plane"
        mode_label = "标定 z_plane（随 XY 变化的斜面）"

    z_plane_summary = {}
    for subject, plane in z_planes.items():
        if plane is None:
            z_plane_summary[subject] = None
            continue
        low, high = _plane_range(plane["coefficients"], x_range, y_range)
        z_plane_summary[subject] = {
            "coefficients": [_round(value) for value in plane["coefficients"]],
            "generation_id": plane["generation_id"],
            "source": plane["source"],
            "safe_box_range": {"min": _round(low), "max": _round(high)},
            "bypassed": fixed_enabled,
        }

    return {
        "mode": mode,
        "mode_label": mode_label,
        "constants": constants,
        "rows": rows,
        "checks": checks,
        "z_plane": z_plane_summary,
        "notes": [
            "XY 不在本表范围：正式模式 XY 由像素标定模型预测后交视觉伺服闭环，伺服全程 Z 不变；"
            "标定模式 XY 由深度世界坐标提供。",
            (
                "标定模式与正式模式共用同一固定 Z（深度只提供粗定位 XY），高度与正式模式逐位相同。"
                if fixed_enabled
                else "标定模式不经过此链路：观察 Z = 深度表面世界 Z + h，抓取 Z = 表面 Z + o，"
                "托盘 Z = 方块观察平面(X,Y) − tray_tcp_below_block_observation_mm。"
            ),
            "以上数值由当前配置实时计算；修改参数后需重启 perception（或 execution 相关的 hardware）节点才在实际运动中生效。",
        ],
    }


def _fixed_mode_rows(block_z, tray_z, height, offset, clearance, descent, lift):
    pick_z = block_z - (height - offset)
    pre_pick_z = pick_z + clearance
    rotate_safe_z = min(pick_z + lift, tray_z)
    return [
        _row("方块观察位", "Z_obs = B", "B", _round(block_z)),
        _row("方块表面 Z", "Z_surface = B − h", f"B − {height:g}", _round(block_z - height)),
        _row("最终抓取 Z（下探终点）", "Z_pick = B − (h − o)", f"B − ({height:g} − {offset:g})", _round(pick_z)),
        _row("预抓取 Z（斜向进入）", "Z_pre = Z_pick + c", f"Z_pick + {clearance:g}", _round(pre_pick_z)),
        _row("抓后抬升 Z", "Z_retreat = T", "T", _round(tray_z)),
        _row("旋转放行阈值", "min(Z_pick + L, T)", f"min(Z_pick + {lift:g}, T)", _round(rotate_safe_z)),
        _row("托盘释放 Z", "Z_place = T", "T", _round(tray_z)),
        _row("摆放下探 Z", "Z_desc = Z_place − d", f"T − {descent:g}", _round(tray_z - descent)),
    ]


def _plane_mode_rows(z_planes, x_range, y_range, height, offset, clearance, descent, lift):
    block = z_planes["block"]
    tray = z_planes["tray"]
    if block is None:
        return [
            {
                "stage": "标定 z_plane 缺失",
                "formula": "—",
                "substitution": "方块标定文件不存在或无 z_plane",
                "value": None,
            }
        ]
    corners = [(x, y) for x in x_range for y in y_range]
    block_at = [
        block["coefficients"][0] * x + block["coefficients"][1] * y + block["coefficients"][2]
        for x, y in corners
    ]
    pick_at = [value - (height - offset) for value in block_at]
    pre_at = [value + clearance for value in pick_at]
    rows = [
        _row("方块观察位", "Z_obs = a·X + b·Y + c", "方块平面(X,Y)", _scalar_or_range(block_at)),
        _row("方块表面 Z", "Z_surface = 平面(X,Y) − h", f"平面 − {height:g}",
             _scalar_or_range([value - height for value in block_at])),
        _row("最终抓取 Z（下探终点）", "Z_pick = 平面(X,Y) − (h − o)", f"平面 − ({height:g} − {offset:g})",
             _scalar_or_range(pick_at)),
        _row("预抓取 Z（斜向进入）", "Z_pre = Z_pick + c", f"Z_pick + {clearance:g}",
             _scalar_or_range(pre_at)),
    ]
    if tray is None:
        rows.append(_row("托盘相关高度", "—", "托盘标定文件不存在或无 z_plane", None))
        return rows
    tray_at = [
        tray["coefficients"][0] * x + tray["coefficients"][1] * y + tray["coefficients"][2]
        for x, y in corners
    ]
    rotate_at = [min(pick + lift, place) for pick, place in zip(pick_at, tray_at)]
    rows += [
        _row("抓后抬升 Z", "Z_retreat = 托盘平面(X,Y)", "托盘平面", _scalar_or_range(tray_at)),
        _row("旋转放行阈值", "min(Z_pick + L, 托盘平面)", "同一角点取小",
             _scalar_or_range(rotate_at)),
        _row("托盘释放 Z", "Z_place = 托盘平面(X,Y)", "托盘平面", _scalar_or_range(tray_at)),
        _row("摆放下探 Z", "Z_desc = Z_place − d", f"托盘平面 − {descent:g}",
             _scalar_or_range([value - descent for value in tray_at])),
    ]
    return rows


def _fixed_mode_checks(block_z, tray_z, height, offset, minimum, blend):
    pick_z = block_z - (height - offset)
    checks = [
        _check(
            "最终抓取 Z ≥ z_min",
            f"{pick_z:.2f} ≥ {minimum:.2f}，余量 {pick_z - minimum:.2f} mm",
            pick_z >= minimum,
        ),
        _check(
            "托盘释放 Z ≥ z_min",
            f"{tray_z:.2f} ≥ {minimum:.2f}，余量 {tray_z - minimum:.2f} mm",
            tray_z >= minimum,
        ),
    ]
    lift_distance = tray_z - pick_z
    checks.append(
        _check(
            "抬升距离 > 抓后圆滑半径",
            f"T − Z_pick = {lift_distance:.2f} mm，圆滑半径 {blend:.2f} mm",
            lift_distance > blend if blend > 0.0 else True,
        )
    )
    return checks


def _plane_mode_checks(z_planes, x_range, y_range, height, offset, minimum, blend):
    block = z_planes["block"]
    tray = z_planes["tray"]
    if block is None:
        return [_check("标定 z_plane 缺失", "无法核对高度下限", False)]
    corners = [(x, y) for x in x_range for y in y_range]
    block_at = [
        block["coefficients"][0] * x + block["coefficients"][1] * y + block["coefficients"][2]
        for x, y in corners
    ]
    pick_at = [value - (height - offset) for value in block_at]
    pick_low = min(pick_at)
    checks = [
        _check(
            "最终抓取 Z ≥ z_min（取跨台最坏角点）",
            f"最小 {pick_low:.2f} ≥ {minimum:.2f}，余量 {pick_low - minimum:.2f} mm",
            pick_low >= minimum,
        ),
    ]
    if tray is None:
        checks.append(_check("托盘标定缺失", "无法核对释放高度", False))
        return checks
    tray_at = [
        tray["coefficients"][0] * x + tray["coefficients"][1] * y + tray["coefficients"][2]
        for x, y in corners
    ]
    tray_low = min(tray_at)
    checks.append(
        _check(
            "托盘释放 Z ≥ z_min（取跨台最坏角点）",
            f"最小 {tray_low:.2f} ≥ {minimum:.2f}，余量 {tray_low - minimum:.2f} mm",
            tray_low >= minimum,
        )
    )
    lift_low = min(place - pick for pick, place in zip(pick_at, tray_at))
    checks.append(
        _check(
            "抬升距离 > 抓后圆滑半径（最坏角点）",
            f"最小抬升 {lift_low:.2f} mm，圆滑半径 {blend:.2f} mm",
            lift_low > blend if blend > 0.0 else True,
        )
    )
    return checks


def _row(stage: str, formula: str, substitution: str, value) -> Dict[str, Any]:
    return {
        "stage": stage,
        "formula": formula,
        "substitution": substitution,
        "value": value,
    }


def _check(name: str, detail: str, ok: bool) -> Dict[str, Any]:
    return {"name": name, "detail": detail, "ok": bool(ok)}
