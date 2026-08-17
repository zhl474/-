#!/home/zhl/fr3env/fr3env/bin/python
# -*- coding: utf-8 -*-
"""九点标定（左右分区版）：像素↔TCP 点对 → 二次多项式 → direct 分支标定 yaml。

复刻旧 point_calibration.py 的拟合逻辑（degree=2 展开为 [1, u, v, u², u·v, v²]
后最小二乘），但产物是现行 schema v2 标定文件，直接被 load_pixel_to_tcp_calibration
与 HighPixelToTcpLocalizer 接受，供 high_tcp_localization.mode=direct 分支使用。

左右分区口径（与 sucker_offset.classify_pixel_side 一致）：
    方块按 u < 640 / u >= 640 分左右两套模型，外加左右全部点合拟的中心回退模型
    （像素缺失/无效时运行时自动回退）；托盘为单模型，不区分左右。

采集口径（重要）：
    - 像素必须取自当前去畸变图（/camera/image_rect；识别调试图、桌面定位全景
      JSON 与它同源）。不要从相机原始图或旧截图取像素。
    - TCP 为吸盘示教位：手动拖动机械臂使吸嘴对准方块顶面/托盘面后记录法兰读数。
    - 每份拟合至少 6 点（二次多项式 6 个系数）；建议每侧 3×3 网格共 9 点。

用法：改本文件开头参数区的点对列表后直接运行
    /home/zhl/fr3env/fr3env/bin/python tools/vision/nine_point_calibration.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

SRC_DIR = Path(__file__).resolve().parents[2]
# 自检需要用正式加载器回读生成的标定文件。
sys.path.insert(0, str(SRC_DIR / "image_process"))

# ============================== 参数区（改这里） ==============================
# 每项 = (px, py, tcp_x, tcp_y, tcp_z)。示例（旧场地数据，仅示意格式）：
#     (145.0, 68.0, -486.46, -229.58, 179.19),
LEFT_PAIRS: list[tuple[float, float, float, float, float]] = [
    # 左半区（u < 640），建议 3×3 网格 ≥9 点
]
RIGHT_PAIRS: list[tuple[float, float, float, float, float]] = [
    # 右半区（u >= 640），建议 3×3 网格 ≥9 点
]
TRAY_PAIRS: list[tuple[float, float, float, float, float]] = [
    # 托盘区域点对，单模型，不区分左右，≥6 点
]

OUTPUT_DIR = SRC_DIR / "image_process" / "config"
BLOCK_CENTER_OUTPUT = "block_pixel_to_tcp_calibration_direct.yaml"
BLOCK_LEFT_OUTPUT = "block_pixel_to_tcp_calibration_direct_left.yaml"
BLOCK_RIGHT_OUTPUT = "block_pixel_to_tcp_calibration_direct_right.yaml"
TRAY_OUTPUT = "tray_pixel_to_tcp_calibration_direct.yaml"

# 与 sucker_offset.DEFAULT_IMAGE_WIDTH_PX / 2 一致的左右分界。
SIDE_SPLIT_U_PX = 640.0
MIN_POINTS = 6
RECOMMENDED_POINTS = 9
# ============================ 参数区结束 ======================================

FEATURE_NAMES = ["1", "u", "v", "u^2", "u*v", "v^2"]


def _parse_pairs(pairs, label):
    """校验点对并返回 (像素 Nx2, TCP XY Nx2, TCP Z N)。"""
    if len(pairs) < MIN_POINTS:
        raise ValueError(
            f"{label} 只有 {len(pairs)} 个点对，至少需要 {MIN_POINTS} 个"
            f"（二次多项式每轴 6 个系数）；建议 {RECOMMENDED_POINTS} 个"
        )
    try:
        values = np.asarray(pairs, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 点对必须是 5 个数值的元组列表") from exc
    if values.shape != (len(pairs), 5) or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} 点对必须每项 5 个有限数值 (px, py, x, y, z)")
    if len(pairs) < RECOMMENDED_POINTS:
        print(f"[警告] {label} 只有 {len(pairs)} 点（建议 ≥{RECOMMENDED_POINTS}），拟合易受单点噪声影响")
    return values[:, :2].copy(), values[:, 2:4].copy(), values[:, 4].copy()


def _check_side(pixels, label, expected_side):
    """分侧点对必须落在自己的半区，否则运行时永远路由不到该模型。"""
    if expected_side == "left":
        crossed = pixels[:, 0] >= SIDE_SPLIT_U_PX
    else:
        crossed = pixels[:, 0] < SIDE_SPLIT_U_PX
    if np.any(crossed):
        bad = [tuple(pixels[index].tolist()) for index in np.flatnonzero(crossed)]
        raise ValueError(
            f"{label} 内有像素越过 u={SIDE_SPLIT_U_PX:.0f} 分界线：{bad}；"
            f"请把{'左' if expected_side == 'left' else '右'}半区的点对放对列表"
        )


def _convex_hull(points):
    """Andrew 单调链凸包，返回逆时针顶点；退化（共线/重合）时报错。"""
    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) < 3:
        raise ValueError("标定像素点少于 3 个不同位置，凸包退化")

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    area = abs(sum(
        hull[i][0] * hull[(i + 1) % len(hull)][1] - hull[(i + 1) % len(hull)][0] * hull[i][1]
        for i in range(len(hull))
    )) / 2.0
    if area <= 1e-6:
        raise ValueError("标定像素凸包面积近似为零（点共线），无法覆盖二维区域")
    return hull


def fit_model(pairs, label, expected_side=None):
    """一份点对 → poly2 XY 模型 + z 平面 + 残差报告；expected_side 校验分侧归属。"""
    pixels, tcp_xy, tcp_z = _parse_pairs(pairs, label)
    if expected_side is not None:
        _check_side(pixels, label, expected_side)
    uv_mean = pixels.mean(axis=0)
    uv_scale = pixels.std(axis=0)
    if np.any(uv_scale <= 1e-12):
        raise ValueError(f"{label} 像素 u 或 v 方向没有离散度，无法归一化拟合")
    normalized = (pixels - uv_mean) / uv_scale
    u, v = normalized[:, 0], normalized[:, 1]
    features = np.column_stack([np.ones_like(u), u, v, u * u, u * v, v * v])
    coef, *_ = np.linalg.lstsq(features, tcp_xy, rcond=None)
    coef = coef.reshape(len(FEATURE_NAMES), 2)

    z_design = np.column_stack([tcp_xy[:, 0], tcp_xy[:, 1], np.ones_like(tcp_z)])
    z_coef, *_ = np.linalg.lstsq(z_design, tcp_z, rcond=None)

    predicted_xy = features @ coef
    delta = predicted_xy - tcp_xy
    euclid = np.hypot(delta[:, 0], delta[:, 1])
    z_residual = z_design @ z_coef - tcp_z
    print(
        f"[拟合] {label}：{len(pairs)} 点 | XY RMSE "
        f"X={np.sqrt(np.mean(delta[:, 0] ** 2)):.3f}mm Y={np.sqrt(np.mean(delta[:, 1] ** 2)):.3f}mm | "
        f"最大二维残差 {euclid.max():.3f}mm | Z 平面 RMSE {np.sqrt(np.mean(z_residual ** 2)):.3f}mm"
    )
    for index in np.argsort(euclid)[-3:]:
        print(
            f"        点对{index + 1} 像素({pixels[index, 0]:.1f}, {pixels[index, 1]:.1f}) "
            f"残差 {euclid[index]:.3f}mm"
        )
    return {
        "uv_mean": uv_mean.tolist(),
        "uv_scale": uv_scale.tolist(),
        "coef": [[float(value) for value in row] for row in coef],
        "z_plane": [float(value) for value in z_coef],
        "pixels": pixels,
        "pairs": [list(map(float, pair)) for pair in pairs],
    }


def build_payload(subject, generation_id, fit, description):
    """把拟合结果组装成 schema v2 标定文档。"""
    pixels = fit["pixels"]
    hull = _convex_hull(pixels)
    return {
        "schema_version": 2,
        "generation_id": generation_id,
        "calibration_type": "pixel_to_tcp_position",
        "calibration_subject": subject,
        "description": description,
        "input": {
            "coordinate": "high_detection_pixel_xy",
            "columns": ["高位检测像素X", "高位检测像素Y"],
            "axes": ["u", "v"],
            "unit": "pixel",
        },
        "output": {
            "coordinate": "tcp_position_xyz",
            "columns": ["实测TCP位置X", "实测TCP位置Y", "实测TCP位置Z"],
            "axes": ["x", "y", "z"],
            "unit": "mm",
            "orientation_included": False,
        },
        "xy_model": {
            "name": "poly2",
            "parameters": {
                "kind": "polynomial",
                "degree": 2,
                "uv_mean": fit["uv_mean"],
                "uv_scale": fit["uv_scale"],
                "coef": fit["coef"],
                "feature_names": list(FEATURE_NAMES),
            },
        },
        "z_plane": {
            "equation": "z = a*x + b*y + c",
            "coefficients": fit["z_plane"],
            "source": "nine_point_pairs_plane_fit",
        },
        "metrics": {
            "sample_count": len(fit["pairs"]),
            "pairs": fit["pairs"],
        },
        "coverage": {
            "sample_count": len(fit["pairs"]),
            "pixel_range": {
                "u": [float(pixels[:, 0].min()), float(pixels[:, 0].max())],
                "v": [float(pixels[:, 1].min()), float(pixels[:, 1].max())],
            },
            "pixel_convex_hull": [list(vertex) for vertex in hull],
            "extrapolation_policy": "diagnostic_only",
        },
        "usage_note": (
            "九点标定 direct 分支标定；Z 在 high_tcp_localization.fixed_tcp_z 启用时被常数覆盖，"
            "z_plane 仅在关闭固定 Z 后生效。"
        ),
    }


def _self_check(path, subject, pairs):
    """用正式加载器回读生成的文件并复测全部点对。"""
    from image_process_lib.pixel_to_tcp_calibration import (
        load_pixel_to_tcp_calibration,
    )

    calibration = load_pixel_to_tcp_calibration(path, expected_subject=subject)
    errors = []
    for px, py, x, y, _z in pairs:
        predicted = calibration.predict((px, py))
        errors.append(float(np.hypot(predicted[0] - x, predicted[1] - y)))
    print(f"[自检] {path.name}：加载成功，XY 回代最大误差 {max(errors):.4f}mm")
    return max(errors)


def main():
    if not LEFT_PAIRS and not RIGHT_PAIRS and not TRAY_PAIRS:
        raise RuntimeError(
            "点对列表为空：请在文件开头参数区填入 LEFT_PAIRS / RIGHT_PAIRS / TRAY_PAIRS"
        )
    generation_id = f"direct-ninepoint-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    print(f"[批次] generation_id = {generation_id}（四份文件同批次，加载器一致性校验要求）")

    plans = []
    if LEFT_PAIRS:
        plans.append((BLOCK_LEFT_OUTPUT, "block", fit_model(LEFT_PAIRS, "左半区方块", expected_side="left"), "九点标定（direct 分支）左半区方块模型"))
    if RIGHT_PAIRS:
        plans.append((BLOCK_RIGHT_OUTPUT, "block", fit_model(RIGHT_PAIRS, "右半区方块", expected_side="right"), "九点标定（direct 分支）右半区方块模型"))
    if LEFT_PAIRS or RIGHT_PAIRS:
        center_pairs = list(LEFT_PAIRS) + list(RIGHT_PAIRS)
        plans.append((BLOCK_CENTER_OUTPUT, "block", fit_model(center_pairs, "方块中心回退"), "九点标定（direct 分支）方块中心回退模型（像素无效时使用）"))
    if TRAY_PAIRS:
        plans.append((TRAY_OUTPUT, "tray", fit_model(TRAY_PAIRS, "托盘"), "九点标定（direct 分支）托盘模型"))

    if not (LEFT_PAIRS and RIGHT_PAIRS and TRAY_PAIRS):
        print("[警告] 点对不全，本次生成的文件不完整；direct 分支需要四份文件齐全才能构建")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs_by_file = {
        BLOCK_LEFT_OUTPUT: LEFT_PAIRS,
        BLOCK_RIGHT_OUTPUT: RIGHT_PAIRS,
        BLOCK_CENTER_OUTPUT: list(LEFT_PAIRS) + list(RIGHT_PAIRS),
        TRAY_OUTPUT: TRAY_PAIRS,
    }
    worst = 0.0
    for filename, subject, fit, description in plans:
        path = OUTPUT_DIR / filename
        path.write_text(
            yaml.safe_dump(
                build_payload(subject, generation_id, fit, description),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        print(f"[写出] {path}")
        worst = max(worst, _self_check(path, subject, pairs_by_file[filename]))
    print(f"[完成] 全部文件自检通过，回代最大误差 {worst:.4f}mm")
    print("[提示] perception.yaml 默认 mode 保持 servo；切 direct 分支：")
    print("       rosparam set /image_process_node/high_tcp_localization_mode direct")


if __name__ == "__main__":
    main()
