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
# 8-18 凌晨整理：方块=panel-180621 队友确认过的 15 条（左5/右10），
# 数据源为伺服 CSV（像素=高位检测像素，TCP=实测吸盘对准位），非现场手工示教。
# 注意：左半区仅 5 点 < 6 点门槛，运行前至少补 1 条（候选见下，按新50点模型
# 留出误差从好到差排序，取消注释即可用；现场新示教的点直接按格式添加）。
LEFT_PAIRS: list[tuple[float, float, float, float, float]] = [
    (369.0, 251.0, -305.877, -114.224, 173.461),  # z_green 180621任务4（确认）
    (349.0, 523.0, -172.788, -124.117, 173.457),  # T 180621任务9（确认）
    (135.0, 127.0, -367.672, -228.709, 173.459),  # z_green 180621任务11（确认）
    (257.0, 427.0, -220.295, -168.933, 173.462),  # L_yellow 180621任务13（确认）
    (143.0, 222.0, -320.648, -225.512, 173.457),  # z_blue 180621任务35（确认）
    # ---- 左侧补点候选（仅 180621 批次未确认行，按新模型留出误差排序；194117 等
    # ---- 中途停止的批次数据不可信，禁止使用）----
    # (236.0, 277.0, -293.883, -179.315, 173.460),  # square 180621任务15 OOF=0.03
    # (200.0, 62.0, -399.345, -196.465, 173.464),  # T 180621任务18 OOF=0.27
    # (240.0, 526.0, -172.091, -178.081, 173.462),  # square 180621任务2 OOF=0.40
    # (330.0, 386.0, -240.402, -133.027, 173.457),  # L_blue 180621任务31 OOF=0.40
    # (233.0, 193.0, -334.647, -180.674, 173.459),  # square 180621任务24 OOF=0.48
    # (369.0, 68.0, -396.147, -113.731, 173.455),  # T 180621任务17 OOF=0.50
]
RIGHT_PAIRS: list[tuple[float, float, float, float, float]] = [
    (991.0, 359.0, -251.256, 191.577, 173.459),  # square 180621任务1（确认）
    (915.0, 223.0, -317.426, 155.010, 173.456),  # z_blue 180621任务3（确认）
    (1006.0, 93.0, -380.860, 199.022, 173.456),  # z_blue 180621任务5（确认）
    (942.0, 548.0, -157.561, 167.635, 173.455),  # L_yellow 180621任务16（确认）
    (1051.0, 165.0, -345.965, 221.331, 173.461),  # line 180621任务19（确认）
    (1141.0, 217.0, -319.523, 264.756, 173.458),  # z_blue 180621任务20（确认）
    (1061.0, 266.0, -296.598, 225.253, 173.461),  # line 180621任务21（确认）
    (1138.0, 319.0, -269.919, 262.810, 173.461),  # z_blue 180621任务23（确认）
    (978.0, 261.0, -298.707, 185.642, 173.462),  # z_green 180621任务29（确认）
    (1053.0, 417.0, -221.896, 221.544, 173.455),  # z_green 180621任务34（确认）
]
TRAY_PAIRS: list[tuple[float, float, float, float, float]] = [
    (582.0, 323.9, -269.600, -8.714, 182.458),  # 180621任务1
    (662.5, 244.0, -309.013, 31.349, 182.461),  # 180621任务2
    (622.0, 364.5, -249.569, 11.101, 182.455),  # 180621任务3
    (702.9, 284.4, -288.873, 51.359, 182.460),  # 180621任务4
    (805.1, 22.0, -417.923, 101.831, 182.458),  # 180621任务5
    (682.4, 405.3, -229.046, 41.099, 182.454),  # 180621任务6
    (482.2, 61.8, -399.733, -58.226, 182.454),  # 180621任务7
    (602.1, 283.8, -289.486, 1.283, 182.457),  # 180621任务8
    (663.0, 142.9, -358.966, 31.534, 182.463),  # 180621任务9
    (482.2, 82.0, -389.803, -58.302, 182.466),  # 180621任务10
    (783.9, 203.9, -328.355, 91.473, 182.460),  # 180621任务11
    (643.6, 21.6, -418.810, 21.927, 182.464),  # 180621任务12
    (462.3, 21.2, -419.736, -68.116, 182.465),  # 180621任务13
    (481.5, 283.4, -290.219, -58.872, 182.457),  # 180621任务14
    (823.9, 365.6, -248.073, 111.226, 182.459),  # 180621任务15
    (803.7, 345.4, -258.248, 101.261, 182.457),  # 180621任务16
    (621.2, 506.2, -179.076, 10.675, 182.461),  # 180621任务17
    (581.9, 243.5, -309.545, -8.775, 182.459),  # 180621任务18
    (481.1, 363.9, -250.488, -59.106, 182.453),  # 180621任务19
    (662.8, 183.5, -338.971, 31.458, 182.459),  # 180621任务20
    (582.0, 303.8, -279.786, -8.753, 182.459),  # 180621任务21
    (803.1, 547.5, -157.400, 100.503, 182.457),  # 180621任务22
    (742.6, 486.5, -188.218, 70.892, 182.456),  # 180621任务23
    (682.5, 385.1, -239.013, 41.190, 182.464),  # 180621任务24
    (603.1, 41.7, -409.153, 1.843, 182.458),  # 180621任务25
    (744.1, 102.5, -378.589, 71.678, 182.462),  # 180621任务26
    (784.1, 304.7, -278.495, 91.390, 182.456),  # 180621任务27
    (823.7, 426.1, -218.039, 111.006, 182.457),  # 180621任务28
    (782.8, 507.1, -177.916, 90.655, 182.456),  # 180621任务29
    (521.5, 343.9, -260.069, -38.806, 182.464),  # 180621任务30
    (722.7, 425.6, -218.700, 61.072, 182.456),  # 180621任务31
    (723.1, 284.5, -288.710, 61.317, 182.464),  # 180621任务32
    (600.9, 546.5, -159.036, 0.541, 182.460),  # 180621任务33
    (520.9, 404.5, -230.061, -39.117, 182.465),  # 180621任务34
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
