#!/home/zhl/fr3env/fr3env/bin/python
# -*- coding: utf-8 -*-
"""分析 prepare_task 自动导出的「高位定位全景_*.json」，定位系统误差来源。

直接改下方参数区再运行，不需要命令行参数：
    /home/zhl/fr3env/fr3env/bin/python tools/vision/定位全景分析.py

支持三种用法（按文件里的"模式"字段自动分流）：
1. 正式模式文件：140 格点 TCP 对理想等距网格的最小二乘仿射拟合残差、
   相邻格点间距统计、Z 平面拟合（可与 yaml 里的 z_plane 系数对照）、
   残差矢量图 + 热力图（存到 JSON 同目录）。
2. 标定模式文件：深度实测世界坐标、拟合 Z 平面逐项打印。
3. PAIR_PATH 配对（正式 + 标定两份）：同一物理摆位下，
   托盘采样点 预测TCP − 深度换算TCP 的逐点差值，直接暴露问题环节
   （格点识别 / XY 标定模型 / Z 平面）。
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import yaml

SRC_DIR = Path(__file__).resolve().parents[2]
PACKAGE_DIR = SRC_DIR / "image_process"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from image_process_lib.board_scene_detector import BOARD_COL_COUNT, BOARD_ROW_COUNT  # noqa: E402
from image_process_lib.depth_rough_localization import DepthRoughLocalizer  # noqa: E402

# ----------------------------- 直接运行配置 -----------------------------
# 输入文件：留空 = 自动取输出目录里最新的 高位定位全景_*.json
INPUT_PATH = "/home/zhl/桌面/高位定位全景_标定模式_2026-08-17_18-09-17.json"
# 输出目录（与 image_node 的 ~localization_panorama_dir 保持一致）
PANORAMA_DIR = "/home/zhl/桌面"
# 配对文件（可选）：正式+标定各一份时填另一份路径做逐点对比；留空跳过
PAIR_PATH = ""
# 残差最大的点打印前 N 个
TOP_N = 15
# 是否输出 matplotlib 图（正式模式：残差矢量图 + 热力图）
PLOT = True
# -----------------------------------------------------------------------

EXECUTION_CONFIG_PATH = SRC_DIR / "competition" / "config" / "execution.yaml"
PERCEPTION_CONFIG_PATH = PACKAGE_DIR / "config" / "perception.yaml"


def load_latest_panorama():
    candidates = sorted(
        Path(PANORAMA_DIR).glob("高位定位全景_*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"{PANORAMA_DIR} 下没有 高位定位全景_*.json，请先跑一次 prepare_task"
        )
    return candidates[-1]


def load_document(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _fmt(value, digits=2):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def collect_grid_points(document):
    """返回 [(row, col, pixel, tcp_xyz)]，跳过有错误或缺失的格点。"""
    points = []
    for entry in document.get("托盘格点", []):
        if entry.get("错误"):
            continue
        tcp = entry.get("TCP_X")
        if tcp is None:
            continue
        points.append(
            (
                float(entry["行"]),
                float(entry["列"]),
                (entry["像素u"], entry["像素v"]),
                np.array([entry["TCP_X"], entry["TCP_Y"], entry["TCP_Z"]]),
            )
        )
    return points


def fit_ideal_grid_residuals(points):
    """对 (行,列)->TCP X/Y 做最小二乘仿射拟合，返回每点残差。"""
    rows = np.array([row for row, _col, _pixel, _tcp in points])
    cols = np.array([col for _row, col, _pixel, _tcp in points])
    design = np.column_stack([rows, cols, np.ones(len(points))])
    tcp = np.array([tcp for _row, _col, _pixel, tcp in points])
    coef_x, _res, _rank, _sv = np.linalg.lstsq(design, tcp[:, 0], rcond=None)
    coef_y, _res, _rank, _sv = np.linalg.lstsq(design, tcp[:, 1], rcond=None)
    predicted = np.column_stack([design @ coef_x, design @ coef_y])
    delta = tcp[:, :2] - predicted
    return rows, cols, tcp, delta


def neighbor_spacing(points):
    """相邻格点 TCP 间距：列向（同行相邻列）与行向（同列相邻行）。"""
    grid = {
        (int(row), int(col)): tcp
        for row, col, _pixel, tcp in points
        if float(row).is_integer() and float(col).is_integer()
    }
    col_spacings = []
    for row in range(1, BOARD_ROW_COUNT + 1):
        for col in range(1, BOARD_COL_COUNT):
            if (row, col) in grid and (row, col + 1) in grid:
                col_spacings.append(
                    float(np.linalg.norm(grid[(row, col + 1)] - grid[(row, col)]))
                )
    row_spacings = []
    for row in range(1, BOARD_ROW_COUNT):
        for col in range(1, BOARD_COL_COUNT + 1):
            if (row, col) in grid and (row + 1, col) in grid:
                row_spacings.append(
                    float(np.linalg.norm(grid[(row + 1, col)] - grid[(row, col)]))
                )
    return np.asarray(col_spacings), np.asarray(row_spacings), grid


def fit_tcp_z_plane(tcp):
    """在预测 TCP 点上重拟合 z=a*x+b*y+c，用于和 yaml z_plane 对照。"""
    design = np.column_stack([tcp[:, 0], tcp[:, 1], np.ones(len(tcp))])
    coefficients, _res, _rank, _sv = np.linalg.lstsq(design, tcp[:, 2], rcond=None)
    residuals = design @ coefficients - tcp[:, 2]
    return coefficients, float(np.sqrt(np.mean(residuals**2)))


def print_stats(label, values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        print(f"  {label}: 无数据")
        return
    print(
        f"  {label}: 均值={values.mean():.3f} 标准差={values.std():.3f} "
        f"最小={values.min():.3f} 最大={values.max():.3f} (n={len(values)})"
    )


def print_calibration_info(document):
    calibration = document.get("标定") or {}
    for subject, info in calibration.items():
        if not isinstance(info, dict):
            continue
        z_plane = info.get("Z平面系数a_b_c")
        z_text = (
            "-"
            if z_plane is None
            else "a={:+.6f} b={:+.6f} c={:.3f}".format(*z_plane)
        )
        print(
            f"  {subject}: 模型={info.get('XY模型')} 批次={info.get('实验批次')} "
            f"样本数={info.get('标定样本数')} 凸包顶点={info.get('凸包顶点数')}"
        )
        print(f"    yaml Z平面: {z_text}")


def analyze_formal(document, path):
    print("=" * 72)
    print(f"[正式模式] {path}")
    print(f"生成时间: {document.get('生成时间')}  图像: {document.get('图像')}")
    print(f"托盘旋转角: {_fmt(document.get('托盘旋转角deg'))} deg")
    print("\n标定信息:")
    print_calibration_info(document)

    points = collect_grid_points(document)
    if len(points) < 6:
        print(f"\n有效格点仅 {len(points)} 个，无法做网格拟合分析")
        return
    print(f"\n有效格点: {len(points)}/{BOARD_ROW_COUNT * BOARD_COL_COUNT}")

    rows, cols, tcp, delta = fit_ideal_grid_residuals(points)
    magnitude = np.hypot(delta[:, 0], delta[:, 1])
    print("\n■ 相对理想等距网格的残差（排除整体缩放/旋转/剪切后的局部畸变）:")
    print_stats("残差模长 mm", magnitude)
    print_stats("残差dX mm", delta[:, 0])
    print_stats("残差dY mm", delta[:, 1])

    print("\n■ 按行分组残差均值（暴露左右/上下半区系统漂移）:")
    print("  行   dX均值   dY均值   |残差|   Z均值")
    for row in range(1, BOARD_ROW_COUNT + 1):
        mask = rows == row
        if not mask.any():
            continue
        print(
            f"  {row:>3d}  {_fmt(delta[mask, 0].mean(), 3):>8}  "
            f"{_fmt(delta[mask, 1].mean(), 3):>8}  "
            f"{_fmt(magnitude[mask].mean(), 3):>8}  {_fmt(tcp[mask, 2].mean(), 2)}"
        )
    print("\n■ 按列分组残差均值:")
    print("  列   dX均值   dY均值   |残差|   Z均值")
    for col in range(1, BOARD_COL_COUNT + 1):
        mask = cols == col
        if not mask.any():
            continue
        print(
            f"  {col:>3d}  {_fmt(delta[mask, 0].mean(), 3):>8}  "
            f"{_fmt(delta[mask, 1].mean(), 3):>8}  "
            f"{_fmt(magnitude[mask].mean(), 3):>8}  {_fmt(tcp[mask, 2].mean(), 2)}"
        )

    print(f"\n■ 残差 Top-{min(TOP_N, len(points))}:")
    order = np.argsort(magnitude)[::-1][:TOP_N]
    print("  行   列   dX      dY      |残差|   像素(u,v)")
    pixels = {int(row): {} for row in rows}
    for row, col, pixel, _tcp in points:
        pixels.setdefault(int(row), {})[int(col)] = pixel
    for index in order:
        pixel = pixels.get(int(rows[index]), {}).get(int(cols[index]), ("-", "-"))
        print(
            f"  {int(rows[index]):>3d}  {int(cols[index]):>3d}  "
            f"{_fmt(delta[index, 0]):>7}  {_fmt(delta[index, 1]):>7}  "
            f"{_fmt(magnitude[index]):>7}   ({_fmt(pixel[0], 1)},{_fmt(pixel[1], 1)})"
        )

    print("\n■ 相邻格点 TCP 间距（刚性托盘应等距，波动=标定畸变+格点识别误差）:")
    col_spacings, row_spacings, _grid = neighbor_spacing(points)
    print_stats("列向间距 mm（同行相邻列）", col_spacings)
    print_stats("行向间距 mm（同列相邻行）", row_spacings)

    print("\n■ 预测 TCP 的 Z 平面（在 140 点上重拟合，对照 yaml 系数看斜面是否一致）:")
    coefficients, rmse = fit_tcp_z_plane(tcp)
    print(
        f"  重拟合: a={coefficients[0]:+.6f} b={coefficients[1]:+.6f} "
        f"c={coefficients[2]:.3f}  RMSE={rmse:.3f} mm"
    )
    print_stats("格点 TCP_Z mm", tcp[:, 2])

    blocks = document.get("方块", [])
    print(f"\n■ 方块预测（{len(blocks)} 个）:")
    print("  类别  像素(u,v)        观察TCP(X,Y,Z)             表面Z   抓取Z  凸包内")
    for block in blocks:
        print(
            f"  {str(block.get('类别')):>4}  "
            f"({_fmt(block.get('像素u'), 1)},{_fmt(block.get('像素v'), 1)})"
            f"{'':>6}({_fmt(block.get('观察TCP_X'))}, {_fmt(block.get('观察TCP_Y'))}, "
            f"{_fmt(block.get('观察TCP_Z'))})  {_fmt(block.get('表面Z毫米'))}  "
            f"{_fmt(block.get('抓取Z毫米'))}  {block.get('凸包内')}"
        )
        if block.get("错误"):
            print(f"       错误: {block['错误']}")

    unsafe = [
        entry
        for entry in document.get("托盘格点", [])
        if entry.get("安全越界轴")
    ]
    if unsafe:
        print(f"\n⚠ 安全越界格点 {len(unsafe)} 个:")
        for entry in unsafe[:10]:
            print(
                f"  ({entry['行']},{entry['列']}) 越界轴={entry['安全越界轴']} "
                f"TCP=({entry['TCP_X']:.1f},{entry['TCP_Y']:.1f},{entry['TCP_Z']:.1f})"
            )

    if PLOT:
        plot_formal(document, path, rows, cols, tcp, delta, magnitude)


def plot_formal(document, path, rows, cols, tcp, delta, magnitude):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"\n（matplotlib 不可用，跳过画图: {exc}）")
        return

    figure, axes = plt.subplots(1, 2, figsize=(16, 8))
    axis = axes[0]
    quiver = axis.quiver(
        tcp[:, 0],
        tcp[:, 1],
        delta[:, 0],
        delta[:, 1],
        magnitude,
        cmap="jet",
        angles="xy",
        scale_units="xy",
        scale=5.0,
    )
    axis.plot(tcp[:, 0], tcp[:, 1], ".", color="lightgray", markersize=3)
    axis.invert_yaxis()
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("TCP X (mm)")
    axis.set_ylabel("TCP Y (mm)")
    axis.set_title(f"残差矢量图（放大5倍，均值|残差|={magnitude.mean():.2f}mm）")
    figure.colorbar(quiver, ax=axis, label="残差模长 mm")

    axis = axes[1]
    grid_magnitude = np.full((BOARD_ROW_COUNT, BOARD_COL_COUNT), np.nan)
    for row, col, value in zip(
        rows.astype(int), cols.astype(int), magnitude
    ):
        grid_magnitude[row - 1, col - 1] = value
    image = axis.pcolormesh(
        np.arange(1, BOARD_COL_COUNT + 2),
        np.arange(1, BOARD_ROW_COUNT + 2),
        grid_magnitude,
        cmap="jet",
        shading="flat",
    )
    axis.invert_yaxis()
    axis.set_xlabel("列")
    axis.set_ylabel("行")
    axis.set_title("残差模长热力图（行1=画面下方）")
    figure.colorbar(image, ax=axis, label="残差模长 mm")

    figure.suptitle(
        f"高位定位全景（正式） {document.get('生成时间')} 生成: {Path(path).name}"
    )
    figure.tight_layout()
    output_path = str(path) + "分析.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"\n图已保存: {output_path}")


def analyze_calibration(document, path):
    print("=" * 72)
    print(f"[标定模式] {path}")
    print(f"生成时间: {document.get('生成时间')}  图像: {document.get('图像')}")
    plane = document.get("方块观察Z平面")
    if plane:
        a, b, c = plane["系数a_b_c"]
        print(
            f"方块观察 Z 平面: a={a:+.6f} b={b:+.6f} c={c:.3f} "
            f"RMSE={plane['RMSE毫米']:.3f} mm"
        )
    else:
        print("方块观察 Z 平面: 未拟合（未识别到托盘）")

    blocks = document.get("方块", [])
    print(f"\n■ 方块深度实测（{len(blocks)} 个）:")
    print("  类别  深度世界(X,Y,Z)             观察TCP_Z  MAD    抓取Z")
    for block in blocks:
        print(
            f"  {str(block.get('类别')):>4}  "
            f"({_fmt(block.get('深度世界X'))}, {_fmt(block.get('深度世界Y'))}, "
            f"{_fmt(block.get('深度世界Z'))}){'':>4}"
            f"{_fmt(block.get('观察TCP_Z')):>9}  "
            f"{_fmt(block.get('深度MAD毫米')):>5}  {_fmt(block.get('抓取Z毫米'))}"
        )
    world_z = [
        block["深度世界Z"]
        for block in blocks
        if block.get("深度世界Z") is not None
    ]
    if world_z:
        print_stats("方块表面深度世界Z mm", world_z)

    tray = document.get("托盘采样点", [])
    print(f"\n■ 托盘采样点深度实测（{len(tray)} 个）:")
    print("  行    列    深度世界(X,Y,Z)             托盘TCP_Z")
    for entry in tray:
        print(
            f"  {_fmt(entry.get('行'), 1):>4}  {_fmt(entry.get('列'), 1):>4}  "
            f"({_fmt(entry.get('深度世界X'))}, {_fmt(entry.get('深度世界Y'))}, "
            f"{_fmt(entry.get('深度世界Z'))}){'':>4}{_fmt(entry.get('托盘TCP_Z')):>9}"
        )
    tray_world_z = [
        entry["深度世界Z"] for entry in tray if entry.get("深度世界Z") is not None
    ]
    if tray_world_z:
        print_stats("托盘点深度世界Z mm", tray_world_z)


def build_grid_tcp_map(formal_document):
    """{(row,col): tcp_xyz}，仅整数格点，供小数行列双线性插值。"""
    grid = {}
    for entry in formal_document.get("托盘格点", []):
        if entry.get("错误") or entry.get("TCP_X") is None:
            continue
        row, col = entry["行"], entry["列"]
        if float(row).is_integer() and float(col).is_integer():
            grid[(int(row), int(col))] = np.array(
                [entry["TCP_X"], entry["TCP_Y"], entry["TCP_Z"]]
            )
    return grid


def bilinear_grid_tcp(grid, row, col):
    """在整数格点 TCP 上按行列双线性插值（与 interpolate_grid_point 同思路）。"""
    row0 = max(1, int(math.floor(row)))
    col0 = max(1, int(math.floor(col)))
    row1 = min(BOARD_ROW_COUNT, row0 + 1)
    col1 = min(BOARD_COL_COUNT, col0 + 1)
    row_t = float(row) - row0
    col_t = float(col) - col0
    corners = (
        grid.get((row0, col0)),
        grid.get((row0, col1)),
        grid.get((row1, col0)),
        grid.get((row1, col1)),
    )
    if any(corner is None for corner in corners):
        raise KeyError(f"({row},{col}) 周围格点缺失，无法插值")
    top = corners[0] * (1.0 - col_t) + corners[1] * col_t
    bottom = corners[2] * (1.0 - col_t) + corners[3] * col_t
    return top * (1.0 - row_t) + bottom * row_t


def load_depth_localizer():
    """与标定模式同一套装配：shooting_pose + 手眼矩阵 → 相机偏移。"""
    with open(EXECUTION_CONFIG_PATH, "r", encoding="utf-8") as file:
        execution_config = yaml.safe_load(file) or {}
    with open(PERCEPTION_CONFIG_PATH, "r", encoding="utf-8") as file:
        perception_config = yaml.safe_load(file) or {}

    def find_hand_eye(node):
        if isinstance(node, dict):
            if "hand_eye_matrix" in node:
                return node["hand_eye_matrix"]
            for value in node.values():
                found = find_hand_eye(value)
                if found is not None:
                    return found
        return None

    hand_eye_relative = find_hand_eye(perception_config)
    if not hand_eye_relative:
        raise ValueError("perception.yaml 里找不到 hand_eye_matrix")
    hand_eye_path = Path(hand_eye_relative)
    if not hand_eye_path.is_absolute():
        hand_eye_path = SRC_DIR / hand_eye_path
    return DepthRoughLocalizer(
        execution_config["shooting_pose"],
        np.load(hand_eye_path),
    )


def observation_height_mm():
    with open(PERCEPTION_CONFIG_PATH, "r", encoding="utf-8") as file:
        perception_config = yaml.safe_load(file) or {}
    return float(perception_config["pick_height"]["block_observation_height_mm"])


def compare_pair(formal_document, formal_path, calibration_document, calibration_path):
    print("=" * 72)
    print("[配对对比] 正式预测 vs 标定深度实测")
    print(f"  正式: {formal_path}")
    print(f"  标定: {calibration_path}")
    print("  （两份文件应对同一物理摆位先后采集，否则对比无意义）")
    try:
        localizer = load_depth_localizer()
    except Exception as exc:
        print(f"  无法装配深度换算（跳过对比）: {exc}")
        return

    grid = build_grid_tcp_map(formal_document)
    if len(grid) < BOARD_ROW_COUNT * BOARD_COL_COUNT:
        print(f"  ⚠ 正式格点仅 {len(grid)}/140 有效，插值可能跳过部分采样点")

    tray = calibration_document.get("托盘采样点", [])
    records = []
    for entry in tray:
        if entry.get("深度世界X") is None:
            continue
        try:
            predicted = bilinear_grid_tcp(grid, entry["行"], entry["列"])
        except KeyError:
            continue
        depth_tcp_xy = localizer.tcp_xy_from_world(
            [entry["深度世界X"], entry["深度世界Y"], entry["深度世界Z"]]
        )
        diff_xy = predicted[:2] - depth_tcp_xy
        records.append((entry["行"], entry["列"], predicted, depth_tcp_xy, diff_xy))
    if records:
        print(f"\n■ 托盘采样点: 预测TCP − 深度换算TCP（{len(records)} 点）:")
        print("  行    列    dX       dY       预测Z     标定托盘Z")
        diffs = np.array([record[4] for record in records])
        for row, col, predicted, _depth_xy, diff in records:
            source = next(
                (
                    item
                    for item in tray
                    if item["行"] == row and item["列"] == col
                ),
                {},
            )
            print(
                f"  {_fmt(row, 1):>4}  {_fmt(col, 1):>4}  "
                f"{_fmt(diff[0], 3):>8}  {_fmt(diff[1], 3):>8}  "
                f"{_fmt(predicted[2]):>8}  {_fmt(source.get('托盘TCP_Z')):>8}"
            )
        print_stats("托盘 dX mm", diffs[:, 0])
        print_stats("托盘 dY mm", diffs[:, 1])
        print(
            "  解读: dX/dY 均值大 = XY 标定模型系统偏移；标准差大/分区域不同 = "
            "局部畸变（格点识别或模型欠拟合）；预测Z与标定托盘Z差 = Z 平面不一致。"
        )
    else:
        print("\n■ 托盘采样点: 没有可对比的点")

    formal_blocks = formal_document.get("方块", [])
    calibration_blocks = calibration_document.get("方块", [])
    if formal_blocks and calibration_blocks:
        height = observation_height_mm()
        print(f"\n■ 方块配对（最近像素 ≤60px，观察高度 {height} mm）:")
        print("  类别  dX       dY       预测观察Z  深度观察Z")
        for calibration_block in calibration_blocks:
            if calibration_block.get("深度世界X") is None:
                continue
            pixel = np.array(
                [calibration_block["像素u"], calibration_block["像素v"]]
            )
            candidates = [
                formal_block
                for formal_block in formal_blocks
                if not formal_block.get("错误")
                and formal_block.get("类别") == calibration_block.get("类别")
            ]
            if not candidates:
                continue
            distances = [
                float(
                    np.hypot(
                        formal_block["像素u"] - pixel[0],
                        formal_block["像素v"] - pixel[1],
                    )
                )
                for formal_block in candidates
            ]
            best = candidates[int(np.argmin(distances))]
            if min(distances) > 60.0:
                print(
                    f"  {calibration_block.get('类别')}: 最近像素距离 {min(distances):.0f}px 超限，跳过"
                )
                continue
            depth_tcp_xy = localizer.tcp_xy_from_world(
                [
                    calibration_block["深度世界X"],
                    calibration_block["深度世界Y"],
                    calibration_block["深度世界Z"],
                ]
            )
            diff_xy = (
                np.array([best["观察TCP_X"], best["观察TCP_Y"]]) - depth_tcp_xy
            )
            depth_observation_z = calibration_block["深度世界Z"] + height
            print(
                f"  {str(best.get('类别')):>4}  {_fmt(diff_xy[0], 3):>8}  "
                f"{_fmt(diff_xy[1], 3):>8}  {_fmt(best.get('观察TCP_Z')):>8}  "
                f"{_fmt(depth_observation_z):>8}"
            )


def main():
    input_path = Path(INPUT_PATH) if INPUT_PATH else load_latest_panorama()
    document = load_document(input_path)
    mode = document.get("模式")
    if mode == "正式":
        analyze_formal(document, input_path)
    elif mode == "标定":
        analyze_calibration(document, input_path)
    else:
        raise ValueError(f"未知模式字段: {mode!r}（文件: {input_path}）")

    if PAIR_PATH:
        other_path = Path(PAIR_PATH)
        other = load_document(other_path)
        if mode == "正式":
            compare_pair(document, input_path, other, other_path)
        else:
            compare_pair(other, other_path, document, input_path)


if __name__ == "__main__":
    main()
