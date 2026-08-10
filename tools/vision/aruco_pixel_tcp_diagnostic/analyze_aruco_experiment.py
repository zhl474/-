#!/home/zhl/fr3env/fr3env/bin/python
"""ArUco 高位像素—低位 TCP 独立诊断实验：离线模型对比。

只读取实验 CSV，不生成也不覆盖任何正式 calibration YAML，不做去畸变。
直接复用正式标定分析中的映射函数（fit_xy_mapping / cross_validate_xy_mapping 等）。
参数集中在文件开头，不使用命令行参数。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ==================== 运行参数（直接修改本文件后运行）====================
INPUT_CSV = Path("/home/zhl/桌面/aruco诊断实验/20260808-220525/aruco实验数据.csv")  # 实机实验输出的 CSV 路径
OUTPUT_DIR = Path("/home/zhl/桌面/aruco诊断实验/20260808-220525/离线分析")  # 离线分析结果输出目录
CV_FOLDS = 5  # 交叉验证折数
RANDOM_SEED = 42  # 交叉验证固定随机种子（保证结果可复现）
SIMPLE_MODEL_SLACK = 0.05  # 选型容差：最简模型 CV 误差不超过最优模型误差的 1+该值 时优先选择最简模型

TOOLS_VISION_DIR = Path(__file__).resolve().parents[1]
SRC_ROOT = TOOLS_VISION_DIR.parents[1]
import sys  # noqa: E402

for _path in (TOOLS_VISION_DIR, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pixel_to_tcp_calibration_analysis import (  # noqa: E402
    MODEL_NAMES,
    choose_xy_model,
    configure_matplotlib,
    cross_validate_xy_mapping,
    fit_xy_mapping,
    predict_xy_mapping,
    read_csv_auto,
)

# ==================== 列与标签定义（与实机 CSV 对齐）====================
COL_EVENT = "事件"
COL_HIGH_UV = ["高位检测像素X", "高位检测像素Y"]
COL_ACTUAL_XY = ["实测TCP位置X", "实测TCP位置Y"]
COL_ZERO_XY = ["零误差等效TCP位置X", "零误差等效TCP位置Y"]
SUCCESS_EVENT = "伺服成功"

LABEL_MAIN = "zero_error"
LABEL_ACTUAL = "actual"
LABEL_NAMES = {LABEL_MAIN: "主标签 high_uv→零误差等效TCP", LABEL_ACTUAL: "辅助标签 high_uv→实测TCP"}
LABEL_COLUMNS = {LABEL_MAIN: COL_ZERO_XY, LABEL_ACTUAL: COL_ACTUAL_XY}

MODEL_CONTRAST_COLUMNS = [
    "标签类型", "模型", "样本数", "折数",
    "训练状态", "训练二维RMSE_mm", "训练X_RMSE_mm", "训练Y_RMSE_mm",
    "训练二维MAE_mm", "训练二维P95_mm", "训练最大误差_mm",
    "CV状态", "CV二维RMSE_mm", "CV二维MAE_mm", "CV二维P95_mm",
    "CV二维最大误差_mm", "CV_X_RMSE_mm", "CV_Y_RMSE_mm", "失败原因",
]
OOF_COLUMNS = [
    "标签类型", "样本序号", "高位检测像素X", "高位检测像素Y",
    "标签TCP位置X", "标签TCP位置Y",
] + [f"{name}_OOF误差_mm" for name in MODEL_NAMES]


def _metrics_for(fitted_model, model_name, uv, tcp_xy):
    """全量训练的二维误差指标。"""
    predictions = predict_xy_mapping(model_name, fitted_model, uv)
    residual = predictions - tcp_xy
    distance = np.linalg.norm(residual, axis=1)
    axis_rmse = np.sqrt(np.mean(residual**2, axis=0))
    return {
        "训练状态": "成功",
        "训练二维RMSE_mm": float(np.sqrt(np.mean(distance**2))),
        "训练X_RMSE_mm": float(axis_rmse[0]),
        "训练Y_RMSE_mm": float(axis_rmse[1]),
        "训练二维MAE_mm": float(np.mean(distance)),
        "训练二维P95_mm": float(np.percentile(distance, 95)),
        "训练最大误差_mm": float(np.max(distance)),
    }


def analyze_label_set(uv: np.ndarray, tcp_xy: np.ndarray):
    """对一种标签分别拟合四种模型；返回逐模型结果、逐模型 OOF 与选型建议。

    单个模型全量拟合失败时只记训练失败；全量可行但 CV 无法完成时保留训练
    指标并将 CV 状态标为“数据不足”，不强行改变验证方法。
    """
    uv = np.asarray(uv, dtype=float)
    tcp_xy = np.asarray(tcp_xy, dtype=float)
    if uv.ndim != 2 or tcp_xy.ndim != 2 or uv.shape[0] != tcp_xy.shape[0]:
        raise ValueError("像素与 TCP 标签点数量不一致")
    if not np.isfinite(uv).all() or not np.isfinite(tcp_xy).all():
        raise ValueError("像素或 TCP 标签包含非有限数值")

    results = []
    oof_per_model = {}
    for model_name in MODEL_NAMES:
        entry = {
            "模型": model_name,
            "样本数": int(len(uv)),
            "折数": int(CV_FOLDS),
            "训练状态": "失败",
            "CV状态": "未执行",
            "失败原因": "",
        }
        try:
            fitted = fit_xy_mapping(model_name, uv, tcp_xy)
        except Exception as exc:  # noqa: BLE001
            entry["失败原因"] = f"全量训练失败: {exc}"
            results.append(entry)
            continue
        entry.update(_metrics_for(fitted, model_name, uv, tcp_xy))
        try:
            summary, oof = cross_validate_xy_mapping(
                model_name, uv, tcp_xy, int(CV_FOLDS), int(RANDOM_SEED)
            )
        except Exception as exc:  # noqa: BLE001
            entry["CV状态"] = "数据不足"
            entry["失败原因"] = f"交叉验证无法完成: {exc}"
            results.append(entry)
            continue
        for key, value in summary.items():
            if key not in ("模型", "样本数", "折数"):
                entry[key] = value
        entry["CV状态"] = "完成"
        oof_per_model[model_name] = oof
        results.append(entry)

    completed = [entry for entry in results if entry.get("CV状态") == "完成"]
    if completed:
        selection = choose_xy_model(completed)
    else:
        selection = {
            "CV最优模型": None,
            "CV最优RMSE_mm": None,
            "推荐最简模型": "无",
            "说明": "所有模型均无法完成交叉验证（样本不足或数值退化）",
        }
    return results, oof_per_model, selection


def analyze_experiment(input_csv: Path, output_dir: Path) -> dict:
    """从实验 CSV 读取成功样本，输出模型对比、OOF 误差与分析图表。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df, encoding = read_csv_auto(input_csv)
    report = {
        "输入CSV": str(input_csv),
        "CSV编码": encoding,
        "输出目录": str(output_dir),
        "总行数": int(len(df)),
    }

    success = df[df[COL_EVENT] == SUCCESS_EVENT].copy()
    report["成功行数"] = int(len(success))
    for column in [*COL_HIGH_UV, *COL_ACTUAL_XY, *COL_ZERO_XY]:
        if column in success.columns:
            success[column] = pd.to_numeric(success[column], errors="coerce")

    contrast_rows = []
    oof_rows = []
    label_reports = {}
    plot_data = {}
    for label_key in (LABEL_MAIN, LABEL_ACTUAL):
        label_columns = LABEL_COLUMNS[label_key]
        available = [column for column in label_columns if column in success.columns]
        if len(available) != 2:
            label_reports[label_key] = {"样本数": 0, "错误": "CSV 缺少标签列"}
            continue
        finite = success.loc[
            np.isfinite(success[label_columns].to_numpy(dtype=float)).all(axis=1)
            & np.isfinite(success[COL_HIGH_UV].to_numpy(dtype=float)).all(axis=1)
        ]
        label_report = {
            "标签名称": LABEL_NAMES[label_key],
            "样本数": int(len(finite)),
        }
        if len(finite) < 1:
            label_reports[label_key] = label_report
            continue
        uv = finite[COL_HIGH_UV].to_numpy(dtype=float)
        tcp_xy = finite[label_columns].to_numpy(dtype=float)
        results, oof_per_model, selection = analyze_label_set(uv, tcp_xy)
        oof_distances = {
            model_name: np.linalg.norm(oof - tcp_xy, axis=1)
            for model_name, oof in oof_per_model.items()
        }
        if label_key == LABEL_MAIN:
            plot_data["uv"] = uv
            plot_data["oof_distances"] = oof_distances
        for entry in results:
            row = {"标签类型": LABEL_NAMES[label_key]}
            row.update(entry)
            contrast_rows.append(row)
        for index in range(len(uv)):
            oof_row = {
                "标签类型": LABEL_NAMES[label_key],
                "样本序号": int(index) + 1,
                "高位检测像素X": uv[index, 0],
                "高位检测像素Y": uv[index, 1],
                "标签TCP位置X": tcp_xy[index, 0],
                "标签TCP位置Y": tcp_xy[index, 1],
            }
            for model_name in MODEL_NAMES:
                if model_name in oof_distances:
                    oof_row[f"{model_name}_OOF误差_mm"] = float(oof_distances[model_name][index])
                else:
                    oof_row[f"{model_name}_OOF误差_mm"] = ""
            oof_rows.append(oof_row)
        label_report["模型结果"] = results
        label_report["选型建议"] = selection
        label_reports[label_key] = label_report
    report["标签"] = label_reports

    pd.DataFrame(contrast_rows, columns=MODEL_CONTRAST_COLUMNS).to_csv(
        output_dir / "模型对比.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(oof_rows, columns=OOF_COLUMNS).to_csv(
        output_dir / "逐样本OOF误差.csv", index=False, encoding="utf-8-sig"
    )
    report["结论"] = _conclusion_text(label_reports)
    (output_dir / "分析报告.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_cv_comparison(label_reports, output_dir / "模型CV误差对比.png")
    _plot_oof_spatial(plot_data, output_dir / "OOF误差空间分布.png")
    return report


def _cv_rmse(label_reports, label_key, model_name):
    entry = _model_entry(label_reports, label_key, model_name)
    if entry is None or entry.get("CV状态") != "完成":
        return None
    return float(entry["CV二维RMSE"])


def _model_entry(label_reports, label_key, model_name):
    label_report = label_reports.get(label_key, {})
    for entry in label_report.get("模型结果", []):
        if entry["模型"] == model_name:
            return entry
    return None


def _conclusion_text(label_reports) -> str:
    """根据主标签模型对比生成简短结论文本。"""
    lines = []
    for label_key in (LABEL_MAIN, LABEL_ACTUAL):
        label_report = label_reports.get(label_key, {})
        name = label_report.get("标签名称", label_key)
        lines.append(f"{name}: 有效样本 {label_report.get('样本数', 0)} 个")
        affine = _cv_rmse(label_reports, label_key, "affine")
        if affine is None:
            lines.append("  affine 交叉验证不可用，无法比较。")
            continue
        best_value = None
        best_name = None
        for model_name in MODEL_NAMES:
            value = _cv_rmse(label_reports, label_key, model_name)
            if value is not None and (best_value is None or value < best_value):
                best_value = value
                best_name = model_name
        if best_name is None:
            lines.append("  所有模型 CV 均不可用。")
            continue
        lines.append(
            f"  affine CV={affine:.3f}mm，最优 {best_name} CV={best_value:.3f}mm"
            f"（改善 {100.0 * (1.0 - best_value / affine):.1f}%）"
        )
    main_affine = _cv_rmse(label_reports, LABEL_MAIN, "affine")
    main_best_value = min(
        (value for model_name in MODEL_NAMES
         for value in [_cv_rmse(label_reports, LABEL_MAIN, model_name)] if value is not None),
        default=None,
    )
    if main_affine is not None and main_best_value is not None:
        if main_best_value < main_affine * 0.9:
            lines.append(
                "结论：ArUco 目标下复杂模型仍显著优于 affine，"
                "标定非线性可能主要来自相机/几何链路而非方块识别目标定义。"
            )
        else:
            lines.append(
                "结论：ArUco 目标下各模型误差接近，"
                "原标定非线性可能主要来自方块/托盘识别目标定义或静止残差。"
            )
    else:
        lines.append("结论：主标签样本不足以完成模型对比，需要更多有效样本。")
    return "；".join(lines)


def _plot_cv_comparison(label_reports, output_path):
    configure_matplotlib()
    import matplotlib.pyplot as plt  # noqa: PLC0415

    figure, axis = plt.subplots(figsize=(9, 5))
    x_positions = np.arange(len(MODEL_NAMES))
    width = 0.35
    for offset, (label_key, display) in enumerate(
        ((LABEL_MAIN, "主标签"), (LABEL_ACTUAL, "辅助标签"))
    ):
        values = [_cv_rmse(label_reports, label_key, model_name) for model_name in MODEL_NAMES]
        bars = axis.bar(
            x_positions + (offset - 0.5) * width,
            [np.nan if value is None else value for value in values],
            width,
            label=display,
        )
        for bar, value in zip(bars, values):
            if value is not None:
                axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}",
                          ha="center", va="bottom", fontsize=8)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(MODEL_NAMES)
    axis.set_ylabel("CV 二维 RMSE (mm)")
    axis.set_title("ArUco 实验：像素到 TCP 模型交叉验证误差对比")
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _plot_oof_spatial(plot_data, output_path):
    configure_matplotlib()
    import matplotlib.pyplot as plt  # noqa: PLC0415

    uv = plot_data.get("uv")
    oof_distances = plot_data.get("oof_distances", {})
    if uv is None or not oof_distances:
        return
    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes_flat = axes.reshape(-1)
    for index, model_name in enumerate(MODEL_NAMES):
        axis = axes_flat[index]
        distances = oof_distances.get(model_name)
        if distances is None:
            axis.set_visible(False)
            continue
        scatter = axis.scatter(uv[:, 0], uv[:, 1], c=distances, cmap="viridis", s=40)
        figure.colorbar(scatter, ax=axis, label="OOF 二维误差 (mm)")
        axis.set_title(model_name)
        axis.set_xlabel("高位像素 X")
        axis.set_ylabel("高位像素 Y")
        axis.grid(linestyle="--", alpha=0.3)
    figure.suptitle("ArUco 实验：主标签逐样本 OOF 误差空间分布（高位像素空间）")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    report = analyze_experiment(INPUT_CSV, OUTPUT_DIR)
    print(json.dumps(report["结论"], ensure_ascii=False, indent=2))
    print(f"分析完成，输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
