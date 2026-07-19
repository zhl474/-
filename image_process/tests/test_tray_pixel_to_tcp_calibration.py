"""托盘高位像素到 TCP 标定的共享核心与格点诊断测试。"""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from image_process_lib.pixel_to_tcp_calibration import load_pixel_to_tcp_calibration
from tools.vision.visual_servo_geometry_analysis import TRAY_ANALYSIS_SPEC, run_geometry_analysis


def test_tray_analysis_outputs_grid_diagnostics_and_subject_yaml(tmp_path):
    """34 个合成托盘格点应导出仿射标定、行列 OOF 误差和 tray 元数据。"""
    rows = []
    for task_index in range(34):
        tray_row = 1.0 + task_index // 6
        tray_col = 1.0 + task_index % 6
        pixel_x = 500.0 + tray_col * 37.0
        pixel_y = 100.0 + tray_row * 29.0
        tcp_x = -0.45 * pixel_x + 0.02 * pixel_y + 100.0
        tcp_y = 0.01 * pixel_x + 0.52 * pixel_y - 150.0
        tcp_z = 190.0 + 0.01 * tcp_x + 0.02 * tcp_y
        world_x = -250.0 + tray_col * 20.0
        world_y = -80.0 + tray_row * 25.0
        world_z = 3.0 + 0.01 * world_x + 0.02 * world_y
        common = {
            "运行编号": 1,
            "任务序号": task_index + 1,
            "方块类别": "托盘目标",
            "托盘行": tray_row,
            "托盘列": tray_col,
            "高位检测像素X": pixel_x,
            "高位检测像素Y": pixel_y,
            "高位世界坐标X": world_x,
            "高位世界坐标Y": world_y,
            "高位世界坐标Z": world_z,
            "实测TCP位置X": tcp_x,
            "实测TCP位置Y": tcp_y,
            "实测TCP位置Z": tcp_z,
        }
        rows.append({**common, "事件": "伺服开始"})
        rows.append({**common, "事件": "伺服成功"})

    input_path = tmp_path / "托盘标定输入.csv"
    output_dir = tmp_path / "托盘分析结果"
    pd.DataFrame(rows).to_csv(input_path, index=False, encoding="utf-8-sig")
    run_geometry_analysis(
        SimpleNamespace(
            input_csv=str(input_path),
            output_dir=str(output_dir),
            expected_count=34,
            plane_rmse_tol=None,
            plane_max_tol=None,
            planarity_ratio_tol=0.01,
            parallel_angle_tol_deg=1.0,
            world_to_tcp_rotation=None,
            unique_value_tol=1e-9,
            cv_folds=5,
            random_seed=42,
            simple_model_slack=0.05,
            skip_plots=True,
        ),
        TRAY_ANALYSIS_SPEC,
    )

    calibration_path = output_dir / "托盘像素到TCP标定结果.yaml"
    grid_error_path = output_dir / "托盘格点OOF误差.csv"
    calibration = load_pixel_to_tcp_calibration(
        calibration_path,
        expected_subject="tray",
    )
    grid_errors = pd.read_csv(grid_error_path, encoding="utf-8-sig")

    assert calibration.model_name == "affine"
    assert calibration.metadata["calibration_subject"] == "tray"
    assert calibration.metadata["metrics"]["sample_count"] == 34
    assert calibration.metadata["coverage"]["extrapolation_policy"] == "diagnostic_only"
    assert {"托盘行", "托盘列", "affine_OOF三维误差"}.issubset(grid_errors.columns)
    assert len(grid_errors) == 34
