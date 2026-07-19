"""像素到 TCP 标定 YAML 的生成、加载和安全边界测试。"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from competition_lib.pixel_to_tcp_calibration import load_pixel_to_tcp_calibration
from tools.vision.visual_servo_geometry_analysis import analyze


def _homography_payload():
    return {
        "schema_version": 1,
        "calibration_type": "pixel_to_tcp_position",
        "input": {"coordinate": "high_detection_pixel_xy", "unit": "pixel"},
        "output": {"coordinate": "tcp_position_xyz", "unit": "mm"},
        "model": {
            "name": "homography",
            "parameters": {
                "kind": "homography",
                "H": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "plane_centroid": [10.0, 20.0, 30.0],
                "plane_basis": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                "plane_normal": [0.0, 0.0, 1.0],
            },
        },
        "coverage": {
            "pixel_convex_hull": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        },
    }


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _run_affine_analysis(tmp_path: Path) -> Path:
    """构造 35 个平面点，验证分析脚本能生成可调用的仿射标定文件。"""
    rows = []
    for task_index in range(35):
        row_index, column_index = divmod(task_index, 7)
        pixel_x = 100.0 + column_index * 28.0
        pixel_y = 80.0 + row_index * 31.0
        tcp_x = 0.5 * pixel_x - 0.2 * pixel_y + 300.0
        tcp_y = -0.1 * pixel_x + 0.4 * pixel_y - 120.0
        tcp_z = 0.01 * tcp_x + 0.02 * tcp_y + 200.0
        world_x = -200.0 + column_index * 20.0
        world_y = 150.0 + row_index * 25.0
        world_z = 0.01 * world_x + 0.02 * world_y + 50.0
        common = {
            "运行编号": 1,
            "任务序号": task_index + 1,
            "方块类别": "测试方块",
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

    input_path = tmp_path / "标定输入.csv"
    output_dir = tmp_path / "分析结果"
    pd.DataFrame(rows).to_csv(input_path, index=False, encoding="utf-8-sig")
    analyze(
        SimpleNamespace(
            input_csv=str(input_path),
            output_dir=str(output_dir),
            expected_count=35,
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
        )
    )
    return output_dir


def test_analysis_outputs_loadable_affine_tcp_calibration(tmp_path):
    output_dir = _run_affine_analysis(tmp_path)
    calibration_path = output_dir / "像素到TCP标定结果.yaml"
    report_path = output_dir / "分析报告.json"

    assert calibration_path.is_file()
    calibration = load_pixel_to_tcp_calibration(calibration_path)
    assert calibration.model_name == "affine"
    assert calibration.metadata["metrics"]["sample_count"] == 35
    assert calibration.metadata["output"]["coordinate"] == "tcp_position_xyz"
    assert calibration.metadata["output"]["orientation_included"] is False

    pixel = [184.0, 142.0]
    expected = [0.5 * pixel[0] - 0.2 * pixel[1] + 300.0, -0.1 * pixel[0] + 0.4 * pixel[1] - 120.0, 0.0]
    expected[2] = 0.01 * expected[0] + 0.02 * expected[1] + 200.0
    assert np.allclose(calibration.predict(pixel), expected, atol=1e-8)
    assert '"selected_model": "affine"' in report_path.read_text(encoding="utf-8")


def test_homography_calibration_prediction_and_coverage_rejection(tmp_path):
    calibration = load_pixel_to_tcp_calibration(_write_yaml(tmp_path / "单应.yaml", _homography_payload()))

    assert np.allclose(calibration.predict([2.0, 3.0]), [12.0, 23.0, 30.0])
    assert calibration.is_pixel_within_coverage([10.0, 5.0]) is True
    with pytest.raises(ValueError, match="凸包之外"):
        calibration.predict([11.0, 5.0])
    assert np.allclose(
        calibration.predict([11.0, 5.0], allow_extrapolation=True),
        [21.0, 25.0, 30.0],
    )


def test_loader_rejects_invalid_model_shape_and_homography_denominator(tmp_path):
    malformed_path = tmp_path / "损坏.yaml"
    malformed_path.write_text("model: [未闭合", encoding="utf-8")
    with pytest.raises(ValueError, match="不是合法 YAML"):
        load_pixel_to_tcp_calibration(malformed_path)

    invalid_shape = _homography_payload()
    invalid_shape["model"] = {
        "name": "affine",
        "parameters": {
            "kind": "polynomial",
            "degree": 1,
            "uv_mean": [0.0, 0.0],
            "uv_scale": [1.0, 1.0],
            "feature_names": ["1", "u", "v"],
            "coef": [[1.0, 2.0, 3.0]],
        },
    }
    with pytest.raises(ValueError, match="coef"):
        load_pixel_to_tcp_calibration(_write_yaml(tmp_path / "坏系数.yaml", invalid_shape))

    zero_denominator = _homography_payload()
    zero_denominator["model"]["parameters"]["H"][2] = [0.0, 0.0, 0.0]
    calibration = load_pixel_to_tcp_calibration(_write_yaml(tmp_path / "零分母.yaml", zero_denominator))
    with pytest.raises(ValueError, match="齐次分母"):
        calibration.predict([2.0, 3.0])
