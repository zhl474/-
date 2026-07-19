"""精简像素到 TCP 标定工具的数据门禁与产物测试。"""

import json

import pandas as pd
import pytest

from image_process_lib.pixel_to_tcp_calibration import load_pixel_to_tcp_calibration
from tools.vision.pixel_to_tcp_calibration_analysis import CalibrationJob, analyze_job


def _calibration_rows(count=34, *, failure_index=None, nonplanar=False, collinear_pixels=False):
    """构造覆盖二维像素区域的仿射平面标定点。"""
    rows = []
    for index in range(count):
        row_index, column_index = divmod(index, 7)
        pixel_x = 100.0 + column_index * 31.0
        pixel_y = 80.0 if collinear_pixels else 80.0 + row_index * 37.0
        tcp_x = 0.45 * pixel_x - 0.12 * pixel_y + 20.0
        tcp_y = -0.08 * pixel_x + 0.52 * pixel_y - 35.0
        tcp_z = 190.0 + 0.01 * tcp_x + 0.02 * tcp_y
        if nonplanar:
            tcp_z += float((index % 3) - 1) * 12.0
        rows.append(
            {
                "方块类别": f"测试类别{index % 7}",
                "事件": "伺服失败" if index == failure_index else "伺服成功",
                "高位检测像素X": pixel_x,
                "高位检测像素Y": pixel_y,
                "实测TCP位置X": tcp_x,
                "实测TCP位置Y": tcp_y,
                "实测TCP位置Z": tcp_z,
            }
        )
    return rows


def _make_job(tmp_path, subject, rows, expected_count=34):
    label = "方块" if subject == "block" else "托盘"
    input_path = tmp_path / f"{label}标定.csv"
    output_dir = tmp_path / f"{label}结果"
    filename = f"{subject}_pixel_to_tcp_calibration.yaml"
    pd.DataFrame(rows).to_csv(input_path, index=False, encoding="utf-8-sig")
    return CalibrationJob(
        label=label,
        subject=subject,
        input_csv=input_path,
        expected_success_count=expected_count,
        output_dir=output_dir,
        calibration_filename=filename,
    )


@pytest.mark.parametrize("subject", ["block", "tray"])
def test_simple_analysis_outputs_loadable_affine_calibration(tmp_path, subject):
    job = _make_job(tmp_path, subject, _calibration_rows())

    assert analyze_job(job) is True

    calibration_path = job.output_dir / job.calibration_filename
    calibration = load_pixel_to_tcp_calibration(
        calibration_path,
        expected_subject=subject,
    )
    assert calibration.model_name == "affine"
    assert calibration.metadata["metrics"]["sample_count"] == 34
    assert (job.output_dir / "TCP平面拟合.png").is_file()
    assert (job.output_dir / "映射模型交叉验证对比.png").is_file()
    assert (job.output_dir / "逐点OOF误差.csv").is_file()
    report = json.loads((job.output_dir / "标定检查报告.json").read_text(encoding="utf-8"))
    assert report["质量门禁通过"] is True
    assert report["TCP平面检查"]["平面判定通过"] is True


def test_failure_row_and_missing_success_count_refuse_yaml(tmp_path):
    job = _make_job(tmp_path, "block", _calibration_rows(failure_index=3))

    assert analyze_job(job) is False

    assert not (job.output_dir / job.calibration_filename).exists()
    report = json.loads((job.output_dir / "标定检查报告.json").read_text(encoding="utf-8"))
    assert report["伺服失败数量"] == 1
    assert report["伺服成功数量"] == 33
    assert report["质量门禁通过"] is False


def test_nonplanar_tcp_points_refuse_yaml(tmp_path):
    job = _make_job(tmp_path, "block", _calibration_rows(nonplanar=True))

    assert analyze_job(job) is False

    report = json.loads((job.output_dir / "标定检查报告.json").read_text(encoding="utf-8"))
    assert report["TCP平面检查"]["平面判定通过"] is False
    assert not (job.output_dir / job.calibration_filename).exists()


def test_collinear_pixel_coverage_refuses_yaml_but_writes_report(tmp_path):
    job = _make_job(tmp_path, "tray", _calibration_rows(collinear_pixels=True))

    assert analyze_job(job) is False

    report = json.loads((job.output_dir / "标定检查报告.json").read_text(encoding="utf-8"))
    assert any("共线" in problem for problem in report["问题"])
    assert not (job.output_dir / job.calibration_filename).exists()
