"""精简像素到 TCP 标定工具的数据门禁与产物测试。"""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from image_process_lib.pixel_to_tcp_calibration import load_pixel_to_tcp_calibration
from tools.vision.pixel_to_tcp_calibration_analysis import (
    CalibrationJob,
    analyze_calibration_pair,
    analyze_job,
)


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


def _v2_rows(subject, count=34, *, block_mad=0.2):
    """构造带稳定深度诊断的新标定 CSV。"""
    rows = []
    for index in range(count):
        row_index, column_index = divmod(index, 7)
        pixel_x = 100.0 + column_index * 31.0
        pixel_y = 80.0 + row_index * 37.0
        tcp_x = 0.45 * pixel_x - 0.12 * pixel_y - 300.0
        tcp_y = -0.08 * pixel_x + 0.52 * pixel_y - 35.0
        block_target_z = 0.01 * tcp_x + 0.02 * tcp_y + 203.0
        if subject == "block":
            target_z = block_target_z
            world_z = target_z - 192.0
            actual_z = target_z + (index % 2) * 0.3
            source = "stable_depth_xyz"
            depth_mad = block_mad
        else:
            # 托盘深度 Z 和实测反馈 Z 都故意大幅变化，生成器不得用于托盘 Z 平面。
            target_z = block_target_z - 7.0
            world_z = 500.0 + (index % 5) * 25.0
            actual_z = 800.0 + index * 3.0
            source = "stable_depth_xy_block_plane_z"
            depth_mad = 20.0
        rows.append({
            "方块类别": f"测试类别{index % 7}",
            "事件": "伺服成功",
            "高位检测像素X": pixel_x,
            "高位检测像素Y": pixel_y,
            "高位世界坐标Z": world_z,
            "深度有效帧数": 15,
            "深度MAD毫米": depth_mad,
            "粗定位来源": source,
            "标定目标TCP位置Z": target_z,
            "实测TCP位置X": tcp_x,
            "实测TCP位置Y": tcp_y,
            "实测TCP位置Z": actual_z,
        })
    return rows


def _add_experiment_fields(rows, session_id="diagnostic_session"):
    enriched = []
    for index, source in enumerate(rows, start=1):
        row = dict(source)
        row.update({
            "实验批次ID": session_id,
            "任务序号": index,
            "目标类型": "方块" if "stable_depth_xyz" == row["粗定位来源"] else "托盘",
            "高位检测角度deg": float((index % 6) * 15),
            "最终命令TCP位置X": row["实测TCP位置X"] - 0.05,
            "最终命令TCP位置Y": row["实测TCP位置Y"] + 0.03,
            "实测减命令TCP位置X": 0.05,
            "实测减命令TCP位置Y": -0.03,
            "最终像素误差X": 0.2,
            "最终像素误差Y": -0.1,
            "静止像素误差均值X": 1.0,
            "静止像素误差均值Y": 2.0,
            "静止像素误差标准差X": 0.1,
            "静止像素误差标准差Y": 0.2,
            "静止采样有效帧数": 20,
            "静止采样请求帧数": 20,
            "静止采样完整": True,
            "零误差等效TCP位置X": row["实测TCP位置X"] + 0.4,
            "零误差等效TCP位置Y": row["实测TCP位置Y"] + 0.2,
        })
        enriched.append(row)
    return enriched


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


def test_pair_analysis_generates_same_batch_v2_and_derived_tray_plane(tmp_path):
    block_job = _make_job(tmp_path, "block", _v2_rows("block"))
    tray_job = _make_job(tmp_path, "tray", _v2_rows("tray"))

    assert analyze_calibration_pair(block_job, tray_job) is True

    block_path = block_job.output_dir / block_job.calibration_filename
    tray_path = tray_job.output_dir / tray_job.calibration_filename
    block_document = yaml.safe_load(block_path.read_text(encoding="utf-8"))
    tray_document = yaml.safe_load(tray_path.read_text(encoding="utf-8"))
    block_calibration = load_pixel_to_tcp_calibration(block_path, expected_subject="block")
    tray_calibration = load_pixel_to_tcp_calibration(tray_path, expected_subject="tray")

    assert block_document["schema_version"] == tray_document["schema_version"] == 2
    assert block_document["generation_id"] == tray_document["generation_id"]
    block_coefficients = np.asarray(block_document["z_plane"]["coefficients"])
    tray_coefficients = np.asarray(tray_document["z_plane"]["coefficients"])
    assert tray_coefficients[:2] == pytest.approx(block_coefficients[:2])
    assert block_coefficients[2] - tray_coefficients[2] == pytest.approx(7.0)
    pixel = [180.0, 150.0]
    assert block_calibration.predict(pixel)[2] - tray_calibration.predict(pixel)[2] == pytest.approx(7.0)
    report = json.loads(
        (block_job.output_dir / "标定检查报告.json").read_text(encoding="utf-8")
    )
    assert report["实验日志诊断"]["可用"] is False


def test_pair_analysis_outputs_experiment_diagnostics_for_schema_v3(tmp_path):
    session_id = "diagnostic_session"
    block_job = _make_job(
        tmp_path, "block", _add_experiment_fields(_v2_rows("block"), session_id)
    )
    tray_job = _make_job(
        tmp_path, "tray", _add_experiment_fields(_v2_rows("tray"), session_id)
    )
    archive_dir = tmp_path / "实验日志" / session_id
    archive_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"事件": "成功后静止帧", "任务序号": index} for index in range(1, 35)]
    ).to_csv(archive_dir / "方块视觉伺服逐轮.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{"事件": "成功后静止帧", "任务序号": index} for index in range(1, 35)]
    ).to_csv(archive_dir / "托盘视觉伺服逐轮.csv", index=False, encoding="utf-8-sig")

    assert analyze_calibration_pair(block_job, tray_job) is True

    report = json.loads(
        (block_job.output_dir / "标定检查报告.json").read_text(encoding="utf-8")
    )
    diagnostic = report["实验日志诊断"]
    assert diagnostic["可用"] is True
    assert diagnostic["静止采样完整目标数"] == 34
    assert set(diagnostic["不同标签模型比较"]) == {
        "实测TCP",
        "最终命令TCP",
        "零误差等效TCP",
    }
    assert diagnostic["逐轮日志"]["总行数"] == 34
    assert (block_job.output_dir / "逐目标终止诊断.csv").is_file()
    assert (block_job.output_dir / "实验空间矢量诊断.png").is_file()
    assert (block_job.output_dir / "CV随机种子稳定性.png").is_file()


def test_pair_analysis_removes_both_candidates_when_block_depth_gate_fails(tmp_path):
    block_job = _make_job(
        tmp_path,
        "block",
        _v2_rows("block", block_mad=1.01),
    )
    tray_job = _make_job(tmp_path, "tray", _v2_rows("tray"))
    block_path = block_job.output_dir / block_job.calibration_filename
    tray_path = tray_job.output_dir / tray_job.calibration_filename
    block_path.parent.mkdir(parents=True, exist_ok=True)
    tray_path.parent.mkdir(parents=True, exist_ok=True)
    block_path.write_text("旧候选", encoding="utf-8")
    tray_path.write_text("旧候选", encoding="utf-8")

    assert analyze_calibration_pair(block_job, tray_job) is False

    assert not block_path.exists()
    assert not tray_path.exists()
