"""像素到 TCP 标定 YAML 的生成、加载和安全边界测试。"""

from pathlib import Path
import numpy as np
import pytest
import yaml

from image_process_lib.pixel_to_tcp_calibration import load_pixel_to_tcp_calibration


SRC_DIR = Path(__file__).resolve().parents[2]


def _homography_payload(subject="block"):
    payload = {
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
    if subject is not None:
        payload["calibration_subject"] = subject
    return payload


def _v2_affine_payload(subject="block", generation_id="batch-1", z_c=100.0):
    """构造像素到 XY 仿射模型和独立 Z 平面。"""
    return {
        "schema_version": 2,
        "generation_id": generation_id,
        "calibration_type": "pixel_to_tcp_position",
        "calibration_subject": subject,
        "input": {"coordinate": "high_detection_pixel_xy", "unit": "pixel"},
        "output": {"coordinate": "tcp_position_xyz", "unit": "mm"},
        "xy_model": {
            "name": "affine",
            "parameters": {
                "kind": "polynomial",
                "degree": 1,
                "uv_mean": [0.0, 0.0],
                "uv_scale": [1.0, 1.0],
                "feature_names": ["1", "u", "v"],
                "coef": [[10.0, 20.0], [1.0, 0.0], [0.0, 1.0]],
            },
        },
        "z_plane": {
            "equation": "z = a*x + b*y + c",
            "coefficients": [0.1, 0.2, z_c],
            "source": "external_depth",
        },
        "coverage": {
            "pixel_convex_hull": [
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 10.0],
                [0.0, 10.0],
            ],
        },
    }


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def test_homography_calibration_uses_hull_only_for_coverage_diagnostics(tmp_path):
    calibration = load_pixel_to_tcp_calibration(
        _write_yaml(tmp_path / "单应.yaml", _homography_payload()),
        expected_subject="block",
    )

    assert np.allclose(calibration.predict([2.0, 3.0]), [12.0, 23.0, 30.0])
    assert calibration.is_pixel_within_coverage([10.0, 5.0]) is True
    assert calibration.is_pixel_within_coverage([11.0, 5.0]) is False
    assert np.allclose(
        calibration.predict([11.0, 5.0]),
        [21.0, 25.0, 30.0],
    )


def test_schema_v2_predicts_xy_first_and_z_from_independent_plane(tmp_path):
    calibration = load_pixel_to_tcp_calibration(
        _write_yaml(tmp_path / "v2方块.yaml", _v2_affine_payload()),
        expected_subject="block",
    )

    prediction = calibration.predict([2.0, 3.0])

    assert calibration.schema_version == 2
    assert prediction[:2] == pytest.approx([12.0, 23.0])
    assert prediction[2] == pytest.approx(0.1 * 12.0 + 0.2 * 23.0 + 100.0)


def test_schema_v2_rejects_missing_generation_and_invalid_z_plane(tmp_path):
    missing_generation = _v2_affine_payload()
    del missing_generation["generation_id"]
    with pytest.raises(ValueError, match="generation_id"):
        load_pixel_to_tcp_calibration(
            _write_yaml(tmp_path / "无批次.yaml", missing_generation),
            expected_subject="block",
        )

    invalid_plane = _v2_affine_payload()
    invalid_plane["z_plane"]["coefficients"] = [0.1, 0.2]
    with pytest.raises(ValueError, match="z_plane.coefficients"):
        load_pixel_to_tcp_calibration(
            _write_yaml(tmp_path / "坏平面.yaml", invalid_plane),
            expected_subject="block",
        )


def test_loader_rejects_subject_mismatch_and_missing_subject(tmp_path):
    block_path = _write_yaml(tmp_path / "方块.yaml", _homography_payload("block"))
    with pytest.raises(ValueError, match="期望 tray.*文件为 block"):
        load_pixel_to_tcp_calibration(block_path, expected_subject="tray")

    legacy_path = _write_yaml(tmp_path / "旧方块.yaml", _homography_payload(None))
    with pytest.raises(ValueError, match="期望 block.*文件为 缺失"):
        load_pixel_to_tcp_calibration(legacy_path, expected_subject="block")
    # 不指定主体时继续兼容旧分析输出。
    assert load_pixel_to_tcp_calibration(legacy_path).model_name == "homography"


def test_loader_reports_missing_file(tmp_path):
    with pytest.raises(ValueError, match="无法读取 TCP 标定文件"):
        load_pixel_to_tcp_calibration(
            tmp_path / "不存在的标定.yaml",
            expected_subject="block",
        )


def test_old_competition_calibration_module_is_removed_and_not_imported():
    old_module = SRC_DIR / "competition" / "competition_lib" / "pixel_to_tcp_calibration.py"
    assert not old_module.exists()

    stale_imports = []
    for package_name in ("camera", "competition", "control", "image_process"):
        for source_path in (SRC_DIR / package_name).rglob("*.py"):
            if "tests" in source_path.parts:
                continue
            source_text = source_path.read_text(encoding="utf-8")
            if "competition_lib.pixel_to_tcp_calibration" in source_text:
                stale_imports.append(str(source_path.relative_to(SRC_DIR)))
    assert stale_imports == []


@pytest.mark.parametrize("pixel", [[np.nan, 1.0], [np.inf, 1.0], [1.0, -np.inf]])
def test_prediction_rejects_nonfinite_pixel(tmp_path, pixel):
    calibration = load_pixel_to_tcp_calibration(
        _write_yaml(tmp_path / "有效标定.yaml", _homography_payload()),
        expected_subject="block",
    )

    with pytest.raises(ValueError, match="pixel_xy.*有限数值"):
        calibration.predict(pixel)


def test_prediction_rejects_nonfinite_overflow_result(tmp_path):
    overflow_payload = _homography_payload()
    overflow_payload["model"]["parameters"]["H"][0][0] = 1e308
    calibration = load_pixel_to_tcp_calibration(
        _write_yaml(tmp_path / "溢出标定.yaml", overflow_payload),
        expected_subject="block",
    )

    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="标定预测结果包含非有限数值"):
            calibration.predict([2.0, 3.0])


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
    calibration = load_pixel_to_tcp_calibration(
        _write_yaml(tmp_path / "零分母.yaml", zero_denominator),
        expected_subject="block",
    )
    with pytest.raises(ValueError, match="齐次分母"):
        calibration.predict([2.0, 3.0])
