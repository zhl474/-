"""高位像素到 TCP 粗定位器的主体、覆盖范围和机械安全边界测试。"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from image_process_lib.high_pixel_to_tcp_localizer import HighPixelToTcpLocalizer


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _affine_payload(subject, base_xyz):
    """构造像素 [u, v] 到 [base_x+u, base_y+v, base_z] 的简单标定。"""
    return {
        "schema_version": 1,
        "calibration_type": "pixel_to_tcp_position",
        "calibration_subject": subject,
        "input": {"coordinate": "high_detection_pixel_xy", "unit": "pixel"},
        "output": {"coordinate": "tcp_position_xyz", "unit": "mm"},
        "model": {
            "name": "affine",
            "parameters": {
                "kind": "polynomial",
                "degree": 1,
                "uv_mean": [0.0, 0.0],
                "uv_scale": [1.0, 1.0],
                "feature_names": ["1", "u", "v"],
                "coef": [
                    [float(base_xyz[0]), float(base_xyz[1]), float(base_xyz[2])],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
            },
        },
        "coverage": {
            "pixel_convex_hull": [
                [0.0, 0.0],
                [100.0, 0.0],
                [100.0, 100.0],
                [0.0, 100.0],
            ],
        },
    }


def _v2_affine_payload(subject, generation_id, z_c):
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
                "coef": [[-300.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            },
        },
        "z_plane": {
            "equation": "z = a*x + b*y + c",
            "coefficients": [0.01, 0.02, z_c],
            "source": "external_depth",
        },
        "coverage": {
            "pixel_convex_hull": [
                [0.0, 0.0],
                [100.0, 0.0],
                [100.0, 100.0],
                [0.0, 100.0],
            ],
        },
    }


def _write_calibrations(tmp_path):
    block_path = tmp_path / "方块标定.yaml"
    tray_path = tmp_path / "托盘标定.yaml"
    block_path.write_text(
        yaml.safe_dump(_affine_payload("block", [-300.0, 0.0, 200.0]), sort_keys=False),
        encoding="utf-8",
    )
    tray_path.write_text(
        yaml.safe_dump(_affine_payload("tray", [-250.0, 50.0, 210.0]), sort_keys=False),
        encoding="utf-8",
    )
    return block_path, tray_path


def test_localizer_distinguishes_subjects_and_uses_shooting_rpy(tmp_path):
    block_path, tray_path = _write_calibrations(tmp_path)
    localizer = HighPixelToTcpLocalizer(
        block_path,
        tray_path,
        shooting_pose=[-300.0, 0.0, 500.0, 179.0, 1.0, -90.0],
    )

    assert localizer.locate_block([10.0, 20.0]) == [-290.0, 20.0, 200.0, 179.0, 1.0, -90.0]
    assert localizer.locate_tray([10.0, 20.0]) == [-240.0, 70.0, 210.0, 179.0, 1.0, -90.0]


def test_localizer_allows_sample_hull_outside_pixel_but_rejects_unknown_subject(tmp_path):
    block_path, tray_path = _write_calibrations(tmp_path)
    localizer = HighPixelToTcpLocalizer(
        block_path,
        tray_path,
        shooting_pose=[-300.0, 0.0, 500.0, 180.0, 0.0, -90.0],
    )

    # (101, 50) 位于 0..100 的样本凸包外，但预测 TCP 仍在机械安全范围内。
    assert localizer.locate_block([101.0, 50.0]) == [
        -199.0,
        50.0,
        200.0,
        180.0,
        0.0,
        -90.0,
    ]
    with pytest.raises(ValueError, match=r"board.*\[10\.0, 20\.0\]"):
        localizer.locate("board", [10.0, 20.0])


def test_localizer_rejects_unsafe_xyz_and_accepts_injected_bounds(tmp_path):
    block_path, tray_path = _write_calibrations(tmp_path)
    restricted = HighPixelToTcpLocalizer(
        block_path,
        tray_path,
        shooting_pose=[-300.0, 0.0, 500.0, 180.0, 0.0, -90.0],
        tcp_min_xyz=[-444.224, -263.279, 205.0],
        tcp_max_xyz=[-148.17, 315.925, None],
    )
    with pytest.raises(ValueError, match=r"方块.*\[10\.0, 20\.0\].*超出安全范围"):
        restricted.locate_block([10.0, 20.0])

    injected = HighPixelToTcpLocalizer(
        block_path,
        tray_path,
        shooting_pose=[-300.0, 0.0, 500.0, 180.0, 0.0, -90.0],
        tcp_min_xyz=[-310.0, -10.0, 190.0],
        tcp_max_xyz=[-200.0, 100.0, 220.0],
    )
    assert injected.locate_block([10.0, 20.0])[:3] == [-290.0, 20.0, 200.0]


@pytest.mark.parametrize(
    "base_xyz",
    [
        [-460.0, 0.0, 200.0],
        [-100.0, 0.0, 200.0],
        [-300.0, -300.0, 200.0],
        [-300.0, 300.0, 200.0],
        [-300.0, 0.0, 160.0],
    ],
)
def test_localizer_rejects_each_configured_xyz_safety_boundary(tmp_path, base_xyz):
    block_path = tmp_path / "越界方块标定.yaml"
    tray_path = tmp_path / "安全托盘标定.yaml"
    block_path.write_text(
        yaml.safe_dump(_affine_payload("block", base_xyz), sort_keys=False),
        encoding="utf-8",
    )
    tray_path.write_text(
        yaml.safe_dump(_affine_payload("tray", [-250.0, 50.0, 210.0]), sort_keys=False),
        encoding="utf-8",
    )
    localizer = HighPixelToTcpLocalizer(
        block_path,
        tray_path,
        shooting_pose=[-300.0, 0.0, 500.0, 180.0, 0.0, -90.0],
    )

    with pytest.raises(ValueError, match="超出安全范围"):
        localizer.locate_block([10.0, 20.0])


def test_localizer_intentionally_has_no_tcp_z_upper_limit(tmp_path):
    block_path = tmp_path / "高Z方块标定.yaml"
    tray_path = tmp_path / "安全托盘标定.yaml"
    block_path.write_text(
        yaml.safe_dump(
            _affine_payload("block", [-300.0, 0.0, 1_000_000.0]),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    tray_path.write_text(
        yaml.safe_dump(_affine_payload("tray", [-250.0, 50.0, 210.0]), sort_keys=False),
        encoding="utf-8",
    )
    localizer = HighPixelToTcpLocalizer(
        block_path,
        tray_path,
        shooting_pose=[-300.0, 0.0, 500.0, 180.0, 0.0, -90.0],
    )

    assert localizer.locate_block([10.0, 20.0])[2] == 1_000_000.0


@pytest.mark.parametrize(
    "boundary_xyz",
    [
        [-444.224, -263.279, 165.0],
        [-148.17, 315.925, 165.0],
    ],
)
def test_localizer_xyz_safety_boundaries_are_inclusive(tmp_path, boundary_xyz):
    block_path = tmp_path / "边界方块标定.yaml"
    tray_path = tmp_path / "安全托盘标定.yaml"
    block_path.write_text(
        yaml.safe_dump(_affine_payload("block", boundary_xyz), sort_keys=False),
        encoding="utf-8",
    )
    tray_path.write_text(
        yaml.safe_dump(_affine_payload("tray", [-250.0, 50.0, 210.0]), sort_keys=False),
        encoding="utf-8",
    )
    localizer = HighPixelToTcpLocalizer(
        block_path,
        tray_path,
        shooting_pose=[-300.0, 0.0, 500.0, 180.0, 0.0, -90.0],
    )

    assert localizer.locate_block([0.0, 0.0])[:3] == boundary_xyz


def test_open_loop_safety_offset_checks_executed_tcp_but_returns_raw_prediction(tmp_path):
    block_path = tmp_path / "开环方块标定.yaml"
    tray_path = tmp_path / "开环托盘标定.yaml"
    block_path.write_text(
        yaml.safe_dump(_affine_payload("block", [-140.0, 20.0, 200.0]), sort_keys=False),
        encoding="utf-8",
    )
    tray_path.write_text(
        yaml.safe_dump(_affine_payload("tray", [-140.0, 20.0, 200.0]), sort_keys=False),
        encoding="utf-8",
    )

    closed_loop = HighPixelToTcpLocalizer(
        block_path,
        tray_path,
        shooting_pose=[-300.0, 0.0, 500.0, 180.0, 0.0, -90.0],
    )
    with pytest.raises(ValueError, match="预测 TCP XYZ.*超出安全范围"):
        closed_loop.locate_block([0.0, 0.0])

    open_loop = HighPixelToTcpLocalizer(
        block_path,
        tray_path,
        shooting_pose=[-300.0, 0.0, 500.0, 180.0, 0.0, -90.0],
        safety_xy_offset=[-94.1, -13.8],
    )

    # 校验使用偏移后的 [-234.1, 6.2]，返回值仍是标定模型的原始预测。
    assert open_loop.locate_block([0.0, 0.0])[:3] == pytest.approx(
        [-140.0, 20.0, 200.0]
    )
    assert open_loop.locate_tray([0.0, 0.0])[:3] == pytest.approx(
        [-140.0, 20.0, 200.0]
    )


def test_open_loop_safety_offset_reports_raw_offset_and_executed_tcp(tmp_path):
    block_path = tmp_path / "偏移越界方块标定.yaml"
    tray_path = tmp_path / "安全托盘标定.yaml"
    block_path.write_text(
        yaml.safe_dump(_affine_payload("block", [-440.0, -250.0, 200.0]), sort_keys=False),
        encoding="utf-8",
    )
    tray_path.write_text(
        yaml.safe_dump(_affine_payload("tray", [-300.0, 0.0, 200.0]), sort_keys=False),
        encoding="utf-8",
    )
    localizer = HighPixelToTcpLocalizer(
        block_path,
        tray_path,
        shooting_pose=[-300.0, 0.0, 500.0, 180.0, 0.0, -90.0],
        safety_xy_offset=[-94.1, -13.8],
    )

    with pytest.raises(ValueError) as error:
        localizer.locate_block([0.0, 0.0])

    message = str(error.value)
    assert "预测 TCP XYZ [-440.0, -250.0, 200.0]" in message
    assert "安全校验 XY 偏移 [-94.1, -13.8]" in message
    assert "待执行 TCP XYZ [-534.1, -263.8, 200.0]" in message
    assert "超出安全范围" in message


def test_structured_assessment_reports_actual_tcp_and_all_violated_axes(tmp_path):
    block_path = tmp_path / "结构化方块标定.yaml"
    tray_path = tmp_path / "结构化托盘标定.yaml"
    block_path.write_text(
        yaml.safe_dump(_affine_payload("block", [-440.0, -260.0, 160.0]), sort_keys=False),
        encoding="utf-8",
    )
    tray_path.write_text(
        yaml.safe_dump(_affine_payload("tray", [-300.0, 0.0, 200.0]), sort_keys=False),
        encoding="utf-8",
    )
    localizer = HighPixelToTcpLocalizer(
        block_path,
        tray_path,
        shooting_pose=[-300.0, 0.0, 500.0, 180.0, 0.0, -90.0],
        safety_xy_offset=[-94.1, -13.8],
    )

    assessment = localizer.assess("block", [0.0, 0.0])

    assert assessment.predicted_tcp_xyz == (-440.0, -260.0, 160.0)
    assert assessment.safety_tcp_xyz == pytest.approx((-534.1, -273.8, 160.0))
    assert assessment.safety_min_xyz == (-444.224, -263.279, 163.0)
    assert assessment.safety_max_xyz[:2] == (-148.17, 315.925)
    assert np.isposinf(assessment.safety_max_xyz[2])
    assert assessment.violated_axes == ("X", "Y", "Z")
    assert assessment.safe is False


def test_structured_assessment_closed_loop_uses_raw_prediction(tmp_path):
    block_path, tray_path = _write_calibrations(tmp_path)
    localizer = HighPixelToTcpLocalizer(
        block_path,
        tray_path,
        shooting_pose=[-300.0, 0.0, 500.0, 180.0, 0.0, -90.0],
    )

    assessment = localizer.assess("block", [10.0, 20.0])

    assert assessment.safe is True
    assert assessment.predicted_tcp_xyz == assessment.safety_tcp_xyz


def test_safety_xy_offset_never_changes_z_validation(tmp_path):
    block_path = tmp_path / "低位方块标定.yaml"
    tray_path = tmp_path / "安全托盘标定.yaml"
    block_path.write_text(
        yaml.safe_dump(_affine_payload("block", [-300.0, 0.0, 160.0]), sort_keys=False),
        encoding="utf-8",
    )
    tray_path.write_text(
        yaml.safe_dump(_affine_payload("tray", [-300.0, 0.0, 200.0]), sort_keys=False),
        encoding="utf-8",
    )
    localizer = HighPixelToTcpLocalizer(
        block_path,
        tray_path,
        shooting_pose=[-300.0, 0.0, 500.0, 180.0, 0.0, -90.0],
        safety_xy_offset=[-94.1, -13.8],
    )

    with pytest.raises(ValueError, match="待执行 TCP XYZ.*160.0.*超出安全范围"):
        localizer.locate_block([0.0, 0.0])


@pytest.mark.parametrize("offset", [[1.0], [1.0, 2.0, 3.0], [float("nan"), 0.0]])
def test_constructor_rejects_invalid_safety_xy_offset(tmp_path, offset):
    block_path, tray_path = _write_calibrations(tmp_path)

    with pytest.raises(ValueError, match="safety_xy_offset 必须包含 2 个有限数值"):
        HighPixelToTcpLocalizer(
            block_path,
            tray_path,
            shooting_pose=[-300.0, 0.0, 500.0, 180.0, 0.0, -90.0],
            safety_xy_offset=offset,
        )


def test_tray_strict_planarity_false_metadata_does_not_block_runtime(tmp_path):
    block_payload = _affine_payload("block", [-300.0, 0.0, 200.0])
    tray_payload = _affine_payload("tray", [-250.0, 50.0, 210.0])
    tray_payload["diagnostics"] = {"strict_planarity_passed": False}
    block_path = tmp_path / "方块标定.yaml"
    tray_path = tmp_path / "托盘标定.yaml"
    block_path.write_text(
        yaml.safe_dump(block_payload, sort_keys=False),
        encoding="utf-8",
    )
    tray_path.write_text(
        yaml.safe_dump(tray_payload, sort_keys=False),
        encoding="utf-8",
    )
    localizer = HighPixelToTcpLocalizer(
        block_path,
        tray_path,
        shooting_pose=[-300.0, 0.0, 500.0, 180.0, 0.0, -90.0],
    )

    assert localizer.locate_tray([10.0, 20.0])[:3] == [-240.0, 70.0, 210.0]


def test_deployed_calibrations_load_with_strict_subjects_and_safe_predictions():
    block_path = PACKAGE_DIR / "config" / "block_pixel_to_tcp_calibration.yaml"
    tray_path = PACKAGE_DIR / "config" / "tray_pixel_to_tcp_calibration.yaml"
    localizer = HighPixelToTcpLocalizer(
        block_path,
        tray_path,
        shooting_pose=[-300.0, 0.0, 500.0, 180.0, 0.0, -90.0],
    )

    block_pose = localizer.locate_block([650.0, 300.0])
    tray_pose = localizer.locate_tray([700.0, 300.0])
    assert len(block_pose) == len(tray_pose) == 6
    assert np.all(np.isfinite(block_pose + tray_pose))
    assert block_pose[3:] == tray_pose[3:] == [180.0, 0.0, -90.0]


def test_reported_block_pixel_outside_sample_hull_is_accepted_when_tcp_is_safe():
    block_path = PACKAGE_DIR / "config" / "block_pixel_to_tcp_calibration.yaml"
    tray_path = PACKAGE_DIR / "config" / "tray_pixel_to_tcp_calibration.yaml"
    localizer = HighPixelToTcpLocalizer(
        block_path,
        tray_path,
        shooting_pose=[-300.0, 0.0, 500.0, 180.0, 0.0, -90.0],
    )

    pose = localizer.locate_block([1063.0, 571.0])

    assert pose == pytest.approx(
        [-152.0510504722284, 216.944753678346, 203.99675126128128, 180.0, 0.0, -90.0],
        abs=1e-6,
    )


def test_constructor_rejects_swapped_subject_calibrations(tmp_path):
    block_path, tray_path = _write_calibrations(tmp_path)
    with pytest.raises(ValueError, match="方块.*期望 block.*文件为 tray"):
        HighPixelToTcpLocalizer(
            tray_path,
            block_path,
            shooting_pose=[-300.0, 0.0, 500.0, 180.0, 0.0, -90.0],
        )


def test_v2_pair_keeps_same_slopes_and_exact_seven_mm_height_difference(tmp_path):
    block_path = tmp_path / "v2方块.yaml"
    tray_path = tmp_path / "v2托盘.yaml"
    block_path.write_text(
        yaml.safe_dump(_v2_affine_payload("block", "batch-1", 203.0), sort_keys=False),
        encoding="utf-8",
    )
    tray_path.write_text(
        yaml.safe_dump(_v2_affine_payload("tray", "batch-1", 196.0), sort_keys=False),
        encoding="utf-8",
    )
    localizer = HighPixelToTcpLocalizer(
        block_path,
        tray_path,
        shooting_pose=[-250.0, 0.0, 380.0, 180.0, 0.0, 90.0],
    )

    block = localizer.locate_block([10.0, 20.0])
    tray = localizer.locate_tray([10.0, 20.0])

    assert block[:2] == tray[:2] == [-290.0, 20.0]
    assert block[2] - tray[2] == pytest.approx(7.0)


def test_localizer_rejects_v1_v2_mix_and_v2_batch_mismatch(tmp_path):
    v1_block = tmp_path / "v1方块.yaml"
    v2_block = tmp_path / "v2方块.yaml"
    tray = tmp_path / "v2托盘.yaml"
    v1_block.write_text(
        yaml.safe_dump(_affine_payload("block", [-300.0, 0.0, 200.0]), sort_keys=False),
        encoding="utf-8",
    )
    v2_block.write_text(
        yaml.safe_dump(_v2_affine_payload("block", "batch-a", 203.0), sort_keys=False),
        encoding="utf-8",
    )
    tray.write_text(
        yaml.safe_dump(_v2_affine_payload("tray", "batch-b", 196.0), sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema 版本不一致"):
        HighPixelToTcpLocalizer(
            v1_block,
            tray,
            shooting_pose=[-250.0, 0.0, 380.0, 180.0, 0.0, 90.0],
        )
    with pytest.raises(ValueError, match="批次不一致"):
        HighPixelToTcpLocalizer(
            v2_block,
            tray,
            shooting_pose=[-250.0, 0.0, 380.0, 180.0, 0.0, 90.0],
        )


def test_fixed_tcp_z_overrides_z_but_keeps_xy_prediction(tmp_path):
    block_path, tray_path = _write_calibrations(tmp_path)
    localizer = HighPixelToTcpLocalizer(
        block_path,
        tray_path,
        shooting_pose=[-300.0, 0.0, 500.0, 180.0, 0.0, -90.0],
        fixed_tcp_z_mm={"block": 205.0, "tray": 214.0},
    )

    block_pose = localizer.locate_block([10.0, 20.0])
    tray_pose = localizer.locate_tray([30.0, 40.0])

    # XY 仍按像素变化，Z 恒为固定常数。
    assert block_pose[:2] == [-290.0, 20.0]
    assert tray_pose[:2] == [-220.0, 90.0]
    assert block_pose[2] == pytest.approx(205.0)
    assert tray_pose[2] == pytest.approx(214.0)
    assert localizer.fixed_tcp_z_mm == {"block": 205.0, "tray": 214.0}


def test_fixed_tcp_z_on_v2_ignores_tilted_plane(tmp_path):
    block_path = tmp_path / "斜面方块.yaml"
    tray_path = tmp_path / "斜面托盘.yaml"
    block_path.write_text(
        yaml.safe_dump(_v2_affine_payload("block", "batch-1", 203.0), sort_keys=False),
        encoding="utf-8",
    )
    tray_path.write_text(
        yaml.safe_dump(_v2_affine_payload("tray", "batch-1", 196.0), sort_keys=False),
        encoding="utf-8",
    )
    localizer = HighPixelToTcpLocalizer(
        block_path,
        tray_path,
        shooting_pose=[-250.0, 0.0, 380.0, 180.0, 0.0, 90.0],
        fixed_tcp_z_mm={"block": 200.0, "tray": 207.0},
    )

    # v2 标定的 z_plane 带 a/b 斜率，固定模式下不同像素的 Z 也必须完全相同。
    assert localizer.locate_block([10.0, 20.0])[2] == pytest.approx(200.0)
    assert localizer.locate_block([90.0, 80.0])[2] == pytest.approx(200.0)
    assert localizer.locate_tray([10.0, 20.0])[2] == pytest.approx(207.0)


def test_fixed_tcp_z_assessment_and_summary_report_constant_z(tmp_path):
    block_path, tray_path = _write_calibrations(tmp_path)
    localizer = HighPixelToTcpLocalizer(
        block_path,
        tray_path,
        shooting_pose=[-300.0, 0.0, 500.0, 180.0, 0.0, -90.0],
        fixed_tcp_z_mm={"block": 205.0, "tray": 214.0},
    )

    assessment = localizer.assess("block", [10.0, 20.0])

    assert assessment.predicted_tcp_xyz == (-290.0, 20.0, 205.0)
    assert assessment.safety_tcp_xyz == (-290.0, 20.0, 205.0)
    assert assessment.violated_axes == ()
    summary = localizer.calibration_summary("block")
    assert summary["TCP_Z来源"] == "fixed_constant=205.000"


def test_default_mode_keeps_calibration_z_plane_and_reports_source(tmp_path):
    block_path, tray_path = _write_calibrations(tmp_path)
    localizer = HighPixelToTcpLocalizer(
        block_path,
        tray_path,
        shooting_pose=[-300.0, 0.0, 500.0, 180.0, 0.0, -90.0],
    )

    assert localizer.fixed_tcp_z_mm is None
    assert localizer.locate_block([10.0, 20.0])[2] == pytest.approx(200.0)
    assert localizer.calibration_summary("block")["TCP_Z来源"] == "calibration_z_plane"


@pytest.mark.parametrize(
    "fixed, message",
    [
        ({"block": 205.0}, r"固定 TCP Z 必须同时提供.*托盘"),
        ({"tray": 214.0}, r"固定 TCP Z 必须同时提供.*方块"),
        ({"block": float("nan"), "tray": 214.0}, r"方块固定 TCP Z 必须是有限数值"),
        ({"block": True, "tray": 214.0}, r"方块固定 TCP Z 必须是有限数值"),
        ({"block": 150.0, "tray": 214.0}, r"方块固定 TCP Z=150\.000 mm 低于安全下限 190\.000"),
        ([205.0, 214.0], r"fixed_tcp_z_mm 必须是含 block 和 tray 键的字典"),
    ],
)
def test_constructor_rejects_incomplete_or_unsafe_fixed_tcp_z(tmp_path, fixed, message):
    block_path, tray_path = _write_calibrations(tmp_path)

    with pytest.raises(ValueError, match=message):
        HighPixelToTcpLocalizer(
            block_path,
            tray_path,
            shooting_pose=[-300.0, 0.0, 500.0, 180.0, 0.0, -90.0],
            tcp_min_xyz=[-444.224, -263.279, 190.0],
            fixed_tcp_z_mm=fixed,
        )
