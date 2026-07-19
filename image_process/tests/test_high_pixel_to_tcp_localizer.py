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
        [-151.053132, 219.159510, 201.435111, 180.0, 0.0, -90.0],
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
