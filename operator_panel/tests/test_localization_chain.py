"""定位 Z 计算链路纯函数与只读端点的数值正确性测试。"""

from pathlib import Path

import pytest
import yaml

from operator_panel_lib.localization_chain import build_localization_z_chain


def _write_yaml(path: Path, data):
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_configs(
    tmp_path,
    *,
    fixed_enabled=False,
    block_z=173.46,
    tray_z=182.46,
    with_calibrations=True,
    z_min=155.0,
):
    perception = {
        "pick_height": {"block_observation_height_mm": 167.0},
        "high_tcp_localization": {
            "safe_x_range_mm": [-300.0, -200.0],
            "safe_y_range_mm": [-100.0, 100.0],
        },
    }
    if fixed_enabled:
        perception["high_tcp_localization"]["fixed_tcp_z"] = {
            "enabled": True,
            "block_observation_z_mm": block_z,
            "tray_z_mm": tray_z,
        }
    execution = {
        "motion": {
            "pick_surface_offset_mm": 151.0,
            "pick_approach_clearance_mm": 5.0,
            "place_descent_offset_mm": 5.0,
            "pick_rotate_safe_lift_mm": 13.0,
            "pick_retreat_blend_radius_mm": 5.0,
            "minimum_tcp_z_mm": z_min,
        }
    }
    perception_path = tmp_path / "perception.yaml"
    execution_path = tmp_path / "execution.yaml"
    _write_yaml(perception_path, perception)
    _write_yaml(execution_path, execution)

    block_path = tmp_path / "block.yaml"
    tray_path = tmp_path / "tray.yaml"
    if with_calibrations:
        for path, (a, b, c) in (
            (block_path, (0.0, 0.0, 173.0)),
            (tray_path, (0.0, 0.0, 182.0)),
        ):
            _write_yaml(path, {
                "schema_version": 2,
                "generation_id": "batch-test",
                "z_plane": {
                    "equation": "z = a*x + b*y + c",
                    "coefficients": [a, b, c],
                    "source": "external_depth",
                },
            })
    return perception_path, execution_path, block_path, tray_path


def _row_values(chain):
    return {row["stage"]: row["value"] for row in chain["rows"]}


def test_fixed_mode_computes_all_derived_heights(tmp_path):
    paths = _write_configs(tmp_path, fixed_enabled=True)

    chain = build_localization_z_chain(*paths[:2], paths[2], paths[3])

    assert chain["mode"] == "fixed_constant"
    values = _row_values(chain)
    assert values["方块观察位"] == 173.46
    assert values["方块表面 Z"] == 6.46
    assert values["最终抓取 Z（下探终点）"] == 157.46
    assert values["预抓取 Z（斜向进入）"] == 162.46
    assert values["抓后抬升 Z"] == 182.46
    assert values["旋转放行阈值"] == 170.46
    assert values["托盘释放 Z"] == 182.46
    assert values["摆放下探 Z"] == 177.46
    assert all(check["ok"] for check in chain["checks"])
    # 固定模式下标定平面只是对照，标记为已被绕过。
    assert chain["z_plane"]["block"]["bypassed"] is True


def test_plane_mode_computes_cornerwise_ranges(tmp_path):
    perception_path, execution_path, block_path, tray_path = _write_configs(
        tmp_path, fixed_enabled=False
    )
    # 换成带斜率的平面，验证角点区间与同角点取小。
    for path, c in ((block_path, 173.0), (tray_path, 182.0)):
        _write_yaml(path, {
            "schema_version": 2,
            "generation_id": "batch-tilt",
            "z_plane": {
                "equation": "z = a*x + b*y + c",
                "coefficients": [0.01, -0.02, c],
                "source": "external_depth",
            },
        })

    chain = build_localization_z_chain(perception_path, execution_path, block_path, tray_path)

    assert chain["mode"] == "calibration_z_plane"
    values = _row_values(chain)
    # 角点：x∈[-300,-200] 0.01x∈[-3,-2]；y∈[-100,100] -0.02y∈[-2,2]。
    assert values["方块观察位"] == {"min": 168.0, "max": 173.0}
    assert values["最终抓取 Z（下探终点）"] == {"min": 152.0, "max": 157.0}
    assert values["托盘释放 Z"] == {"min": 177.0, "max": 182.0}
    # 旋转阈值必须同角点取小：最坏角点 (-300,100) 为 min(152+13, 173)=165。
    assert values["旋转放行阈值"] == {"min": 165.0, "max": 170.0}
    assert chain["z_plane"]["block"]["bypassed"] is False


def test_fixed_mode_below_limit_reports_failing_check(tmp_path):
    paths = _write_configs(tmp_path, fixed_enabled=True, block_z=170.0)

    chain = build_localization_z_chain(*paths[:2], paths[2], paths[3])

    pick_check = next(c for c in chain["checks"] if "抓取" in c["name"])
    assert pick_check["ok"] is False
    assert "154.00" in pick_check["detail"]


def test_plane_mode_without_calibration_files_degrades_gracefully(tmp_path):
    perception_path, execution_path, _, _ = _write_configs(
        tmp_path, fixed_enabled=False, with_calibrations=False
    )

    chain = build_localization_z_chain(
        perception_path, execution_path,
        tmp_path / "missing-block.yaml", tmp_path / "missing-tray.yaml",
    )

    assert chain["mode"] == "calibration_z_plane"
    assert chain["rows"][0]["value"] is None
    assert any(not check["ok"] for check in chain["checks"])
    assert chain["z_plane"] == {"block": None, "tray": None}


def test_fixed_enabled_but_missing_constants_raises(tmp_path):
    perception_path, execution_path, _, _ = _write_configs(tmp_path, fixed_enabled=False)
    perception = yaml.safe_load(perception_path.read_text(encoding="utf-8"))
    perception["high_tcp_localization"]["fixed_tcp_z"] = {"enabled": True}
    _write_yaml(perception_path, perception)

    with pytest.raises(ValueError, match="缺少 block_observation_z_mm 或 tray_z_mm"):
        build_localization_z_chain(perception_path, execution_path)


def test_missing_motion_key_raises_clear_error(tmp_path):
    perception_path, execution_path, _, _ = _write_configs(tmp_path)
    execution = yaml.safe_load(execution_path.read_text(encoding="utf-8"))
    del execution["motion"]["pick_surface_offset_mm"]
    _write_yaml(execution_path, execution)

    with pytest.raises(ValueError, match="缺少配置项.*pick_surface_offset_mm"):
        build_localization_z_chain(perception_path, execution_path)
