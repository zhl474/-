"""高位定位链路分流 Router：模式路由、shadow 对比、direct 缺失回退与接口转发测试。"""

import sys
from pathlib import Path

import pytest
import yaml

SRC_DIR = Path(__file__).resolve().parents[2]
IMAGE_PROCESS_DIR = SRC_DIR / "image_process"
if str(IMAGE_PROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(IMAGE_PROCESS_DIR))

from image_process_lib.high_pixel_to_tcp_localizer import (  # noqa: E402
    HighPixelToTcpLocalizer,
)
from image_process_lib.high_tcp_localization_router import (  # noqa: E402
    _TRAY_SHADOW_FLUSH_COUNT,
    HighTcpLocalizationRouter,
)

_SHOOTING_POSE = [-300.0, 0.0, 500.0, 180.0, 0.0, -90.0]
_GENERATION = "router-test-batch"


def _v2_affine_payload(subject, base_xy, z_c, generation_id=_GENERATION):
    """affine 模型：predict([u, v]) = [base_x+u, base_y+v, z_c]。"""
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
                "coef": [[float(base_xy[0]), float(base_xy[1])], [1.0, 0.0], [0.0, 1.0]],
            },
        },
        "z_plane": {
            "equation": "z = a*x + b*y + c",
            "coefficients": [0.0, 0.0, z_c],
            "source": "router_test",
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


def _write_payload(tmp_path, filename, payload):
    path = tmp_path / filename
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _build_localizers(tmp_path, with_direct=True):
    servo = HighPixelToTcpLocalizer(
        _write_payload(tmp_path, "servo_block.yaml", _v2_affine_payload("block", (-300.0, 0.0), 200.0)),
        _write_payload(tmp_path, "servo_tray.yaml", _v2_affine_payload("tray", (-250.0, 50.0), 210.0)),
        shooting_pose=_SHOOTING_POSE,
    )
    direct = None
    if with_direct:
        direct = HighPixelToTcpLocalizer(
            _write_payload(tmp_path, "direct_center.yaml", _v2_affine_payload("block", (-400.0, 0.0), 200.0)),
            _write_payload(tmp_path, "direct_tray.yaml", _v2_affine_payload("tray", (-350.0, 60.0), 210.0)),
            shooting_pose=_SHOOTING_POSE,
            block_calibration_left_path=_write_payload(
                tmp_path, "direct_left.yaml", _v2_affine_payload("block", (-410.0, -5.0), 200.0)
            ),
            block_calibration_right_path=_write_payload(
                tmp_path, "direct_right.yaml", _v2_affine_payload("block", (-420.0, 10.0), 200.0)
            ),
            split_models=True,
        )
    return servo, direct


class _Recorder:
    """收集 warn 输出并按需抛错，替代 rospy.logwarn。"""

    def __init__(self):
        self.messages = []

    def __call__(self, message, *args):
        self.messages.append(message % args if args else message)


def _router(tmp_path, mode, with_direct=True):
    """按常量模式构造 Router，附带 warn 记录器。"""
    servo, direct = _build_localizers(tmp_path, with_direct=with_direct)
    recorder = _Recorder()
    router = HighTcpLocalizationRouter(
        servo,
        direct,
        lambda: mode,
        warn=recorder,
    )
    return router, recorder


def test_servo_mode_keeps_current_behavior(tmp_path):
    router, recorder = _router(tmp_path, "servo")
    assert router.locate_block([100.0, 220.0])[:3] == pytest.approx([-200.0, 220.0, 200.0])
    assert router.locate_tray([700.0, 220.0])[:3] == pytest.approx([450.0, 270.0, 210.0])
    assert recorder.messages == []


def test_direct_mode_routes_by_pixel_side(tmp_path):
    router, _ = _router(tmp_path, "direct")
    # 左像素走 direct 左模型（基准 -410, -5）。
    assert router.locate_block([100.0, 220.0])[:3] == pytest.approx([-310.0, 215.0, 200.0])
    # 右像素走 direct 右模型（基准 -420, 10）。
    assert router.locate_block([900.0, 220.0])[:3] == pytest.approx([480.0, 230.0, 200.0])
    # (0,0) 缺失标记回退 direct 中心模型（基准 -400, 0）。
    assert router.locate_block([0.0, 0.0])[:3] == pytest.approx([-400.0, 0.0, 200.0])
    # 托盘恒用 direct 托盘模型（基准 -350, 60）。
    assert router.locate_tray([900.0, 220.0])[:3] == pytest.approx([550.0, 280.0, 210.0])


def test_unknown_mode_falls_back_to_servo_with_warning(tmp_path):
    router, recorder = _router(tmp_path, "nonsense")
    assert router.locate_block([100.0, 220.0])[:3] == pytest.approx([-200.0, 220.0, 200.0])
    assert any("未知高位定位链路模式" in message for message in recorder.messages)


def test_mode_provider_exception_falls_back_to_servo(tmp_path):
    servo, direct = _build_localizers(tmp_path)
    recorder = _Recorder()

    def broken_provider():
        raise RuntimeError("rosparam 不可用")

    router = HighTcpLocalizationRouter(servo, direct, broken_provider, warn=recorder)
    assert router.locate_block([100.0, 220.0])[:3] == pytest.approx([-200.0, 220.0, 200.0])
    assert any("读取高位定位链路模式失败" in message for message in recorder.messages)


def test_shadow_returns_servo_result_and_logs_delta(tmp_path):
    router, recorder = _router(tmp_path, "shadow")
    pose = router.locate_block([100.0, 220.0])
    # shadow 正式输出仍是 servo 结果。
    assert pose[:3] == pytest.approx([-200.0, 220.0, 200.0])
    assert any(
        "九点标定shadow" in message and "方块" in message for message in recorder.messages
    )
    delta_message = next(m for m in recorder.messages if "ΔXY" in m)
    # servo (-200, 220) vs direct 左 (-310, 215)：ΔXY=hypot(110, 5)=110.11。
    assert "ΔXY=110.11mm" in delta_message


def test_shadow_tray_accumulates_and_flushes_summary(tmp_path):
    router, recorder = _router(tmp_path, "shadow")
    for index in range(_TRAY_SHADOW_FLUSH_COUNT - 1):
        router.locate_tray([500.0 + index, 200.0])
    assert not any("托盘" in message for message in recorder.messages)
    # 第 140 个托盘点触发冲刷汇总。
    router.locate_tray([500.0, 200.0])
    assert any(
        f"托盘 {_TRAY_SHADOW_FLUSH_COUNT} 点" in message for message in recorder.messages
    )


def test_direct_missing_instance_warns_once_and_uses_servo(tmp_path):
    router, recorder = _router(tmp_path, "direct", with_direct=False)
    assert router.locate_block([100.0, 220.0])[:3] == pytest.approx([-200.0, 220.0, 200.0])
    router.locate_block([150.0, 220.0])
    warnings = [m for m in recorder.messages if "direct 九点标定实例未构建" in m]
    assert len(warnings) == 1, "缺失告警只应出现一次"


def test_assess_routes_by_mode_and_shadow_compares(tmp_path):
    router, recorder = _router(tmp_path, "direct")
    assessment = router.assess("block", [100.0, 220.0])
    assert assessment.predicted_tcp_xyz == pytest.approx((-310.0, 215.0, 200.0))
    assert assessment.safe is True

    shadow_router, _ = _router(tmp_path, "shadow")
    shadow_assessment = shadow_router.assess("block", [100.0, 220.0])
    assert shadow_assessment.predicted_tcp_xyz == pytest.approx((-200.0, 220.0, 200.0))


def test_update_dynamic_bounds_forwards_to_both_branches(tmp_path):
    router, _ = _router(tmp_path, "servo")
    router.update_dynamic_bounds(shooting_pose=[-300.0, 0.0, 500.0, 179.0, 1.0, -88.0])
    assert router.locate_block([100.0, 220.0])[3:] == [179.0, 1.0, -88.0]
    # direct 分支同样收到动态位姿更新。
    direct_router, _ = _router(tmp_path, "direct")
    direct_router.update_dynamic_bounds(shooting_pose=[-300.0, 0.0, 500.0, 179.0, 1.0, -88.0])
    assert direct_router.locate_block([100.0, 220.0])[3:] == [179.0, 1.0, -88.0]


def test_fixed_tcp_z_and_summary_forwarding(tmp_path):
    servo, direct = _build_localizers(tmp_path)
    recorder = _Recorder()

    router = HighTcpLocalizationRouter(servo, direct, lambda: "servo", warn=recorder)
    assert router.fixed_tcp_z_mm == servo.fixed_tcp_z_mm

    summary = router.calibration_summary("block")
    assert summary["生效链路"] == "servo"
    assert "九点标定版" in summary

    direct_router = HighTcpLocalizationRouter(
        servo, direct, lambda: "direct", warn=recorder
    )
    direct_summary = direct_router.calibration_summary("block")
    assert direct_summary["生效链路"] == "direct"
    assert "伺服标定版" in direct_summary


def test_is_pixel_within_coverage_routes_by_mode(tmp_path):
    router, _ = _router(tmp_path, "servo")
    assert router.is_pixel_within_coverage("block", [50.0, 50.0]) is True
    assert router.is_pixel_within_coverage("block", [500.0, 500.0]) is False
