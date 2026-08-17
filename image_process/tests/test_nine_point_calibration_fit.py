"""九点标定工具：点对拟合、schema v2 产出与正式加载器回读的一致性测试。"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

SRC_DIR = Path(__file__).resolve().parents[2]
IMAGE_PROCESS_DIR = SRC_DIR / "image_process"
if str(IMAGE_PROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(IMAGE_PROCESS_DIR))
TOOLS_VISION_DIR = SRC_DIR / "tools" / "vision"
if str(TOOLS_VISION_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_VISION_DIR))

import nine_point_calibration as npc  # noqa: E402
from image_process_lib.pixel_to_tcp_calibration import (  # noqa: E402
    _polynomial_feature_names,
    load_pixel_to_tcp_calibration,
)


def _quadratic_pairs(pixels, seed=0):
    """从已知二次多项式生成点对：x=f(u,v)、y=g(u,v)、z=a*x+b*y+c。"""
    rng = np.random.default_rng(seed)
    x_coef = np.array([-300.0, 0.32, -0.05, 1.2e-5, -3.0e-6, 4.0e-5])
    y_coef = np.array([20.0, -0.04, 0.28, 2.0e-6, 5.0e-5, -1.0e-5])
    u = np.asarray([p[0] for p in pixels], dtype=float)
    v = np.asarray([p[1] for p in pixels], dtype=float)
    features = np.column_stack([np.ones_like(u), u, v, u * u, u * v, v * v])
    xy = np.column_stack([features @ x_coef, features @ y_coef])
    noise = rng.normal(0.0, 1e-9, size=xy.shape)
    xy = xy + noise
    z = 0.001 * xy[:, 0] + 0.002 * xy[:, 1] + 175.0
    return [
        (float(u_i), float(v_i), float(xy[i, 0]), float(xy[i, 1]), float(z[i]))
        for i, (u_i, v_i) in enumerate(pixels)
    ]


_LEFT_PIXELS = [(u, v) for u in (100.0, 300.0, 500.0) for v in (100.0, 300.0, 500.0)]
_RIGHT_PIXELS = [(u, v) for u in (700.0, 900.0, 1100.0) for v in (100.0, 300.0, 500.0)]
_TRAY_PIXELS = [(u, v) for u in (480.0, 640.0, 800.0) for v in (120.0, 360.0, 600.0)]


def test_feature_names_match_loader_poly2_order():
    assert npc.FEATURE_NAMES == _polynomial_feature_names(2)


def test_fit_and_payload_round_trip_through_official_loader(tmp_path):
    pairs = _quadratic_pairs(_LEFT_PIXELS)
    fit = npc.fit_model(pairs, "左半区方块", expected_side="left")
    payload = npc.build_payload("block", "direct-ninepoint-test", fit, "测试")
    path = tmp_path / "九点标定.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    calibration = load_pixel_to_tcp_calibration(path, expected_subject="block")

    for px, py, x, y, z in pairs:
        predicted = calibration.predict((px, py))
        assert predicted[0] == pytest.approx(x, abs=1e-4)
        assert predicted[1] == pytest.approx(y, abs=1e-4)
        assert predicted[2] == pytest.approx(z, abs=1e-4)


def test_fit_model_rejects_too_few_points():
    pairs = _quadratic_pairs([(100.0, 100.0), (300.0, 200.0), (500.0, 300.0)])
    with pytest.raises(ValueError, match="至少需要 6 个"):
        npc.fit_model(pairs, "左半区方块", expected_side="left")


def test_fit_model_rejects_cross_side_pixels():
    pairs = _quadratic_pairs([(100.0, 100.0), (700.0, 100.0)] + _LEFT_PIXELS[2:])
    with pytest.raises(ValueError, match="越过 u=640 分界线"):
        npc.fit_model(pairs, "左半区方块", expected_side="left")


def test_build_payload_rejects_collinear_pixels(tmp_path):
    # 像素共线：拟合本身可行，但凸包退化必须被 build_payload 拦下。
    collinear = [(float(100 + 40 * i), float(100 + 30 * i)) for i in range(9)]
    fit = npc.fit_model(_quadratic_pairs(collinear), "共线点")
    with pytest.raises(ValueError, match="凸包面积近似为零"):
        npc.build_payload("block", "direct-ninepoint-test", fit, "测试")


def test_main_writes_four_files_and_self_checks(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(npc, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(npc, "LEFT_PAIRS", _quadratic_pairs(_LEFT_PIXELS, seed=1))
    monkeypatch.setattr(npc, "RIGHT_PAIRS", _quadratic_pairs(_RIGHT_PIXELS, seed=2))
    monkeypatch.setattr(npc, "TRAY_PAIRS", _quadratic_pairs(_TRAY_PIXELS, seed=3))

    npc.main()

    written = sorted(path.name for path in tmp_path.glob("*.yaml"))
    assert written == [
        "block_pixel_to_tcp_calibration_direct.yaml",
        "block_pixel_to_tcp_calibration_direct_left.yaml",
        "block_pixel_to_tcp_calibration_direct_right.yaml",
        "tray_pixel_to_tcp_calibration_direct.yaml",
    ]
    output = capsys.readouterr().out
    assert "自检" in output and "完成" in output
    generations = set()
    for filename in written:
        document = yaml.safe_load((tmp_path / filename).read_text(encoding="utf-8"))
        generations.add(document["generation_id"])
    assert len(generations) == 1, "四份文件必须同批次 generation_id"

    block_left = load_pixel_to_tcp_calibration(
        tmp_path / "block_pixel_to_tcp_calibration_direct_left.yaml",
        expected_subject="block",
    )
    # 左模型对左半区像素回代应贴近原点对。
    for px, py, x, y, _z in npc.LEFT_PAIRS:
        predicted = block_left.predict((px, py))
        assert np.hypot(predicted[0] - x, predicted[1] - y) < 1e-3


def test_main_raises_when_all_pair_lists_empty(monkeypatch):
    monkeypatch.setattr(npc, "LEFT_PAIRS", [])
    monkeypatch.setattr(npc, "RIGHT_PAIRS", [])
    monkeypatch.setattr(npc, "TRAY_PAIRS", [])
    with pytest.raises(RuntimeError, match="点对列表为空"):
        npc.main()
