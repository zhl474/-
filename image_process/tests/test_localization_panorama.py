"""高位定位全景 JSON 组装函数的纯数据测试。"""

from types import SimpleNamespace

import numpy as np
import pytest

from image_process_lib.board_scene_detector import BOARD_COL_COUNT, BOARD_ROW_COUNT
from image_process_lib.depth_rough_localization import fit_z_plane
from image_process_lib.localization_panorama import (
    build_calibration_panorama_document,
    build_formal_panorama_document,
)
from image_process_lib.task_planner import ObservedBlock, PlacementTarget


def _synthetic_grid(poison_cell=None):
    """构造 1-based 的 15×11 嵌套格点表，可指定一个格子写入 NaN。"""
    grid = [
        [None for _ in range(BOARD_COL_COUNT + 1)]
        for _ in range(BOARD_ROW_COUNT + 1)
    ]
    for row in range(1, BOARD_ROW_COUNT + 1):
        for col in range(1, BOARD_COL_COUNT + 1):
            px, py = 100.0 + col * 20.0, 120.0 + row * 15.0
            if poison_cell == (row, col):
                px, py = float("nan"), float("nan")
            grid[row][col] = np.array([px, py], dtype=np.float32)
    return grid


class StubLocalizer:
    """与 HighPixelToTcpLocalizer 的鸭子类型接口保持一致。"""

    def __init__(self, poison_pixel=None):
        self.poison_pixel = poison_pixel

    def assess(self, subject, pixel_xy):
        if self.poison_pixel is not None and tuple(pixel_xy) == self.poison_pixel:
            raise ValueError("毒化像素触发预测失败")
        return SimpleNamespace(
            predicted_tcp_xyz=(-250.0 + pixel_xy[0] * 0.1, 20.0, 180.0),
            violated_axes=(),
        )

    def is_pixel_within_coverage(self, subject, pixel_xy, tolerance_px=0.0):
        return pixel_xy[0] < 400.0

    def calibration_summary(self, subject):
        return {
            "schema_version": 2,
            "实验批次": "测试批次",
            "XY模型": "poly2",
            "Z平面系数a_b_c": [-0.005, 0.001, 173.4],
            "凸包顶点数": 6,
            "标定样本数": 35,
        }


def _observed_block(category="I", pixel=(200.0, 300.0)):
    return ObservedBlock(
        category=category,
        observation_pose=(-230.0, 20.0, 173.0, -180.0, 0.0, 90.0),
        detected_angle_deg=15.0,
        source_id=0,
        pick_surface_z_mm=6.0,
        pick_surface_z_valid=True,
        high_detected_pixel_xy=pixel,
    )


def test_formal_panorama_covers_all_140_grid_points():
    document = build_formal_panorama_document(
        generated_at="2026-08-16T15:00:00",
        image_shape=(720, 1280),
        board_angle_deg=1.25,
        board_grid_points=_synthetic_grid(),
        observed_blocks=[_observed_block()],
        localizer=StubLocalizer(),
        pick_surface_offset_mm=151.0,
        block_calibration_sha256="aaa",
        tray_calibration_sha256="bbb",
    )

    assert document["模式"] == "正式"
    assert document["图像"] == {"宽": 1280, "高": 720}
    assert document["托盘旋转角deg"] == pytest.approx(1.25)

    grid_entries = document["托盘格点"]
    assert len(grid_entries) == BOARD_ROW_COUNT * BOARD_COL_COUNT
    assert {(entry["行"], entry["列"]) for entry in grid_entries} == {
        (row, col)
        for row in range(1, BOARD_ROW_COUNT + 1)
        for col in range(1, BOARD_COL_COUNT + 1)
    }
    assert all(entry["错误"] == "" for entry in grid_entries)
    assert all(entry["TCP_X"] is not None for entry in grid_entries)
    # 像素 u=100+20*col → u<400 即 col<15，全部格点都应在凸包内。
    assert all(entry["凸包内"] is True for entry in grid_entries)

    block_entry = document["方块"][0]
    assert block_entry["类别"] == "I"
    assert block_entry["抓取Z毫米"] == pytest.approx(6.0 + 151.0)
    assert block_entry["凸包内"] is True

    assert document["标定"]["方块"]["文件sha256"] == "aaa"
    assert document["标定"]["托盘"]["文件sha256"] == "bbb"
    assert document["标定"]["方块"]["XY模型"] == "poly2"


def test_formal_panorama_single_point_failure_does_not_infect_others():
    grid = _synthetic_grid(poison_cell=(7, 5))
    document = build_formal_panorama_document(
        generated_at="2026-08-16T15:00:00",
        image_shape=(720, 1280),
        board_angle_deg=0.0,
        board_grid_points=grid,
        observed_blocks=[],
        localizer=StubLocalizer(),
        pick_surface_offset_mm=151.0,
        block_calibration_sha256="aaa",
        tray_calibration_sha256="bbb",
    )
    entries = {(entry["行"], entry["列"]): entry for entry in document["托盘格点"]}
    poisoned = entries[(7, 5)]
    assert poisoned["错误"].startswith("格点像素插值失败")
    assert poisoned["TCP_X"] is None
    healthy = entries[(7, 4)]
    assert healthy["错误"] == ""
    assert healthy["TCP_X"] is not None


def test_formal_panorama_without_board_grid():
    document = build_formal_panorama_document(
        generated_at="2026-08-16T15:00:00",
        image_shape=(720, 1280),
        board_angle_deg=None,
        board_grid_points=None,
        observed_blocks=[],
        localizer=StubLocalizer(),
        pick_surface_offset_mm=151.0,
        block_calibration_sha256="aaa",
        tray_calibration_sha256="bbb",
    )
    assert document["托盘格点"] == []
    assert document["托盘旋转角deg"] is None


def test_calibration_panorama_keeps_depth_and_plane():
    observed = ObservedBlock(
        category="L",
        observation_pose=(-240.0, 10.0, 175.0, -180.0, 0.0, 90.0),
        detected_angle_deg=-30.0,
        pick_surface_z_mm=8.0,
        pick_surface_z_valid=True,
        high_detected_pixel_xy=(210.0, 260.0),
        high_world_position=(-320.0, 40.0, 8.0),
        high_world_position_valid=True,
        depth_valid_frame_count=10,
        depth_median_mm=520.0,
        depth_mad_mm=0.8,
    )
    plane = fit_z_plane(
        [
            (-240.0, 10.0, 175.0),
            (-340.0, 10.0, 176.0),
            (-240.0, 110.0, 175.5),
        ]
    )
    placement = PlacementTarget(
        index=0,
        row=7.5,
        col=5.5,
        desired_angle_deg=0.0,
        category="I",
        observation_pose=(-280.0, 60.0, 166.0, -180.0, 0.0, 90.0),
        high_detected_pixel_xy=(300.0, 330.0),
        high_world_position=(-355.0, 55.0, 150.0),
        high_world_position_valid=True,
        depth_valid_frame_count=10,
        depth_median_mm=512.0,
        depth_mad_mm=0.6,
    )
    document = build_calibration_panorama_document(
        generated_at="2026-08-16T15:00:00",
        image_shape=(720, 1280),
        board_angle_deg=0.5,
        board_grid_points=_synthetic_grid(),
        observed_blocks=[observed],
        placement_targets=[placement],
        block_plane=plane,
        pick_surface_offset_mm=151.0,
    )

    assert document["模式"] == "标定"
    assert document["方块观察Z平面"]["RMSE毫米"] >= 0.0
    assert len(document["方块观察Z平面"]["系数a_b_c"]) == 3
    assert len(document["托盘格点像素"]) == BOARD_ROW_COUNT * BOARD_COL_COUNT

    block_entry = document["方块"][0]
    assert block_entry["深度世界Z"] == pytest.approx(8.0)
    assert block_entry["抓取Z毫米"] == pytest.approx(159.0)
    assert block_entry["观察TCP_Z"] == pytest.approx(175.0)

    tray_entry = document["托盘采样点"][0]
    assert tray_entry["行"] == pytest.approx(7.5)
    assert tray_entry["深度世界X"] == pytest.approx(-355.0)
    assert tray_entry["托盘TCP_Z"] == pytest.approx(166.0)


def test_calibration_panorama_plane_missing_writes_null():
    document = build_calibration_panorama_document(
        generated_at="2026-08-16T15:00:00",
        image_shape=(720, 1280),
        board_angle_deg=None,
        board_grid_points=None,
        observed_blocks=[_observed_block()],
        placement_targets=[],
        block_plane=None,
        pick_surface_offset_mm=151.0,
    )
    assert document["方块观察Z平面"] is None
    assert document["托盘格点像素"] == []
    block_entry = document["方块"][0]
    # 正式字段仍在；深度字段缺省为 None 且不抛异常。
    assert block_entry["观察TCP_Z"] == pytest.approx(173.0)
    assert block_entry["深度世界X"] is None
