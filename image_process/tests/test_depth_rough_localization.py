"""深度世界坐标粗定位和方块/托盘 Z 解耦测试。"""

import numpy as np
import pytest

from image_process_lib.depth_rough_localization import DepthRoughLocalizer, fit_z_plane


def test_block_surface_world_xyz_is_converted_to_observation_and_pick_tcp_z():
    wrist_to_camera = np.eye(4)
    wrist_to_camera[:3, 3] = [10.0, 20.0, 30.0]
    localizer = DepthRoughLocalizer(
        [-250.0, 0.0, 380.0, 0.0, 0.0, 0.0],
        wrist_to_camera,
    )

    pose = localizer.block_observation_pose([-200.0, 50.0, 8.0], 192.0)

    assert pose == [-210.0, 30.0, 200.0, 0.0, 0.0, 0.0]
    # 深度返回的是方块表面世界 Z，抓取 TCP 高度仍由表面 Z 加 162 mm 得到。
    assert 8.0 + 162.0 == pytest.approx(pose[2] - 30.0)


def test_tray_depth_z_never_changes_tray_tcp_pose_z():
    localizer = DepthRoughLocalizer(
        [-250.0, 0.0, 380.0, 0.0, 0.0, 0.0],
        np.eye(4),
    )

    low_depth_pose = localizer.tray_observation_pose([-225.0, 30.0, 1.0], 194.35)
    noisy_depth_pose = localizer.tray_observation_pose([-225.0, 30.0, 999.0], 194.35)

    assert low_depth_pose == noisy_depth_pose
    assert low_depth_pose[:3] == [-225.0, 30.0, 194.35]


def test_block_and_tray_planes_keep_exact_seven_mm_difference():
    block_points = [
        [-300.0, 0.0, 200.0],
        [-250.0, 50.0, 201.5],
        [-200.0, -20.0, 200.6],
        [-180.0, 70.0, 202.6],
    ]
    plane = fit_z_plane(block_points)

    for x_mm, y_mm in [(-300.0, 0.0), (-225.0, 30.0), (-180.0, 70.0)]:
        block_z = plane.predict(x_mm, y_mm)
        tray_z = block_z - 7.0
        assert block_z - tray_z == pytest.approx(7.0)
    assert plane.rmse_mm == pytest.approx(0.0, abs=1e-10)


def test_block_plane_rejects_collinear_xy_points():
    with pytest.raises(ValueError, match="共线"):
        fit_z_plane([
            [-300.0, 0.0, 200.0],
            [-250.0, 0.0, 200.0],
            [-200.0, 0.0, 200.0],
        ])
