"""相机几何诊断核心的无硬件测试。"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml


DIAGNOSTIC_DIR = Path(__file__).resolve().parents[1]
ARUCO_DIR = DIAGNOSTIC_DIR.parents[0] / "aruco_pixel_tcp_diagnostic"
TOOLS_VISION_DIR = DIAGNOSTIC_DIR.parents[0]
SRC_ROOT = TOOLS_VISION_DIR.parents[1]
for path in (DIAGNOSTIC_DIR, ARUCO_DIR, TOOLS_VISION_DIR, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diagnostic_core import (  # noqa: E402
    CircleStabilityTracker,
    calibrate_from_observations,
    create_asymmetric_object_points,
    detect_asymmetric_circles,
    estimate_capture_pose,
    fit_plane_svd,
    heldout_reprojection_errors,
    make_board_material_masks,
    plane_quadratic_cross_validation,
    summarize_capture_coverage,
    undistort_pixel_coordinates,
    validate_image_size,
)
from analyze_aruco_experiment import (  # noqa: E402
    analyze_experiment,
    analyze_label_set,
    load_camera_calibration,
    undistort_high_pixels,
)


PATTERN_SIZE = (4, 7)


def test_asymmetric_object_points_follow_opencv_row_order():
    points = create_asymmetric_object_points(PATTERN_SIZE, 10.0)
    assert points.shape == (28, 3)
    np.testing.assert_allclose(
        points[:8],
        [
            [0, 0, 0], [20, 0, 0], [40, 0, 0], [60, 0, 0],
            [10, 10, 0], [30, 10, 0], [50, 10, 0], [70, 10, 0],
        ],
    )
    np.testing.assert_allclose(points[-1], [60, 60, 0])


def _synthetic_white_circle_grid():
    image = np.zeros((520, 620, 3), dtype=np.uint8)
    expected = []
    for row in range(PATTERN_SIZE[1]):
        for column in range(PATTERN_SIZE[0]):
            point = (120 + (2 * column + row % 2) * 50, 90 + row * 50)
            expected.append(point)
            cv2.circle(image, point, 14, (255, 255, 255), -1, cv2.LINE_AA)
    return image, np.asarray(expected, dtype=float)


def test_white_asymmetric_grid_detection_and_failure_are_explicit():
    image, expected = _synthetic_white_circle_grid()
    result = detect_asymmetric_circles(image, PATTERN_SIZE)
    assert result.found, result.message
    assert result.centers.shape == (28, 2)
    # findCirclesGrid 可能按等价方向输出，但集合必须逐点匹配。
    for point in expected:
        assert np.min(np.linalg.norm(result.centers - point, axis=1)) < 0.3

    failed = detect_asymmetric_circles(np.zeros_like(image), PATTERN_SIZE)
    assert not failed.found
    assert failed.centers is None
    assert "未检测" in failed.message


def test_circle_detection_can_be_locked_to_one_method():
    image, _expected = _synthetic_white_circle_grid()
    # 模拟黑底上的低亮度圆形纹理；不应进入真实白点候选集。
    cv2.circle(image, (570, 470), 14, (38, 38, 38), -1, cv2.LINE_AA)
    raw = detect_asymmetric_circles(image, PATTERN_SIZE, methods="raw_white")
    assert raw.found
    assert raw.branch == "原始灰度-白点"
    assert raw.keypoint_count == 28
    with pytest.raises(ValueError, match="未知圆点检测方法"):
        detect_asymmetric_circles(image, PATTERN_SIZE, methods="随机分支")


def test_circle_stability_requires_consecutive_low_jitter_frames():
    _image, centers = _synthetic_white_circle_grid()
    tracker = CircleStabilityTracker(
        required_frames=3,
        max_rms_jitter_px=0.5,
        max_point_jitter_px=1.0,
    )
    assert not tracker.update(centers)["stable"]
    assert not tracker.update(centers + 0.1)["stable"]
    assert tracker.update(centers - 0.1)["stable"]
    moved = tracker.update(centers + 5.0)
    assert not moved["stable"]
    assert moved["count"] == 1
    failed = tracker.update(None)
    assert failed["count"] == 0


def _project_capture_pose(rotation_vector, translation_vector):
    width, height = 1280, 720
    focal = width / (2.0 * np.tan(np.radians(86.0) / 2.0))
    camera_matrix = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    object_points = create_asymmetric_object_points(PATTERN_SIZE, 30.0)
    projected, _ = cv2.projectPoints(
        object_points,
        np.asarray(rotation_vector, dtype=float),
        np.asarray(translation_vector, dtype=float),
        camera_matrix,
        np.zeros(5),
    )
    return projected.reshape(-1, 2)


def test_capture_pose_and_coverage_provide_numeric_guidance():
    frontal = estimate_capture_pose(
        _project_capture_pose([0.0, 0.0, 0.0], [-90.0, -90.0, 450.0]),
        PATTERN_SIZE,
        (1280, 720),
    )
    assert frontal["position"] == "中央"
    assert frontal["tilt_direction"] == "正视"
    assert frontal["tilt_deg"] < 1.0

    right_near = estimate_capture_pose(
        _project_capture_pose([0.0, 0.32, 0.0], [-90.0, -90.0, 450.0]),
        PATTERN_SIZE,
        (1280, 720),
    )
    assert right_near["tilt_direction"] == "右侧靠近"
    assert 15.0 < right_near["tilt_deg"] < 22.0

    empty = summarize_capture_coverage([], minimum_samples=20)
    assert not empty["ready"]
    assert empty["progress_ratio"] == 0.0
    assert "中央" in empty["next_instruction"]


def _synthetic_calibration_observations(seed=7):
    rng = np.random.default_rng(seed)
    object_template = create_asymmetric_object_points(PATTERN_SIZE, 20.0)
    camera_matrix = np.array(
        [[700.0, 0.0, 640.0], [0.0, 705.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    distortion = np.array([0.12, -0.08, 0.0015, -0.001, 0.035], dtype=float)
    object_points = []
    image_points = []
    attempts = 0
    while len(object_points) < 24 and attempts < 200:
        attempts += 1
        rvec = rng.uniform([-0.35, -0.35, -0.2], [0.35, 0.35, 0.2])
        tvec = np.array(
            [rng.uniform(-230, 120), rng.uniform(-150, 70), rng.uniform(420, 760)],
            dtype=float,
        )
        projected, _ = cv2.projectPoints(
            object_template,
            rvec,
            tvec,
            camera_matrix,
            distortion,
        )
        image = projected.reshape(-1, 2)
        if (
            np.min(image[:, 0]) < 15
            or np.max(image[:, 0]) > 1265
            or np.min(image[:, 1]) < 15
            or np.max(image[:, 1]) > 705
        ):
            continue
        object_points.append(object_template.copy())
        image_points.append(image.astype(np.float32))
    assert len(object_points) == 24
    return object_points, image_points, camera_matrix, distortion


def test_five_parameter_calibration_recovers_synthetic_camera_and_beats_zero_model():
    object_points, image_points, true_k, _true_d = _synthetic_calibration_observations()
    fitted = calibrate_from_observations(object_points, image_points, (1280, 720))
    assert fitted["rms"] < 1e-3
    np.testing.assert_allclose(fitted["K"], true_k, rtol=2e-3, atol=0.5)
    standard_cv = heldout_reprojection_errors(
        object_points,
        image_points,
        (1280, 720),
        folds=5,
        seed=42,
    )
    zero_cv = heldout_reprojection_errors(
        object_points,
        image_points,
        (1280, 720),
        folds=5,
        seed=42,
        zero_distortion=True,
    )
    assert standard_cv["rmse"] < zero_cv["rmse"] * 0.2
    assert len(standard_cv["per_view_errors"]) == len(object_points)


def test_pixel_undistortion_uses_same_k_coordinate_frame():
    camera_matrix = np.array(
        [[700.0, 0.0, 640.0], [0.0, 700.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    distortion = np.array([0.1, -0.02, 0.001, -0.001, 0.01], dtype=float)
    original = np.array([[640.0, 360.0], [50.0, 50.0]], dtype=float)
    corrected = undistort_pixel_coordinates(original, camera_matrix, distortion)
    np.testing.assert_allclose(corrected[0], original[0], atol=1e-10)
    assert np.linalg.norm(corrected[1] - original[1]) > 1.0


def test_resolution_mismatch_is_rejected_without_resize():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="禁止自动缩放"):
        validate_image_size(image, (1280, 720))


def test_plane_and_quadratic_surface_are_distinguished_by_heldout_error():
    grid_x, grid_y = np.meshgrid(np.linspace(-120, 120, 35), np.linspace(-90, 90, 29))
    plane_z = 500.0 + 0.015 * grid_x - 0.02 * grid_y
    plane_points = np.column_stack([grid_x.ravel(), grid_y.ravel(), plane_z.ravel()])
    plane = fit_plane_svd(plane_points)
    assert plane.rmse < 1e-10

    bowl_z = plane_z + 0.00035 * grid_x**2 + 0.0005 * grid_y**2
    bowl_points = np.column_stack([grid_x.ravel(), grid_y.ravel(), bowl_z.ravel()])
    comparison = plane_quadratic_cross_validation(bowl_points, folds=5, seed=42)
    assert comparison["plane_rmse_mm"] > 1.0
    assert comparison["quadratic_improvement_ratio"] > 0.95


def test_material_masks_are_disjoint_and_inside_board():
    _image, centers = _synthetic_white_circle_grid()
    board, white, black, spacing = make_board_material_masks((520, 620), centers)
    assert spacing == pytest.approx(np.sqrt(2.0) * 50.0, rel=0.02)
    assert np.count_nonzero(board) > 0
    assert np.count_nonzero(white) > 0
    assert np.count_nonzero(black) > 0
    assert not np.any(white & black)
    assert np.all(white <= board)
    assert np.all(black <= board)


def test_camera_calibration_yaml_loader_rejects_wrong_model_and_reads_valid(tmp_path):
    valid_path = tmp_path / "相机标定.yaml"
    valid_path.write_text(
        yaml.safe_dump(
            {
                "model": "pinhole_radtan_5",
                "image_width": 1280,
                "image_height": 720,
                "K": [[700, 0, 640], [0, 700, 360], [0, 0, 1]],
                "D": [0.1, -0.02, 0, 0, 0.01],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    _data, camera_matrix, distortion, size = load_camera_calibration(valid_path)
    assert size == (1280, 720)
    assert camera_matrix.shape == (3, 3)
    assert distortion.shape == (5,)

    wrong_path = tmp_path / "错误模型.yaml"
    wrong_path.write_text("model: fisheye\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pinhole_radtan_5"):
        load_camera_calibration(wrong_path)


def test_corrected_pixels_restore_planar_homography_in_aruco_analysis():
    rng = np.random.default_rng(123)
    tcp_xy = rng.uniform([-240.0, -160.0], [240.0, 160.0], size=(90, 2))
    object_xyz = np.column_stack([tcp_xy, np.zeros(len(tcp_xy))]).astype(np.float64)
    camera_matrix = np.array(
        [[700.0, 0.0, 640.0], [0.0, 700.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    distortion = np.array([0.22, -0.11, 0.001, -0.001, 0.05], dtype=float)
    raw_uv, _ = cv2.projectPoints(
        object_xyz,
        np.array([0.08, -0.12, 0.04]),
        np.array([10.0, -5.0, 620.0]),
        camera_matrix,
        distortion,
    )
    raw_uv = raw_uv.reshape(-1, 2)
    corrected_uv = undistort_high_pixels(raw_uv, camera_matrix, distortion)
    raw_results, _raw_oof, _raw_selection = analyze_label_set(raw_uv, tcp_xy)
    corrected_results, _corrected_oof, _corrected_selection = analyze_label_set(
        corrected_uv,
        tcp_xy,
    )

    def homography_rmse(results):
        return next(
            float(item["CV二维RMSE"])
            for item in results
            if item["模型"] == "homography" and item["CV状态"] == "完成"
        )

    assert homography_rmse(corrected_results) < homography_rmse(raw_results) * 0.05


def test_full_aruco_ab_writes_independent_outputs(tmp_path):
    rng = np.random.default_rng(321)
    tcp_xy = rng.uniform([-220.0, -140.0], [220.0, 140.0], size=(48, 2))
    object_xyz = np.column_stack([tcp_xy, np.zeros(len(tcp_xy))]).astype(np.float64)
    camera_matrix = np.array(
        [[700.0, 0.0, 640.0], [0.0, 700.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    distortion = np.array([0.2, -0.1, 0.001, -0.001, 0.04], dtype=float)
    raw_uv, _ = cv2.projectPoints(
        object_xyz,
        np.array([0.05, -0.1, 0.03]),
        np.array([5.0, 2.0, 620.0]),
        camera_matrix,
        distortion,
    )
    raw_uv = raw_uv.reshape(-1, 2)
    csv_path = tmp_path / "aruco实验数据.csv"
    pd.DataFrame(
        {
            "事件": ["伺服成功"] * len(raw_uv),
            "高位检测像素X": raw_uv[:, 0],
            "高位检测像素Y": raw_uv[:, 1],
            "实测TCP位置X": tcp_xy[:, 0],
            "实测TCP位置Y": tcp_xy[:, 1],
            "零误差等效TCP位置X": tcp_xy[:, 0],
            "零误差等效TCP位置Y": tcp_xy[:, 1],
        }
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")
    calibration_path = tmp_path / "相机标定.yaml"
    calibration_path.write_text(
        yaml.safe_dump(
            {
                "model": "pinhole_radtan_5",
                "image_width": 1280,
                "image_height": 720,
                "K": camera_matrix.tolist(),
                "D": distortion.tolist(),
                "quality_pass": True,
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "分析"
    report = analyze_experiment(csv_path, output_dir, calibration_path)
    assert "畸变A_B" in report
    assert report["畸变A_B"]["单应改善比例"] > 0.2
    assert (output_dir / "畸变A_B模型对比.csv").is_file()
    assert (output_dir / "畸变A_B逐样本OOF误差.csv").is_file()
    assert (output_dir / "畸变A_B模型CV误差.png").is_file()
    assert (output_dir / "畸变A_B单应残差矢量.png").is_file()
