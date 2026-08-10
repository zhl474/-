"""ArUco 诊断实验无硬件单元测试：纯合成数据验证核心与离线分析逻辑。"""

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

DIAGNOSTIC_DIR = Path(__file__).resolve().parents[1]
TOOLS_VISION_DIR = DIAGNOSTIC_DIR.parents[1]
SRC_ROOT = TOOLS_VISION_DIR.parents[1]
for _path in (DIAGNOSTIC_DIR, TOOLS_VISION_DIR, SRC_ROOT, SRC_ROOT / "image_process"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from aruco_diagnostic_core import (  # noqa: E402
    CenterRefineConfig,
    check_image_size_constant,
    create_aruco_detector,
    detect_all_markers,
    detect_aruco_center,
    diagonal_intersection,
    draw_servo_overlay,
    image_center_uv,
    median_center,
    refine_marker_center,
    static_center_stats,
    validate_motion_pose,
    zero_error_tcp_xy,
)
from analyze_aruco_experiment import (  # noqa: E402
    LABEL_MAIN,
    analyze_experiment,
    analyze_label_set,
)

DETECTOR = create_aruco_detector("DICT_6X6_50")
MARKER_SIZE = 200


def _marker_image(marker_id=0, size=MARKER_SIZE):
    return cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50),
        int(marker_id),
        int(size),
    )


def _place_marker(canvas_size=(640, 480), offset=(80, 90), marker_id=0):
    canvas = np.full((canvas_size[1], canvas_size[0]), 255, dtype=np.uint8)
    marker = _marker_image(marker_id)
    x0, y0 = int(offset[0]), int(offset[1])
    canvas[y0:y0 + MARKER_SIZE, x0:x0 + MARKER_SIZE] = marker
    center = (x0 + MARKER_SIZE / 2.0, y0 + MARKER_SIZE / 2.0)
    return canvas, center


def test_detect_marker_id0_on_synthetic_image():
    image, true_center = _place_marker()
    result = detect_aruco_center(image, DETECTOR, marker_id=0)
    assert result.found, result.message
    assert np.linalg.norm(
        np.asarray(result.center_uv) - np.asarray(true_center)
    ) < 1.5
    # 200px 标记占据整数像素 [80,279]，中央黑白边界位于半像素 179.5/189.5。
    assert result.center_uv == pytest.approx((179.5, 189.5), abs=0.1)
    assert np.all(np.isfinite(result.rough_center_uv))
    assert np.all(np.isfinite(result.refine_delta_uv))
    assert result.center_contrast > 200.0


def test_low_contrast_blurred_big_marker_refines_center():
    """模拟实机暗码格：中央灰度角点应比错误外框粗中心更接近真值。"""
    marker = cv2.resize(_marker_image(0), (450, 450), interpolation=cv2.INTER_NEAREST)
    marker = np.where(marker > 0, 43, 29).astype(np.uint8)
    image = np.full((720, 1280), 180, dtype=np.uint8)
    image[135:585, 415:865] = marker
    image = cv2.GaussianBlur(image, (3, 3), 0.8)
    noise = np.random.default_rng(123).normal(0.0, 0.8, image.shape)
    image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)

    result = detect_aruco_center(image, DETECTOR, marker_id=0)
    assert result.found, result.message
    true_center = np.array([639.5, 359.5])
    refined_error = np.linalg.norm(np.asarray(result.center_uv) - true_center)
    rough_error = np.linalg.norm(np.asarray(result.rough_center_uv) - true_center)
    assert refined_error < 0.2
    assert refined_error < rough_error
    assert result.center_contrast >= 5.0


def test_non_checkerboard_marker_id_is_rejected_without_fallback():
    """ID=1 中央不是棋盘角点，即使 ArUco 解码成功也不能混入粗中心。"""
    image, _ = _place_marker(marker_id=1)
    result = detect_aruco_center(image, DETECTOR, marker_id=1)
    assert not result.found
    assert "不是黑白棋盘角点" in result.message
    assert np.all(np.isfinite(result.rough_center_uv))
    assert np.isnan(result.px) and np.isnan(result.py)


def test_center_refinement_rejects_low_contrast_and_excessive_shift():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    marker = _marker_image(0)
    low_contrast = np.where(marker > 0, 34, 30).astype(np.uint8)
    corners = np.array(
        [[0.0, 0.0], [199.0, 0.0], [199.0, 199.0], [0.0, 199.0]],
        dtype=float,
    )
    low_result = refine_marker_center(
        low_contrast, corners, (99.5, 99.5), dictionary, 0
    )
    assert not low_result.found
    assert "黑白分离度" in low_result.message

    shift_result = refine_marker_center(
        marker,
        corners,
        (101.5, 99.5),
        dictionary,
        0,
        CenterRefineConfig(max_shift_cell_ratio=0.01),
    )
    assert not shift_result.found
    assert "修正量" in shift_result.message


def test_reject_wrong_id_and_no_marker():
    wrong_id_image, _ = _place_marker(marker_id=1)
    result = detect_aruco_center(wrong_id_image, DETECTOR, marker_id=0)
    assert not result.found
    assert "ID" in result.message

    blank = np.full((480, 640), 255, dtype=np.uint8)
    result = detect_aruco_center(blank, DETECTOR, marker_id=0)
    assert not result.found


def test_reject_duplicate_target_id():
    image = np.full((480, 700), 255, dtype=np.uint8)
    marker = _marker_image(0)
    image[100:300, 80:280] = marker
    image[100:300, 400:600] = marker
    result = detect_aruco_center(image, DETECTOR, marker_id=0)
    assert not result.found
    assert "多个" in result.message


def test_diagonal_intersection_matches_projected_center_not_corner_average():
    homography = np.array(
        [
            [1.2, 0.1, 120.0],
            [0.05, 1.1, 60.0],
            [0.0015, -0.0012, 1.0],
        ],
        dtype=float,
    )
    canvas = np.full((700, 900), 255, dtype=np.uint8)
    canvas[100:100 + MARKER_SIZE, 200:200 + MARKER_SIZE] = _marker_image(0)
    warped = cv2.warpPerspective(canvas, homography, (600, 450))

    projected = homography @ np.array([300.0, 200.0, 1.0])
    true_center = projected[:2] / projected[2]

    result = detect_aruco_center(warped, DETECTOR, marker_id=0)
    assert result.found, result.message
    detected_center = np.asarray(result.center_uv)
    corner_mean = np.asarray(result.corners, dtype=float).mean(axis=0)
    assert np.linalg.norm(detected_center - true_center) < 4.0
    assert np.linalg.norm(corner_mean - true_center) > 5.0


def test_same_detector_high_low_and_error_sign():
    width, height = 640, 480
    marker_center = (450.0, 150.0)
    image = np.full((height, width), 255, dtype=np.uint8)
    x0 = int(marker_center[0] - MARKER_SIZE / 2)
    y0 = int(marker_center[1] - MARKER_SIZE / 2)
    image[y0:y0 + MARKER_SIZE, x0:x0 + MARKER_SIZE] = _marker_image(0)

    high_result = detect_aruco_center(image, DETECTOR, marker_id=0)
    low_result = detect_aruco_center(image, DETECTOR, marker_id=0)
    assert high_result.found and low_result.found
    assert np.isclose(high_result.px, low_result.px) and np.isclose(high_result.py, low_result.py)

    ref_x, ref_y = image_center_uv(width, height)
    assert np.isclose(low_result.dx_px, marker_center[0] - ref_x, atol=1.0)
    assert np.isclose(low_result.dy_px, marker_center[1] - ref_y, atol=1.0)


def test_pose_validation_rejects_dangerous_poses():
    safe = [-200.0, 50.0, 200.0, -180.0, 0.0, 90.0]
    ok, _ = validate_motion_pose(safe, 165.0, [-520.224, -148.17], [-263.279, 315.925])
    assert ok

    bad_nan = list(safe)
    bad_nan[0] = float("nan")
    assert not validate_motion_pose(bad_nan, 165.0, [-520.224, -148.17], [-263.279, 315.925])[0]

    bad_z = list(safe)
    bad_z[2] = 160.0
    assert not validate_motion_pose(bad_z, 165.0, [-520.224, -148.17], [-263.279, 315.925])[0]

    bad_x = list(safe)
    bad_x[0] = 100.0
    assert not validate_motion_pose(bad_x, 165.0, [-520.224, -148.17], [-263.279, 315.925])[0]

    bad_y = list(safe)
    bad_y[1] = 1000.0
    assert not validate_motion_pose(bad_y, 165.0, [-520.224, -148.17], [-263.279, 315.925])[0]


def test_zero_error_tcp_uses_unclamped_matrix_correction():
    matrix = [[0.0, 0.2], [0.2, 0.0]]
    actual = (10.0, 20.0)
    huge_error = (100.0, 50.0)
    equivalent = zero_error_tcp_xy(actual, huge_error, matrix, complete=True)
    expected = (
        actual[0] + matrix[0][1] * huge_error[1],
        actual[1] + matrix[1][0] * huge_error[0],
    )
    assert np.allclose(equivalent, expected)


def test_zero_error_gated_by_static_completeness():
    matrix = [[0.0, 0.2], [0.2, 0.0]]
    assert zero_error_tcp_xy((1.0, 2.0), (0.5, 0.5), matrix, complete=False) is None
    with pytest.raises(ValueError):
        zero_error_tcp_xy((1.0, float("nan")), (0.5, 0.5), matrix, complete=True)


def test_static_stats_completeness_threshold():
    valid = [(100.0, 200.0), (101.0, 199.0), (102.0, 201.0)]
    invalid = (float("nan"), float("nan"))
    incomplete = valid + [invalid] * 17
    stats = static_center_stats(incomplete, 20, 0.8)
    assert stats["有效帧数"] == 3
    assert not stats["完整"]

    complete = valid + [(100.0 + i, 200.0) for i in range(13)]
    assert len(complete) == 16
    stats = static_center_stats(complete, 20, 0.8)
    assert stats["完整"]
    assert np.isclose(stats["均值"][0], np.mean([center[0] for center in complete]))


def test_small_sample_skips_poly_models_but_keeps_affine_and_homography():
    uv, tcp = _synthetic_points(5)
    results, oof_per_model, selection = analyze_label_set(uv, tcp)
    by_model = {entry["模型"]: entry for entry in results}
    assert by_model["affine"]["CV状态"] == "完成"
    assert by_model["homography"]["CV状态"] == "完成"
    assert by_model["poly2"]["训练状态"] == "失败"
    assert by_model["poly3"]["训练状态"] == "失败"
    assert "affine" in oof_per_model and "homography" in oof_per_model
    assert "poly2" not in oof_per_model and "poly3" not in oof_per_model
    assert selection["推荐最简模型"] == "affine"


def test_full_training_ok_but_cv_insufficient_keeps_train_metrics():
    uv, tcp = _synthetic_points(10)
    results, oof_per_model, selection = analyze_label_set(uv, tcp)
    by_model = {entry["模型"]: entry for entry in results}
    assert by_model["affine"]["CV状态"] == "完成"
    assert by_model["poly2"]["CV状态"] == "完成"
    assert by_model["poly3"]["训练状态"] == "成功"
    assert by_model["poly3"]["CV状态"] == "数据不足"
    assert np.isfinite(by_model["poly3"]["训练二维RMSE_mm"])
    assert "poly3" not in oof_per_model
    assert selection["推荐最简模型"] == "affine"


def test_analyze_experiment_writes_all_outputs(tmp_path):
    csv_path = tmp_path / "aruco实验数据.csv"
    uv, tcp = _synthetic_points(10)
    rows = []
    for index in range(10):
        rows.append({
            "样本号": index + 1,
            "事件": "伺服成功",
            "高位检测像素X": uv[index, 0],
            "高位检测像素Y": uv[index, 1],
            "实测TCP位置X": tcp[index, 0],
            "实测TCP位置Y": tcp[index, 1],
            "零误差等效TCP位置X": tcp[index, 0] + 0.3,
            "零误差等效TCP位置Y": tcp[index, 1] - 0.2,
        })
    csv_path.write_text(
        ",".join(rows[0].keys()) + "\n" +
        "\n".join(",".join(str(rows[index][key]) for key in rows[0]) for index in range(10)),
        encoding="utf-8-sig",
    )

    output_dir = tmp_path / "离线分析"
    report = analyze_experiment(csv_path, output_dir)

    for filename in (
        "模型对比.csv", "逐样本OOF误差.csv", "分析报告.json",
        "模型CV误差对比.png", "OOF误差空间分布.png",
    ):
        assert (output_dir / filename).is_file(), filename
    main_label = report["标签"][LABEL_MAIN]
    assert main_label["样本数"] == 10
    by_model = {entry["模型"]: entry for entry in main_label["模型结果"]}
    assert by_model["affine"]["CV状态"] == "完成"
    assert by_model["poly3"]["CV状态"] == "数据不足"
    assert json.loads((output_dir / "分析报告.json").read_text(encoding="utf-8"))["总行数"] == 10


def _synthetic_points(count):
    """生成固定种子的非共线散点：uv 与 tcp 之间为精确仿射关系。"""
    rng = np.random.default_rng(7)
    uv = rng.uniform([100.0, 150.0], [900.0, 800.0], size=(int(count), 2))
    matrix = np.array([[0.1, 0.05], [0.02, 0.12]], dtype=float)
    tcp = uv @ matrix.T + np.array([10.0, -5.0])
    return uv, tcp


def test_helper_pure_functions():
    quad = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    center = diagonal_intersection(quad)
    assert np.allclose(center, (50.0, 50.0))
    assert median_center([(1.0, 10.0), (3.0, 30.0), (2.0, 20.0)]) == (2.0, 20.0)
    assert check_image_size_constant(640, 480, (640, 480))
    assert not check_image_size_constant(320, 240, (640, 480))


def test_detect_all_markers_lists_ids_sizes_and_corners():
    image, _ = _place_marker()
    results = detect_all_markers(image, ("DICT_6X6_50", "DICT_5X5_50", "DICT_4X4_50"))
    assert results["DICT_6X6_50"]["ids"] == [0]
    assert len(results["DICT_6X6_50"]["sizes_px"]) == 1
    assert results["DICT_6X6_50"]["sizes_px"][0] == pytest.approx(MARKER_SIZE, abs=2.0)
    assert len(results["DICT_6X6_50"]["corners"]) == 1
    json.dumps(results["DICT_6X6_50"]["corners"])
    assert results["DICT_5X5_50"]["ids"] == []
    assert results["DICT_4X4_50"]["ids"] == []


def test_detect_all_markers_relaxed_params_still_detect():
    image, _ = _place_marker()
    results = detect_all_markers(
        image, ("DICT_6X6_50",),
        corner_refine=False, min_perimeter_rate=0.01,
    )
    assert results["DICT_6X6_50"]["ids"] == [0]


def test_detect_all_markers_empty_on_blank_image():
    blank = np.full((480, 640), 255, dtype=np.uint8)
    results = detect_all_markers(blank, ("DICT_6X6_50", "DICT_ARUCO_ORIGINAL"))
    assert results["DICT_6X6_50"]["ids"] == []
    assert results["DICT_ARUCO_ORIGINAL"]["ids"] == []


def test_detect_all_markers_includes_rejected_fields():
    image, _ = _place_marker()
    results = detect_all_markers(image, ("DICT_6X6_50",))
    found = results["DICT_6X6_50"]
    assert isinstance(found["rejected_count"], int)
    assert isinstance(found["rejected_corners"], list)
    json.dumps(found)  # rejected 字段必须可序列化，供 CSV 诊断使用


def test_big_marker_detected_with_default_enlarged_threshold_window():
    """模拟低位尺度：450px 标记（低位实测约 447px）在 1280x720 画面中。"""
    marker = _marker_image(0)
    marker_big = cv2.resize(marker, (450, 450), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((720, 1280), 255, dtype=np.uint8)
    canvas[135:585, 415:865] = marker_big
    result = detect_aruco_center(canvas, DETECTOR, marker_id=0)
    assert result.found, result.message
    assert result.center_uv[0] == pytest.approx(640.0, abs=2.0)
    assert result.center_uv[1] == pytest.approx(360.0, abs=2.0)


def test_small_marker_still_detected_with_enlarged_window():
    """默认窗口放大后，高位小尺度（约 200px）也必须仍然检出。"""
    image, true_center = _place_marker()
    result = detect_aruco_center(image, DETECTOR, marker_id=0)
    assert result.found, result.message
    assert np.linalg.norm(
        np.asarray(result.center_uv) - np.asarray(true_center)
    ) < 1.5


def test_draw_servo_overlay_keeps_frame_size():
    image, _ = _place_marker()
    overlay = draw_servo_overlay(
        image,
        round_no=5,
        error_xy=(1.5, -2.5),
        center_uv=(320.0, 240.0),
        corners=None,
        rejected_corners=[[[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]]],
    )
    assert overlay.shape == image.shape
    assert not np.array_equal(overlay, image)

    bare = draw_servo_overlay(image, round_no=1)
    assert bare.shape == image.shape


def _storage_definition():
    return (
        {"参数A": 1, "参数B": [2, 3]},
        {"配置A": {"阈值": 0.5}, "配置B": "固定"},
    )


def _append_fake_sample(run, experiment, sample_no, event=None):
    experiment.sample_counter = int(sample_no)
    row = {column: "" for column in run.CSV_COLUMNS}
    row["样本号"] = int(sample_no)
    row["事件"] = event or run.SUCCESS_EVENT
    experiment.record(row)


def test_experiment_resume_appends_without_duplicate_header(tmp_path):
    import run_aruco_experiment as run

    parameters, configuration = _storage_definition()
    root = tmp_path / "实验根目录"
    experiment, previous, state = run.create_or_resume_experiment(
        root, False, None, parameters, configuration, batch_name="批次001"
    )
    assert previous is None and state is None
    metadata, run_index = run.start_experiment_metadata(
        experiment, previous, parameters, configuration
    )
    _append_fake_sample(run, experiment, 1)
    experiment.close()
    run.finish_experiment_metadata(experiment, metadata, run_index, "正常结束")

    resumed, previous, state = run.create_or_resume_experiment(
        root, True, root / "批次001", parameters, configuration
    )
    assert resumed.record_count == 1
    assert resumed.sample_counter == 1
    assert state["orphan_sample_numbers"] == set()
    metadata, run_index = run.start_experiment_metadata(
        resumed, previous, parameters, configuration
    )
    _append_fake_sample(run, resumed, 2)
    resumed.close()
    run.finish_experiment_metadata(resumed, metadata, run_index, "正常结束")

    csv_path = root / "批次001" / "aruco实验数据.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
    assert reader.fieldnames == run.CSV_COLUMNS
    assert [int(row["样本号"]) for row in rows] == [1, 2]
    assert csv_path.read_text(encoding="utf-8-sig").count("样本号,事件,时间戳") == 1

    saved_metadata = json.loads(
        (root / "批次001" / "实验元数据.json").read_text(encoding="utf-8")
    )
    assert len(saved_metadata["运行历史"]) == 2
    assert saved_metadata["统计"]["CSV记录数"] == 2
    assert saved_metadata["统计"]["最大样本号"] == 2


def test_resume_skips_orphan_file_number_and_refuses_overwrite(tmp_path):
    import run_aruco_experiment as run

    parameters, configuration = _storage_definition()
    root = tmp_path / "实验根目录"
    experiment, previous, _ = run.create_or_resume_experiment(
        root, False, None, parameters, configuration, batch_name="批次残留"
    )
    metadata, run_index = run.start_experiment_metadata(
        experiment, previous, parameters, configuration
    )
    for sample_no in range(1, 5):
        _append_fake_sample(run, experiment, sample_no)
    experiment.close()
    run.finish_experiment_metadata(experiment, metadata, run_index, "正常结束")
    orphan = root / "批次残留" / "样本005_高位_原图.png"
    orphan.write_bytes(b"interrupted")

    resumed, _, state = run.create_or_resume_experiment(
        root, True, root / "批次残留", parameters, configuration
    )
    assert resumed.sample_counter == 5
    assert state["orphan_sample_numbers"] == {5}
    with pytest.raises(RuntimeError, match="禁止覆盖"):
        resumed.image_paths(5)
    assert resumed.image_paths(6)["high_original"].name == "样本006_高位_原图.png"
    resumed.close()


def test_resume_rejects_missing_path_schema_and_config_mismatch(tmp_path):
    import run_aruco_experiment as run

    parameters, configuration = _storage_definition()
    root = tmp_path / "实验根目录"
    with pytest.raises(RuntimeError, match="必须设置"):
        run.validate_resume_batch(root, None, parameters, configuration)

    experiment, previous, _ = run.create_or_resume_experiment(
        root, False, None, parameters, configuration, batch_name="批次校验"
    )
    metadata, run_index = run.start_experiment_metadata(
        experiment, previous, parameters, configuration
    )
    experiment.close()
    run.finish_experiment_metadata(experiment, metadata, run_index, "正常结束")
    batch_dir = root / "批次校验"
    csv_before = (batch_dir / "aruco实验数据.csv").read_bytes()
    metadata_before = (batch_dir / "实验元数据.json").read_bytes()
    with pytest.raises(RuntimeError, match="实验条件"):
        run.validate_resume_batch(
            root, batch_dir, {**parameters, "参数A": 999}, configuration
        )
    assert (batch_dir / "aruco实验数据.csv").read_bytes() == csv_before
    assert (batch_dir / "实验元数据.json").read_bytes() == metadata_before

    outside = tmp_path / "根目录外"
    outside.mkdir()
    with pytest.raises(RuntimeError, match="OUTPUT_ROOT"):
        run.validate_resume_batch(root, outside, parameters, configuration)

    (batch_dir / "aruco实验数据.csv").write_text(
        "错误表头\n", encoding="utf-8-sig"
    )
    with pytest.raises(RuntimeError, match="表头"):
        run.validate_resume_batch(root, batch_dir, parameters, configuration)


def test_resume_marks_unfinished_run_as_interrupted(tmp_path):
    import run_aruco_experiment as run

    parameters, configuration = _storage_definition()
    root = tmp_path / "实验根目录"
    experiment, previous, _ = run.create_or_resume_experiment(
        root, False, None, parameters, configuration, batch_name="批次中断"
    )
    run.start_experiment_metadata(experiment, previous, parameters, configuration)
    experiment.close()  # 模拟进程硬中断：不调用 finish_experiment_metadata。

    resumed, previous, _ = run.create_or_resume_experiment(
        root, True, root / "批次中断", parameters, configuration
    )
    metadata, run_index = run.start_experiment_metadata(
        resumed, previous, parameters, configuration
    )
    assert metadata["运行历史"][-2]["状态"] == "异常中断"
    resumed.close()
    run.finish_experiment_metadata(resumed, metadata, run_index, "正常结束")


def test_open_video_writer_produces_readable_avi(tmp_path):
    """录像写入后必须能被 OpenCV 正常读回。"""
    import run_aruco_experiment as run

    video_path = tmp_path / "servo.avi"
    writer, actual_path = run.open_video_writer(video_path, 10.0, (640, 480))
    assert writer is not None and writer.isOpened()
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    for _ in range(5):
        writer.write(frame)
    writer.release()
    reader = cv2.VideoCapture(str(actual_path))
    assert reader.isOpened()
    ok, _read_frame = reader.read()
    reader.release()
    assert ok


def test_move_checked_always_waits_for_stable():
    """所有运动（含每轮伺服修正）必须向 MoveArm 服务传 wait_until_stable=True。"""
    import run_aruco_experiment as run

    class FakeServices:
        def __init__(self):
            self.calls = []

        def move_to(self, pose, speed, wait_until_stable=True):
            self.calls.append((list(pose), speed, bool(wait_until_stable)))

    fake = FakeServices()
    pose = [-200.0, 50.0, 200.0, -180.0, 0.0, 90.0]
    run.move_checked(fake, pose, 50, 165.0, [-520.224, -148.17], [-263.279, 315.925])
    run.move_checked(
        fake, pose, 30, 165.0, [-520.224, -148.17], [-263.279, 315.925],
        wait_until_stable=True,
    )
    assert len(fake.calls) == 2
    for _, speed, wait_until_stable in fake.calls:
        assert wait_until_stable is True

    # 安全检查失败时绝不能发出运动命令
    bad_pose = list(pose)
    bad_pose[2] = 100.0
    with pytest.raises(RuntimeError):
        run.move_checked(fake, bad_pose, 50, 165.0, [-520.224, -148.17], [-263.279, 315.925])
    assert len(fake.calls) == 2
