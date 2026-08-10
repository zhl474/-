#!/home/zhl/fr3env/fr3env/bin/python
"""Gemini 335 工程彩色图的非对称圆阵标定与验证工具。

模式、路径和板参数都集中在文件开头；不使用命令行参数。
"""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diagnostic_core import (  # noqa: E402
    CIRCLE_DETECTION_METHODS,
    PINHOLE_MODEL_NAME,
    CircleStabilityTracker,
    calibrate_from_observations,
    create_asymmetric_object_points,
    detect_asymmetric_circles,
    draw_circle_order,
    estimate_capture_pose,
    find_similar_signature,
    heldout_reprojection_errors,
    put_chinese_text,
    read_image,
    sample_signature,
    save_image,
    summarize_capture_coverage,
    to_builtin,
    undistort_pixel_coordinates,
    validate_image_size,
)


# ==================== 运行参数（直接修改本文件）====================
# 可选值：board_check / capture / calibrate / verify
MODE = "verify"

CAMERA_TOPIC = "/camera/image_raw"
EXPECTED_IMAGE_SIZE = (1280, 720)  # (宽, 高)，不允许自动缩放
PATTERN_SIZE = (4, 7)  # 暂按“每行4点、共7行”定义
BOARD_CONFIRMED = True  # 看过完整正视照片并确认顺序后改为 True
GRID_STEP_MM = 1.0  # 相邻行纵向间距，也是同行圆心水平间距的一半
GRID_STEP_IS_PHYSICAL = False  # 已实测毫米值后改为 True

# 标定采集和离线求解只使用这一种方法，禁止逐帧在多个预处理分支间切换。
# 可选值见 CIRCLE_DETECTION_METHODS；应先在板型确认模式中选定，再保持整批一致。
DETECTION_METHOD = "raw_white"

# 当前现场原图中真实白点中心灰度约110以上，黑底伪圆约37；避免使用过低阈值。
BLOB_DETECTOR_OPTIONS = {
    "min_threshold": 50.0,
    "max_threshold": 220.0,
    "threshold_step": 5.0,
    "min_area": 50.0,
    "max_area": 10000.0,
    "min_circularity": 0.50,
    "min_convexity": 0.70,
    "min_inertia_ratio": 0.35,
}

# 连续若干帧的28个圆心都基本静止后才允许保存，防止运动模糊和点序跳变。
STABLE_REQUIRED_FRAMES = 5
STABLE_MAX_RMS_JITTER_PX = 0.65
STABLE_MAX_POINT_JITTER_PX = 1.50

# 采集导航只使用近似内参估算倾角，不参与最终 K/D 求解。
APPROXIMATE_HORIZONTAL_FOV_DEG = 86.0
GUIDE_MINIMUM_SAMPLES = 20

OUTPUT_ROOT = Path.home() / "桌面" / "相机几何诊断" / "圆点板标定"
BOARD_CHECK_IMAGE = None  # 例如 Path("/绝对路径/标定板正视图.png")；None 时订阅相机
CALIBRATION_IMAGE_DIR = Path.home() / "桌面" / "相机几何诊断" / "圆点板标定" / "20260810-161036" / "原始图"
CALIBRATION_OUTPUT_DIR = Path.home() / "桌面" / "相机几何诊断" / "圆点板标定" / "20260810-161036" / "标定结果"
VERIFY_CALIBRATION_YAML = CALIBRATION_OUTPUT_DIR / "相机标定.yaml"
VERIFY_IMAGE_PATHS = []  # 留空时使用 ROS 实时画面
VENDOR_PROFILE_JSON = None  # 可填写 camera_profile_probe.py 生成的 JSON
OLD_PROJECT_CALIBRATION = SRC_ROOT / "competition" / "config" / "calibration_matrix.yaml"

MIN_VALID_IMAGES = 15
RECOMMENDED_VALID_IMAGES = 20
CV_FOLDS = 5
RANDOM_SEED = 42
MAX_TOTAL_RMS_PX = 0.5
MAX_SINGLE_VIEW_RMS_PX = 1.0
DISPLAY_SCALE = 0.72  # 只缩放显示窗口，不改变保存图和计算坐标
ROS_WAIT_TIMEOUT_SEC = 3.0
PIXELS_TO_CHECK = [
    (640.0, 360.0),
    (100.0, 100.0),
    (1180.0, 100.0),
    (100.0, 620.0),
    (1180.0, 620.0),
]


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _now():
    return datetime.now(timezone.utc).astimezone()


def _display(image, title="rgb_circle_calibration"):
    scale = float(DISPLAY_SCALE)
    shown = image
    if 0.0 < scale < 1.0:
        shown = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow(title, shown)


def _detect_board(image, pattern_size=PATTERN_SIZE):
    """所有正式采集和求解统一走同一个检测方法。"""
    return detect_asymmetric_circles(
        image,
        pattern_size,
        methods=(DETECTION_METHOD,),
        blob_detector_options=BLOB_DETECTOR_OPTIONS,
    )


def _progress_text(counts, targets, order):
    return "  ".join(f"{name}{counts[name]}/{targets[name]}" for name in order)


def _draw_capture_dashboard(image, detection, stability, pose, coverage, similar_index):
    """在原始画面右侧绘制定量采集导航，不覆盖标定板。"""
    height = image.shape[0]
    panel_width = 560
    panel = np.full((height, panel_width, 3), 28, dtype=np.uint8)
    green = (80, 230, 80)
    yellow = (0, 210, 255)
    red = (70, 70, 255)
    white = (235, 235, 235)
    muted = (175, 175, 175)
    y = 18

    put_chinese_text(panel, "圆点板采集导航", (18, y), white, 28)
    y += 42
    detection_color = green if detection.found else red
    put_chinese_text(
        panel,
        f"检测：{'成功' if detection.found else '失败'}  方法：{detection.branch}",
        (18, y),
        detection_color,
        20,
    )
    y += 31
    blob_color = green if detection.keypoint_count == math.prod(PATTERN_SIZE) else yellow
    put_chinese_text(
        panel,
        f"候选圆点：{detection.keypoint_count}/28",
        (18, y),
        blob_color,
        20,
    )
    y += 29
    stable_text = (
        f"稳定：{stability['count']}/{stability['required']}"
        + ("，可以保存" if stability["stable"] else "，请保持静止")
    )
    put_chinese_text(panel, stable_text, (18, y), green if stability["stable"] else yellow, 21)
    y += 30
    if stability["rms_jitter_px"] is not None:
        put_chinese_text(
            panel,
            f"圆心抖动 RMS={stability['rms_jitter_px']:.2f}px，最大={stability['max_jitter_px']:.2f}px",
            (18, y),
            muted,
            18,
        )
    y += 31

    if pose is None:
        put_chinese_text(panel, "当前姿态：等待完整检测", (18, y), red, 21)
    else:
        center = pose["center_normalized"]
        put_chinese_text(
            panel,
            f"当前位置：{pose['position']}  中心=({center[0]:.0%},{center[1]:.0%})",
            (18, y),
            white,
            21,
        )
        y += 29
        put_chinese_text(
            panel,
            f"当前尺寸：{pose['size_class']}  长边比例={pose['size_ratio']:.0%}",
            (18, y),
            yellow if pose["size_class"] in ("过远", "过近") else white,
            21,
        )
        y += 29
        put_chinese_text(
            panel,
            f"当前倾斜：{pose['tilt_direction']}  约{pose['tilt_deg']:.1f}°",
            (18, y),
            yellow if pose["tilt_deg"] > 35.0 else white,
            21,
        )
    y += 40

    ratio = float(coverage["progress_ratio"])
    put_chinese_text(
        panel,
        f"已保存 {coverage['sample_count']}/{coverage['minimum_samples']}，覆盖进度 {ratio:.0%}",
        (18, y),
        green if coverage["ready"] else white,
        22,
    )
    y += 31
    bar_left, bar_right = 18, panel_width - 22
    cv2.rectangle(panel, (bar_left, y), (bar_right, y + 18), (100, 100, 100), 1)
    cv2.rectangle(
        panel,
        (bar_left + 1, y + 1),
        (bar_left + 1 + int((bar_right - bar_left - 2) * ratio), y + 17),
        green,
        -1,
    )
    y += 32

    position_counts = coverage["position_counts"]
    position_targets = coverage["position_targets"]
    put_chinese_text(panel, "位置覆盖（板中心九宫格）", (18, y), white, 20)
    y += 27
    for row in (("左上", "上", "右上"), ("左", "中央", "右"), ("左下", "下", "右下")):
        put_chinese_text(
            panel,
            _progress_text(position_counts, position_targets, row),
            (32, y),
            muted,
            18,
        )
        y += 25

    put_chinese_text(
        panel,
        "尺寸：" + _progress_text(
            coverage["scale_counts"], coverage["scale_targets"], ("远", "中", "近")
        ),
        (18, y),
        white,
        19,
    )
    y += 29
    put_chinese_text(
        panel,
        "倾斜：" + _progress_text(
            coverage["tilt_counts"],
            coverage["tilt_targets"],
            ("正视", "左侧靠近", "右侧靠近"),
        ),
        (18, y),
        white,
        18,
    )
    y += 25
    put_chinese_text(
        panel,
        "      " + _progress_text(
            coverage["tilt_counts"],
            coverage["tilt_targets"],
            ("上侧靠近", "下侧靠近"),
        ),
        (18, y),
        white,
        18,
    )
    y += 32
    if similar_index is not None:
        put_chinese_text(panel, f"提示：与已保存样本{similar_index + 1}相似", (18, y), yellow, 19)
        y += 26
    put_chinese_text(panel, "下一步：", (18, y), yellow if not coverage["ready"] else green, 21)
    y += 27
    instruction = coverage["next_instruction"]
    lines = [instruction[index : index + 24] for index in range(0, len(instruction), 24)]
    for line in lines[:3]:
        put_chinese_text(panel, line, (32, y), white, 19)
        y += 25
    put_chinese_text(panel, "空格保存有效帧，d保存诊断帧，q退出", (18, height - 42), muted, 18)
    return np.hstack([image, panel])


def _wait_ros_image():
    import rospy
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image

    message = rospy.wait_for_message(CAMERA_TOPIC, Image, timeout=float(ROS_WAIT_TIMEOUT_SEC))
    return CvBridge().imgmsg_to_cv2(message, "bgr8")


def _load_reference_calibration(path):
    if path is None or not Path(path).is_file():
        return None
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    camera_matrix = data.get("K", data.get("camera_matrix"))
    distortion = data.get("D", data.get("dist_coeff"))
    if isinstance(camera_matrix, dict):
        camera_matrix = camera_matrix.get("data")
    if isinstance(distortion, dict):
        distortion = distortion.get("data")
    try:
        k = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
        d = np.asarray(distortion, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return {"路径": str(path), "错误": "无法解析 K/D"}
    return {"路径": str(path), "K": k.tolist(), "D": d.tolist()}


def _load_vendor_profile(path):
    if path is None or not Path(path).is_file():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"路径": str(path), "读取错误": str(exc)}


def run_board_check():
    """同时尝试两个候选 patternSize，只负责让人工锁定板定义。"""
    if BOARD_CHECK_IMAGE is not None:
        image = read_image(BOARD_CHECK_IMAGE)
        candidates = (PATTERN_SIZE, (PATTERN_SIZE[1], PATTERN_SIZE[0]))
        canvases = []
        for candidate in candidates:
            detection = _detect_board(image, candidate)
            canvas = draw_circle_order(image, detection, candidate)
            put_chinese_text(
                canvas,
                f"规格={candidate}，分支={detection.branch}",
                (15, 5),
                (0, 255, 255) if detection.found else (0, 0, 255),
                21,
            )
            canvases.append(canvas)
        _display(np.hstack(canvases), "board check: left=current right=transposed")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    import rospy

    rospy.init_node("asymmetric_circle_board_check", anonymous=True)
    print("板型确认模式：q 退出；窗口左侧为当前定义，右侧为转置定义。")
    while not rospy.is_shutdown():
        image = _wait_ros_image()
        validate_image_size(image, EXPECTED_IMAGE_SIZE)
        canvases = []
        for candidate in (PATTERN_SIZE, (PATTERN_SIZE[1], PATTERN_SIZE[0])):
            detection = _detect_board(image, candidate)
            canvas = draw_circle_order(image, detection, candidate)
            put_chinese_text(
                canvas,
                f"规格={candidate}，分支={detection.branch}",
                (15, 5),
                (0, 255, 255) if detection.found else (0, 0, 255),
                21,
            )
            canvases.append(canvas)
        _display(np.hstack(canvases), "board check: left=current right=transposed")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def run_capture():
    """人工控制保存检测成功的工程原始彩色帧。"""
    if not BOARD_CONFIRMED:
        raise RuntimeError("请先运行 board_check 并确认完整标定板，再把 BOARD_CONFIRMED 改为 True")
    import rospy

    rospy.init_node("asymmetric_circle_capture", anonymous=True)
    run_time = _now()
    output_dir = OUTPUT_ROOT / run_time.strftime("%Y%m%d-%H%M%S")
    raw_dir = output_dir / "原始图"
    overlay_dir = output_dir / "检测标注"
    diagnostic_dir = output_dir / "诊断帧"
    raw_dir.mkdir(parents=True, exist_ok=False)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "开始时间": run_time.isoformat(timespec="seconds"),
        "图像来源": CAMERA_TOPIC,
        "图像尺寸": list(EXPECTED_IMAGE_SIZE),
        "pattern_size": list(PATTERN_SIZE),
        "固定检测方法": DETECTION_METHOD,
        "Blob检测参数": BLOB_DETECTOR_OPTIONS,
        "采集稳定门禁": {
            "连续帧数": int(STABLE_REQUIRED_FRAMES),
            "圆心RMS抖动上限像素": float(STABLE_MAX_RMS_JITTER_PX),
            "单点最大抖动上限像素": float(STABLE_MAX_POINT_JITTER_PX),
        },
        "采集导航说明": (
            f"倾角由约{APPROXIMATE_HORIZONTAL_FOV_DEG:.1f}°水平视场估算，"
            "只用于采集覆盖导航，不参与最终K/D求解"
        ),
        "保存样本": [],
        "诊断帧": [],
    }
    signatures = []
    saved_poses = []
    tracker = CircleStabilityTracker(
        required_frames=STABLE_REQUIRED_FRAMES,
        max_rms_jitter_px=STABLE_MAX_RMS_JITTER_PX,
        max_point_jitter_px=STABLE_MAX_POINT_JITTER_PX,
    )
    print(
        f"采集模式：固定检测方法={DETECTION_METHOD}；"
        "圆心连续稳定后按空格/回车保存，q 退出。"
    )
    try:
        while not rospy.is_shutdown():
            image = _wait_ros_image()
            validate_image_size(image, EXPECTED_IMAGE_SIZE)
            detection = _detect_board(image)
            stability = tracker.update(detection.centers if detection.found else None)
            overlay = draw_circle_order(image, detection, PATTERN_SIZE)
            similar_index = None
            current_signature = None
            current_pose = None
            if detection.found:
                current_signature = sample_signature(detection.centers, EXPECTED_IMAGE_SIZE)
                similar_index = find_similar_signature(current_signature, signatures)
                current_pose = estimate_capture_pose(
                    detection.centers,
                    PATTERN_SIZE,
                    EXPECTED_IMAGE_SIZE,
                    approximate_hfov_deg=APPROXIMATE_HORIZONTAL_FOV_DEG,
                )
            coverage = summarize_capture_coverage(
                saved_poses,
                minimum_samples=GUIDE_MINIMUM_SAMPLES,
            )
            dashboard = _draw_capture_dashboard(
                overlay,
                detection,
                stability,
                current_pose,
                coverage,
                similar_index,
            )
            _display(dashboard, "capture dashboard: space save, q quit")
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                diagnostic_index = len(manifest["诊断帧"]) + 1
                diagnostic_raw_path = diagnostic_dir / f"诊断{diagnostic_index:03d}_原图.png"
                diagnostic_overlay_path = diagnostic_dir / f"诊断{diagnostic_index:03d}_界面.png"
                save_image(diagnostic_raw_path, image)
                save_image(diagnostic_overlay_path, dashboard)
                manifest["诊断帧"].append(
                    {
                        "序号": diagnostic_index,
                        "原图": str(diagnostic_raw_path),
                        "界面": str(diagnostic_overlay_path),
                        "检测成功": bool(detection.found),
                        "检测方法": detection.branch,
                        "候选圆点数": int(detection.keypoint_count),
                        "稳定状态": stability,
                    }
                )
                (output_dir / "采集清单.json").write_text(
                    json.dumps(to_builtin(manifest), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"已保存诊断帧：{diagnostic_raw_path}")
                continue
            if key not in (10, 13, 32):
                continue
            if not detection.found:
                print(f"未保存：{detection.message}")
                continue
            if not stability["stable"]:
                print(
                    f"未保存：圆心只稳定 {stability['count']}/{stability['required']} 帧，"
                    "请停止移动并等待稳定提示"
                )
                continue
            sample_index = len(manifest["保存样本"]) + 1
            raw_path = raw_dir / f"样本{sample_index:03d}.png"
            overlay_path = overlay_dir / f"样本{sample_index:03d}_检测.png"
            save_image(raw_path, image)
            saved_poses.append(current_pose)
            coverage_after_save = summarize_capture_coverage(
                saved_poses,
                minimum_samples=GUIDE_MINIMUM_SAMPLES,
            )
            dashboard_after_save = _draw_capture_dashboard(
                overlay,
                detection,
                stability,
                current_pose,
                coverage_after_save,
                similar_index,
            )
            save_image(overlay_path, dashboard_after_save)
            record = {
                "样本号": sample_index,
                "原始图": str(raw_path),
                "检测标注": str(overlay_path),
                "检测分支": detection.branch,
                "圆心": detection.centers.tolist(),
                "采样特征": current_signature,
                "采集姿态估算": current_pose,
                "保存时稳定状态": stability,
                "采样后覆盖进度": coverage_after_save,
                "相似样本提示": None if similar_index is None else similar_index + 1,
            }
            manifest["保存样本"].append(record)
            manifest["当前覆盖进度"] = coverage_after_save
            signatures.append(current_signature)
            tracker.reset()
            (output_dir / "采集清单.json").write_text(
                json.dumps(to_builtin(manifest), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            warning = "（与已有样本相似，请人工判断）" if similar_index is not None else ""
            print(f"已保存有效样本 {sample_index}{warning}: {raw_path}")
            print(f"下一步：{coverage_after_save['next_instruction']}")
    finally:
        manifest["结束时间"] = _now().isoformat(timespec="seconds")
        (output_dir / "采集清单.json").write_text(
            json.dumps(to_builtin(manifest), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        cv2.destroyAllWindows()
        print(f"采集目录：{output_dir}")


def _image_files(directory):
    paths = [path for path in Path(directory).iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(paths, key=lambda item: item.name)


def _calibration_summary(result):
    per_view = np.asarray(result["per_view_errors"], dtype=float)
    return {
        "总体RMS_px": float(result["rms"]),
        "K": result["K"].tolist(),
        "D": result["D"].tolist(),
        "内参标准差": result["intrinsic_std"].tolist(),
        "逐图RMS_px": per_view.tolist(),
        "逐图平均RMS_px": float(np.mean(per_view)),
        "逐图最大RMS_px": float(np.max(per_view)),
    }


def run_calibrate():
    """重新检测全部原图、标定并保存完整质量报告。"""
    if not BOARD_CONFIRMED:
        raise RuntimeError("BOARD_CONFIRMED=False：板型未确认，禁止产出标定参数")
    image_paths = _image_files(CALIBRATION_IMAGE_DIR)
    if not image_paths:
        raise RuntimeError(f"标定图片目录为空: {CALIBRATION_IMAGE_DIR}")
    output_dir = Path(CALIBRATION_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "检测标注"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    object_template = create_asymmetric_object_points(PATTERN_SIZE, GRID_STEP_MM)
    object_points = []
    image_points = []
    valid_records = []
    failed_records = []
    print(f"开始离线检测 {len(image_paths)} 张图片。")
    for image_path in image_paths:
        image = read_image(image_path)
        validate_image_size(image, EXPECTED_IMAGE_SIZE)
        detection = _detect_board(image)
        overlay = draw_circle_order(image, detection, PATTERN_SIZE)
        save_image(overlay_dir / f"{image_path.stem}_检测.png", overlay)
        if not detection.found:
            failed_records.append({"文件": str(image_path), "原因": detection.message})
            print(f"检测失败，不加入标定：{image_path.name}：{detection.message}")
            continue
        object_points.append(object_template.copy())
        image_points.append(detection.centers.copy())
        valid_records.append(
            {
                "文件": str(image_path),
                "检测分支": detection.branch,
                "圆心数量": int(len(detection.centers)),
                "采集姿态估算": estimate_capture_pose(
                    detection.centers,
                    PATTERN_SIZE,
                    EXPECTED_IMAGE_SIZE,
                    approximate_hfov_deg=APPROXIMATE_HORIZONTAL_FOV_DEG,
                ),
            }
        )
    if len(valid_records) < int(MIN_VALID_IMAGES):
        raise RuntimeError(
            f"有效图片只有 {len(valid_records)} 张，少于硬性下限 {MIN_VALID_IMAGES} 张；"
            "请补采后再标定"
        )

    standard = calibrate_from_observations(
        object_points,
        image_points,
        EXPECTED_IMAGE_SIZE,
        zero_distortion=False,
    )
    zero = calibrate_from_observations(
        object_points,
        image_points,
        EXPECTED_IMAGE_SIZE,
        zero_distortion=True,
    )
    heldout_standard = heldout_reprojection_errors(
        object_points,
        image_points,
        EXPECTED_IMAGE_SIZE,
        folds=CV_FOLDS,
        seed=RANDOM_SEED,
        zero_distortion=False,
    )
    heldout_zero = heldout_reprojection_errors(
        object_points,
        image_points,
        EXPECTED_IMAGE_SIZE,
        folds=CV_FOLDS,
        seed=RANDOM_SEED,
        zero_distortion=True,
    )
    standard_per_view = np.asarray(standard["per_view_errors"], dtype=float)
    zero_per_view = np.asarray(zero["per_view_errors"], dtype=float)
    heldout_standard_per_view = np.asarray(heldout_standard["per_view_errors"], dtype=float)
    heldout_zero_per_view = np.asarray(heldout_zero["per_view_errors"], dtype=float)
    sampling_coverage = summarize_capture_coverage(
        [record["采集姿态估算"] for record in valid_records],
        minimum_samples=GUIDE_MINIMUM_SAMPLES,
    )
    for index, record in enumerate(valid_records):
        record.update(
            {
                "五参数单图RMS_px": float(standard_per_view[index]),
                "零畸变单图RMS_px": float(zero_per_view[index]),
                "五参数留出RMS_px": float(heldout_standard_per_view[index]),
                "零畸变留出RMS_px": float(heldout_zero_per_view[index]),
            }
        )
    warnings = []
    if len(valid_records) < int(RECOMMENDED_VALID_IMAGES):
        warnings.append(
            f"有效图片 {len(valid_records)} 张，少于建议值 {RECOMMENDED_VALID_IMAGES} 张"
        )
    if not sampling_coverage["ready"]:
        warnings.append(
            "采样覆盖未达到导航默认要求；下一步："
            + sampling_coverage["next_instruction"]
        )
    if not GRID_STEP_IS_PHYSICAL:
        warnings.append("GRID_STEP_MM 尚未标记为实测毫米，K/D 有效，但外参平移尺度不是毫米")
    if standard["rms"] > float(MAX_TOTAL_RMS_PX):
        warnings.append(
            f"总体 RMS={standard['rms']:.4f}px 超过门禁 {MAX_TOTAL_RMS_PX:.4f}px"
        )
    if float(np.max(standard_per_view)) > float(MAX_SINGLE_VIEW_RMS_PX):
        warnings.append(
            f"最差单图 RMS={np.max(standard_per_view):.4f}px 超过门禁 "
            f"{MAX_SINGLE_VIEW_RMS_PX:.4f}px"
        )
    quality_pass = (
        standard["rms"] <= float(MAX_TOTAL_RMS_PX)
        and float(np.max(standard_per_view)) <= float(MAX_SINGLE_VIEW_RMS_PX)
        and len(valid_records) >= int(RECOMMENDED_VALID_IMAGES)
    )
    worst_indices = np.argsort(standard_per_view)[::-1][: min(5, len(valid_records))]
    worst_views = [valid_records[int(index)] for index in worst_indices]
    now = _now()
    calibration_document = {
        "schema_version": 1,
        "generation_time": now.isoformat(timespec="seconds"),
        "calibration_type": "camera_intrinsic_and_distortion",
        "model": PINHOLE_MODEL_NAME,
        "image_source": {
            "topic": CAMERA_TOPIC,
            "coordinate": "camera_node 发布的工程彩色帧",
            "preprocessing": "未在本工具内 resize/crop/undistort",
            "circle_detection_method": DETECTION_METHOD,
            "blob_detector_options": BLOB_DETECTOR_OPTIONS,
        },
        "image_width": int(EXPECTED_IMAGE_SIZE[0]),
        "image_height": int(EXPECTED_IMAGE_SIZE[1]),
        "K": standard["K"].tolist(),
        "D": standard["D"].tolist(),
        "parameters": {
            "fx": float(standard["K"][0, 0]),
            "fy": float(standard["K"][1, 1]),
            "cx": float(standard["K"][0, 2]),
            "cy": float(standard["K"][1, 2]),
            "k1": float(standard["D"][0]),
            "k2": float(standard["D"][1]),
            "p1": float(standard["D"][2]),
            "p2": float(standard["D"][3]),
            "k3": float(standard["D"][4]),
        },
        "board": {
            "type": "asymmetric_circles_grid",
            "pattern_size": list(PATTERN_SIZE),
            "point_count": int(math.prod(PATTERN_SIZE)),
            "grid_step_mm": float(GRID_STEP_MM),
            "grid_step_is_physical_mm": bool(GRID_STEP_IS_PHYSICAL),
            "object_point_definition": "x=(2*j+i%2)*s, y=i*s, z=0，按行优先",
        },
        "metrics": {
            "valid_image_count": len(valid_records),
            "failed_image_count": len(failed_records),
            "standard_model": _calibration_summary(standard),
            "zero_distortion_baseline": _calibration_summary(zero),
            "heldout_standard": to_builtin(heldout_standard),
            "heldout_zero_distortion": to_builtin(heldout_zero),
            "sampling_coverage": to_builtin(sampling_coverage),
            "heldout_standard_improvement_over_zero_ratio": float(
                1.0 - heldout_standard["rmse"] / max(heldout_zero["rmse"], 1e-12)
            ),
        },
        "samples": valid_records,
        "failed_samples": failed_records,
        "worst_views": worst_views,
        "references": {
            "vendor_profile": _load_vendor_profile(VENDOR_PROFILE_JSON),
            "old_project_calibration": _load_reference_calibration(OLD_PROJECT_CALIBRATION),
        },
        "quality_thresholds": {
            "minimum_recommended_images": int(RECOMMENDED_VALID_IMAGES),
            "maximum_total_rms_px": float(MAX_TOTAL_RMS_PX),
            "maximum_single_view_rms_px": float(MAX_SINGLE_VIEW_RMS_PX),
        },
        "quality_pass": bool(quality_pass),
        "warnings": warnings,
        "usage_warning": "首轮仅供独立诊断；不得直接送入旧 pixel→TCP 或视觉伺服矩阵",
    }
    yaml_path = output_dir / "相机标定.yaml"
    yaml_path.write_text(
        yaml.safe_dump(to_builtin(calibration_document), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    report_path = output_dir / "标定质量报告.json"
    report_path.write_text(
        json.dumps(to_builtin(calibration_document), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"有效图片：{len(valid_records)}，失败图片：{len(failed_records)}")
    print(f"五参数总体 RMS：{standard['rms']:.6f} px")
    print(f"五参数留出 RMS：{heldout_standard['rmse']:.6f} px")
    print(f"零畸变留出 RMS：{heldout_zero['rmse']:.6f} px")
    print("最差图片：")
    for item in worst_views:
        print(f"  {item['文件']}：{item['五参数单图RMS_px']:.6f} px")
    print(f"质量门禁：{'通过' if quality_pass else '未通过'}")
    for warning in warnings:
        print(f"警告：{warning}")
    print(f"参数文件：{yaml_path}")


def load_calibration(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if data.get("model") != PINHOLE_MODEL_NAME:
        raise ValueError(f"不支持的模型: {data.get('model')}")
    k = np.asarray(data.get("K"), dtype=float).reshape(3, 3)
    d = np.asarray(data.get("D"), dtype=float).reshape(-1)
    size = (int(data["image_width"]), int(data["image_height"]))
    if len(d) != 5 or not np.all(np.isfinite(k)) or not np.all(np.isfinite(d)):
        raise ValueError("标定文件中的 K/D 无效")
    return data, k, d, size


def correct_image_same_pixel_frame(image, k, d):
    """用 P=K 去畸变，保持像素坐标定义不额外改变。"""
    height, width = image.shape[:2]
    map_x, map_y = cv2.initUndistortRectifyMap(
        k,
        d,
        np.eye(3, dtype=float),
        k,
        (width, height),
        cv2.CV_32FC1,
    )
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _print_pixel_corrections(k, d):
    original = np.asarray(PIXELS_TO_CHECK, dtype=float).reshape(-1, 2)
    corrected = undistort_pixel_coordinates(original, k, d)
    print("\n单点像素畸变修正量（P=K，同一像素坐标系）")
    print("原始(u,v) -> 理想(u',v') | Δu, Δv, 总修正量")
    for source, target in zip(original, corrected):
        delta = target - source
        print(
            f"({source[0]:9.3f},{source[1]:9.3f}) -> "
            f"({target[0]:9.3f},{target[1]:9.3f}) | "
            f"{delta[0]:+8.3f}, {delta[1]:+8.3f}, {np.linalg.norm(delta):8.3f} px"
        )


def run_verify():
    """加载已有 K/D，对文件或实时帧做独立验证。"""
    _data, k, d, calibrated_size = load_calibration(VERIFY_CALIBRATION_YAML)
    if calibrated_size != tuple(EXPECTED_IMAGE_SIZE):
        raise RuntimeError(
            f"标定文件尺寸 {calibrated_size} 与当前期望 {EXPECTED_IMAGE_SIZE} 不一致"
        )
    _print_pixel_corrections(k, d)
    paths = [Path(path) for path in VERIFY_IMAGE_PATHS]
    if paths:
        for path in paths:
            image = read_image(path)
            validate_image_size(image, calibrated_size)
            corrected = correct_image_same_pixel_frame(image, k, d)
            comparison = np.hstack([image, corrected])
            put_chinese_text(comparison, "原始工程图像", (15, 5), (0, 255, 255), 23)
            put_chinese_text(
                comparison,
                "去畸变图像（P=K）",
                (image.shape[1] + 15, 5),
                (0, 255, 255),
                23,
            )
            _display(comparison, "verify: any key next, q quit")
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
        return

    import rospy

    rospy.init_node("camera_undistortion_verify", anonymous=True)
    print("实时验证：q 退出。左侧原图，右侧 P=K 去畸变图。")
    while not rospy.is_shutdown():
        image = _wait_ros_image()
        validate_image_size(image, calibrated_size)
        corrected = correct_image_same_pixel_frame(image, k, d)
        comparison = np.hstack([image, corrected])
        put_chinese_text(comparison, "原始工程图像", (15, 5), (0, 255, 255), 23)
        put_chinese_text(
            comparison,
            "去畸变图像（P=K）",
            (image.shape[1] + 15, 5),
            (0, 255, 255),
            23,
        )
        _display(comparison, "verify: any key next, q quit")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def main():
    if DETECTION_METHOD not in CIRCLE_DETECTION_METHODS:
        raise ValueError(
            f"DETECTION_METHOD={DETECTION_METHOD!r} 无效；"
            f"可选值：{', '.join(CIRCLE_DETECTION_METHODS)}"
        )
    modes = {
        "board_check": run_board_check,
        "capture": run_capture,
        "calibrate": run_calibrate,
        "verify": run_verify,
    }
    if MODE not in modes:
        raise ValueError(f"未知 MODE={MODE!r}，可选值：{', '.join(modes)}")
    modes[MODE]()


if __name__ == "__main__":
    main()
