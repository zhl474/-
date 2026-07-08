#!/home/zhl/fr3env/fr3env/bin/python
import argparse
import os
import sys
from statistics import median

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

from image_process_lib.block_detection import get_mask
from image_process_lib.template_config import TEMPLATE_CONFIG_PATH


DETECT_MODEL_PATH = "/home/zhl/SingleArmTetris/SingleArmTetris/src/competition/model/best5.14.pt"
CALIBRATION_MATRIX_PATH = "/home/zhl/SingleArmTetris/SingleArmTetris/src/competition/config/calibration_matrix.yaml"
DEFAULT_VIS_DIR = "/home/zhl/SingleArmTetris/SingleArmTetris/src/competition/config/template_size_vis"
DEFAULT_CAMERA_TOPIC = "/camera/image_raw"

CATEGORY_NAME_MAP = {
    "LR": "L_blue",
    "LL": "L_yellow",
    "ZL": "z_blue",
    "ZR": "z_green",
    "O": "square",
    "suqare": "square",
    "Line": "line",
}

CATEGORY_ORDER = ["line", "square", "L_yellow", "L_blue", "z_blue", "z_green", "T"]

def parse_args():
    parser = argparse.ArgumentParser(
        description="测量俄罗斯方块上表面外接矩形，并打印模板几何参数建议值；不传参数时默认使用相机获取一帧图像。"
    )
    parser.add_argument("--image", action="append", default=[], help="输入图片路径，可重复传入多张")
    parser.add_argument("--camera", action="store_true", help="从 ROS 相机话题读取图像；不传 --image 时默认启用")
    parser.add_argument("--topic", default=DEFAULT_CAMERA_TOPIC, help="ROS 图像话题")
    parser.add_argument("--frames", type=int, default=1, help="从相机读取的帧数，默认 1 帧")
    parser.add_argument("--timeout", type=float, default=30.0, help="等待相机帧超时时间")
    parser.add_argument("--detect-model", default=DETECT_MODEL_PATH, help="方块检测模型路径")
    parser.add_argument("--output", default=TEMPLATE_CONFIG_PATH, help="仅用于提示配置文件路径，不会自动写入")
    parser.add_argument("--vis-dir", default=DEFAULT_VIS_DIR, help="可视化图片保存目录")
    parser.add_argument("--save-vis", action="store_true", help="保存每次测量的可视化图片，默认不保存")
    parser.add_argument("--no-window", action="store_true", help="不弹出 OpenCV 窗口，改用终端确认")
    parser.add_argument("--auto-accept", action="store_true", help="自动接受所有有效测量")
    parser.add_argument("--accumulate", action="store_true", help="兼容旧参数；当前脚本只统计本次接受测量")
    parser.add_argument("--category", action="append", choices=CATEGORY_ORDER, help="只标定指定类别，可重复传入")
    parser.add_argument("--detect-conf", type=float, default=0.45, help="检测置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="检测 NMS IoU 阈值")
    parser.add_argument("--crop-margin", type=int, default=8, help="检测框裁剪外扩像素")
    parser.add_argument("--mask-thresh", type=int, default=127, help="mask 二值化阈值")
    parser.add_argument("--min-area", type=float, default=50.0, help="最大轮廓最小面积")
    parser.add_argument("--calibration", default=CALIBRATION_MATRIX_PATH, help="相机内参配置路径")
    parser.add_argument("--no-undistort", action="store_true", help="不做去畸变，直接处理原图")
    return parser.parse_args()


def normalize_category(category):
    return CATEGORY_NAME_MAP.get(category, category)


def load_camera_params(calibration_path):
    if not os.path.exists(calibration_path):
        print(f"提示：未找到相机内参配置，跳过去畸变: {calibration_path}")
        return None

    with open(calibration_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return np.array(data["camera_matrix"]), np.array(data["dist_coeff"])


def undistort_image(img_bgr, camera_params):
    """Gemini335 图像直接参与标定测量，不再额外去畸变。"""
    return img_bgr


def iter_image_inputs(image_paths):
    for image_path in image_paths:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"警告：无法读取图片，跳过: {image_path}")
            continue
        yield img_bgr, os.path.basename(image_path)


def iter_camera_inputs(topic, frames, timeout):
    import rospy
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image

    rospy.init_node("template_size_calibration", anonymous=True)
    bridge = CvBridge()
    for frame_idx in range(frames):
        print(f"等待相机帧 {frame_idx + 1}/{frames}: {topic}")
        msg = rospy.wait_for_message(topic, Image, timeout=timeout)
        yield bridge.imgmsg_to_cv2(msg, "bgr8"), f"camera_{frame_idx + 1}"


def threshold_and_measure(mask, mask_thresh, min_area):
    _, binary = cv2.threshold(mask, mask_thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("mask 中没有轮廓")

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < min_area:
        raise RuntimeError(f"最大轮廓面积过小: {area:.1f}")

    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    if width <= 0 or height <= 0:
        raise RuntimeError("最小外接矩形尺寸无效")

    long_side = max(width, height)
    short_side = min(width, height)
    return binary, contour, rect, float(long_side), float(short_side), float(area)


def draw_measurement_visual(frame, mask, rect, crop_box, category, score, long_side, short_side):
    x1, y1, x2, y2 = crop_box
    visual = frame.copy()

    roi = visual[y1:y2, x1:x2]
    mask_bool = mask > 0
    colored_roi = roi.copy()
    colored_roi[mask_bool] = (0, 255, 0)
    roi[:] = cv2.addWeighted(roi, 0.65, colored_roi, 0.35, 0)

    box = cv2.boxPoints(rect)
    box = np.intp(box)
    box[:, 0] += x1
    box[:, 1] += y1
    cv2.drawContours(visual, [box], 0, (0, 0, 255), 2)
    cv2.rectangle(visual, (x1, y1), (x2, y2), (255, 0, 0), 2)

    text_lines = [
        f"类别: {category}",
        f"置信度: {score:.3f}",
        f"长边: {long_side:.1f}px",
        f"短边: {short_side:.1f}px",
        "回车/y: 接受, n/空格: 跳过, q: 退出",
    ]
    text_x = max(10, min(x1, visual.shape[1] - 430))
    text_y = max(10, y1 - 150)
    cv2.rectangle(visual, (text_x - 8, text_y - 8), (text_x + 430, text_y + 150), (0, 0, 0), -1)
    for line_idx, text in enumerate(text_lines):
        cv2.putText(
            visual,
            text,
            (text_x, text_y + line_idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return visual


def ask_user_to_accept(visual, args, category, source_name, det_idx):
    vis_path = None
    if args.save_vis:
        os.makedirs(args.vis_dir, exist_ok=True)
        stem = os.path.splitext(source_name)[0]
        vis_path = os.path.join(args.vis_dir, f"{stem}_{det_idx}_{category}.jpg")
        cv2.imwrite(vis_path, visual)
        print(f"已保存可视化图片: {vis_path}")

    if args.auto_accept:
        return "accept"

    prompt = "接受当前测量？[y/回车=接受, n/空格=跳过, q=退出] "
    if args.no_window:
        answer = input(prompt).strip().lower()
        if answer in ("", "y", "yes"):
            return "accept"
        if answer == "q":
            return "quit"
        return "skip"

    try:
        cv2.imshow("template_size_calibration", visual)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("template_size_calibration")
    except cv2.error:
        answer = input(prompt).strip().lower()
        if answer in ("", "y", "yes"):
            return "accept"
        if answer == "q":
            return "quit"
        return "skip"

    if key in (13, ord("y"), ord("Y")):
        return "accept"
    if key in (ord("q"), ord("Q")):
        return "quit"
    return "skip"


def init_measurements():
    return {category: [] for category in CATEGORY_ORDER}


def estimate_geometry_from_measurement(category, long_side, short_side):
    """从单次外接矩形测量推导模板几何参数。"""
    if category in ("L_yellow", "L_blue", "z_blue", "z_green", "T"):
        block_px = 2.0 * short_side - long_side
        connector_px = 2.0 * long_side - 3.0 * short_side
    elif category == "line":
        block_px = short_side
        connector_px = (long_side - 4.0 * block_px) / 3.0
    else:
        return None

    if block_px <= 0 or connector_px <= 0:
        return None

    return {
        "category": category,
        "block_px": float(block_px),
        "connector_px": float(connector_px),
    }


def collect_geometry_estimates(measurements):
    estimates = []
    for category, category_measurements in measurements.items():
        for measurement in category_measurements:
            estimate = estimate_geometry_from_measurement(
                category,
                float(measurement["long_side"]),
                float(measurement["short_side"]),
            )
            if estimate is not None:
                estimates.append(estimate)
    return estimates


def process_frame(frame, source_name, detect_model, camera_params, measurements, args):
    frame = undistort_image(frame, camera_params)
    h, w = frame.shape[:2]
    result = detect_model(frame, iou=args.iou, conf=args.detect_conf, verbose=False)
    if not args.no_window:
        cv2.imshow("检测结果", result[0].plot())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if len(result) == 0 or result[0].boxes is None or len(result[0].boxes) == 0:
        print(f"提示：{source_name} 没有检测到方块")
        return True

    for det_idx, det in enumerate(result[0].boxes.data.tolist()):
        x1, y1, x2, y2, score, cid = det
        category = normalize_category(detect_model.names[int(cid)])
        if category == "board":
            continue
        if args.category and category not in args.category:
            continue

        crop_x1 = max(0, int(x1) - args.crop_margin)
        crop_y1 = max(0, int(y1) - args.crop_margin)
        crop_x2 = min(w, int(x2) + args.crop_margin)
        crop_y2 = min(h, int(y2) + args.crop_margin)
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            print(f"警告：裁剪区域无效，跳过 {category}")
            continue

        cropped_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        try:
            mask, _ = get_mask(cropped_img, category)
            binary, _, rect, long_side, short_side, area = threshold_and_measure(
                mask,
                args.mask_thresh,
                args.min_area,
            )
        except Exception as exc:
            print(f"警告：{source_name} 的 {category} 测量失败: {exc}")
            continue

        visual = draw_measurement_visual(
            frame,
            binary,
            rect,
            (crop_x1, crop_y1, crop_x2, crop_y2),
            category,
            float(score),
            long_side,
            short_side,
        )
        action = ask_user_to_accept(visual, args, category, source_name, det_idx)
        if action == "quit":
            return False
        if action == "skip":
            print(f"跳过：{category} 长边={long_side:.1f}px 短边={short_side:.1f}px")
            continue

        measurements.setdefault(category, [])
        measurements[category].append(
            {
                "long_side": long_side,
                "short_side": short_side,
                "score": float(score),
                "source": source_name,
                "is_new": True,
            }
        )
        estimate = estimate_geometry_from_measurement(category, long_side, short_side)
        if estimate is None:
            print(f"接受：{category} 长边={long_side:.1f}px 短边={short_side:.1f}px 面积={area:.1f}")
        else:
            print(
                f"接受：{category} 长边={long_side:.1f}px 短边={short_side:.1f}px 面积={area:.1f}，"
                f"建议子块={estimate['block_px']:.1f}px 连接处={estimate['connector_px']:.1f}px"
            )

    return True


def print_measurement_group(title, measurements):
    print(f"\n{title}：")
    has_measurement = False
    for category in CATEGORY_ORDER:
        category_measurements = measurements.get(category, [])
        if not category_measurements:
            continue
        has_measurement = True
        longs = [m["long_side"] for m in category_measurements]
        shorts = [m["short_side"] for m in category_measurements]
        print(
            f"{category}: 长边={median(longs):.1f}px, "
            f"短边={median(shorts):.1f}px, 样本数={len(category_measurements)}"
        )
    if not has_measurement:
        print("无")


def print_geometry_estimates(estimates):
    print("\n几何参数推导：")
    if not estimates:
        print("无可用推导结果；至少需要 line 或 L/T/Z 类别的有效测量")
        return None

    for category in CATEGORY_ORDER:
        category_estimates = [item for item in estimates if item["category"] == category]
        if not category_estimates:
            continue
        block_values = [item["block_px"] for item in category_estimates]
        connector_values = [item["connector_px"] for item in category_estimates]
        print(
            f"{category}: 子块={median(block_values):.1f}px, "
            f"连接处={median(connector_values):.1f}px, 样本数={len(category_estimates)}"
        )

    block_px = median([item["block_px"] for item in estimates])
    connector_px = median([item["connector_px"] for item in estimates])
    print(f"\n建议值 median：子块={block_px:.1f}px，连接处={connector_px:.1f}px")
    return block_px, connector_px


def print_square_check(measurements, suggested_geometry):
    square_measurements = measurements.get("square", [])
    if not square_measurements or suggested_geometry is None:
        return

    block_px, connector_px = suggested_geometry
    expected_side = 2.0 * block_px + connector_px
    longs = [m["long_side"] for m in square_measurements]
    shorts = [m["short_side"] for m in square_measurements]
    measured_side = median([(long_side + short_side) / 2.0 for long_side, short_side in zip(longs, shorts)])
    print(
        f"square 一致性检查：实测边长约 {measured_side:.1f}px，"
        f"建议几何边长 {expected_side:.1f}px，偏差 {measured_side - expected_side:.1f}px"
    )


def print_yaml_snippet(suggested_geometry, profile_name="high"):
    if suggested_geometry is None:
        return

    block_px, connector_px = suggested_geometry
    block_px = int(round(block_px))
    connector_px = int(round(connector_px))
    snippet = {
        "template_sizes": {
            "active_profile": profile_name,
            "profiles": {
                profile_name: {
                    "block_px": block_px,
                    "connector_px": connector_px,
                },
                "low": {
                    "block_px": block_px,
                    "connector_px": connector_px,
                },
            },
        },
    }

    print("\n可复制到配置文件的 YAML 片段：")
    print(yaml.safe_dump(snippet, allow_unicode=True, sort_keys=False).rstrip())


def print_summary(measurements):
    print_measurement_group("本次接受测量", measurements)
    estimates = collect_geometry_estimates(measurements)
    suggested_geometry = print_geometry_estimates(estimates)
    print_square_check(measurements, suggested_geometry)
    print_yaml_snippet(suggested_geometry)


def main():
    args = parse_args()
    camera_params = None
    measurements = init_measurements()
    if args.accumulate:
        print("提示：当前脚本不再读取历史 measurements，只统计本次接受测量。")

    detect_model = YOLO(args.detect_model)
    input_iterators = [iter_image_inputs(args.image)]
    use_camera = args.camera or len(args.image) == 0#不传参数时默认使用相机获取一帧图像
    if use_camera:
        input_iterators.append(iter_camera_inputs(args.topic, args.frames, args.timeout))

    has_input = False
    should_continue = True
    for input_iterator in input_iterators:
        for frame, source_name in input_iterator:
            has_input = True
            should_continue = process_frame(
                frame,
                source_name,
                detect_model,
                camera_params,
                measurements,
                args,
            )
            if not should_continue:
                break
        if not should_continue:
            break

    if not has_input:
        print("没有可处理的输入图像")
        return

    new_accepted_count = sum(
        1
        for category_measurements in measurements.values()
        for measurement in category_measurements
        if measurement.get("is_new", False)
    )
    if new_accepted_count == 0:
        print("没有接受任何新测量，未输出模板几何建议")
        return

    print_summary(measurements)
    print(f"\n配置文件路径：{args.output}")
    print("提示：脚本不会自动修改配置文件，请按需要手动复制上述 YAML 片段。")


if __name__ == "__main__":
    main()
