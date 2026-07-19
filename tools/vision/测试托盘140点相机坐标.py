#!/home/zhl/fr3env/fr3env/bin/python
"""直接检查托盘 140 个格点从像素到相机坐标的转换结果。

本脚本只使用相机 SDK 的内参与深度图，不启动 ROS，不读取机械臂位姿，
也不加载或使用手眼标定矩阵。运行前请停止占用 Gemini335 的 camera_node。
"""

import csv
import os
import sys
from datetime import datetime

import cv2
import numpy as np


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMAGE_PROCESS_DIR = os.path.join(SRC_DIR, "image_process")
if IMAGE_PROCESS_DIR not in sys.path:
    sys.path.insert(0, IMAGE_PROCESS_DIR)

from akai_gemini335 import AkaiGemini335
from image_process_lib.board_scene_detector import (
    BOARD_COL_COUNT,
    BOARD_POINT_COUNT,
    BOARD_ROW_COUNT,
    _put_chinese_text,
    board_grid_detect,
)


# ========================= 直接修改的运行参数 =========================
# 相机参数与 camera_node 使用同一份 YAML，但本脚本不使用任何手眼矩阵。
CAMERA_CONFIG_PATH = os.path.join(SRC_DIR, "camera", "config", "新相机参数.yaml")
# 初始化后先丢弃若干帧，使自动曝光和深度流稳定。
WARMUP_FRAME_COUNT = 10
# CSV 和标注图的输出目录；每次运行会生成带时间戳的新文件。
OUTPUT_DIR = os.path.expanduser("~/桌面/托盘相机坐标测试")
# 是否分别保存坐标表和全部格点标注图。
SAVE_CSV = True
SAVE_ANNOTATED_IMAGE = True


def save_image(image_path, image):
    """保存图片，兼容中文目录和文件名。"""
    if image is None:
        return False
    image_dir = os.path.dirname(image_path)
    if image_dir:
        os.makedirs(image_dir, exist_ok=True)
    extension = os.path.splitext(image_path)[1] or ".jpg"
    ok, encoded_image = cv2.imencode(extension, image)
    if not ok:
        return False
    encoded_image.tofile(image_path)
    return True


def read_stable_frame(camera):
    """预热后取得一组同时返回的彩色图和对齐深度图。"""
    if WARMUP_FRAME_COUNT < 0:
        raise ValueError("WARMUP_FRAME_COUNT 不能小于 0")

    for frame_index in range(WARMUP_FRAME_COUNT):
        color_image, depth_image = camera.read()
        if color_image is None or depth_image is None:
            print(f"预热帧 {frame_index + 1}/{WARMUP_FRAME_COUNT} 读取失败，继续等待下一帧")

    color_image, depth_image = camera.read()
    if color_image is None or depth_image is None:
        raise RuntimeError("相机读取失败，未取得可用的彩色图和深度图")
    if color_image.ndim != 3 or color_image.shape[2] != 3:
        raise RuntimeError(f"彩色图格式异常: shape={color_image.shape}")
    if depth_image.ndim < 2:
        raise RuntimeError(f"深度图格式异常: shape={depth_image.shape}")
    if color_image.shape[:2] != depth_image.shape[:2]:
        raise RuntimeError(
            "彩色图与深度图尺寸不一致，无法确认像素对齐："
            f"彩色图={color_image.shape[:2]}，深度图={depth_image.shape[:2]}"
        )
    return color_image, depth_image


def make_record(row, col, pixel_xy, depth_image, camera):
    """把一个浮点格点像素转换为相机坐标，并保留完整诊断信息。"""
    pixel_x, pixel_y = float(pixel_xy[0]), float(pixel_xy[1])
    # 与正式表面高度服务保持一致：先四舍五入到整数深度像素。
    sample_x = int(round(pixel_x))
    sample_y = int(round(pixel_y))
    image_height, image_width = depth_image.shape[:2]
    record = {
        "row": row,
        "col": col,
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "sample_x": sample_x,
        "sample_y": sample_y,
        "depth_mm": None,
        "camera_x_mm": None,
        "camera_y_mm": None,
        "camera_z_mm": None,
        "valid": False,
        "message": "",
    }

    if not (0 <= sample_x < image_width and 0 <= sample_y < image_height):
        record["message"] = f"深度采样像素越界，图像尺寸=({image_width},{image_height})"
        return record

    depth_value = float(depth_image[sample_y, sample_x])
    record["depth_mm"] = depth_value
    if not np.isfinite(depth_value) or depth_value <= 0.0:
        record["message"] = "深度无效"
        return record

    try:
        # 只调用 SDK 的像素+深度转相机坐标接口；此处绝不做手眼或世界坐标变换。
        camera_point = np.asarray(
            camera.depth_pixel2cam_point3d(sample_x, sample_y, depth_value=depth_value),
            dtype=float,
        ).reshape(-1)
        if camera_point.size != 3 or not np.all(np.isfinite(camera_point)):
            raise ValueError(f"SDK 返回无效相机坐标: {camera_point.tolist()}")
        record["camera_x_mm"] = float(camera_point[0])
        record["camera_y_mm"] = float(camera_point[1])
        record["camera_z_mm"] = float(camera_point[2])
        record["valid"] = True
        record["message"] = "成功"
    except Exception as exc:
        record["message"] = f"相机坐标转换失败: {exc}"
    return record


def convert_all_grid_points(grid_points, depth_image, camera):
    """按固定的 1-based 行列约定转换全部 140 个托盘格点。"""
    records = []
    for row in range(1, BOARD_ROW_COUNT + 1):
        for col in range(1, BOARD_COL_COUNT + 1):
            point = grid_points[row][col]
            if point is None:
                raise RuntimeError(f"托盘格点缺失: 行={row}，列={col}")
            records.append(make_record(row, col, point, depth_image, camera))
    if len(records) != BOARD_POINT_COUNT:
        raise RuntimeError(f"格点转换数量异常: 期望 {BOARD_POINT_COUNT}，实际 {len(records)}")
    return records


def print_records(records):
    """在终端逐行打印像素、深度和原始相机坐标。"""
    print("\n托盘格点像素到相机坐标结果（单位：像素、毫米）")
    print("行 列 | 原始像素(x,y) | 深度采样像素(x,y) | 深度 | 相机坐标(X,Y,Z) | 状态")
    print("-" * 112)
    for record in records:
        pixel_text = f"({record['pixel_x']:8.2f},{record['pixel_y']:8.2f})"
        sample_text = f"({record['sample_x']:4d},{record['sample_y']:4d})"
        depth_text = "       -" if record["depth_mm"] is None else f"{record['depth_mm']:8.2f}"
        if record["valid"]:
            camera_text = (
                f"({record['camera_x_mm']:8.2f},"
                f"{record['camera_y_mm']:8.2f},{record['camera_z_mm']:8.2f})"
            )
        else:
            camera_text = "(       -,       -,       -)"
        print(
            f"{record['row']:2d} {record['col']:2d} | {pixel_text} | {sample_text} | "
            f"{depth_text} | {camera_text} | {record['message']}"
        )


def print_summary(records):
    """输出有效深度数量与相机坐标范围，方便判断托盘平面是否稳定。"""
    valid_records = [record for record in records if record["valid"]]
    invalid_records = [record for record in records if not record["valid"]]
    print("\n汇总")
    print(f"总格点数：{len(records)}，有效相机坐标点：{len(valid_records)}，无效点：{len(invalid_records)}")
    if not valid_records:
        print("没有有效相机坐标，无法统计 XYZ 范围。")
        return

    coordinates = np.array(
        [[record["camera_x_mm"], record["camera_y_mm"], record["camera_z_mm"]] for record in valid_records],
        dtype=float,
    )
    axis_names = ("X", "Y", "Z")
    for axis_index, axis_name in enumerate(axis_names):
        values = coordinates[:, axis_index]
        print(
            f"相机坐标 {axis_name} 范围："
            f"最小={np.min(values):.2f} mm，最大={np.max(values):.2f} mm，"
            f"跨度={np.ptp(values):.2f} mm"
        )


def save_records_csv(csv_path, records):
    """保存中文表头 CSV，供 Excel 或后续脚本直接比较。"""
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "行",
                "列",
                "原始像素X",
                "原始像素Y",
                "深度采样像素X",
                "深度采样像素Y",
                "深度值毫米",
                "相机X毫米",
                "相机Y毫米",
                "相机Z毫米",
                "是否有效",
                "状态",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record["row"],
                    record["col"],
                    f"{record['pixel_x']:.6f}",
                    f"{record['pixel_y']:.6f}",
                    record["sample_x"],
                    record["sample_y"],
                    _format_csv_number(record["depth_mm"]),
                    _format_csv_number(record["camera_x_mm"]),
                    _format_csv_number(record["camera_y_mm"]),
                    _format_csv_number(record["camera_z_mm"]),
                    "是" if record["valid"] else "否",
                    record["message"],
                ]
            )


def _format_csv_number(value):
    """把缺失数值写为空白单元格，避免 CSV 出现字符串 None。"""
    return "" if value is None else f"{float(value):.6f}"


def draw_annotated_image(color_image, records):
    """绘制全部格点行列号；绿色有效，红色代表深度或转换无效。"""
    image = color_image.copy()
    for record in records:
        center = (int(round(record["pixel_x"])), int(round(record["pixel_y"])))
        color = (0, 255, 0) if record["valid"] else (0, 0, 255)
        cv2.circle(image, center, 4, color, -1)
        # 标签格式为“行,列”，140 个点均有标签，避免依赖 OpenCV 的中文字体支持。
        cv2.putText(
            image,
            f"{record['row']},{record['col']}",
            (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            color,
            1,
            cv2.LINE_AA,
        )
    _put_chinese_text(image, "托盘140个格点：绿色有效，红色深度或相机坐标无效", (15, 30), (0, 255, 255), 22)
    return image


def main():
    """执行一次取图、托盘识别和 140 点相机坐标诊断。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    camera = None
    try:
        print("正在连接 Gemini335；请确认 camera_node 已停止，避免相机被占用。")
        camera = AkaiGemini335(yaml_path=CAMERA_CONFIG_PATH)
        color_image, depth_image = read_stable_frame(camera)
        print(f"已获取同帧彩色图和深度图，图像尺寸：{color_image.shape[1]}x{color_image.shape[0]}")

        detection = board_grid_detect(color_image, debug_path=None)
        if not detection["found"]:
            debug_image = detection.get("debug_image")
            if debug_image is None:
                debug_image = color_image
            debug_path = os.path.join(OUTPUT_DIR, f"{run_name}_托盘识别失败.jpg")
            if save_image(debug_path, debug_image):
                print(f"托盘识别调试图已保存：{debug_path}")
            raise RuntimeError(detection["message"])

        records = convert_all_grid_points(detection["grid_points"], depth_image, camera)
        print_records(records)
        print_summary(records)

        if SAVE_CSV:
            csv_path = os.path.join(OUTPUT_DIR, f"{run_name}_托盘140点相机坐标.csv")
            save_records_csv(csv_path, records)
            print(f"CSV 结果已保存：{csv_path}")
        if SAVE_ANNOTATED_IMAGE:
            image_path = os.path.join(OUTPUT_DIR, f"{run_name}_托盘140点相机坐标标注.jpg")
            annotated_image = draw_annotated_image(color_image, records)
            if not save_image(image_path, annotated_image):
                raise RuntimeError(f"标注图保存失败: {image_path}")
            print(f"格点标注图已保存：{image_path}")

        print("\n完成：以上 XYZ 为相机 SDK 原始相机坐标，未经过任何手眼或世界坐标变换。")
    finally:
        if camera is not None:
            camera.release()
            print("相机已释放。")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\n测试失败：{error}")
        raise SystemExit(1)
