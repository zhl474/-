#!/home/zhl/fr3env/fr3env/bin/python
"""ArUco 相机画面探针：实时显示相机画面，诊断标记能否被识别。

用途：
- 手动把机械臂摆到任意姿态（例如低位观察位），运行本工具查看相机到底看到了什么。
- 回车保存当前帧并打印多字典检测诊断；输入 q 退出。

不移动机械臂，只订阅相机话题；参数集中在文件开头。
"""

import sys
from datetime import datetime
from pathlib import Path

import cv2
import rospy

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aruco_diagnostic_core import (  # noqa: E402
    create_aruco_detector,
    detect_all_markers,
    detect_aruco_center,
    draw_servo_overlay,
    put_chinese_text,
)
from run_aruco_experiment import FreshImageReader  # noqa: E402

# ==================== 运行参数（直接修改本文件后运行）====================
IMAGE_TOPIC = "/camera/image_rect"  # 去畸变相机图像话题
IMAGE_TIMEOUT_SEC = 1.0  # 单次等待新相机帧的超时时间（秒）
OUTPUT_DIR = Path("/home/zhl/桌面/aruco探针")  # 回车保存的图片输出目录
DICT_NAMES = (  # 多字典诊断列表（第一个是正式检测字典）
    "DICT_6X6_50", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_5X5_50", "DICT_4X4_50", "DICT_ARUCO_ORIGINAL",
)
TARGET_MARKER_ID = 0  # 实时叠加只精定位这个 ID 的中央棋盘角点
RELAXED_MIN_PERIMETER_RATE = 0.01  # 放宽模式：更小的标记最小周长比例
WINDOW_NAME = "aruco_probe"  # 实时窗口名

FORMAL_DETECTOR = create_aruco_detector(DICT_NAMES[0])


def _print_diagnosis(image, label):
    """对当前帧执行标准与放宽两轮多字典检测并打印。"""
    height, width = image.shape[:2]
    print(f"\n===== {label}（{width}x{height}）=====")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(
        f"亮度 min={gray.min()} max={gray.max()} mean={gray.mean():.1f}"
        f" std={gray.std():.1f}"
    )
    for corner_refine, min_perimeter_rate, mode in (
        (True, None, "标准"),
        (False, RELAXED_MIN_PERIMETER_RATE, "放宽"),
    ):
        print(f"--- {mode}参数（细化={corner_refine}，minPerimeterRate={min_perimeter_rate}）---")
        results = detect_all_markers(
            image, DICT_NAMES,
            corner_refine=corner_refine, min_perimeter_rate=min_perimeter_rate,
        )
        for dict_name, found in results.items():
            if not found["ids"] and not found["rejected_count"]:
                continue
            if found["ids"]:
                details = ", ".join(
                    f"ID={marker_id} 尺寸≈{size:.0f}px"
                    for marker_id, size in zip(found["ids"], found["sizes_px"])
                )
                print(f"  {dict_name}: 检出 {len(found['ids'])} 个 -> {details}")
            if found["rejected_count"]:
                print(f"  {dict_name}: rejected 候选 {found['rejected_count']} 个（找到四边形但解码失败）")
    print("=======================================================")


def _draw_overlay(image):
    """叠加绿色粗外框、青色粗中心和红色中央精中心。"""
    height, width = image.shape[:2]
    result = detect_aruco_center(image, FORMAL_DETECTOR, TARGET_MARKER_ID)
    canvas = draw_servo_overlay(
        image,
        round_no="探针",
        error_xy=(result.dx_px, result.dy_px) if result.found else None,
        center_uv=result.center_uv if result.found else None,
        rough_center_uv=result.rough_center_uv,
        refine_delta_uv=result.refine_delta_uv,
        center_contrast=result.center_contrast,
        corners=result.corners,
        rejected_corners=result.rejected_corners,
    )
    status = f"ID={TARGET_MARKER_ID} OK" if result.found else result.message
    put_chinese_text(
        canvas, status,
        (10, max(0, height - 58)),
        (0, 200, 0) if result.found else (0, 0, 255), font_size=20,
    )
    put_chinese_text(
        canvas, "Enter：保存并诊断　q：退出",
        (10, max(0, height - 30)), (0, 200, 0), font_size=20,
    )
    return canvas


def main():
    rospy.init_node("aruco_frame_probe", anonymous=True)
    reader = FreshImageReader(IMAGE_TOPIC)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("ArUco 相机画面探针启动")
    print("操作：回车 = 保存当前帧并打印诊断；q = 退出")
    print("提示：先把机械臂手动摆到低位观察位，观察画面中是否出现绿色四边形。")

    while not rospy.is_shutdown():
        image, error = reader.fresh_image(IMAGE_TIMEOUT_SEC)
        if image is None:
            print(f"图像获取失败: {error}")
            continue
        overlay = _draw_overlay(image)
        cv2.imshow(WINDOW_NAME, overlay)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        if key == 13:  # Enter
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            original_path = OUTPUT_DIR / f"{stamp}_原图.png"
            overlay_path = OUTPUT_DIR / f"{stamp}_调试图.png"
            cv2.imwrite(str(original_path), image)
            cv2.imwrite(str(overlay_path), overlay)
            print(f"已保存: {original_path}")
            _print_diagnosis(image, stamp)

    cv2.destroyAllWindows()
    print("探针退出")


if __name__ == "__main__":
    main()
