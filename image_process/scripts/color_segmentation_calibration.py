#!/home/zhl/fr3env/fr3env/bin/python
import os
import sys

import cv2
import numpy as np
import yaml


IMAGE_PROCESS_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if IMAGE_PROCESS_SRC_DIR not in sys.path:
    sys.path.insert(0, IMAGE_PROCESS_SRC_DIR)

from image_process_lib.template_config import (
    TEMPLATE_CONFIG_PATH,
    load_color_segmentation_config,
    load_template_config,
)


# =========================
# 直接改这里，不需要命令行传参
# =========================
TARGET_CATEGORY = "T"  # 可选: line / square / L_yellow / L_blue / z_blue / z_green / T
CAMERA_TOPIC = "/camera/image_raw"
PATCH_SIZE = 3
WAIT_TIMEOUT = 30.0
WRITE_BACK = True  # True: 按 s 写入 template_config.yaml；False: 按 s 只打印建议值
WINDOW_NAME = "color_segmentation_calibration"


def _put_chinese_text(image, text, org, color, font_size=24):
    """在 OpenCV 图像上绘制中文文字。"""
    try:
        from PIL import Image, ImageDraw, ImageFont

        font_candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        font = None
        for font_path in font_candidates:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        if font is None:
            font = ImageFont.load_default()

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(pil_image)
        b, g, r = color
        draw.text(org, text, font=font, fill=(r, g, b))
        image[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(
            image,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


def _validate_settings():
    """启动前检查类别和 patch 参数，避免点开窗口后才发现配置错误。"""
    if PATCH_SIZE <= 0:
        raise ValueError("PATCH_SIZE 必须为正数")
    if PATCH_SIZE % 2 == 0:
        raise ValueError("PATCH_SIZE 建议使用奇数，方便以鼠标点为中心取 patch")

    # 这里会严格检查 template_config.yaml 中 color_segmentation 和当前类别是否存在。
    load_color_segmentation_config(TARGET_CATEGORY, config_path=TEMPLATE_CONFIG_PATH)


def _read_one_ros_image():
    """从 ROS 图像话题读取一帧 BGR 图像。"""
    import rospy
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image

    rospy.init_node("color_segmentation_calibration", anonymous=True)
    print(f"等待相机图像: topic={CAMERA_TOPIC}, timeout={WAIT_TIMEOUT}s")
    msg = rospy.wait_for_message(CAMERA_TOPIC, Image, timeout=WAIT_TIMEOUT)
    bridge = CvBridge()
    return bridge.imgmsg_to_cv2(msg, "bgr8")


def _clip_patch_bounds(image_shape, x, y, patch_size):
    """根据鼠标点裁剪 patch 边界，靠近图像边缘时自动缩小。"""
    image_h, image_w = image_shape[:2]
    half = patch_size // 2
    x1 = max(0, int(x) - half)
    y1 = max(0, int(y) - half)
    x2 = min(image_w, int(x) + half + 1)
    y2 = min(image_h, int(y) + half + 1)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"点击坐标无效: x={x}, y={y}")
    return x1, y1, x2, y2


def _measure_color(image_bgr, x, y):
    """计算鼠标点和周围 patch 的 BGR/RGB 标定信息。"""
    image_h, image_w = image_bgr.shape[:2]
    if x < 0 or y < 0 or x >= image_w or y >= image_h:
        raise ValueError(f"点击坐标超出图像范围: x={x}, y={y}, image={image_w}x{image_h}")

    x1, y1, x2, y2 = _clip_patch_bounds(image_bgr.shape, x, y, PATCH_SIZE)
    patch_bgr = image_bgr[y1:y2, x1:x2]

    point_bgr = image_bgr[int(y), int(x)].astype(np.uint8)
    patch_bgr_mean = patch_bgr.reshape(-1, 3).mean(axis=0)
    patch_rgb_mean = patch_bgr_mean[::-1]
    rgb = [int(round(float(value))) for value in patch_rgb_mean.tolist()]

    return {
        "x": int(x),
        "y": int(y),
        "patch_box": (x1, y1, x2, y2),
        "point_bgr": [int(v) for v in point_bgr.tolist()],
        "point_rgb": [int(v) for v in point_bgr[::-1].tolist()],
        "patch_bgr_mean": [float(v) for v in patch_bgr_mean.tolist()],
        "patch_rgb_mean": [float(v) for v in patch_rgb_mean.tolist()],
        "rgb": rgb,
    }


def _format_float_list(values, digits=1):
    return "[" + ", ".join(f"{float(value):.{digits}f}" for value in values) + "]"


def _print_result(result):
    """在终端打印一次取色结果。"""
    print("\n========== 颜色标定结果 ==========")
    print(f"类别: {TARGET_CATEGORY}")
    print(f"坐标: x={result['x']}, y={result['y']}")
    print(f"单点 BGR: {result['point_bgr']}")
    print(f"单点 RGB: {result['point_rgb']}")
    print(f"patch BGR均值: {_format_float_list(result['patch_bgr_mean'])}")
    print(f"patch RGB均值: {_format_float_list(result['patch_rgb_mean'])}")
    print(f"建议写入: {TARGET_CATEGORY}: rgb: {result['rgb']}")
    print("YAML 片段:")
    print(f"  {TARGET_CATEGORY}:")
    print(f"    rgb: {result['rgb']}")
    print("==================================")


def _draw_overlay(image_bgr, result=None):
    """生成带中文提示和点击结果的显示图。"""
    visual = image_bgr.copy()
    panel_h = 150 if result is None else 245
    overlay = visual.copy()
    cv2.rectangle(overlay, (0, 0), (visual.shape[1], panel_h), (0, 0, 0), -1)
    visual = cv2.addWeighted(overlay, 0.55, visual, 0.45, 0)

    _put_chinese_text(visual, f"请选择 {TARGET_CATEGORY} 类方块颜色", (14, 12), (255, 255, 255), 25)
    _put_chinese_text(visual, "左键取色，s保存/写入，r重置，q退出", (14, 48), (0, 255, 255), 23)
    _put_chinese_text(visual, f"patch={PATCH_SIZE}x{PATCH_SIZE}, WRITE_BACK={WRITE_BACK}", (14, 82), (220, 220, 220), 21)

    if result is None:
        _put_chinese_text(visual, "当前未取色", (14, 116), (180, 220, 255), 22)
        return visual

    x = result["x"]
    y = result["y"]
    x1, y1, x2, y2 = result["patch_box"]
    cv2.drawMarker(
        visual,
        (x, y),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=2,
    )
    cv2.rectangle(visual, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 2)

    lines = [
        f"坐标: x={x}, y={y}",
        f"BGR: {result['point_bgr']}  RGB: {result['point_rgb']}",
        f"patch RGB均值: {_format_float_list(result['patch_rgb_mean'])}",
        f"建议 rgb: {result['rgb']}",
    ]
    for index, text in enumerate(lines):
        _put_chinese_text(visual, text, (14, 116 + index * 31), (180, 240, 255), 21)

    return visual


def _require_existing_color_config(data):
    """确认 YAML 中已有 color_segmentation 和当前类别，避免静默创建新配置。"""
    color_config = data.get("color_segmentation")
    if not isinstance(color_config, dict):
        raise ValueError("template_config.yaml 缺少 color_segmentation")
    categories = color_config.get("categories")
    if not isinstance(categories, dict):
        raise ValueError("template_config.yaml 的 color_segmentation 缺少 categories")
    category_config = categories.get(TARGET_CATEGORY)
    if not isinstance(category_config, dict):
        raise KeyError(f"template_config.yaml 中没有类别: {TARGET_CATEGORY}")
    if "rgb" not in category_config:
        raise ValueError(f"template_config.yaml 中类别 {TARGET_CATEGORY} 缺少 rgb")


def _replace_rgb_line_preserving_yaml(text, rgb):
    """只替换当前类别下面的 rgb 行，保留 YAML 其它注释和格式。"""
    lines = text.splitlines(keepends=True)
    in_color_segmentation = False
    in_categories = False
    in_target_category = False
    category_indent = None

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            in_color_segmentation = stripped == "color_segmentation:"
            in_categories = False
            in_target_category = False
            category_indent = None
            continue

        if not in_color_segmentation:
            continue

        if indent == 2 and stripped == "categories:":
            in_categories = True
            in_target_category = False
            category_indent = None
            continue

        if not in_categories:
            continue

        if indent == 4 and stripped.endswith(":"):
            category_name = stripped[:-1]
            in_target_category = category_name == TARGET_CATEGORY
            category_indent = indent if in_target_category else None
            continue

        if in_target_category and indent > category_indent and stripped.startswith("rgb:"):
            prefix = line[:indent]
            newline = "\n" if line.endswith("\n") else ""
            lines[index] = f"{prefix}rgb: [{rgb[0]}, {rgb[1]}, {rgb[2]}]{newline}"
            return "".join(lines)

    raise ValueError(f"没有在 template_config.yaml 中找到 {TARGET_CATEGORY}.rgb 行")


def _write_rgb_to_template_config(rgb):
    """把当前类别的 rgb 写回 template_config.yaml。"""
    data = load_template_config(TEMPLATE_CONFIG_PATH)
    _require_existing_color_config(data)

    with open(TEMPLATE_CONFIG_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    new_text = _replace_rgb_line_preserving_yaml(text, rgb)
    yaml.safe_load(new_text)

    with open(TEMPLATE_CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(new_text)

    print(f"已写入 {TEMPLATE_CONFIG_PATH}: {TARGET_CATEGORY}.rgb = {rgb}")


def main():
    _validate_settings()
    image_bgr = _read_one_ros_image()
    if image_bgr is None or image_bgr.size == 0:
        raise RuntimeError("没有获取到有效图像")

    state = {
        "image_bgr": image_bgr,
        "result": None,
        "display": _draw_overlay(image_bgr),
    }

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        result = _measure_color(state["image_bgr"], x, y)
        state["result"] = result
        state["display"] = _draw_overlay(state["image_bgr"], result)
        _print_result(result)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print(f"请选择 {TARGET_CATEGORY} 类方块颜色")
    print("左键取色，s保存/写入，r重置，q退出")
    print(f"当前 WRITE_BACK={WRITE_BACK}，按 s 时{'会写入 YAML' if WRITE_BACK else '只打印建议 YAML'}")

    while True:
        cv2.imshow(WINDOW_NAME, state["display"])
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("r"), ord("R")):
            print("重新读取一帧相机图像...")
            try:
                new_image_bgr = _read_one_ros_image()
                if new_image_bgr is None or new_image_bgr.size == 0:
                    raise RuntimeError("没有获取到有效图像")
                state["image_bgr"] = new_image_bgr
                state["result"] = None
                state["display"] = _draw_overlay(new_image_bgr)
                print("已重置当前取色结果，并刷新相机图像")
            except Exception as exc:
                print(f"重新读取相机图像失败，保留当前图像: {exc}")
            continue
        if key in (ord("s"), ord("S")):
            if state["result"] is None:
                print("还没有取色，请先左键点击方块上表面")
                continue
            if WRITE_BACK:
                _write_rgb_to_template_config(state["result"]["rgb"])
            else:
                _print_result(state["result"])
                print("WRITE_BACK=False，未修改 template_config.yaml")

    cv2.destroyWindow(WINDOW_NAME)


if __name__ == "__main__":
    main()
