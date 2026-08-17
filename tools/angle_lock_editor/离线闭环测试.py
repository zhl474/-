#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V2 角度锁定编辑的离线闭环测试：不碰 ROS 与机械臂。

用一张保存的高位照片走完与 image_node._run_v2_angle_lock_editor 完全
相同的链路：YOLO 出框 → V2 边缘模板匹配 → 创建编辑会话 →（弹窗或模拟）
提交锁定 → rematch_blocks_with_angle_locks 重匹配 → 打印前后对比。

自动模式=True 时不弹窗：脚本直接调编辑器提交用的同一个
提交角度锁定结果() 写 commit.json（锁第 1 个方块 +10 度），用于
无人工介入的冒烟验证；False 为真实弹窗，需在图形桌面终端运行。

运行：/home/zhl/fr3env/fr3env/bin/python tools/angle_lock_editor/离线闭环测试.py
"""

import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

SRC = Path(__file__).resolve().parents[2]
for _path in (str(SRC), str(SRC / "image_process")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import cv2  # noqa: E402
import yaml  # noqa: E402

from image_process_lib.angle_lock_edit_session import (  # noqa: E402
    编辑取消退出码,
    角度锁定会话环境变量,
    创建角度锁定编辑会话,
    读取已提交角度锁定,
    提交角度锁定结果,
)
from image_process_lib.block_scene_detector import detect_blocks_yolo  # noqa: E402
from image_process_lib.edge_template_detector import (  # noqa: E402
    EdgeTemplateConfig,
    EdgeTemplateMatcher,
)
from image_process_lib.template_config import load_template_geometry  # noqa: E402

# ---------------- 参数区 ----------------
# 注意：7.png 是 8-16 的图，当时 block_px=37；8-17 起配置改为 36（按当前相机标定）。
# 用当前配置跑这张旧图轮廓会内缩约 1px/格（整块约 3px），匹配分数只低 0.6%，
# 不影响测锁定编辑逻辑本身；要像素级贴合就用下方抓帧或换当前原图。
IMAGE_PATH = str(SRC / "tools" / "vision" / "7.png")
MODEL_PATH = str(SRC / "competition" / "model" / "best5.14.pt")
PERCEPTION_YAML = str(SRC / "image_process" / "config" / "perception.yaml")
OUTPUT_DIR = str(Path(__file__).resolve().parent / "output")
# True=不弹窗冒烟（自动锁第 1 个方块 +10 度）；False=真实弹窗人工编辑。
自动模式 = False
自动锁定序号 = 1  # 1 起，与编辑器窗口左上角标号一致
自动锁定角度偏移 = 10.0  # 在该块原 theta 上加的度数
# 非空时先从 /camera/image_rect 抓一帧存到该路径并改用它（需感知节点在跑），
# 用于拿当前尺度的原图；留空则直接用 IMAGE_PATH。
抓帧保存路径 = ""
# ----------------------------------------

EDITOR_SCRIPT = str(Path(__file__).resolve().parent / "angle_lock_editor.py")


def 规范化角度(theta):
    value = (float(theta) + 180.0) % 360.0 - 180.0
    return 0.0 if abs(value) < 1e-9 else value


def 打印方块(blocks, title):
    print(f"\n== {title} ==")
    print(f"{'序号':>4} {'类别':<10} {'theta':>9} {'px':>9} {'py':>9}")
    for block in blocks:
        print(
            f"{block['index']:>4} {block['category']:<10} "
            f"{float(block['theta']):>9.2f} {float(block['px']):>9.2f} "
            f"{float(block['py']):>9.2f}"
        )


def 抓帧(path):
    """从 /camera/image_rect 抓一帧当前相机原图，返回路径。"""
    import rospy
    from sensor_msgs.msg import Image as RosImage
    from cv_bridge import CvBridge

    rospy.init_node("angle_lock_offline_grab", anonymous=True, disable_signals=True)
    message = rospy.wait_for_message("/camera/image_rect", RosImage, timeout=5.0)
    frame = CvBridge().imgmsg_to_cv2(message, "bgr8")
    cv2.imwrite(path, frame)
    print(f"已抓取当前相机帧：{path}（{frame.shape[1]}x{frame.shape[0]}）")
    return path


def main() -> int:
    if 抓帧保存路径:
        try:
            抓帧(抓帧保存路径)
        except Exception as exc:
            print(f"抓帧失败（感知节点没在跑？）：{exc}")
            return 1
        image_path = 抓帧保存路径
    else:
        image_path = IMAGE_PATH
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"测试图读取失败：{image_path}")
        return 1

    with open(PERCEPTION_YAML, "r", encoding="utf-8") as file:
        perception = yaml.safe_load(file)
    config = EdgeTemplateConfig.from_mapping(
        perception.get("block_recognition", {}).get("v2")
    )
    template_geometry = load_template_geometry("high")

    print("加载 YOLO 检测模型……")
    from ultralytics import YOLO  # noqa: E402

    model = YOLO(MODEL_PATH)
    detections = detect_blocks_yolo(image, model)
    print(f"YOLO 出框 {len(detections)} 个：", [d["category"] for d in detections])
    if not detections:
        print("没有检测到方块，换一张高位照片再试。")
        return 1

    matcher = EdgeTemplateMatcher(template_geometry, config)
    blocks, debug_image = matcher.detect_blocks(image, detections)
    if not blocks:
        print("V2 边缘模板匹配全部失败。")
        return 1
    for offset, block in enumerate(blocks, start=1):
        block["index"] = offset
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(str(Path(OUTPUT_DIR) / "V2初步结果.jpg"), debug_image)
    打印方块(blocks, "V2 初步识别结果")

    with tempfile.TemporaryDirectory(prefix="angle_lock_offline_") as temp_dir:
        manifest_path = 创建角度锁定编辑会话(
            temp_dir,
            image,
            blocks,
            template_geometry,
            asdict(config),
        )

        if 自动模式:
            base_theta = float(blocks[自动锁定序号 - 1]["theta"])
            auto_theta = 规范化角度(base_theta + 自动锁定角度偏移)
            print(f"\n[自动模式] 直接模拟编辑器提交：锁第 {自动锁定序号} 个方块 "
                  f"theta {base_theta:.2f} -> {auto_theta:.2f}")
            提交角度锁定结果(manifest_path, {自动锁定序号: auto_theta})
            locks = 读取已提交角度锁定(manifest_path)
        else:
            if not os.environ.get("DISPLAY", "").strip():
                print("当前终端没有 DISPLAY，弹窗模式需在图形桌面终端运行。")
                return 1
            child_env = os.environ.copy()
            child_env[角度锁定会话环境变量] = str(manifest_path)
            child_env["PYTHONUNBUFFERED"] = "1"
            print("\n即将弹出角度锁定编辑窗：点击选中，A/D 微调，Z/C 粗调，"
                  "T 精确输入，Enter 提交，Esc 放弃。")
            process = subprocess.Popen(
                [sys.executable, EDITOR_SCRIPT],
                cwd=str(SRC),
                env=child_env,
            )
            process.wait()
            exit_code = int(process.returncode)
            if exit_code == 编辑取消退出码:
                print("已放弃修改（Esc），本轮按原识别结果继续，闭环正常。")
                return 0
            if exit_code != 0:
                print(f"编辑器子进程异常退出，退出码={exit_code}")
                return 1
            locks = 读取已提交角度锁定(manifest_path)

        if not locks:
            print("没有锁定任何方块，按原识别结果继续，闭环正常。")
            return 0

        zero_based_locks = {int(index) - 1: theta for index, theta in locks.items()}
        rematched, rematch_debug = matcher.rematch_blocks_with_angle_locks(
            image,
            blocks,
            zero_based_locks,
        )
        for offset, block in enumerate(rematched, start=1):
            block["index"] = offset
        cv2.imwrite(str(Path(OUTPUT_DIR) / "锁定重匹配结果.jpg"), rematch_debug)
        打印方块(rematched, "锁定重匹配结果（* = 被锁定）")
        print("\n前后对比：")
        for block in rematched:
            offset = int(block["index"])
            old = blocks[offset - 1]
            mark = "*" if (offset - 1) in zero_based_locks else " "
            print(
                f"{mark}第 {offset} 个 {block['category']:<10} "
                f"theta {float(old['theta']):8.2f} -> {float(block['theta']):8.2f}  "
                f"px {float(old['px']):8.2f} -> {float(block['px']):8.2f}  "
                f"py {float(old['py']):8.2f} -> {float(block['py']):8.2f}"
            )
    print(f"\n对比图已保存到 {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
