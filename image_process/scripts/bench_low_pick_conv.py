#!/home/zhl/fr3env/fr3env/bin/python
"""低位抓取点对齐紧凑核无 Padding 卷积性能基准（可直接运行，参数在文件开头修改）。"""

# ---------- 可调参数（直接修改本段） ----------
块像素 = 83
连接像素 = 11
搜索半径 = 30
角度中心 = 20.0
角度窗口 = 5.0
循环次数 = 50
使用CUDA = True
# ---------------------------------------------

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMAGE_PROCESS_DIR = os.path.join(SRC_DIR, "image_process")
if IMAGE_PROCESS_DIR not in sys.path:
    sys.path.insert(0, IMAGE_PROCESS_DIR)

from image_process_lib.template_match.kernels_create import (
    build_angle_values,
    create_base_shape,
    create_pick_aligned_kernels,
    embed_in_center,
    rotate_image,
)
from image_process_lib.template_match.template_match import load_img

类别表 = ["L_yellow", "L_blue", "T", "z_blue", "z_green", "square", "line"]


def 合成掩码(category, theta):
    base = create_base_shape(category, 块像素, 连接像素)
    height, width = base.shape
    length = int(np.ceil(np.sqrt(width ** 2 + height ** 2)))
    if length % 2 == 0:
        length += 1
    rotated = rotate_image(embed_in_center(base, length), theta)
    return (rotated > 0.5).astype(np.uint8) * 255


def main():
    device = "cuda" if 使用CUDA and torch.cuda.is_available() else "cpu"
    print(f"基准设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    全部通过 = True
    for category in 类别表:
        angles = build_angle_values(
            category, angle_step=1.0, angle_center=-角度中心, angle_window=角度窗口
        )
        prepared = create_pick_aligned_kernels(
            category, 块像素, 连接像素, angles, device=device, safety_margin_px=2,
        )
        kernel_h, kernel_w = prepared["kernel_size"]
        ax, ay = prepared["pick_anchor"]
        # 构造一级 ROI 大小的输入掩码：核 + 2*搜索半径，中心放置方块。
        roi_h = kernel_h + 2 * 搜索半径
        roi_w = kernel_w + 2 * 搜索半径
        binary = 合成掩码(category, -角度中心)
        ys, xs = np.nonzero(binary)
        bbox_h = int(ys.max()) - int(ys.min()) + 1
        bbox_w = int(xs.max()) - int(xs.min()) + 1
        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        top = (roi_h - bbox_h) // 2
        left = (roi_w - bbox_w) // 2
        mask[top:top + bbox_h, left:left + bbox_w] = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        tensor = load_img(mask, device=device)
        kernels = prepared["kernels"]
        output = F.conv2d(tensor, kernels, padding=0, stride=1)
        output_size = (output.shape[2], output.shape[3])
        for _ in range(3):
            F.conv2d(tensor, kernels, padding=0, stride=1)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(循环次数):
            F.conv2d(tensor, kernels, padding=0, stride=1)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / 循环次数
        print(
            f"{category}: 候选 {len(angles)} 个角度，公共核 {kernel_w}x{kernel_h}，"
            f"ROI {roi_w}x{roi_h}，输出 {output_size[1]}x{output_size[0]}，"
            f"单次卷积 {elapsed_ms:.2f} ms"
        )
        if output_size != (2 * 搜索半径 + 1, 2 * 搜索半径 + 1):
            print(f"  !! 输出尺寸异常：期望 {2 * 搜索半径 + 1}x{2 * 搜索半径 + 1}")
            全部通过 = False
        if device == "cuda" and elapsed_ms > 10.0:
            全部通过 = False
    if device == "cuda":
        print("验收结论: 输出 61x61，各类别单次 GPU 卷积均不超过 10 ms" if 全部通过 else "验收结论: 未通过验收")
    else:
        print("验收结论: CPU 模式仅用于功能检查，不参与 10 ms 验收")


if __name__ == "__main__":
    main()
