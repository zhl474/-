#!/home/zhl/fr3env/fr3env/bin/python
"""高位模板筛选匹配 GPU 卷积性能基准（可直接运行，参数在文件开头修改）。"""

# ---------- 可调参数（直接修改本段） ----------
块像素 = 38
连接像素 = 5
最小候选角度 = 3
尺寸容差 = 4
放宽容差 = 8
核安全边距 = 2
平移余量 = 4
循环次数 = 50
使用CUDA = True
# ---------------------------------------------

import math
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
    build_angle_foreground_metadata,
    compute_screened_input_padding,
    create_base_shape,
    create_screened_kernels,
    embed_in_center,
    rotate_image,
    select_screen_angles,
)
from image_process_lib.template_match.template_match import load_img

类别表 = ["L_yellow", "L_blue", "T", "z_blue", "z_green", "square", "line"]


def 旋转二值模板(category, theta):
    base = create_base_shape(category, 块像素, 连接像素)
    height, width = base.shape
    length = int(math.ceil(math.sqrt(width ** 2 + height ** 2)))
    if length % 2 == 0:
        length += 1
    rotated = rotate_image(embed_in_center(base, length), theta)
    return (rotated > 0.5).astype(np.uint8)


def 类别最坏候选(category, metadata):
    worst = None
    for angle in sorted(metadata):
        binary = 旋转二值模板(category, angle)
        ys, xs = np.nonzero(binary)
        mask_w = int(xs.max()) - int(xs.min()) + 1
        mask_h = int(ys.max()) - int(ys.min()) + 1
        candidates, _ = select_screen_angles(
            metadata, mask_w, mask_h, 尺寸容差, 放宽容差, 最小候选角度
        )
        if worst is None or len(candidates) > len(worst[0]):
            worst = (candidates, binary)
    return worst


def main():
    device = "cuda" if 使用CUDA and torch.cuda.is_available() else "cpu"
    print(f"基准设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    全部通过 = True
    for category in 类别表:
        metadata = build_angle_foreground_metadata(category, 块像素, 连接像素)
        candidates, binary = 类别最坏候选(category, metadata)
        if candidates is None:
            print(f"{category}: 无候选角度，跳过")
            全部通过 = False
            continue
        prepared = create_screened_kernels(
            category, 块像素, 连接像素, candidates, device=device,
            safety_margin_px=核安全边距,
        )
        kernel_h, kernel_w = prepared["kernel_size"]
        pad_info = compute_screened_input_padding(
            binary.shape[0], binary.shape[1], kernel_h, kernel_w, 平移余量
        )
        padded = np.zeros((pad_info["target_h"], pad_info["target_w"]), dtype=np.uint8)
        padded[
            pad_info["pad_top"]:pad_info["pad_top"] + binary.shape[0],
            pad_info["pad_left"]:pad_info["pad_left"] + binary.shape[1],
        ] = binary
        tensor = load_img(padded, device=device)
        kernels = prepared["kernels"]
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
            f"{category}: 最坏候选 {len(candidates)} 个角度，"
            f"公共核 {kernel_w}x{kernel_h}，单次卷积 {elapsed_ms:.2f} ms"
        )
        if elapsed_ms > 5.0:
            全部通过 = False
    if device == "cuda":
        print("验收结论: 各类别单次 GPU 卷积均不超过 5 ms" if 全部通过 else "验收结论: 存在超过 5 ms 的类别")
    else:
        print("验收结论: CPU 模式仅用于功能检查，不参与 5 ms 验收")


if __name__ == "__main__":
    main()
