import torch
import torch.nn.functional as F
import cv2
from typing import List, Tuple
import numpy as np
import time
from .kernels_create import (
    create_rotation_kernels,
    get_template_rect_size,
    show_all_kernels_grid,
    show_kernel,
)


def load_img(img, device: str = "cuda") -> torch.Tensor:
    """
    加载图像，返回 shape (1, 1, H, W) 的 float 张量，值域 {0, 1}
    """
    if img is None:
        raise FileNotFoundError("无法读取图像")
    # 转为 torch tensor，增加 batch 和 channel 维度
    tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    tensor = tensor / 255.0
    if device != "cpu":
        tensor = tensor.to(torch.float16)
    tensor = tensor.to(device)
    return tensor


def _finish_timing_stage(timing_output, stage_name, started_at, device):
    """结束一个模板匹配计时阶段，CUDA 时等待异步任务完成。"""
    if timing_output is None:
        return
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    previous_ms = timing_output["阶段毫秒"].get(stage_name)
    timing_output["阶段毫秒"][stage_name] = elapsed_ms + (previous_ms or 0.0)


def _normalize_kernel_hw(kernel_size):
    """把旧方核尺寸或新矩形核尺寸统一为（高，宽）。"""
    if isinstance(kernel_size, (tuple, list)):
        if len(kernel_size) != 2:
            raise ValueError("矩形模板核尺寸必须为（高，宽）")
        kernel_h, kernel_w = int(kernel_size[0]), int(kernel_size[1])
    else:
        kernel_h = kernel_w = int(kernel_size)
    if kernel_h <= 0 or kernel_w <= 0:
        raise ValueError("模板核尺寸必须为正数")
    return kernel_h, kernel_w


def match_template(image: torch.Tensor, template: torch.Tensor, kernel_size, angles) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    用卷积进行模板匹配。
    image: 大图，shape (1, 1, H, W)，值域 0/1
    template: 单个模板 (1, 1, h, w) 或模板列表
    angles: 每个模板对应的真实旋转角度
    返回: (特征图 batch, 最佳匹配位置列表)
          特征图形状 (N, H_out, W_out)，N 为模板个数
          位置列表每个元素为 (top, left) 像素坐标
    """

    # 使用 conv2d 做互相关（卷积不翻转，等价于模板匹配中的相关性）
    # groups=1 表示普通卷积
    kernel_h, kernel_w = _normalize_kernel_hw(kernel_size)
    pad = (kernel_h // 2, kernel_w // 2)
    # feature_map = F.conv2d(image, template, padding=pad, stride=1)
    feature_map_small = F.conv2d(image, template, padding=pad, stride=2)
    # 上采样回原尺寸
    feature_map = F.interpolate(
        feature_map_small,
        size=image.shape[-2:],   # (H, W)
        mode="nearest"
    )

    feature_map = feature_map.squeeze(0)  # (N, H, W)
    N, H, W = feature_map.shape

    # 🔥 全局最大
    flat_idx = torch.argmax(feature_map)
    # 转换成 (angle, y, x)
    angle_idx = flat_idx // (H * W)
    remain = flat_idx % (H * W)
    top = remain // W
    left = remain % W

    best_match = {
        "angle": float(angles[angle_idx.item()]),
        "y": top.item(),
        "x": left.item(),
        "score": feature_map[angle_idx, top, left].item()
    }

    best_kernel = show_kernel(template, angle_idx.item(),show=0)
    return best_kernel, best_match


def crop_image_for_search(image, search_center=None, search_radius=None, kernel_size=0):
    """按搜索中心和半径裁剪输入图像，并返回裁剪图及其在原图里的偏移。"""
    if search_center is None or search_radius is None:
        return image, (0, 0)

    radius = float(search_radius)
    if radius <= 0:
        return image, (0, 0)

    center_x, center_y = search_center
    kernel_h, kernel_w = _normalize_kernel_hw(kernel_size)
    pad = int(max(kernel_h, kernel_w) // 2 + radius)
    image_h, image_w = image.shape[:2]
    x1 = max(0, int(round(float(center_x))) - pad)
    y1 = max(0, int(round(float(center_y))) - pad)
    x2 = min(image_w, int(round(float(center_x))) + pad + 1)
    y2 = min(image_h, int(round(float(center_y))) + pad + 1)
    if x2 <= x1 or y2 <= y1:
        return image, (0, 0)

    return image[y1:y2, x1:x2], (x1, y1)


def get_rect(
    image,
    block_px,
    connector_px,
    category,
    img_bgr2,
    crop_x,
    crop_y,
    angle_step=1,
    angle_center=None,
    angle_window=None,
    angle_values=None,
    search_center=None,
    search_radius=None,
    debug_output=None,
    timing_output=None,
    prepared_templates=None,
):
    # 2. 加载模板（可以是一个或多个）
    template_started_at = time.perf_counter() if timing_output is not None else None
    if prepared_templates is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        kernels, kernel_size, angles = create_rotation_kernels(
            block_px,
            connector_px,
            category,
            device=device,
            angle_step=angle_step,
            angle_center=angle_center,
            angle_window=angle_window,
            angle_values=angle_values,
        )
        _finish_timing_stage(timing_output, "模板生成", template_started_at, device)
    else:
        kernels = prepared_templates["kernels"]
        kernel_size = prepared_templates["kernel_size"]
        angles = prepared_templates["angles"]
        device = str(kernels.device.type)
        if timing_output is not None and timing_output["阶段毫秒"].get("模板生成") is None:
            timing_output["阶段毫秒"]["模板生成"] = 0.0
    if timing_output is not None:
        timing_output["后端"] = device
        timing_output["模板数量"] = int(kernels.shape[0])
        kernel_h, kernel_w = _normalize_kernel_hw(kernel_size)
        timing_output["模板核尺寸"] = (int(kernel_w), int(kernel_h))

    tensor_started_at = time.perf_counter() if timing_output is not None else None
    cropped_image, (offset_x, offset_y) = crop_image_for_search(
        image,
        search_center=search_center,
        search_radius=search_radius,
        kernel_size=kernel_size,
    )
    image_tensor = load_img(cropped_image, device=device)
    _finish_timing_stage(timing_output, "张量准备", tensor_started_at, device)
    if timing_output is not None:
        timing_output["匹配图尺寸"] = (
            int(cropped_image.shape[1]),
            int(cropped_image.shape[0]),
        )

    # show_all_kernels_grid(kernels)
    # 3. 执行匹配
    match_started_at = time.perf_counter() if timing_output is not None else None
    best_kernel, positions = match_template(image_tensor, kernels, kernel_size, angles)
    _finish_timing_stage(timing_output, "卷积选优", match_started_at, device)

    postprocess_started_at = time.perf_counter() if timing_output is not None else None
    center = (positions['x'] + offset_x, positions['y'] + offset_y)  # 注意：OpenCV 用 (x, y)

    # 矩形尺寸由子块和连接处像素计算，不再从旧外接矩形标定结果读取。
    rect_size = get_template_rect_size(category, block_px, connector_px)

    # 构造 rotated rect
    rect = (center, rect_size, -1*positions['angle'])#这个opencv顺时针转是正的,模版逆时针是正的
    start_x = center[0] - (best_kernel.shape[1] // 2)
    start_y = center[1] - (best_kernel.shape[0] // 2)
    _finish_timing_stage(timing_output, "匹配收尾", postprocess_started_at, device)

    debug_started_at = time.perf_counter() if timing_output is not None else None
    if debug_output is not None:
        debug_output.update({
            "best_kernel": best_kernel.copy(),
            "match_center": (float(center[0]), float(center[1])),
            "template_top_left": (float(start_x), float(start_y)),
            "search_offset": (int(offset_x), int(offset_y)),
            "angle": float(positions["angle"]),
            "score": float(positions["score"]),
        })
    if img_bgr2 is not None:
        contours, _ = cv2.findContours(best_kernel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(
            img_bgr2,
            contours,
            -1,
            (0, 255, 0),
            1,
            offset=(int(crop_x) + int(start_x), int(crop_y) + int(start_y)),
        )
    _finish_timing_stage(timing_output, "检测调试图", debug_started_at, device)
    # cv2.imshow("match", vis_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return rect
