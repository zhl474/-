import torch
import torch.nn.functional as F
import cv2
from typing import List, Tuple
import numpy as np
import time
from .kernels_create import (
    ScreenedMatchFallbackError,
    build_angle_foreground_metadata,
    compute_screened_input_padding,
    create_rotation_kernels,
    create_screened_kernels,
    normalize_template_runs,
    select_screen_angles,
    show_all_kernels_grid,
    show_kernel,
    template_rect_size_from_runs,
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


def match_template(
    image,
    template,
    kernel_size,
    angles,
    anchors=None,
    use_same_padding=True,
) -> Tuple[torch.Tensor, dict]:
    """
    用卷积进行模板匹配。
    image: 大图，shape (1, 1, H, W)，值域 0/1
    template: 单个模板 (1, 1, h, w) 或模板列表
    angles: 每个模板对应的真实旋转角度
    anchors: 每个模板的旋转中心相对模板左上角的位置；None 时取核中心
    use_same_padding: True 时用 same padding，输出位置即核中心对准的输入位置；
                      False 时无 padding，输出位置为核左上角对准的输入位置
    返回: (特征图 batch, 最佳匹配信息)
          特征图形状 (N, H_out, W_out)，N 为模板个数
          位置信息包含 angle、y、x、score、anchor
    """

    # 使用 conv2d 做互相关（卷积不翻转，等价于模板匹配中的相关性）
    # groups=1 表示普通卷积
    kernel_h, kernel_w = _normalize_kernel_hw(kernel_size)
    if use_same_padding:
        pad = (kernel_h // 2, kernel_w // 2)
        feature_map = F.conv2d(image, template, padding=pad, stride=1)
    else:
        feature_map = F.conv2d(image, template, padding=0, stride=1)
    # feature_map_small = F.conv2d(image, template, padding=pad, stride=2)
    # 上采样回原尺寸
    # feature_map = F.interpolate(
    #     feature_map_small,
    #     size=image.shape[-2:],   # (H, W)
    #     mode="nearest"
    # )

    feature_map = feature_map.squeeze(0)  # (N, H, W)
    N, H, W = feature_map.shape

    # 🔥 全局最大
    flat_idx = torch.argmax(feature_map)
    # 转换成 (angle, y, x)
    angle_idx = flat_idx // (H * W)
    remain = flat_idx % (H * W)
    top = remain // W
    left = remain % W

    if anchors is None:
        anchor = (kernel_w // 2, kernel_h // 2)
    else:
        anchor = anchors[angle_idx.item()]

    best_match = {
        "angle": float(angles[angle_idx.item()]),
        "angle_index": angle_idx.item(),
        "y": top.item(),
        "x": left.item(),
        "score": feature_map[angle_idx, top, left].item(),
        "anchor": anchor,
    }

    best_kernel = show_kernel(template, angle_idx.item(), show=0)
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


def _match_rect_screened(
    image,
    block_px,
    connector_px,
    category,
    img_bgr2,
    crop_x,
    crop_y,
    screening_config,
    debug_output,
    timing_output,
    template_runs=None,
):
    """高位尺寸筛角 + 紧边框模板 + 无 padding 卷积的快速匹配。

    输入 Mask 前景宽高先筛出候选角度，候选不足或尺寸非法时抛出
    ScreenedMatchFallbackError 由调用方回退旧路径；GPU 运行错误不在此捕获。
    template_runs 缺省时按 block_px/connector_px 理想展开。
    """
    if image is None or getattr(image, "ndim", 0) != 2 or image.size == 0:
        raise ScreenedMatchFallbackError("输入 Mask 为空或不是二维图")
    binary_mask = (np.asarray(image) > 0).astype(np.uint8)
    ys, xs = np.nonzero(binary_mask)
    if len(xs) == 0:
        raise ScreenedMatchFallbackError("输入 Mask 前景为空")
    mask_w = int(np.max(xs)) - int(np.min(xs)) + 1
    mask_h = int(np.max(ys)) - int(np.min(ys)) + 1
    image_h, image_w = binary_mask.shape

    size_tolerance_px = int(screening_config.get("size_tolerance_px", 4))
    relaxed_size_tolerance_px = int(screening_config.get("relaxed_size_tolerance_px", 8))
    min_candidate_angles = int(screening_config.get("min_candidate_angles", 3))
    kernel_safety_margin_px = int(screening_config.get("kernel_safety_margin_px", 2))
    minimum_translation_margin_px = int(screening_config.get("minimum_translation_margin_px", 4))

    screen_started_at = time.perf_counter() if timing_output is not None else None
    metadata = build_angle_foreground_metadata(
        category, block_px, connector_px, template_runs=template_runs
    )
    candidates, tolerance_used = select_screen_angles(
        metadata,
        mask_w,
        mask_h,
        size_tolerance_px,
        relaxed_size_tolerance_px,
        min_candidate_angles,
    )
    _finish_timing_stage(timing_output, "角度筛选", screen_started_at, "cpu")
    if tolerance_used is None:
        raise ScreenedMatchFallbackError(
            f"尺寸筛选后候选角度不足 {min_candidate_angles} 个（Mask 前景 {mask_w}x{mask_h}）"
        )

    template_started_at = time.perf_counter() if timing_output is not None else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prepared = create_screened_kernels(
        category,
        block_px,
        connector_px,
        candidates,
        device=device,
        safety_margin_px=kernel_safety_margin_px,
        template_runs=template_runs,
    )
    _finish_timing_stage(timing_output, "候选模板生成", template_started_at, device)
    kernels = prepared["kernels"]
    kernel_h, kernel_w = prepared["kernel_size"]
    anchors = prepared["anchors"]
    if kernel_h <= 0 or kernel_w <= 0:
        raise ScreenedMatchFallbackError("候选卷积核尺寸非法")
    for anchor_x, anchor_y in anchors:
        if not (0 <= anchor_x < kernel_w and 0 <= anchor_y < kernel_h):
            raise ScreenedMatchFallbackError("候选角度锚点越界")

    pad_info = compute_screened_input_padding(
        image_h,
        image_w,
        kernel_h,
        kernel_w,
        minimum_translation_margin_px,
    )
    target_h, target_w = pad_info["target_h"], pad_info["target_w"]
    if target_h <= 0 or target_w <= 0:
        raise ScreenedMatchFallbackError("补边后输入画布尺寸非法")
    tensor_started_at = time.perf_counter() if timing_output is not None else None
    padded = np.zeros((target_h, target_w), dtype=np.float32)
    padded[
        pad_info["pad_top"]:pad_info["pad_top"] + image_h,
        pad_info["pad_left"]:pad_info["pad_left"] + image_w,
    ] = (binary_mask * 255).astype(np.float32)
    image_tensor = load_img(padded, device=device)
    _finish_timing_stage(timing_output, "输入补边", tensor_started_at, device)

    match_started_at = time.perf_counter() if timing_output is not None else None
    best_kernel, positions = match_template(
        image_tensor,
        kernels,
        (kernel_h, kernel_w),
        prepared["angles"],
        anchors=anchors,
        use_same_padding=False,
    )
    _finish_timing_stage(timing_output, "卷积选优", match_started_at, device)

    postprocess_started_at = time.perf_counter() if timing_output is not None else None
    out_x = positions["x"]
    out_y = positions["y"]
    anchor_x, anchor_y = positions["anchor"]
    center_x = out_x + anchor_x - pad_info["pad_left"]
    center_y = out_y + anchor_y - pad_info["pad_top"]
    start_x = out_x - pad_info["pad_left"]
    start_y = out_y - pad_info["pad_top"]
    rect_size = template_rect_size_from_runs(
        *normalize_template_runs(category, template_runs, block_px, connector_px)
    )
    rect = ((float(center_x), float(center_y)), rect_size, -1.0 * positions["angle"])
    _finish_timing_stage(timing_output, "匹配收尾", postprocess_started_at, device)

    debug_started_at = time.perf_counter() if timing_output is not None else None
    if debug_output is not None:
        debug_output.update({
            "mask_fg_size": (mask_w, mask_h),
            "tolerance_px": tolerance_used,
            "full_angle_count": len(metadata),
            "candidate_angle_count": len(candidates),
            "kernel_size": (kernel_w, kernel_h),
            "padding": (
                pad_info["pad_left"],
                pad_info["pad_right"],
                pad_info["pad_top"],
                pad_info["pad_bottom"],
            ),
            "conv_output_size": (target_h - kernel_h + 1, target_w - kernel_w + 1),
            "anchor": (anchor_x, anchor_y),
            "best_kernel": best_kernel.copy(),
            "match_center": (float(center_x), float(center_y)),
            "template_top_left": (float(start_x), float(start_y)),
            "search_offset": (0, 0),
            "angle": float(positions["angle"]),
            "score": float(positions["score"]),
            "screening_fallback": False,
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
    return rect


def match_pick_aligned(
    mask,
    kernels,
    kernel_size,
    angles,
    rect_center_anchors,
    pick_anchors,
    debug_output=None,
    timing_output=None,
):
    """低位抓取点对齐匹配：无 padding、stride=1。

    mask 为 ROI 内二值前景（0/255）；kernels 为按抓取点对齐的公共候选核。
    返回矩形中心与抓取点（均在 mask 坐标）、角度与分数。
    """
    if mask is None or getattr(mask, "ndim", 0) != 2 or mask.size == 0:
        raise ValueError("低位匹配输入 Mask 为空")
    device = str(kernels.device.type)

    tensor_started_at = time.perf_counter() if timing_output is not None else None
    image_tensor = load_img(np.asarray(mask), device=device)
    _finish_timing_stage(timing_output, "张量准备", tensor_started_at, device)

    match_started_at = time.perf_counter() if timing_output is not None else None
    best_kernel, positions = match_template(
        image_tensor,
        kernels,
        kernel_size,
        angles,
        anchors=pick_anchors,
        use_same_padding=False,
    )
    _finish_timing_stage(timing_output, "卷积选优", match_started_at, device)

    postprocess_started_at = time.perf_counter() if timing_output is not None else None
    angle_index = positions["angle_index"]
    out_x, out_y = positions["x"], positions["y"]
    rect_center = (
        out_x + rect_center_anchors[angle_index][0],
        out_y + rect_center_anchors[angle_index][1],
    )
    pick_point = (
        out_x + pick_anchors[angle_index][0],
        out_y + pick_anchors[angle_index][1],
    )
    conv_h = mask.shape[0] - int(kernel_size[0]) + 1
    conv_w = mask.shape[1] - int(kernel_size[1]) + 1
    if timing_output is not None:
        timing_output["匹配图尺寸"] = (conv_h, conv_w)
        timing_output["模板数量"] = len(angles)
        timing_output["模板核尺寸"] = (int(kernel_size[1]), int(kernel_size[0]))
        timing_output["后端"] = device
    _finish_timing_stage(timing_output, "匹配收尾", postprocess_started_at, device)

    debug_started_at = time.perf_counter() if timing_output is not None else None
    if debug_output is not None:
        debug_output.update({
            "best_kernel": best_kernel.copy(),
            "template_top_left": (float(out_x), float(out_y)),
            "match_center": (float(rect_center[0]), float(rect_center[1])),
            "pick_point": (float(pick_point[0]), float(pick_point[1])),
            "angle": float(positions["angle"]),
            "score": float(positions["score"]),
            "conv_output_size": (conv_h, conv_w),
        })
    _finish_timing_stage(timing_output, "检测调试图", debug_started_at, device)

    return {
        "rect_center": rect_center,
        "pick_point": pick_point,
        "angle": float(positions["angle"]),
        "score": float(positions["score"]),
        "out_position": (out_x, out_y),
    }


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
    screening_config=None,
    template_runs=None,
):
    screening_enabled = (
        screening_config is not None
        and bool(screening_config.get("enabled", True))
        and search_center is None
        and angle_values is None
        and angle_center is None
    )
    if screening_enabled:
        try:
            return _match_rect_screened(
                image,
                block_px,
                connector_px,
                category,
                img_bgr2,
                crop_x,
                crop_y,
                screening_config,
                debug_output,
                timing_output,
                template_runs=template_runs,
            )
        except ScreenedMatchFallbackError as fallback:
            if debug_output is not None:
                debug_output["screening_fallback"] = True
                debug_output["screening_fallback_reason"] = str(fallback)
            if not bool(screening_config.get("legacy_fallback_enabled", True)):
                raise RuntimeError(f"高位筛选匹配回退被禁用: {fallback}") from fallback
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
            template_runs=template_runs,
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
    rect_size = template_rect_size_from_runs(
        *normalize_template_runs(category, template_runs, block_px, connector_px)
    )

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
