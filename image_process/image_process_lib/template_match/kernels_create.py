import math

import cv2
import numpy as np
import torch


# 每个类别的逻辑子块坐标。坐标只描述基础朝向，旋转角度由卷积核生成阶段处理。
TETRIS_BLOCKS = {
    "L_yellow": [(2, 0), (0, 1), (1, 1), (2, 1)],
    "L_blue": [(0, 0), (0, 1), (1, 1), (2, 1)],
    "T": [(1, 0), (0, 1), (1, 1), (2, 1)],
    "z_blue": [(0, 0), (1, 0), (1, 1), (2, 1)],
    "z_green": [(1, 0), (2, 0), (0, 1), (1, 1)],
    "square": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "line": [(0, 0), (1, 0), (2, 0), (3, 0)],
}

ROTATION_TOTAL_ANGLE = {
    "L_blue": 360,
    "L_yellow": 360,
    "T": 360,
    "z_blue": 180,
    "z_green": 180,
    "square": 90,
    "line": 180,
}


def _normalize_angle(angle, period):
    angle = float(angle) % float(period)
    if np.isclose(angle, period, atol=1e-9):
        return 0.0
    return angle


def _validate_angle_step(angle_step):
    try:
        angle_step = float(angle_step)
    except (TypeError, ValueError) as exc:
        raise ValueError("angle_step 必须是数字") from exc

    if angle_step <= 0:
        raise ValueError("angle_step 必须为正数")

    return angle_step


def build_angle_values(category, angle_step=1, angle_center=None, angle_window=None, angle_values=None):
    """生成实际用于模板旋转的角度列表。"""
    if category not in ROTATION_TOTAL_ANGLE:
        raise ValueError(f"未知方块类别: {category}")

    period = float(ROTATION_TOTAL_ANGLE[category])
    if angle_values is not None:
        angles = [_normalize_angle(angle, period) for angle in angle_values]
    else:
        angle_step = _validate_angle_step(angle_step)
        if angle_center is None or angle_window is None:
            count = int(math.ceil(period / angle_step))
            angles = [_normalize_angle(i * angle_step, period) for i in range(count)]
        else:
            angle_window = float(angle_window)
            if angle_window < 0:
                raise ValueError("angle_window 必须大于等于 0")

            start = float(angle_center) - angle_window
            end = float(angle_center) + angle_window
            count = int(math.floor((end - start) / angle_step)) + 1
            angles = [_normalize_angle(start + i * angle_step, period) for i in range(count)]
            end_angle = _normalize_angle(end, period)
            if not any(np.isclose(angle, end_angle, atol=1e-6) for angle in angles):
                angles.append(end_angle)

    deduped = []
    seen = set()
    for angle in angles:
        key = round(float(angle), 6)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(float(angle))

    if not deduped:
        raise ValueError("角度列表不能为空")

    return deduped


def _validate_geometry(block_px, connector_px):
    try:
        block_px = int(round(float(block_px)))
        connector_px = int(round(float(connector_px)))
    except (TypeError, ValueError) as exc:
        raise ValueError("block_px 和 connector_px 必须是数字") from exc

    if block_px <= 0 or connector_px <= 0:
        raise ValueError("block_px 和 connector_px 必须为正数")

    return block_px, connector_px


def _category_cells(category):
    if category not in TETRIS_BLOCKS:
        raise ValueError(f"未知方块类别: {category}")
    return TETRIS_BLOCKS[category]


def get_template_rect_size(category, block_px, connector_px):
    """按类别和几何参数计算基础模板的外接矩形尺寸。"""
    block_px, connector_px = _validate_geometry(block_px, connector_px)
    cells = _category_cells(category)
    grid_w = max(x for x, _ in cells) + 1
    grid_h = max(y for _, y in cells) + 1
    width = grid_w * block_px + (grid_w - 1) * connector_px
    height = grid_h * block_px + (grid_h - 1) * connector_px
    return width, height


def create_base_shape(category, block_px, connector_px):
    """生成未旋转的基础模板，像素值为 0/1。"""
    block_px, connector_px = _validate_geometry(block_px, connector_px)
    cells = set(_category_cells(category))
    width, height = get_template_rect_size(category, block_px, connector_px)
    canvas = np.zeros((height, width), dtype=np.float32)

    step = block_px + connector_px

    for x, y in cells:
        x0 = x * step
        y0 = y * step
        canvas[y0:y0 + block_px, x0:x0 + block_px] = 1.0

    for x, y in cells:
        x0 = x * step
        y0 = y * step
        if (x + 1, y) in cells:
            canvas[y0:y0 + block_px, x0 + block_px:x0 + block_px + connector_px] = 1.0
        if (x, y + 1) in cells:
            canvas[y0 + block_px:y0 + block_px + connector_px, x0:x0 + block_px] = 1.0

    # 四个子块围住同一个连接交点时，填充中心连接区域，避免 square 中心留空。
    max_x = max(x for x, _ in cells)
    max_y = max(y for _, y in cells)
    for x in range(max_x):
        for y in range(max_y):
            if {(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)}.issubset(cells):
                x0 = x * step + block_px
                y0 = y * step + block_px
                canvas[y0:y0 + connector_px, x0:x0 + connector_px] = 1.0

    return canvas


def show_kernel(kernels, index, show=1):
    kernel = kernels[index, 0].detach().cpu().numpy()
    # 确保是二维数组。
    if kernel.ndim == 3 and kernel.shape[0] == 1:
        kernel = kernel.squeeze(0)
    elif kernel.ndim == 3 and kernel.shape[2] == 1:
        kernel = kernel.squeeze(-1)

    # 归一化到 0-255，便于 OpenCV 显示和画轮廓。
    if kernel.min() < 0 or kernel.max() > 1:
        kernel_normalized = (kernel - kernel.min()) / (kernel.max() - kernel.min()) * 255
        kernel_uint8 = kernel_normalized.astype(np.uint8)
    elif kernel.max() > 0:
        scale = 255 / kernel.max()
        kernel_uint8 = (kernel * scale).astype(np.uint8)
    else:
        kernel_uint8 = kernel.astype(np.uint8)

    if show:
        cv2.imshow("Kernel Visualization", kernel_uint8)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return kernel_uint8


def show_all_kernels_grid(kernels, cols=12):
    # 仅调试绘图时加载 matplotlib，避免正式节点启动时创建字体缓存。
    import matplotlib.pyplot as plt

    k = kernels.shape[0]
    length = kernels.shape[2]

    rows = math.ceil(k / cols)
    canvas = np.zeros((rows * length, cols * length))

    for i in range(k):
        r = i // cols
        c = i % cols
        kernel = kernels[i, 0].detach().cpu().numpy()
        canvas[r * length:(r + 1) * length, c * length:(c + 1) * length] = kernel

    plt.figure(figsize=(12, 12))
    plt.imshow(canvas, cmap="gray")
    plt.title("所有旋转模板")
    plt.show()


def embed_in_center(base_shape: np.ndarray, length: int) -> np.ndarray:
    canvas = np.zeros((length, length), dtype=np.float32)

    h, w = base_shape.shape
    start_y = length // 2 - h // 2
    start_x = length // 2 - w // 2

    canvas[start_y:start_y + h, start_x:start_x + w] = base_shape
    return canvas


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    length = img.shape[0]
    center = (length // 2, length // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img,
        matrix,
        (length, length),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def create_kernels(base_shape: np.ndarray, angles, device=None):
    """生成一个类别的所有旋转卷积核。"""
    h, w = base_shape.shape

    length = int(math.ceil(math.sqrt(w ** 2 + h ** 2)))
    if length % 2 == 0:
        length += 1

    base_canvas = embed_in_center(base_shape, length)
    angles = [float(angle) for angle in angles]
    angle_num = len(angles)
    kernel_dtype = np.float32 if device is None or str(device) == "cpu" else np.float16
    kernels = np.zeros((angle_num, length, length), dtype=kernel_dtype)

    for i, angle in enumerate(angles):
        rotated = rotate_image(base_canvas, angle)
        kernels[i] = (rotated > 0.5).astype(kernel_dtype)

    kernels_tensor = torch.from_numpy(kernels).unsqueeze(1)
    if device is not None:
        kernels_tensor = kernels_tensor.to(device)

    return kernels_tensor, length, angles


def create_rotation_kernels(
    block_px,
    connector_px,
    category,
    device=None,
    angle_step=1,
    angle_center=None,
    angle_window=None,
    angle_values=None,
):
    """按子块和连接处像素生成指定类别的旋转模板。"""
    base_shape = create_base_shape(category, block_px, connector_px)
    angles = build_angle_values(
        category,
        angle_step=angle_step,
        angle_center=angle_center,
        angle_window=angle_window,
        angle_values=angle_values,
    )
    return create_kernels(
        base_shape,
        angles,
        device=device,
    )


def _template_l_pick_in_canvas(binary, bbox, anchor, rect_size, template_angle):
    """按 coreect_LL_location 相同规则计算 L 形模板抓取点（画布坐标）。

    binary 为模板紧边框二值图，bbox 为其在画布中的左上角，anchor 为旋转中心
    相对紧边框左上角的偏移；返回抓取点画布坐标，两候选都无实体时回退到矩形中心。
    """
    x1, y1 = bbox[0], bbox[1]
    center_x = anchor[0] + x1
    center_y = anchor[1] + y1
    rect = ((float(center_x), float(center_y)), rect_size, -1.0 * float(template_angle))
    box = np.intp(cv2.boxPoints(rect))
    long_side = rect_size[0] if rect_size[0] > rect_size[1] else rect_size[1]

    def _edge_length(point_a, point_b):
        return math.sqrt(
            (float(point_a[0]) - float(point_b[0])) ** 2
            + (float(point_a[1]) - float(point_b[1])) ** 2
        )

    l1 = _edge_length(box[0], box[1])
    l2 = _edge_length(box[0], box[2])
    l3 = _edge_length(box[0], box[3])
    if 0.9 * long_side <= l1 <= 1.1 * long_side:
        long1, long2 = (box[0], box[1]), (box[2], box[3])
    elif 0.9 * long_side <= l2 <= 1.1 * long_side:
        long1, long2 = (box[0], box[2]), (box[1], box[3])
    elif 0.9 * long_side <= l3 <= 1.1 * long_side:
        long1, long2 = (box[0], box[3]), (box[1], box[2])
    else:
        long1, long2 = (box[0], box[1]), (box[2], box[3])

    def _edge_midpoint(edge):
        return (
            (float(edge[0][0]) + float(edge[1][0])) / 2.0,
            (float(edge[0][1]) + float(edge[1][1])) / 2.0,
        )

    mid1 = _edge_midpoint(long1)
    mid2 = _edge_midpoint(long2)
    point1 = (
        int((center_x + mid1[0]) / 2.0),
        int((center_y + mid1[1]) / 2.0),
    )
    point2 = (
        int((center_x + mid2[0]) / 2.0),
        int((center_y + mid2[1]) / 2.0),
    )

    def _covered(point):
        px = point[0] - x1
        py = point[1] - y1
        if 0 <= px < binary.shape[1] and 0 <= py < binary.shape[0]:
            return bool(binary[py, px] >= 1)
        return False

    if _covered(point1):
        return point1
    if _covered(point2):
        return point2
    return (int(round(center_x)), int(round(center_y)))


def create_pick_aligned_kernels(
    category,
    block_px,
    connector_px,
    angles,
    device=None,
    safety_margin_px=2,
):
    """按抓取点对齐生成低位候选角度公共卷积核，保留各角度独立锚点。

    每个角度生成纯黑底白前景的紧边框二值模板（四周保留 safety_margin_px
    黑边），记录矩形中心锚点 rect_center_anchors 与抓取点锚点 pick_anchors；
    所有模板按 pick_anchors 对齐到公共核同一像素 pick_anchor 后补齐统一大小。
    """
    if not angles:
        raise ValueError("候选角度列表不能为空")
    block_px, connector_px = _validate_geometry(block_px, connector_px)
    safety_margin_px = int(safety_margin_px)
    if safety_margin_px < 0:
        raise ValueError("safety_margin_px 必须为非负整数")
    metadata = build_angle_foreground_metadata(category, block_px, connector_px)
    rect_size = get_template_rect_size(category, block_px, connector_px)
    is_l = category in ("L_yellow", "L_blue")
    crops = []
    rect_center_anchors = []
    pick_anchors = []
    for angle in angles:
        item = metadata.get(angle)
        if item is None:
            raise ValueError(f"候选角度缺少前景元数据: {angle}")
        binary = item["binary"]
        anchor_x, anchor_y = item["anchor"]
        crop_h = binary.shape[0] + 2 * safety_margin_px
        crop_w = binary.shape[1] + 2 * safety_margin_px
        crop = np.zeros((crop_h, crop_w), dtype=np.float32)
        crop[
            safety_margin_px:safety_margin_px + binary.shape[0],
            safety_margin_px:safety_margin_px + binary.shape[1],
        ] = binary
        rect_anchor = (anchor_x + safety_margin_px, anchor_y + safety_margin_px)
        if is_l:
            pick_canvas = _template_l_pick_in_canvas(
                binary, item["bbox"], item["anchor"], rect_size, angle
            )
            pick = (
                pick_canvas[0] - item["bbox"][0] + safety_margin_px,
                pick_canvas[1] - item["bbox"][1] + safety_margin_px,
            )
        else:
            pick = rect_anchor
        crops.append(crop)
        rect_center_anchors.append(rect_anchor)
        pick_anchors.append(pick)

    common_pick_x = max(pick[0] for pick in pick_anchors)
    common_pick_y = max(pick[1] for pick in pick_anchors)
    kernel_w = max(
        common_pick_x - pick[0] + crop.shape[1]
        for pick, crop in zip(pick_anchors, crops)
    )
    kernel_h = max(
        common_pick_y - pick[1] + crop.shape[0]
        for pick, crop in zip(pick_anchors, crops)
    )
    if kernel_w <= 0 or kernel_h <= 0:
        raise ValueError("公共卷积核尺寸非法")
    kernel_dtype = np.float32 if device is None or str(device) == "cpu" else np.float16
    common = np.zeros((len(crops), kernel_h, kernel_w), dtype=kernel_dtype)
    kernel_rect_anchors = []
    kernel_pick_anchors = []
    for index, (crop, pick, rect_anchor) in enumerate(zip(crops, pick_anchors, rect_center_anchors)):
        top = common_pick_y - pick[1]
        left = common_pick_x - pick[0]
        common[index, top:top + crop.shape[0], left:left + crop.shape[1]] = crop
        # 锚点统一换算成公共核坐标：裁剪图左上角在核中的位置 + 裁剪图内相对锚点。
        kernel_rect_anchors.append((left + rect_anchor[0], top + rect_anchor[1]))
        kernel_pick_anchors.append((left + pick[0], top + pick[1]))
    kernels_tensor = torch.from_numpy(common).unsqueeze(1)
    if device is not None:
        kernels_tensor = kernels_tensor.to(device)
    return {
        "kernels": kernels_tensor,
        "kernel_size": (kernel_h, kernel_w),
        "angles": [float(angle) for angle in angles],
        "rect_center_anchors": kernel_rect_anchors,
        "pick_anchors": kernel_pick_anchors,
        "pick_anchor": (common_pick_x, common_pick_y),
    }


class ScreenedMatchFallbackError(RuntimeError):
    """高位尺寸筛选匹配无法继续，应回退到全角度、same-padding 慢匹配。"""


_ANGLE_FOREGROUND_METADATA_CACHE = {}
_SCREENED_KERNEL_CACHE = {}
_SCREENED_KERNEL_CACHE_CAP = 8


def build_angle_foreground_metadata(category, block_px, connector_px, angle_step=1.0):
    """按类别生成全角度二值模板的前景紧边框元数据并缓存（纯 CPU，不生成 GPU 核）。

    每个角度记录白色前景宽高、前景在安全方形画布中的紧边框、
    旋转中心相对紧边框左上角的锚点，以及紧边框二值模板。
    """
    block_px, connector_px = _validate_geometry(block_px, connector_px)
    cache_key = (str(category), block_px, connector_px, float(angle_step))
    cached = _ANGLE_FOREGROUND_METADATA_CACHE.get(cache_key)
    if cached is not None:
        return cached
    angles = build_angle_values(category, angle_step=angle_step)
    base_shape = create_base_shape(category, block_px, connector_px)
    height, width = base_shape.shape
    length = int(math.ceil(math.sqrt(width ** 2 + height ** 2)))
    if length % 2 == 0:
        length += 1
    center = length // 2
    base_canvas = embed_in_center(base_shape, length)
    metadata = {}
    for angle in angles:
        rotated = rotate_image(base_canvas, angle)
        binary = (rotated > 0.5).astype(np.uint8)
        ys, xs = np.nonzero(binary)
        if len(xs) == 0:
            raise ValueError(f"旋转模板为空: 类别={category}, 角度={angle}")
        x1, y1 = int(np.min(xs)), int(np.min(ys))
        x2, y2 = int(np.max(xs)) + 1, int(np.max(ys)) + 1
        metadata[angle] = {
            "fg_w": x2 - x1,
            "fg_h": y2 - y1,
            "bbox": (x1, y1, x2, y2),
            "anchor": (center - x1, center - y1),
            "binary": binary[y1:y2, x1:x2].copy(),
        }
    _ANGLE_FOREGROUND_METADATA_CACHE[cache_key] = metadata
    return metadata


def select_screen_angles(
    metadata,
    mask_w,
    mask_h,
    size_tolerance_px,
    relaxed_size_tolerance_px,
    min_candidate_angles,
):
    """按 Mask 白色前景宽高筛选候选角度。

    先按 size_tolerance_px 筛选；候选不足 min_candidate_angles 个时改用
    relaxed_size_tolerance_px。返回 (候选角度列表, 使用的容差)；仍不足时
    容差为 None，调用方应回退旧路径。
    """
    def _within(angle, tolerance):
        item = metadata[angle]
        return (
            abs(item["fg_w"] - mask_w) <= tolerance
            and abs(item["fg_h"] - mask_h) <= tolerance
        )

    sorted_angles = sorted(metadata)
    candidates = [angle for angle in sorted_angles if _within(angle, size_tolerance_px)]
    if len(candidates) >= min_candidate_angles:
        return candidates, size_tolerance_px
    candidates = [angle for angle in sorted_angles if _within(angle, relaxed_size_tolerance_px)]
    if len(candidates) >= min_candidate_angles:
        return candidates, relaxed_size_tolerance_px
    return candidates, None


def compute_screened_input_padding(image_h, image_w, kernel_h, kernel_w, margin):
    """计算无 padding 卷积前的手动补边尺寸，保证卷积核可放入并留有平移余量。"""
    margin = int(margin)
    if margin < 0:
        raise ValueError("minimum_translation_margin_px 必须为非负整数")
    image_w, image_h = int(image_w), int(image_h)
    kernel_w, kernel_h = int(kernel_w), int(kernel_h)
    if image_w <= 0 or image_h <= 0 or kernel_w <= 0 or kernel_h <= 0:
        raise ValueError("输入和卷积核尺寸必须为正")
    target_w = max(image_w, kernel_w + 2 * margin)
    target_h = max(image_h, kernel_h + 2 * margin)
    pad_left = (target_w - image_w) // 2
    pad_right = target_w - image_w - pad_left
    pad_top = (target_h - image_h) // 2
    pad_bottom = target_h - image_h - pad_top
    return {
        "pad_left": pad_left,
        "pad_right": pad_right,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "target_w": target_w,
        "target_h": target_h,
    }


def create_screened_kernels(
    category,
    block_px,
    connector_px,
    angles,
    device=None,
    safety_margin_px=2,
):
    """为候选角度生成公共紧边框卷积核 batch，并保留各角度独立的旋转中心锚点。

    各模板按自身前景紧边框四周保留 safety_margin_px 黑边，统一放在公共核左上角，
    右侧和下侧不足部分填 0。返回 kernels、kernel_size、angles、anchors。
    """
    if not angles:
        raise ScreenedMatchFallbackError("候选角度列表为空")
    block_px, connector_px = _validate_geometry(block_px, connector_px)
    safety_margin_px = int(safety_margin_px)
    if safety_margin_px < 0:
        raise ValueError("kernel_safety_margin_px 必须为非负整数")
    cache_key = (
        str(device),
        str(category),
        block_px,
        connector_px,
        tuple(round(float(angle), 6) for angle in angles),
        safety_margin_px,
    )
    cached = _SCREENED_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        _SCREENED_KERNEL_CACHE[cache_key] = _SCREENED_KERNEL_CACHE.pop(cache_key)
        return cached
    metadata = build_angle_foreground_metadata(category, block_px, connector_px)
    crops = []
    anchors = []
    max_w = 0
    max_h = 0
    for angle in angles:
        item = metadata.get(angle)
        if item is None:
            raise ScreenedMatchFallbackError(f"候选角度缺少前景元数据: {angle}")
        binary = item["binary"]
        anchor_x, anchor_y = item["anchor"]
        crop_h = binary.shape[0] + 2 * safety_margin_px
        crop_w = binary.shape[1] + 2 * safety_margin_px
        crop = np.zeros((crop_h, crop_w), dtype=np.float32)
        crop[
            safety_margin_px:safety_margin_px + binary.shape[0],
            safety_margin_px:safety_margin_px + binary.shape[1],
        ] = binary
        crops.append(crop)
        anchors.append((anchor_x + safety_margin_px, anchor_y + safety_margin_px))
        max_w = max(max_w, crop_w)
        max_h = max(max_h, crop_h)
    if max_w <= 0 or max_h <= 0:
        raise ScreenedMatchFallbackError("公共卷积核尺寸非法")
    kernel_dtype = np.float32 if device is None or str(device) == "cpu" else np.float16
    common = np.zeros((len(crops), max_h, max_w), dtype=kernel_dtype)
    for index, crop in enumerate(crops):
        common[index, :crop.shape[0], :crop.shape[1]] = crop
    kernels_tensor = torch.from_numpy(common).unsqueeze(1)
    if device is not None:
        kernels_tensor = kernels_tensor.to(device)
    prepared = {
        "kernels": kernels_tensor,
        "kernel_size": (max_h, max_w),
        "angles": [float(angle) for angle in angles],
        "anchors": anchors,
    }
    _SCREENED_KERNEL_CACHE[cache_key] = prepared
    while len(_SCREENED_KERNEL_CACHE) > _SCREENED_KERNEL_CACHE_CAP:
        _SCREENED_KERNEL_CACHE.pop(next(iter(_SCREENED_KERNEL_CACHE)))
    return prepared
