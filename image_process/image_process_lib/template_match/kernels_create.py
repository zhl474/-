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
