import torch
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

# 逻辑 L 形
base_L_yellow = np.zeros((37+5+37, 37+5+37+5+37), dtype=np.float32)
base_L_yellow[0:37+5,37+5+37+5:] = 1
base_L_yellow[37+5:,:] = 1

base_L_blue = np.zeros((37+5+37, 37+5+37+5+37), dtype=np.float32)
base_L_blue[0:37+5,:37] = 1
base_L_blue[37+5:,:] = 1

base_T = np.zeros((37+5+37, 37+5+37+5+37), dtype=np.float32)
base_T[0:37+5,37+5:37+5+37] = 1
base_T[37+5:,:] = 1

base_square = np.array([
    [1, 1],
    [1, 1]
], dtype=np.float32)
base_line = np.array([
    [1, 1, 1, 1]
], dtype=np.float32)

base_z_blue = np.zeros((37+5+37, 37+5+37+5+37), dtype=np.float32)
base_z_blue[0:37,:37+5+37] = 1
base_z_blue[37:37+5,37+5:37+5+37] = 1
base_z_blue[37+5:,37+5:] = 1

base_z_green = np.zeros((37+5+37, 37+5+37+5+37), dtype=np.float32)
base_z_green[0:37,37+5:] = 1
base_z_green[37:37+5,37+5:37+5+37] = 1
base_z_green[37+5:,:37+5+37] = 1



# ==============================
# 1️⃣ 显示函数
# ==============================

def show_kernel(kernels, index,show=1):
    kernel = kernels[index, 0].detach().cpu().numpy()
    # 确保是2D数组
    if kernel.ndim == 3 and kernel.shape[0] == 1:
        kernel = kernel.squeeze(0)  # 从 (1, H, W) -> (H, W)
    elif kernel.ndim == 3 and kernel.shape[2] == 1:
        kernel = kernel.squeeze(-1)  # 从 (H, W, 1) -> (H, W)

    # 归一化到 0-255（如果 kernel 值不在这个范围）
    if kernel.min() < 0 or kernel.max() > 1:
        kernel_normalized = (kernel - kernel.min()) / (kernel.max() - kernel.min()) * 255
        kernel_uint8 = kernel_normalized.astype(np.uint8)
    else:
        A = 255/kernel.max()
        kernel_uint8 = (kernel * A).astype(np.uint8)

    # 显示
    if show:
        cv2.imshow("Kernel Visualization", kernel_uint8)
        cv2.waitKey(0)  # 等待按键
        cv2.destroyAllWindows()  # 关闭窗口
    return kernel_uint8


def show_all_kernels_grid(kernels, cols=12):
    k = kernels.shape[0]
    L = kernels.shape[2]

    rows = math.ceil(k / cols)
    canvas = np.zeros((rows * L, cols * L))

    for i in range(k):
        r = i // cols
        c = i % cols
        kernel = kernels[i, 0].detach().cpu().numpy()
        canvas[r*L:(r+1)*L, c*L:(c+1)*L] = kernel

    plt.figure(figsize=(12, 12))
    plt.imshow(canvas, cmap="gray")
    plt.title("All Rotation Kernels")
    plt.show()


# ==============================
# 2️⃣ 将逻辑形状放大为真实像素尺寸
# ==============================

def upscale_shape(base_shape: np.ndarray,
                  total_w: int,
                  total_h: int) -> np.ndarray:
    """
    将俄罗斯方块逻辑结构放大为真实像素尺寸
    """

    grid_h, grid_w = base_shape.shape

    block_w = total_w // grid_w
    block_h = total_h // grid_h

    canvas = np.zeros((grid_h * block_h, grid_w * block_w),
                      dtype=np.float32)

    for y in range(grid_h):
        for x in range(grid_w):
            if base_shape[y, x] == 1:
                canvas[
                    y*block_h:(y+1)*block_h,
                    x*block_w:(x+1)*block_w
                ] = 1.0

    return canvas


# ==============================
# 3️⃣ 嵌入中心
# ==============================

def embed_in_center(base_shape: np.ndarray, L: int) -> np.ndarray:
    canvas = np.zeros((L, L), dtype=np.float32)

    h, w = base_shape.shape

    start_y = L // 2 - h // 2
    start_x = L // 2 - w // 2

    canvas[start_y:start_y + h,
           start_x:start_x + w] = base_shape

    return canvas


# ==============================
# 4️⃣ 旋转
# ==============================

def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    L = img.shape[0]
    center = (L // 2, L // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        img,
        M,
        (L, L),
        flags=cv2.INTER_NEAREST,   # 必须最近邻
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return rotated


# ==============================
# 5️⃣ 主函数：生成旋转卷积核
# ==============================

def create_kernels(base_shape: np.ndarray,category,total_w: int,total_h: int,angle_step: int,total_angle:int, device=None):

    # 1️⃣ 放大到真实尺寸
    if category=="line" or category=="square":
        base_shape = upscale_shape(base_shape,total_w,total_h)

    h, w = base_shape.shape

    # 2️⃣ 计算旋转安全画布
    L = int(math.ceil(math.sqrt(w**2 + h**2)))
    if L % 2 == 0:
        L += 1

    # 3️⃣ 嵌入中心
    base_canvas = embed_in_center(base_shape, L)

    #强化边缘部分
    erode_kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(base_canvas, erode_kernel, iterations=1)
    edge = cv2.subtract(base_canvas,eroded)
    base_canvas = base_canvas+edge*4
    # cv2.imshow("eroded", eroded)
    # cv2.imshow("base_canvas1", edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    angle_num = int(total_angle / angle_step)
    kernels = np.zeros((angle_num, L, L),
                       dtype=np.float16)

    # 4️⃣ 生成旋转模板
    for i, angle in enumerate(range(0, total_angle, angle_step)):
        rotated = rotate_image(base_canvas, angle)
        rotated = (rotated > 0.5).astype(np.float16)
        kernels[i] = rotated

    # 5️⃣ 转 torch
    kernels_tensor = torch.from_numpy(kernels).unsqueeze(1)

    # 6️⃣ 归一化（可选）
    # kernel_sum = kernels_tensor.sum(dim=(2, 3), keepdim=True)
    # kernels_tensor = kernels_tensor / (kernel_sum + 1e-6)

    if device is not None:
        kernels_tensor = kernels_tensor.to(device)

    return kernels_tensor, L


# ==============================
# 6️⃣ 测试
# ==============================

def create_rotation_kernels(w, h, category, device=None):
    if(category=="L_blue"):
        return create_kernels(
        base_L_blue,
        category,
        total_w=w,
        total_h=h,
        angle_step=1,
        total_angle=360,
        device=device
    )
    elif(category=="L_yellow"):
        return create_kernels(
        base_L_yellow,
        category,
        total_w=w,
        total_h=h,
        angle_step=1,
        total_angle=360,
        device=device
    )
    elif(category=="z_blue"):
        return create_kernels(
        base_z_blue,
        category,
        total_w=w,
        total_h=h,
        angle_step=1,
        total_angle=180,
        device=device
    )
    elif(category=="z_green"):
        return create_kernels(
        base_z_green,
        category,
        total_w=w,
        total_h=h,
        angle_step=1,
        total_angle=180,
        device=device
    )
    elif(category=="square"):#这个东西不是0，是O，o
        return create_kernels(
        base_square,
        category,
        total_w=w,
        total_h=h,
        angle_step=1,
        total_angle=90,
        device=device
    )
    elif(category=="T"):
        return create_kernels(
        base_T,
        category,
        total_w=w,
        total_h=h,
        angle_step=1,
        total_angle=360,
        device=device
    )
    elif(category=="line"):
        return create_kernels(
        base_line,
        category,
        total_w=w,
        total_h=h,
        angle_step=1,
        total_angle=180,
        device=device
    )
    else:
        raise ValueError(f"未知方块类别: {category}")

if __name__ == "__main__":

    

    # 已知外接矩形尺寸
    W = 109
    H = 70
    kernels, L = create_kernels(
        base_T,
        total_w=W,
        total_h=H,
        angle_step=1,
        total_angle=180,
        device="cuda"
    )

    print("Kernel shape:", kernels.shape)
    print("Kernel size:", L)

    # 显示某个角度
    show_kernel(kernels, 45)

    # 或显示全部
    # show_all_kernels_grid(kernels, cols=12)
