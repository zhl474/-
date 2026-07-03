import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple
import time
from .kernels_create import create_rotation_kernels ,show_all_kernels_grid,show_kernel


def load_img(img,device: str = "cuda") -> torch.Tensor:
    """
    加载图像，返回 shape (1, 1, H, W) 的 float 张量，值域 {0, 1}
    """
    if img is None:
        raise FileNotFoundError("无法读取图像")
    # 转为 torch tensor，增加 batch 和 channel 维度
    tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    tensor = tensor / 255.0
    tensor = tensor.to(torch.float16)
    tensor = tensor.to(device)
    return tensor

def match_template(image: torch.Tensor,template: torch.Tensor,kernel_size) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    用卷积进行模板匹配。
    image: 大图，shape (1, 1, H, W)，值域 0/1
    template: 单个模板 (1, 1, h, w) 或模板列表
    device: "cuda" 或 "cpu"
    返回: (特征图 batch, 最佳匹配位置列表)
          特征图形状 (N, H_out, W_out)，N 为模板个数
          位置列表每个元素为 (top, left) 像素坐标
    """
    
    # 使用 conv2d 做互相关（卷积不翻转，等价于模板匹配中的相关性）
    # groups=1 表示普通卷积
    pad = kernel_size // 2
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
        "angle_index": angle_idx.item(),
        "y": top.item(),
        "x": left.item(),
        "score": feature_map[angle_idx, top, left].item()
    }

    best_kernel = show_kernel(template, angle_idx.item(),show=0)
    return best_kernel, best_match

def get_rect(image,w,h,category,img_bgr2,crop_x,crop_y):
    # 2. 加载模板（可以是一个或多个）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"使用设备: {device}")
    vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_tensor = load_img(image,device=device)
    #设置最小分辨率
    angle_step = 1
    kernels, kernel_size = create_rotation_kernels(w, h, category, device=device)
    # show_all_kernels_grid(kernels)
    # 3. 执行匹配
    start_time = time.time()
    best_kernel, positions = match_template(image_tensor, kernels, kernel_size)
    # end_time = time.time()
    # # 计算运行时间
    # elapsed_time = end_time - start_time
    # print(f"模版匹配{category}代码运行时间: {elapsed_time:.6f} 秒")
    # print(positions)
    center = (positions['x'], positions['y'])  # 注意：OpenCV 用 (x, y)
    
    # 矩形尺寸（你原始模板的 w, h）
    rect_size = (w, h)

    # 构造 rotated rect
    rect = (center, rect_size, -1*positions['angle_index'])#这个opencv顺时针转是正的,模版逆时针是正的
    # box = cv2.boxPoints(rect)
    # box = np.int32(box)

    start_x = positions['x'] - (best_kernel.shape[1] // 2)
    start_y = positions['y'] - (best_kernel.shape[0] // 2)
    contours, _ = cv2.findContours(best_kernel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 1, offset=(int(start_x), int(start_y)))
    cv2.drawContours(img_bgr2, contours, -1, (0, 255, 0), 1, offset=(int(crop_x)+int(start_x), int(crop_y)+int(start_y)))
    # cv2.imshow("match", vis_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return rect

# ========== 使用示例 ==========
if __name__ == "__main__":
    # 设备选择
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 1. 加载大图并二值化
    image = cv2.imread("/home/zhl/机器人竞赛_zhb/SingleArmTetris/template_match/图片/2.png")
    vis_img = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 先变灰度
    image_tensor = load_img(image,device=device)
    print(f"大图尺寸: {image_tensor.shape}")

    # 2. 加载模板（可以是一个或多个）
    w, h = 69, 69   #正方形
    # w, h = 140, 33   #长条
    # w, h = 109, 70   #长方形
    angle_step = 1
    kernels, kernel_size = create_rotation_kernels(w, h, "square", device=device)
    show_kernel(kernels, 0)
    print(f"卷积核形状: {kernels.shape}")  # [180, 1, kernel_size, kernel_size]
    print(f"卷积核大小: {kernel_size}")

    # 3. 执行匹配
    feature_maps, positions = match_template(image_tensor, kernels, kernel_size)
    print(positions)
    center = (positions['left'], positions['top'])  # 注意：OpenCV 用 (x, y)

    # 矩形尺寸（你原始模板的 w, h）
    rect_size = (w, h)

    # 构造 rotated rect
    rect = (center, rect_size, positions['angle_index'])
    box = cv2.boxPoints(rect)
    box = np.int32(box)


    cv2.drawContours(vis_img, [box], 0, (0, 255, 0), 2)

    cv2.imshow("match", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 5. 可视化第一个模板的匹配结果
    # visualize_result(image_tensor, templates[0], feature_maps, positions[0], template_idx=0)
