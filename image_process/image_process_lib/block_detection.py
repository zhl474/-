import os

import cv2
import numpy as np
import math
from ultralytics import YOLO
import sys
import tensorrt

print("实际解释器：", sys.executable)
print("TensorRT 路径：", tensorrt.__file__)
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# 实际运行路径由 image_node 从 perception.yaml 注入；这里保留可独立调用时的默认值。
SEG_MODEL_PATH = os.path.join(SRC_DIR, "competition", "model", "best_seg.engine")
SEG_CONF = 0.25
TOP_SURFACE_CLASS_NAME = "top_surface"
_SEG_MODEL = None
ALLOW_COLOR_FALLBACK = True


def detect_dominant_color(image, v_threshold=120, h_bins=30, s_bins=32):
    """
    检测图像中L型方块上表面的主要颜色（通过过滤亮色区域并分析HSV直方图）

    参数:
    image: 输入BGR图像 (ROI区域)
    v_threshold: 亮度阈值，低于此值的像素被视为上表面 (0-255)
    h_bins: 色调(H)直方图的分组数量
    s_bins: 饱和度(S)直方图的分组数量

    返回:
    (h, s, v): 主要颜色的HSV值
    """
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 创建亮度掩码：过滤掉高亮区域（侧边）
    mask = (v < v_threshold).astype(np.uint8)

    # 计算带掩码的2D直方图 (H和S通道)
    hist = cv2.calcHist([hsv], [0, 1], mask, [h_bins, s_bins], [0, 180, 0, 256])

    # 找到直方图峰值位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hist)
    peak_h_idx, peak_s_idx = max_loc[1], max_loc[0]  # 注意OpenCV返回格式(x=s, y=h)

    # 计算直方图bin宽度
    h_bin_width = 180 / h_bins
    s_bin_width = 256 / s_bins

    # 计算峰值bin的中心值
    h_center = (peak_h_idx + 0.5) * h_bin_width
    s_center = (peak_s_idx + 0.5) * s_bin_width

    # 提取峰值bin内的所有像素
    bin_mask = np.zeros_like(h, dtype=np.uint8)
    bin_mask[
        (h >= peak_h_idx * h_bin_width) &
        (h < (peak_h_idx + 1) * h_bin_width) &
        (s >= peak_s_idx * s_bin_width) &
        (s < (peak_s_idx + 1) * s_bin_width) &
        (mask == 1)
    ] = 255

    # 计算实际像素平均值
    mean_h = cv2.mean(h, mask=bin_mask)[0]
    mean_s = cv2.mean(s, mask=bin_mask)[0]
    mean_v = cv2.mean(v, mask=bin_mask)[0]

    return (mean_h, mean_s, mean_v)


def _print_red_warning(message):
    print(f"\033[91m警告：{message}\033[0m")


def _get_seg_model():
    """懒加载上表面分割模型，避免每个裁剪图重复读取权重。"""
    global _SEG_MODEL
    if _SEG_MODEL is None:
        _SEG_MODEL = YOLO(SEG_MODEL_PATH, task="segment")
    return _SEG_MODEL


def _get_top_surface_class_ids(model):
    """从模型类别表中找出 top_surface 类别编号。"""
    return [
        int(class_id)
        for class_id, class_name in model.names.items()
        if class_name == TOP_SURFACE_CLASS_NAME
    ]


def _get_mask_by_yolo_seg(img_input):
    """使用 YOLO-Seg 提取裁剪图中的上表面二值掩码。"""
    if img_input is None or img_input.size == 0:
        raise ValueError("输入裁剪图为空")

    model = _get_seg_model()
    results = model(img_input, conf=SEG_CONF, verbose=False, retina_masks=True)
    if len(results) == 0:
        raise RuntimeError("YOLO-Seg 没有返回结果")

    result = results[0]
    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        raise RuntimeError("YOLO-Seg 没有检测到上表面 mask")

    top_surface_ids = _get_top_surface_class_ids(model)
    selected_indices = list(range(len(result.masks.data)))
    if top_surface_ids and result.boxes is not None and result.boxes.cls is not None:
        cls_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        selected_indices = [
            idx
            for idx, cls_id in enumerate(cls_ids)
            if cls_id in top_surface_ids
        ]

    if not selected_indices:
        raise RuntimeError("YOLO-Seg 结果中没有 top_surface 类别")

    best_mask = None
    best_area = -1
    for idx in selected_indices:
        mask = result.masks.data[idx].detach().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8)
        area = int(mask.sum())
        if area > best_area:
            best_area = area
            best_mask = mask

    if best_mask is None or best_area <= 0:
        raise RuntimeError("YOLO-Seg 返回的 top_surface mask 为空")

    target_h, target_w = img_input.shape[:2]
    if best_mask.shape != (target_h, target_w):
        best_mask = cv2.resize(
            best_mask,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST
        )

    return (best_mask * 255).astype(np.uint8)


def _get_mask_by_color(img_input,category):#白色[6,5.355,234]
    (lh,ls,lv)=detect_dominant_color(img_input, v_threshold=230, h_bins=30, s_bins=32)
    hsv = cv2.cvtColor(img_input, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    # LOWER_COLOR_LL = np.array([0,100,120])      # 下限[ 26  72 195]，亮区28,147,254
    # UPPER_COLOR_LL = np.array([35,171,185])  # 上限
    LOWER_COLOR_LL = np.array([lh-20,ls-20,lv-20])      # 下限[ 26  72 195]，亮区28,147,254
    UPPER_COLOR_LL = np.array([lh+20,ls+20,lv+10])  # 上限
    LOWER_COLOR_ZL = np.array([lh-20,ls-20,lv-20])      # 下限[116  64  88]
    UPPER_COLOR_ZL = np.array([lh+20,ls+20,lv+10])  # 上限
    LOWER_COLOR_ZR = np.array([lh-20,ls-20,lv-20])      # 下限[ 26  72 195]
    UPPER_COLOR_ZR =np.array([lh+20,ls+20,lv+10]) # 上限
    LOWER_COLOR_O = np.array([0,95,hsv[int(h/2.0)][int(w/2.0)][2]-30])      # 下限[178 119 169]
    UPPER_COLOR_O = np.array([45,140,hsv[int(h/2.0)][int(w/2.0)][2]-5])  # 上限
    LOWER_COLOR_O2 = np.array([145,95,hsv[int(h/2.0)][int(w/2.0)][2]-30])      # 下限[178 119 169]
    UPPER_COLOR_O2 = np.array([180,140,hsv[int(h/2.0)][int(w/2.0)][2]-5])  # 上限
    LOWER_COLOR_LR = np.array([lh-20,ls-20,lv-20])     # 下限[110  53  96]
    UPPER_COLOR_LR = np.array([lh+20,ls+20,lv+10]) # 上限
    LOWER_COLOR_line =  np.array([lh-12,ls-12,lv-20])      # 下限[177  127 130]
    UPPER_COLOR_line =  np.array([lh+12,ls+12,lv+10])  # 上限
    LOWER_COLOR_T = np.array([lh-20,ls-20,lv-20])
    UPPER_COLOR_T = np.array([lh+20,ls+20,lv+10])  # 上限
    if(category=="L_yellow"):
        mask = cv2.inRange(hsv, LOWER_COLOR_LL, UPPER_COLOR_LL)
    elif(category=="L_blue"):
        mask = cv2.inRange(hsv, LOWER_COLOR_LR, UPPER_COLOR_LR)
    elif(category=="z_blue"):
        mask = cv2.inRange(hsv, LOWER_COLOR_ZL, UPPER_COLOR_ZL)
    elif(category=="z_green"):
        mask = cv2.inRange(hsv, LOWER_COLOR_ZR, UPPER_COLOR_ZR)
    elif(category=="square"):
        mask = cv2.inRange(hsv, LOWER_COLOR_O, UPPER_COLOR_O)+cv2.inRange(hsv, LOWER_COLOR_O2, UPPER_COLOR_O2)
        mask = cv2.inRange(mask,np.array([100]),np.array([1000]))
    elif(category=="line"):
        mask = cv2.inRange(hsv, LOWER_COLOR_line, UPPER_COLOR_line)
    elif(category=="T"):
        mask = cv2.inRange(hsv, LOWER_COLOR_T, UPPER_COLOR_T)
    else:
        raise ValueError(f"未知方块类别: {category}")
    kernel = np.ones((1, 1), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel2)
    mask2=fill_holes(mask)
    mask=mask2 | mask
    mask = cv2.erode(mask, kernel2)
    return mask,hsv


def get_mask(img_input,category):
    hsv = cv2.cvtColor(img_input, cv2.COLOR_BGR2HSV)
    try:
        mask = _get_mask_by_yolo_seg(img_input)
        return mask,hsv
    except Exception as exc:
        if not ALLOW_COLOR_FALLBACK:
            raise
        _print_red_warning(f"YOLO-Seg 上表面分割失败，回退旧颜色分割。原因：{exc}")
        return _get_mask_by_color(img_input,category)


def draw_mask_on_full_image(full_image, mask, crop_x1, crop_y1, color=(0, 255, 0), alpha=0.35, draw_contour=True):
    """把 ROI 内的上表面掩码叠加回大图对应位置。"""
    if full_image is None or full_image.size == 0:
        raise ValueError("输入大图为空")
    if mask is None or mask.size == 0:
        raise ValueError("输入掩码为空")

    mask_h, mask_w = mask.shape[:2]
    image_h, image_w = full_image.shape[:2]
    crop_x1 = int(crop_x1)
    crop_y1 = int(crop_y1)
    x1 = max(0, crop_x1)
    y1 = max(0, crop_y1)
    x2 = min(image_w, crop_x1 + mask_w)
    y2 = min(image_h, crop_y1 + mask_h)
    if x2 <= x1 or y2 <= y1:
        return full_image

    # 大图边界裁剪后，需要同步裁剪 ROI 掩码，防止靠边方块越界。
    mask_x1 = x1 - crop_x1
    mask_y1 = y1 - crop_y1
    mask_x2 = mask_x1 + (x2 - x1)
    mask_y2 = mask_y1 + (y2 - y1)
    roi = full_image[y1:y2, x1:x2]
    mask_roi = mask[mask_y1:mask_y2, mask_x1:mask_x2] > 0
    if not np.any(mask_roi):
        return full_image

    colored_roi = roi.copy()
    colored_roi[mask_roi] = color
    blended_roi = cv2.addWeighted(roi, 1.0 - alpha, colored_roi, alpha, 0)
    roi[mask_roi] = blended_roi[mask_roi]

    if draw_contour:
        contour_mask = (mask_roi.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi, contours, -1, color, 1)

    return full_image


def get_length(point1,point2):
        return math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
def coreect_LL_location(box,mask,rect):#获取L方块长边偏下的位置，L方块系中间吸不到（不用细看）
    px=0
    py=0
    if(rect[1][0]>rect[1][1]):
        long_side=rect[1][0]
    else:
        long_side=rect[1][1]
    px=rect[0][0]
    py=rect[0][1]
    l1=get_length(box[0],box[1])
    l2=get_length(box[0],box[2])
    l3=get_length(box[0],box[3])
    if(l1/long_side>=0.9 and l1/long_side<=1.1):
        long1=[box[0],box[1]]
        long2=[box[2],box[3]]
    elif(l2/long_side>=0.9 and l2/long_side<=1.1):
        long1=[box[0],box[2]]
        long2=[box[1],box[3]]
    elif(l3/long_side>=0.9 and l3/long_side<=1.1):
        long1=[box[0],box[3]]
        long2=[box[1],box[2]]
    point1=[int((px+(long1[0][0]+long1[1][0])/2.0)/2.0),int((py+(long1[0][1]+long1[1][1])/2.0)/2.0)]
    point2=[int((px+(long2[0][0]+long2[1][0])/2.0)/2.0),int((py+(long2[0][1]+long2[1][1])/2.0)/2.0)]
    # print("mask1:",mask[point1[0]][point1[1]],point1)
    # print("mask2:",mask[point2[0]][point2[1]],point2)
    # mask[point1[0]][point1[1]]=255
    # mask[point2[0]][point2[1]]=255
    if(mask[point1[1]][point1[0]]>=1):#是是方块内
        return point1[0],point1[1]
    else:
        return point2[0],point2[1]


def fill_holes(mask):#对检测到的mask进行填充
    """
    使用 floodFill 填充二值图像中的空洞

    参数:
        mask (numpy.ndarray): 二值化掩码图像

    返回:
        numpy.ndarray: 填充空洞后的图像
    """
    # 1. 创建稍大的临时图像用于 floodFill
    h, w = mask.shape
    temp = np.zeros((h+2, w+2), dtype=np.uint8)

    # 2. 将原始图像复制到临时图像中心
    temp[1:-1, 1:-1] = mask

    # 3. 创建一个掩码用于 floodFill (需要比原图大2个像素)
    mask_flood = np.zeros((h+4, w+4), dtype=np.uint8)

    # 4. 从边界点开始填充背景
    # 注意：我们需要填充的是背景，然后将背景反转
    cv2.floodFill(temp, mask_flood, (0, 0), 255)

    # 5. 反转填充结果：现在背景为255，前景为0
    filled = cv2.bitwise_not(temp)

    # 6. 裁剪回原始尺寸
    return filled[1:-1, 1:-1]
