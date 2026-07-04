import time
import copy
from matplotlib import pyplot as plt
# 阿凯机器人工具箱
from akai import tf3d, MM, M, RAD, DEG
from akai_fr import AkaiFr
from akai_gemini335 import AkaiGemini335

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from ultralytics import YOLO

def board_detect(img):
    np.set_printoptions(suppress=True, precision=4)
    # 创建相机对象
    # camera = AkaiGemini335()

    model_path = "/home/zhl/SingleArmTetris/SingleArmTetris/src/competition/model/best.pt"
    model = YOLO(model_path)
    # img, depth_img = camera.read()
    orig_img = img.copy()
    results = model(img)
    # cv2.imwrite("/home/zhl/桌面/托盘识别.png",results[0].plot())
    # 默认只有一个检测目标
    result = results[0]
    # 获取 OBB 四个顶点坐标 (xyxyxyxy 格式)
    # shape: (1, 8)
    obb_points = result.obb.xyxyxyxy.cpu().numpy()[0]

    # reshape 成 4x2
    pts = obb_points.reshape(4, 2).astype(np.float32)

    # 计算宽高
    width = int(
        max(
            np.linalg.norm(pts[0] - pts[1]),
            np.linalg.norm(pts[2] - pts[3])
        )
    )

    height = int(
        max(
            np.linalg.norm(pts[1] - pts[2]),
            np.linalg.norm(pts[3] - pts[0])
        )
    )

    # 目标矩形
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(pts, dst_pts)

    # 裁剪图
    crop_img = cv2.warpPerspective(orig_img, M, (width, height))
    # plt.imshow(crop_img[:, :, ::-1])
    # plt.show()

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.equalizeHist(gray)
    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7
    )
    cv2.imshow("result", th)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 创建 blob detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 25
    params.maxArea = 200
    params.filterByCircularity = True
    params.minCircularity = 0.3

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)
    print(len(keypoints))
    img_draw = cv2.drawKeypoints(
        crop_img,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imshow("blobs", img_draw)
    cv2.imwrite("/home/zhl/桌面/blobs.jpg",img_draw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 提取点
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    # # 1️⃣ PCA
    # mean, eigenvectors = cv2.PCACompute(points, mean=None)

    # center = mean[0]
    # axis1 = eigenvectors[0]
    # axis2 = eigenvectors[1]

    # # 2️⃣ 投影到两个轴
    # proj1 = np.dot(points - center, axis1)
    # proj2 = np.dot(points - center, axis2)

    # # 3️⃣ 找 min/max
    # min1, max1 = np.min(proj1), np.max(proj1)
    # min2, max2 = np.min(proj2), np.max(proj2)

    # # 4️⃣ 组合四个角（在 PCA 坐标系）
    # corners = [
    #     center + min1*axis1 + min2*axis2,
    #     center + max1*axis1 + min2*axis2,
    #     center + max1*axis1 + max2*axis2,
    #     center + min1*axis1 + max2*axis2
    # ]

    # corners = np.array(corners)
    # # print(corners)
    # 1️⃣ PCA
    mean, eigenvectors = cv2.PCACompute(points, mean=None)
    center = mean[0]
    axis1 = eigenvectors[0]
    axis2 = eigenvectors[1]

    # 2️⃣ 投影
    proj1 = np.dot(points - center, axis1)
    proj2 = np.dot(points - center, axis2)

    # 3️⃣ 四个角：用“组合极值”找真实点
    idx_tl = np.argmin(proj1 + proj2)
    idx_tr = np.argmax(proj1 - proj2)
    idx_br = np.argmax(proj1 + proj2)
    idx_bl = np.argmin(proj1 - proj2)

    corners = np.array([
        points[idx_tl],
        points[idx_tr],
        points[idx_br],
        points[idx_bl]
    ])
    corners = np.array(corners)
    print("角点",corners)

    # 需要用逆矩阵
    M_inv = np.linalg.inv(M)
    corners_pix_position = corners
    i = 0
    for corner in corners:
        # 齐次坐标
        point_crop = np.array([corner[0], corner[1], 1]).reshape(3, 1)

        # 反变换
        point_orig = np.dot(M_inv, point_crop)

        # 归一化
        point_orig = point_orig / point_orig[2]

        x_orig = point_orig[0]
        y_orig = point_orig[1]
        corners_pix_position[i][0] = x_orig
        corners_pix_position[i][1] = y_orig
        i=i+1
        cv2.circle(img, (int(x_orig), int(y_orig)), 3, (0, 0, 255), -1)
        #print("裁剪图坐标:", (corner[0], corner[1]))
        #print("转换回原图坐标:", (x_orig, y_orig))

    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 1️⃣ 按 x 排序
    x_sorted = corners_pix_position[np.argsort(corners_pix_position[:, 0]), :]

    # 2️⃣ 左边两个点
    left = x_sorted[:2]
    right = x_sorted[2:]

    # 3️⃣ 左边按 y 排序
    left = left[np.argsort(left[:, 1])]
    left_top, left_bottom = left

    # 4️⃣ 右边按 y 排序
    right = right[np.argsort(right[:, 1])]
    right_top, right_bottom = right
    # 5️⃣ 组合
    
    return np.array([left_top, left_bottom, right_top, right_bottom])

if __name__ == "__main__":
    image_path = "/home/zhl/图片/数据集/15_Color.png"
    img = cv2.imread(image_path)
    board_detect(img)