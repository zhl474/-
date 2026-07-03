import cv2
import numpy as np
import matplotlib.pyplot as plt

# 创建一个测试图像
img = np.zeros((100, 100), dtype=np.uint8)
img[20:80, 20:80] = 255  # 白色方块

# 腐蚀操作
kernel = np.ones((5, 5), np.uint8)
eroded = cv2.erode(img, kernel, iterations=1)

# 计算边缘（原始 - 腐蚀）
edge = cv2.subtract(img, eroded)

# 创建显示画布
base_canvas = edge.copy()

# 显示
cv2.imshow("Original", img)
cv2.imshow("Eroded", eroded)
cv2.imshow("Edge (Original - Eroded)", edge)
cv2.imshow("Base Canvas", base_canvas)

# 等待按键
cv2.waitKey(0)
cv2.destroyAllWindows()

