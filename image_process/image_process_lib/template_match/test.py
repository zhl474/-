import cv2
import numpy as np


def main():
    # 创建一个测试图像。
    img = np.zeros((100, 100), dtype=np.uint8)
    img[20:80, 20:80] = 255

    # 腐蚀操作。
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(img, kernel, iterations=1)

    # 计算边缘（原始 - 腐蚀）。
    edge = cv2.subtract(img, eroded)
    base_canvas = edge.copy()

    cv2.imshow("原图", img)
    cv2.imshow("腐蚀后", eroded)
    cv2.imshow("边缘", edge)
    cv2.imshow("显示画布", base_canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
