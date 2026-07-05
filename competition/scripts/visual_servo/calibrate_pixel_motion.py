#!/home/zhl/fr3env/fr3env/bin/python
import time

import numpy as np

from visual_servo_common import (
    VISUAL_SERVO_CONFIG_PATH,
    load_visual_servo_config,
    save_visual_servo_config,
)


# ===== 手动参数区：需要改参数时直接改这里 =====
# 标定方式：
# 1. 先让相机看到方块，调用视觉偏差服务或看调试图，记录初始像素误差 [dx_px, dy_px]。
# 2. 用机械臂上位机手动移动一段 XY，记录上位机显示的实际移动距离 [dx_mm, dy_mm]。
# 3. 再记录移动后的像素误差 [dx_px, dy_px]。
# 4. 至少做两次方向不平行的移动，脚本会求出 [像素误差] -> [机械臂XY修正] 的 2x2 矩阵。

# 第一次手动移动前的像素误差，单位 px。
MOVE1_PIXEL_BEFORE = [0.0, 0.0]

# 第一次手动移动后的像素误差，单位 px。
MOVE1_PIXEL_AFTER = [0.0, 0.0]

# 第一次上位机记录的机械臂实际移动量 [X, Y]，单位 mm。
MOVE1_ROBOT_DELTA_MM = [0.0, 0.0]

# 第二次手动移动前的像素误差，单位 px。
MOVE2_PIXEL_BEFORE = [0.0, 0.0]

# 第二次手动移动后的像素误差，单位 px。
MOVE2_PIXEL_AFTER = [0.0, 0.0]

# 第二次上位机记录的机械臂实际移动量 [X, Y]，单位 mm。
MOVE2_ROBOT_DELTA_MM = [0.0, 0.0]

# 是否把算出来的矩阵直接写入 visual_servo.yaml。
# 首次检查计算结果时可以先改成 False，只打印不写文件。
WRITE_CONFIG = True


def _as_vector(name, value):
    """把手动填写的列表转成二维向量，并检查是否填错长度或数值。"""
    vec = np.array(value, dtype=float)
    if vec.shape != (2,):
        raise ValueError(f"{name} 必须是长度为2的列表，例如 [10.0, -3.0]")
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{name} 包含无效数值: {value}")
    return vec


def calculate_pixel_to_robot_matrix():
    """根据两组人工标定数据计算像素误差到机械臂修正量的矩阵。

    这里使用的是误差变化量：pixel_delta = after - before。
    robot_delta 是你在上位机里实际移动的 XY 距离。两组数据组成：
        robot_delta = pixel_to_robot_matrix * pixel_delta
    视觉伺服运行时会直接用当前像素误差乘这个矩阵得到修正量。
    """
    p1_before = _as_vector("MOVE1_PIXEL_BEFORE", MOVE1_PIXEL_BEFORE)
    p1_after = _as_vector("MOVE1_PIXEL_AFTER", MOVE1_PIXEL_AFTER)
    r1_delta = _as_vector("MOVE1_ROBOT_DELTA_MM", MOVE1_ROBOT_DELTA_MM)
    p2_before = _as_vector("MOVE2_PIXEL_BEFORE", MOVE2_PIXEL_BEFORE)
    p2_after = _as_vector("MOVE2_PIXEL_AFTER", MOVE2_PIXEL_AFTER)
    r2_delta = _as_vector("MOVE2_ROBOT_DELTA_MM", MOVE2_ROBOT_DELTA_MM)

    pixel_delta_1 = p1_after - p1_before
    pixel_delta_2 = p2_after - p2_before
    pixel_delta_matrix = np.column_stack([pixel_delta_1, pixel_delta_2])
    robot_delta_matrix = np.column_stack([r1_delta, r2_delta])

    if np.linalg.norm(pixel_delta_1) < 1e-6 or np.linalg.norm(pixel_delta_2) < 1e-6:
        raise ValueError("像素变化太小，无法标定；请手动移动更明显一点")
    if abs(np.linalg.det(pixel_delta_matrix)) < 1e-6:
        raise ValueError(f"两次像素变化方向接近平行，无法求2x2矩阵: {pixel_delta_matrix}")

    pixel_to_robot = robot_delta_matrix @ np.linalg.inv(pixel_delta_matrix)
    if not np.all(np.isfinite(pixel_to_robot)):
        raise ValueError(f"计算出的矩阵包含无效值: {pixel_to_robot}")
    return pixel_to_robot, pixel_delta_matrix, robot_delta_matrix


def main():
    pixel_to_robot, pixel_delta_matrix, robot_delta_matrix = calculate_pixel_to_robot_matrix()
    print("两次手动移动对应的像素变化矩阵，每一列是一组 after-before，单位 px:")
    print(pixel_delta_matrix)
    print("两次手动移动对应的机械臂实际移动矩阵，每一列是一组 [dx, dy]，单位 mm:")
    print(robot_delta_matrix)
    print("计算得到 pixel_to_robot_matrix，单位 mm/px:")
    print(pixel_to_robot)

    if not WRITE_CONFIG:
        print("WRITE_CONFIG=False，只打印结果，不写配置文件。")
        return

    config = load_visual_servo_config()
    config["pixel_to_robot_matrix"] = pixel_to_robot.tolist()
    config.setdefault("last_calibration", {})
    config["last_calibration"]["pixel_motion"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_visual_servo_config(config)
    print(f"已写入配置文件: {VISUAL_SERVO_CONFIG_PATH}")


if __name__ == "__main__":
    main()
