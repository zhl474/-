#!/home/zhl/fr3env/fr3env/bin/python
"""测试机械臂从当前位置按指定 XYZ 偏移运动时的指令耗时。"""

import time

from akai_fr import AkaiFr


# ========================= 直接修改的运行参数 =========================
MOVE_SPEED = 100
# 相对当前位置的工具坐标系 XYZ 偏移量，单位：毫米。
XYZ_OFFSET = [2.0, 1.0, 0.0]


def main():
    arm = AkaiFr()
    arm.set_speed(MOVE_SPEED)
    arm.set_tcf(1, [0, 0, 0, 0, 0, 0])

    # 第一个位置：读取当前工具坐标系位姿。
    ret, tool_pose = arm.get_tool_pose()
    print(f"获取当前工具位姿返回值: {ret}")
    print(f"工具坐标系姿态 [XYZRPY]: {tool_pose}")
    # 此 SDK 的返回值为布尔值，True 表示位姿读取成功。
    if not ret:
        raise RuntimeError(f"读取机械臂当前位姿失败，返回值: {ret}")

    # 第二个位置：保持姿态不变，仅在当前位置的 XYZ 上叠加偏移。
    TARGET_POSE = list(tool_pose)
    for index, offset in enumerate(XYZ_OFFSET):
        TARGET_POSE[index] += offset
    print(f"目标工具坐标系姿态 [XYZRPY]: {TARGET_POSE}")

    input("请确认运动路径安全，按回车开始运动: ")
    start_time = time.perf_counter()
    arm.set_tool_pose(TARGET_POSE)
    elapsed_time = time.perf_counter() - start_time
    print(f"arm.set_tool_pose(TARGET_POSE) 耗时: {elapsed_time:.6f} 秒")


if __name__ == "__main__":
    main()
