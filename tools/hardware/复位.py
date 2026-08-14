#!/home/zhl/fr3env/fr3env/bin/python
"""机械臂直接复位测试。运行前确认工作空间安全。"""

from akai_fr import AkaiFr


# ========================= 直接修改的运行参数 =========================
TARGET_POSE = [-250.4151306152343, 22.14801216125488, 380.0, -180, 0, 90]
MOVE_SPEED = 100


def main():
    arm = AkaiFr()
    arm.set_speed(MOVE_SPEED)
    arm.set_tcf(1, [0, 0, 0, 0, 0, 0])
    # input(f"确认安全后按回车移动到复位位姿: {TARGET_POSE}")
    arm.set_tool_pose(TARGET_POSE)


if __name__ == "__main__":
    main()
