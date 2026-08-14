#!/home/zhl/fr3env/fr3env/bin/python
"""直接测试电子吸盘状态。"""

from akai_fr import AkaiElectricSucker, AkaiFr
import time

# 可选值："吸气"、"喷气"、"关闭"。
TARGET_STATE = "关闭"
TARGET_POSE = [-250.4151306152343, 12.14801216125488, 380.0, -180, 0, 90]
MOVE_SPEED = 100

def main():
    arm = AkaiFr()
    arm.set_speed(MOVE_SPEED)
    arm.set_tcf(1, [0, 0, 0, 0, 0, 0])
    # input(f"确认安全后按回车移动到复位位姿: {TARGET_POSE}")
    
    sucker = AkaiElectricSucker(arm)
    arm.set_tool_pose(TARGET_POSE)
    sucker.set_solenoid_valve(True)
    sucker.set_pump_motor(True)
    # time.sleep(0.005)
    
    # if TARGET_STATE == "吸气":
    #     sucker.set_solenoid_valve(False)
    #     sucker.set_pump_motor(True)
    # elif TARGET_STATE == "喷气":
    #     sucker.set_solenoid_valve(True)
    #     sucker.set_pump_motor(True)
    # elif TARGET_STATE == "关闭":
    #     sucker.set_solenoid_valve(False)
    #     sucker.set_pump_motor(False)
    # else:
    #     raise ValueError(f"未知吸盘状态: {TARGET_STATE}")
    # print(f"吸盘状态已设置为: {TARGET_STATE}")


if __name__ == "__main__":
    main()
