#!/home/zhl/fr3env/fr3env/bin/python
"""直接测试电子吸盘状态。"""

from akai_fr import AkaiElectricSucker, AkaiFr


# 可选值："吸气"、"喷气"、"关闭"。
TARGET_STATE = "关闭"


def main():
    arm = AkaiFr()
    sucker = AkaiElectricSucker(arm)
    if TARGET_STATE == "吸气":
        sucker.set_solenoid_valve(False)
        sucker.set_pump_motor(True)
    elif TARGET_STATE == "喷气":
        sucker.set_solenoid_valve(True)
        sucker.set_pump_motor(True)
    elif TARGET_STATE == "关闭":
        sucker.set_solenoid_valve(False)
        sucker.set_pump_motor(False)
    else:
        raise ValueError(f"未知吸盘状态: {TARGET_STATE}")
    print(f"吸盘状态已设置为: {TARGET_STATE}")


if __name__ == "__main__":
    main()
