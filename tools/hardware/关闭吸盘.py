#!/home/zhl/fr3env/fr3env/bin/python
"""直接关闭电子吸盘。"""

from akai_fr import AkaiElectricSucker, AkaiFr


def main():
    arm = AkaiFr()
    sucker = AkaiElectricSucker(arm)
    sucker.set_solenoid_valve(False)
    sucker.set_pump_motor(False)
    print("吸盘已关闭")


if __name__ == "__main__":
    main()
