#!/home/zhl/fr3env/fr3env/bin/python
import os

from servo_eccentric_compensator import ServoEccentricCompensator


# ========================= 直接修改的运行参数 =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "servo_eccentric_compensation.yaml")
TEST_ANGLE_DEG = 90.0
NOMINAL_POSE = [-9, -351.252, 320.273, 180, 0, -170]


def main():
    compensator = ServoEccentricCompensator(CONFIG_PATH, debug=True)
    compensation = compensator.get_compensation(TEST_ANGLE_DEG)
    p_cmd = compensator.apply(NOMINAL_POSE, TEST_ANGLE_DEG)

    print(f"输入角度(deg): {TEST_ANGLE_DEG}")
    print(f"补偿量(mm): {compensation.tolist()}")
    print(f"补偿前目标点: {NOMINAL_POSE}")
    print(f"补偿后目标点: {p_cmd.tolist()}")


if __name__ == "__main__":
    main()
