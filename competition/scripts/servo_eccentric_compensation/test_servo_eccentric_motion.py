#!/home/zhl/fr3env/fr3env/bin/python
import argparse
import os
import time
import warnings

import serial
from serial.tools import list_ports
from akai_fr import AkaiFr, AkaiElectricSucker

from servo_eccentric_compensator import ServoEccentricCompensator


# ==================== 手动填写区 ====================
# 这里填写不考虑偏心补偿时，希望吸盘真实中心对准的机械臂目标点，单位 mm。
TARGET_POSE = [-9, -351.252, 320.273, 180, 0, -170]

# 这里填写要测试的舵机角度，单位 deg；不要求等间隔。
TEST_ANGLES = [0, 90, 180, 270]

# 机械臂运动速度。
ARM_SPEED = 50

# 舵机发出角度后等待的时间，单位秒；如果舵机转得慢可以调大。
SERVO_WAIT_SEC = 1.5

# 舵机串口配置，和 control/scripts/controller.py 保持一致。
SERVO_PORT = "/dev/servo_motor"
SERVO_BAUDRATE = 115200

# 吸盘状态：none 不操作吸盘，off 关闭吸盘，in 吸气，out 喷气。
SUCKER_STATE = "off"
# ====================================================


class DirectServo:
    """直接按控制节点协议发送舵机角度。"""

    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.ser = self._open_serial()

    def _open_serial(self):
        try:
            ser = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=1)
            print(f"舵机串口已打开: {self.port}, 波特率: {self.baudrate}")
            return ser
        except serial.SerialException as e:
            available_ports = [
                f"{item.device} ({item.description}, 序列号: {item.serial_number})"
                for item in list_ports.comports()
            ]
            print(f"舵机串口打开失败: {self.port}, 错误: {e}")
            print(f"当前可用串口: {available_ports}")
            raise

    def rotate_to(self, angle):
        angle = float(angle)
        if angle < 0:
            angle = 0.0
            warnings.warn("旋转角小于0，已限制为0")
        if angle > 360:
            angle = 360.0
            warnings.warn("旋转角大于360，已限制为360")

        data = f"{angle}E"
        self.ser.write(data.encode())
        print(f"舵机角度已发送: {data}")
        return angle

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


class DirectArmAndSucker:
    """直接控制机械臂和吸盘，写法参考 control/scripts/controller.py。"""

    def __init__(self, speed):
        self.arm = AkaiFr()
        self.arm.set_speed(speed)
        self.speed = speed
        tcf = [0, 0, 0, 0, 0, 0.0]
        self.arm.set_tcf(1, tcf)
        self.sucker = AkaiElectricSucker(self.arm)

    def move_to(self, pose):
        self.arm.set_speed(self.speed)
        self.arm.set_tool_pose(list(pose))

    def suck_in(self):
        self.sucker.set_solenoid_valve(False)   # 电磁阀关闭
        self.sucker.set_pump_motor(True)        # 气泵电机开启

    def suck_out(self):
        self.sucker.set_solenoid_valve(True)    # 电磁阀开启
        self.sucker.set_pump_motor(True)        # 气泵电机开启

    def sucker_off(self):
        self.sucker.set_solenoid_valve(False)   # 电磁阀关闭
        self.sucker.set_pump_motor(False)       # 气泵电机关闭

    def set_sucker_state(self, state):
        if state == "none":
            return
        if state == "in":
            self.suck_in()
        elif state == "out":
            self.suck_out()
        elif state == "off":
            self.sucker_off()
        else:
            raise ValueError(f"无效吸盘状态: {state}")


def parse_angles(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "servo_eccentric_compensation.yaml")

    parser = argparse.ArgumentParser(description="实机测试吸盘舵机偏心补偿")
    parser.add_argument("--config", default=default_config, help="补偿 yaml 配置路径")
    parser.add_argument("--angles", default=None, help="测试角度，例如: 0,45,90,180")
    parser.add_argument("--pose", type=float, nargs=6, default=None, help="目标 pose: x y z rx ry rz")
    parser.add_argument("--speed", type=float, default=ARM_SPEED, help="机械臂速度")
    parser.add_argument("--servo-port", default=SERVO_PORT, help="舵机串口")
    parser.add_argument("--servo-baudrate", type=int, default=SERVO_BAUDRATE, help="舵机串口波特率")
    parser.add_argument("--servo-wait", type=float, default=SERVO_WAIT_SEC, help="舵机转动等待时间")
    parser.add_argument(
        "--sucker",
        choices=["none", "off", "in", "out"],
        default=SUCKER_STATE,
        help="测试过程中设置吸盘状态",
    )
    args = parser.parse_args()

    target_pose = args.pose if args.pose is not None else TARGET_POSE
    test_angles = parse_angles(args.angles) if args.angles else TEST_ANGLES

    compensator = ServoEccentricCompensator(args.config, debug=False)
    servo = DirectServo(args.servo_port, args.servo_baudrate)
    arm = DirectArmAndSucker(args.speed)

    print("目标点为未补偿 nominal pose，脚本会按 p_cmd = p_nominal + C(theta) 运动。")
    print(f"nominal 目标点: {target_pose}")
    print(f"测试角度: {test_angles}")
    print(f"补偿配置: {args.config}")
    input("确认机械臂运动区域安全后按回车开始...")

    try:
        arm.set_sucker_state(args.sucker)
        for idx, theta in enumerate(test_angles, start=1):
            print(f"\n第 {idx}/{len(test_angles)} 次测试，舵机角度: {theta} deg")
            actual_theta = servo.rotate_to(theta)
            time.sleep(args.servo_wait)

            compensation = compensator.get_compensation(actual_theta)
            cmd_pose = compensator.apply(target_pose, actual_theta).tolist()
            print(f"补偿量(mm): {compensation.tolist()}")
            print(f"补偿前目标点: {target_pose}")
            print(f"补偿后目标点: {cmd_pose}")

            arm.move_to(cmd_pose)
            input("请检查吸盘是否对准；检查完成后按回车继续下一个角度...")
    finally:
        servo.close()
        print("测试结束，舵机串口已关闭。")


if __name__ == "__main__":
    main()
