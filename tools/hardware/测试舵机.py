#!/home/zhl/fr3env/fr3env/bin/python
"""直接发送一次舵机角度命令。"""

import serial


# ========================= 直接修改的运行参数 =========================
SERIAL_PORT = "/dev/servo_motor"
SERIAL_BAUDRATE = 115200
TARGET_ANGLE_DEG = 90


def main():
    angle = float(TARGET_ANGLE_DEG)
    if not 0.0 <= angle <= 360.0:
        raise ValueError("TARGET_ANGLE_DEG 必须在 0 到 360 度之间")
    connection = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1)
    try:
        command = f"{angle}E"
        connection.write(command.encode("ascii"))
        print(f"舵机指令已发送: {command}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
