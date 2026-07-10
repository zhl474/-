#!/home/zhl/fr3env/fr3env/bin/python
"""测试舵机串口能否打开。"""

import serial


# ========================= 直接修改的运行参数 =========================
SERIAL_PORT = "/dev/servo_motor"
SERIAL_BAUDRATE = 115200


def main():
    connection = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1)
    print(f"串口打开成功: {SERIAL_PORT}")
    connection.close()


if __name__ == "__main__":
    main()
