#!/home/zhl/fr3env/fr3env/bin/python
"""让舵机在 0 度和 360 度之间反复转动。

默认一直循环，按 Ctrl+C 停止。
"""

import time

import serial


# ========================= 直接修改的运行参数 =========================
SERIAL_PORT = "/dev/servo_motor"
SERIAL_BAUDRATE = 115200

MIN_ANGLE_DEG = 0.0
MAX_ANGLE_DEG = 360.0

# 每次发送角度后等待的时间，需大于舵机完成转动所需的时间
MOVE_WAIT_SECONDS = 3.0

# 0 表示无限循环；大于 0 时表示往返的次数
REPEAT_COUNT = 0


def send_angle(connection, angle):
    """向舵机发送目标角度。"""
    command = f"{float(angle)}E"
    connection.write(command.encode("ascii"))
    connection.flush()
    print(f"舵机指令已发送: {command}")


def main():
    if not 0.0 <= MIN_ANGLE_DEG <= 360.0:
        raise ValueError("MIN_ANGLE_DEG 必须在 0 到 360 度之间")
    if not 0.0 <= MAX_ANGLE_DEG <= 360.0:
        raise ValueError("MAX_ANGLE_DEG 必须在 0 到 360 度之间")
    if MIN_ANGLE_DEG >= MAX_ANGLE_DEG:
        raise ValueError("MIN_ANGLE_DEG 必须小于 MAX_ANGLE_DEG")
    if MOVE_WAIT_SECONDS <= 0:
        raise ValueError("MOVE_WAIT_SECONDS 必须大于 0")
    if REPEAT_COUNT < 0:
        raise ValueError("REPEAT_COUNT 不能小于 0")

    connection = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1)
    completed_count = 0

    try:
        print(
            f"舵机开始在 {MIN_ANGLE_DEG} 度和 {MAX_ANGLE_DEG} 度之间"
            "往复转动，按 Ctrl+C 停止。"
        )

        # 先转到起始位置，后面每轮都是一次完整的往返
        send_angle(connection, MIN_ANGLE_DEG)
        time.sleep(MOVE_WAIT_SECONDS)

        while REPEAT_COUNT == 0 or completed_count < REPEAT_COUNT:
            send_angle(connection, MAX_ANGLE_DEG)
            time.sleep(MOVE_WAIT_SECONDS)

            send_angle(connection, MIN_ANGLE_DEG)
            time.sleep(MOVE_WAIT_SECONDS)

            completed_count += 1
            print(f"已完成 {completed_count} 次往返")
    except KeyboardInterrupt:
        print("\n收到停止指令，舵机测试已结束。")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
