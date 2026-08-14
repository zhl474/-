#!/home/zhl/fr3env/fr3env/bin/python
"""手动控制直连硬件自测：验证脱离 ROS 时机械臂/吸盘/舵机的直连通道。

运行：
    /home/zhl/fr3env/fr3env/bin/python 手动直连自测.py

默认只做只读握手，不发出任何运动/吸盘/舵机指令。要执行对应动作，
把下方对应开关改为 True 再运行。危险动作务必先确认现场安全。
"""

import sys
from pathlib import Path

import serial

SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR / "operator_panel"))

from operator_panel_lib.direct_hardware import DirectHardware

# ========================= 直接修改的运行参数 =========================
ARM_IP = "192.168.58.2"
ARM_PORT = 20003
SERVO_PORT = "/dev/servo_motor"
SERVO_BAUDRATE = 115200
MINIMUM_Z_MM = 165.0

# 只读开关（默认开，不改变硬件状态）。
RUN_GET_POSE = True            # 读 TCP 位姿
RUN_IS_STOPPED = True          # 读运动停止状态
RUN_SERVO_HANDSHAKE = True     # 仅打开串口再关闭

# 危险动作开关（默认全部关闭，需人工改 True 才会执行）。
RUN_SET_SUCTION = False        # 设置吸盘
RUN_ROTATE_TOOL = False        # 舵机转到 TOOL_ANGLE_DEG
RUN_MOVE_ARM = False           # 机械臂移动到 TARGET_POSE
RUN_STOP_MOTION = False        # 发 StopMotion 并确认停稳（急停通道）

SUCTION_STATE = 2              # 0=吸气，1=喷气，2=关闭
TOOL_ANGLE_DEG = 180.0
TARGET_POSE = [-250.415, 22.148, 380.0, -180.0, 0.0, 90.0]
MOVE_SPEED = 50
# =====================================================================


def _panel_config():
    return {
        "manual_control": {
            "arm_ip": ARM_IP,
            "arm_port": ARM_PORT,
            "servo_port": SERVO_PORT,
            "servo_baudrate": SERVO_BAUDRATE,
            "minimum_z_mm": MINIMUM_Z_MM,
        }
    }


def _servo_handshake():
    connection = serial.Serial(SERVO_PORT, SERVO_BAUDRATE, timeout=1)
    try:
        return {"opened": True, "port": SERVO_PORT}
    finally:
        connection.close()


def main():
    direct = DirectHardware(_panel_config())

    def run(name, fn):
        try:
            print(f"[OK]   {name}: {fn()}")
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")

    print(f"机械臂 XML-RPC: http://{ARM_IP}:{ARM_PORT}")
    print(f"舵机串口: {SERVO_PORT} @ {SERVO_BAUDRATE}")
    print("=" * 60)

    if RUN_GET_POSE:
        run("读取 TCP 位姿", direct.get_pose)
    if RUN_IS_STOPPED:
        run("读取停止状态", lambda: {"is_stopped": direct.is_stopped()})
    if RUN_SERVO_HANDSHAKE:
        run("舵机串口握手", _servo_handshake)
    if RUN_SET_SUCTION:
        run(f"设置吸盘 state={SUCTION_STATE}", lambda: direct.set_suction(SUCTION_STATE))
    if RUN_ROTATE_TOOL:
        run(f"舵机转到 {TOOL_ANGLE_DEG}°", lambda: direct.rotate_tool(TOOL_ANGLE_DEG))
    if RUN_MOVE_ARM:
        run("机械臂复位运动", lambda: direct.move_arm(TARGET_POSE, MOVE_SPEED, wait_until_stable=True))
    if RUN_STOP_MOTION:
        run("急停通道 StopMotion", lambda: direct.stop_motion())

    print("=" * 60)
    print("自测结束。危险动作开关默认关闭，未执行的项不会出现在上方结果中。")


if __name__ == "__main__":
    main()
