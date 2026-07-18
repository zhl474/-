#!/home/zhl/fr3env/fr3env/bin/python
"""机械臂、末端舵机与电子吸盘控制节点。"""

import math

import rospy
import serial
from serial.tools import list_ports

from akai_fr import AkaiElectricSucker, AkaiFr
from control.srv import (
    MoveArm,
    MoveArmResponse,
    RotateTool,
    RotateToolResponse,
    SetSuction,
    SetSuctionResponse,
)


class ControlNode:
    def __init__(self):
        self.minimum_z = float(rospy.get_param("~minimum_z", 165.0))
        self.arm = AkaiFr()
        self.arm.set_speed(80)
        self.arm.set_tcf(1, [0, 0, 0, 0, 0, 0])
        self.sucker = AkaiElectricSucker(self.arm)
        self.servo_serial = self._open_servo_serial()

        self.arm_service = rospy.Service("/control/move_arm", MoveArm, self.move_arm)
        self.motor_service = rospy.Service("/control/rotate_tool", RotateTool, self.rotate_tool)
        self.suction_service = rospy.Service("/control/set_suction", SetSuction, self.set_suction)
        rospy.on_shutdown(self.close)
        rospy.loginfo("机械臂、吸盘和舵机控制节点已启动")

    def _open_servo_serial(self):
        port = rospy.get_param("~servo_port", "/dev/servo_motor")
        baudrate = int(rospy.get_param("~servo_baudrate", 115200))
        try:
            connection = serial.Serial(port=port, baudrate=baudrate, timeout=1)
            rospy.loginfo("舵机串口已打开: %s，波特率: %d", port, baudrate)
            return connection
        except serial.SerialException as exc:
            available = [f"{item.device} ({item.description})" for item in list_ports.comports()]
            rospy.logerr("舵机串口打开失败: %s；可用串口: %s", exc, available)
            raise

    @staticmethod
    def _finite(values):
        return all(math.isfinite(float(value)) for value in values)

    def move_arm(self, request):
        pose = [float(value) for value in request.pose]
        z_was_clamped = False
        if len(pose) != 6 or not self._finite(pose):
            return MoveArmResponse(success=False, message="机械臂位姿必须包含 6 个有限数值")
        if request.speed <= 0:
            return MoveArmResponse(success=False, message="机械臂速度必须大于 0")
        if pose[2] < self.minimum_z:
            requested_z = pose[2]
            pose[2] = self.minimum_z
            z_was_clamped = True
            # 使用错误级别日志，让终端以红色突出显示安全钳制警告。
            rospy.logerr(
                "安全警告：目标 Z=%.2f mm 低于安全下限 %.2f mm，已自动调整为 %.2f mm 后继续运动",
                requested_z,
                self.minimum_z,
                pose[2],
            )
        try:
            self.arm.set_speed(int(request.speed))
            result = self.arm.arm.MoveL(pose, tool=0, user=0, vel=int(request.speed))
            if result is False:
                return MoveArmResponse(success=False, message="机械臂 MoveL 返回失败")
            if z_was_clamped:
                return MoveArmResponse(
                    success=True,
                    message=(
                        f"目标 Z={requested_z:.2f} mm 低于安全下限 {self.minimum_z:.2f} mm，"
                        f"已自动调整为 {pose[2]:.2f} mm 并完成运动"
                    ),
                )
            return MoveArmResponse(success=True, message="机械臂运动完成")
        except Exception as exc:
            rospy.logerr("机械臂运动异常: %s", exc)
            return MoveArmResponse(success=False, message=str(exc))

    def rotate_tool(self, request):
        angle = float(request.angle_deg)
        if not math.isfinite(angle) or not 0.0 <= angle <= 360.0:
            return RotateToolResponse(success=False, message="舵机角度必须在 0 到 360 度之间")
        try:
            self.servo_serial.write(f"{angle}E".encode("ascii"))
            return RotateToolResponse(success=True, message="舵机指令已发送")
        except Exception as exc:
            rospy.logerr("舵机指令发送失败: %s", exc)
            return RotateToolResponse(success=False, message=str(exc))

    def set_suction(self, request):
        try:
            if request.state == request.SUCK:
                self.sucker.set_solenoid_valve(False)
                self.sucker.set_pump_motor(True)
                message = "吸盘开始吸气"
            elif request.state == request.BLOW:
                self.sucker.set_solenoid_valve(True)
                self.sucker.set_pump_motor(True)
                message = "吸盘开始喷气"
            elif request.state == request.OFF:
                self.sucker.set_solenoid_valve(False)
                self.sucker.set_pump_motor(False)
                message = "吸盘已关闭"
            else:
                return SetSuctionResponse(success=False, message=f"未知吸盘状态: {request.state}")
            return SetSuctionResponse(success=True, message=message)
        except Exception as exc:
            rospy.logerr("吸盘控制失败: %s", exc)
            return SetSuctionResponse(success=False, message=str(exc))

    def close(self):
        if getattr(self, "servo_serial", None) is not None and self.servo_serial.is_open:
            self.servo_serial.close()


def main():
    rospy.init_node("control_node")
    ControlNode()
    rospy.spin()


if __name__ == "__main__":
    main()
