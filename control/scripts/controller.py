#!/home/zhl/fr3env/fr3env/bin/python
"""机械臂、末端舵机与电子吸盘控制节点。"""

import math
import time

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
        self.move_arm_timing_debug = bool(rospy.get_param("~move_arm_timing_debug", False))
        stability = rospy.get_param("~arm_stability", {})
        self.stable_timeout = float(stability.get("timeout_seconds", 5.0))
        self.stable_poll_interval = float(stability.get("poll_interval_seconds", 0.005))
        self.stable_duration = float(stability.get("duration_seconds", 0.01))
        self.linear_speed_threshold = float(stability.get("linear_speed_threshold_mm_s", 3.0))
        self.angular_speed_threshold = float(stability.get("angular_speed_threshold_deg_s", 3.0))
        self.position_tolerance = float(stability.get("position_tolerance_mm", 1.0))
        self.orientation_tolerance = float(stability.get("orientation_tolerance_deg", 0.5))
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

    @staticmethod
    def _angle_error(actual, target):
        return abs((float(actual) - float(target) + 180.0) % 360.0 - 180.0)

    @staticmethod
    def _rpc_value(name, result, value_length=None):
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise RuntimeError(f"{name} 返回格式异常: {result!r}")
        if result[0] != 0:
            raise RuntimeError(f"{name} 查询失败，错误码: {result[0]}")
        value = result[1]
        if value_length is not None:
            if not isinstance(value, (tuple, list)) or len(value) != value_length:
                raise RuntimeError(f"{name} 数据格式异常: {value!r}")
            return [float(item) for item in value]
        return value

    def _wait_until_arm_stable(self, target_pose, motion_started_at=None):
        """等待机械臂到达目标位姿，并返回运动与保稳阶段耗时。"""
        if motion_started_at is None:
            motion_started_at = time.monotonic()
        deadline = time.monotonic() + self.stable_timeout
        stable_since = None
        motion_finished_at = None

        while not rospy.is_shutdown():
            motion_done = bool(
                self._rpc_value(
                    "GetRobotMotionDone",
                    self.arm.arm.GetRobotMotionDone(),
                )
            )
            tcp_speed = self._rpc_value(
                "GetActualTCPCompositeSpeed",
                self.arm.arm.GetActualTCPCompositeSpeed(),
                value_length=2,
            )
            actual_pose = self._rpc_value(
                "GetActualTCPPose",
                self.arm.arm.GetActualTCPPose(),
                value_length=6,
            )

            position_error = math.dist(actual_pose[:3], target_pose[:3])
            orientation_error = max(
                self._angle_error(actual_pose[index], target_pose[index])
                for index in range(3, 6)
            )
            is_stable = (
                motion_done
                and abs(tcp_speed[0]) <= self.linear_speed_threshold
                and abs(tcp_speed[1]) <= self.angular_speed_threshold
                and position_error <= self.position_tolerance
                and orientation_error <= self.orientation_tolerance
            )

            now = time.monotonic()
            if motion_done and motion_finished_at is None:
                motion_finished_at = now
            if is_stable:
                if stable_since is None:
                    stable_since = now
                if now - stable_since >= self.stable_duration:
                    return (
                        motion_finished_at - motion_started_at,
                        now - motion_finished_at,
                    )
            else:
                stable_since = None

            if now >= deadline:
                raise TimeoutError(
                    "等待机械臂停稳超时: "
                    f"motion_done={motion_done}, "
                    f"线速度={tcp_speed[0]:.3f} mm/s, "
                    f"姿态速度={tcp_speed[1]:.3f} °/s, "
                    f"位置误差={position_error:.3f} mm, "
                    f"姿态误差={orientation_error:.3f}°"
                )
            time.sleep(self.stable_poll_interval)

        raise RuntimeError("ROS 正在关闭，停止等待机械臂停稳")

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
            motion_started_at = time.monotonic()
            result = self.arm.arm.MoveL(pose, tool=0, user=0, vel=int(request.speed))
            if (isinstance(result, bool) and not result) or (
                not isinstance(result, bool) and result != 0
            ):
                return MoveArmResponse(success=False, message=f"机械臂 MoveL 返回失败: {result!r}")
            if request.wait_until_stable:
                motion_seconds, stabilization_seconds = self._wait_until_arm_stable(
                    pose,
                    motion_started_at,
                )
                if getattr(self, "move_arm_timing_debug", False):
                    rospy.loginfo(
                        "机械臂阶段耗时：运动=%.1f ms，保稳=%.1f ms，总计=%.1f ms",
                        motion_seconds * 1000.0,
                        stabilization_seconds * 1000.0,
                        (motion_seconds + stabilization_seconds) * 1000.0,
                    )
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
