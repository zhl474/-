"""手动控制的直连硬件层：不依赖 ROS，直接复用厂商 SDK 与串口操作硬件。

仅当 control_node 未运行时由 coordinator 分流使用。机械臂与吸盘严格照抄
``control/scripts/controller.py`` 的调用方式（``AkaiFr`` / ``AkaiElectricSucker``），
舵机照抄 ``tools/hardware/测试舵机.py``，急停照抄 controller.py 的独立 XML-RPC
StopMotion 通道。

机械臂控制器支持多客户端（control_node 与 camera_node 已同时各持有一个
``AkaiFr``），因此这里懒创建一个 ``AkaiFr`` 并复用，不做裸 XML-RPC 重实现。
"""

import math
import threading
import time
import xmlrpc.client

import numpy as np
import serial
from akai_fr import AkaiElectricSucker, AkaiFr
from competition_lib.config import load_execution_config

from .constants import HAND_EYE_MATRIX_PATH


class DirectHardwareError(RuntimeError):
    """直连硬件操作失败。"""


# 机械臂错误码 → 中文说明（来自 akai_fr.AkaiFr.error_code_dict）。
ERROR_CODE_NAMES = {
    0: "成功",
    1: "驱动器错误",
    2: "关节位置超范围",
    3: "碰撞",
    4: "奇异点",
    5: "从站错误",
    6: "命令点错误",
    7: "IO 错误",
    8: "夹爪错误",
    9: "文件错误",
    10: "参数错误",
    11: "外部轴超范围",
    12: "关节配置警告",
    14: "关节命令点错误",
    112: "运动学无解",
}


class _TimeoutTransport(xmlrpc.client.Transport):
    """为急停用的独立 XML-RPC 连接设置超时（照抄 controller.py）。"""

    def __init__(self, timeout_seconds):
        super().__init__()
        self.timeout_seconds = float(timeout_seconds)

    def make_connection(self, host):
        connection = super().make_connection(host)
        connection.timeout = self.timeout_seconds
        return connection


class DirectHardware:
    """吸盘 / 舵机 / 机械臂复位 / 读位姿 / 急停的直连实现。"""

    # 停稳判定阈值，与 controller.yaml / controller.py 默认值保持一致。
    LINEAR_SPEED_THRESHOLD_MM_S = 3.0
    ANGULAR_SPEED_THRESHOLD_DEG_S = 3.0
    POSITION_TOLERANCE_MM = 1.0
    ORIENTATION_TOLERANCE_DEG = 0.5
    STABLE_DURATION_SECONDS = 0.01
    POLL_INTERVAL_SECONDS = 0.02
    STABLE_TIMEOUT_SECONDS = 20.0

    def __init__(self, panel_config):
        manual = panel_config["manual_control"]
        self.arm_ip = str(manual["arm_ip"])
        self.arm_port = int(manual["arm_port"])
        self.servo_port = str(manual["servo_port"])
        self.servo_baudrate = int(manual["servo_baudrate"])
        self.stop_rpc_timeout = 2.0
        self._arm = None
        self._sucker = None
        self._arm_lock = threading.RLock()
        self._servo = None
        self._servo_lock = threading.RLock()

    # ------------------------------------------------------------ SDK 实例
    def _ensure_arm(self):
        """懒创建一个 AkaiFr 并复用，照 controller.py 的初始化。"""
        if self._arm is not None:
            return self._arm
        with self._arm_lock:
            if self._arm is not None:
                return self._arm
            arm = AkaiFr()
            wrist_to_camera = np.load(HAND_EYE_MATRIX_PATH)
            if wrist_to_camera.shape != (4, 4) or not np.all(np.isfinite(wrist_to_camera)):
                raise DirectHardwareError("手眼标定矩阵必须是有限的 4x4 矩阵")
            arm.set_tmat_wrist2camera(wrist_to_camera)
            arm.set_tcf(1, [0, 0, 0, 0, 0, 0])
            self._sucker = AkaiElectricSucker(arm)
            self._arm = arm
            return self._arm

    @staticmethod
    def _error_name(code):
        return ERROR_CODE_NAMES.get(int(code), f"错误码 {code}")

    @staticmethod
    def _rpc_value(name, result, value_length=None):
        """解析厂商 SDK 返回（[错误码, 值]），照抄 controller.py 的 _rpc_value。"""
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise DirectHardwareError(f"{name} 返回格式异常: {result!r}")
        if result[0] != 0:
            raise DirectHardwareError(f"{name} 查询失败，错误码 {result[0]}")
        value = result[1]
        if value_length is not None:
            if not isinstance(value, (tuple, list)) or len(value) != value_length:
                raise DirectHardwareError(f"{name} 数据格式异常: {value!r}")
            return [float(item) for item in value]
        return value

    # ------------------------------------------------------------------ 吸盘
    def set_suction(self, state):
        """state: 0=吸气，1=喷气，2=关闭。照 controller.py set_suction。"""
        state = int(state)
        self._ensure_arm()
        sucker = self._sucker
        try:
            if state == 0:  # 吸气：阀关 + 泵开
                sucker.set_solenoid_valve(False)
                sucker.set_pump_motor(True)
                message = "吸盘开始吸气"
            elif state == 1:  # 喷气：阀开 + 泵开
                sucker.set_solenoid_valve(True)
                sucker.set_pump_motor(True)
                message = "吸盘开始喷气"
            elif state == 2:  # 关闭：阀关 + 泵关
                sucker.set_solenoid_valve(False)
                sucker.set_pump_motor(False)
                message = "吸盘已关闭"
            else:
                raise DirectHardwareError("吸盘状态只能是吸气、喷气或关闭")
        except DirectHardwareError:
            raise
        except Exception as exc:
            raise DirectHardwareError(f"吸盘控制失败：{exc}") from exc
        return {"success": True, "message": message, "state": state}

    # ------------------------------------------------------------------ 舵机
    def _ensure_servo(self):
        """懒打开舵机串口并长驻复用，照 controller.py 的 servo_serial。"""
        if self._servo is not None:
            return self._servo
        with self._servo_lock:
            if self._servo is not None:
                return self._servo
            try:
                connection = serial.Serial(self.servo_port, self.servo_baudrate, timeout=1)
            except serial.SerialException as exc:
                raise DirectHardwareError(f"舵机串口打开失败：{exc}") from None
            self._servo = connection
            return self._servo

    def release_servo(self):
        """关闭长驻的舵机串口，让 control_node 能重新打开（照 controller.py close）。"""
        with self._servo_lock:
            if self._servo is not None:
                try:
                    self._servo.close()
                finally:
                    self._servo = None

    def release(self):
        """释放直连层占用的硬件资源（当前只有舵机串口）。"""
        self.release_servo()

    def rotate_tool(self, angle_deg):
        """照 controller.py rotate_tool：长驻串口 write，不每次开关串口。"""
        angle = float(angle_deg)
        if not math.isfinite(angle) or not 0.0 <= angle <= 360.0:
            raise DirectHardwareError("舵机角度必须位于 0～360°")
        connection = self._ensure_servo()
        connection.write(f"{angle}E".encode("ascii"))
        connection.flush()
        return {"success": True, "message": "舵机指令已发送", "angle_deg": angle}

    # ------------------------------------------------------------ 机械臂复位
    @staticmethod
    def _angle_error(actual, target):
        return abs((float(actual) - float(target) + 180.0) % 360.0 - 180.0)

    def _motion_state(self, arm):
        """读取运动完成信号与 TCP 合速度，照 controller.py 的 SDK 查询。"""
        motion_done = bool(self._rpc_value("GetRobotMotionDone", arm.arm.GetRobotMotionDone()))
        speeds = self._rpc_value(
            "GetActualTCPCompositeSpeed", arm.arm.GetActualTCPCompositeSpeed(), value_length=2
        )
        linear_speed = abs(float(speeds[0]))
        angular_speed = abs(float(speeds[1]))
        stopped = (
            motion_done
            and linear_speed <= self.LINEAR_SPEED_THRESHOLD_MM_S
            and angular_speed <= self.ANGULAR_SPEED_THRESHOLD_DEG_S
        )
        return stopped, linear_speed, angular_speed

    def move_arm(self, pose, speed, wait_until_stable=True):
        """照 controller.py move_arm：set_speed → MoveL → 停稳轮询，带 Z 钳制。"""
        pose = [float(value) for value in pose]
        if len(pose) != 6 or not all(math.isfinite(value) for value in pose):
            raise DirectHardwareError("机械臂位姿必须包含 6 个有限数值")
        speed = int(speed)
        if speed <= 0:
            raise DirectHardwareError("机械臂速度必须大于 0")

        try:
            minimum_tcp_z_mm = float(load_execution_config().minimum_tcp_z_mm)
        except Exception as exc:
            raise DirectHardwareError(
                f"无法读取唯一 TCP 高度配置 motion.minimum_tcp_z_mm：{exc}"
            ) from exc

        z_was_clamped = False
        requested_z = pose[2]
        if pose[2] < minimum_tcp_z_mm:
            pose[2] = minimum_tcp_z_mm
            z_was_clamped = True

        arm = self._ensure_arm()

        # 运行前查运动状态，避免与上一段未停稳的运动冲突。
        stopped, _linear, _angular = self._motion_state(arm)
        if not stopped:
            raise DirectHardwareError("机械臂仍在运动，请等待停稳后再操作")

        arm.set_speed(speed)
        motion_started_at = time.monotonic()
        result = arm.arm.MoveL(pose, tool=0, user=0, vel=speed, blendR=-1.0)
        if (isinstance(result, bool) and not result) or (
            not isinstance(result, bool) and result != 0
        ):
            raise DirectHardwareError(
                f"MoveL 失败：{self._error_name(result)}，目标位姿={pose}"
            )

        if wait_until_stable:
            self._wait_until_stable(arm, pose, motion_started_at)

        if z_was_clamped:
            raise DirectHardwareError(
                f"请求 Z={requested_z:.2f} mm 低于安全下限 "
                f"{minimum_tcp_z_mm:.2f} mm，已调整到 {pose[2]:.2f} mm，"
                "已完成运动，但本次运动判定失败"
            )
        return {"success": True, "message": "机械臂运动完成", "pose": pose}

    def _wait_until_stable(self, arm, target_pose, motion_started_at):
        """等待机械臂到达目标并停稳，照 controller.py 的 _wait_until_arm_stable。"""
        deadline = time.monotonic() + self.STABLE_TIMEOUT_SECONDS
        stable_since = None
        while True:
            motion_done = bool(self._rpc_value("GetRobotMotionDone", arm.arm.GetRobotMotionDone()))
            speeds = self._rpc_value(
                "GetActualTCPCompositeSpeed", arm.arm.GetActualTCPCompositeSpeed(), value_length=2
            )
            actual_pose = self._rpc_value("GetActualTCPPose", arm.arm.GetActualTCPPose(), value_length=6)
            linear_speed = abs(float(speeds[0]))
            angular_speed = abs(float(speeds[1]))

            position_error = math.dist(actual_pose[:3], target_pose[:3])
            orientation_error = max(
                self._angle_error(actual_pose[index], target_pose[index])
                for index in range(3, 6)
            )
            is_stable = (
                motion_done
                and linear_speed <= self.LINEAR_SPEED_THRESHOLD_MM_S
                and angular_speed <= self.ANGULAR_SPEED_THRESHOLD_DEG_S
                and position_error <= self.POSITION_TOLERANCE_MM
                and orientation_error <= self.ORIENTATION_TOLERANCE_DEG
            )

            now = time.monotonic()
            if is_stable:
                if stable_since is None:
                    stable_since = now
                if now - stable_since >= self.STABLE_DURATION_SECONDS:
                    return
            else:
                stable_since = None

            if now >= deadline:
                raise DirectHardwareError(
                    "等待机械臂停稳超时："
                    f"motion_done={motion_done}, 线速度={linear_speed:.3f} mm/s, "
                    f"姿态速度={angular_speed:.3f} °/s, "
                    f"位置误差={position_error:.3f} mm, "
                    f"姿态误差={orientation_error:.3f}°"
                )
            time.sleep(self.POLL_INTERVAL_SECONDS)

    # ---------------------------------------------------------------- 读位姿
    def get_pose(self):
        """照 controller.py get_actual_pose：TCP 位姿 + 相机光心位姿。"""
        arm = self._ensure_arm()
        tcp_pose = self._rpc_value("GetActualTCPPose", arm.arm.GetActualTCPPose(), value_length=6)
        success, camera_pose = arm.get_camera_pose()
        if not success or camera_pose is None:
            raise DirectHardwareError("获取相机光心位姿失败")
        camera_pose = [float(value) for value in camera_pose]
        if len(camera_pose) != 6 or not all(math.isfinite(value) for value in camera_pose):
            raise DirectHardwareError("相机光心位姿无效")
        return {
            "success": True,
            "tcp_pose": [float(value) for value in tcp_pose],
            "camera_pose": camera_pose,
            "message": "读取 TCP 与相机光心位姿成功",
        }

    # ------------------------------------------------------------ 急停 / 停止
    def _create_stop_rpc(self):
        """独立 XML-RPC 连接，使停止不等待正在阻塞的 MoveL（照 controller.py）。"""
        transport = _TimeoutTransport(self.stop_rpc_timeout)
        return xmlrpc.client.ServerProxy(
            f"http://{self.arm_ip}:{self.arm_port}", transport=transport, allow_none=True
        )

    @staticmethod
    def _raw_rpc_result(name, result, value_count=0):
        """解析原始 XML-RPC 返回，照抄 controller.py 的 _raw_rpc_result。"""
        if isinstance(result, (tuple, list)):
            values = list(result)
        else:
            values = [result]
        if not values or values[0] != 0:
            raise DirectHardwareError(f"{name} 失败，错误码 {values[0] if values else '空返回'}")
        if value_count == 0:
            return None
        payload = values[1:]
        if len(payload) == 1 and isinstance(payload[0], (tuple, list)):
            payload = list(payload[0])
        if len(payload) < value_count:
            raise DirectHardwareError(f"{name} 返回数据不足: {result!r}")
        return payload[:value_count]

    def _query_motion_state_raw(self, rpc):
        motion_done = bool(
            self._raw_rpc_result("GetRobotMotionDone", rpc.GetRobotMotionDone(), value_count=1)[0]
        )
        speeds = self._raw_rpc_result(
            "GetActualTCPCompositeSpeed", rpc.GetActualTCPCompositeSpeed(1), value_count=2
        )
        linear_speed, angular_speed = (abs(float(value)) for value in speeds)
        stopped = (
            motion_done
            and linear_speed <= self.LINEAR_SPEED_THRESHOLD_MM_S
            and angular_speed <= self.ANGULAR_SPEED_THRESHOLD_DEG_S
        )
        return stopped, linear_speed, angular_speed

    def stop_motion(self, timeout_seconds=2.0):
        """独立连接调 StopMotion 并确认停稳，照 controller.py stop_arm。"""
        rpc = self._create_stop_rpc()
        self._raw_rpc_result("StopMotion", rpc.StopMotion())
        deadline = time.monotonic() + timeout_seconds
        last_state = None
        while time.monotonic() < deadline:
            last_state = self._query_motion_state_raw(rpc)
            if last_state[0]:
                return {
                    "success": True,
                    "stop_latched": True,
                    "message": "机械臂已停止，停止锁保持",
                }
            time.sleep(self.POLL_INTERVAL_SECONDS)
        raise DirectHardwareError("StopMotion 后未在限定时间内确认停稳")

    def is_stopped(self):
        rpc = self._create_stop_rpc()
        stopped, _linear, _angular = self._query_motion_state_raw(rpc)
        return stopped
