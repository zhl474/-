#!/home/zhl/fr3env/fr3env/bin/python
"""机械臂、末端舵机与电子吸盘控制节点。"""

import math
import os
import threading
import time
import xmlrpc.client

import numpy as np
import rospy
import serial
import yaml
from serial.tools import list_ports

from akai_fr import AkaiElectricSucker, AkaiFr
from control.srv import (
    ClearArmStop,
    ClearArmStopResponse,
    GetActualPose,
    GetActualPoseResponse,
    GetControlStatus,
    GetControlStatusResponse,
    MoveArm,
    MoveArmResponse,
    RotateTool,
    RotateToolResponse,
    SetSuction,
    SetSuctionResponse,
    StopArm,
    StopArmResponse,
)


PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, ".."))
DEFAULT_HAND_EYE_MATRIX = os.path.join(SRC_DIR, "camera", "config", "T_wrist2camera.npy")
EXECUTION_CONFIG_PATH = os.path.join(SRC_DIR, "competition", "config", "execution.yaml")


def _load_minimum_tcp_z_mm(config_path=EXECUTION_CONFIG_PATH):
    """从唯一执行配置读取 TCP 最低安全高度，不提供硬编码后备值。"""
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            execution_config = yaml.safe_load(config_file) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(
            f"无法读取唯一 TCP 高度配置 {config_path}: {exc}"
        ) from exc

    if not isinstance(execution_config, dict):
        raise ValueError("execution.yaml 顶层必须是字典")
    motion_config = execution_config.get("motion")
    if not isinstance(motion_config, dict) or "minimum_tcp_z_mm" not in motion_config:
        raise ValueError(
            "execution.yaml 缺少唯一安全高度字段 motion.minimum_tcp_z_mm"
        )
    raw_value = motion_config["minimum_tcp_z_mm"]
    if isinstance(raw_value, bool):
        raise ValueError("motion.minimum_tcp_z_mm 必须是大于 0 的有限数值")
    try:
        minimum_tcp_z_mm = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("motion.minimum_tcp_z_mm 必须是大于 0 的有限数值") from exc
    if not math.isfinite(minimum_tcp_z_mm) or minimum_tcp_z_mm <= 0.0:
        raise ValueError("motion.minimum_tcp_z_mm 必须是大于 0 的有限数值")
    return minimum_tcp_z_mm


class _TimeoutTransport(xmlrpc.client.Transport):
    """为高优先级停止连接设置独立超时，避免网络异常时永久阻塞。"""

    def __init__(self, timeout_seconds):
        super().__init__()
        self.timeout_seconds = float(timeout_seconds)

    def make_connection(self, host):
        connection = super().make_connection(host)
        connection.timeout = self.timeout_seconds
        return connection


class ControlNode:
    SUCTION_UNKNOWN = -1

    def __init__(self):
        self.minimum_tcp_z_mm = _load_minimum_tcp_z_mm()
        rospy.loginfo(
            "已从唯一执行配置读取 TCP 最低安全高度：%.3f mm（%s）",
            self.minimum_tcp_z_mm,
            EXECUTION_CONFIG_PATH,
        )
        self.move_arm_timing_debug = bool(rospy.get_param("~move_arm_timing_debug", False))
        stability = rospy.get_param("~arm_stability", {})
        self.stable_timeout = float(stability.get("timeout_seconds", 5.0))
        self.stable_poll_interval = float(stability.get("poll_interval_seconds", 0.005))
        self.stable_duration = float(stability.get("duration_seconds", 0.01))
        self.linear_speed_threshold = float(stability.get("linear_speed_threshold_mm_s", 3.0))
        self.angular_speed_threshold = float(stability.get("angular_speed_threshold_deg_s", 3.0))
        self.position_tolerance = float(stability.get("position_tolerance_mm", 1.0))
        self.orientation_tolerance = float(stability.get("orientation_tolerance_deg", 0.5))
        stop_motion = rospy.get_param("~stop_motion", {})
        self.stop_rpc_timeout = float(stop_motion.get("rpc_timeout_seconds", 1.0))
        self.stop_confirm_timeout = float(stop_motion.get("confirm_timeout_seconds", 2.0))
        if self.stop_rpc_timeout <= 0.0 or self.stop_confirm_timeout <= 0.0:
            raise ValueError("StopMotion 通信和确认超时必须大于 0")
        self.state_lock = threading.RLock()
        self.stop_latched = False
        self.motion_active = False
        self.commanded_suction_state = self.SUCTION_UNKNOWN
        self.servo_target_known = False
        self.servo_target_angle_deg = 0.0
        self.arm = AkaiFr()
        self.robot_ip = str(
            rospy.get_param(
                "~robot_ip",
                getattr(getattr(self.arm, "arm", None), "ip_address", "192.168.58.2"),
            )
        ).strip()
        if not self.robot_ip:
            raise ValueError("robot_ip 不能为空")
        hand_eye_matrix_path = rospy.get_param("~hand_eye_matrix", DEFAULT_HAND_EYE_MATRIX)
        wrist_to_camera = np.load(hand_eye_matrix_path)
        if wrist_to_camera.shape != (4, 4) or not np.all(np.isfinite(wrist_to_camera)):
            raise ValueError("手眼标定矩阵必须是有限的 4x4 矩阵")
        # 读取实测相机光心位姿时使用，与相机节点保持同一份手眼标定。
        self.arm.set_tmat_wrist2camera(wrist_to_camera)
        self.arm.set_speed(80)
        self.arm.set_tcf(1, [0, 0, 0, 0, 0, 0])
        self.sucker = AkaiElectricSucker(self.arm)
        self.servo_serial = self._open_servo_serial()

        self.arm_service = rospy.Service("/control/move_arm", MoveArm, self.move_arm)
        self.actual_pose_service = rospy.Service(
            "/control/get_actual_pose", GetActualPose, self.get_actual_pose
        )
        self.motor_service = rospy.Service("/control/rotate_tool", RotateTool, self.rotate_tool)
        self.suction_service = rospy.Service("/control/set_suction", SetSuction, self.set_suction)
        self.stop_arm_service = rospy.Service(
            "/control/stop_arm", StopArm, self.stop_arm
        )
        self.clear_arm_stop_service = rospy.Service(
            "/control/clear_arm_stop", ClearArmStop, self.clear_arm_stop
        )
        self.control_status_service = rospy.Service(
            "/control/get_status", GetControlStatus, self.get_control_status
        )
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

    def _ensure_runtime_state(self):
        """兼容不执行构造函数的单元测试，同时集中初始化运行状态。"""
        if not hasattr(self, "state_lock"):
            self.state_lock = threading.RLock()
        defaults = {
            "stop_latched": False,
            "motion_active": False,
            "commanded_suction_state": self.SUCTION_UNKNOWN,
            "servo_target_known": False,
            "servo_target_angle_deg": 0.0,
            "robot_ip": "192.168.58.2",
            "stop_rpc_timeout": 1.0,
            "stop_confirm_timeout": 2.0,
            "stable_poll_interval": 0.005,
            "linear_speed_threshold": 3.0,
            "angular_speed_threshold": 3.0,
        }
        for name, value in defaults.items():
            if not hasattr(self, name):
                setattr(self, name, value)

    def _create_stop_rpc(self):
        """每次创建独立 XML-RPC 连接，使停止命令不等待正在执行的 MoveL。"""
        self._ensure_runtime_state()
        transport = _TimeoutTransport(self.stop_rpc_timeout)
        return xmlrpc.client.ServerProxy(
            f"http://{self.robot_ip}:20003",
            transport=transport,
            allow_none=True,
        )

    @staticmethod
    def _raw_rpc_result(name, result, value_count=0):
        """解析控制器原始 XML-RPC 返回值，并兼容测试替身的嵌套列表。"""
        if isinstance(result, (tuple, list)):
            values = list(result)
        else:
            values = [result]
        if not values or values[0] != 0:
            code = values[0] if values else "空返回"
            raise RuntimeError(f"{name} 失败，错误码: {code}")
        if value_count == 0:
            return None
        payload = values[1:]
        if len(payload) == 1 and isinstance(payload[0], (tuple, list)):
            payload = list(payload[0])
        if len(payload) < value_count:
            raise RuntimeError(f"{name} 返回数据不足: {result!r}")
        return payload[:value_count]

    def _query_motion_state(self, rpc=None):
        """读取运动完成信号和 TCP 合速度，返回（是否已停、线速度、姿态速度）。"""
        rpc = rpc or self._create_stop_rpc()
        motion_done = bool(
            self._raw_rpc_result(
                "GetRobotMotionDone",
                rpc.GetRobotMotionDone(),
                value_count=1,
            )[0]
        )
        speeds = self._raw_rpc_result(
            "GetActualTCPCompositeSpeed",
            rpc.GetActualTCPCompositeSpeed(1),
            value_count=2,
        )
        linear_speed, angular_speed = (abs(float(value)) for value in speeds)
        stopped = (
            motion_done
            and linear_speed <= self.linear_speed_threshold
            and angular_speed <= self.angular_speed_threshold
        )
        return stopped, linear_speed, angular_speed

    def _wait_until_motion_stopped(self, rpc):
        """StopMotion 后等待控制器反馈真正停止，并覆盖命令发出瞬间的竞态。

        MoveL 使用厂商客户端时可能正在另一个线程里阻塞。停止锁会先阻止所有
        后续请求；这里还会周期性重发 StopMotion，确保一个恰好越过锁检查、
        但晚于首条 StopMotion 到达控制器的 MoveL 也会被终止。
        """
        deadline = time.monotonic() + self.stop_confirm_timeout
        last_state = None
        last_restop_at = 0.0
        while time.monotonic() < deadline:
            last_state = self._query_motion_state(rpc)
            with self.state_lock:
                command_handler_active = bool(self.motion_active)
            if last_state[0] and not command_handler_active:
                return last_state
            now = time.monotonic()
            if now - last_restop_at >= 0.05:
                self._raw_rpc_result("StopMotion", rpc.StopMotion())
                last_restop_at = now
            time.sleep(max(0.005, self.stable_poll_interval))
        if last_state is None:
            raise TimeoutError("StopMotion 后没有取得运动状态")
        with self.state_lock:
            command_handler_active = bool(self.motion_active)
        raise TimeoutError(
            "StopMotion 后未在限定时间内确认停稳："
            f"线速度={last_state[1]:.3f} mm/s，"
            f"姿态速度={last_state[2]:.3f} °/s，"
            f"运动服务仍在执行={command_handler_active}"
        )

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
        self._ensure_runtime_state()
        with self.state_lock:
            if self.stop_latched:
                raise RuntimeError("等待机械臂停稳时收到停止锁，本次运动已中止")
        if motion_started_at is None:
            motion_started_at = time.monotonic()
        deadline = time.monotonic() + self.stable_timeout
        stable_since = None
        motion_finished_at = None

        while not rospy.is_shutdown():
            with self.state_lock:
                if self.stop_latched:
                    raise RuntimeError("等待机械臂停稳时收到停止锁，本次运动已中止")
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
        self._ensure_runtime_state()
        with self.state_lock:
            if self.stop_latched:
                return MoveArmResponse(
                    success=False,
                    message="机械臂停止锁已生效；请检查现场并解除停止锁后再运动",
                )
        pose = [float(value) for value in request.pose]
        blend_enabled = bool(getattr(request, "blend_enabled", False))
        blend_radius_mm = float(getattr(request, "blend_radius_mm", 0.0))
        z_was_clamped = False
        if len(pose) != 6 or not self._finite(pose):
            return MoveArmResponse(success=False, message="机械臂位姿必须包含 6 个有限数值")
        if request.speed <= 0:
            return MoveArmResponse(success=False, message="机械臂速度必须大于 0")
        if (
            not math.isfinite(blend_radius_mm)
            or blend_radius_mm < 0.0
            or blend_radius_mm > 1000.0
        ):
            return MoveArmResponse(
                success=False,
                message="圆滑半径必须是 0 到 1000 mm 之间的有限数值",
            )
        if blend_enabled and blend_radius_mm == 0.0:
            return MoveArmResponse(
                success=False,
                message="启用圆滑过渡时，圆滑半径必须大于 0 mm",
            )
        if blend_enabled and request.wait_until_stable:
            return MoveArmResponse(
                success=False,
                message="圆滑过渡点不会精确到位，不能同时等待该点停稳",
            )
        sdk_blend_radius_mm = blend_radius_mm if blend_enabled else -1.0
        if pose[2] < self.minimum_tcp_z_mm:
            requested_z = pose[2]
            pose[2] = self.minimum_tcp_z_mm
            z_was_clamped = True
            # 使用错误级别日志，让终端以红色突出显示安全钳制警告。
            rospy.logerr(
                "安全警告：目标 Z=%.2f mm 低于安全下限 %.2f mm，已自动调整为 %.2f mm 后继续运动",
                requested_z,
                self.minimum_tcp_z_mm,
                pose[2],
            )
        try:
            with self.state_lock:
                if self.stop_latched:
                    return MoveArmResponse(
                        success=False,
                        message="机械臂停止锁已生效；本次运动已拒绝",
                    )
                self.motion_active = True
            self.arm.set_speed(int(request.speed))
            motion_started_at = time.monotonic()
            result = self.arm.arm.MoveL(
                pose,
                tool=0,
                user=0,
                vel=int(request.speed),
                blendR=sdk_blend_radius_mm,
            )
            if (isinstance(result, bool) and not result) or (
                not isinstance(result, bool) and result != 0
            ):
                return MoveArmResponse(success=False, message=f"机械臂 MoveL 返回失败: {result!r}")
            with self.state_lock:
                if self.stop_latched:
                    return MoveArmResponse(
                        success=False,
                        message="机械臂运动已被停止锁中止，不会继续后续动作",
                    )
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
            with self.state_lock:
                if self.stop_latched:
                    return MoveArmResponse(
                        success=False,
                        message="机械臂运动已被停止锁中止，不会继续后续动作",
                    )
            if z_was_clamped:
                motion_status = "已提交圆滑过渡运动" if blend_enabled else "已完成运动"
                return MoveArmResponse(
                    success=False,
                    message=(
                        f"请求 Z={requested_z:.2f} mm 低于安全下限 "
                        f"{self.minimum_tcp_z_mm:.2f} mm，已调整到 {pose[2]:.2f} mm，"
                        f"{motion_status}，但本次运动判定失败"
                    ),
                )
            if blend_enabled:
                return MoveArmResponse(success=True, message="机械臂圆滑过渡运动已提交")
            return MoveArmResponse(success=True, message="机械臂运动完成")
        except Exception as exc:
            rospy.logerr("机械臂运动异常: %s", exc)
            return MoveArmResponse(success=False, message=str(exc))
        finally:
            with self.state_lock:
                self.motion_active = False

    def get_actual_pose(self, _request):
        """读取控制器实测 TCP 与相机光心在机器人基坐标系下的位姿。"""
        zero_pose = [0.0] * 6
        try:
            tcp_pose = self._rpc_value(
                "GetActualTCPPose",
                self.arm.arm.GetActualTCPPose(),
                value_length=6,
            )
            success, camera_pose = self.arm.get_camera_pose()
            if not success or camera_pose is None:
                raise RuntimeError("获取相机光心位姿失败")
            camera_pose = [float(value) for value in camera_pose]
            if len(camera_pose) != 6 or not self._finite(camera_pose):
                raise RuntimeError(f"相机光心位姿无效: {camera_pose}")
            return GetActualPoseResponse(
                success=True,
                tcp_pose=tcp_pose,
                camera_pose=camera_pose,
                message="读取实测 TCP 与相机光心位姿成功",
            )
        except Exception as exc:
            rospy.logwarn("读取实测位姿失败: %s", exc)
            return GetActualPoseResponse(
                success=False,
                tcp_pose=zero_pose,
                camera_pose=zero_pose,
                message=str(exc),
            )

    def rotate_tool(self, request):
        self._ensure_runtime_state()
        with self.state_lock:
            if self.stop_latched:
                return RotateToolResponse(
                    success=False,
                    message="机械臂停止锁已生效；舵机指令已拒绝",
                )
        angle = float(request.angle_deg)
        if not math.isfinite(angle) or not 0.0 <= angle <= 360.0:
            return RotateToolResponse(success=False, message="舵机角度必须在 0 到 360 度之间")
        try:
            with self.state_lock:
                # 串口写入很短，和停止锁共用临界区可明确保证指令先后关系。
                if self.stop_latched:
                    return RotateToolResponse(
                        success=False,
                        message="机械臂停止锁已生效；舵机指令已拒绝",
                    )
                self.servo_serial.write(f"{angle}E".encode("ascii"))
                self.servo_target_known = True
                self.servo_target_angle_deg = angle
            return RotateToolResponse(success=True, message="舵机指令已发送")
        except Exception as exc:
            rospy.logerr("舵机指令发送失败: %s", exc)
            return RotateToolResponse(success=False, message=str(exc))

    def set_suction(self, request):
        self._ensure_runtime_state()
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
            with self.state_lock:
                self.commanded_suction_state = int(request.state)
            return SetSuctionResponse(success=True, message=message)
        except Exception as exc:
            rospy.logerr("吸盘控制失败: %s", exc)
            return SetSuctionResponse(success=False, message=str(exc))

    def stop_arm(self, _request):
        """锁存停止状态并通过独立 XML-RPC 连接终止当前机械臂运动。"""
        self._ensure_runtime_state()
        with self.state_lock:
            self.stop_latched = True
        try:
            rpc = self._create_stop_rpc()
            self._raw_rpc_result("StopMotion", rpc.StopMotion())
            self._wait_until_motion_stopped(rpc)
            rospy.logwarn("机械臂软件停止已确认；停止锁保持生效")
            return StopArmResponse(
                success=True,
                stop_latched=True,
                message="机械臂运动已停止；吸盘状态保持，停止锁仍生效",
            )
        except Exception as exc:
            rospy.logerr("机械臂软件停止未确认: %s", exc)
            return StopArmResponse(
                success=False,
                stop_latched=True,
                message=f"软件停止未确认，请立即准备使用物理急停：{exc}",
            )

    def clear_arm_stop(self, _request):
        """仅在确认机械臂已经停止后解除锁存，不自动复位或恢复旧任务。"""
        self._ensure_runtime_state()
        with self.state_lock:
            if not self.stop_latched:
                return ClearArmStopResponse(
                    success=True,
                    stop_latched=False,
                    message="停止锁当前未生效",
                )
            if self.motion_active:
                return ClearArmStopResponse(
                    success=False,
                    stop_latched=True,
                    message="旧的机械臂运动服务仍在执行，不能解除停止锁",
                )
        try:
            stopped, linear_speed, angular_speed = self._query_motion_state()
            if not stopped:
                return ClearArmStopResponse(
                    success=False,
                    stop_latched=True,
                    message=(
                        "机械臂尚未确认停稳，不能解除停止锁："
                        f"线速度={linear_speed:.3f} mm/s，"
                        f"姿态速度={angular_speed:.3f} °/s"
                    ),
                )
            with self.state_lock:
                if self.motion_active:
                    return ClearArmStopResponse(
                        success=False,
                        stop_latched=True,
                        message="旧的机械臂运动服务尚未结束，停止锁保持生效",
                    )
                self.stop_latched = False
            return ClearArmStopResponse(
                success=True,
                stop_latched=False,
                message="停止锁已解除；旧任务不会自动恢复",
            )
        except Exception as exc:
            return ClearArmStopResponse(
                success=False,
                stop_latched=True,
                message=f"无法确认机械臂停稳，停止锁保持生效：{exc}",
            )

    def get_control_status(self, _request):
        """返回控制节点的软件锁存与最近命令状态；吸盘和舵机均无位置反馈。"""
        self._ensure_runtime_state()
        motion_state_known = False
        motion_done = False
        message = "控制节点状态已读取"
        try:
            motion_done = bool(self._query_motion_state()[0])
            motion_state_known = True
        except Exception as exc:
            message = f"控制状态已读取，但机器人运动状态未知：{exc}"
        with self.state_lock:
            if self.motion_active:
                motion_done = False
                message = "机械臂运动服务仍在执行"
            return GetControlStatusResponse(
                success=True,
                stop_latched=bool(self.stop_latched),
                motion_state_known=motion_state_known,
                motion_done=motion_done,
                commanded_suction_state=int(self.commanded_suction_state),
                servo_target_known=bool(self.servo_target_known),
                servo_target_angle_deg=float(self.servo_target_angle_deg),
                message=message,
            )

    def close(self):
        if getattr(self, "servo_serial", None) is not None and self.servo_serial.is_open:
            self.servo_serial.close()


def main():
    rospy.init_node("control_node")
    ControlNode()
    rospy.spin()


if __name__ == "__main__":
    main()
