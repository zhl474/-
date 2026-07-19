"""粗定位与视觉伺服抓放任务状态机。"""

from enum import Enum
import math
import time

import rospy

from .config import load_execution_config, load_visual_servo_config
from .ros_clients import RobotClients
from .servo_csv_logger import DEFAULT_SERVO_CSV_OUTPUT_DIR, ServoCsvLogger
from .visual_servo import apply_camera_to_sucker_offset, run_offset_visual_servo_alignment


class TaskState(Enum):
    IDLE = "空闲"
    PREPARING = "准备任务"
    PICK_COARSE = "抓取粗定位"
    PICK_ALIGN = "抓取视觉对准"
    PICKING = "下探抓取"
    PLACE_COARSE = "摆放粗定位"
    PLACE_ALIGN = "摆放视觉对准"
    PLACING = "下放摆放"
    COMPLETED = "完成"
    FAILED = "失败"


class ServoAnglePlanner:
    def __init__(self, config, rotate_func):
        self.last_angle = config.initial_motor_angle_deg
        self.velocity = config.motor_velocity_deg_per_sec
        self.lower_margin = config.motor_lower_margin_deg
        self.upper_margin = config.motor_upper_margin_deg
        self.rotate_func = rotate_func

    def plan(self, rotation_delta_deg):
        wait_sec = 0.0
        target = self.last_angle + float(rotation_delta_deg)
        if target > 360.0:
            safe_pick_angle = self.upper_margin - float(rotation_delta_deg)
            wait_sec = abs(self.last_angle - safe_pick_angle) / self.velocity
            self.rotate_func(safe_pick_angle)
            self.last_angle = safe_pick_angle
        elif target < 0.0:
            safe_pick_angle = self.lower_margin - float(rotation_delta_deg)
            wait_sec = abs(self.last_angle - safe_pick_angle) / self.velocity
            self.rotate_func(safe_pick_angle)
            self.last_angle = safe_pick_angle
        return self.last_angle, self.last_angle + float(rotation_delta_deg), wait_sec

    def commit_place_angle(self, place_angle):
        self.rotate_func(place_angle)
        self.last_angle = float(place_angle)


class TaskRunner:
    def __init__(
        self,
        clients=None,
        execution_config=None,
        visual_config=None,
        servo_csv_output_dir=DEFAULT_SERVO_CSV_OUTPUT_DIR,
        servo_csv_logger=None,
    ):
        self.clients = clients or RobotClients()
        self.config = execution_config or load_execution_config()
        self.visual_config = visual_config or load_visual_servo_config()
        self.servo_csv_logger = servo_csv_logger or ServoCsvLogger(servo_csv_output_dir)
        self.angle_planner = ServoAnglePlanner(self.config, self.clients.rotate_tool)
        self.state = TaskState.IDLE
        self.holding_block = False
        self.execution_start_time = None

    def _set_state(self, state):
        self.state = state
        if state is TaskState.COMPLETED:
            rospy.loginfo("任务状态: %s", state.value)
            if self.execution_start_time is not None:
                elapsed_sec = time.monotonic() - self.execution_start_time
                rospy.loginfo("全部方块抓放完成，总耗时: %.2f 秒", elapsed_sec)
                self.execution_start_time = None
        elif state is TaskState.FAILED:
            rospy.logerr("任务状态: %s", state.value)

    def _align(self, offset_func, start_pose, log_label, event_callback=None):
        return run_offset_visual_servo_alignment(
            offset_func,
            self.clients.move_arm,
            start_pose,
            self.visual_config,
            speed=self.config.servo_speed,
            error_threshold_px=self.config.error_threshold_px,
            max_step_mm=self.config.max_step_mm,
            min_step_mm=self.config.min_step_mm,
            max_iter=self.config.max_iter,
            success_stable_frames=self.config.success_stable_frames,
            max_missed_frames=self.config.max_missed_frames,
            settle_sec=self.config.settle_sec,
            timing_debug=self.config.timing_debug,
            log_label=log_label,
            event_callback=event_callback,
        )

    @staticmethod
    def _finite_values(values, length):
        try:
            result = [float(value) for value in values]
        except (TypeError, ValueError):
            return [""] * length
        if len(result) != length or not all(math.isfinite(value) for value in result):
            return [""] * length
        return result

    @staticmethod
    def _pose_fields(values, prefix):
        """把六维位姿展开为 CSV 字段。"""
        x, y, z, roll, pitch, yaw = TaskRunner._finite_values(values, 6)
        return {
            f"{prefix}X": x,
            f"{prefix}Y": y,
            f"{prefix}Z": z,
            f"{prefix}R": roll,
            f"{prefix}P": pitch,
            f"{prefix}YAW": yaw,
        }

    def _servo_base_row(self, target, side, rough_pose, task_index):
        """整理同一任务所有事件共用的高位粗定位数据。"""
        prefix = "pick" if side == "pick" else "place"
        detected_x, detected_y = self._finite_values(
            getattr(target, f"{prefix}_high_detected_pixel_xy", None), 2
        )
        depth_x, depth_y = self._finite_values(
            getattr(target, f"{prefix}_high_depth_sample_pixel_xy", None), 2
        )
        center_x, center_y = self._finite_values(
            getattr(target, f"{prefix}_high_image_center_xy", None), 2
        )
        world_valid = bool(getattr(target, f"{prefix}_high_world_position_valid", False))
        world_position = self._finite_values(
            getattr(target, f"{prefix}_high_world_position", None), 3
        ) if world_valid else ["", "", ""]
        source = str(getattr(target, f"{prefix}_rough_localization_source", "") or "未提供")
        row = {
            "任务序号": "" if task_index is None else int(task_index),
            "方块类别": str(getattr(target, "category", "") or ""),
            "托盘行": getattr(target, "row", ""),
            "托盘列": getattr(target, "col", ""),
            "高位检测像素X": detected_x,
            "高位检测像素Y": detected_y,
            "深度采样像素X": depth_x,
            "深度采样像素Y": depth_y,
            "高位图像中心X": center_x,
            "高位图像中心Y": center_y,
            "高位世界坐标X": world_position[0],
            "高位世界坐标Y": world_position[1],
            "高位世界坐标Z": world_position[2],
            "高位世界坐标有效": world_valid,
            "粗定位来源": source,
        }
        row.update(self._pose_fields(rough_pose, "粗定位末端命令"))
        return row

    def _record_servo_event(self, target, side, rough_pose, task_index, event_data):
        """把闭环回调事件合并为方块或托盘 CSV 的一行。"""
        row = self._servo_base_row(target, side, rough_pose, task_index)
        row.update(event_data)
        self.servo_csv_logger.write("block" if side == "pick" else "board", row)

    def _read_actual_pose_for_csv(self):
        """读取实测位姿失败时只写入错误字段，绝不影响抓放流程。"""
        empty = {
            "实测TCP位置X": "",
            "实测TCP位置Y": "",
            "实测TCP位置Z": "",
            "实测TCP姿态R": "",
            "实测TCP姿态P": "",
            "实测TCP姿态YAW": "",
            "实测相机光心位置X": "",
            "实测相机光心位置Y": "",
            "实测相机光心位置Z": "",
            "实测相机光心姿态R": "",
            "实测相机光心姿态P": "",
            "实测相机光心姿态YAW": "",
        }
        get_actual_pose = getattr(self.clients, "get_actual_pose", None)
        if not callable(get_actual_pose):
            empty["实测位姿读取信息"] = "控制客户端未提供实测位姿服务"
            return empty
        try:
            response = get_actual_pose()
            if not getattr(response, "success", False):
                empty["实测位姿读取信息"] = str(getattr(response, "message", "未知错误"))
                return empty
            tcp_x, tcp_y, tcp_z, tcp_r, tcp_p, tcp_yaw = self._finite_values(response.tcp_pose, 6)
            camera_x, camera_y, camera_z, camera_r, camera_p, camera_yaw = self._finite_values(
                response.camera_pose, 6
            )
            return {
                "实测TCP位置X": tcp_x,
                "实测TCP位置Y": tcp_y,
                "实测TCP位置Z": tcp_z,
                "实测TCP姿态R": tcp_r,
                "实测TCP姿态P": tcp_p,
                "实测TCP姿态YAW": tcp_yaw,
                "实测相机光心位置X": camera_x,
                "实测相机光心位置Y": camera_y,
                "实测相机光心位置Z": camera_z,
                "实测相机光心姿态R": camera_r,
                "实测相机光心姿态P": camera_p,
                "实测相机光心姿态YAW": camera_yaw,
                "实测位姿读取信息": "成功",
            }
        except Exception as exc:
            empty["实测位姿读取信息"] = f"读取异常: {exc}"
            return empty

    def _response_csv_fields(self, response):
        """把最终低位检测响应转换为 CSV 字段。"""
        values = self._finite_values(
            [
                getattr(response, "px", None),
                getattr(response, "py", None),
                getattr(response, "dx_px", None),
                getattr(response, "dy_px", None),
            ],
            4,
        )
        target_x, target_y, error_x, error_y = values
        center_x = target_x - error_x if target_x != "" and error_x != "" else ""
        center_y = target_y - error_y if target_y != "" and error_y != "" else ""
        maximum_error = max(abs(error_x), abs(error_y)) if error_x != "" and error_y != "" else ""
        return {
            "低位目标像素X": target_x,
            "低位目标像素Y": target_y,
            "低位图像中心X": center_x,
            "低位图像中心Y": center_y,
            "像素误差X": error_x,
            "像素误差Y": error_y,
            "最大像素误差": maximum_error,
        }

    def _record_servo_result(
        self, target, side, rough_pose, task_index, success, command_pose, response, message
    ):
        """写入成功或失败的最终事件，并附上不影响流程的实测位姿结果。"""
        event_data = {
            "事件": "伺服成功" if success else "伺服失败",
            "伺服轮次": "",
            "识别成功": bool(success),
            "消息": str(message),
        }
        event_data.update(self._response_csv_fields(response))
        event_data.update(self._pose_fields(command_pose, "末端命令"))
        event_data.update(self._read_actual_pose_for_csv())
        self._record_servo_event(target, side, rough_pose, task_index, event_data)

    def _pick(self, target, task_index=None):
        if not getattr(target, "pick_surface_z_valid", False):
            raise RuntimeError("方块缺少有效深度高度，已取消抓取")
        pick_surface_z_mm = float(getattr(target, "pick_surface_z_mm", float("nan")))
        if not math.isfinite(pick_surface_z_mm):
            raise RuntimeError("方块表面深度高度无效，已取消抓取")

        _, place_angle, motor_wait = self.angle_planner.plan(target.rotation_delta_deg)
        rough_pose = list(target.pick_observation_pose)
        self._set_state(TaskState.PICK_COARSE)
        self.clients.move_arm(rough_pose, self.config.arm_speed, wait_until_stable=True)
        if motor_wait > 0:
            time.sleep(motor_wait)

        self._set_state(TaskState.PICK_ALIGN)
        self._record_servo_event(target, "pick", rough_pose, task_index, {
            "事件": "伺服开始", "伺服轮次": 0, "识别成功": "", "消息": "",
        })
        try:
            success, camera_pose, last_response, message = self._align(
                lambda: self.clients.detect_block_offset(target.category, target.detected_angle_deg),
                rough_pose,
                "方块视觉伺服",
                event_callback=lambda event: self._record_servo_event(
                    target, "pick", rough_pose, task_index, event
                ),
            )
        except Exception as exc:
            self._record_servo_result(target, "pick", rough_pose, task_index, False, rough_pose, None, str(exc))
            raise
        if not success:
            self._record_servo_result(
                target, "pick", rough_pose, task_index, False, camera_pose, last_response, message
            )
            raise RuntimeError(f"方块视觉伺服失败: {message}")
        self._record_servo_result(
            target, "pick", rough_pose, task_index, True, camera_pose, last_response, message
        )

        sucker_pose = apply_camera_to_sucker_offset(camera_pose, self.visual_config)
        sucker_pose[2] = self.config.lift_z
        self.clients.move_arm(sucker_pose, self.config.arm_speed)

        self._set_state(TaskState.PICKING)
        pick_pose = list(sucker_pose)
        pick_pose[2] = pick_surface_z_mm + self.config.pick_surface_offset_mm
        self.clients.move_arm(pick_pose, self.config.pick_speed)
        self.clients.set_suction(RobotClients.SUCK)
        self.holding_block = True
        self.clients.move_arm(sucker_pose, self.config.arm_speed)
        self.angle_planner.commit_place_angle(place_angle)

    def _place(self, target, task_index=None):
        rough_pose = list(target.place_observation_pose)
        self._set_state(TaskState.PLACE_COARSE)
        self.clients.move_arm(rough_pose, self.config.arm_speed, wait_until_stable=True)

        self._set_state(TaskState.PLACE_ALIGN)
        self._record_servo_event(target, "place", rough_pose, task_index, {
            "事件": "伺服开始", "伺服轮次": 0, "识别成功": "", "消息": "",
        })
        try:
            success, camera_pose, last_response, message = self._align(
                lambda: self.clients.detect_board_offset(target.row, target.col),
                rough_pose,
                "托盘视觉伺服",
                event_callback=lambda event: self._record_servo_event(
                    target, "place", rough_pose, task_index, event
                ),
            )
        except Exception as exc:
            self._record_servo_result(target, "place", rough_pose, task_index, False, rough_pose, None, str(exc))
            raise
        if not success:
            self._record_servo_result(
                target, "place", rough_pose, task_index, False, camera_pose, last_response, message
            )
            raise RuntimeError(f"托盘视觉伺服失败: {message}")
        self._record_servo_result(
            target, "place", rough_pose, task_index, True, camera_pose, last_response, message
        )

        place_pose = apply_camera_to_sucker_offset(camera_pose, self.visual_config)
        self.clients.move_arm(place_pose, self.config.arm_speed)

        self._set_state(TaskState.PLACING)
        self.clients.set_suction(RobotClients.BLOW)
        self.holding_block = False

    def prepare(self, advanced=False, place_order=()):
        self._set_state(TaskState.PREPARING)
        self.clients.move_arm(self.config.shooting_pose, self.config.arm_speed)
        return self.clients.prepare_task(advanced=advanced, place_order=place_order)

    def execute_all(self, task_count):
        paths = self.servo_csv_logger.open()
        print(
            f"视觉伺服 CSV 已覆盖创建：方块={paths['block']}，托盘={paths['board']}"
        )
        try:
            for index in range(int(task_count)):
                target = self.clients.get_task_target(index)
                rospy.loginfo("执行第 %d/%d 个任务，类别=%s", index + 1, task_count, target.category)
                self._pick(target, task_index=index + 1)
                self._place(target, task_index=index + 1)
            self.clients.set_suction(RobotClients.OFF)
            self._set_state(TaskState.COMPLETED)
        except Exception:
            self._set_state(TaskState.FAILED)
            if self.holding_block:
                rospy.logerr("任务失败时仍持有方块，保持吸盘状态并停止自动运动")
            print("\033[91m任务失败，当前视觉伺服 CSV 已保存。\033[0m")
            raise
        finally:
            self.servo_csv_logger.close()

    def run_interactive(self):
        advanced = input("是否进行进阶任务？输入 y 启用: ").strip().lower() == "y"
        place_order = []
        if advanced:
            place_order = [int(value) for value in input("输入进阶任务 7 项顺序: ").split()]
            if len(place_order) != 7:
                raise ValueError("进阶任务顺序必须正好包含 7 个整数")

        while True:
            response = self.prepare(advanced=advanced, place_order=place_order)
            if not response.success:
                print(f"\033[91m粗定位失败: {response.message}\033[0m")
                input("请调整托盘、方块或光照后按回车重新识别...")
                continue

            print(response.message)
            if input("\033[93m识别结果满意请输入 1；其他输入将重新识别: \033[0m").strip() == "1":
                break
        input("按回车开始抓放全部方块...")
        self.execution_start_time = time.monotonic()
        self.execute_all(response.task_count)
