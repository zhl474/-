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
        mode_name = "标定采集模式" if self.config.calibration_mode else "正式运行模式"
        print(f"\033[96m当前模式：{mode_name}\033[0m")

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

    def _align(self, offset_func, start_pose, log_label):
        start_pose = self._validate_motion_pose(start_pose, f"{log_label}起始位")

        def move_checked(pose, *args, **kwargs):
            """视觉伺服每轮运动前复核位姿，避免无效识别结果生成危险命令。"""
            checked_pose = self._validate_motion_pose(pose, f"{log_label}修正位")
            return self.clients.move_arm(checked_pose, *args, **kwargs)

        return run_offset_visual_servo_alignment(
            offset_func,
            move_checked,
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
        )

    def _validate_motion_pose(self, pose, label):
        """校验一条正式 TCP 运动命令，并返回规范化后的六维位姿。"""
        try:
            values = [float(value) for value in pose]
        except (TypeError, ValueError):
            raise RuntimeError(f"{label}必须包含 6 个有限数值")
        if len(values) != 6 or not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"{label}必须包含 6 个有限数值")
        if values[2] < self.config.minimum_tcp_z_mm:
            raise RuntimeError(
                f"{label} Z={values[2]:.2f} mm 低于 TCP 安全下限 "
                f"{self.config.minimum_tcp_z_mm:.2f} mm"
            )
        return values

    @staticmethod
    def _validate_finite_value(value, label):
        """校验单个任务高度等标量，避免 NaN 或无穷值进入运动计算。"""
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise RuntimeError(f"{label}必须是有限数值")
        if not math.isfinite(number):
            raise RuntimeError(f"{label}必须是有限数值")
        return number

    @staticmethod
    def _finite_values(values, length):
        try:
            result = [float(value) for value in values]
        except (TypeError, ValueError):
            return [""] * length
        if len(result) != length or not all(math.isfinite(value) for value in result):
            return [""] * length
        return result

    def _read_actual_pose_for_csv(self):
        """读取实测 TCP XYZ；读取失败时留空且不影响原失败原因。"""
        empty = {
            "实测TCP位置X": "",
            "实测TCP位置Y": "",
            "实测TCP位置Z": "",
        }
        get_actual_pose = getattr(self.clients, "get_actual_pose", None)
        if not callable(get_actual_pose):
            return empty
        try:
            response = get_actual_pose()
            if not getattr(response, "success", False):
                return empty
            tcp_x, tcp_y, tcp_z, _tcp_r, _tcp_p, _tcp_yaw = self._finite_values(
                response.tcp_pose, 6
            )
            return {
                "实测TCP位置X": tcp_x,
                "实测TCP位置Y": tcp_y,
                "实测TCP位置Z": tcp_z,
            }
        except Exception:
            return empty

    def _record_servo_result(self, target, side, success):
        """标定模式仅写入一条最终成功或失败记录。"""
        if not self.config.calibration_mode or not self.servo_csv_logger.is_open:
            return False
        prefix = "pick" if side == "pick" else "place"
        detected_x, detected_y = self._finite_values(
            getattr(target, f"{prefix}_high_detected_pixel_xy", None), 2
        )
        row = {
            "方块类别": str(getattr(target, "category", "") or ""),
            "事件": "伺服成功" if success else "伺服失败",
            "高位检测像素X": detected_x,
            "高位检测像素Y": detected_y,
        }
        row.update(self._read_actual_pose_for_csv())
        return self.servo_csv_logger.write("block" if side == "pick" else "board", row)

    def _pick(self, target, task_index=None):
        if not getattr(target, "pick_surface_z_valid", False):
            raise RuntimeError("方块缺少有效抓取表面高度，已取消抓取")
        pick_surface_z_mm = self._validate_finite_value(
            getattr(target, "pick_surface_z_mm", None),
            "方块抓取表面高度",
        )
        pick_z_mm = self._validate_finite_value(
            pick_surface_z_mm + self.config.pick_surface_offset_mm,
            "最终抓取高度",
        )
        rough_pose = self._validate_motion_pose(target.pick_observation_pose, "方块观察位")
        lift_pose = list(rough_pose)
        lift_pose[2] = self.config.lift_z
        lift_pose = self._validate_motion_pose(lift_pose, "抓取抬升位")
        pick_pose = list(lift_pose)
        pick_pose[2] = pick_z_mm
        pick_pose = self._validate_motion_pose(pick_pose, "最终抓取位")

        _, place_angle, motor_wait = self.angle_planner.plan(target.rotation_delta_deg)
        self._set_state(TaskState.PICK_COARSE)
        self.clients.move_arm(rough_pose, self.config.arm_speed, wait_until_stable=True)
        if motor_wait > 0:
            time.sleep(motor_wait)

        self._set_state(TaskState.PICK_ALIGN)
        try:
            success, camera_pose, last_response, message = self._align(
                lambda: self.clients.detect_block_offset(target.category, target.detected_angle_deg),
                rough_pose,
                "方块视觉伺服",
            )
        except Exception as exc:
            self._record_servo_result(target, "pick", False)
            raise
        if not success:
            self._record_servo_result(target, "pick", False)
            raise RuntimeError(f"方块视觉伺服失败: {message}")
        self._record_servo_result(target, "pick", True)

        sucker_pose = apply_camera_to_sucker_offset(camera_pose, self.visual_config)
        sucker_pose[2] = self.config.lift_z
        sucker_pose = self._validate_motion_pose(sucker_pose, "吸盘抓取抬升位")
        self.clients.move_arm(sucker_pose, self.config.arm_speed)

        self._set_state(TaskState.PICKING)
        pick_pose = list(sucker_pose)
        pick_pose[2] = pick_z_mm
        pick_pose = self._validate_motion_pose(pick_pose, "最终抓取位")
        self.clients.move_arm(pick_pose, self.config.pick_speed)
        if not self.config.calibration_mode:
            self.clients.set_suction(RobotClients.SUCK)
            self.holding_block = True
        self.clients.move_arm(sucker_pose, self.config.arm_speed)
        self.angle_planner.commit_place_angle(place_angle)

    def _place(self, target, task_index=None):
        rough_pose = self._validate_motion_pose(target.place_observation_pose, "托盘观察位")
        self._set_state(TaskState.PLACE_COARSE)
        self.clients.move_arm(rough_pose, self.config.arm_speed, wait_until_stable=True)

        self._set_state(TaskState.PLACE_ALIGN)
        try:
            success, camera_pose, last_response, message = self._align(
                lambda: self.clients.detect_board_offset(target.row, target.col),
                rough_pose,
                "托盘视觉伺服",
            )
        except Exception as exc:
            self._record_servo_result(target, "place", False)
            raise
        if not success:
            self._record_servo_result(target, "place", False)
            raise RuntimeError(f"托盘视觉伺服失败: {message}")
        self._record_servo_result(target, "place", True)

        place_pose = apply_camera_to_sucker_offset(camera_pose, self.visual_config)
        # 托盘标定给出的观察 Z 同时就是吹气释放 Z，此处只应用吸盘 XY 偏移。
        place_pose = self._validate_motion_pose(place_pose, "最终摆放位")
        self.clients.move_arm(place_pose, self.config.arm_speed)

        self._set_state(TaskState.PLACING)
        if not self.config.calibration_mode:
            self.clients.set_suction(RobotClients.BLOW)
        self.holding_block = False

    def prepare(self, advanced=False, place_order=()):
        self._set_state(TaskState.PREPARING)
        shooting_pose = self._validate_motion_pose(self.config.shooting_pose, "高位拍摄位")
        self.clients.move_arm(shooting_pose, self.config.arm_speed, wait_until_stable=True)
        return self.clients.prepare_task(advanced=advanced, place_order=place_order)

    def execute_all(self, task_count):
        try:
            if self.config.calibration_mode:
                paths = self.servo_csv_logger.open()
                print(
                    f"标定 CSV 已覆盖创建：方块={paths['block']}，"
                    f"托盘={paths['board']}"
                )
                # 标定采集必须先确认气泵和电磁阀已关闭，失败则不允许开始运动。
                self.clients.set_suction(RobotClients.OFF)
            for index in range(int(task_count)):
                target = self.clients.get_task_target(index)
                rospy.loginfo("执行第 %d/%d 个任务，类别=%s", index + 1, task_count, target.category)
                self._pick(target, task_index=index + 1)
                self._place(target, task_index=index + 1)
            if not self.config.calibration_mode:
                self.clients.set_suction(RobotClients.OFF)
            self._set_state(TaskState.COMPLETED)
        except Exception:
            self._set_state(TaskState.FAILED)
            if self.holding_block:
                rospy.logerr("任务失败时仍持有方块，保持吸盘状态并停止自动运动")
            if self.config.calibration_mode:
                print("\033[91m任务失败，当前标定 CSV 已保存。\033[0m")
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
        action_name = "标定采集" if self.config.calibration_mode else "抓放"
        input(f"按回车开始{action_name}全部方块...")
        self.execution_start_time = time.monotonic()
        self.execute_all(response.task_count)
