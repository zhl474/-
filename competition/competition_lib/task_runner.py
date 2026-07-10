"""粗定位与视觉伺服抓放任务状态机。"""

from enum import Enum
import time

import rospy

from .config import load_execution_config, load_visual_servo_config
from .ros_clients import RobotClients
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
    def __init__(self, clients=None, execution_config=None, visual_config=None):
        self.clients = clients or RobotClients()
        self.config = execution_config or load_execution_config()
        self.visual_config = visual_config or load_visual_servo_config()
        self.angle_planner = ServoAnglePlanner(self.config, self.clients.rotate_tool)
        self.state = TaskState.IDLE
        self.holding_block = False

    def _set_state(self, state):
        self.state = state
        rospy.loginfo("任务状态: %s", state.value)

    def _align(self, offset_func, start_pose):
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
        )

    def _pick(self, target):
        _, place_angle, motor_wait = self.angle_planner.plan(target.rotation_delta_deg)
        rough_pose = list(target.pick_observation_pose)
        self._set_state(TaskState.PICK_COARSE)
        self.clients.move_arm(rough_pose, self.config.arm_speed)
        if motor_wait > 0:
            time.sleep(motor_wait)

        self._set_state(TaskState.PICK_ALIGN)
        success, camera_pose, _, message = self._align(
            lambda: self.clients.detect_block_offset(target.category, target.detected_angle_deg),
            rough_pose,
        )
        if not success:
            raise RuntimeError(f"方块视觉伺服失败: {message}")

        sucker_pose = apply_camera_to_sucker_offset(camera_pose, self.visual_config)
        sucker_pose[2] = self.config.lift_z
        self.clients.move_arm(sucker_pose, self.config.arm_speed)

        self._set_state(TaskState.PICKING)
        pick_pose = list(sucker_pose)
        pick_pose[2] = self.config.pick_z
        self.clients.move_arm(pick_pose, self.config.pick_speed)
        self.clients.set_suction(RobotClients.SUCK)
        self.holding_block = True
        self.clients.move_arm(sucker_pose, self.config.arm_speed)
        self.angle_planner.commit_place_angle(place_angle)

    def _place(self, target):
        rough_pose = list(target.place_observation_pose)
        self._set_state(TaskState.PLACE_COARSE)
        self.clients.move_arm(rough_pose, self.config.arm_speed)

        self._set_state(TaskState.PLACE_ALIGN)
        success, camera_pose, _, message = self._align(
            lambda: self.clients.detect_board_offset(target.row, target.col),
            rough_pose,
        )
        if not success:
            raise RuntimeError(f"托盘视觉伺服失败: {message}")

        place_pose = apply_camera_to_sucker_offset(camera_pose, self.visual_config)
        place_pose[2] = self.config.place_high_z
        self.clients.move_arm(place_pose, self.config.arm_speed)

        self._set_state(TaskState.PLACING)
        down_pose = list(place_pose)
        down_pose[2] = self.config.place_down_z
        self.clients.move_arm(down_pose, self.config.pick_speed)
        self.clients.set_suction(RobotClients.BLOW)
        self.holding_block = False
        lift_pose = list(down_pose)
        lift_pose[2] += self.config.place_lift_step_mm
        self.clients.move_arm(lift_pose, self.config.arm_speed)

    def prepare(self, advanced=False, place_order=()):
        self._set_state(TaskState.PREPARING)
        self.clients.move_arm(self.config.shooting_pose, self.config.arm_speed)
        return self.clients.prepare_task(advanced=advanced, place_order=place_order)

    def execute_all(self, task_count):
        try:
            for index in range(int(task_count)):
                target = self.clients.get_task_target(index)
                rospy.loginfo("执行第 %d/%d 个任务，类别=%s", index + 1, task_count, target.category)
                self._pick(target)
                self._place(target)
            self.clients.set_suction(RobotClients.OFF)
            self._set_state(TaskState.COMPLETED)
        except Exception:
            self._set_state(TaskState.FAILED)
            if self.holding_block:
                rospy.logerr("任务失败时仍持有方块，保持吸盘状态并停止自动运动")
            raise

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
        self.execute_all(response.task_count)
