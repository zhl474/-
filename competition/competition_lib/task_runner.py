"""粗定位与视觉伺服抓放任务状态机。"""

from enum import Enum
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import rospy

from .config import load_execution_config, load_visual_servo_config
from .ros_clients import RobotClients
from .servo_csv_logger import DEFAULT_SERVO_CSV_OUTPUT_DIR, ServoCsvLogger
from .visual_servo import apply_camera_to_sucker_offset, run_offset_visual_servo_alignment
from image_process_lib.servo_angle_model import (
    plan_servo_angle_transition,
    worst_case_servo_reset_seconds,
)


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


@dataclass(frozen=True)
class ServoRotationEstimate:
    """一次舵机指令的预计运动区间，时间基于命令服务成功返回的时刻。"""

    start_angle_deg: float
    target_angle_deg: float
    estimated_duration_sec: float
    command_accepted_at: float
    ready_at: float


class ServoAnglePlanner:
    def __init__(self, config, rotate_func):
        self.last_angle = config.initial_motor_angle_deg
        self.velocity = config.motor_velocity_deg_per_sec
        self.lower_margin = config.motor_lower_margin_deg
        self.upper_margin = config.motor_upper_margin_deg
        self.rotate_func = rotate_func

    def _command_rotation(self, target_angle):
        """发送绝对角度，并在服务确认命令已写出后开始保守计时。"""
        start_angle = float(self.last_angle)
        target_angle = float(target_angle)
        estimated_duration_sec = abs(target_angle - start_angle) / self.velocity

        # rotate_func 失败时会抛出异常，此时不得更新软件记录的舵机角度。
        self.rotate_func(target_angle)
        command_accepted_at = time.monotonic()
        self.last_angle = target_angle
        return ServoRotationEstimate(
            start_angle_deg=start_angle,
            target_angle_deg=target_angle,
            estimated_duration_sec=estimated_duration_sec,
            command_accepted_at=command_accepted_at,
            ready_at=command_accepted_at + estimated_duration_sec,
        )

    def plan(self, rotation_delta_deg):
        transition = plan_servo_angle_transition(
            self.last_angle,
            rotation_delta_deg,
            self.lower_margin,
            self.upper_margin,
        )
        pre_pick_rotation = None
        if transition.pre_pick_target_angle_deg is not None:
            pre_pick_rotation = self._command_rotation(
                transition.pre_pick_target_angle_deg
            )
        return (
            transition.pick_angle_deg,
            transition.place_angle_deg,
            pre_pick_rotation,
        )

    def commit_place_angle(self, place_angle):
        return self._command_rotation(place_angle)


class TaskRunner:
    def __init__(
        self,
        clients=None,
        execution_config=None,
        visual_config=None,
        servo_csv_output_dir=DEFAULT_SERVO_CSV_OUTPUT_DIR,
        servo_csv_logger=None,
        experiment_session_id=None,
    ):
        self.clients = clients or RobotClients()
        self.config = execution_config or load_execution_config()
        self.visual_config = visual_config or load_visual_servo_config()
        self.servo_csv_logger = servo_csv_logger or ServoCsvLogger(
            servo_csv_output_dir,
            session_id=experiment_session_id,
        )
        self.angle_planner = ServoAnglePlanner(self.config, self.clients.rotate_tool)
        self.state = TaskState.IDLE
        self.holding_block = False
        self.execution_start_time = None
        self.visual_servo_enabled = self.config.visual_servo_enabled
        if self.config.calibration_mode and not self.visual_servo_enabled:
            rospy.logwarn(
                "标定采集依赖视觉伺服精确位姿，已忽略 servo.enabled=false 并强制开启视觉伺服"
            )
            self.visual_servo_enabled = True
        mode_name = "标定采集模式" if self.config.calibration_mode else "正式运行模式"
        servo_mode_name = "开启（闭环）" if self.visual_servo_enabled else "关闭（开环）"
        print(f"\033[96m当前模式：{mode_name}；视觉伺服：{servo_mode_name}\033[0m")

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

    @staticmethod
    def _print_prepare_failure(message):
        """安全汇总直接按原格式显示，其他失败保留原粗定位前缀。"""
        message = str(message)
        if message.startswith((
            "高位初步安全检查失败：",
            "高位最终安全检查失败：",
        )):
            print(f"\033[91m{message}\033[0m")
            return
        print(f"\033[91m粗定位失败: {message}\033[0m")

    def _timed_call(self, label, operation, *args, **kwargs):
        """在开启调试时记录一次任务步骤的端到端耗时。"""
        if not self.config.timing_debug:
            return operation(*args, **kwargs)
        started_at = time.monotonic()
        try:
            return operation(*args, **kwargs)
        finally:
            elapsed_ms = (time.monotonic() - started_at) * 1000.0
            rospy.loginfo("任务步骤耗时：%s=%.1f ms", label, elapsed_ms)

    def _wait_for_motor_rotation(self, rotation, purpose):
        """在关键动作前只补足舵机预计运动尚未被其它步骤覆盖的时间。"""
        if rotation is None:
            return 0.0

        now = time.monotonic()
        elapsed_sec = max(0.0, now - rotation.command_accepted_at)
        remaining_sec = max(0.0, rotation.ready_at - now)
        if remaining_sec > 0.0:
            time.sleep(remaining_sec)

        if self.config.timing_debug:
            rospy.loginfo(
                "舵机旋转等待：阶段=%s，角度=%.1f°→%.1f°，估算=%.1f ms，"
                "已与其它步骤重叠=%.1f ms，补足=%.1f ms",
                purpose,
                rotation.start_angle_deg,
                rotation.target_angle_deg,
                rotation.estimated_duration_sec * 1000.0,
                min(elapsed_sec, rotation.estimated_duration_sec) * 1000.0,
                remaining_sec * 1000.0,
            )
        return remaining_sec

    def _align(
        self,
        offset_func,
        start_pose,
        log_label,
        error_threshold_px,
        event_callback=None,
    ):
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
            error_threshold_px=error_threshold_px,
            max_step_mm=self.config.max_step_mm,
            min_step_mm=self.config.min_step_mm,
            max_iter=self.config.max_iter,
            success_stable_frames=self.config.success_stable_frames,
            max_missed_frames=self.config.max_missed_frames,
            settle_sec=self.config.settle_sec,
            timing_debug=self.config.timing_debug,
            log_label=log_label,
            event_callback=event_callback,
            post_success_sample_frames=(
                self.config.post_success_sample_frames
                if self.config.calibration_mode
                else 0
            ),
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

    @staticmethod
    def _now_iso():
        """生成带本地时区的实验时间，方便和跨节点录像对齐。"""
        return datetime.now().astimezone().isoformat(timespec="milliseconds")

    def _read_actual_pose_for_csv(self):
        """读取实测 TCP 和相机六维位姿；失败时留空且不影响主流程。"""
        empty = {
            "实测TCP位置X": "",
            "实测TCP位置Y": "",
            "实测TCP位置Z": "",
            "实测TCP姿态R": "",
            "实测TCP姿态P": "",
            "实测TCP姿态YAW": "",
            "实测相机位置X": "",
            "实测相机位置Y": "",
            "实测相机位置Z": "",
            "实测相机姿态R": "",
            "实测相机姿态P": "",
            "实测相机姿态YAW": "",
        }
        get_actual_pose = getattr(self.clients, "get_actual_pose", None)
        if not callable(get_actual_pose):
            return empty
        try:
            response = get_actual_pose()
            if not getattr(response, "success", False):
                return empty
            tcp_pose = self._finite_values(
                response.tcp_pose, 6
            )
            camera_pose = self._finite_values(response.camera_pose, 6)
            if "" in tcp_pose or "" in camera_pose:
                return empty
            names = (
                "实测TCP位置X",
                "实测TCP位置Y",
                "实测TCP位置Z",
                "实测TCP姿态R",
                "实测TCP姿态P",
                "实测TCP姿态YAW",
                "实测相机位置X",
                "实测相机位置Y",
                "实测相机位置Z",
                "实测相机姿态R",
                "实测相机姿态P",
                "实测相机姿态YAW",
            )
            return dict(zip(names, [*tcp_pose, *camera_pose]))
        except Exception:
            return empty

    @staticmethod
    def _round_pose_fields(actual_pose):
        """把最终位姿字段改名为逐轮位姿字段。"""
        mapping = {
            "实测TCP位置X": "本轮实测TCP位置X",
            "实测TCP位置Y": "本轮实测TCP位置Y",
            "实测TCP位置Z": "本轮实测TCP位置Z",
            "实测TCP姿态R": "本轮实测TCP姿态R",
            "实测TCP姿态P": "本轮实测TCP姿态P",
            "实测TCP姿态YAW": "本轮实测TCP姿态YAW",
            "实测相机位置X": "本轮实测相机位置X",
            "实测相机位置Y": "本轮实测相机位置Y",
            "实测相机位置Z": "本轮实测相机位置Z",
            "实测相机姿态R": "本轮实测相机姿态R",
            "实测相机姿态P": "本轮实测相机姿态P",
            "实测相机姿态YAW": "本轮实测相机姿态YAW",
        }
        return {target: actual_pose.get(source, "") for source, target in mapping.items()}

    def _build_trial_context(self, target, side, task_index):
        """生成汇总和逐轮日志共用的稳定关联字段。"""
        prefix = "pick" if side == "pick" else "place"
        detected_x, detected_y = self._finite_values(
            getattr(target, f"{prefix}_high_detected_pixel_xy", None), 2
        )
        image_center_x, image_center_y = self._finite_values(
            getattr(target, f"{prefix}_high_image_center_xy", None), 2
        )
        return {
            "实验批次ID": self.servo_csv_logger.session_id,
            "任务序号": "" if task_index is None else int(task_index),
            "目标类型": "方块" if side == "pick" else "托盘",
            "方块类别": str(getattr(target, "category", "") or ""),
            "托盘行": getattr(target, "row", "") if side == "place" else "",
            "托盘列": getattr(target, "col", "") if side == "place" else "",
            "高位检测像素X": detected_x,
            "高位检测像素Y": detected_y,
            "高位图像中心X": image_center_x,
            "高位图像中心Y": image_center_y,
            "高位检测角度deg": getattr(target, "detected_angle_deg", ""),
            "目标旋转增量deg": getattr(target, "rotation_delta_deg", ""),
            "试次开始时间": self._now_iso(),
        }

    def _make_round_callback(self, target_key, context, events):
        """把控制层事件补齐实验上下文，并在每次运动后读取实际位姿。"""
        def callback(event):
            row = dict(context)
            row.update(event)
            row["全试次记录序号"] = len(events) + 1
            if row.get("事件") == "执行修正":
                row.update(self._round_pose_fields(self._read_actual_pose_for_csv()))
            self.servo_csv_logger.write_round(target_key, row)
            events.append(row)

        return callback

    @staticmethod
    def _response_value(response, name):
        try:
            value = float(getattr(response, name))
        except (AttributeError, TypeError, ValueError):
            return ""
        return value if math.isfinite(value) else ""

    def _event_statistics(self, events, success):
        """从逐轮事件计算终止残差和闭环过程统计。"""
        control_events = [
            row
            for row in events
            if row.get("事件") in {"目标丢失", "执行修正", "稳定帧"}
        ]
        static_rows = [row for row in events if row.get("事件") == "成功后静止帧"]
        static_errors = []
        for row in static_rows:
            try:
                dx = float(row["像素误差X"])
                dy = float(row["像素误差Y"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(dx) and math.isfinite(dy):
                static_errors.append((dx, dy))
        requested = self.config.post_success_sample_frames if success else 0
        required = int(math.ceil(0.8 * requested)) if requested else 0
        statistics = {
            "伺服总轮数": len({row.get("伺服轮次") for row in control_events}),
            "执行修正次数": sum(row.get("事件") == "执行修正" for row in events),
            "目标丢失次数": sum(row.get("事件") == "目标丢失" for row in events),
            "稳定帧数": sum(row.get("事件") == "稳定帧" for row in events),
            "静止采样请求帧数": requested,
            "静止采样有效帧数": len(static_errors),
            "静止采样完整": bool(requested and len(static_errors) >= required),
            "静止像素误差均值X": "",
            "静止像素误差均值Y": "",
            "静止像素误差标准差X": "",
            "静止像素误差标准差Y": "",
            "静止像素误差P95": "",
        }
        if static_errors:
            array = np.asarray(static_errors, dtype=float)
            norms = np.linalg.norm(array, axis=1)
            statistics.update(
                {
                    "静止像素误差均值X": float(np.mean(array[:, 0])),
                    "静止像素误差均值Y": float(np.mean(array[:, 1])),
                    "静止像素误差标准差X": float(np.std(array[:, 0])),
                    "静止像素误差标准差Y": float(np.std(array[:, 1])),
                    "静止像素误差P95": float(np.percentile(norms, 95)),
                }
            )
        return statistics

    def _record_servo_result(
        self,
        target,
        side,
        success,
        failure_message="",
        *,
        task_index=None,
        context=None,
        events=None,
        final_command_pose=None,
        last_response=None,
    ):
        """标定模式仅写入一条最终成功或失败记录。"""
        if not self.config.calibration_mode or not self.servo_csv_logger.is_open:
            return False
        prefix = "pick" if side == "pick" else "place"
        detected_x, detected_y = self._finite_values(
            getattr(target, f"{prefix}_high_detected_pixel_xy", None), 2
        )
        depth_x, depth_y = self._finite_values(
            getattr(target, f"{prefix}_high_depth_sample_pixel_xy", None), 2
        )
        world_valid = bool(
            getattr(target, f"{prefix}_high_world_position_valid", False)
        )
        world_x, world_y, world_z = (
            self._finite_values(
                getattr(target, f"{prefix}_high_world_position", None), 3
            )
            if world_valid
            else ["", "", ""]
        )
        rough_pose = self._finite_values(
            getattr(target, f"{prefix}_observation_pose", None), 6
        )
        context = dict(context or self._build_trial_context(target, side, task_index))
        events = list(events or [])
        command_pose = self._finite_values(final_command_pose, 6)
        actual_pose = self._read_actual_pose_for_csv()
        statistics = self._event_statistics(events, success)
        row = {
            "方块类别": str(getattr(target, "category", "") or ""),
            "事件": "伺服成功" if success else "伺服失败",
            "高位检测像素X": detected_x,
            "高位检测像素Y": detected_y,
            "深度采样像素X": depth_x,
            "深度采样像素Y": depth_y,
            "高位世界坐标X": world_x,
            "高位世界坐标Y": world_y,
            "高位世界坐标Z": world_z,
            "深度有效帧数": getattr(
                target, f"{prefix}_depth_valid_frame_count", ""
            ),
            "深度中位数毫米": getattr(target, f"{prefix}_depth_median_mm", ""),
            "深度MAD毫米": getattr(target, f"{prefix}_depth_mad_mm", ""),
            "粗定位TCP位置X": rough_pose[0],
            "粗定位TCP位置Y": rough_pose[1],
            "粗定位TCP位置Z": rough_pose[2],
            "标定目标TCP位置Z": getattr(
                target,
                f"{prefix}_calibration_target_tcp_z_mm",
                "",
            ),
            "粗定位来源": str(
                getattr(target, f"{prefix}_rough_localization_source", "") or ""
            ),
            "失败信息": "" if success else str(failure_message or "视觉伺服失败"),
            **context,
            "伺服开始时间": context.get("试次开始时间", ""),
            "伺服结束时间": self._now_iso(),
            **statistics,
            "最终低位目标像素X": self._response_value(last_response, "px"),
            "最终低位目标像素Y": self._response_value(last_response, "py"),
            "最终像素误差X": self._response_value(last_response, "dx_px"),
            "最终像素误差Y": self._response_value(last_response, "dy_px"),
            "最终低位检测角度deg": self._response_value(
                last_response, "detected_angle_deg"
            ),
            "最终低位匹配得分": self._response_value(last_response, "score"),
            "最终命令TCP位置X": command_pose[0],
            "最终命令TCP位置Y": command_pose[1],
            "最终命令TCP位置Z": command_pose[2],
            "最终命令TCP姿态R": command_pose[3],
            "最终命令TCP姿态P": command_pose[4],
            "最终命令TCP姿态YAW": command_pose[5],
        }
        final_px = row["最终低位目标像素X"]
        final_py = row["最终低位目标像素Y"]
        final_dx = row["最终像素误差X"]
        final_dy = row["最终像素误差Y"]
        row["最终低位图像中心X"] = (
            final_px - final_dx if "" not in (final_px, final_dx) else ""
        )
        row["最终低位图像中心Y"] = (
            final_py - final_dy if "" not in (final_py, final_dy) else ""
        )
        row.update(actual_pose)
        actual_xyz = [actual_pose.get(name, "") for name in (
            "实测TCP位置X", "实测TCP位置Y", "实测TCP位置Z"
        )]
        for axis, actual, command in zip("XYZ", actual_xyz, command_pose[:3]):
            row[f"实测减命令TCP位置{axis}"] = (
                actual - command if "" not in (actual, command) else ""
            )
        mean_error = [
            statistics["静止像素误差均值X"],
            statistics["静止像素误差均值Y"],
        ]
        if "" not in mean_error and "" not in actual_xyz[:2]:
            matrix = np.asarray(self.visual_config["pixel_to_robot_matrix"], dtype=float)
            zero_xy = np.asarray(actual_xyz[:2], dtype=float) + matrix @ np.asarray(
                mean_error, dtype=float
            )
            row["零误差等效TCP位置X"] = float(zero_xy[0])
            row["零误差等效TCP位置Y"] = float(zero_xy[1])
        else:
            row["零误差等效TCP位置X"] = ""
            row["零误差等效TCP位置Y"] = ""
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
        pre_pick_z_mm = self._validate_finite_value(
            pick_z_mm + self.config.pick_approach_clearance_mm,
            "预抓取高度",
        )
        rough_pose = self._validate_motion_pose(target.pick_observation_pose, "方块观察位")

        target_type = str(getattr(target, "target_type", "pick_place") or "pick_place")
        is_calibration_block = self.config.calibration_mode and target_type == "block"
        if is_calibration_block:
            # 标定方块没有托盘目标，模拟下探后退回本方块观察高度。
            retreat_z_mm = rough_pose[2]
            retreat_label = "标定方块退回位"
        else:
            # 正式任务吸住后原地抬到托盘伺服高度，再以相同 Z 进入托盘区域。
            place_observation_pose = self._validate_motion_pose(
                target.place_observation_pose,
                "托盘观察位",
            )
            retreat_z_mm = place_observation_pose[2]
            retreat_label = "吸取后托盘高度抬升位"

        retreat_blend_radius_mm = None
        if not self.config.calibration_mode and target_type == "pick_place":
            configured_blend_radius_mm = self.config.pick_retreat_blend_radius_mm
            if configured_blend_radius_mm > 0.0:
                lift_distance_mm = retreat_z_mm - pick_z_mm
                if configured_blend_radius_mm >= lift_distance_mm:
                    raise RuntimeError(
                        "抓后抬升圆滑半径必须小于竖直抬升距离："
                        f"半径={configured_blend_radius_mm:.2f} mm，"
                        f"抬升距离={lift_distance_mm:.2f} mm"
                    )
                retreat_blend_radius_mm = configured_blend_radius_mm

        # 闭环最终 XY 尚未得到，先用粗定位 XY 校验本次全部运动高度。
        for height, label in (
            (pre_pick_z_mm, "预抓取位"),
            (pick_z_mm, "最终抓取位"),
            (retreat_z_mm, retreat_label),
        ):
            height_check_pose = list(rough_pose)
            height_check_pose[2] = height
            self._validate_motion_pose(height_check_pose, label)

        _, place_angle, pre_pick_rotation = self._timed_call(
            "舵机角度规划与预旋转指令",
            self.angle_planner.plan,
            target.rotation_delta_deg,
        )
        self._set_state(TaskState.PICK_COARSE)
        if self.visual_servo_enabled:
            self._timed_call(
                "方块观察位运动",
                self.clients.move_arm,
                rough_pose,
                self.config.arm_speed,
                wait_until_stable=True,
            )

            self._set_state(TaskState.PICK_ALIGN)
            trial_context = self._build_trial_context(target, "pick", task_index)
            trial_events = []
            round_callback = self._make_round_callback(
                "block", trial_context, trial_events
            ) if self.config.calibration_mode else None
            try:
                success, camera_pose, last_response, message = self._align(
                    lambda: self.clients.detect_block_offset(
                        target.category,
                        target.detected_angle_deg,
                    ),
                    rough_pose,
                    "方块视觉伺服",
                    self.config.block_error_threshold_px,
                    event_callback=round_callback,
                )
            except Exception as exc:
                self._record_servo_result(
                    target,
                    "pick",
                    False,
                    str(exc),
                    task_index=task_index,
                    context=trial_context,
                    events=trial_events,
                )
                raise
            if not success:
                self._record_servo_result(
                    target,
                    "pick",
                    False,
                    message,
                    task_index=task_index,
                    context=trial_context,
                    events=trial_events,
                    final_command_pose=camera_pose,
                    last_response=last_response,
                )
                raise RuntimeError(f"方块视觉伺服失败: {message}")
            self._record_servo_result(
                target,
                "pick",
                True,
                task_index=task_index,
                context=trial_context,
                events=trial_events,
                final_command_pose=camera_pose,
                last_response=last_response,
            )
        else:
            camera_pose = rough_pose

        # 相机位姿转换到吸盘 XY 后，直接斜向进入动态预抓取位并等待停稳。
        pre_pick_pose = apply_camera_to_sucker_offset(camera_pose, self.visual_config)
        pre_pick_pose[2] = pre_pick_z_mm
        pre_pick_pose = self._validate_motion_pose(pre_pick_pose, "预抓取位")
        self._timed_call(
            "预抓取位运动",
            self.clients.move_arm,
            pre_pick_pose,
            self.config.pick_approach_speed,
            wait_until_stable=True,
        )

        # 无论开环还是闭环，都只在真正下探前检查抓取前预旋转是否完成。
        self._wait_for_motor_rotation(pre_pick_rotation, "抓取前预旋转")
        self._set_state(TaskState.PICKING)
        pick_pose = list(pre_pick_pose)
        pick_pose[2] = pick_z_mm
        pick_pose = self._validate_motion_pose(pick_pose, "最终抓取位")
        self._timed_call(
            "抓取下探运动",
            self.clients.move_arm,
            pick_pose,
            self.config.pick_speed,
            wait_until_stable=False,
        )
        # input("吸取方块后请确认吸盘已吸住方块，按回车继续...")
        if not self.config.calibration_mode:
            self._timed_call(
                "吸盘吸气服务",
                self.clients.set_suction,
                RobotClients.SUCK,
            )
            self.holding_block = True
        retreat_pose = list(pre_pick_pose)
        retreat_pose[2] = retreat_z_mm
        retreat_pose = self._validate_motion_pose(retreat_pose, retreat_label)
        self._timed_call(
            "抓后抬升运动",
            self.clients.move_arm,
            retreat_pose,
            self.config.arm_speed,
            wait_until_stable=False,
            blend_radius_mm=retreat_blend_radius_mm,
        )
        place_rotation = self._timed_call(
            "摆放角度舵机指令",
            self.angle_planner.commit_place_angle,
            place_angle,
        )
        return place_rotation

    def _place(self, target, task_index=None, place_rotation=None):
        rough_pose = self._validate_motion_pose(target.place_observation_pose, "托盘观察位")
        open_loop_place_pose = None
        if not self.visual_servo_enabled:
            open_loop_place_pose = apply_camera_to_sucker_offset(
                rough_pose,
                self.visual_config,
            )
            # 托盘高位标定给出的 Z 就是释放 Z，只应用吸盘 XY 偏移。
            open_loop_place_pose = self._validate_motion_pose(
                open_loop_place_pose,
                "开环最终摆放位",
            )

        self._set_state(TaskState.PLACE_COARSE)
        if self.visual_servo_enabled:
            self._timed_call(
                "托盘观察位运动",
                self.clients.move_arm,
                rough_pose,
                self.config.arm_speed,
                wait_until_stable=True,
            )

            self._set_state(TaskState.PLACE_ALIGN)
            trial_context = self._build_trial_context(target, "place", task_index)
            trial_events = []
            round_callback = self._make_round_callback(
                "board", trial_context, trial_events
            ) if self.config.calibration_mode else None
            try:
                success, camera_pose, last_response, message = self._align(
                    lambda: self.clients.detect_board_offset(target.row, target.col),
                    rough_pose,
                    "托盘视觉伺服",
                    self.config.tray_error_threshold_px,
                    event_callback=round_callback,
                )
            except Exception as exc:
                self._record_servo_result(
                    target,
                    "place",
                    False,
                    str(exc),
                    task_index=task_index,
                    context=trial_context,
                    events=trial_events,
                )
                raise
            if not success:
                self._record_servo_result(
                    target,
                    "place",
                    False,
                    message,
                    task_index=task_index,
                    context=trial_context,
                    events=trial_events,
                    final_command_pose=camera_pose,
                    last_response=last_response,
                )
                raise RuntimeError(f"托盘视觉伺服失败: {message}")
            self._record_servo_result(
                target,
                "place",
                True,
                task_index=task_index,
                context=trial_context,
                events=trial_events,
                final_command_pose=camera_pose,
                last_response=last_response,
            )

            place_pose = apply_camera_to_sucker_offset(camera_pose, self.visual_config)
            # 托盘标定给出的观察 Z 同时就是吹气释放 Z，此处只应用吸盘 XY 偏移。
            place_pose = self._validate_motion_pose(place_pose, "最终摆放位")
            self._timed_call(
                "最终摆放位运动",
                self.clients.move_arm,
                place_pose,
                self.config.arm_speed,
                wait_until_stable=True,
            )
        else:
            place_pose = open_loop_place_pose
            self._timed_call(
                "最终摆放位运动",
                self.clients.move_arm,
                place_pose,
                self.config.arm_speed,
                wait_until_stable=True,
            )

        if not self.config.calibration_mode:
            if place_rotation is None:
                raise RuntimeError("摆放前缺少本次舵机旋转记录，已禁止喷气")
            # 机械臂搬运可以覆盖舵机转动时间，但喷气前必须确认预计旋转已经完成。
            self._wait_for_motor_rotation(place_rotation, "抓取后摆放旋转")

        self._set_state(TaskState.PLACING)
        if not self.config.calibration_mode:
            self._timed_call(
                "吸盘喷气服务",
                self.clients.set_suction,
                RobotClients.BLOW,
            )
        self.holding_block = False

    def prepare(self, advanced=False, place_order=()):
        self._set_state(TaskState.PREPARING)
        # 复位发生在拍摄和用户确认之前，因此不计入比赛方案成本。
        initial_angle = float(self.config.initial_motor_angle_deg)
        self.clients.rotate_tool(initial_angle)
        reset_wait_seconds = worst_case_servo_reset_seconds(
            initial_angle,
            self.config.motor_velocity_deg_per_sec,
        )
        # 无位置反馈时必须按 0°/360° 两端中的最坏角差保守等待。
        time.sleep(reset_wait_seconds)
        self.angle_planner.last_angle = initial_angle
        shooting_pose = self._validate_motion_pose(self.config.shooting_pose, "高位拍摄位")
        self.clients.move_arm(shooting_pose, self.config.arm_speed, wait_until_stable=True)
        return self.clients.prepare_task(advanced=advanced, place_order=place_order)

    @staticmethod
    def _git_metadata():
        """尽力记录源码版本；Git 不可用时不得阻断标定。"""
        source_root = Path(__file__).resolve().parents[2]
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(source_root),
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            status = subprocess.run(
                ["git", "status", "--short"],
                cwd=str(source_root),
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            return {
                "Git提交": revision.stdout.strip() if revision.returncode == 0 else "",
                "Git工作区有修改": bool(status.stdout.strip()) if status.returncode == 0 else None,
                "Git工作区状态": status.stdout.splitlines() if status.returncode == 0 else [],
            }
        except Exception:  # noqa: BLE001
            return {"Git提交": "", "Git工作区有修改": None, "Git工作区状态": []}

    def _experiment_metadata(self):
        archive_dir = self.servo_csv_logger.archive_dir
        metadata = {
            "执行配置": asdict(self.config),
            "视觉伺服配置": self.visual_config,
            "静止采样完整阈值": int(
                math.ceil(0.8 * self.config.post_success_sample_frames)
            ),
            "调试归档文件": {
                "方块低位视频": str(archive_dir / "方块视觉伺服调试.avi"),
                "托盘低位视频": str(archive_dir / "托盘视觉伺服调试.avi"),
                "方块上表面掩码": str(archive_dir / "方块上表面掩码.jpg"),
                "高位方块模板匹配": str(archive_dir / "高位方块模板匹配结果.jpg"),
                "托盘格点粗定位": str(archive_dir / "托盘格点粗定位.jpg"),
            },
        }
        metadata.update(self._git_metadata())
        return metadata

    def execute_all(self, task_count):
        experiment_status = "未开始"
        try:
            if self.config.calibration_mode:
                paths = self.servo_csv_logger.open(
                    metadata=self._experiment_metadata()
                )
                print(
                    f"标定 CSV 已覆盖创建：方块={paths['block']}，"
                    f"托盘={paths['board']}\n"
                    f"实验批次归档：{paths['archive_dir']}"
                )
                experiment_status = "进行中"
                self.servo_csv_logger.update_metadata(
                    {"实验状态": experiment_status, "计划任务数量": int(task_count)}
                )
                # 标定采集必须先确认气泵和电磁阀已关闭，失败则不允许开始运动。
                self.clients.set_suction(RobotClients.OFF)
            for index in range(int(task_count)):
                task_started_at = time.monotonic()
                target = self._timed_call(
                    f"读取第 {index + 1} 个任务目标",
                    self.clients.get_task_target,
                    index,
                )
                target_type = str(
                    getattr(target, "target_type", "pick_place") or "pick_place"
                )
                if target_type != "pick_place":
                    raise RuntimeError(
                        "正式运行只接受 pick_place 目标，收到目标类型 "
                        f"{target_type!r}；标定采集请改用 calibration.py 启动"
                    )
                rospy.loginfo("执行第 %d/%d 个任务，类别=%s", index + 1, task_count, target.category)
                place_rotation = self._pick(target, task_index=index + 1)
                self._place(
                    target,
                    task_index=index + 1,
                    place_rotation=place_rotation,
                )
                if self.config.timing_debug:
                    rospy.loginfo(
                        "单块任务总耗时：第 %d/%d 块=%.1f ms",
                        index + 1,
                        task_count,
                        (time.monotonic() - task_started_at) * 1000.0,
                    )
            if not self.config.calibration_mode:
                self.clients.set_suction(RobotClients.OFF)
            experiment_status = "完成"
            self._set_state(TaskState.COMPLETED)
        except Exception:
            experiment_status = "失败"
            self._set_state(TaskState.FAILED)
            if self.holding_block:
                rospy.logerr("任务失败时仍持有方块，保持吸盘状态并停止自动运动")
            if self.config.calibration_mode:
                print("\033[91m任务失败，当前标定 CSV 已保存。\033[0m")
            raise
        finally:
            if self.config.calibration_mode and self.servo_csv_logger.archive_dir.exists():
                self.servo_csv_logger.update_metadata({"实验状态": experiment_status})
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
                self._print_prepare_failure(response.message)
                input("请调整托盘、方块或光照后按回车重新识别...")
                continue

            print(response.message)
            if input("\033[93m识别结果满意请输入 1；其他输入将重新识别: \033[0m").strip() == "1":
                break
        action_name = "标定采集" if self.config.calibration_mode else "抓放"
        input(f"按回车开始{action_name}全部方块...")
        self.execution_start_time = time.monotonic()
        self.execute_all(response.task_count)


class CalibrationTaskRunner(TaskRunner):
    """独立标定采集执行器：block 目标只拾取，tray 目标只抵达，全程关闭吸吹气。"""

    def __init__(
        self,
        clients=None,
        execution_config=None,
        visual_config=None,
        servo_csv_output_dir=DEFAULT_SERVO_CSV_OUTPUT_DIR,
        servo_csv_logger=None,
        experiment_session_id=None,
    ):
        base_config = execution_config or load_execution_config()
        forced_config = replace(base_config, calibration_mode=True)
        super().__init__(
            clients=clients,
            execution_config=forced_config,
            visual_config=visual_config,
            servo_csv_output_dir=servo_csv_output_dir,
            servo_csv_logger=servo_csv_logger,
            experiment_session_id=experiment_session_id,
        )
        print("\033[96m当前模式：独立标定采集（方块=拾取，托盘=抵达，不吸不吹）\033[0m")

    def execute_all(self, task_count, block_count=0, tray_count=0):
        experiment_status = "未开始"
        try:
            paths = self.servo_csv_logger.open(metadata=self._experiment_metadata())
            print(
                f"标定 CSV 已覆盖创建：方块={paths['block']}，"
                f"托盘={paths['board']}\n"
                f"实验批次归档：{paths['archive_dir']}"
            )
            experiment_status = "进行中"
            self.servo_csv_logger.update_metadata(
                {
                    "实验状态": experiment_status,
                    "计划任务数量": int(task_count),
                    "计划方块数量": int(block_count),
                    "计划托盘数量": int(tray_count),
                }
            )
            # 标定采集必须先确认气泵和电磁阀已关闭，失败则不允许开始运动。
            self.clients.set_suction(RobotClients.OFF)
            block_index = 0
            tray_index = 0
            for index in range(int(task_count)):
                target = self.clients.get_task_target(index)
                target_type = str(getattr(target, "target_type", "") or "")
                if target_type == "block":
                    block_index += 1
                    rospy.loginfo("标定方块 %d/%d", block_index, block_count)
                    self._pick(target, task_index=block_index)
                elif target_type == "tray":
                    tray_index += 1
                    rospy.loginfo("标定托盘点 %d/%d", tray_index, tray_count)
                    self._place(target, task_index=tray_index)
                else:
                    raise RuntimeError(
                        f"未知目标类型 {target_type!r}，已停止标定"
                    )
            experiment_status = "完成"
            self._set_state(TaskState.COMPLETED)
        except Exception:
            experiment_status = "失败"
            self._set_state(TaskState.FAILED)
            print("\033[91m标定失败，当前标定 CSV 已保存。\033[0m")
            raise
        finally:
            if self.servo_csv_logger.archive_dir.exists():
                self.servo_csv_logger.update_metadata({"实验状态": experiment_status})
            self.servo_csv_logger.close()

    def run_interactive(self):
        while True:
            response = self.prepare()
            if not response.success:
                print(f"\033[91m标定准备失败: {response.message}\033[0m")
                input("请调整方块、托盘或光照后按回车重新识别...")
                continue

            print(response.message)
            if input("\033[93m识别结果满意请输入 1；其他输入将重新识别: \033[0m").strip() == "1":
                break
        input("按回车开始标定采集...")
        self.execution_start_time = time.monotonic()
        self.execute_all(
            response.task_count,
            block_count=getattr(response, "block_count", 0),
            tray_count=getattr(response, "tray_count", 0),
        )
