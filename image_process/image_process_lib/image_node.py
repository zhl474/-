#!/home/zhl/fr3env/fr3env/bin/python
"""图像快照、任务规划和视觉伺服检测的 ROS 服务组合节点。"""

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

import yaml
from ultralytics import YOLO

import image_process_lib.block_detection as block_detection_module
import image_process_lib.block_scene_detector as block_scene_detector_module
import image_process_lib.board_scene_detector as board_scene_detector_module
from image_process_lib.template_config import load_template_geometry
from image_process_lib.board_scene_detector import (
    BOARD_COL_COUNT,
    BOARD_ROW_COUNT,
    board_grid_detect,
    draw_grid_debug,
    interpolate_grid_point,
)
from image_process_lib.board_servo_detector import detect_nearest_board_dot_in_roi
from image_process_lib.block_category import BLOCK_CATEGORY_NAMES, normalize_category_name
from image_process_lib.block_scene_detector import (
    detect_blocks_in_image,
    rematch_blocks_from_masks,
)
from image_process_lib.block_servo_detector import detect_block_with_high_prior_roi
from image_process_lib.advanced_planner import AdvancedPlanner
from image_process_lib.debug_output import DebugVideoRecorder, save_image_to_path
from image_process_lib.depth_rough_localization import DepthRoughLocalizer, fit_z_plane
from image_process_lib.high_pixel_to_tcp_localizer import HighPixelToTcpLocalizer
from image_process_lib.high_mask_edit_session import (
    会话环境变量,
    创建高位Mask编辑会话,
    预览设备环境变量,
    编辑取消退出码,
    读取已提交高位Mask,
)
from image_process_lib.task_planner import (
    ObservedBlock,
    PlacementTarget,
    TaskTarget,
    assign_blocks_to_targets,
    load_task_layout,
    select_calibration_tray_points,
)
from image_process_lib.arm_motion_time import get_default_arm_motion_time_model
from image_process_lib.board_candidate_selector import (
    BoardCandidateSelector,
    BoardCandidateSelectorConfig,
)
from image_process_lib.dynamic_board_report import (
    build_dynamic_board_selection_report,
)
from image_process_lib.dynamic_board_runtime import (
    OperatorPromptBroker,
    OperatorPromptChoiceError,
    OperatorPromptConflict,
    atomic_write_json,
    prompt_dynamic_selection_failure,
    sha256_file,
)
from image_process_lib.final_board_selector import (
    DynamicBoardSelectionError,
    FinalBoardSelector,
    FinalBoardSelectorConfig,
)
from image_process_lib.task_sequence_optimizer import (
    MotionModelSpeedMismatchError,
    TaskSequenceOptimizerConfig,
    build_task_plan_report,
    decide_optimizer_mode,
    optimize_task_sequence,
    validate_motion_model_speeds,
)
from image_process_lib.v5_board_library import load_v5_board_library

from image_process.srv import (
    DetectBlockOffset,
    DetectBlockOffsetResponse,
    DetectBoardOffset,
    DetectBoardOffsetResponse,
    GetTaskTarget,
    GetTaskTargetResponse,
    GetOperatorPrompt,
    GetOperatorPromptResponse,
    PrepareTask,
    PrepareTaskResponse,
    RespondOperatorPrompt,
    RespondOperatorPromptResponse,
)
from camera.srv import GetStableWorldPoints


PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, ".."))
COMPETITION_DIR = os.path.join(SRC_DIR, "competition")
EXECUTION_CONFIG_PATH = os.path.join(COMPETITION_DIR, "config", "execution.yaml")
VISUAL_SERVO_CONFIG_PATH = os.path.join(COMPETITION_DIR, "config", "visual_servo.yaml")
DETECTION_MODEL_PATH = os.path.join(COMPETITION_DIR, "model", "best5.14.pt")
BOARD_MODEL_PATH = os.path.join(COMPETITION_DIR, "model", "best.pt")
SEGMENTATION_MODEL_PATH = os.path.join(COMPETITION_DIR, "model", "best_seg.engine")
JINJIE_LIB_PATH = os.path.join(SRC_DIR, "jinjie", "jinjie_libtetris.so")
TASK_LAYOUT_PATH = os.path.join(PACKAGE_DIR, "config", "task_layout.yaml")
PERCEPTION_CONFIG_PATH = os.path.join(PACKAGE_DIR, "config", "perception.yaml")
DEFAULT_V5_BOARD_LIBRARY_PATH = os.path.join(
    PACKAGE_DIR,
    "config",
    "v5_board_library_v1.npz",
)
DEFAULT_BLOCK_CALIBRATION_PATH = os.path.join(
    PACKAGE_DIR,
    "config",
    "block_pixel_to_tcp_calibration.yaml",
)
DEFAULT_TRAY_CALIBRATION_PATH = os.path.join(
    PACKAGE_DIR,
    "config",
    "tray_pixel_to_tcp_calibration.yaml",
)
DEFAULT_DEBUG_DIR = os.path.expanduser("~/.ros/single_arm_tetris")
DEFAULT_HIGH_MASK_EDITOR_SCRIPT = os.path.join(
    SRC_DIR,
    "tools",
    "high_mask_editor_demo",
    "high_mask_session_editor.py",
)


class HighTcpSafetyCheckError(RuntimeError):
    """正式高位准备中的可恢复批量安全检查失败。"""

    def __init__(self, stage, violation_lines):
        self.stage = str(stage)
        self.violation_lines = tuple(str(line) for line in violation_lines)
        tail = (
            "本轮未打开 Mask 编辑器"
            if self.stage == "初步"
            else "请重新识别或重新编辑 Mask"
        )
        lines = [f"高位{self.stage}安全检查失败："]
        lines.extend(f"- {line}" for line in self.violation_lines)
        lines.append(tail)
        super().__init__("\n".join(lines))

    @property
    def violation_count(self):
        """返回本轮发生定位或越位问题的目标数量。"""
        return len(self.violation_lines)


class ImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_lock = threading.Lock()
        self.image_condition = threading.Condition(self.image_lock)
        self.prepare_task_lock = threading.Lock()
        self.high_mask_editor_process_lock = threading.Lock()
        self.active_high_mask_editor_process = None
        self.latest_image = None
        self.latest_image_stamp = None
        self.fresh_image_timeout_sec = float(rospy.get_param("~fresh_image_timeout_sec", 0.5))
        if not np.isfinite(self.fresh_image_timeout_sec) or self.fresh_image_timeout_sec <= 0.0:
            raise ValueError("fresh_image_timeout_sec 必须是大于 0 的有限数值")
        self.image_topic = rospy.get_param("~image_topic", "/camera/image_rect")
        if not isinstance(self.image_topic, str) or not self.image_topic.strip():
            raise ValueError("image_topic 必须是非空字符串")
        self.image_topic = self.image_topic.strip()
        self.task_targets = []
        self.last_board_layout = []
        self.board_grid_points = None
        self.board_grid_image_shape = None
        self.board_grid_image = None

        perception_config_path = rospy.get_param("~perception_config", PERCEPTION_CONFIG_PATH)
        with open(perception_config_path, "r", encoding="utf-8") as config_file:
            perception_config = yaml.safe_load(config_file) or {}
        with open(EXECUTION_CONFIG_PATH, "r", encoding="utf-8") as config_file:
            execution_config = yaml.safe_load(config_file) or {}
        # 标定/正式模式由 launch 文件的 ~calibration_mode 参数决定，
        # 不再读取 execution.yaml 的 calibration_mode 开关。
        calibration_mode_value = rospy.get_param("~calibration_mode", False)
        if not isinstance(calibration_mode_value, bool):
            raise ValueError("calibration_mode 必须是布尔值 true 或 false")
        self.calibration_mode = calibration_mode_value
        self.interaction_mode = str(
            rospy.get_param("~interaction_mode", "terminal")
        ).strip().lower()
        if self.interaction_mode not in ("terminal", "web"):
            raise ValueError("interaction_mode 只能是 terminal 或 web")
        self.operator_prompt_broker = OperatorPromptBroker()
        experiment_session_id = str(
            rospy.get_param("~experiment_session_id", "") or ""
        ).strip()
        if experiment_session_id and not re.fullmatch(
            r"[A-Za-z0-9_.-]+", experiment_session_id
        ):
            raise ValueError("experiment_session_id 包含非法路径字符")
        experiment_archive_root = str(
            rospy.get_param(
                "~experiment_archive_root",
                "/home/zhl/桌面/标定数据/实验日志",
            )
        ).strip()
        self.experiment_archive_dir = None
        self.task_plan_report_dir = None
        if experiment_session_id:
            self.task_plan_report_dir = os.path.join(
                experiment_archive_root,
                experiment_session_id,
            )
        if self.calibration_mode and experiment_session_id:
            self.experiment_archive_dir = self.task_plan_report_dir
            os.makedirs(self.experiment_archive_dir, exist_ok=True)
        model_config = perception_config.get("models", {})
        calibration_config = perception_config.get("calibration", {})

        def src_path(value, fallback):
            raw_path = str(value or fallback)
            return raw_path if os.path.isabs(raw_path) else os.path.join(SRC_DIR, raw_path)

        manual_editor_config = perception_config.get("high_mask_manual_editor", {})
        if not isinstance(manual_editor_config, dict):
            raise ValueError("perception.yaml 的 high_mask_manual_editor 必须是字典")
        manual_editor_enabled = rospy.get_param(
            "~high_mask_manual_editor_enabled",
            manual_editor_config.get("enabled", False),
        )
        if not isinstance(manual_editor_enabled, bool):
            raise ValueError("high_mask_manual_editor.enabled 必须是布尔值")
        self.high_mask_manual_editor_enabled = manual_editor_enabled
        self.high_mask_editor_script_path = src_path(
            rospy.get_param(
                "~high_mask_editor_script_path",
                manual_editor_config.get("script"),
            ),
            DEFAULT_HIGH_MASK_EDITOR_SCRIPT,
        )
        self.high_mask_editor_preview_device = str(rospy.get_param(
            "~high_mask_editor_preview_device",
            manual_editor_config.get("preview_device", "cuda"),
        )).strip().lower()
        if self.high_mask_editor_preview_device not in ("cpu", "cuda"):
            raise ValueError("high_mask_manual_editor.preview_device 只能是 cpu 或 cuda")
        if self.high_mask_manual_editor_enabled and not os.path.isfile(
            self.high_mask_editor_script_path
        ):
            raise FileNotFoundError(
                f"高位 Mask 编辑子进程脚本不存在: {self.high_mask_editor_script_path}"
            )

        detection_model_path = src_path(
            rospy.get_param("~detection_model_path", model_config.get("detection")),
            DETECTION_MODEL_PATH,
        )
        board_model_path = src_path(
            rospy.get_param("~board_model_path", model_config.get("board")),
            BOARD_MODEL_PATH,
        )
        segmentation_model_path = src_path(
            rospy.get_param("~segmentation_model_path", model_config.get("segmentation")),
            SEGMENTATION_MODEL_PATH,
        )
        block_calibration_path = src_path(
            rospy.get_param(
                "~block_pixel_to_tcp_calibration_path",
                calibration_config.get("block_pixel_to_tcp"),
            ),
            DEFAULT_BLOCK_CALIBRATION_PATH,
        )
        tray_calibration_path = src_path(
            rospy.get_param(
                "~tray_pixel_to_tcp_calibration_path",
                calibration_config.get("tray_pixel_to_tcp"),
            ),
            DEFAULT_TRAY_CALIBRATION_PATH,
        )
        self.block_calibration_path = block_calibration_path
        self.tray_calibration_path = tray_calibration_path
        hand_eye_matrix_path = src_path(
            calibration_config.get("hand_eye_matrix"),
            os.path.join(SRC_DIR, "camera", "config", "T_wrist2camera.npy"),
        )
        self.task_layout_path = rospy.get_param("~task_layout_path", TASK_LAYOUT_PATH)
        self.advanced_library_path = rospy.get_param("~advanced_library_path", JINJIE_LIB_PATH)
        model_paths = {
            "方块检测模型": detection_model_path,
            "托盘检测模型": board_model_path,
            "上表面分割模型": segmentation_model_path,
        }
        for model_name, model_path in model_paths.items():
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"{model_name}不存在: {model_path}")
            rospy.loginfo("%s路径: %s", model_name, model_path)

        # 两个检测模块采用懒加载；这里只注入配置路径，不改变其内部识别流程。
        board_scene_detector_module.BOARD_MODEL_PATH = board_model_path
        board_scene_detector_module._board_model = None
        block_detection_module.SEG_MODEL_PATH = segmentation_model_path
        block_detection_module._SEG_MODEL = None
        self.model = YOLO(detection_model_path)

        self.board_theta = 0.0
        self.save_top_surface_mask_vis = rospy.get_param("~save_top_surface_mask_vis", False)
        self.top_surface_mask_vis_path = rospy.get_param(
            "~top_surface_mask_vis_path",
            os.path.join(DEFAULT_DEBUG_DIR, "方块上表面掩码.jpg")
        )
        # 高位全场识别的模板匹配轮廓图，用于核对边框是否贴合方块。
        self.high_template_match_debug_path = rospy.get_param(
            "~high_template_match_debug_path",
            os.path.join(DEFAULT_DEBUG_DIR, "高位方块模板匹配结果.jpg")
        )
        visual_servo_debug_config = perception_config.get("visual_servo_debug", {})
        if not isinstance(visual_servo_debug_config, dict):
            raise ValueError("perception.yaml 的 visual_servo_debug 必须是字典")
        debug_enabled = visual_servo_debug_config.get("enabled", False)
        debug_output_dir = rospy.get_param(
            "~visual_servo_debug_output_dir",
            visual_servo_debug_config.get("output_dir", DEFAULT_DEBUG_DIR),
        )
        if not isinstance(debug_enabled, bool):
            raise ValueError("visual_servo_debug.enabled 必须是布尔值")
        if not isinstance(debug_output_dir, str):
            raise ValueError("visual_servo_debug.output_dir 必须是字符串")
        self.visual_servo_debug_enabled = debug_enabled
        self.visual_servo_debug_output_dir = debug_output_dir.strip()
        try:
            self.visual_servo_debug_video_fps = float(visual_servo_debug_config.get("video_fps", 10.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("visual_servo_debug.video_fps 必须是数值") from exc
        if not self.visual_servo_debug_output_dir:
            raise ValueError("visual_servo_debug.output_dir 不能为空")
        if not np.isfinite(self.visual_servo_debug_video_fps) or self.visual_servo_debug_video_fps <= 0.0:
            raise ValueError("visual_servo_debug.video_fps 必须是大于 0 的有限数值")

        # 关闭低位调试时不创建录像器，避免任何图像构建、编码或文件写入。
        self.visual_servo_debug_recorder = None
        self.visual_board_debug_recorder = None
        if self.visual_servo_debug_enabled:
            video_output_dir = self.experiment_archive_dir or self.visual_servo_debug_output_dir
            block_video_name = "方块视觉伺服调试.avi"
            board_video_name = "托盘视觉伺服调试.avi"
            self.visual_servo_debug_recorder = DebugVideoRecorder(
                os.path.join(video_output_dir, block_video_name),
                fps=self.visual_servo_debug_video_fps,
                enabled=True,
                latest_alias_path=(
                    os.path.join(self.visual_servo_debug_output_dir, block_video_name)
                    if self.experiment_archive_dir
                    else None
                ),
            )
            self.visual_board_debug_recorder = DebugVideoRecorder(
                os.path.join(video_output_dir, board_video_name),
                fps=self.visual_servo_debug_video_fps,
                enabled=True,
                latest_alias_path=(
                    os.path.join(self.visual_servo_debug_output_dir, board_video_name)
                    if self.experiment_archive_dir
                    else None
                ),
            )
        self.visual_board_grid_debug_path = rospy.get_param(
            "~visual_board_grid_debug_path",
            os.path.join(DEFAULT_DEBUG_DIR, "托盘格点粗定位.jpg")
        )
        board_servo = perception_config["board_servo"]
        block_servo = perception_config["block_servo"]
        self.board_low_roi_half_size = rospy.get_param("~board_low_roi_half_size", board_servo["roi_half_size"])
        self.board_low_blackhat_kernel_size = rospy.get_param(
            "~board_low_blackhat_kernel_size", board_servo["blackhat_kernel_size"]
        )
        self.board_low_min_dot_area = rospy.get_param("~board_low_min_dot_area", board_servo["min_dot_area"])
        self.board_low_max_dot_area = rospy.get_param("~board_low_max_dot_area", board_servo["max_dot_area"])
        self.board_low_min_dot_circularity = rospy.get_param(
            "~board_low_min_dot_circularity", board_servo["min_dot_circularity"]
        )
        self.board_low_max_dot_aspect_ratio = rospy.get_param(
            "~board_low_max_dot_aspect_ratio", board_servo["max_dot_aspect_ratio"]
        )
        def low_match_int_param(name, default, positive=False):
            value = rospy.get_param(f"~block_low_{name}", block_servo.get(name, default))
            if isinstance(value, bool):
                raise ValueError(f"block_servo.{name} 必须是非负整数")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"block_servo.{name} 必须是非负整数") from exc
            if not math.isfinite(number) or not number.is_integer() or number < 0:
                raise ValueError(f"block_servo.{name} 必须是非负整数")
            if positive and number <= 0:
                raise ValueError(f"block_servo.{name} 必须是正整数")
            return int(number)

        self.block_low_search_radius_px = low_match_int_param("search_radius_px", 30)
        self.block_low_fallback_search_radius_px = low_match_int_param("fallback_search_radius_px", 50)
        if self.block_low_fallback_search_radius_px < self.block_low_search_radius_px:
            raise ValueError("block_servo.fallback_search_radius_px 必须大于等于 search_radius_px")
        self.block_low_boundary_guard_px = low_match_int_param("boundary_guard_px", 3)
        self.block_low_kernel_safety_margin_px = low_match_int_param("kernel_safety_margin_px", 2)
        self.block_low_legacy_fallback_enabled = bool(
            rospy.get_param(
                "~block_low_legacy_fallback_enabled",
                block_servo.get("legacy_fallback_enabled", True),
            )
        )
        self.block_low_white_s_max = rospy.get_param("~block_low_white_s_max", block_servo["white_s_max"])
        self.block_low_white_v_min = rospy.get_param("~block_low_white_v_min", block_servo["white_v_min"])
        self.block_low_min_foreground_area = rospy.get_param(
            "~block_low_min_foreground_area", block_servo["min_foreground_area"]
        )
        self.block_angle_window_deg = float(
            rospy.get_param("~block_angle_window_deg", block_servo["angle_window_deg"])
        )
        self.block_angle_step_deg = float(rospy.get_param("~block_angle_step_deg", block_servo["angle_step_deg"]))

        fallback_config = perception_config.get("fallback", {})
        block_detection_module.ALLOW_COLOR_FALLBACK = bool(fallback_config.get("color_segmentation", True))

        high_match_config = perception_config.get("high_template_match", {})
        if not isinstance(high_match_config, dict):
            raise ValueError("high_template_match 必须是字典")

        def high_match_int_param(name, default, positive=False):
            value = rospy.get_param(f"~high_template_{name}", high_match_config.get(name, default))
            if isinstance(value, bool):
                raise ValueError(f"high_template_match.{name} 必须是非负整数")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"high_template_match.{name} 必须是非负整数") from exc
            if not math.isfinite(number) or not number.is_integer() or number < 0:
                raise ValueError(f"high_template_match.{name} 必须是非负整数")
            if positive and number <= 0:
                raise ValueError(f"high_template_match.{name} 必须是正整数")
            return int(number)

        high_match_enabled = bool(
            rospy.get_param("~high_template_enabled", high_match_config.get("enabled", True))
        )
        high_match_size_tolerance_px = high_match_int_param("size_tolerance_px", 4)
        high_match_relaxed_size_tolerance_px = high_match_int_param("relaxed_size_tolerance_px", 8)
        high_match_min_candidate_angles = high_match_int_param("min_candidate_angles", 3, positive=True)
        if high_match_relaxed_size_tolerance_px < high_match_size_tolerance_px:
            raise ValueError(
                "high_template_match.relaxed_size_tolerance_px 必须大于等于 size_tolerance_px"
            )
        high_match_kernel_safety_margin_px = high_match_int_param("kernel_safety_margin_px", 2)
        high_match_minimum_translation_margin_px = high_match_int_param(
            "minimum_translation_margin_px", 4
        )
        high_match_legacy_fallback_enabled = bool(
            rospy.get_param(
                "~high_template_legacy_fallback_enabled",
                high_match_config.get("legacy_fallback_enabled", True),
            )
        )
        block_scene_detector_module.HIGH_SCREENING_CONFIG = {
            "enabled": high_match_enabled,
            "size_tolerance_px": high_match_size_tolerance_px,
            "relaxed_size_tolerance_px": high_match_relaxed_size_tolerance_px,
            "min_candidate_angles": high_match_min_candidate_angles,
            "kernel_safety_margin_px": high_match_kernel_safety_margin_px,
            "minimum_translation_margin_px": high_match_minimum_translation_margin_px,
            "legacy_fallback_enabled": high_match_legacy_fallback_enabled,
        }

        servo_config = execution_config.get("servo", {})
        if not isinstance(servo_config, dict):
            raise ValueError("servo 必须是字典")
        visual_servo_enabled = servo_config.get("enabled", True)
        if not isinstance(visual_servo_enabled, bool):
            raise ValueError("servo.enabled 必须是 YAML 布尔值 true 或 false")
        # 标定模式与任务节点保持一致，即使配置关闭也按闭环方式校验观察 TCP。
        self.visual_servo_enabled = self.calibration_mode or visual_servo_enabled

        with open(VISUAL_SERVO_CONFIG_PATH, "r", encoding="utf-8") as config_file:
            visual_servo_config = yaml.safe_load(config_file) or {}
        if not isinstance(visual_servo_config, dict):
            raise ValueError("visual_servo.yaml 必须是字典")
        try:
            camera_to_sucker_offset = np.asarray(
                visual_servo_config.get("camera_to_sucker_offset_mm"),
                dtype=float,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("camera_to_sucker_offset_mm 必须包含 2 个有限数值") from exc
        if camera_to_sucker_offset.shape != (2,) or not np.all(
            np.isfinite(camera_to_sucker_offset)
        ):
            raise ValueError("camera_to_sucker_offset_mm 必须包含 2 个有限数值")
        self.high_tcp_safety_xy_offset = (
            (0.0, 0.0)
            if self.visual_servo_enabled
            else tuple(float(value) for value in camera_to_sucker_offset)
        )
        self.camera_to_sucker_offset_mm = tuple(
            float(value) for value in camera_to_sucker_offset
        )
        self.visual_servo_timing_debug = bool(
            rospy.get_param(
                "~visual_servo_timing_debug",
                servo_config.get("timing_debug", False),
            )
        )
        self.shooting_angle = [float(value) for value in execution_config["shooting_pose"]]
        if len(self.shooting_angle) != 6 or not np.all(np.isfinite(self.shooting_angle)):
            raise ValueError("shooting_pose 必须包含 6 个有限数值")
        motion_config = execution_config.get("motion", {})
        if not isinstance(motion_config, dict) or "minimum_tcp_z_mm" not in motion_config:
            raise ValueError(
                "execution.yaml 缺少唯一安全高度字段 motion.minimum_tcp_z_mm"
            )
        raw_minimum_tcp_z_mm = motion_config["minimum_tcp_z_mm"]
        if isinstance(raw_minimum_tcp_z_mm, bool):
            raise ValueError("minimum_tcp_z_mm 必须是大于 0 的有限数值")
        try:
            self.minimum_tcp_z_mm = float(raw_minimum_tcp_z_mm)
        except (TypeError, ValueError) as exc:
            raise ValueError("minimum_tcp_z_mm 必须是大于 0 的有限数值") from exc
        self.pick_surface_offset_mm = float(motion_config.get("pick_surface_offset_mm", 0.0))
        self.pick_approach_clearance_mm = float(
            motion_config.get("pick_approach_clearance_mm", 0.0)
        )
        self.arm_speed = float(motion_config.get("arm_speed", 0.0))
        self.pick_approach_speed = float(
            motion_config.get("pick_approach_speed", 0.0)
        )
        if not np.isfinite(self.minimum_tcp_z_mm) or self.minimum_tcp_z_mm <= 0.0:
            raise ValueError("minimum_tcp_z_mm 必须是大于 0 的有限数值")
        if not np.isfinite(self.pick_surface_offset_mm):
            raise ValueError("pick_surface_offset_mm 必须是有限数值")
        if (
            not np.isfinite(self.pick_approach_clearance_mm)
            or self.pick_approach_clearance_mm <= 0.0
        ):
            raise ValueError("pick_approach_clearance_mm 必须是大于 0 的有限数值")

        motor_config = execution_config.get("tool_motor", {})
        try:
            self.initial_motor_angle_deg = float(motor_config["initial_angle_deg"])
            self.motor_velocity_deg_per_sec = float(motor_config["velocity_deg_per_sec"])
            self.motor_lower_margin_deg = float(motor_config["lower_margin_deg"])
            self.motor_upper_margin_deg = float(motor_config["upper_margin_deg"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("tool_motor 配置缺失或格式无效") from exc

        optimizer_config = perception_config.get("task_sequence_optimizer", {})
        if not isinstance(optimizer_config, dict):
            raise ValueError("task_sequence_optimizer 必须是字典")
        self.task_sequence_optimizer_mode = str(
            rospy.get_param(
                "~task_sequence_optimizer_mode",
                optimizer_config.get("mode", "shadow"),
            )
        ).strip().lower()
        if self.task_sequence_optimizer_mode not in ("legacy", "shadow", "execute"):
            raise ValueError(
                "task_sequence_optimizer.mode 只能是 legacy、shadow 或 execute"
            )

        def positive_optimizer_int(name, default):
            value = rospy.get_param(
                f"~task_sequence_optimizer_{name}",
                optimizer_config.get(name, default),
            )
            if isinstance(value, bool):
                raise ValueError(f"task_sequence_optimizer.{name} 必须是正整数")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"task_sequence_optimizer.{name} 必须是正整数"
                ) from exc
            if not math.isfinite(number) or not number.is_integer() or number <= 0:
                raise ValueError(f"task_sequence_optimizer.{name} 必须是正整数")
            return int(number)

        self.task_sequence_optimizer_beam_width = positive_optimizer_int(
            "beam_width", 1000
        )
        self.task_sequence_optimizer_report_top_candidates = positive_optimizer_int(
            "report_top_candidates", 20
        )
        self.last_task_plan_result = None

        dynamic_config = perception_config.get("dynamic_board_selection", {})
        if not isinstance(dynamic_config, dict):
            raise ValueError("dynamic_board_selection 必须是字典")
        self.dynamic_board_selection_mode = str(rospy.get_param(
            "~dynamic_board_selection_mode",
            dynamic_config.get("mode", "shadow"),
        )).strip().lower()
        if self.dynamic_board_selection_mode not in ("disabled", "shadow", "execute"):
            raise ValueError(
                "dynamic_board_selection.mode 只能是 disabled、shadow 或 execute"
            )

        def positive_dynamic_int(name, default):
            value = rospy.get_param(
                f"~dynamic_board_selection_{name}",
                dynamic_config.get(name, default),
            )
            if isinstance(value, bool):
                raise ValueError(f"dynamic_board_selection.{name} 必须是正整数")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"dynamic_board_selection.{name} 必须是正整数"
                ) from exc
            if not math.isfinite(number) or not number.is_integer() or number <= 0.0:
                raise ValueError(f"dynamic_board_selection.{name} 必须是正整数")
            return int(number)

        self.dynamic_board_coarse_top_k = positive_dynamic_int("coarse_top_k", 300)
        self.dynamic_board_final_candidate_k = positive_dynamic_int(
            "final_candidate_k", 20
        )
        keep_boundary_ties = rospy.get_param(
            "~dynamic_board_selection_keep_coarse_boundary_ties",
            dynamic_config.get("keep_coarse_boundary_ties", True),
        )
        if not isinstance(keep_boundary_ties, bool):
            raise ValueError(
                "dynamic_board_selection.keep_coarse_boundary_ties 必须是布尔值"
            )
        self.dynamic_board_keep_coarse_boundary_ties = keep_boundary_ties
        self.dynamic_board_comparison_beam_width = positive_dynamic_int(
            "comparison_beam_width", 1000
        )
        self.dynamic_board_comparison_returned_candidates = positive_dynamic_int(
            "comparison_returned_candidates", 1
        )
        self.dynamic_board_comparison_worker_count = positive_dynamic_int(
            "comparison_worker_count", 12
        )
        self.dynamic_board_confirmation_candidate_k = positive_dynamic_int(
            "confirmation_candidate_k", 10
        )
        self.dynamic_board_confirmation_beam_width = positive_dynamic_int(
            "confirmation_beam_width", 5000
        )
        self.dynamic_board_confirmation_returned_candidates = positive_dynamic_int(
            "confirmation_returned_candidates", 20
        )
        self.dynamic_board_confirmation_worker_count = positive_dynamic_int(
            "confirmation_worker_count", 10
        )

        def finite_dynamic_seconds(name, default, allow_zero):
            value = rospy.get_param(
                f"~dynamic_board_selection_{name}",
                dynamic_config.get(name, default),
            )
            try:
                seconds = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"dynamic_board_selection.{name} 必须是有限秒数"
                ) from exc
            if not math.isfinite(seconds) or seconds < 0.0 or (
                not allow_zero and seconds == 0.0
            ):
                qualifier = "非负" if allow_zero else "大于 0"
                raise ValueError(
                    f"dynamic_board_selection.{name} 必须是{qualifier}有限秒数"
                )
            return seconds

        self.dynamic_board_soft_time_budget_sec = finite_dynamic_seconds(
            "soft_time_budget_sec", 10.0, True
        )
        self.dynamic_board_failure_prompt_timeout_sec = finite_dynamic_seconds(
            "failure_prompt_timeout_sec", 60.0, False
        )
        raw_library_path = rospy.get_param(
            "~dynamic_board_selection_library_path",
            dynamic_config.get("library_path", DEFAULT_V5_BOARD_LIBRARY_PATH),
        )
        self.dynamic_board_library_path = src_path(
            raw_library_path,
            DEFAULT_V5_BOARD_LIBRARY_PATH,
        )
        self.dynamic_board_runtime_config = {
            "mode": self.dynamic_board_selection_mode,
            "library_path": self.dynamic_board_library_path,
            "coarse_top_k": self.dynamic_board_coarse_top_k,
            "final_candidate_k": self.dynamic_board_final_candidate_k,
            "keep_coarse_boundary_ties": self.dynamic_board_keep_coarse_boundary_ties,
            "comparison_beam_width": self.dynamic_board_comparison_beam_width,
            "comparison_returned_candidates": (
                self.dynamic_board_comparison_returned_candidates
            ),
            "comparison_worker_count": self.dynamic_board_comparison_worker_count,
            "confirmation_candidate_k": (
                self.dynamic_board_confirmation_candidate_k
            ),
            "confirmation_beam_width": self.dynamic_board_confirmation_beam_width,
            "confirmation_returned_candidates": (
                self.dynamic_board_confirmation_returned_candidates
            ),
            "confirmation_worker_count": self.dynamic_board_confirmation_worker_count,
            "soft_time_budget_sec": self.dynamic_board_soft_time_budget_sec,
            "failure_prompt_timeout_sec": self.dynamic_board_failure_prompt_timeout_sec,
        }
        self.dynamic_board_library = None
        self.dynamic_board_candidate_selector = None
        self.dynamic_final_board_selector = None
        self.last_dynamic_board_decision = None
        self.last_dynamic_board_report_path = None
        if (
            not self.calibration_mode
            and self.dynamic_board_selection_mode == "execute"
            and self.visual_servo_enabled
        ):
            raise ValueError(
                "动态盘面 execute 要求 execution.yaml 中 servo.enabled=false"
            )
        if (
            not self.calibration_mode
            and self.dynamic_board_selection_mode != "disabled"
        ):
            self.dynamic_board_library = load_v5_board_library(
                self.dynamic_board_library_path
            )
            self.dynamic_board_candidate_selector = BoardCandidateSelector(
                self.dynamic_board_library,
                BoardCandidateSelectorConfig(
                    coarse_top_k=self.dynamic_board_coarse_top_k,
                    final_candidate_k=self.dynamic_board_final_candidate_k,
                    keep_coarse_boundary_ties=(
                        self.dynamic_board_keep_coarse_boundary_ties
                    ),
                ),
            )
            self.dynamic_final_board_selector = FinalBoardSelector(
                self.dynamic_board_library,
                FinalBoardSelectorConfig(
                    comparison_beam_width=(
                        self.dynamic_board_comparison_beam_width
                    ),
                    comparison_returned_candidates=(
                        self.dynamic_board_comparison_returned_candidates
                    ),
                    comparison_worker_count=(
                        self.dynamic_board_comparison_worker_count
                    ),
                    confirmation_candidate_k=(
                        self.dynamic_board_confirmation_candidate_k
                    ),
                    confirmation_beam_width=(
                        self.dynamic_board_confirmation_beam_width
                    ),
                    confirmation_returned_candidates=(
                        self.dynamic_board_confirmation_returned_candidates
                    ),
                    confirmation_worker_count=(
                        self.dynamic_board_confirmation_worker_count
                    ),
                    soft_time_budget_sec=self.dynamic_board_soft_time_budget_sec,
                ),
            )
            rospy.loginfo(
                "V5 动态盘面库已加载：%d 张，SHA256=%s",
                self.dynamic_board_library.board_count,
                self.dynamic_board_library.source_sha256,
            )

        localization_config = perception_config.get("high_tcp_localization", {})
        if not isinstance(localization_config, dict):
            raise ValueError("high_tcp_localization 必须是字典")

        def finite_range(name, default):
            values = np.asarray(localization_config.get(name, default), dtype=float)
            if values.shape != (2,) or not np.all(np.isfinite(values)) or values[0] >= values[1]:
                raise ValueError(f"high_tcp_localization.{name} 必须是递增的两个有限数值")
            return tuple(float(value) for value in values)

        safe_x_range_mm = finite_range("safe_x_range_mm", [-444.224, -148.17])
        safe_y_range_mm = finite_range("safe_y_range_mm", [-263.279, 315.925])
        self.safe_x_range_mm = safe_x_range_mm
        self.safe_y_range_mm = safe_y_range_mm

        pick_height_config = perception_config.get("pick_height", {})
        if not isinstance(pick_height_config, dict):
            raise ValueError("pick_height 必须是字典")
        self.block_observation_height_mm = float(
            pick_height_config.get("block_observation_height_mm", 192.0)
        )
        if (
            not np.isfinite(self.block_observation_height_mm)
            or self.block_observation_height_mm <= 0.0
        ):
            raise ValueError("block_observation_height_mm 必须是大于 0 的有限数值")

        depth_config = perception_config.get("calibration_depth", {})
        if not isinstance(depth_config, dict):
            raise ValueError("calibration_depth 必须是字典")
        required_depth_keys = (
            "frame_count",
            "min_valid_frames",
            "capture_timeout_sec",
            "block_max_mad_mm",
            "block_plane_max_rmse_mm",
            "tray_tcp_below_block_observation_mm",
        )
        missing_depth_keys = [key for key in required_depth_keys if key not in depth_config]
        if missing_depth_keys:
            raise ValueError(
                "calibration_depth 缺少配置项: " + ", ".join(missing_depth_keys)
            )
        self.calibration_depth_frame_count = int(depth_config["frame_count"])
        self.calibration_depth_min_valid_frames = int(depth_config["min_valid_frames"])
        self.calibration_depth_capture_timeout_sec = float(
            depth_config["capture_timeout_sec"]
        )
        self.block_depth_max_mad_mm = float(depth_config["block_max_mad_mm"])
        self.block_plane_max_rmse_mm = float(depth_config["block_plane_max_rmse_mm"])
        self.tray_tcp_below_block_observation_mm = float(
            depth_config["tray_tcp_below_block_observation_mm"]
        )
        if self.calibration_depth_frame_count <= 0:
            raise ValueError("calibration_depth.frame_count 必须是大于 0 的整数")
        if self.calibration_depth_min_valid_frames <= 0:
            raise ValueError("calibration_depth.min_valid_frames 必须是大于 0 的整数")
        if self.calibration_depth_min_valid_frames > self.calibration_depth_frame_count:
            raise ValueError(
                "calibration_depth.min_valid_frames="
                f"{self.calibration_depth_min_valid_frames} 不能大于 "
                "calibration_depth.frame_count="
                f"{self.calibration_depth_frame_count}"
            )
        positive_depth_values = {
            "capture_timeout_sec": self.calibration_depth_capture_timeout_sec,
            "block_max_mad_mm": self.block_depth_max_mad_mm,
            "block_plane_max_rmse_mm": self.block_plane_max_rmse_mm,
        }
        for name, value in positive_depth_values.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"calibration_depth.{name} 必须是大于 0 的有限数值"
                )
        if not np.isfinite(self.tray_tcp_below_block_observation_mm):
            raise ValueError(
                "calibration_depth.tray_tcp_below_block_observation_mm 必须是有限数值"
            )

        self.high_tcp_localizer = None
        self.depth_rough_localizer = None
        self.stable_world_points_client = None
        if self.calibration_mode:
            wrist_to_camera = np.load(hand_eye_matrix_path)
            self.depth_rough_localizer = DepthRoughLocalizer(
                self.shooting_angle,
                wrist_to_camera,
            )
            self.stable_world_points_client = rospy.ServiceProxy(
                "/camera/stable_world_points",
                GetStableWorldPoints,
            )
            rospy.loginfo("标定模式：高位粗定位使用批量稳定深度 XYZ")
        else:
            self.high_tcp_localizer = HighPixelToTcpLocalizer(
                block_calibration_path=block_calibration_path,
                tray_calibration_path=tray_calibration_path,
                shooting_pose=self.shooting_angle,
                tcp_min_xyz=(safe_x_range_mm[0], safe_y_range_mm[0], self.minimum_tcp_z_mm),
                tcp_max_xyz=(safe_x_range_mm[1], safe_y_range_mm[1], None),
                safety_xy_offset=self.high_tcp_safety_xy_offset,
            )
            safety_mode = "闭环原始预测 TCP" if self.visual_servo_enabled else "开环吸盘偏置后 TCP"
            rospy.loginfo(
                "正式模式：高位 TCP 标定已加载，方块=%s，托盘=%s，安全校验=%s",
                block_calibration_path,
                tray_calibration_path,
                safety_mode,
            )
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)
        self.prepare_task_service = rospy.Service("/perception/prepare_task", PrepareTask, self.prepare_task)
        self.get_task_target_service = rospy.Service(
            "/perception/get_task_target", GetTaskTarget, self.get_task_target
        )
        self.block_offset_service = rospy.Service(
            "/perception/block_offset", DetectBlockOffset, self.detect_block_offset_service
        )
        self.board_offset_service = rospy.Service(
            "/perception/board_offset", DetectBoardOffset, self.detect_board_offset_service
        )
        self.get_operator_prompt_service_handle = rospy.Service(
            "/perception/get_operator_prompt",
            GetOperatorPrompt,
            self.get_operator_prompt_service,
        )
        self.respond_operator_prompt_service_handle = rospy.Service(
            "/perception/respond_operator_prompt",
            RespondOperatorPrompt,
            self.respond_operator_prompt_service,
        )
        rospy.on_shutdown(self.close_runtime_resources)
        rospy.loginfo("图像处理节点订阅去畸变彩图: %s", self.image_topic)
        rospy.loginfo("图像处理服务已启动")

    def terminate_high_mask_editor_process(self):
        """节点退出时终止仍在等待人工输入的独立 GUI 子进程。"""
        with self.high_mask_editor_process_lock:
            process = self.active_high_mask_editor_process
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)
        except Exception as exc:
            rospy.logwarn("终止高位 Mask 编辑子进程失败: %s", exc)

    def close_runtime_resources(self):
        """节点退出时统一关闭 GUI 子进程和调试视频。"""
        prompt_broker = getattr(self, "operator_prompt_broker", None)
        if prompt_broker is not None:
            prompt_broker.close()
        self.terminate_high_mask_editor_process()
        self.close_debug_video_recorders()

    def get_operator_prompt_service(self, _request):
        """向网页控制台返回当前待处理的固定选项提示。"""
        prompt = self.operator_prompt_broker.snapshot()
        return GetOperatorPromptResponse(**prompt)

    def respond_operator_prompt_service(self, request):
        """接受网页对当前提示的回答，拒绝任意或过期内容。"""
        try:
            choice = self.operator_prompt_broker.respond(
                request.prompt_id,
                request.choice,
            )
            return RespondOperatorPromptResponse(
                success=True,
                code="accepted",
                message=f"已接受人工选择：{choice}",
            )
        except OperatorPromptChoiceError as exc:
            return RespondOperatorPromptResponse(
                success=False,
                code="invalid_choice",
                message=str(exc),
            )
        except OperatorPromptConflict as exc:
            return RespondOperatorPromptResponse(
                success=False,
                code="conflict",
                message=str(exc),
            )

    def choose_dynamic_selection_failure(self, reason, allow_continue=False):
        """按启动入口选择网页按钮或原终端字符交互。"""
        timeout = getattr(
            self,
            "dynamic_board_failure_prompt_timeout_sec",
            60.0,
        )
        if getattr(self, "interaction_mode", "terminal") == "web":
            rospy.logwarn("V5 动态盘面等待网页人工选择：%s", reason)
            return self.operator_prompt_broker.request_dynamic_failure(
                reason,
                timeout,
                allow_continue=allow_continue,
            )
        return prompt_dynamic_selection_failure(
            reason,
            timeout,
            allow_continue=allow_continue,
        )

    def close_debug_video_recorders(self):
        """节点退出时释放视频文件句柄，避免最后几帧没有写入文件。"""
        if self.visual_servo_debug_recorder is not None:
            self.visual_servo_debug_recorder.release()
        if self.visual_board_debug_recorder is not None:
            self.visual_board_debug_recorder.release()

    def record_visual_servo_debug_frame(self, debug_panel):
        """仅在低位调试开启时追加方块视觉伺服调试视频帧。"""
        if not self.visual_servo_debug_enabled or self.visual_servo_debug_recorder is None:
            return False
        return self.visual_servo_debug_recorder.write(debug_panel)

    def record_visual_board_debug_frame(self, debug_panel):
        """仅在低位调试开启时追加托盘视觉伺服调试视频帧。"""
        if not self.visual_servo_debug_enabled or self.visual_board_debug_recorder is None:
            return False
        return self.visual_board_debug_recorder.write(debug_panel)

    def save_experiment_debug_image(self, latest_path, image):
        """保存最新调试图，并在标定会话目录保留同名归档。"""
        latest_saved = save_image_to_path(latest_path, image)
        archive_dir = getattr(self, "experiment_archive_dir", None)
        if archive_dir:
            archive_path = os.path.join(archive_dir, os.path.basename(latest_path))
            if os.path.abspath(archive_path) != os.path.abspath(latest_path):
                save_image_to_path(archive_path, image)
        return latest_saved

    def log_visual_servo_detection_timing(
        self,
        target_type,
        request_started_at,
        snapshot_finished_at,
        detection_started_at=None,
        detection_finished_at=None,
        debug_started_at=None,
        debug_finished_at=None,
        block_timing=None,
    ):
        """调试时输出图像服务内部耗时，所有时间戳均在日志输出前采集。"""
        if not getattr(self, "visual_servo_timing_debug", False):
            return

        def elapsed_ms(started_at, finished_at):
            if started_at is None or finished_at is None:
                return "-"
            return f"{(finished_at - started_at) * 1000:.1f}ms"

        finished_at = debug_finished_at or detection_finished_at or snapshot_finished_at
        print(
            f"[视觉伺服识别耗时][{target_type}] "
            f"取图={elapsed_ms(request_started_at, snapshot_finished_at)}，"
            f"识别={elapsed_ms(detection_started_at, detection_finished_at)}，"
            f"调试图输出={elapsed_ms(debug_started_at, debug_finished_at)}，"
            f"服务内总={elapsed_ms(request_started_at, finished_at)}"
        )
        if target_type == "方块" and block_timing is not None:
            self.log_block_servo_stage_timing(block_timing)

    def log_block_servo_stage_timing(self, timing_info):
        """输出低位方块识别的可归因分段耗时。"""
        if not getattr(self, "visual_servo_timing_debug", False):
            return

        def format_ms(value):
            return "未执行" if value is None else f"{float(value):.1f}ms"

        stages_ms = timing_info.get("阶段毫秒", {})
        roi_size = timing_info.get("ROI尺寸")
        match_image_size = timing_info.get("匹配图尺寸")
        roi_text = "-" if roi_size is None else f"{roi_size[0]}x{roi_size[1]}"
        match_image_text = (
            "-" if match_image_size is None else f"{match_image_size[0]}x{match_image_size[1]}"
        )
        template_count = timing_info.get("模板数量")
        kernel_size = timing_info.get("模板核尺寸")
        template_text = "未执行"
        if template_count is not None and kernel_size is not None:
            if isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 2:
                kernel_text = f"{kernel_size[0]}x{kernel_size[1]}"
            else:
                kernel_text = str(kernel_size)
            template_text = f"{template_count}×{kernel_text}"

        config_ms = timing_info.get("模板几何配置毫秒")
        detector_total_ms = timing_info.get("总计毫秒")
        total_ms = None
        if config_ms is not None and detector_total_ms is not None:
            total_ms = float(config_ms) + float(detector_total_ms)

        print(
            f"[方块低位分段] 类别={timing_info.get('类别') or '-'} "
            f"状态={timing_info.get('状态') or '-'} "
            f"后端={timing_info.get('后端') or '未执行'} "
            f"ROI={roi_text} 匹配图={match_image_text} 模板={template_text} "
            f"前景面积={timing_info.get('前景面积') if timing_info.get('前景面积') is not None else '-'} "
            f"模式={timing_info.get('匹配模式') or '-'} "
            f"缓存={timing_info.get('模板缓存') or '-'}"
        )
        print(
            f"配置={format_ms(config_ms)}，"
            f"先验ROI={format_ms(stages_ms.get('先验ROI'))}，"
            f"RGB分割={format_ms(stages_ms.get('RGB分割'))}，"
            f"模板生成={format_ms(stages_ms.get('模板生成'))}，"
            f"张量准备={format_ms(stages_ms.get('张量准备'))}，"
            f"卷积选优={format_ms(stages_ms.get('卷积选优'))}，"
            f"匹配收尾={format_ms(stages_ms.get('匹配收尾'))}，"
            f"检测调试图={format_ms(stages_ms.get('检测调试图'))}，"
            f"合计={format_ms(total_ms)}"
        )

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            image_stamp = getattr(getattr(msg, "header", None), "stamp", None)
            with self.image_condition:
                self.latest_image = image
                self.latest_image_stamp = image_stamp
                self.image_condition.notify_all()
        except Exception as e:
            rospy.logerr("图像转换失败: %s" % e)

    def get_image_snapshot_newer_than(self, request_stamp):
        """等待发布时间晚于本次请求的图像，避免使用请求前的缓存帧。"""
        with self.image_condition:
            deadline = time.monotonic() + self.fresh_image_timeout_sec
            while (
                self.latest_image is None
                or self.latest_image_stamp is None
                or self.latest_image_stamp <= request_stamp
            ):
                remaining_sec = deadline - time.monotonic()
                if remaining_sec <= 0.0:
                    return None
                self.image_condition.wait(remaining_sec)
            return self.latest_image.copy()

    @staticmethod
    def make_high_localization_diagnostic(
        px,
        py,
        image_shape,
        depth_sample_pixel_xy=(0.0, 0.0),
        world_position=(0.0, 0.0, 0.0),
        world_position_valid=False,
        source="tcp_calibration",
        depth_valid_frame_count=0,
        depth_median_mm=0.0,
        depth_mad_mm=0.0,
        calibration_target_tcp_z_mm=0.0,
    ):
        """整理高位粗定位和稳定深度诊断字段。"""
        height, width = image_shape[:2]
        return {
            "high_detected_pixel_xy": (float(px), float(py)),
            "high_depth_sample_pixel_xy": tuple(
                float(value) for value in depth_sample_pixel_xy
            ),
            "high_image_center_xy": (float(width) / 2.0, float(height) / 2.0),
            "high_world_position": tuple(float(value) for value in world_position),
            "high_world_position_valid": bool(world_position_valid),
            "rough_localization_source": str(source),
            "depth_valid_frame_count": int(depth_valid_frame_count),
            "depth_median_mm": float(depth_median_mm),
            "depth_mad_mm": float(depth_mad_mm),
            "calibration_target_tcp_z_mm": float(calibration_target_tcp_z_mm),
        }

    def resolve_pick_surface_height(self, px, py, predicted_tcp_z_mm):
        """正式模式由观察 TCP Z 反推出方块上表面绝对 Z。"""
        del px, py
        predicted_tcp_z_mm = float(predicted_tcp_z_mm)
        if not np.isfinite(predicted_tcp_z_mm):
            raise ValueError("方块标定观察 TCP Z 不是有限数值")
        surface_z_mm = predicted_tcp_z_mm - self.block_observation_height_mm

        if not np.isfinite(surface_z_mm):
            raise ValueError(f"方块表面高度不是有限数值: {surface_z_mm}")
        pick_tcp_z_mm = surface_z_mm + self.pick_surface_offset_mm
        if not np.isfinite(pick_tcp_z_mm) or pick_tcp_z_mm < self.minimum_tcp_z_mm:
            raise ValueError(
                f"最终抓取 TCP Z={pick_tcp_z_mm:.3f} mm 低于安全下限 "
                f"{self.minimum_tcp_z_mm:.3f} mm"
            )
        return float(surface_z_mm), (0.0, 0.0)

    def _validate_calibration_pose(self, pose, label):
        """校验深度粗定位生成的六维 TCP 位姿。"""
        values = np.asarray(pose, dtype=float)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{label}必须包含 6 个有限数值")
        tcp_xyz = values[:3]
        is_safe = (
            self.safe_x_range_mm[0] <= tcp_xyz[0] <= self.safe_x_range_mm[1]
            and self.safe_y_range_mm[0] <= tcp_xyz[1] <= self.safe_y_range_mm[1]
            and tcp_xyz[2] >= self.minimum_tcp_z_mm
        )
        if not is_safe:
            minimum_xyz = [
                float(self.safe_x_range_mm[0]),
                float(self.safe_y_range_mm[0]),
                float(self.minimum_tcp_z_mm),
            ]
            maximum_xyz = [
                float(self.safe_x_range_mm[1]),
                float(self.safe_y_range_mm[1]),
                None,
            ]
            raise ValueError(
                f"{label} TCP XYZ {tcp_xyz.tolist()} 超出安全范围："
                f"最小值 {minimum_xyz}，最大值 {maximum_xyz}"
            )
        return values.tolist()

    def _query_stable_world_points(self, pixel_xy):
        """批量查询同一批深度帧中的世界 XYZ，并校验响应结构。"""
        if self.stable_world_points_client is None:
            raise RuntimeError("标定模式未初始化批量稳定深度客户端")
        pixels = np.asarray(pixel_xy, dtype=float)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.all(np.isfinite(pixels)):
            raise ValueError("稳定深度查询像素必须是有限的 N×2 数组")
        xs = [int(round(value)) for value in pixels[:, 0]]
        ys = [int(round(value)) for value in pixels[:, 1]]
        response = self.stable_world_points_client(
            xs,
            ys,
            self.calibration_depth_frame_count,
            self.calibration_depth_min_valid_frames,
            self.calibration_depth_capture_timeout_sec,
        )
        if not getattr(response, "success", False):
            raise RuntimeError(f"稳定深度 XYZ 查询失败: {response.message}")
        point_valid = list(response.point_valid)
        valid_counts = list(response.valid_frame_counts)
        medians = list(response.depth_median_mm)
        mads = list(response.depth_mad_mm)
        world_flat = np.asarray(response.world_xyz, dtype=float)
        count = len(pixels)
        if (
            len(point_valid) != count
            or len(valid_counts) != count
            or len(medians) != count
            or len(mads) != count
            or world_flat.shape != (count * 3,)
        ):
            raise RuntimeError("稳定深度 XYZ 服务返回字段长度不一致")
        world = world_flat.reshape(count, 3)
        samples = []
        for index in range(count):
            valid_count = int(valid_counts[index])
            depth_median = float(medians[index])
            depth_mad = float(mads[index])
            if (
                not point_valid[index]
                or valid_count < self.calibration_depth_min_valid_frames
                or not np.all(np.isfinite(world[index]))
                or not np.isfinite(depth_median)
                or depth_median <= 0.0
                or not np.isfinite(depth_mad)
                or depth_mad < 0.0
            ):
                raise RuntimeError(
                    f"像素 ({xs[index]}, {ys[index]}) 的稳定深度结果无效"
                )
            samples.append({
                "pixel": (float(xs[index]), float(ys[index])),
                "world": world[index].copy(),
                "valid_count": valid_count,
                "depth_median_mm": depth_median,
                "depth_mad_mm": depth_mad,
            })
        return samples

    def compute_board_theta_from_grid_points(self, grid_points):
        """只根据托盘四角像素计算托盘旋转角，不再依赖九点坐标标定。"""
        left_top = grid_points[BOARD_ROW_COUNT][1]
        left_bottom = grid_points[1][1]
        right_top = grid_points[BOARD_ROW_COUNT][BOARD_COL_COUNT]
        right_bottom = grid_points[1][BOARD_COL_COUNT]
        dx1 = right_top[0] - left_top[0]
        dx2 = right_bottom[0] - left_bottom[0]
        dy1 = right_top[1] - left_top[1]
        dy2 = right_bottom[1] - left_bottom[1]
        dx = (dx1 + dx2) / 2.0
        dy = (dy1 + dy2) / 2.0
        if abs(dx) < 1e-6:
            return 0.0
        return float(np.degrees(np.arctan2(dy, dx)))

    def _detect_board_for_task(self, image):
        """识别并缓存高位托盘格点。"""
        result = board_grid_detect(image, debug_path=None)
        debug_image = result.get("debug_image")
        if debug_image is None:
            debug_image = image
        self.save_experiment_debug_image(self.visual_board_grid_debug_path, debug_image)
        if not result["found"]:
            raise RuntimeError(result["message"])
        self.board_grid_points = result["grid_points"]
        self.board_grid_image_shape = image.shape[:2]
        self.board_grid_image = image.copy()
        self.board_theta = self.compute_board_theta_from_grid_points(self.board_grid_points)
        rospy.loginfo("托盘像素角度: %.2f 度", self.board_theta)

    def _run_high_mask_manual_editor(self, image, blocks):
        """在独立无 ROS 子进程中编辑 Mask，并只返回严格校验后的二值结果。"""
        if not self.high_mask_manual_editor_enabled:
            return [block["mask"].copy() for block in blocks]
        if not os.environ.get("DISPLAY", "").strip():
            raise RuntimeError(
                "高位 Mask 人工编辑已启用，但图像节点环境中没有 DISPLAY；"
                "请从图形桌面终端启动节点，或关闭 high_mask_manual_editor.enabled"
            )
        if not os.path.isfile(self.high_mask_editor_script_path):
            raise RuntimeError(f"高位 Mask 编辑脚本不存在: {self.high_mask_editor_script_path}")

        with tempfile.TemporaryDirectory(prefix="single_arm_tetris_high_mask_") as temp_dir:
            manifest_path = 创建高位Mask编辑会话(temp_dir, image, blocks)
            child_env = os.environ.copy()
            child_env[会话环境变量] = str(manifest_path)
            child_env[预览设备环境变量] = self.high_mask_editor_preview_device
            if self.high_mask_editor_preview_device == "cpu":
                child_env["CUDA_VISIBLE_DEVICES"] = ""
            child_env["PYTHONUNBUFFERED"] = "1"
            matplotlib_cache = os.path.join(temp_dir, "matplotlib_cache")
            os.makedirs(matplotlib_cache, exist_ok=True)
            child_env["MPLCONFIGDIR"] = matplotlib_cache

            rospy.loginfo(
                "等待高位 Mask 人工编辑，预览设备=%s，"
                "按 Enter/Q 提交，Esc 取消本轮识别",
                self.high_mask_editor_preview_device,
            )
            process = subprocess.Popen(
                [sys.executable, self.high_mask_editor_script_path],
                cwd=SRC_DIR,
                env=child_env,
            )
            with self.high_mask_editor_process_lock:
                if self.active_high_mask_editor_process is not None:
                    process.terminate()
                    process.wait(timeout=2.0)
                    raise RuntimeError("已有高位 Mask 编辑子进程正在运行")
                self.active_high_mask_editor_process = process
            try:
                while process.poll() is None:
                    if rospy.is_shutdown():
                        self.terminate_high_mask_editor_process()
                        raise RuntimeError("ROS 节点正在退出，已取消高位 Mask 编辑")
                    time.sleep(0.1)
                exit_code = int(process.returncode)
            finally:
                with self.high_mask_editor_process_lock:
                    if self.active_high_mask_editor_process is process:
                        self.active_high_mask_editor_process = None

            if exit_code == 编辑取消退出码:
                raise RuntimeError("用户取消了本轮高位 Mask 编辑，请重新识别")
            if exit_code != 0:
                raise RuntimeError(f"高位 Mask 编辑子进程异常退出，退出码={exit_code}")
            return 读取已提交高位Mask(manifest_path, blocks)

    def _detect_blocks_automatic(self, image):
        """自动识别高位方块，保留初始 Mask 和重匹配所需的完整数据。"""
        geometry = load_template_geometry("high")
        blocks, debug_image = detect_blocks_in_image(
            image,
            self.model,
            template_geometry=geometry,
            crop_margin=8,
            save_mask_overlay=self.save_top_surface_mask_vis,
        )
        if not blocks:
            raise RuntimeError("高位没有识别到方块")
        return blocks, debug_image, geometry

    def _edit_and_rematch_blocks(self, image, blocks, geometry, debug_image):
        """按配置人工编辑 Mask，并返回使用编辑结果重匹配后的完整方块。"""
        if getattr(self, "high_mask_manual_editor_enabled", False):
            edited_masks = self._run_high_mask_manual_editor(image, blocks)
            blocks, debug_image = rematch_blocks_from_masks(
                image,
                blocks,
                edited_masks,
                template_geometry=geometry,
                save_mask_overlay=self.save_top_surface_mask_vis,
            )
        return blocks, debug_image

    def _save_high_block_debug_images(self, blocks, debug_image):
        """保存最终采用的高位模板匹配和 Mask 调试图。"""
        self.save_experiment_debug_image(self.high_template_match_debug_path, debug_image)
        if self.save_top_surface_mask_vis:
            mask_image = blocks[-1].get("mask_overlay")
            if mask_image is None:
                mask_image = debug_image
            self.save_experiment_debug_image(self.top_surface_mask_vis_path, mask_image)

    @staticmethod
    def _summarize_detected_blocks(blocks):
        """提取规划需要的类别、像素和角度，并统计各类别数量。"""
        recognized = []
        counts = {category: 0 for category in BLOCK_CATEGORY_NAMES}
        for block in blocks:
            category = normalize_category_name(block["category"])
            if category not in counts:
                rospy.logwarn("忽略未知方块类别: %s", category)
                continue
            recognized.append({
                "category": category,
                "px": float(block["px"]),
                "py": float(block["py"]),
                "theta": float(block["theta"]),
            })
            counts[category] += 1
        if not recognized:
            raise RuntimeError("没有可用于任务规划的已知类别方块")
        return recognized, [counts[category] for category in BLOCK_CATEGORY_NAMES]

    def _detect_blocks_raw(self, image):
        """保持标定流程兼容：自动识别后立即编辑，再返回规划字段。"""
        blocks, debug_image, geometry = self._detect_blocks_automatic(image)
        blocks, debug_image = self._edit_and_rematch_blocks(
            image,
            blocks,
            geometry,
            debug_image,
        )
        self._save_high_block_debug_images(blocks, debug_image)
        return self._summarize_detected_blocks(blocks)

    def _build_observed_blocks_for_task(self, blocks, image_shape):
        """正式模式使用部署标定把已确认的方块识别结果转成粗观察位。"""
        observed_blocks = []
        category_order = {
            category: index for index, category in enumerate(BLOCK_CATEGORY_NAMES)
        }
        # 人工修正完成后再稳定排序编号，使日志、DP 平分和重复运行完全可复现。
        ordered_blocks = sorted(
            blocks,
            key=lambda block: (
                category_order[normalize_category_name(block["category"])],
                float(block["px"]),
                float(block["py"]),
                float(block["theta"]),
            ),
        )
        for source_id, block in enumerate(ordered_blocks):
            category = block["category"]
            servo_pose = self.high_tcp_localizer.locate_block(
                (block["px"], block["py"])
            )
            pick_surface_z_mm, depth_sample_pixel_xy = self.resolve_pick_surface_height(
                block["px"],
                block["py"],
                servo_pose[2],
            )
            diagnostic = self.make_high_localization_diagnostic(
                block["px"],
                block["py"],
                image_shape,
                depth_sample_pixel_xy,
            )
            observed_blocks.append(
                ObservedBlock(
                    category=category,
                    observation_pose=tuple(servo_pose),
                    detected_angle_deg=float(block["theta"]),
                    source_id=source_id,
                    pick_surface_z_mm=pick_surface_z_mm,
                    pick_surface_z_valid=True,
                    **diagnostic,
                )
            )
        return observed_blocks

    def _detect_blocks_for_task(self, image):
        """兼容旧调用：完成整套识别后生成正式方块观察目标。"""
        blocks, count_list = self._detect_blocks_raw(image)
        return self._build_observed_blocks_for_task(blocks, image.shape), count_list

    def _load_layout_for_request(self, request, cube_counts):
        if not request.advanced:
            return load_task_layout(self.task_layout_path), "基础任务布局"
        if len(request.place_order) != 7:
            raise ValueError("进阶任务顺序必须正好包含 7 个整数")
        layout, fill_line = AdvancedPlanner(self.advanced_library_path).build_layout(
            cube_counts, request.place_order
        )
        return layout, f"进阶任务布局，极限填满 {fill_line} 行"

    @staticmethod
    def _layout_to_board_dicts(layout):
        """把布局（含 cells）序列化为统一盘面结构，供网页只读展示。"""
        return [
            {
                "col": float(item["col"]),
                "row": float(item["row"]),
                "angle_deg": float(item["angle_deg"]),
                "category": str(item["category"]),
                "cells": [[int(col), int(row)] for col, row in item["cells"]],
            }
            for item in layout
        ]

    def _v5_decision_to_board_dicts(self, decision):
        """从 V5 决策 + 盘面库重建盘面结构（含 cells），供网页只读展示。"""
        library = self.dynamic_board_library
        targets = []
        for pid in decision.placement_pids:
            pid = int(pid)
            category = str(library.category_names[int(library.placement_category[pid])])
            targets.append({
                "col": float(library.placement_col[pid]),
                "row": float(library.placement_row[pid]),
                "angle_deg": float(library.placement_yaw_clockwise_deg[pid]),
                "category": category,
                "cells": [[int(col), int(row)] for col, row in library.placement_cells[pid]],
            })
        return targets

    def _build_placement_targets(self, layout, image_shape):
        """把布局格点转换成摆放粗观察位。"""
        targets = []
        for item in layout:
            target_point = interpolate_grid_point(
                self.board_grid_points,
                float(item["row"]),
                float(item["col"]),
            )
            servo_pose = self.high_tcp_localizer.locate_tray(
                (target_point[0], target_point[1])
            )
            diagnostic = self.make_high_localization_diagnostic(
                target_point[0],
                target_point[1],
                image_shape,
            )
            targets.append(
                PlacementTarget(
                    index=int(item["index"]),
                    row=float(item["row"]),
                    col=float(item["col"]),
                    desired_angle_deg=float(item["angle_deg"]),
                    category=normalize_category_name(item["category"]),
                    observation_pose=tuple(servo_pose),
                    cells=tuple(tuple(cell) for cell in item["cells"]),
                    **diagnostic,
                )
            )
        return targets

    @staticmethod
    def _dynamic_center_key(row, col):
        """V5 行列均是整数或 .5，可直接作为稳定缓存键。"""
        return float(row), float(col)

    def _build_dynamic_target_center_cache(self, placement_ids, image_shape):
        """按行列去重执行“格点插值→托盘 TCP”，并构造 PID XY 查表。"""
        library = self.dynamic_board_library
        if library is None:
            raise RuntimeError("V5 动态盘面库未加载")
        placement_xy = np.full((library.placement_count, 2), np.nan, dtype=np.float64)
        cache_by_center = {}
        records_by_center = {}
        for pid_value in sorted(set(int(value) for value in placement_ids)):
            pid = int(pid_value)
            if pid < 0 or pid >= library.placement_count:
                raise ValueError(f"动态盘面 PID 越界：{pid}")
            row = float(library.placement_row[pid])
            col = float(library.placement_col[pid])
            key = self._dynamic_center_key(row, col)
            cached = cache_by_center.get(key)
            if cached is None:
                target_point = interpolate_grid_point(
                    self.board_grid_points,
                    row,
                    col,
                )
                pixel_xy = (float(target_point[0]), float(target_point[1]))
                servo_pose = tuple(
                    float(value)
                    for value in self.high_tcp_localizer.locate_tray(pixel_xy)
                )
                diagnostic = self.make_high_localization_diagnostic(
                    pixel_xy[0],
                    pixel_xy[1],
                    image_shape,
                )
                cached = {
                    "row": row,
                    "col": col,
                    "pixel_xy": pixel_xy,
                    "observation_pose": servo_pose,
                    "diagnostic": diagnostic,
                }
                cache_by_center[key] = cached
                records_by_center[key] = {
                    "行": row,
                    "列": col,
                    "目标中心像素XY": list(pixel_xy),
                    "TCP观察位姿": list(servo_pose),
                    "高位定位诊断": diagnostic,
                    "placement_PID": [],
                }
            records_by_center[key]["placement_PID"].append(pid)
            placement_xy[pid] = cached["observation_pose"][:2]
        records = tuple(
            records_by_center[key]
            for key in sorted(records_by_center)
        )
        return placement_xy, cache_by_center, records

    def _build_dynamic_placement_targets(self, placement_ids, cache_by_center):
        """只为 relaxed 前 top-k 盘面构造完整 PlacementTarget。"""
        library = self.dynamic_board_library
        targets = {}
        for pid_value in sorted(set(int(value) for value in placement_ids)):
            pid = int(pid_value)
            row = float(library.placement_row[pid])
            col = float(library.placement_col[pid])
            key = self._dynamic_center_key(row, col)
            if key not in cache_by_center:
                raise RuntimeError(f"PID {pid} 缺少目标中心 TCP 缓存")
            cached = cache_by_center[key]
            targets[pid] = PlacementTarget(
                index=pid,
                row=row,
                col=col,
                desired_angle_deg=float(
                    library.placement_yaw_clockwise_deg[pid]
                ),
                category=library.category_names[
                    int(library.placement_category[pid])
                ],
                observation_pose=cached["observation_pose"],
                cells=tuple(
                    tuple(int(value) for value in cell)
                    for cell in library.placement_cells[pid]
                ),
                **cached["diagnostic"],
            )
        return targets

    def _run_dynamic_board_selection(
        self,
        observed_blocks,
        image_shape,
        allow_motion_speed_mismatch=False,
    ):
        """完整执行粗筛、relaxed、低 Beam 比较和前 K 名高 Beam 复核。"""
        if (
            self.dynamic_board_candidate_selector is None
            or self.dynamic_final_board_selector is None
        ):
            raise RuntimeError("动态盘面选择器未初始化")
        state = {
            "tray_center_pixel_xy": None,
            "coarse_result": None,
            "relaxed_result": None,
            "target_center_cache": (),
            "decision": None,
            "motion_speed_mismatch_override": bool(
                allow_motion_speed_mismatch
            ),
        }
        self._current_dynamic_board_run = state
        tray_center = interpolate_grid_point(
            self.board_grid_points,
            row=7.5,
            col=5.5,
        )
        tray_center = (float(tray_center[0]), float(tray_center[1]))
        state["tray_center_pixel_xy"] = tray_center
        coarse_result = self.dynamic_board_candidate_selector.select_coarse(
            observed_blocks,
            tray_center,
        )
        state["coarse_result"] = coarse_result
        required_pids = self.dynamic_board_candidate_selector.required_placement_ids(
            coarse_result
        )
        placement_xy, center_cache, center_records = (
            self._build_dynamic_target_center_cache(required_pids, image_shape)
        )
        state["target_center_cache"] = center_records
        relaxed_result = self.dynamic_board_candidate_selector.select_relaxed(
            coarse_result,
            observed_blocks,
            placement_xy,
            self.board_theta,
        )
        state["relaxed_result"] = relaxed_result
        final_pids = {
            int(pid)
            for candidate in relaxed_result.candidates
            for pid in self.dynamic_board_library.board_target_pid[
                candidate.board_index
            ].flat
            if int(pid) >= 0
        }
        placement_targets_by_pid = self._build_dynamic_placement_targets(
            final_pids,
            center_cache,
        )
        motion_model = get_default_arm_motion_time_model()
        if allow_motion_speed_mismatch:
            rospy.logwarn(
                "已人工确认忽略路程时间标定速度不一致："
                "标定=%g，arm_speed=%g，pick_approach_speed=%g。"
                "实际执行速度不会被修改，但盘面预测时间不可视为真实秒数。",
                float(motion_model.move_speed_percent),
                self.arm_speed,
                self.pick_approach_speed,
            )
        else:
            validate_motion_model_speeds(
                motion_model,
                self.arm_speed,
                self.pick_approach_speed,
            )
        self.dynamic_motion_model_source_path = getattr(
            motion_model,
            "source_path",
            None,
        )
        decision = self.dynamic_final_board_selector.select(
            relaxed_result,
            observed_blocks,
            placement_targets_by_pid,
            self.board_theta,
            self._make_task_sequence_optimizer_config(),
            motion_model,
        )
        state["decision"] = decision
        self.last_dynamic_board_decision = decision
        for attempt in decision.comparison_attempts:
            if attempt.succeeded:
                rospy.loginfo(
                    "V5比较 %s：简化=%.6f s，舵机重放=%.6f s，耗时=%.3f s",
                    attempt.board_id,
                    attempt.simplified_cost_seconds,
                    attempt.servo_replay_total_seconds,
                    attempt.elapsed_seconds,
                )
            else:
                rospy.logwarn(
                    "V5比较 %s 失败：%s",
                    attempt.board_id,
                    attempt.error_message,
                )
        rospy.loginfo(
            "V5低Beam比较完成：%d 张，实际线程=%d，墙钟耗时=%.3f s",
            len(decision.comparison_attempts),
            decision.comparison_worker_count,
            decision.comparison_elapsed_seconds,
        )
        for attempt in decision.confirmation_attempts:
            if attempt.succeeded:
                rospy.loginfo(
                    "V5高Beam复核 %s：简化=%.6f s，舵机重放=%.6f s，"
                    "耗时=%.3f s",
                    attempt.board_id,
                    attempt.simplified_cost_seconds,
                    attempt.servo_replay_total_seconds,
                    attempt.elapsed_seconds,
                )
            else:
                rospy.logwarn(
                    "V5高Beam复核 %s 失败：%s",
                    attempt.board_id,
                    attempt.error_message,
                )
        rospy.loginfo(
            "V5高Beam复核完成：%d 张，实际线程=%d，墙钟耗时=%.3f s",
            len(decision.confirmation_attempts),
            decision.confirmation_worker_count,
            decision.confirmation_elapsed_seconds,
        )
        rospy.loginfo(
            "V5唯一盘面=%s，指纹=%s，总耗时=%.3f s",
            decision.board_id,
            decision.decision_fingerprint,
            decision.total_elapsed_seconds,
        )
        if decision.soft_time_budget_exceeded:
            rospy.logwarn(
                "V5动态盘面选择耗时 %.3f s，超过软预算 %.3f s；"
                "结果不受影响",
                decision.total_elapsed_seconds,
                self.dynamic_board_soft_time_budget_sec,
            )
        return state

    def _save_dynamic_board_report(
        self,
        observed_blocks,
        image_shape,
        outcome,
        error_message="",
        human_failure_choice="",
        actual_planner_message="",
        comparison_attempts=(),
        confirmation_attempts=(),
    ):
        """尽力原子写入动态报告；报告失败不改变已选任务。"""
        report_dir = self.task_plan_report_dir
        if not report_dir:
            rospy.logwarn("未配置实验批次目录，跳过动态盘面报告写入")
            return
        state = getattr(self, "_current_dynamic_board_run", {}) or {}
        try:
            tray_center = state.get("tray_center_pixel_xy")
            if tray_center is None:
                point = interpolate_grid_point(
                    self.board_grid_points,
                    row=7.5,
                    col=5.5,
                )
                tray_center = (float(point[0]), float(point[1]))
            hashes = {
                "方块像素到TCP": sha256_file(self.block_calibration_path),
                "托盘像素到TCP": sha256_file(self.tray_calibration_path),
                "机械臂运动时间": sha256_file(
                    getattr(self, "dynamic_motion_model_source_path", None)
                ),
            }
            optimizer_config = self._make_task_sequence_optimizer_config()
            runtime_config = dict(self.dynamic_board_runtime_config)
            runtime_config["task_sequence_optimizer"] = {
                "shooting_pose": list(optimizer_config.shooting_pose),
                "camera_to_sucker_offset_mm": list(
                    optimizer_config.camera_to_sucker_offset_mm
                ),
                "pick_surface_offset_mm": optimizer_config.pick_surface_offset_mm,
                "pick_approach_clearance_mm": (
                    optimizer_config.pick_approach_clearance_mm
                ),
                "motor_velocity_deg_per_sec": (
                    optimizer_config.motor_velocity_deg_per_sec
                ),
                "initial_motor_angle_deg": optimizer_config.initial_motor_angle_deg,
                "motor_lower_margin_deg": optimizer_config.motor_lower_margin_deg,
                "motor_upper_margin_deg": optimizer_config.motor_upper_margin_deg,
            }
            runtime_config["motion_model_path"] = str(
                getattr(self, "dynamic_motion_model_source_path", "") or ""
            )
            runtime_config["arm_speed"] = self.arm_speed
            runtime_config["pick_approach_speed"] = self.pick_approach_speed
            runtime_config["motion_speed_mismatch_override"] = bool(
                state.get("motion_speed_mismatch_override", False)
            )
            decision = state.get("decision")
            document = build_dynamic_board_selection_report(
                mode=self.dynamic_board_selection_mode,
                outcome=outcome,
                observed_blocks=observed_blocks,
                board_grid_points=self.board_grid_points,
                tray_center_pixel_xy=tray_center,
                board_angle_deg=self.board_theta,
                image_shape=image_shape,
                target_center_cache=state.get("target_center_cache", ()),
                library=self.dynamic_board_library,
                runtime_config=runtime_config,
                calibration_sha256=hashes,
                coarse_result=state.get("coarse_result"),
                relaxed_result=state.get("relaxed_result"),
                decision=decision,
                comparison_attempts=comparison_attempts,
                confirmation_attempts=confirmation_attempts,
                error_message=error_message,
                human_failure_choice=human_failure_choice,
                actual_planner_message=actual_planner_message,
            )
            report_path = os.path.join(report_dir, "动态盘面选择报告.json")
            atomic_write_json(report_path, document)
            self.last_dynamic_board_report_path = report_path
            rospy.loginfo("动态盘面选择报告已写入: %s", report_path)
        except Exception as exc:
            rospy.logwarn("动态盘面选择报告写入失败，不影响本轮任务: %s", exc)

    def _make_task_sequence_optimizer_config(self):
        """把已校验的执行配置投影为纯规划器配置。"""
        return TaskSequenceOptimizerConfig(
            shooting_pose=tuple(self.shooting_angle),
            camera_to_sucker_offset_mm=self.camera_to_sucker_offset_mm,
            pick_surface_offset_mm=self.pick_surface_offset_mm,
            pick_approach_clearance_mm=self.pick_approach_clearance_mm,
            motor_velocity_deg_per_sec=self.motor_velocity_deg_per_sec,
            initial_motor_angle_deg=self.initial_motor_angle_deg,
            motor_lower_margin_deg=self.motor_lower_margin_deg,
            motor_upper_margin_deg=self.motor_upper_margin_deg,
            beam_width=self.task_sequence_optimizer_beam_width,
            report_top_candidates=self.task_sequence_optimizer_report_top_candidates,
        )

    def _save_task_plan_report(self, result):
        """尽力写入本轮中文 JSON；失败不得影响 shadow 或 execute 规划结果。"""
        report_dir = self.task_plan_report_dir
        if not report_dir:
            rospy.logwarn("未配置实验批次目录，跳过任务规划报告写入")
            return
        report_path = os.path.join(report_dir, "任务规划报告.json")
        try:
            os.makedirs(report_dir, exist_ok=True)
            document = build_task_plan_report(result)
            with open(report_path, "w", encoding="utf-8") as report_file:
                json.dump(document, report_file, ensure_ascii=False, indent=2)
                report_file.write("\n")
            rospy.loginfo("任务规划报告已写入: %s", report_path)
        except Exception as exc:
            rospy.logwarn("任务规划报告写入失败，不影响本轮任务: %s", exc)

    def _plan_required_tasks(self, observed_blocks, placement_targets):
        """按 legacy/shadow/execute 规则生成本轮必选任务列表。"""
        legacy_tasks = assign_blocks_to_targets(
            observed_blocks,
            placement_targets,
            board_angle_deg=self.board_theta,
        )
        mode = self.task_sequence_optimizer_mode

        def run_new_optimizer():
            motion_model = get_default_arm_motion_time_model()
            validate_motion_model_speeds(
                motion_model,
                self.arm_speed,
                self.pick_approach_speed,
            )
            return optimize_task_sequence(
                observed_blocks,
                placement_targets,
                board_angle_deg=self.board_theta,
                config=self._make_task_sequence_optimizer_config(),
                motion_model=motion_model,
                legacy_tasks=legacy_tasks,
            )

        decision = decide_optimizer_mode(
            mode,
            legacy_tasks,
            self.visual_servo_enabled,
            run_new_optimizer,
        )
        self.last_task_plan_result = decision.plan_result
        if decision.status == "legacy":
            return list(decision.tasks), "旧匈牙利规划"
        if decision.status == "shadow_failed":
            rospy.logwarn(
                "V1 shadow 新规划失败，本轮继续执行旧方案: %s",
                decision.error_message,
            )
            return list(decision.tasks), "V1 shadow 失败回退旧匈牙利方案"

        result = decision.plan_result
        self._save_task_plan_report(result)
        if decision.status == "shadow":
            rospy.loginfo(
                "V1 shadow：旧方案 %.3f s，固定顺序 DP %.3f s，"
                "Beam %.3f s；本轮仍返回旧方案",
                result.legacy_plan.simplified_cost_seconds,
                result.fixed_order_plan.simplified_cost_seconds,
                result.beam_best_plan.simplified_cost_seconds,
            )
            return list(decision.tasks), "V1 shadow（执行旧匈牙利方案）"
        return list(decision.tasks), "V1 execute 优化规划"

    def _plan_advanced_tasks(self, observed_blocks, placement_targets):
        """只匹配同类实体，并严格保留进阶任务动态库给出的目标顺序。"""
        expected_indices = tuple(range(len(placement_targets)))
        target_indices = tuple(int(target.index) for target in placement_targets)
        if target_indices != expected_indices:
            raise RuntimeError(
                "进阶任务目标序号没有严格按动态库输出顺序连续编号"
            )

        tasks = assign_blocks_to_targets(
            observed_blocks,
            placement_targets,
            board_angle_deg=self.board_theta,
        )
        task_indices = tuple(int(task.index) for task in tasks)
        expected_categories = tuple(
            normalize_category_name(target.category)
            for target in placement_targets
        )
        actual_categories = tuple(
            normalize_category_name(task.category)
            for task in tasks
        )
        if task_indices != expected_indices or actual_categories != expected_categories:
            raise RuntimeError("进阶任务规划结果改变了裁判指定的摆放顺序")

        self.last_task_plan_result = None
        return list(tasks), "进阶任务固定顺序，旧匈牙利实体分配"

    def _placement_specs(self, layout):
        """先计算全部托盘目标像素，供一次批量深度查询使用。"""
        specs = []
        for item in layout:
            target_point = interpolate_grid_point(
                self.board_grid_points,
                float(item["row"]),
                float(item["col"]),
            )
            specs.append((item, (float(target_point[0]), float(target_point[1]))))
        return specs

    @staticmethod
    def _format_tcp_xyz(tcp_xyz):
        """使用现场易读的固定三位小数格式显示 TCP XYZ。"""
        return "[" + ", ".join(f"{float(value):.3f}" for value in tcp_xyz) + "]"

    @staticmethod
    def _format_pixel_xy(pixel_xy):
        """使用固定三位小数显示检测或编辑后的高位像素中心。"""
        return "[" + ", ".join(f"{float(value):.3f}" for value in pixel_xy) + "]"

    @staticmethod
    def _format_high_tcp_axis_violations(assessment):
        """逐轴显示越界值和本次评估实际使用的安全限位。"""
        axis_indices = {"X": 0, "Y": 1, "Z": 2}
        details = []
        for axis_name in assessment.violated_axes:
            axis_index = axis_indices[axis_name]
            value = float(assessment.safety_tcp_xyz[axis_index])
            minimum = float(assessment.safety_min_xyz[axis_index])
            maximum = float(assessment.safety_max_xyz[axis_index])
            maximum_text = f"{maximum:.3f}" if np.isfinite(maximum) else "+∞"
            details.append(
                f"{axis_name}={value:.3f} mm 越界"
                f"（当前限位 [{minimum:.3f}, {maximum_text}] mm）"
            )
        return "、".join(details)

    def _collect_high_tcp_safety_violations(self, blocks, layout):
        """检查完整正式任务的方块和托盘目标，并返回全部问题。"""
        candidates = []
        for index, block in enumerate(blocks, start=1):
            candidates.append((
                f"方块 {index}（{block['category']}）",
                "block",
                (float(block["px"]), float(block["py"])),
            ))
        for fallback_index, (item, point) in enumerate(
            self._placement_specs(layout),
            start=1,
        ):
            target_index = int(item.get("index", fallback_index - 1)) + 1
            candidates.append((f"托盘目标 {target_index}", "tray", point))

        violations = []
        for label, subject, pixel_xy in candidates:
            pixel_text = self._format_pixel_xy(pixel_xy)
            try:
                assessment = self.high_tcp_localizer.assess(subject, pixel_xy)
            except Exception as exc:
                violations.append(
                    f"{label}：像素坐标 {pixel_text} px，TCP 预测失败：{exc}"
                )
                continue
            if assessment.safe:
                continue
            tcp_text = self._format_tcp_xyz(assessment.safety_tcp_xyz)
            axis_text = self._format_high_tcp_axis_violations(assessment)
            violations.append(
                f"{label}：像素坐标 {pixel_text} px，"
                f"预测实际 TCP {tcp_text} mm，{axis_text}"
            )
        return violations

    def _validate_high_task_safety(self, blocks, layout, stage):
        """全量检查正式高位任务，失败时抛出带完整现场摘要的可恢复错误。"""
        violations = self._collect_high_tcp_safety_violations(blocks, layout)
        if violations:
            raise HighTcpSafetyCheckError(stage, violations)

    def _build_calibration_block_observed(self, blocks, samples, image_shape):
        """把方块深度样本转成标定 ObservedBlock，并校验 MAD、位姿和抓取高度。"""
        observed_blocks = []
        for block, sample in zip(blocks, samples):
            high_pixel = (float(block["px"]), float(block["py"]))
            block_label = f"方块 {block['category']}（block）高位像素 {high_pixel!r}"
            if sample["depth_mad_mm"] > self.block_depth_max_mad_mm:
                raise RuntimeError(
                    f"方块 {block['category']} 深度 MAD={sample['depth_mad_mm']:.3f} mm "
                    f"超过阈值 {self.block_depth_max_mad_mm:.3f} mm"
                )
            pose = self.depth_rough_localizer.block_observation_pose(
                sample["world"],
                self.block_observation_height_mm,
            )
            pose = self._validate_calibration_pose(pose, f"{block_label} 深度粗定位")
            pick_tcp_z = float(sample["world"][2]) + self.pick_surface_offset_mm
            if not np.isfinite(pick_tcp_z) or pick_tcp_z < self.minimum_tcp_z_mm:
                raise ValueError(
                    f"{block_label} 最终抓取 TCP Z={pick_tcp_z:.3f} mm "
                    f"低于安全下限 {self.minimum_tcp_z_mm:.3f} mm"
                )
            diagnostic = self.make_high_localization_diagnostic(
                block["px"],
                block["py"],
                image_shape,
                depth_sample_pixel_xy=sample["pixel"],
                world_position=sample["world"],
                world_position_valid=True,
                source="stable_depth_xyz",
                depth_valid_frame_count=sample["valid_count"],
                depth_median_mm=sample["depth_median_mm"],
                depth_mad_mm=sample["depth_mad_mm"],
                calibration_target_tcp_z_mm=pose[2],
            )
            observed_blocks.append(
                ObservedBlock(
                    category=block["category"],
                    observation_pose=tuple(pose),
                    detected_angle_deg=block["theta"],
                    pick_surface_z_mm=float(sample["world"][2]),
                    pick_surface_z_valid=True,
                    **diagnostic,
                )
            )
        return observed_blocks

    def _fit_calibration_block_plane(self, observed_blocks):
        """由方块观察 TCP 点拟合 Z 平面，拒绝少点、共线和超限平面。"""
        block_observation_points = [
            np.asarray(block.observation_pose[:3], dtype=float)
            for block in observed_blocks
        ]
        if len(block_observation_points) < 3:
            raise RuntimeError(
                "托盘标定至少需要 3 个不共线方块提供观察 Z 平面，请增加方块数量"
            )
        try:
            block_plane = fit_z_plane(block_observation_points)
        except ValueError as exc:
            raise RuntimeError(
                f"方块观察点无法拟合 Z 平面（{exc}），请重新摆放方块使其不共线"
            ) from exc
        if block_plane.rmse_mm > self.block_plane_max_rmse_mm:
            raise RuntimeError(
                f"方块观察 TCP 平面 RMSE={block_plane.rmse_mm:.3f} mm "
                f"超过阈值 {self.block_plane_max_rmse_mm:.3f} mm"
            )
        return block_plane

    def _build_calibration_tray_targets(self, specs, samples, block_plane, image_shape):
        """把托盘格点规格和深度样本转成标定 PlacementTarget。"""
        placement_targets = []
        for (item, point), sample in zip(specs, samples):
            tray_tcp_xy = self.depth_rough_localizer.tcp_xy_from_world(sample["world"])
            tray_tcp_z = (
                block_plane.predict(tray_tcp_xy[0], tray_tcp_xy[1])
                - self.tray_tcp_below_block_observation_mm
            )
            pose = self.depth_rough_localizer.tray_observation_pose(
                sample["world"],
                tray_tcp_z,
            )
            pose = self._validate_calibration_pose(
                pose,
                f"托盘目标 {item['index']}（tray）高位像素 {point!r} 深度粗定位",
            )
            diagnostic = self.make_high_localization_diagnostic(
                point[0],
                point[1],
                image_shape,
                depth_sample_pixel_xy=sample["pixel"],
                world_position=sample["world"],
                world_position_valid=True,
                source="stable_depth_xy_block_plane_z",
                depth_valid_frame_count=sample["valid_count"],
                depth_median_mm=sample["depth_median_mm"],
                depth_mad_mm=sample["depth_mad_mm"],
                calibration_target_tcp_z_mm=tray_tcp_z,
            )
            placement_targets.append(
                PlacementTarget(
                    index=int(item["index"]),
                    row=float(item["row"]),
                    col=float(item["col"]),
                    desired_angle_deg=float(item["angle_deg"]),
                    category=normalize_category_name(item["category"]),
                    observation_pose=tuple(pose),
                    **diagnostic,
                )
            )
        return placement_targets

    def _build_calibration_targets(self, blocks, layout, image_shape):
        """标定模式用深度 XYZ 生成方块和托盘粗位姿及完整诊断。"""
        placement_specs = self._placement_specs(layout)
        pixels = [(block["px"], block["py"]) for block in blocks]
        pixels.extend(point for _item, point in placement_specs)
        samples = self._query_stable_world_points(pixels)
        observed_blocks = self._build_calibration_block_observed(
            blocks,
            samples[:len(blocks)],
            image_shape,
        )
        block_plane = self._fit_calibration_block_plane(observed_blocks)
        placement_targets = self._build_calibration_tray_targets(
            placement_specs,
            samples[len(blocks):],
            block_plane,
            image_shape,
        )
        return observed_blocks, placement_targets

    def _prepare_calibration_targets(self, image):
        """标定模式准备：全部方块目标在前，可选 34 个托盘随机目标在后。"""
        blocks, _counts = self._detect_blocks_raw(image)
        tray_found = False
        try:
            self._detect_board_for_task(image)
            tray_found = True
        except Exception as exc:
            self.board_grid_points = None
            self.board_grid_image = None
            self.board_grid_image_shape = None
            rospy.logwarn("未识别到托盘，本轮标定只采集方块: %s", exc)
        block_pixels = [(block["px"], block["py"]) for block in blocks]
        tray_items = []
        tray_pixels = []
        if tray_found:
            for index, spec in enumerate(select_calibration_tray_points()):
                tray_items.append({
                    "index": index,
                    "row": spec["row"],
                    "col": spec["col"],
                    "angle_deg": 0.0,
                    "category": "",
                })
            tray_pixels = [
                interpolate_grid_point(self.board_grid_points, item["row"], item["col"])
                for item in tray_items
            ]
        samples = self._query_stable_world_points([*block_pixels, *tray_pixels])
        observed_blocks = self._build_calibration_block_observed(
            blocks,
            samples[:len(blocks)],
            image.shape[:2],
        )
        block_targets = [
            TaskTarget(
                index=index,
                category=block.category,
                row=0.0,
                col=0.0,
                pick_observation_pose=tuple(float(value) for value in block.observation_pose),
                place_observation_pose=(0.0,) * 6,
                detected_angle_deg=float(block.detected_angle_deg),
                rotation_delta_deg=0.0,
                pick_surface_z_mm=float(block.pick_surface_z_mm),
                pick_surface_z_valid=bool(block.pick_surface_z_valid),
                target_type="block",
                pick_high_detected_pixel_xy=tuple(
                    float(value) for value in block.high_detected_pixel_xy
                ),
                pick_high_depth_sample_pixel_xy=tuple(
                    float(value) for value in block.high_depth_sample_pixel_xy
                ),
                pick_high_image_center_xy=tuple(
                    float(value) for value in block.high_image_center_xy
                ),
                pick_high_world_position=tuple(
                    float(value) for value in block.high_world_position
                ),
                pick_high_world_position_valid=bool(block.high_world_position_valid),
                pick_rough_localization_source=str(block.rough_localization_source),
                pick_depth_valid_frame_count=int(block.depth_valid_frame_count),
                pick_depth_median_mm=float(block.depth_median_mm),
                pick_depth_mad_mm=float(block.depth_mad_mm),
                pick_calibration_target_tcp_z_mm=float(
                    block.calibration_target_tcp_z_mm
                ),
            )
            for index, block in enumerate(observed_blocks)
        ]
        tray_targets = []
        if tray_found:
            block_plane = self._fit_calibration_block_plane(observed_blocks)
            placement_targets = self._build_calibration_tray_targets(
                list(zip(tray_items, tray_pixels)),
                samples[len(blocks):],
                block_plane,
                image.shape[:2],
            )
            tray_targets = [
                TaskTarget(
                    index=len(block_targets) + index,
                    category=str(target.category),
                    row=float(target.row),
                    col=float(target.col),
                    pick_observation_pose=(0.0,) * 6,
                    place_observation_pose=tuple(
                        float(value) for value in target.observation_pose
                    ),
                    detected_angle_deg=0.0,
                    rotation_delta_deg=0.0,
                    pick_surface_z_mm=0.0,
                    pick_surface_z_valid=False,
                    target_type="tray",
                    place_high_detected_pixel_xy=tuple(
                        float(value) for value in target.high_detected_pixel_xy
                    ),
                    place_high_depth_sample_pixel_xy=tuple(
                        float(value) for value in target.high_depth_sample_pixel_xy
                    ),
                    place_high_image_center_xy=tuple(
                        float(value) for value in target.high_image_center_xy
                    ),
                    place_high_world_position=tuple(
                        float(value) for value in target.high_world_position
                    ),
                    place_high_world_position_valid=bool(
                        target.high_world_position_valid
                    ),
                    place_rough_localization_source=str(
                        target.rough_localization_source
                    ),
                    place_depth_valid_frame_count=int(target.depth_valid_frame_count),
                    place_depth_median_mm=float(target.depth_median_mm),
                    place_depth_mad_mm=float(target.depth_mad_mm),
                    place_calibration_target_tcp_z_mm=float(
                        target.calibration_target_tcp_z_mm
                    ),
                )
                for index, target in enumerate(placement_targets)
            ]
        return block_targets, tray_targets

    def prepare_task(self, request):
        """串行执行高位准备，禁止并发弹出两个人工编辑窗口。"""
        prepare_lock = getattr(self, "prepare_task_lock", None)
        acquired = prepare_lock is None or prepare_lock.acquire(blocking=False)
        if not acquired:
            return PrepareTaskResponse(
                success=False,
                task_count=0,
                block_count=0,
                tray_count=0,
                message="已有一轮高位识别或人工编辑正在进行，请勿并发请求",
                layout_json="",
            )
        try:
            return self._prepare_task_locked(request)
        finally:
            if prepare_lock is not None:
                prepare_lock.release()

    def _prepare_task_locked(self, request):
        """用同一高位图像快照完成托盘、方块识别与任务规划。"""
        self.task_targets = []
        self.last_board_layout = []
        self.board_grid_points = None
        self.board_grid_image_shape = None
        self.board_grid_image = None
        self.last_task_plan_result = None
        self.last_dynamic_board_decision = None
        self.last_dynamic_board_report_path = None
        self._current_dynamic_board_run = {}
        request_stamp = rospy.Time.now()
        image = self.get_image_snapshot_newer_than(request_stamp)
        if image is None:
            return PrepareTaskResponse(
                success=False,
                task_count=0,
                block_count=0,
                tray_count=0,
                message=f"等待高位新图像超时（{self.fresh_image_timeout_sec:.1f} 秒）",
                layout_json="",
            )
        block_count = 0
        tray_count = 0
        try:
            if self.calibration_mode:
                block_targets, tray_targets = self._prepare_calibration_targets(image)
                self.task_targets = [*block_targets, *tray_targets]
                block_count = len(block_targets)
                tray_count = len(tray_targets)
                if tray_targets:
                    tray_coords = "，".join(
                        f"({target.row:g},{target.col:g})" for target in tray_targets
                    )
                    message = (
                        f"标定准备完成：方块 {block_count} 个，"
                        f"托盘 {tray_count} 个（行列：{tray_coords}）"
                    )
                else:
                    message = f"标定准备完成：方块 {block_count} 个，未识别到托盘，只采集方块"
            else:
                self._detect_board_for_task(image)
                raw_blocks, raw_debug_image, geometry = self._detect_blocks_automatic(image)
                preliminary_blocks, cube_counts = self._summarize_detected_blocks(raw_blocks)
                layout, layout_message = self._load_layout_for_request(request, cube_counts)
                self.last_board_layout = self._layout_to_board_dicts(layout)
                dynamic_execute_requested = (
                    not request.advanced
                    and getattr(
                        self,
                        "dynamic_board_selection_mode",
                        "disabled",
                    ) == "execute"
                )
                # execute 的目标由筛选后的 PID 决定；此时只预检 source。
                # 若后续人工回退，会对固定 YAML 重做完整安全检查。
                initial_safety_layout = [] if dynamic_execute_requested else layout
                self._validate_high_task_safety(
                    preliminary_blocks,
                    initial_safety_layout,
                    "初步",
                )

                final_blocks, final_debug_image = self._edit_and_rematch_blocks(
                    image,
                    raw_blocks,
                    geometry,
                    raw_debug_image,
                )
                self._save_high_block_debug_images(final_blocks, final_debug_image)
                final_blocks, _final_counts = self._summarize_detected_blocks(final_blocks)
                self._validate_high_task_safety(
                    final_blocks,
                    initial_safety_layout,
                    "最终",
                )

                observed_blocks = self._build_observed_blocks_for_task(
                    final_blocks,
                    image.shape,
                )
                dynamic_mode = (
                    getattr(self, "dynamic_board_selection_mode", "disabled")
                    if not request.advanced
                    else "disabled"
                )
                dynamic_state = None
                dynamic_error = None
                if dynamic_mode != "disabled":
                    try:
                        dynamic_state = self._run_dynamic_board_selection(
                            observed_blocks,
                            image.shape[:2],
                        )
                    except Exception as exc:
                        dynamic_error = exc
                        rospy.logwarn("V5 动态盘面选择失败: %s", exc)

                if dynamic_mode == "execute" and dynamic_state is not None:
                    decision = dynamic_state["decision"]
                    self.task_targets = list(decision.tasks)
                    self.last_board_layout = self._v5_decision_to_board_dicts(decision)
                    layout_message = f"V5 动态盘面 {decision.board_id}"
                    planner_message = (
                        "动态 execute 确认路径，指纹="
                        f"{decision.decision_fingerprint}"
                    )
                    self._save_dynamic_board_report(
                        observed_blocks,
                        image.shape[:2],
                        outcome="execute成功，执行动态盘面",
                        actual_planner_message=planner_message,
                    )
                elif dynamic_mode == "execute":
                    comparison_attempts = getattr(
                        dynamic_error,
                        "comparison_attempts",
                        getattr(dynamic_error, "attempts", ()),
                    )
                    confirmation_attempts = getattr(
                        dynamic_error,
                        "confirmation_attempts",
                        (),
                    )
                    failure_choice = self.choose_dynamic_selection_failure(
                        str(dynamic_error),
                        allow_continue=isinstance(
                            dynamic_error,
                            MotionModelSpeedMismatchError,
                        ),
                    )
                    if failure_choice == "continue_dynamic":
                        try:
                            # 只有速度不一致会展示此选项；重跑时仅跳过该校验。
                            dynamic_state = self._run_dynamic_board_selection(
                                observed_blocks,
                                image.shape[:2],
                                allow_motion_speed_mismatch=True,
                            )
                        except Exception as continue_exc:
                            continue_comparison_attempts = getattr(
                                continue_exc,
                                "comparison_attempts",
                                getattr(continue_exc, "attempts", ()),
                            )
                            continue_confirmation_attempts = getattr(
                                continue_exc,
                                "confirmation_attempts",
                                (),
                            )
                            self._save_dynamic_board_report(
                                observed_blocks,
                                image.shape[:2],
                                outcome="速度不一致人工继续，但动态重试失败",
                                error_message=(
                                    f"原始失败：{dynamic_error}；"
                                    f"忽略速度校验后失败：{continue_exc}"
                                ),
                                human_failure_choice="continue_dynamic",
                                comparison_attempts=(
                                    continue_comparison_attempts
                                ),
                                confirmation_attempts=(
                                    continue_confirmation_attempts
                                ),
                            )
                            raise DynamicBoardSelectionError(
                                "V5 动态盘面忽略速度校验后仍失败："
                                f"{continue_exc}",
                                continue_comparison_attempts,
                                continue_confirmation_attempts,
                            ) from continue_exc
                        decision = dynamic_state["decision"]
                        self.task_targets = list(decision.tasks)
                        self.last_board_layout = self._v5_decision_to_board_dicts(decision)
                        layout_message = f"V5 动态盘面 {decision.board_id}"
                        planner_message = (
                            "动态 execute 已人工忽略时间标定速度不一致，"
                            f"确认路径，指纹={decision.decision_fingerprint}"
                        )
                        self._save_dynamic_board_report(
                            observed_blocks,
                            image.shape[:2],
                            outcome="速度不一致人工确认，执行动态盘面",
                            error_message=str(dynamic_error),
                            human_failure_choice="continue_dynamic",
                            actual_planner_message=planner_message,
                        )
                    elif failure_choice != "fixed_yaml":
                        self._save_dynamic_board_report(
                            observed_blocks,
                            image.shape[:2],
                            outcome="execute失败，停止本轮",
                            error_message=str(dynamic_error),
                            human_failure_choice="stop",
                            comparison_attempts=comparison_attempts,
                            confirmation_attempts=confirmation_attempts,
                        )
                        raise DynamicBoardSelectionError(
                            "V5 动态盘面选择失败，本轮已停止："
                            f"{dynamic_error}",
                            comparison_attempts,
                            confirmation_attempts,
                        )
                    else:
                        try:
                            # 人工选择回退后重做固定盘面安全检查，不沿用旧结论。
                            self._validate_high_task_safety(
                                final_blocks,
                                layout,
                                "回退",
                            )
                            placement_targets = self._build_placement_targets(
                                layout,
                                image.shape[:2],
                            )
                            self.task_targets, fixed_message = self._plan_required_tasks(
                                observed_blocks,
                                placement_targets,
                            )
                        except Exception as fallback_exc:
                            self._save_dynamic_board_report(
                                observed_blocks,
                                image.shape[:2],
                                outcome="execute失败，固定YAML回退也失败",
                                error_message=(
                                    f"动态失败：{dynamic_error}；"
                                    f"固定YAML回退失败：{fallback_exc}"
                                ),
                                human_failure_choice="fixed_yaml",
                                comparison_attempts=comparison_attempts,
                                confirmation_attempts=confirmation_attempts,
                            )
                            raise
                        planner_message = f"动态 execute 失败，人工回退；{fixed_message}"
                        self._save_dynamic_board_report(
                            observed_blocks,
                            image.shape[:2],
                            outcome="execute失败，执行固定YAML回退",
                            error_message=str(dynamic_error),
                            human_failure_choice="fixed_yaml",
                            actual_planner_message=planner_message,
                            comparison_attempts=comparison_attempts,
                            confirmation_attempts=confirmation_attempts,
                        )
                else:
                    # 进阶任务严格保留动态库顺序；必选任务才允许进入 V1 优化。
                    try:
                        placement_targets = self._build_placement_targets(
                            layout,
                            image.shape[:2],
                        )
                        if request.advanced:
                            self.task_targets, planner_message = (
                                self._plan_advanced_tasks(
                                    observed_blocks,
                                    placement_targets,
                                )
                            )
                        else:
                            self.task_targets, planner_message = (
                                self._plan_required_tasks(
                                    observed_blocks,
                                    placement_targets,
                                )
                            )
                    except Exception as fixed_exc:
                        if dynamic_mode == "shadow":
                            dynamic_text = (
                                "动态成功"
                                if dynamic_state is not None
                                else f"动态失败：{dynamic_error}"
                            )
                            self._save_dynamic_board_report(
                                observed_blocks,
                                image.shape[:2],
                                outcome="shadow完成，但固定YAML规划失败",
                                error_message=(
                                    f"{dynamic_text}；固定YAML失败：{fixed_exc}"
                                ),
                                comparison_attempts=getattr(
                                    dynamic_error,
                                    "comparison_attempts",
                                    getattr(dynamic_error, "attempts", ()),
                                ),
                                confirmation_attempts=getattr(
                                    dynamic_error,
                                    "confirmation_attempts",
                                    (),
                                ),
                            )
                        raise
                    if dynamic_mode == "shadow":
                        if dynamic_state is not None:
                            decision = dynamic_state["decision"]
                            planner_message = (
                                f"{planner_message}；V5 shadow 选出 "
                                f"{decision.board_id}（未执行）"
                            )
                            self._save_dynamic_board_report(
                                observed_blocks,
                                image.shape[:2],
                                outcome="shadow成功，执行固定YAML",
                                actual_planner_message=planner_message,
                            )
                        else:
                            planner_message = (
                                f"{planner_message}；V5 shadow 失败，已自动忽略"
                            )
                            self._save_dynamic_board_report(
                                observed_blocks,
                                image.shape[:2],
                                outcome="shadow失败，执行固定YAML",
                                error_message=str(dynamic_error),
                                actual_planner_message=planner_message,
                                comparison_attempts=getattr(
                                    dynamic_error,
                                    "comparison_attempts",
                                    getattr(dynamic_error, "attempts", ()),
                                ),
                                confirmation_attempts=getattr(
                                    dynamic_error,
                                    "confirmation_attempts",
                                    (),
                                ),
                            )
                message = (
                    f"{layout_message}准备完成，共 {len(self.task_targets)} 个任务；"
                    f"{planner_message}"
                )
            rospy.loginfo(message)
            return PrepareTaskResponse(
                success=True,
                task_count=len(self.task_targets),
                block_count=block_count,
                tray_count=tray_count,
                message=message,
                layout_json=json.dumps(self.last_board_layout),
            )
        except HighTcpSafetyCheckError as exc:
            self.task_targets = []
            rospy.logwarn(
                "高位%s安全检查发现 %d 个问题，已停止本轮任务准备",
                exc.stage,
                exc.violation_count,
            )
            return PrepareTaskResponse(
                success=False,
                task_count=0,
                block_count=0,
                tray_count=0,
                message=str(exc),
                layout_json="",
            )
        except Exception as exc:
            self.task_targets = []
            rospy.logerr("任务准备失败: %s", exc)
            return PrepareTaskResponse(
                success=False,
                task_count=0,
                block_count=0,
                tray_count=0,
                message=str(exc),
                layout_json="",
            )

    def get_task_target(self, request):
        """按执行顺序返回一个完整抓放目标。"""
        index = int(request.index)
        if index < 0 or index >= len(self.task_targets):
            return GetTaskTargetResponse(
                success=False,
                target_type="",
                pick_observation_pose=[0.0] * 6,
                place_observation_pose=[0.0] * 6,
                row=0.0,
                col=0.0,
                category="",
                detected_angle_deg=0.0,
                rotation_delta_deg=0.0,
                pick_surface_z_mm=0.0,
                pick_surface_z_valid=False,
                pick_high_detected_pixel_xy=[0.0] * 2,
                pick_high_depth_sample_pixel_xy=[0.0] * 2,
                pick_high_image_center_xy=[0.0] * 2,
                pick_high_world_position=[0.0] * 3,
                pick_high_world_position_valid=False,
                pick_rough_localization_source="",
                pick_depth_valid_frame_count=0,
                pick_depth_median_mm=0.0,
                pick_depth_mad_mm=0.0,
                pick_calibration_target_tcp_z_mm=0.0,
                place_high_detected_pixel_xy=[0.0] * 2,
                place_high_depth_sample_pixel_xy=[0.0] * 2,
                place_high_image_center_xy=[0.0] * 2,
                place_high_world_position=[0.0] * 3,
                place_high_world_position_valid=False,
                place_rough_localization_source="",
                place_depth_valid_frame_count=0,
                place_depth_median_mm=0.0,
                place_depth_mad_mm=0.0,
                place_calibration_target_tcp_z_mm=0.0,
                message=f"任务序号越界: {index}，当前任务数: {len(self.task_targets)}",
            )
        target = self.task_targets[index]
        return GetTaskTargetResponse(
            success=True,
            target_type=str(getattr(target, "target_type", "pick_place") or "pick_place"),
            pick_observation_pose=list(target.pick_observation_pose),
            place_observation_pose=list(target.place_observation_pose),
            row=target.row,
            col=target.col,
            category=target.category,
            detected_angle_deg=target.detected_angle_deg,
            rotation_delta_deg=target.rotation_delta_deg,
            pick_surface_z_mm=target.pick_surface_z_mm,
            pick_surface_z_valid=target.pick_surface_z_valid,
            pick_high_detected_pixel_xy=list(target.pick_high_detected_pixel_xy),
            pick_high_depth_sample_pixel_xy=list(target.pick_high_depth_sample_pixel_xy),
            pick_high_image_center_xy=list(target.pick_high_image_center_xy),
            pick_high_world_position=list(target.pick_high_world_position),
            pick_high_world_position_valid=target.pick_high_world_position_valid,
            pick_rough_localization_source=target.pick_rough_localization_source,
            pick_depth_valid_frame_count=target.pick_depth_valid_frame_count,
            pick_depth_median_mm=target.pick_depth_median_mm,
            pick_depth_mad_mm=target.pick_depth_mad_mm,
            pick_calibration_target_tcp_z_mm=target.pick_calibration_target_tcp_z_mm,
            place_high_detected_pixel_xy=list(target.place_high_detected_pixel_xy),
            place_high_depth_sample_pixel_xy=list(target.place_high_depth_sample_pixel_xy),
            place_high_image_center_xy=list(target.place_high_image_center_xy),
            place_high_world_position=list(target.place_high_world_position),
            place_high_world_position_valid=target.place_high_world_position_valid,
            place_rough_localization_source=target.place_rough_localization_source,
            place_depth_valid_frame_count=target.place_depth_valid_frame_count,
            place_depth_median_mm=target.place_depth_median_mm,
            place_depth_mad_mm=target.place_depth_mad_mm,
            place_calibration_target_tcp_z_mm=target.place_calibration_target_tcp_z_mm,
            message="读取任务目标成功",
        )

    def make_visual_servo_result(
        self,
        found=False,
        target_type="",
        category="",
        px=0,
        py=0,
        dx_px=0,
        dy_px=0,
        theta=0,
        score=0,
        message="",
    ):
        """统一组织视觉伺服偏差结果，方块和托盘服务都必须返回这个结构。

        found 表示本帧是否识别到目标。
        px/py 是目标点在图像中的像素坐标。
        dx_px/dy_px 是目标点相对图像中心的像素偏差，闭环控制只依赖这两个值移动机械臂。
        theta/score/category 是给方块识别预留的附加信息；托盘分支可以保持默认值。
        """
        return {
            "found": bool(found),
            "target_type": str(target_type),
            "category": str(category),
            "px": float(px),
            "py": float(py),
            "dx_px": float(dx_px),
            "dy_px": float(dy_px),
            "theta": float(theta),
            "score": float(score),
            "message": str(message),
        }

    def make_block_angle_prior_message(self, angle_center, angle_window, angle_step):
        """生成低位模板角度先验摘要，便于确认 competition.py 的 t 已通信到图像节点。"""
        if angle_center is None or angle_window is None:
            return "角度先验: 未启用"
        return (
            f"角度先验: center={float(angle_center):.1f}, "
            f"window={float(angle_window):.1f}, step={float(angle_step):.1f}"
        )

    def detect_board_visual_offset(self, row, col):
        """识别低位托盘目标点相对相机中心的像素偏差。

        这里只负责图像识别和像素偏差计算，不控制机械臂。
        低位时托盘通常不完整入画，因此只在画面中心小 ROI 内找托盘圆点。
        row/col 支持整数和 .5，小数目标会用相邻圆点平均成虚拟目标点。
        """
        request_started_at = time.perf_counter()
        request_stamp = rospy.Time.now()
        img_bgr1 = self.get_image_snapshot_newer_than(request_stamp)
        snapshot_finished_at = time.perf_counter()
        if img_bgr1 is None:
            self.log_visual_servo_detection_timing(
                "托盘", request_started_at, snapshot_finished_at
            )
            return self.make_visual_servo_result(
                found=False,
                target_type="board",
                message=f"等待视觉新帧超时（{self.fresh_image_timeout_sec:.1f} 秒）",
            )

        try:
            detection_started_at = time.perf_counter()
            h, w = img_bgr1.shape[:2]
            board_bgr = img_bgr1
            center_x = w / 2.0
            center_y = h / 2.0
            detect_result = detect_nearest_board_dot_in_roi(
                board_bgr,
                (center_x, center_y),
                roi_half_size=self.board_low_roi_half_size,
                blackhat_kernel_size=self.board_low_blackhat_kernel_size,
                min_area=self.board_low_min_dot_area,
                max_area=self.board_low_max_dot_area,
                min_circularity=self.board_low_min_dot_circularity,
                max_aspect_ratio=self.board_low_max_dot_aspect_ratio,
                row=row,
                col=col,
                debug_enabled=self.visual_servo_debug_enabled,
            )
            detection_finished_at = time.perf_counter()
            debug_image = detect_result.get("debug_image")
            debug_panel = detect_result.get("debug_panel")
            if not detect_result["found"]:
                debug_started_at = time.perf_counter()
                if self.visual_servo_debug_enabled:
                    self.record_visual_board_debug_frame(debug_panel)
                debug_finished_at = time.perf_counter()
                self.log_visual_servo_detection_timing(
                    "托盘",
                    request_started_at,
                    snapshot_finished_at,
                    detection_started_at,
                    detection_finished_at,
                    debug_started_at,
                    debug_finished_at,
                )
                return self.make_visual_servo_result(
                    found=False,
                    target_type="board",
                    message=detect_result["message"],
                )

            target_point = detect_result["point"]
            dx_px = float(target_point[0] - center_x)
            dy_px = float(target_point[1] - center_y)
            debug_started_at = time.perf_counter()
            if self.visual_servo_debug_enabled:
                self.record_visual_board_debug_frame(debug_panel)
            debug_finished_at = time.perf_counter()
            self.log_visual_servo_detection_timing(
                "托盘",
                request_started_at,
                snapshot_finished_at,
                detection_started_at,
                detection_finished_at,
                debug_started_at,
                debug_finished_at,
            )

            return self.make_visual_servo_result(
                found=True,
                target_type="board",
                category="board_low_dot",
                px=float(target_point[0]),
                py=float(target_point[1]),
                dx_px=dx_px,
                dy_px=dy_px,
                theta=0,
                score=1.0,
                message=detect_result.get("message", "低位托盘目标点识别成功"),
            )
        except Exception as exc:
            self.log_visual_servo_detection_timing(
                "托盘",
                request_started_at,
                snapshot_finished_at,
                locals().get("detection_started_at"),
                time.perf_counter(),
            )
            rospy.logerr("托盘视觉伺服目标检测失败: %s" % exc)
            print("\033[91m托盘视觉伺服目标检测失败，请重新识别。\033[0m")
            return self.make_visual_servo_result(
                found=False,
                target_type="board",
                message=str(exc),
            )

    def detect_block_visual_offset(self, expected_category, angle_step, angle_center, angle_window):
        """使用高位类别和角度先验识别低位方块像素偏差。"""
        prior_message = self.make_block_angle_prior_message(angle_center, angle_window, angle_step)
        request_started_at = time.perf_counter()
        request_stamp = rospy.Time.now()
        image = self.get_image_snapshot_newer_than(request_stamp)
        snapshot_finished_at = time.perf_counter()
        if image is None:
            self.log_visual_servo_detection_timing(
                "方块", request_started_at, snapshot_finished_at
            )
            return self.make_visual_servo_result(
                found=False,
                target_type="block",
                message=(
                    f"等待视觉新帧超时（{self.fresh_image_timeout_sec:.1f} 秒），"
                    f"{prior_message}"
                ),
            )

        category = normalize_category_name((expected_category or "").strip())
        if not category:
            self.log_visual_servo_detection_timing(
                "方块", request_started_at, snapshot_finished_at
            )
            return self.make_visual_servo_result(
                found=False,
                target_type="block",
                message=f"低位识别缺少高位类别，{prior_message}",
            )
        try:
            height, width = image.shape[:2]
            center_x, center_y = width / 2.0, height / 2.0
            detection_started_at = time.perf_counter()
            geometry_started_at = time.perf_counter() if self.visual_servo_timing_debug else None
            template_geometry = load_template_geometry("low")
            geometry_finished_at = time.perf_counter() if self.visual_servo_timing_debug else None
            target_block = detect_block_with_high_prior_roi(
                image,
                template_geometry=template_geometry,
                category=category,
                high_theta_deg=float(angle_center),
                angle_window=float(angle_window),
                angle_step=float(angle_step),
                search_radius_px=self.block_low_search_radius_px,
                fallback_search_radius_px=self.block_low_fallback_search_radius_px,
                boundary_guard_px=self.block_low_boundary_guard_px,
                kernel_safety_margin_px=self.block_low_kernel_safety_margin_px,
                legacy_fallback_enabled=self.block_low_legacy_fallback_enabled,
                white_s_max=self.block_low_white_s_max,
                white_v_min=self.block_low_white_v_min,
                min_foreground_area=self.block_low_min_foreground_area,
                debug_enabled=self.visual_servo_debug_enabled,
                timing_enabled=self.visual_servo_timing_debug,
            )
            detection_finished_at = time.perf_counter()
            block_timing = target_block.get("timing")
            if block_timing is not None:
                block_timing["模板几何配置毫秒"] = (
                    (geometry_finished_at - geometry_started_at) * 1000.0
                )
            debug_image = target_block.get("debug_image")
            if self.visual_servo_debug_enabled and debug_image is None:
                debug_image = image.copy()
            debug_panel = target_block.get("debug_panel")
            if self.visual_servo_debug_enabled:
                cv2.drawMarker(
                    debug_image,
                    (int(center_x), int(center_y)),
                    (255, 0, 0),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=24,
                    thickness=2,
                )
            if not target_block["found"]:
                debug_started_at = time.perf_counter()
                if self.visual_servo_debug_enabled:
                    self.record_visual_servo_debug_frame(debug_panel)
                debug_finished_at = time.perf_counter()
                self.log_visual_servo_detection_timing(
                    "方块",
                    request_started_at,
                    snapshot_finished_at,
                    detection_started_at,
                    detection_finished_at,
                    debug_started_at,
                    debug_finished_at,
                    block_timing=block_timing,
                )
                return self.make_visual_servo_result(
                    found=False,
                    target_type="block",
                    category=category,
                    message=f"{target_block['message']}，{prior_message}",
                )

            px = float(target_block["px"])
            py = float(target_block["py"])
            if self.visual_servo_debug_enabled:
                cv2.line(
                    debug_image,
                    (int(center_x), int(center_y)),
                    (int(px), int(py)),
                    (255, 0, 0),
                    1,
                )
            debug_started_at = time.perf_counter()
            if self.visual_servo_debug_enabled:
                self.record_visual_servo_debug_frame(debug_panel)
            debug_finished_at = time.perf_counter()
            self.log_visual_servo_detection_timing(
                "方块",
                request_started_at,
                snapshot_finished_at,
                detection_started_at,
                detection_finished_at,
                debug_started_at,
                debug_finished_at,
                block_timing=block_timing,
            )
            return self.make_visual_servo_result(
                found=True,
                target_type="block",
                category=target_block["category"],
                px=px,
                py=py,
                dx_px=px - center_x,
                dy_px=py - center_y,
                theta=float(target_block["theta"]),
                score=float(target_block["score"]),
                message=f"低位先验 ROI 模板匹配成功，{prior_message}",
            )
        except Exception as exc:
            self.log_visual_servo_detection_timing(
                "方块",
                request_started_at,
                snapshot_finished_at,
                locals().get("detection_started_at"),
                time.perf_counter(),
            )
            rospy.logerr("方块视觉伺服目标检测失败: %s", exc)
            return self.make_visual_servo_result(
                found=False,
                target_type="block",
                category=category,
                message=f"{exc}，{prior_message}",
            )

    def detect_block_offset_service(self, request):
        """方块低位视觉伺服服务入口。"""
        result = self.detect_block_visual_offset(
            expected_category=request.category,
            angle_step=self.block_angle_step_deg,
            angle_center=float(request.high_angle_deg),
            angle_window=self.block_angle_window_deg,
        )
        return DetectBlockOffsetResponse(
            found=result["found"],
            px=result["px"],
            py=result["py"],
            dx_px=result["dx_px"],
            dy_px=result["dy_px"],
            detected_angle_deg=result["theta"],
            score=result["score"],
            message=result["message"],
        )

    def detect_board_offset_service(self, request):
        """托盘低位视觉伺服服务入口。"""
        result = self.detect_board_visual_offset(request.row, request.col)
        return DetectBoardOffsetResponse(
            found=result["found"],
            px=result["px"],
            py=result["py"],
            dx_px=result["dx_px"],
            dy_px=result["dy_px"],
            message=result["message"],
        )
