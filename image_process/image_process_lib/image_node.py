#!/home/zhl/fr3env/fr3env/bin/python
"""图像快照、任务规划和视觉伺服检测的 ROS 服务组合节点。"""

import os
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
from image_process_lib.block_scene_detector import detect_blocks_in_image
from image_process_lib.block_servo_detector import detect_block_with_high_prior_roi
from image_process_lib.advanced_planner import AdvancedPlanner
from image_process_lib.debug_output import DebugVideoRecorder, save_image_to_path
from image_process_lib.depth_rough_localization import DepthRoughLocalizer, fit_z_plane
from image_process_lib.high_pixel_to_tcp_localizer import HighPixelToTcpLocalizer
from image_process_lib.task_planner import (
    ObservedBlock,
    PlacementTarget,
    assign_blocks_to_targets,
    load_task_layout,
)

from image_process.srv import (
    DetectBlockOffset,
    DetectBlockOffsetResponse,
    DetectBoardOffset,
    DetectBoardOffsetResponse,
    GetTaskTarget,
    GetTaskTargetResponse,
    PrepareTask,
    PrepareTaskResponse,
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


class ImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_lock = threading.Lock()
        self.image_condition = threading.Condition(self.image_lock)
        self.latest_image = None
        self.latest_image_stamp = None
        self.fresh_image_timeout_sec = float(rospy.get_param("~fresh_image_timeout_sec", 0.5))
        if not np.isfinite(self.fresh_image_timeout_sec) or self.fresh_image_timeout_sec <= 0.0:
            raise ValueError("fresh_image_timeout_sec 必须是大于 0 的有限数值")
        self.task_targets = []
        self.board_grid_points = None
        self.board_grid_image_shape = None
        self.board_grid_image = None

        perception_config_path = rospy.get_param("~perception_config", PERCEPTION_CONFIG_PATH)
        with open(perception_config_path, "r", encoding="utf-8") as config_file:
            perception_config = yaml.safe_load(config_file) or {}
        model_config = perception_config.get("models", {})
        calibration_config = perception_config.get("calibration", {})

        def src_path(value, fallback):
            raw_path = str(value or fallback)
            return raw_path if os.path.isabs(raw_path) else os.path.join(SRC_DIR, raw_path)

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
        debug_output_dir = visual_servo_debug_config.get("output_dir", DEFAULT_DEBUG_DIR)
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
            self.visual_servo_debug_recorder = DebugVideoRecorder(
                os.path.join(self.visual_servo_debug_output_dir, "方块视觉伺服调试.avi"),
                fps=self.visual_servo_debug_video_fps,
                enabled=True,
            )
            self.visual_board_debug_recorder = DebugVideoRecorder(
                os.path.join(self.visual_servo_debug_output_dir, "托盘视觉伺服调试.avi"),
                fps=self.visual_servo_debug_video_fps,
                enabled=True,
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
        self.block_low_roi_expand_px = rospy.get_param("~block_low_roi_expand_px", block_servo["roi_expand_px"])
        self.block_low_rectified_roi_enabled = rospy.get_param(
            "~block_low_rectified_roi_enabled",
            block_servo.get("rectified_roi_enabled", False),
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

        with open(EXECUTION_CONFIG_PATH, "r", encoding="utf-8") as config_file:
            execution_config = yaml.safe_load(config_file) or {}
        calibration_mode_value = execution_config.get("calibration_mode", False)
        if not isinstance(calibration_mode_value, bool):
            raise ValueError("calibration_mode 必须是 YAML 布尔值 true 或 false")
        self.calibration_mode = calibration_mode_value
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
        self.minimum_tcp_z_mm = float(motion_config.get("minimum_tcp_z_mm", 165.0))
        self.pick_surface_offset_mm = float(motion_config.get("pick_surface_offset_mm", 0.0))
        if not np.isfinite(self.minimum_tcp_z_mm) or self.minimum_tcp_z_mm <= 0.0:
            raise ValueError("minimum_tcp_z_mm 必须是大于 0 的有限数值")
        if not np.isfinite(self.pick_surface_offset_mm):
            raise ValueError("pick_surface_offset_mm 必须是有限数值")

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
            "tray_tcp_below_block_observation_mm": self.tray_tcp_below_block_observation_mm,
        }
        for name, value in positive_depth_values.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"calibration_depth.{name} 必须是大于 0 的有限数值"
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
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
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
        rospy.on_shutdown(self.close_debug_video_recorders)
        rospy.loginfo("图像处理服务已启动")

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
            f"ROI矫正={format_ms(stages_ms.get('ROI矫正'))}，"
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
        save_image_to_path(self.visual_board_grid_debug_path, debug_image)
        if not result["found"]:
            raise RuntimeError(result["message"])
        self.board_grid_points = result["grid_points"]
        self.board_grid_image_shape = image.shape[:2]
        self.board_grid_image = image.copy()
        self.board_theta = self.compute_board_theta_from_grid_points(self.board_grid_points)
        rospy.loginfo("托盘像素角度: %.2f 度", self.board_theta)

    def _detect_blocks_raw(self, image):
        """识别高位方块，仅整理类别、像素和角度，不生成机械臂位姿。"""
        geometry = load_template_geometry("high")
        blocks, debug_image = detect_blocks_in_image(
            image,
            self.model,
            template_geometry=geometry,
            crop_margin=8,
            save_mask_overlay=self.save_top_surface_mask_vis,
        )
        save_image_to_path(self.high_template_match_debug_path, debug_image)
        if not blocks:
            raise RuntimeError("高位没有识别到方块")
        if self.save_top_surface_mask_vis:
            mask_image = blocks[-1].get("mask_overlay")
            if mask_image is None:
                mask_image = debug_image
            save_image_to_path(self.top_surface_mask_vis_path, mask_image)
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

    def _detect_blocks_for_task(self, image):
        """正式模式识别高位方块并使用部署标定生成粗观察位。"""
        blocks, count_list = self._detect_blocks_raw(image)
        observed_blocks = []
        for block in blocks:
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
                image.shape,
                depth_sample_pixel_xy,
            )
            observed_blocks.append(
                ObservedBlock(
                    category=category,
                    observation_pose=tuple(servo_pose),
                    detected_angle_deg=float(block["theta"]),
                    pick_surface_z_mm=pick_surface_z_mm,
                    pick_surface_z_valid=True,
                    **diagnostic,
                )
            )
        return observed_blocks, count_list

    def _load_layout_for_request(self, request, cube_counts):
        if not request.advanced:
            return load_task_layout(self.task_layout_path), "基础任务布局"
        if len(request.place_order) != 7:
            raise ValueError("进阶任务顺序必须正好包含 7 个整数")
        layout, fill_line = AdvancedPlanner(self.advanced_library_path).build_layout(
            cube_counts, request.place_order
        )
        return layout, f"进阶任务布局，极限填满 {fill_line} 行"

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
                    **diagnostic,
                )
            )
        return targets

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

    def _build_calibration_targets(self, blocks, layout, image_shape):
        """标定模式用深度 XYZ 生成方块和托盘粗位姿及完整诊断。"""
        placement_specs = self._placement_specs(layout)
        pixels = [(block["px"], block["py"]) for block in blocks]
        pixels.extend(point for _item, point in placement_specs)
        samples = self._query_stable_world_points(pixels)
        block_samples = samples[:len(blocks)]
        tray_samples = samples[len(blocks):]

        observed_blocks = []
        block_observation_points = []
        for block, sample in zip(blocks, block_samples):
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
            block_observation_points.append(pose[:3])
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

        block_plane = fit_z_plane(block_observation_points)
        if block_plane.rmse_mm > self.block_plane_max_rmse_mm:
            raise RuntimeError(
                f"方块观察 TCP 平面 RMSE={block_plane.rmse_mm:.3f} mm "
                f"超过阈值 {self.block_plane_max_rmse_mm:.3f} mm"
            )

        placement_targets = []
        for (item, point), sample in zip(placement_specs, tray_samples):
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
        return observed_blocks, placement_targets

    def prepare_task(self, request):
        """用同一高位图像快照完成托盘、方块识别与任务规划。"""
        self.task_targets = []
        self.board_grid_points = None
        self.board_grid_image_shape = None
        self.board_grid_image = None
        request_stamp = rospy.Time.now()
        image = self.get_image_snapshot_newer_than(request_stamp)
        if image is None:
            return PrepareTaskResponse(
                success=False,
                task_count=0,
                message=f"等待高位新图像超时（{self.fresh_image_timeout_sec:.1f} 秒）",
            )
        try:
            self._detect_board_for_task(image)
            if self.calibration_mode:
                blocks, cube_counts = self._detect_blocks_raw(image)
                layout, layout_message = self._load_layout_for_request(request, cube_counts)
                observed_blocks, placement_targets = self._build_calibration_targets(
                    blocks,
                    layout,
                    image.shape[:2],
                )
            else:
                observed_blocks, cube_counts = self._detect_blocks_for_task(image)
                layout, layout_message = self._load_layout_for_request(request, cube_counts)
                placement_targets = self._build_placement_targets(layout, image.shape[:2])
            self.task_targets = assign_blocks_to_targets(
                observed_blocks,
                placement_targets,
                board_angle_deg=self.board_theta,
            )
            message = f"{layout_message}准备完成，共 {len(self.task_targets)} 个任务"
            rospy.loginfo(message)
            return PrepareTaskResponse(
                success=True,
                task_count=len(self.task_targets),
                message=message,
            )
        except Exception as exc:
            self.task_targets = []
            rospy.logerr("任务准备失败: %s", exc)
            return PrepareTaskResponse(success=False, task_count=0, message=str(exc))

    def get_task_target(self, request):
        """按执行顺序返回一个完整抓放目标。"""
        index = int(request.index)
        if index < 0 or index >= len(self.task_targets):
            return GetTaskTargetResponse(
                success=False,
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
                roi_expand_px=self.block_low_roi_expand_px,
                white_s_max=self.block_low_white_s_max,
                white_v_min=self.block_low_white_v_min,
                min_foreground_area=self.block_low_min_foreground_area,
                debug_enabled=self.visual_servo_debug_enabled,
                timing_enabled=self.visual_servo_timing_debug,
                rectified_roi_enabled=self.block_low_rectified_roi_enabled,
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
