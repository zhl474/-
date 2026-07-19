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
from image_process_lib.rough_localization import RoughLocalizer
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
from camera.srv import PixelToWorld


PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, ".."))
COMPETITION_DIR = os.path.join(SRC_DIR, "competition")
CAMERA_DIR = os.path.join(SRC_DIR, "camera")
VISUAL_SERVO_CONFIG_PATH = os.path.join(COMPETITION_DIR, "config", "visual_servo.yaml")
EXECUTION_CONFIG_PATH = os.path.join(COMPETITION_DIR, "config", "execution.yaml")
T_WRIST2CAMERA_PATH = os.path.join(CAMERA_DIR, "config", "T_wrist2camera.npy")
DETECTION_MODEL_PATH = os.path.join(COMPETITION_DIR, "model", "best5.14.pt")
BOARD_MODEL_PATH = os.path.join(COMPETITION_DIR, "model", "best.pt")
SEGMENTATION_MODEL_PATH = os.path.join(COMPETITION_DIR, "model", "best_seg.engine")
JINJIE_LIB_PATH = os.path.join(SRC_DIR, "jinjie", "jinjie_libtetris.so")
TASK_LAYOUT_PATH = os.path.join(PACKAGE_DIR, "config", "task_layout.yaml")
PERCEPTION_CONFIG_PATH = os.path.join(PACKAGE_DIR, "config", "perception.yaml")
DEFAULT_DEBUG_DIR = os.path.expanduser("~/.ros/single_arm_tetris")


def load_servo_height_offsets_mm(config_path=VISUAL_SERVO_CONFIG_PATH):
    """从视觉伺服配置读取方块和托盘各自的观察高度偏移。"""
    if not os.path.exists(config_path):
        return 200.0, 200.0
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    offsets = (
        float(data.get("block_servo_height_offset_mm", 200.0)),
        float(data.get("board_servo_height_offset_mm", 200.0)),
    )
    if not np.all(np.isfinite(offsets)) or min(offsets) <= 0.0:
        raise ValueError("方块和托盘视觉伺服观察高度偏置必须是大于 0 的有限数值")
    return offsets


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
        hand_eye_matrix_path = rospy.get_param(
            "~hand_eye_matrix_path", src_path(calibration_config.get("hand_eye_matrix"), T_WRIST2CAMERA_PATH)
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
        (
            self.block_servo_height_offset_mm,
            self.board_servo_height_offset_mm,
        ) = load_servo_height_offsets_mm()
        self.T_wrist2camera_mm = np.load(hand_eye_matrix_path)
        if self.T_wrist2camera_mm.shape != (4, 4) or not np.all(np.isfinite(self.T_wrist2camera_mm)):
            raise ValueError("手眼标定矩阵 T_wrist2camera.npy 无效，请重新标定")
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

        # 高位全场识别后，用像素偏差粗估方块机械臂坐标。
        rough_config = perception_config["high_rough_localization"]
        self.high_rough_x_mm_per_pixel = float(rough_config["x_mm_per_pixel"])
        self.high_rough_y_mm_per_pixel = float(rough_config["y_mm_per_pixel"])
        fallback_config = perception_config.get("fallback", {})
        self.depth_fallback_enabled = bool(fallback_config.get("depth_pixel_estimate", True))
        block_detection_module.ALLOW_COLOR_FALLBACK = bool(fallback_config.get("color_segmentation", True))

        self.pixel2world_client = rospy.ServiceProxy("/camera/pixel_to_world", PixelToWorld)
        self.pixel2world_client.wait_for_service()

        with open(EXECUTION_CONFIG_PATH, "r", encoding="utf-8") as config_file:
            execution_config = yaml.safe_load(config_file) or {}
        self.visual_servo_timing_debug = bool(
            rospy.get_param(
                "~visual_servo_timing_debug",
                execution_config.get("servo", {}).get("timing_debug", False),
            )
        )
        self.shooting_angle = [float(value) for value in execution_config["shooting_pose"]]
        if len(self.shooting_angle) != 6 or not np.all(np.isfinite(self.shooting_angle)):
            raise ValueError("shooting_pose 必须包含 6 个有限数值")
        self.rough_localizer = RoughLocalizer(
            shooting_pose=self.shooting_angle,
            wrist_to_camera_mm=self.T_wrist2camera_mm,
            pixel_to_world_client=self.pixel2world_client,
            x_mm_per_pixel=self.high_rough_x_mm_per_pixel,
            y_mm_per_pixel=self.high_rough_y_mm_per_pixel,
            fallback_enabled=self.depth_fallback_enabled,
            warning_func=self.print_yellow_warning,
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

    def print_yellow_warning(self, message):
        """输出黄色警告，现场调试时用于区分深度回退。"""
        rospy.logwarn(message)
        print(f"\033[93m{message}\033[0m")

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

    def get_image_snapshot(self):
        """在锁内复制当前图像，服务处理期间始终使用同一帧。"""
        with self.image_condition:
            return None if self.latest_image is None else self.latest_image.copy()

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

    def make_depth_first_servo_pose(self, px, py, image_shape, label, height_offset_mm):
        """兼容节点内部调用，实际粗定位由 RoughLocalizer 完成。"""
        return self.rough_localizer.locate(px, py, image_shape, label, height_offset_mm)

    @staticmethod
    def make_high_localization_diagnostic(px, py, image_shape, world_position, localization_source):
        """整理高位识别到粗定位的原始数据，供主进程输出诊断日志。"""
        height, width = image_shape[:2]
        world = np.asarray(world_position, dtype=float)
        world_valid = (
            str(localization_source) == "depth"
            and world.shape == (3,)
            and np.all(np.isfinite(world))
        )
        return {
            "high_detected_pixel_xy": (float(px), float(py)),
            # 深度服务内部会对检测像素四舍五入，此处单独保存以便排查一像素误差。
            "high_depth_sample_pixel_xy": (float(round(float(px))), float(round(float(py)))),
            "high_image_center_xy": (float(width) / 2.0, float(height) / 2.0),
            "high_world_position": tuple(world.tolist()) if world_valid else (0.0, 0.0, 0.0),
            "high_world_position_valid": bool(world_valid),
            "rough_localization_source": str(localization_source),
        }

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

    def _detect_blocks_for_task(self, image):
        """识别高位方块并生成粗观察位。"""
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

        observed_blocks = []
        counts = {category: 0 for category in BLOCK_CATEGORY_NAMES}
        for block in blocks:
            category = normalize_category_name(block["category"])
            if category not in counts:
                rospy.logwarn("忽略未知方块类别: %s", category)
                continue
            servo_pose, localization_source, world_position = self.make_depth_first_servo_pose(
                block["px"],
                block["py"],
                image.shape,
                f"方块 {category}",
                self.block_servo_height_offset_mm,
            )
            has_valid_surface_z = localization_source == "depth" and len(world_position) == 3
            diagnostic = self.make_high_localization_diagnostic(
                block["px"],
                block["py"],
                image.shape,
                world_position,
                localization_source,
            )
            counts[category] += 1
            observed_blocks.append(
                ObservedBlock(
                    category=category,
                    observation_pose=tuple(servo_pose),
                    detected_angle_deg=float(block["theta"]),
                    pick_surface_z_mm=float(world_position[2]) if has_valid_surface_z else 0.0,
                    pick_surface_z_valid=has_valid_surface_z,
                    **diagnostic,
                )
            )
        if not observed_blocks:
            raise RuntimeError("没有可用于任务规划的已知类别方块")
        invalid_depth_count = sum(not block.pick_surface_z_valid for block in observed_blocks)
        if invalid_depth_count:
            raise RuntimeError(
                f"{invalid_depth_count} 个方块缺少有效深度高度，已取消任务准备；"
                "请调整相机视野、方块摆放或光照后重试"
            )
        return observed_blocks, [counts[category] for category in BLOCK_CATEGORY_NAMES]

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
            servo_pose, localization_source, world_position = self.make_depth_first_servo_pose(
                target_point[0],
                target_point[1],
                image_shape,
                f"托盘目标 {item['index']}",
                self.board_servo_height_offset_mm,
            )
            diagnostic = self.make_high_localization_diagnostic(
                target_point[0],
                target_point[1],
                image_shape,
                world_position,
                localization_source,
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

    def prepare_task(self, request):
        """用同一高位图像快照完成托盘、方块识别与任务规划。"""
        self.task_targets = []
        self.board_grid_points = None
        self.board_grid_image_shape = None
        self.board_grid_image = None
        image = self.get_image_snapshot()
        if image is None:
            return PrepareTaskResponse(success=False, task_count=0, message="没有可用图像")
        try:
            self._detect_board_for_task(image)
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
                place_high_detected_pixel_xy=[0.0] * 2,
                place_high_depth_sample_pixel_xy=[0.0] * 2,
                place_high_image_center_xy=[0.0] * 2,
                place_high_world_position=[0.0] * 3,
                place_high_world_position_valid=False,
                place_rough_localization_source="",
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
            place_high_detected_pixel_xy=list(target.place_high_detected_pixel_xy),
            place_high_depth_sample_pixel_xy=list(target.place_high_depth_sample_pixel_xy),
            place_high_image_center_xy=list(target.place_high_image_center_xy),
            place_high_world_position=list(target.place_high_world_position),
            place_high_world_position_valid=target.place_high_world_position_valid,
            place_rough_localization_source=target.place_rough_localization_source,
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
