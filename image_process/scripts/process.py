#!/home/zhl/fr3env/fr3env/bin/python
import os
import sys

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

import yaml
from ultralytics import YOLO

IMAGE_PROCESS_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if IMAGE_PROCESS_SRC_DIR not in sys.path:
    sys.path.insert(0, IMAGE_PROCESS_SRC_DIR)

from image_process_lib.Place_optimization import (
    get_all_cube,
    make_list,
    cube_pocess,
    optimize_block_assignment,
    get_cube_location as calc_cube_location,
)
from image_process_lib.template_config import load_template_geometry
from image_process_lib.board_detect import (
    BOARD_COL_COUNT,
    BOARD_ROW_COUNT,
    board_grid_detect,
    detect_nearest_board_dot_in_roi,
    draw_grid_debug,
    interpolate_grid_point,
)
from image_process_lib.block_category import BLOCK_CATEGORY_NAMES, category_to_code
from image_process_lib.single_block_detector import (
    detect_block_with_high_prior_roi,
    detect_blocks_in_image,
    normalize_category_name,
    undistort_bgr_image,
)

from image_process.srv import GetTargetPos, GetTargetPosResponse
from image_process.srv import VisualTargetOffset, VisualTargetOffsetResponse
from image_process.srv import VisualBoardOffset, VisualBoardOffsetResponse
from image_process.srv import VisualServoOffset, VisualServoOffsetResponse
from camera.srv import pixel2world, pixel2worldRequest

from ctypes import CDLL, c_char_p


CALIBRATION_MATRIX_PATH = "/home/zhl/SingleArmTetris/SingleArmTetris/src/competition/config/calibration_matrix.yaml"
VISUAL_SERVO_CONFIG_PATH = "/home/zhl/SingleArmTetris/SingleArmTetris/src/competition/config/visual_servo.yaml"
T_WRIST2CAMERA_PATH = "/home/zhl/SingleArmTetris/SingleArmTetris/src/camera/config/T_wrist2camera.npy"
DETECTION_MODEL_PATH = "/home/zhl/SingleArmTetris/SingleArmTetris/src/competition/model/best5.14.pt"
JINJIE_LIB_PATH = "/home/zhl/SingleArmTetris/SingleArmTetris/src/jinjie/jinjie_libtetris.so"

VISUAL_TARGET_BLOCK = "block"
VISUAL_TARGET_BOARD = "board"

# 高位全场拍摄位姿。competition.py 里也有同一份值，后面应通过服务返回动态规划结果来彻底统一。
BASE_SHOOTING_ANGLE = [-250.4151306152343, 22.14801216125488, 380.3343505859375, -180, 0, 90]


def load_servo_look_z(config_path=VISUAL_SERVO_CONFIG_PATH):
    """从视觉伺服配置读取相机观察高度。"""
    if not os.path.exists(config_path):
        return 200.0
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return float(data.get("servo_look_z", 200.0))


def rpy_degrees_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    """按常用 Rz(yaw) * Ry(pitch) * Rx(roll) 生成旋转矩阵。"""
    roll = np.deg2rad(float(roll_deg))
    pitch = np.deg2rad(float(pitch_deg))
    yaw = np.deg2rad(float(yaw_deg))

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ])
    ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ])
    rz = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ])
    return rz @ ry @ rx


def build_base_pick_list(include_index=False):
    """基础任务摆放表：col、row、目标角度、方块类别。"""
    pick_list = [
        [1, 2, 90, 'L_yellow'], [4, 1.5, 0, 'T'], [7.5, 1, 0, 'line'], [9.5, 2, -90, 'z_green'], [2, 3, 90, 'L_yellow'],
        [4, 3, 180, 'L_blue'], [7, 2, 0, 'L_yellow'], [1.5, 5, 90, 'T'], [3.5, 5, 90, 'z_blue'], [5, 4.5, 0, 'z_green'],
        [6.5, 3.5, 0, 'square'], [9, 5, -90, 'L_blue'], [10, 4.5, 90, 'line'], [7.5, 5.5, 0, 'square'], [1.5, 7, -90, 'z_green'],
        [3.5, 7, 90, 'z_blue'], [6.5, 7, 90, 'T'], [5, 7.5, 90, 'line'], [9, 7, 0, 'L_blue'], [3, 9, -90, 'L_blue'],
        [7, 8.5, 180, 'T'], [9.5, 8.5, 0, 'square'], [1, 10, 90, 'L_yellow'], [4.5, 10, 90, 'T'], [6.5, 11, 90, 'z_blue'],
        [8.5, 10, 0, 'line'], [2.5, 11, 90, 'z_blue'], [8, 12, 90, 'L_yellow'], [9.5, 12, -90, 'z_green'], [1.5, 12.5, 0, 'square'],
        [3.5, 13, -90, 'z_green'], [5, 12.5, 90, 'line'], [6.5, 13, 90, 'z_blue'], [9, 14, 180, 'L_blue']
    ]
    if include_index:
        for index, item in enumerate(pick_list):
            item.append(index)
    return pick_list


def save_image_to_path(image_path, image):
    """保存调试图像，兼容中文路径，并在失败时输出日志。"""
    if image is None:
        rospy.logwarn("调试图像为空，无法保存: %s" % image_path)
        return False
    try:
        dot_index = image_path.rfind(".")
        image_ext = image_path[dot_index:] if dot_index >= 0 else ".jpg"
        ok, encoded_image = cv2.imencode(image_ext, image)
        if not ok:
            rospy.logwarn("调试图像编码失败: %s" % image_path)
            return False
        encoded_image.tofile(image_path)
        return True
    except Exception as exc:
        rospy.logwarn("调试图像保存失败 %s: %s" % (image_path, exc))
        return False


class DebugVideoRecorder:
    """把每次视觉伺服调试图追加写入视频，便于回看闭环过程。"""

    def __init__(self, video_path, fps=10.0, enabled=True):
        self.requested_video_path = video_path
        self.video_path = video_path
        self.fps = max(0.1, float(fps))
        self.enabled = bool(enabled)
        self.writer = None
        self.frame_size = None
        self.codec_name = None
        self.open_failed = False

    def _make_writer_candidates(self):
        """按文件后缀选择编码器，优先使用本机最稳的 MJPG/AVI。"""
        if not self.requested_video_path:
            return []

        base_path, ext = os.path.splitext(self.requested_video_path)
        ext = ext.lower()
        candidates = []
        if ext in (".mp4", ".m4v", ".mov"):
            candidates.extend([
                (base_path + ".avi", "MJPG"),
                (base_path + ".avi", "XVID"),
                (self.requested_video_path, "mp4v"),
                (self.requested_video_path, "avc1"),
            ])
        elif ext == ".avi":
            candidates.extend([
                (self.requested_video_path, "MJPG"),
                (self.requested_video_path, "XVID"),
            ])
        else:
            video_path = self.requested_video_path + ".avi"
            candidates.extend([
                (video_path, "MJPG"),
                (video_path, "XVID"),
            ])

        unique_candidates = []
        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            unique_candidates.append(candidate)
        return unique_candidates

    def _open(self, frame_size):
        if not self.enabled or self.open_failed:
            return False

        for video_path, codec_name in self._make_writer_candidates():
            try:
                video_dir = os.path.dirname(video_path)
                if video_dir:
                    os.makedirs(video_dir, exist_ok=True)

                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                writer = cv2.VideoWriter(video_path, fourcc, self.fps, frame_size)
                if writer.isOpened():
                    self.writer = writer
                    self.video_path = video_path
                    self.frame_size = frame_size
                    self.codec_name = codec_name
                    rospy.loginfo("视觉伺服调试视频开始录制: %s, fps=%.1f, codec=%s" % (
                        self.video_path,
                        self.fps,
                        self.codec_name,
                    ))
                    return True
                writer.release()
            except Exception as exc:
                rospy.logwarn("视觉伺服调试视频初始化失败 %s: %s" % (video_path, exc))

        self.open_failed = True
        rospy.logwarn("视觉伺服调试视频打开失败: %s" % self.requested_video_path)
        return False

    def write(self, image):
        """追加一帧调试图；灰度图会自动转成 BGR。"""
        if not self.enabled or self.open_failed or image is None:
            return False

        frame = image
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        frame_size = (int(frame.shape[1]), int(frame.shape[0]))
        if self.writer is None:
            if not self._open(frame_size):
                return False
        elif self.frame_size != frame_size:
            frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)

        self.writer.write(frame)
        return True

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class ImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.latest_image = None
        self.board_grid_points = None
        self.board_grid_image_shape = None
        self.board_grid_image = None
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
        self.service1 = rospy.Service("get_cube_pos", GetTargetPos,self.get_cube_pos)
        self.service2 = rospy.Service("get_board_pos", GetTargetPos,self.get_board_pos)
        self.get_cube_location_service = rospy.Service("get_cube_location", GetTargetPos,self.get_cube_location1)#这里这样搞是因为不小心把俩函数重名了，导入的函数也有个get_cube_location
        self.get_put_pose_service = rospy.Service("get_put_pose", GetTargetPos,self.get_put_pose)
        self.visual_target_offset_service = rospy.Service(
            "get_visual_target_offset",
            VisualTargetOffset,
            self.get_visual_target_offset,
        )
        self.visual_board_offset_service = rospy.Service(
            "get_visual_board_offset",
            VisualBoardOffset,
            self.get_visual_board_offset,
        )
        self.visual_servo_offset_service = rospy.Service(
            "get_visual_servo_offset",
            VisualServoOffset,
            self.get_visual_servo_offset,
        )

        self.model = YOLO(DETECTION_MODEL_PATH)
        with open(CALIBRATION_MATRIX_PATH) as f:
            data = yaml.safe_load(f)
        self.camera_matrix = np.array(data['camera_matrix'])
        self.dist_coeff = np.array(data['dist_coeff'])

        self.results = None
        self.board_theta = 0.0
        self.servo_look_z = load_servo_look_z()
        self.T_wrist2camera_mm = np.load(T_WRIST2CAMERA_PATH)
        if self.T_wrist2camera_mm.shape != (4, 4) or not np.all(np.isfinite(self.T_wrist2camera_mm)):
            raise ValueError("手眼标定矩阵 T_wrist2camera.npy 无效，请重新标定")
        self.save_top_surface_mask_vis = rospy.get_param("~save_top_surface_mask_vis", False)
        self.top_surface_mask_vis_path = rospy.get_param(
            "~top_surface_mask_vis_path",
            "/home/zhl/桌面/top_surface_masks.jpg"
        )
        self.visual_servo_debug_path = rospy.get_param(
            "~visual_servo_debug_path",
            "/home/zhl/桌面/视觉伺服当前检测.jpg"
        )
        default_visual_servo_debug_video_path = os.path.splitext(self.visual_servo_debug_path)[0] + ".avi"
        self.visual_servo_debug_video_path = rospy.get_param(
            "~visual_servo_debug_video_path",
            default_visual_servo_debug_video_path,
        )
        self.visual_servo_debug_video_fps = rospy.get_param("~visual_servo_debug_video_fps", 10.0)
        self.visual_servo_debug_video_enabled = rospy.get_param("~visual_servo_debug_video_enabled", True)
        self.visual_servo_debug_recorder = DebugVideoRecorder(
            self.visual_servo_debug_video_path,
            fps=self.visual_servo_debug_video_fps,
            enabled=self.visual_servo_debug_video_enabled,
        )
        self.visual_board_debug_path = rospy.get_param(
            "~visual_board_debug_path",
            "/home/zhl/桌面/托盘视觉伺服当前检测.jpg"
        )
        default_visual_board_debug_video_path = os.path.splitext(self.visual_board_debug_path)[0] + ".avi"
        self.visual_board_debug_video_path = rospy.get_param(
            "~visual_board_debug_video_path",
            default_visual_board_debug_video_path,
        )
        self.visual_board_debug_video_fps = rospy.get_param("~visual_board_debug_video_fps", 10.0)
        self.visual_board_debug_video_enabled = rospy.get_param("~visual_board_debug_video_enabled", True)
        self.visual_board_debug_recorder = DebugVideoRecorder(
            self.visual_board_debug_video_path,
            fps=self.visual_board_debug_video_fps,
            enabled=self.visual_board_debug_video_enabled,
        )
        self.visual_board_grid_debug_path = rospy.get_param(
            "~visual_board_grid_debug_path",
            "/home/zhl/桌面/托盘格点粗定位.jpg"
        )
        self.board_low_roi_half_size = rospy.get_param("~board_low_roi_half_size", 90)
        self.board_low_blackhat_kernel_size = rospy.get_param("~board_low_blackhat_kernel_size", 27)
        self.board_low_min_dot_area = rospy.get_param("~board_low_min_dot_area", 200)
        self.board_low_max_dot_area = rospy.get_param("~board_low_max_dot_area", 800)
        self.board_low_min_dot_circularity = rospy.get_param("~board_low_min_dot_circularity", 0.35)
        self.board_low_max_dot_aspect_ratio = rospy.get_param("~board_low_max_dot_aspect_ratio", 1.8)
        self.block_low_roi_expand_px = rospy.get_param("~block_low_roi_expand_px", 50)
        self.block_low_white_s_max = rospy.get_param("~block_low_white_s_max", 45)
        self.block_low_white_v_min = rospy.get_param("~block_low_white_v_min", 180)
        self.block_low_min_foreground_area = rospy.get_param("~block_low_min_foreground_area", 200)

        # 高位全场识别后，用像素偏差粗估方块机械臂坐标。
        self.high_rough_x_mm_per_pixel = 0.5
        self.high_rough_y_mm_per_pixel = 0.5

        self.pixel2world_client = rospy.ServiceProxy("get_world_pos",pixel2world)
        self.pixel2world_client.wait_for_service()

        self.shooting_angle = list(BASE_SHOOTING_ANGLE)
        self.pick_list = build_base_pick_list(include_index=True)
        rospy.on_shutdown(self.close_debug_video_recorders)
        rospy.loginfo("图像处理服务已启动")

    def print_yellow_warning(self, message):
        """输出黄色警告，现场调试时用于区分深度回退。"""
        rospy.logwarn(message)
        print(f"\033[93m{message}\033[0m")

    def close_debug_video_recorders(self):
        """节点退出时释放视频文件句柄，避免最后几帧没有写入文件。"""
        if hasattr(self, "visual_servo_debug_recorder"):
            self.visual_servo_debug_recorder.release()
        if hasattr(self, "visual_board_debug_recorder"):
            self.visual_board_debug_recorder.release()

    def save_visual_servo_debug_frame(self, debug_image, video_image=None):
        """保存当前调试图，同时把关键调试拼图追加到视觉伺服视频。"""
        image_saved = save_image_to_path(self.visual_servo_debug_path, debug_image)
        video_frame = video_image if video_image is not None else debug_image
        video_saved = self.visual_servo_debug_recorder.write(video_frame)
        return image_saved and (video_saved or not self.visual_servo_debug_video_enabled)

    def save_visual_board_debug_frame(self, debug_image, video_image=None):
        """保存托盘当前调试图，同时把关键调试拼图追加到托盘伺服视频。"""
        image_saved = save_image_to_path(self.visual_board_debug_path, debug_image)
        video_frame = video_image if video_image is not None else debug_image
        video_saved = self.visual_board_debug_recorder.write(video_frame)
        return image_saved and (video_saved or not self.visual_board_debug_video_enabled)

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("图像转换失败: %s" % e)

    def make_high_rough_servo_pose(self, px, py, image_shape):
        """深度不可用时沿用旧像素比例粗估，返回视觉伺服观察位姿。"""
        h, w = image_shape[:2]
        center_x = w / 2.0
        center_y = h / 2.0
        predicted_x = self.shooting_angle[0] + (float(py) - center_y) * self.high_rough_y_mm_per_pixel
        predicted_y = self.shooting_angle[1] + (float(px) - center_x) * self.high_rough_x_mm_per_pixel
        return [
            float(predicted_x),
            float(predicted_y),
            float(self.servo_look_z),
            float(self.shooting_angle[3]),
            float(self.shooting_angle[4]),
            float(self.shooting_angle[5]),
        ]

    def query_world_position_from_depth(self, px, py):
        """调用深度相机服务，把 Gemini335 对齐像素转成世界坐标点。"""
        pixel_x = int(round(float(px)))
        pixel_y = int(round(float(py)))
        resp = self.pixel2world_client(pixel2worldRequest(pixel_x, pixel_y))
        world_position = np.array(list(resp.world_position), dtype=float)
        if world_position.shape[0] != 3:
            raise ValueError(f"深度服务返回长度错误: {world_position.tolist()}")
        if not np.all(np.isfinite(world_position)):
            raise ValueError(f"深度服务返回非有限坐标: {world_position.tolist()}")
        if np.allclose(world_position, np.zeros(3), atol=1e-6):
            raise ValueError("深度服务返回失败占位坐标 [0, 0, 0]")
        return world_position

    def make_servo_pose_from_camera_world_point(self, world_position):
        """根据目标世界点生成相机中心对准目标的 MoveL 工具位姿。"""
        desired_camera_xy = np.array([
            float(world_position[0]),
            float(world_position[1]),
        ], dtype=float)
        tool_rotation = rpy_degrees_to_rotation_matrix(
            self.shooting_angle[3],
            self.shooting_angle[4],
            self.shooting_angle[5],
        )
        camera_offset_in_tool = np.array(self.T_wrist2camera_mm[:3, 3], dtype=float)
        camera_offset_in_base = tool_rotation @ camera_offset_in_tool
        tool_position = np.array([
            desired_camera_xy[0] - camera_offset_in_base[0],
            desired_camera_xy[1] - camera_offset_in_base[1],
            float(self.servo_look_z),
        ], dtype=float)
        if not np.all(np.isfinite(tool_position)):
            raise ValueError(f"计算出的工具位姿坐标无效: {tool_position.tolist()}")
        return [
            float(tool_position[0]),
            float(tool_position[1]),
            float(tool_position[2]),
            float(self.shooting_angle[3]),
            float(self.shooting_angle[4]),
            float(self.shooting_angle[5]),
        ]

    def make_depth_first_servo_pose(self, px, py, image_shape, label):
        """优先用深度相机生成观察位；失败时回退旧高位粗估。"""
        try:
            world_position = self.query_world_position_from_depth(px, py)
            servo_pose = self.make_servo_pose_from_camera_world_point(world_position)
            print(
                f"{label}深度观察位",
                "像素", (float(px), float(py)),
                "世界点", world_position.tolist(),
                "pose", servo_pose,
            )
            return servo_pose, "depth", world_position.tolist()
        except Exception as exc:
            self.print_yellow_warning(f"{label}深度定位失败，回退旧粗估: {exc}")
            servo_pose = self.make_high_rough_servo_pose(px, py, image_shape)
            print(
                f"{label}回退观察位",
                "像素", (float(px), float(py)),
                "pose", servo_pose,
            )
            return servo_pose, "fallback", []

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

    def build_pick_target_lists(self):
        """用托盘格点像素生成任务分配目标坐标，深度失败时自动回退旧粗估。"""
        if self.board_grid_points is None or self.board_grid_image_shape is None:
            raise RuntimeError("尚未缓存托盘格点，无法生成任务分配目标坐标")

        category_index_by_name = {
            category: index
            for index, category in enumerate(BLOCK_CATEGORY_NAMES)
        }
        out_list = [[] for _ in BLOCK_CATEGORY_NAMES]
        for item in self.pick_list:
            col = float(item[0])
            row = float(item[1])
            category = str(item[3])
            target_index = int(item[4])
            target_point = interpolate_grid_point(self.board_grid_points, row, col)
            px = float(target_point[0])
            py = float(target_point[1])
            servo_pose, source, _ = self.make_depth_first_servo_pose(
                px,
                py,
                self.board_grid_image_shape,
                f"托盘目标{target_index}",
            )
            category_index = category_index_by_name.get(category)
            if category_index is None:
                self.print_yellow_warning(f"未知托盘目标类别，已跳过: {category}")
                continue
            out_list[category_index].append([servo_pose[0], servo_pose[1], target_index])
            print("托盘任务分配目标", "序号", target_index, "来源", source, "坐标", servo_pose[:3])
        return out_list

    def get_cube_pos(self, req):
        img_bgr1 = self.latest_image
        if img_bgr1 is None:
            rospy.logwarn("没有可用图像，无法识别方块")
            return GetTargetPosResponse([0])
        # 裁剪边距：正数向外扩展 YOLO 框，负数向内收缩 YOLO 框，单位是像素。
        crop_margin = 8
        cube_count=[0,0,0,0,0,0,0]#记录每个方块的放置个数
        img_bgr = img_bgr1
        template_geometry = load_template_geometry("high")#每次服务调用读取一次模板几何配置，便于标定后直接生效
        cube_list=[]#检测到方块储存在这个列表中
        for i in range(7):
            cube_list.append([])
        blocks, img_bgr2 = detect_blocks_in_image(
            img_bgr,
            self.model,
            template_geometry=template_geometry,
            crop_margin=crop_margin,
            save_mask_overlay=self.save_top_surface_mask_vis,
        )
        for block in blocks:
            category = block["category"]
            cube_count=get_all_cube(category,cube_count)#读取每种方块的个数
            px = block["px"]
            py = block["py"]
            print(f"{category}坐标{px,py}")

            #获取角度，不用看
            theta = block["theta"]

            servo_pose, source, world_position = self.make_depth_first_servo_pose(
                px,
                py,
                img_bgr1.shape,
                f"方块{category}",
            )
            cam_point3d = servo_pose[:3]
            print(
                "方块伺服观察位",
                "类别", category,
                "来源", source,
                "世界点", world_position,
                "pose", servo_pose,
            )

            cube_list=make_list(cube_list,category,cam_point3d,theta)

        # cv2.imshow('img_bgr2', img_bgr2)
        # cv2.waitKey(2000)
        # cv2.destroyAllWindows()
        save_image_to_path('/home/zhl/桌面/cube_pos_image.jpg', img_bgr2)
        if self.save_top_surface_mask_vis:
            mask_vis_img = blocks[-1].get("mask_overlay") if blocks else img_bgr2
            if mask_vis_img is None:
                mask_vis_img = img_bgr2
            if save_image_to_path(self.top_surface_mask_vis_path, mask_vis_img):
                rospy.loginfo("上表面掩码可视化已保存: %s" % self.top_surface_mask_vis_path)
            else:
                rospy.logwarn("上表面掩码可视化保存失败: %s" % self.top_surface_mask_vis_path)
        # print("cubelist检查:",cube_list)
        if req.num==-2:
            test = CDLL(JINJIE_LIB_PATH)
            test.IDBS.restype = c_char_p
            place_order=list(map(int,input("输入进阶任务顺序").split()))
            self.pick_list,fill_line=self.get_put_table(cube_count,place_order,test)
            print("实际极限填满",fill_line,"行")
            print(self.pick_list)
            for idx, sublist in enumerate(self.pick_list):
                sublist.append(idx)
        try:
            pick_list2 = self.build_pick_target_lists()
        except Exception as exc:
            rospy.logerr("生成托盘任务分配目标失败: %s" % exc)
            print(f"\033[91m生成托盘任务分配目标失败: {exc}\033[0m")
            return GetTargetPosResponse([0])
        block=cube_pocess(cube_list)
        self.results = optimize_block_assignment(block, pick_list2,self.pick_list)
        
        return GetTargetPosResponse([len(self.pick_list)])
    
    def get_board_pos(self, req):
        self.board_grid_points = None
        self.board_grid_image_shape = None
        self.board_grid_image = None
        img_bgr1 = self.latest_image
        if img_bgr1 is None:
            rospy.logwarn("没有可用图像，无法识别托盘")
            return GetTargetPosResponse([0.0])
        board_bgr = img_bgr1
        if board_bgr is None:
            print("没有图片")
            return GetTargetPosResponse([0.0])
        board_debug_image = board_bgr
        try:
            detect_result = board_grid_detect(board_bgr, debug_path=None)
            board_debug_image = detect_result.get("debug_image")
            if board_debug_image is None:
                board_debug_image = board_bgr
            if not detect_result["found"]:
                save_image_to_path(self.visual_board_grid_debug_path, board_debug_image)
                raise RuntimeError(detect_result["message"])

            self.board_grid_points = detect_result["grid_points"]
            self.board_grid_image_shape = board_bgr.shape[:2]
            self.board_grid_image = board_bgr.copy()
            center_point = (board_bgr.shape[1] / 2.0, board_bgr.shape[0] / 2.0)
            board_debug_image = draw_grid_debug(board_bgr, self.board_grid_points, center_point=center_point)
            save_image_to_path(self.visual_board_grid_debug_path, board_debug_image)

            self.board_theta = self.compute_board_theta_from_grid_points(self.board_grid_points)
            print(f"托盘像素角度: {self.board_theta:.2f}")
        except Exception as exc:
            self.board_grid_points = None
            self.board_grid_image_shape = None
            self.board_grid_image = None
            rospy.logerr("托盘识别失败: %s" % exc)
            print("\033[91m托盘识别失败，请重新识别。\033[0m")
            return GetTargetPosResponse([0.0])

        # save_image_to_path('/home/zhl/桌面/board_bgr.jpg', board_bgr)
        
        return GetTargetPosResponse([1.0])
    
    def get_cube_location1(self, req):
        if self.results is None:
            rospy.logwarn("尚未完成方块识别，无法返回抓取位置")
            return GetTargetPosResponse(array=[0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        if req.num < -1 or req.num >= len(self.pick_list):
            rospy.logwarn("方块序号越界: %s，当前可用数量: %s" % (req.num, len(self.pick_list)))
            return GetTargetPosResponse(array=[0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

        block2, selected, orig_dists, opt_dists, dist_saving, time_used, total_time, orig_total, opt_total = self.results
        if req.num == -1:
            pick_cube = self.pick_list[0]
            x,y,z,t = calc_cube_location(pick_cube,block2,False)
        else:
            pick_cube = self.pick_list[req.num]
            x,y,z,t = calc_cube_location(pick_cube,block2,True)
        xuanzhuan_angle = pick_cube[2] - t + self.board_theta
        #处理旋转角度超过360°的情况（比较极端）-- (-360 ~ 360)
        if xuanzhuan_angle > 360:
            xuanzhuan_angle -= 360
        elif xuanzhuan_angle < -360:
            xuanzhuan_angle += 360
        #舵机为180°舵机，处理舵机旋转超过180°的情况，同时也使总所需旋转角更小 -- (-180 ~ 180)
        if xuanzhuan_angle > 180:
            xuanzhuan_angle -= 360
        elif xuanzhuan_angle < -180:
            xuanzhuan_angle += 360

        if pick_cube[3]=='z_green' or pick_cube[3]=='z_blue' or pick_cube[3]=='line':
            #优化旋转方向  -- (-90 ~ 90)
            if xuanzhuan_angle > 90:
                xuanzhuan_angle -= 180
            elif xuanzhuan_angle < -90:
                xuanzhuan_angle += 180
        elif pick_cube[3]=='square':
            #处理多余旋转 -- (-90 ~ 90)
            if xuanzhuan_angle>0:
                xuanzhuan_angle = (xuanzhuan_angle) % 90
            else:
                xuanzhuan_angle = (xuanzhuan_angle) % -90
            #优化旋转方向 -- (-45 ~ 45)
            if xuanzhuan_angle > 45:
                xuanzhuan_angle -= 90
            elif xuanzhuan_angle < -45:
                xuanzhuan_angle += 90
        category_code = category_to_code(pick_cube[3])
        print("方块种类", pick_cube[3], "目标角度：", pick_cube[2],"识别到的角度：", t,"需要旋转的角度：", xuanzhuan_angle,"位置",x,y,z)
        return GetTargetPosResponse(array=[x,y,z,t,xuanzhuan_angle,float(category_code)])
    
    def get_put_pose(self, req):
        if req.num < 0 or req.num >= len(self.pick_list):
            rospy.logwarn("摆放序号越界: %s，当前可用数量: %s" % (req.num, len(self.pick_list)))
            return GetTargetPosResponse(array=[])

        if self.board_grid_points is None or self.board_grid_image_shape is None:
            rospy.logwarn("尚未缓存托盘格点，无法计算托盘粗观察位")
            print("\033[91m尚未缓存托盘格点，请先在高位完成托盘识别。\033[0m")
            return GetTargetPosResponse(array=[])

        pick_cube = self.pick_list[req.num]
        col = float(pick_cube[0])
        row = float(pick_cube[1])

        try:
            grid_points = self.board_grid_points
            target_point = interpolate_grid_point(grid_points, row, col)
            px = float(target_point[0])
            py = float(target_point[1])

            h, w = self.board_grid_image_shape
            center_x = w / 2.0
            center_y = h / 2.0
            rough_pose, source, world_position = self.make_depth_first_servo_pose(
                px,
                py,
                self.board_grid_image_shape,
                f"托盘摆放{req.num}",
            )

            if self.board_grid_image is not None:
                debug_image = draw_grid_debug(
                    self.board_grid_image,
                    grid_points,
                    target_point=target_point,
                    center_point=(center_x, center_y),
                )
                save_image_to_path(self.visual_board_grid_debug_path, debug_image)
            print(
                "托盘粗观察位",
                "序号", req.num,
                "目标行列", (row, col),
                "目标像素", (px, py),
                "来源", source,
                "世界点", world_position,
                "pose", rough_pose,
            )
            return GetTargetPosResponse(array=rough_pose)
        except Exception as exc:
            rospy.logerr("托盘粗观察位计算失败: %s" % exc)
            print(f"\033[91m托盘粗观察位计算失败: {exc}\033[0m")
            return GetTargetPosResponse(array=[])

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

    def visual_servo_result_to_response(self, result):
        """把内部字典转换成统一 ROS 服务响应。"""
        return VisualServoOffsetResponse(
            found=result["found"],
            target_type=result["target_type"],
            category=result["category"],
            px=result["px"],
            py=result["py"],
            dx_px=result["dx_px"],
            dy_px=result["dy_px"],
            theta=result["theta"],
            score=result["score"],
            message=result["message"],
        )

    def handle_block_visual_servo_request(self, req):
        """方块视觉伺服服务分支。

        这里是胶水代码：只解析服务请求，然后调用真正的方块识别函数。
        真正要改识别算法时，去填 detect_block_visual_offset，不要在这里写图像处理细节。
        """
        expected_category = (req.expected_category or "").strip()
        template_options = self.parse_block_template_match_options(req)
        return self.detect_block_visual_offset(expected_category, **template_options)

    def parse_block_template_match_options(self, req):
        """解析方块低位模板匹配先验，用于缩小模板角度和位置搜索范围。"""
        template_profile = (getattr(req, "template_profile", "") or "low").strip() or "low"
        angle_step = float(getattr(req, "angle_step_deg", 1.0) or 1.0)
        if angle_step <= 0:
            angle_step = 1.0

        options = {
            "template_profile": template_profile,
            "angle_step": angle_step,
            "angle_center": None,
            "angle_window": None,
            "search_center": None,
            "search_radius": None,
        }

        if bool(getattr(req, "use_angle_prior", False)):
            options["angle_center"] = float(getattr(req, "angle_center_deg", 0.0))
            options["angle_window"] = max(0.0, float(getattr(req, "angle_window_deg", 0.0)))

        if bool(getattr(req, "use_position_prior", False)):
            options["search_center"] = (
                float(getattr(req, "search_center_x", 0.0)),
                float(getattr(req, "search_center_y", 0.0)),
            )
            options["search_radius"] = max(0.0, float(getattr(req, "search_radius_px", 0.0)))

        return options

    def make_block_angle_prior_message(self, angle_center, angle_window, angle_step):
        """生成低位模板角度先验摘要，便于确认 competition.py 的 t 已通信到图像节点。"""
        if angle_center is None or angle_window is None:
            return "角度先验: 未启用"
        return (
            f"角度先验: center={float(angle_center):.1f}, "
            f"window={float(angle_window):.1f}, step={float(angle_step):.1f}"
        )

    def handle_board_visual_servo_request(self, req):
        """托盘视觉伺服服务分支。

        这里是胶水代码：只解析目标格点 row/col，然后调用真正的托盘识别函数。
        真正要改托盘识别或目标点计算时，去填 detect_board_visual_offset。
        """
        return self.detect_board_visual_offset(float(req.row), float(req.col))

    def make_unknown_visual_target_result(self, raw_target_type):
        """请求的 target_type 不是 block/board 时，返回统一失败结果。"""
        target_type = (raw_target_type or "").strip().lower()
        return self.make_visual_servo_result(
            found=False,
            target_type=target_type,
            message=f"未知视觉伺服目标类型: {raw_target_type}，应为 block 或 board",
        )

    def detect_board_visual_offset(self, row, col):
        """识别低位托盘目标点相对相机中心的像素偏差。

        这里只负责图像识别和像素偏差计算，不控制机械臂。
        低位时托盘通常不完整入画，因此只在画面中心小 ROI 内找托盘圆点。
        row/col 支持整数和 .5，小数目标会用相邻圆点平均成虚拟目标点。
        """
        img_bgr1 = self.latest_image
        if img_bgr1 is None:
            return self.make_visual_servo_result(
                found=False,
                target_type="board",
                message="没有可用图像",
            )

        try:
            h, w = img_bgr1.shape[:2]
            new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeff, (w, h), 1, (w, h))
            board_bgr = cv2.undistort(img_bgr1, self.camera_matrix, self.dist_coeff, None, new_camera_mtx)#去畸变
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
                debug_path=self.visual_board_debug_path,
                row=row,
                col=col,
            )
            debug_image = detect_result.get("debug_image")
            debug_panel = detect_result.get("debug_panel")
            if not detect_result["found"]:
                self.save_visual_board_debug_frame(debug_image, debug_panel)
                return self.make_visual_servo_result(
                    found=False,
                    target_type="board",
                    message=detect_result["message"],
                )

            target_point = detect_result["point"]
            dx_px = float(target_point[0] - center_x)
            dy_px = float(target_point[1] - center_y)
            self.save_visual_board_debug_frame(debug_image, debug_panel)

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
            rospy.logerr("托盘视觉伺服目标检测失败: %s" % exc)
            print("\033[91m托盘视觉伺服目标检测失败，请重新识别。\033[0m")
            return self.make_visual_servo_result(
                found=False,
                target_type="board",
                message=str(exc),
            )

    def detect_block_visual_offset(
        self,
        expected_category="",
        template_profile="low",
        angle_step=1.0,
        angle_center=None,
        angle_window=None,
        search_center=None,
        search_radius=None,
    ):
        """识别方块目标相对相机中心的像素偏差。

        template_profile="low" 时使用高位传来的类别和角度先验：
        按低位模板尺寸在画面中心生成旋转 ROI，去白背景后直接模板匹配。
        其它 profile 仍保留旧 YOLO 检测流程，供高位或历史调试调用。
        """
        angle_prior_message = self.make_block_angle_prior_message(angle_center, angle_window, angle_step)
        img_bgr1 = self.latest_image
        if img_bgr1 is None:
            return self.make_visual_servo_result(
                found=False,
                target_type="block",
                message=f"没有可用图像，{angle_prior_message}",
            )

        try:
            img_bgr = undistort_bgr_image(img_bgr1, self.camera_matrix, self.dist_coeff)
            h, w = img_bgr.shape[:2]
            center_x = w / 2.0
            center_y = h / 2.0
            expected_category = normalize_category_name((expected_category or "").strip())
            template_geometry = load_template_geometry(template_profile)

            if template_profile == "low":
                if not expected_category:
                    debug_image = np.copy(img_bgr)
                    self.save_visual_servo_debug_frame(debug_image)
                    return self.make_visual_servo_result(
                        found=False,
                        target_type="block",
                        message=f"低位先验 ROI 缺少高位类别，{angle_prior_message}",
                    )
                if angle_center is None or angle_window is None:
                    debug_image = np.copy(img_bgr)
                    self.save_visual_servo_debug_frame(debug_image)
                    return self.make_visual_servo_result(
                        found=False,
                        target_type="block",
                        category=expected_category,
                        message=f"低位先验 ROI 缺少高位角度，{angle_prior_message}",
                    )

                target_block = detect_block_with_high_prior_roi(
                    img_bgr,
                    template_geometry=template_geometry,
                    category=expected_category,
                    high_theta_deg=angle_center,
                    angle_window=angle_window,
                    angle_step=angle_step,
                    roi_expand_px=self.block_low_roi_expand_px,
                    white_s_max=self.block_low_white_s_max,
                    white_v_min=self.block_low_white_v_min,
                    min_foreground_area=self.block_low_min_foreground_area,
                )
                debug_image = target_block.get("debug_image", np.copy(img_bgr))
                debug_panel = target_block.get("debug_panel")
                cv2.drawMarker(
                    debug_image,
                    (int(center_x), int(center_y)),
                    (255, 0, 0),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=24,
                    thickness=2,
                )

                if not target_block["found"]:
                    self.save_visual_servo_debug_frame(debug_image, debug_panel)
                    return self.make_visual_servo_result(
                        found=False,
                        target_type="block",
                        category=expected_category,
                        message=f"{target_block['message']}，{angle_prior_message}",
                    )

                px = target_block["px"]
                py = target_block["py"]
                theta = target_block["theta"]
                dx_px = px - center_x
                dy_px = py - center_y
                cv2.line(
                    debug_image,
                    (int(center_x), int(center_y)),
                    (int(px), int(py)),
                    (255, 0, 0),
                    1,
                )
                self.save_visual_servo_debug_frame(debug_image, debug_panel)
                return self.make_visual_servo_result(
                    found=True,
                    target_type="block",
                    category=target_block["category"],
                    px=float(px),
                    py=float(py),
                    dx_px=float(dx_px),
                    dy_px=float(dy_px),
                    theta=float(theta),
                    score=float(target_block["score"]),
                    message=f"低位先验 ROI 模板匹配成功，{angle_prior_message}",
                )

            blocks, debug_image = detect_blocks_in_image(
                img_bgr,
                self.model,
                template_geometry=template_geometry,
                crop_margin=8,
                save_mask_overlay=self.save_top_surface_mask_vis,
                angle_step=angle_step,
                angle_center=angle_center,
                angle_window=angle_window,
                search_center=search_center,
                search_radius=search_radius,
                expected_category=expected_category,
            )

            cv2.drawMarker(
                debug_image,
                (int(center_x), int(center_y)),
                (255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=24,
                thickness=2,
            )

            if not blocks:
                category_msg = expected_category if expected_category else "任意类别"
                self.save_visual_servo_debug_frame(debug_image)
                return self.make_visual_servo_result(
                    found=False,
                    target_type="block",
                    category=expected_category,
                    message=f"没有检测到目标方块，目标类别: {category_msg}，{angle_prior_message}",
                )

            # 低位对准时相机中心附近的同类方块才是目标。
            target_block = min(
                blocks,
                key=lambda block: (block["px"] - center_x) ** 2 + (block["py"] - center_y) ** 2,
            )
            px = target_block["px"]
            py = target_block["py"]
            theta = target_block["theta"]
            dx_px = px - center_x
            dy_px = py - center_y
            score = target_block["score"]

            cv2.drawMarker(
                debug_image,
                (int(px), int(py)),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=24,
                thickness=2,
            )
            cv2.line(
                debug_image,
                (int(center_x), int(center_y)),
                (int(px), int(py)),
                (255, 0, 0),
                1,
            )
            cv2.putText(
                debug_image,
                f"{target_block['category']} dx={dx_px:.1f}px dy={dy_px:.1f}px theta={theta:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            self.save_visual_servo_debug_frame(debug_image)
            message = (
                f"方块视觉伺服识别成功，共 {len(blocks)} 个候选，"
                f"已选择离画面中心最近的 {target_block['category']}，{angle_prior_message}"
            )
            return self.make_visual_servo_result(
                found=True,
                target_type="block",
                category=target_block["category"],
                px=float(px),
                py=float(py),
                dx_px=float(dx_px),
                dy_px=float(dy_px),
                theta=float(theta),
                score=float(score),
                message=message,
            )
        except Exception as exc:
            rospy.logerr("视觉伺服目标检测失败: %s" % exc)
            return self.make_visual_servo_result(
                found=False,
                target_type="block",
                message=f"{exc}，{angle_prior_message}",
            )

    def get_visual_servo_offset(self, req):
        """统一视觉伺服偏差服务入口。

        请求格式：
        - target_type="block"：识别当前方块目标，expected_category 可选。
        - target_type="board"：识别托盘目标格点，需要 row/col。

        返回格式固定为 VisualServoOffsetResponse。
        本函数只做分流，不写任何具体图像识别算法。
        """
        target_type = (req.target_type or "").strip().lower()
        if target_type == VISUAL_TARGET_BLOCK:
            result = self.handle_block_visual_servo_request(req)
        elif target_type == VISUAL_TARGET_BOARD:
            result = self.handle_board_visual_servo_request(req)
        else:
            result = self.make_unknown_visual_target_result(req.target_type)
        return self.visual_servo_result_to_response(result)

    def get_visual_board_offset(self, req):
        """旧托盘偏差服务兼容包装；正式主流程改用 get_visual_servo_offset。"""
        result = self.detect_board_visual_offset(req.row, req.col)
        return VisualBoardOffsetResponse(
            found=result["found"],
            px=result["px"],
            py=result["py"],
            dx_px=result["dx_px"],
            dy_px=result["dy_px"],
            message=result["message"],
        )
    
    def get_put_table(self,cube_count,place_order,test):
        result = test.IDBS(cube_count[0], cube_count[1], cube_count[2], cube_count[3], cube_count[4], cube_count[5],
                        cube_count[6],place_order[0],place_order[1],place_order[2],place_order[3],place_order[4],place_order[5],place_order[6])  # 调用库里的函数sum，求和函数
        result = result.decode('gbk')
        cube_list0 = result.split(',')
        length = len(cube_list0) // 4
        cube_list = []
        for i in range(length):
            cube_list.append([])
            cube_list[i].append(float(cube_list0[i * 4 + 2])+0.5)
            cube_list[i].append(float(cube_list0[i * 4 + 3])+0.5)
            cube_list[i].append(float(cube_list0[i * 4 + 1]))
            cube_name=normalize_category_name(cube_list0[i * 4])
            cube_list[i].append(cube_name)
        # print(cube_list, cube_list0[-1])  # 打印结果
        cube_sum = 0
        for i in cube_count:
            cube_sum = cube_sum + i * 4
        level = cube_sum // 10
        print("极限填满行", level)
        return cube_list,cube_list0[-1]

    def get_visual_target_offset(self, req):
        """旧方块偏差服务兼容包装；正式主流程改用 get_visual_servo_offset。"""
        result = self.detect_block_visual_offset(req.expected_category)
        return VisualTargetOffsetResponse(
            found=result["found"],
            category=result["category"],
            px=result["px"],
            py=result["py"],
            dx_px=result["dx_px"],
            dy_px=result["dy_px"],
            theta=result["theta"],
            score=result["score"],
            message=result["message"],
        )

if __name__ == "__main__":
    rospy.init_node("image_processor")
    processor = ImageProcessor()
    rospy.spin()
