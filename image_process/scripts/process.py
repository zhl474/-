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
    put_fenlei,
    cube_pocess,
    optimize_block_assignment,
    get_cube_location as calc_cube_location,
    get_put_pose as calc_put_pose,
)
from image_process_lib.template_config import load_template_geometry
from image_process_lib.point_calibration import ArmCalibrator
from image_process_lib.board_detect import detect_nearest_board_dot_in_roi
from image_process_lib.single_block_detector import (
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
DETECTION_MODEL_PATH = "/home/zhl/SingleArmTetris/SingleArmTetris/src/competition/model/best5.14.pt"
JINJIE_LIB_PATH = "/home/zhl/SingleArmTetris/SingleArmTetris/src/jinjie/jinjie_libtetris.so"

VISUAL_TARGET_BLOCK = "block"
VISUAL_TARGET_BOARD = "board"

# 高位全场拍摄位姿。competition.py 里也有同一份值，后面应通过服务返回动态规划结果来彻底统一。
BASE_SHOOTING_ANGLE = [-250.4151306152343, 22.14801216125488, 380.3343505859375, -180, 0, 90]


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

class ImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.latest_image = None
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
        self.calibrator = ArmCalibrator()
        self.save_top_surface_mask_vis = rospy.get_param("~save_top_surface_mask_vis", False)
        self.top_surface_mask_vis_path = rospy.get_param(
            "~top_surface_mask_vis_path",
            "/home/zhl/桌面/top_surface_masks.jpg"
        )
        self.visual_servo_debug_path = rospy.get_param(
            "~visual_servo_debug_path",
            "/home/zhl/桌面/视觉伺服当前检测.jpg"
        )
        self.visual_board_debug_path = rospy.get_param(
            "~visual_board_debug_path",
            "/home/zhl/桌面/托盘视觉伺服当前检测.jpg"
        )
        self.board_low_roi_half_size = rospy.get_param("~board_low_roi_half_size", 120)
        self.board_low_blackhat_kernel_size = rospy.get_param("~board_low_blackhat_kernel_size", 15)
        self.board_low_min_dot_area = rospy.get_param("~board_low_min_dot_area", 20)
        self.board_low_max_dot_area = rospy.get_param("~board_low_max_dot_area", 250)
        self.board_low_min_dot_circularity = rospy.get_param("~board_low_min_dot_circularity", 0.35)
        self.board_low_max_dot_aspect_ratio = rospy.get_param("~board_low_max_dot_aspect_ratio", 1.8)

        # 高位全场识别后，用像素偏差粗估方块机械臂坐标。
        self.high_rough_x_mm_per_pixel = 0.5
        self.high_rough_y_mm_per_pixel = 0.5

        self.pixel2world_client = rospy.ServiceProxy("get_world_pos",pixel2world)
        self.pixel2world_client.wait_for_service()

        self.shooting_angle = list(BASE_SHOOTING_ANGLE)
        self.pick_list = build_base_pick_list(include_index=True)
        rospy.loginfo("图像处理服务已启动")
    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("图像转换失败: %s" % e)

    def get_cube_pos(self, req):
        img_bgr1 = self.latest_image
        if img_bgr1 is None:
            rospy.logwarn("没有可用图像，无法识别方块")
            return GetTargetPosResponse([0])
        # 裁剪边距：正数向外扩展 YOLO 框，负数向内收缩 YOLO 框，单位是像素。
        crop_margin = 8
        cube_count=[0,0,0,0,0,0,0]#记录每个方块的放置个数
        img_bgr = undistort_bgr_image(img_bgr1, self.camera_matrix, self.dist_coeff)#去畸变
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

            cam_point3d = [0,0,0]
################################################       粗定位，移动到大概位置即可      #########################################################
            h, w = img_bgr1.shape[:2]
            center_x = w / 2.0
            center_y = h / 2.0
            predicted_x = self.shooting_angle[0] + (py - center_y) * self.high_rough_y_mm_per_pixel  # 机械臂和相机坐标是反的
            predicted_y = self.shooting_angle[1] + (px - center_x) * self.high_rough_x_mm_per_pixel  # 机械臂和相机坐标是反的
            predicted_z = 200
            cam_point3d[0]=predicted_x
            cam_point3d[1]=predicted_y
            cam_point3d[2]=predicted_z
################################################       深度相机法        #########################################################
            # resp = self.pixel2world_client(pixel2worldRequest(px,py))
            # cam_point3d = list(resp.world_position)

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
        pick_list2=put_fenlei(self.pick_list,self.shooting_angle,self.calibrator)
        block=cube_pocess(cube_list)
        self.results = optimize_block_assignment(block, pick_list2,self.pick_list)
        
        return GetTargetPosResponse([len(self.pick_list)])
    
    def get_board_pos(self, req):
        img_bgr1 = self.latest_image
        if img_bgr1 is None:
            rospy.logwarn("没有可用图像，无法识别托盘")
            return GetTargetPosResponse([0.0])
        h, w = img_bgr1.shape[:2]
        board_bgr = img_bgr1
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeff, (w, h), 1, (w, h))
        board_bgr = cv2.undistort(img_bgr1, self.camera_matrix, self.dist_coeff, None, new_camera_mtx)#去畸变
        if board_bgr is None:
            print("没有图片")
            return GetTargetPosResponse([0.0])
        try:
            self.calibrator.board_detector(board_bgr,self.pixel2world_client)
            self.calibrator.calibrate()#托盘识别结束
        except Exception as exc:
            rospy.logerr("托盘识别失败: %s" % exc)
            print("\033[91m托盘识别失败，请重新识别。\033[0m")
            return GetTargetPosResponse([0.0])

        save_image_to_path('/home/zhl/桌面/board_bgr.jpg', board_bgr)
        
        return GetTargetPosResponse([1.0])
    
    def get_cube_location1(self, req):
        if self.results is None:
            rospy.logwarn("尚未完成方块识别，无法返回抓取位置")
            return GetTargetPosResponse(array=[0.0, 0.0, 0.0, 0.0, 0.0])
        if req.num < -1 or req.num >= len(self.pick_list):
            rospy.logwarn("方块序号越界: %s，当前可用数量: %s" % (req.num, len(self.pick_list)))
            return GetTargetPosResponse(array=[0.0, 0.0, 0.0, 0.0, 0.0])

        block2, selected, orig_dists, opt_dists, dist_saving, time_used, total_time, orig_total, opt_total = self.results
        if req.num == -1:
            pick_cube = self.pick_list[0]
            x,y,z,t = calc_cube_location(pick_cube,block2,False)
        else:
            pick_cube = self.pick_list[req.num]
            x,y,z,t = calc_cube_location(pick_cube,block2,True)
        xuanzhuan_angle = pick_cube[2] - t + self.calibrator.board_theta
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
        print("方块种类", pick_cube[3], "目标角度：", pick_cube[2],"识别到的角度：", t,"需要旋转的角度：", xuanzhuan_angle,"位置",x,y,z)
        return GetTargetPosResponse(array=[x,y,z,t,xuanzhuan_angle])
    
    def get_put_pose(self, req):
        if req.num < 0 or req.num >= len(self.pick_list):
            rospy.logwarn("摆放序号越界: %s，当前可用数量: %s" % (req.num, len(self.pick_list)))
            return GetTargetPosResponse(array=[])

        pick_cube = self.pick_list[req.num]
        if(pick_cube[3]=='square'):
            # put_pose=calc_put_pose(pick_cube[0]-0.035,pick_cube[1]+0.06,self.shooting_angle,self.calibrator)
            put_pose=calc_put_pose(pick_cube[0],pick_cube[1],self.shooting_angle,self.calibrator)
        else:
            # put_pose=calc_put_pose(pick_cube[0]-0.03,pick_cube[1]+0.03,self.shooting_angle,self.calibrator)
            put_pose=calc_put_pose(pick_cube[0],pick_cube[1],self.shooting_angle,self.calibrator)
        return GetTargetPosResponse(array=put_pose)

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
        """识别低位托盘圆点相对相机中心的像素偏差。

        这里只负责图像识别和像素偏差计算，不控制机械臂。
        低位时托盘通常不完整入画，因此只在画面中心小 ROI 内找最近的托盘圆点。
        row/col 只保留给服务兼容和调试图显示，不参与低位圆点选择。
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
            if not detect_result["found"]:
                return self.make_visual_servo_result(
                    found=False,
                    target_type="board",
                    message=detect_result["message"],
                )

            target_point = detect_result["point"]
            dx_px = float(target_point[0] - center_x)
            dy_px = float(target_point[1] - center_y)

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
                message="低位托盘圆点识别成功",
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

        低位视觉伺服和高位全场识别使用同一套流程：
        YOLO 检测候选框，按 expected_category 过滤类别，裁剪候选区域，
        分割上表面，再用模板匹配得到目标点和角度。
        如果画面中有多个同类方块，选择离画面中心最近的一个。
        angle_center/angle_window 和 search_center/search_radius 是模板匹配先验，
        用来缩小旋转角和局部位置搜索范围。
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
                save_image_to_path(self.visual_servo_debug_path, debug_image)
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
            save_image_to_path(self.visual_servo_debug_path, debug_image)
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
