#!/home/zhl/fr3env/fr3env/bin/python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import yaml
from ultralytics import YOLO

from image_process_lib.Place_optimization import get_all_cube, make_list, put_fenlei,cube_pocess,optimize_block_assignment,get_cube_location,get_put_pose
from image_process_lib.template_config import load_template_geometry
from image_process_lib.point_calibration import ArmCalibrator, x_predict,y_predict,z_predict
from image_process_lib.board_detect import board_detect, board_grid_detect, draw_grid_debug, interpolate_grid_point
from image_process_lib.single_block_detector import (
    detect_blocks_in_image,
    detect_single_block_in_image,
    normalize_category_name,
    undistort_bgr_image,
)

from image_process.srv import GetTargetPos, GetTargetPosResponse
from image_process.srv import VisualTargetOffset, VisualTargetOffsetResponse
from image_process.srv import VisualBoardOffset, VisualBoardOffsetResponse
from image_process.srv import VisualServoOffset, VisualServoOffsetResponse
from camera.srv import pixel2world, pixel2worldRequest

from ctypes import * 

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

def make_debug_image_with_mask(debug_image, mask):
    """把检测结果和二值掩码拼在一起，方便现场调阈值。"""
    if debug_image is None or mask is None:
        return debug_image
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return np.hstack((debug_image, mask_bgr))


VISUAL_TARGET_BLOCK = "block"
VISUAL_TARGET_BOARD = "board"


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

        model_path="/home/zhl/SingleArmTetris/SingleArmTetris/src/competition/model/best5.14.pt"
        self.model=YOLO(model_path)
        with open("/home/zhl/SingleArmTetris/SingleArmTetris/src/competition/config/calibration_matrix.yaml") as f:
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

        # 机械臂 X 方向每 1 像素误差对应的开环移动量，单位 mm/px。在shooting_angle拍摄下
        self.HIGH_ROUGH_X_MM_PER_PIXEL = 0.5

        # 机械臂 Y 方向每 1 像素误差对应的开环移动量，单位 mm/px。
        self.HIGH_ROUGH_Y_MM_PER_PIXEL = 0.5

        self.pixel2world_client = rospy.ServiceProxy("get_world_pos",pixel2world)
        self.pixel2world_client.wait_for_service()

        self.shooting_angle = [-250.4151306152343 , 22.14801216125488, 380.3343505859375 , -180, 0, 90]#competition里还有一个
        self.pick_list=[
            [1, 2, 90, 'L_yellow', 0], [4, 1.5, 0, 'T', 1], [7.5, 1, 0, 'line', 2], [9.5, 2, -90, 'z_green', 3], [2, 3, 90, 'L_yellow', 4],
            [4, 3, 180, 'L_blue', 5], [7, 2, 0, 'L_yellow', 6], [1.5, 5, 90, 'T', 7], [3.5, 5, 90, 'z_blue', 8], [5, 4.5, 0, 'z_green', 9],
            [6.5, 3.5, 0, 'square', 10], [9, 5, -90, 'L_blue', 11], [10, 4.5, 90, 'line', 12], [7.5, 5.5, 0, 'square', 13], [1.5, 7, -90, 'z_green', 14],
            [3.5, 7, 90, 'z_blue', 15], [6.5, 7, 90, 'T', 16], [5, 7.5, 90, 'line', 17], [9, 7, 0, 'L_blue', 18], [3, 9, -90, 'L_blue', 19],
            [7, 8.5, 180, 'T', 20],[9.5, 8.5, 0, 'square', 21], [1, 10, 90, 'L_yellow', 22], [4.5, 10, 90, 'T', 23], [6.5, 11, 90, 'z_blue', 24],
            [8.5, 10, 0, 'line', 25], [2.5, 11, 90, 'z_blue', 26], [8, 12, 90, 'L_yellow', 27], [9.5, 12, -90, 'z_green', 28], [1.5, 12.5, 0, 'square', 29],
            [3.5, 13, -90, 'z_green', 30], [5, 12.5, 90, 'line', 31], [6.5, 13, 90, 'z_blue', 32], [9, 14, 180, 'L_blue', 33]
        ]
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
            predicted_x = self.shooting_angle[0]+(py-center_y)*self.HIGH_ROUGH_Y_MM_PER_PIXEL#这机械臂和相机坐标是反的
            predicted_y = self.shooting_angle[1]+(px-center_x)*self.HIGH_ROUGH_X_MM_PER_PIXEL#这机械臂和相机坐标是反的
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
            test = CDLL("/home/zhl/SingleArmTetris/SingleArmTetris/src/jinjie/jinjie_libtetris.so") 
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
        block2, selected, orig_dists, opt_dists, dist_saving, time_used, total_time, orig_total, opt_total = self.results
        if req.num == -1:
            pick_cube = self.pick_list[0]
            x,y,z,t = get_cube_location(pick_cube,block2,False)
        else:
            pick_cube = self.pick_list[req.num]
            x,y,z,t = get_cube_location(pick_cube,block2,True)
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
        pick_cube = self.pick_list[req.num]
        if(pick_cube[3]=='square'):
            # put_pose=get_put_pose(pick_cube[0]-0.035,pick_cube[1]+0.06,self.shooting_angle,self.calibrator)
            put_pose=get_put_pose(pick_cube[0],pick_cube[1],self.shooting_angle,self.calibrator)
        else:
            # put_pose=get_put_pose(pick_cube[0]-0.03,pick_cube[1]+0.03,self.shooting_angle,self.calibrator)
            put_pose=get_put_pose(pick_cube[0],pick_cube[1],self.shooting_angle,self.calibrator)
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
        """解析方块低位模板匹配先验；当前低位临时代码暂不使用这些参数。"""
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
        """识别托盘目标格点相对相机中心的像素偏差。

        下一步真正要填或重写的托盘识别代码就在这里。
        这里只负责图像识别和像素偏差计算，不控制机械臂。
        board_grid_detect 必须识别完整 140 个格点；数量不对时直接返回 found=False。
        row/col 支持整数格点和半格插值，具体排序和插值在 board_detect.py 内部完成。
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
            detect_result = board_grid_detect(board_bgr, debug_path=self.visual_board_debug_path)
            if not detect_result["found"]:
                return self.make_visual_servo_result(
                    found=False,
                    target_type="board",
                    message=detect_result["message"],
                )

            target_point = interpolate_grid_point(
                detect_result["grid_points"],
                row,
                col,
            )
            center_x = w / 2.0
            center_y = h / 2.0
            dx_px = float(target_point[0] - center_x)
            dy_px = float(target_point[1] - center_y)

            debug_image = draw_grid_debug(
                board_bgr,
                detect_result["grid_points"],
                target_point=target_point,
                center_point=(center_x, center_y),
            )
            cv2.putText(
                debug_image,
                f"row={float(row):.2f} col={float(col):.2f} dx={dx_px:.1f} dy={dy_px:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            save_image_to_path(self.visual_board_debug_path, debug_image)
            return self.make_visual_servo_result(
                found=True,
                target_type="board",
                category="board_grid",
                px=float(target_point[0]),
                py=float(target_point[1]),
                dx_px=dx_px,
                dy_px=dy_px,
                theta=0,
                score=1.0,
                message="托盘目标格点识别成功",
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

        下一步真正要填或重写的方块识别代码就在这里。
        当前正式通信接口已经支持 expected_category，但这里仍沿用实测成功的深色旋转矩形检测。
        也就是说：现在还没有真正按 7 类方块类别筛选目标。
        正式多方块同框时，需要后续补上类别过滤、ROI 或距离画面中心优先等目标选择策略。
        """
        img_bgr1 = self.latest_image
        if img_bgr1 is None:
            return self.make_visual_servo_result(
                found=False,
                target_type="block",
                message="没有可用图像",
            )

        try:
            img_bgr = undistort_bgr_image(img_bgr1, self.camera_matrix, self.dist_coeff)
            debug_image = img_bgr.copy()
            h, w = img_bgr.shape[:2]
            center_x = w / 2.0
            center_y = h / 2.0

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            # 相机曝光没调好时，目标会变成灰黑色；用 Otsu 自动阈值找浅色背景上的深色区域。
            _, dark_mask = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )

            # 去掉小噪点并填补目标内部的小孔，保证旋转矩形拟合稳定。
            kernel = np.ones((5, 5), np.uint8)
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(
                dark_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            min_area = max(200.0, float(h * w) * 0.0002)
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

            cv2.drawMarker(
                debug_image,
                (int(center_x), int(center_y)),
                (255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=24,
                thickness=2,
            )

            if not valid_contours:
                save_image_to_path(self.visual_servo_debug_path, make_debug_image_with_mask(debug_image, dark_mask))
                return self.make_visual_servo_result(
                    found=False,
                    target_type="block",
                    message="未检测到深色旋转矩形",
                )

            target_contour = max(valid_contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(target_contour)
            (px, py), (rect_w, rect_h), raw_angle = rect
            if rect_w <= 1e-6 or rect_h <= 1e-6:
                save_image_to_path(self.visual_servo_debug_path, make_debug_image_with_mask(debug_image, dark_mask))
                return self.make_visual_servo_result(
                    found=False,
                    target_type="block",
                    message="深色区域尺寸异常",
                )

            # theta 表示旋转矩形长边相对图像 x 轴的角度，范围约为 -90 到 90 度。
            theta = raw_angle
            if rect_w < rect_h:
                theta += 90.0
            while theta >= 90.0:
                theta -= 180.0
            while theta < -90.0:
                theta += 180.0

            dx_px = px - center_x
            dy_px = py - center_y
            contour_area = cv2.contourArea(target_contour)
            rect_area = rect_w * rect_h
            score = float(contour_area / rect_area) if rect_area > 1e-6 else 0.0

            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(debug_image, [target_contour], -1, (0, 255, 255), 1)
            cv2.drawContours(debug_image, [box], 0, (0, 165, 255), 2)
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
                f"dx={dx_px:.1f}px dy={dy_px:.1f}px theta={theta:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            save_image_to_path(self.visual_servo_debug_path, make_debug_image_with_mask(debug_image, dark_mask))
            message = "检测到深色旋转矩形"
            if expected_category:
                message += f"；注意：当前尚未按类别 {expected_category} 筛选目标"
            return self.make_visual_servo_result(
                found=True,
                target_type="block",
                category="dark_rectangle",
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
                message=str(exc),
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
