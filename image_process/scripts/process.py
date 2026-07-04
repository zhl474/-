#!/home/zhl/fr3env/fr3env/bin/python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import yaml
from ultralytics import YOLO
import torch

from image_process_lib.Place_optimization import get_all_cube, make_list, put_fenlei,cube_pocess,optimize_block_assignment,get_cube_location,get_put_pose
from image_process_lib.block_detection import get_mask, coreect_LL_location
from image_process_lib.template_config import load_template_sizes, get_template_size
from image_process_lib.template_match.template_match import get_rect
from image_process_lib.point_calibration import ArmCalibrator, x_predict,y_predict,z_predict
from image_process_lib.board_detect import board_detect

from image_process.srv import GetTargetPos, GetTargetPosResponse
from camera.srv import pixel2world, pixel2worldRequest

from ctypes import * 

# 进阶任务动态库可能仍输出旧类别名，这里统一转换为新模型类别名。
CATEGORY_NAME_MAP = {
    "LR": "L_blue",
    "LL": "L_yellow",
    "ZL": "z_blue",
    "ZR": "z_green",
    "O": "square",
    "suqare": "square",
    "Line": "line",
}

class ImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
        self.service1 = rospy.Service("get_cube_pos", GetTargetPos,self.get_cube_pos)
        self.service2 = rospy.Service("get_board_pos", GetTargetPos,self.get_board_pos)
        self.get_cube_location_service = rospy.Service("get_cube_location", GetTargetPos,self.get_cube_location1)
        self.get_put_pose_service = rospy.Service("get_put_pose", GetTargetPos,self.get_put_pose)

        model_path="/home/zhl/SingleArmTetris/SingleArmTetris/src/competition/model/best5.14.pt"
        self.model=YOLO(model_path)
        with open("/home/zhl/SingleArmTetris/SingleArmTetris/src/competition/config/calibration_matrix.yaml") as f:
            data = yaml.safe_load(f)
        self.camera_matrix = np.array(data['camera_matrix'])
        self.dist_coeff = np.array(data['dist_coeff'])

        self.results = None
        self.calibrator = ArmCalibrator()

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
        # 裁剪边距：正数向外扩展 YOLO 框，负数向内收缩 YOLO 框，单位是像素。
        crop_margin = 8
        cube_count=[0,0,0,0,0,0,0]#记录每个方块的放置个数
        h, w = img_bgr1.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeff, (w, h), 1, (w, h))
        img_bgr = cv2.undistort(img_bgr1, self.camera_matrix, self.dist_coeff, None, new_camera_mtx)#去畸变
        img_bgr2=np.copy(img_bgr)#复制一个数组，我们会在检测到的图片中画黑框开辅助判断，但黑框会影响颜色分割，所以复制一个数组，img_bgr用来检测画黑框，img_bgr2用来颜色分割
        result = self.model(img_bgr,iou=0.5,conf=0.45)#yolo检测
        template_sizes = load_template_sizes()#每次服务调用读取一次模板尺寸配置，便于标定后直接生效
        # cv2.imwrite('/home/zhl/桌面/yolo识别.jpg', result[0].plot())
        cube_list=[]#检测到方块储存在这个列表中
        for i in range(7):
            cube_list.append([])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        for det in result[0].boxes.data.tolist():
            x1, y1, x2, y2, score, cid = det#取出识别结果
            category = self.model.names[int(cid)]#类别
            category = CATEGORY_NAME_MAP.get(category, category)
            if(category=="board"):
                continue
            cube_count=get_all_cube(category,cube_count)#读取每种方块的个数
            px=(x1+x2)/2
            py=(y1+y2)/2
            crop_x1 = max(0, int(x1) - crop_margin)
            crop_y1 = max(0, int(y1) - crop_margin)
            crop_x2 = min(w, int(x2) + crop_margin)
            crop_y2 = min(h, int(y2) + crop_margin)
            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                rospy.logwarn("裁剪区域无效，跳过方块: %s" % category)
                continue
            cropped_img = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]#根据裁剪边距调整后的 YOLO 框取出图像
            mask,hsv=get_mask(cropped_img,category)#颜色分割，mask是分割的图片二值化图
            # cv2.imshow("mask",mask)
            # cv2.imshow("cropped_img",cropped_img)
            try:
                template_w, template_h = get_template_size(category, template_sizes)
                # print(f"\033[31m模版长度是{category,template_w, template_h}\033[0m")
            except Exception as e:
                rospy.logwarn("模板尺寸配置读取失败，跳过方块 %s: %s" % (category, e))
                continue
            rect = get_rect(mask,template_w, template_h,category,img_bgr2,crop_x1,crop_y1)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            #方块上表面的最小矩形框，进而得到中心点
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            # abs_box = box + [crop_x1, crop_y1]

            px=(int(rect[0][0])+crop_x1)
            py=(int(rect[0][1])+crop_y1)
            if(category=="L_yellow" or category=="L_blue"):#L型方块要特殊处理，吸中间吸不起来
                px,py=coreect_LL_location(box,mask,rect)
                px=(px+crop_x1)
                py=(py+crop_y1)

            #给中心点画圈
            cv2.circle(img_bgr2,(px,py),3, (0, 0, 255), 2)
            

            #获取角度，不用看
            # theta0=get_precise_angle(rect,box,category,mask)#获取角度，不用看
            theta0=rect[2]
            theta = theta0
            if theta0<-180:
                theta = 360+theta0

            cam_point3d = [0,0,0]
################################################       9点标定法(托盘标定那里也要同步改)      #########################################################
            predicted_x = x_predict(px,py)
            predicted_y = y_predict(px,py)
            predicted_z = z_predict(px,py)
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
        cv2.imwrite('/home/zhl/桌面/cube_pos_image.jpg', img_bgr2)
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
        h, w = img_bgr1.shape[:2]
        board_bgr = img_bgr1
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeff, (w, h), 1, (w, h))
        board_bgr = cv2.undistort(img_bgr1, self.camera_matrix, self.dist_coeff, None, new_camera_mtx)#去畸变
        if board_bgr is None:
            print("没有图片")
        self.calibrator.board_detector(board_bgr,self.pixel2world_client)
        self.calibrator.calibrate()#托盘识别结束

        cv2.imwrite('/home/zhl/桌面/board_bgr.jpg', board_bgr)
        
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
            cube_name=CATEGORY_NAME_MAP.get(cube_list0[i * 4], cube_list0[i * 4])
            cube_list[i].append(cube_name)
        # print(cube_list, cube_list0[-1])  # 打印结果
        cube_sum = 0
        for i in cube_count:
            cube_sum = cube_sum + i * 4
        level = cube_sum // 10
        print("极限填满行", level)
        return cube_list,cube_list0[-1]

if __name__ == "__main__":
    rospy.init_node("image_processor")
    processor = ImageProcessor()
    rospy.spin()
