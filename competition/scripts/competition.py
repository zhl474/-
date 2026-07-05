#!/home/zhl/fr3env/fr3env/bin/python
import rospy
import numpy as np
import time
from matplotlib import pyplot as plt
import math
import yaml
import threading
import os

with open("/home/zhl/SingleArmTetris/SingleArmTetris/src/competition/config/calibration_matrix.yaml") as f:
    data = yaml.safe_load(f)
camera_matrix = np.array(data['camera_matrix'])
dist_coeff = np.array(data['dist_coeff'])
import warnings
warnings.filterwarnings("ignore",category=RuntimeWarning)

from image_process.srv import GetTargetPos ,GetTargetPosRequest
from control.srv import arm,armRequest,motor,motorRequest,suck,suckRequest
from sensor_msgs.msg import Image
from servo_eccentric_compensation import ServoEccentricCompensator


if __name__ == "__main__":
    rospy.init_node("competition")
    rospy.wait_for_service("get_cube_pos")
    rospy.wait_for_service("get_board_pos")
    rospy.wait_for_service("get_cube_location")
    rospy.wait_for_service("get_put_pose")
    try:
        get_array = rospy.ServiceProxy("get_cube_pos", GetTargetPos)
        get_board_pos = rospy.ServiceProxy("get_board_pos", GetTargetPos)
        get_cube_location = rospy.ServiceProxy("get_cube_location", GetTargetPos)
        get_put_pose = rospy.ServiceProxy("get_put_pose", GetTargetPos)
        arm_control = rospy.ServiceProxy("arm_control", arm)
        motor_control = rospy.ServiceProxy("motor_control", motor)
        suck_control = rospy.ServiceProxy("suck_control", suck)
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)
    GetTargetPos_req = GetTargetPosRequest()
    arm_req = armRequest()
    arm_req.speed = 80
    motor_req = motorRequest()
    suck_req = suckRequest()
    suck_in = 0
    suck_out = 1
    sucker_off = 2
    script_dir = os.path.dirname(os.path.abspath(__file__))
    servo_compensation_config = os.path.join(
        script_dir,
        "servo_eccentric_compensation",
        "servo_eccentric_compensation.yaml",
    )
    servo_compensator = ServoEccentricCompensator(servo_compensation_config, debug=False)

    def apply_servo_compensation(pose, theta_deg, label):
        # 在抓取/摆放目标点发送前显式补偿，底层 MoveL 不做隐式补偿。
        pose_before = list(pose)
        compensation = servo_compensator.get_compensation(theta_deg)
        pose_after = servo_compensator.apply(pose_before, theta_deg).tolist()
        print(
            f"[吸盘偏心补偿/{label}] theta={float(theta_deg):.3f} deg, "
            f"补偿量(mm)={compensation.tolist()}, "
            f"补偿前={pose_before}, 补偿后={pose_after}"
        )
        return pose_after

    # rospy.set_param('/motor_done', 1)
    motor_done = 1
    def timer_callback():
        global motor_done
        motor_done = 1
    #################################           复位              ################################
    shooting_angle = [-250.4151306152343 , 22.14801216125488, 380.3343505859375 , -180, 0, 90]#process那里还有一个
    arm_req.pose = shooting_angle
    resp = arm_control.call(arm_req)

    #################################           获取图像与摆放方法             ################################
    print("是否进行进阶任务")
    jinjie=input()
    if(jinjie=="y"):
        GetTargetPos_req.num = -2

    pick_list=[[1, 2, 90, 'L_yellow'], [4, 1.5, 0, 'T'], [7.5, 1, 0, 'line'], [9.5, 2, -90, 'z_green'], [2, 3, 90, 'L_yellow'],#基础任务的放置表
            [4, 3, 180, 'L_blue'], [7, 2, 0, 'L_yellow'], [1.5, 5, 90, 'T'], [3.5, 5, 90, 'z_blue'], [5, 4.5, 0, 'z_green'], 
            [6.5, 3.5, 0, 'square'], [9, 5, -90, 'L_blue'], [10, 4.5, 90, 'line'], [7.5, 5.5, 0, 'square'], [1.5, 7, -90, 'z_green'], 
            [3.5, 7, 90, 'z_blue'], [6.5, 7, 90, 'T'], [5, 7.5, 90, 'line'], [9, 7, 0, 'L_blue'], [3, 9, -90, 'L_blue'], [7, 8.5, 180, 'T'], 
            [9.5, 8.5, 0, 'square'], [1, 10, 90, 'L_yellow'], [4.5, 10, 90, 'T'], [6.5, 11, 90, 'z_blue'], [8.5, 10, 0, 'line'], [2.5, 11, 90, 'z_blue'], 
            [8, 12, 90, 'L_yellow'], [9.5, 12, -90, 'z_green'], [1.5, 12.5, 0, 'square'], [3.5, 13, -90, 'z_green'], [5, 12.5, 90, 'line'], [6.5, 13, 90, 'z_blue'], [9, 14, 180, 'L_blue']]


    #################################           开始视觉识别             ################################
    rospy.wait_for_message('/camera/image_raw', Image, timeout=30)
    while True:
        resp = get_board_pos(GetTargetPos_req)
        board_ok = len(resp.array) > 0 and int(resp.array[0]) == 1
        if board_ok:
            resp = get_array(GetTargetPos_req)
            cube_num = int(resp.array[0])
        else:
            # 红色提示托盘识别失败，本轮跳过方块识别，等待人工确认后重新执行循环。
            print("\033[91m托盘识别失败，已跳过方块识别，请调整后重新识别。\033[0m")
            cube_num = 0
        if input("识别结果满意扣1")=="1":
            break


    #################################           运动到第一个方块上方             ################################
    # input("按回车后运动到第一个上方...")
    # GetTargetPos_req.num = -1
    # resp = get_cube_location.call(GetTargetPos_req)
    # x,y,z,t,xuanzhuan_angle=resp.array

    # #因为是270度舵机，所以正反只有135度，为了应对170度的情况需要先预留空间
    # motor_req.angle = 180
    # motor_control.call(motor_req)
   
    # print(f"该方块经9点标定预测的实际坐标、角度：x={x}, y={y}, z={z}, t={t}")
    # #运动到第一个方块上方，所以高度为z+3
    # set_angle=shooting_angle
    # base_x=x
    # base_y=y
    # base_z=shooting_angle[2] - z + 100 -66 
    # set_angle[0]=base_x
    # set_angle[1]=base_y
    # set_angle[2]=z+7
    # set_angle[5]=shooting_angle[5]
    # arm_req.pose = set_angle
    # arm_control.call(arm_req)

    #################################           捡所有方块             ################################
    input("按回车后捡所有方块...")
    angle_velocity = 270
    last_angle = 180
    def down_pick(x,y,z,t,xuanzhuan_angle,index_cube):#根据位置过去捡方块
        global motor_done,last_angle
        wait_time=0#没进if不用等
        if(xuanzhuan_angle>0):
            if last_angle+xuanzhuan_angle>360:#没必要每次都复位
                motor_req.angle = 350-xuanzhuan_angle
                wait_time = (last_angle-motor_req.angle)/angle_velocity
                motor_control.call(motor_req)
                motor_done = 0
                last_angle=motor_req.angle
        else:
            if last_angle+xuanzhuan_angle<0:
                motor_req.angle = 10-xuanzhuan_angle
                wait_time = (last_angle-motor_req.angle)/angle_velocity
                motor_control.call(motor_req)
                motor_done = 0
                last_angle=motor_req.angle
        timer = threading.Timer(wait_time, timer_callback)
        timer.start()
        theta_pick = last_angle
        #先运动到方块上方再下去吸取
        set_angle=list(shooting_angle)
        base_x=x
        base_y=y
        set_angle[0]=base_x
        set_angle[1]=base_y
        set_angle[2]=z+7
        set_angle[5]=shooting_angle[5]

        #如果不是第一个方块就先到上方，第一个方块可以计时开始前先过去，节约点时间
        # if(index_cube!=0):
        arm_req.pose = apply_servo_compensation(set_angle, theta_pick, "抓取上方")
        arm_control.call(arm_req)
        input("按空格继续...")
        #吸方块
        while not motor_done:#等舵机转完
            time.sleep(0.05)
        set_angle[2]=z-8
        arm_req.pose = set_angle
        arm_req.speed = 60
        # 如果恢复下探 MoveL，发送前也要使用 theta_pick 做吸盘偏心补偿。
        # arm_control.call(arm_req)
        # input("捡")
        suck_req.state = suck_in
        suck_control.call(suck_req)
        #shooting_angle=[-240.575,-25.665,522.61,180,0,90]

        #吸到方块后起来一点
        set_angle[2]=z+20
        arm_req.pose = apply_servo_compensation(set_angle, theta_pick, "抓取抬起")
        arm_req.speed = 80
        arm_control.call(arm_req)

        #开始设置舵机角度
        # if(xuanzhuan_angle>0):
        #     motor_req.angle = last_angle+xuanzhuan_angle
        #     motor_control.call(motor_req)
        # else:
        theta_place = last_angle+xuanzhuan_angle
        motor_req.angle = theta_place
        motor_control.call(motor_req)
        last_angle = motor_req.angle
        return theta_pick,theta_place,set_angle

    for i in range(cube_num):
        GetTargetPos_req.num = i
        #先获取方块坐标（x,y,z）和角度t
        resp = get_cube_location.call(GetTargetPos_req)
        x,y,z,t,xuanzhuan_angle=resp.array
        #捡方块
        # input("捡")
        theta_pick,theta_place,set_angle=down_pick(x,y,z,t,xuanzhuan_angle,i)

        #获取这个方块摆放位置的姿态
        resp = get_put_pose.call(GetTargetPos_req)
        put_pose=list(resp.array)
        #print("旋转",angle)

        #运动到放置位置上方较高处，气泵喷气
        # put_pose[2]=205#高度稍微高一点，不然会撞到方块
        put_pose[2]=197#高度稍微高一点，不然会撞到方块
        put_pose[5]=shooting_angle[5]
        # print("目标位置",put_pose)
        arm_req.pose = apply_servo_compensation(put_pose, theta_place, "摆放上方")
        arm_control.call(arm_req)

        # delayed_suck_out(0.2)
        # while(1):
        #time.sleep()
        # input("回车继续")
        put_pose[2]=188
        put_pose[5]=shooting_angle[5]
        arm_req.pose = apply_servo_compensation(put_pose, theta_place, "摆放下放")
        arm_req.speed = 60
        arm_control.call(arm_req)
        input("放")
        suck_req.state = suck_out
        resp = suck_control.call(suck_req)

        # time.sleep(0.1)
        #再上来，准备捡下一个方块
        put_pose[2]=put_pose[2]+20
        arm_req.pose = apply_servo_compensation(put_pose, theta_place, "摆放抬起")
        arm_req.speed = 80
        arm_control.call(arm_req)
        i=i+1
        if(i==34):#捡完34个方块结束
            break

    suck_req.state = sucker_off
    resp = suck_control.call(suck_req)










