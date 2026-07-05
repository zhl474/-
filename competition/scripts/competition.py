#!/home/zhl/fr3env/fr3env/bin/python
import os
import sys
import time
import math
import yaml
import threading
import warnings

import numpy as np
import rospy
from matplotlib import pyplot as plt
from sensor_msgs.msg import Image

from control.srv import arm, armRequest, motor, motorRequest, suck, suckRequest
from image_process.srv import GetTargetPos, GetTargetPosRequest, VisualTargetOffset


warnings.filterwarnings("ignore", category=RuntimeWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VISUAL_SERVO_DIR = os.path.join(SCRIPT_DIR, "visual_servo")
if VISUAL_SERVO_DIR not in sys.path:
    sys.path.insert(0, VISUAL_SERVO_DIR)

try:
    from visual_servo.visual_servo_common import (
        apply_camera_to_sucker_offset,
        load_visual_servo_config,
        run_visual_servo_alignment,
    )
except Exception as exc:
    # 视觉伺服骨架阶段允许公共库暂时不可用；真正启用视觉抓取前必须解决这里。
    apply_camera_to_sucker_offset = None
    load_visual_servo_config = None
    run_visual_servo_alignment = None
    print(f"\033[91m视觉伺服公共库导入失败，视觉抓取暂不可用: {exc}\033[0m")


with open("/home/zhl/SingleArmTetris/SingleArmTetris/src/competition/config/calibration_matrix.yaml") as f:
    data = yaml.safe_load(f)
camera_matrix = np.array(data["camera_matrix"])
dist_coeff = np.array(data["dist_coeff"])


if __name__ == "__main__":
    rospy.init_node("competition")

    # ========================= 主流程开关和待实测参数 =========================
    # 默认先走旧开环抓取，避免未填实的视觉伺服占位函数被误调用。
    USE_VISUAL_PICK = False

    # 方块视觉伺服期望类别；空字符串表示不限制类别，当前只作为未来接口参数保留。
    EXPECTED_BLOCK_CATEGORY = ""

    # 粗定位观察高度：后续需要根据全场识别结果和相机视野实测确认。
    ROUGH_LOOK_Z = 220.0

    # 闭环视觉伺服高度：应和 pixel_to_robot_matrix 标定高度一致，当前只是占位参数。
    SERVO_LOOK_Z = 200.0

    # 下探吸取高度：旧流程使用 z-8，新视觉流程先集中成参数，后续按实机重标定。
    PICK_Z = 180.0

    # 吸取后抬起高度：旧流程使用 z+20，新视觉流程先集中成参数，后续按实机重标定。
    LIFT_Z = 200.0

    # 普通移动速度：视觉流程统一从这里取，避免速度散落在各个分支。
    ARM_SPEED = 80

    # 下探吸取速度：正式启用视觉抓取前建议低速实测。
    PICK_SPEED = 60

    # 视觉伺服收敛阈值，单位像素；连续多帧低于该阈值才认为对准成功。
    VISUAL_ERROR_THRESHOLD_PX = 2.0

    # 视觉伺服单次最大 XY 修正距离，单位 mm；防止识别异常导致一次移动过大。
    VISUAL_MAX_STEP_MM = 5.0

    # 视觉伺服最多迭代次数；每轮最多移动一次，也可能只是等待画面稳定。
    VISUAL_MAX_ITER = 40

    # 连续多少帧满足误差阈值才算成功，避免单帧抖动误判。
    VISUAL_SUCCESS_STABLE_FRAMES = 5

    # 连续多少帧未识别到方块才失败；单帧丢失只等待。
    VISUAL_MAX_MISSED_FRAMES = 5

    # 每次视觉伺服移动后的等待时间，单位秒；用于等机械臂和画面稳定。
    VISUAL_SETTLE_SEC = 0.25

    # ========================= ROS 服务初始化 =========================
    rospy.wait_for_service("get_cube_pos")
    rospy.wait_for_service("get_board_pos")
    rospy.wait_for_service("get_cube_location")
    rospy.wait_for_service("get_put_pose")
    if USE_VISUAL_PICK:
        rospy.wait_for_service("get_visual_target_offset")

    try:
        get_array = rospy.ServiceProxy("get_cube_pos", GetTargetPos)
        get_board_pos = rospy.ServiceProxy("get_board_pos", GetTargetPos)
        get_cube_location = rospy.ServiceProxy("get_cube_location", GetTargetPos)
        get_put_pose = rospy.ServiceProxy("get_put_pose", GetTargetPos)
        visual_target_offset = (
            rospy.ServiceProxy("get_visual_target_offset", VisualTargetOffset)
            if USE_VISUAL_PICK
            else None
        )
        arm_control = rospy.ServiceProxy("arm_control", arm)
        motor_control = rospy.ServiceProxy("motor_control", motor)
        suck_control = rospy.ServiceProxy("suck_control", suck)
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)
        raise

    GetTargetPos_req = GetTargetPosRequest()
    arm_req = armRequest()
    arm_req.speed = ARM_SPEED
    motor_req = motorRequest()
    suck_req = suckRequest()
    suck_in = 0
    suck_out = 1
    sucker_off = 2

    visual_servo_config = load_visual_servo_config() if load_visual_servo_config is not None else {}

    # rospy.set_param('/motor_done', 1)
    motor_done = 1
    angle_velocity = 270
    last_angle = 180

    def apply_pose_compensation(pose, theta_deg, label):
        """预留位姿补偿入口。

        当前舵机不偏心，因此这里不做任何补偿，直接返回原始 pose。
        如果以后重新启用吸盘偏心补偿或其它末端补偿，所有抓取/摆放位姿都从这里接入，
        不要再把补偿逻辑散落到主流程各处。
        """
        return list(pose)

    def send_arm_pose(pose, speed=ARM_SPEED, wait_sec=0.0, theta_deg=0.0, label=""):
        """统一发送机械臂位姿。

        输入 pose 是 [x, y, z, rx, ry, rz]；speed 是 MoveL 速度。
        theta_deg 和 label 当前只传给 apply_pose_compensation，方便以后恢复补偿。
        返回机械臂服务响应。
        """
        compensated_pose = apply_pose_compensation(pose, theta_deg, label)
        arm_req.pose = [float(v) for v in compensated_pose]
        arm_req.speed = int(speed)
        resp = arm_control.call(arm_req)
        if wait_sec > 0:
            time.sleep(wait_sec)
        return resp

    def set_sucker_state(state):
        """统一控制吸盘状态：0 吸气，1 喷气，2 关闭。"""
        suck_req.state = int(state)
        return suck_control.call(suck_req)

    def wait_motor_ready():
        """等待舵机转动完成。

        motor_done 由 timer_callback 在估算转动时间结束后置 1。
        后续如果控制节点能返回真实完成状态，应优先替换这里。
        """
        while not motor_done:
            time.sleep(0.05)

    def make_pose_xy_z(base_pose, x, y, z):
        """基于模板 pose 生成新的 x/y/z 位姿。

        base_pose 通常使用 shooting_angle，保留原 rx/ry/rz，只替换位置坐标。
        返回值供粗定位、下探、抬起等步骤继续修改。
        """
        pose = list(base_pose)
        pose[0] = float(x)
        pose[1] = float(y)
        pose[2] = float(z)
        pose[5] = shooting_angle[5]
        return pose

    def timer_callback():
        global motor_done
        motor_done = 1

    def plan_pick_servo_angle(xuanzhuan_angle):
        """规划抓取角度和摆放角度，并处理舵机避限位。

        输入 xuanzhuan_angle 来自 get_cube_location，表示从抓取方向转到摆放方向的角度差。
        返回 theta_pick、theta_place、wait_time。函数内部会在需要避限位时先转动舵机，
        并更新 last_angle；主流程随后用 theta_pick 做抓取，用 theta_place 做摆放。
        """
        global motor_done, last_angle
        wait_time = 0
        if xuanzhuan_angle > 0:
            if last_angle + xuanzhuan_angle > 360:
                motor_req.angle = 350 - xuanzhuan_angle
                wait_time = (last_angle - motor_req.angle) / angle_velocity
                motor_control.call(motor_req)
                motor_done = 0
                last_angle = motor_req.angle
        else:
            if last_angle + xuanzhuan_angle < 0:
                motor_req.angle = 10 - xuanzhuan_angle
                wait_time = (last_angle - motor_req.angle) / angle_velocity
                motor_control.call(motor_req)
                motor_done = 0
                last_angle = motor_req.angle

        timer = threading.Timer(wait_time, timer_callback)
        timer.start()
        theta_pick = last_angle
        theta_place = last_angle + xuanzhuan_angle
        return theta_pick, theta_place, wait_time

    def choose_block_rough_camera_pose(x, y, z, t, index_cube):
        """选择方块视觉伺服开始前的相机粗定位位姿。

        输入 x/y/z/t 来自 get_cube_location，目前仍是旧九点预测结果。
        这个函数要解决的问题是：机械臂先去哪一个高度和 xy 位置，才能让目标方块稳定进入相机视野。
        该策略需要现场验证不同区域、不同方块姿态下旧预测误差是否足够小，所以当前先保守留空。
        返回值未来会交给 send_arm_pose 和 align_camera_to_block 使用。
        """
        message = "粗定位现在只使用1像素=0.5mm"
        print(f"\033[91m{message}\033[0m")
        return True, list(shooting_angle), message

    def choose_block_pick_heights(x, y, z, t, index_cube):
        """决定视觉抓取的下探高度和抬起高度。

        输入 x/y/z/t 来自 get_cube_location；视觉伺服对准后，相机中心会对准方块，
        但最终吸盘下探高度是否还能沿用旧 z-8，需要现场重新确认。
        返回 pick_z 和 lift_z，分别供下探吸取和吸后抬起使用。
        """
        message = "choose_block_pick_heights 尚未实测：暂用集中配置 PICK_Z/LIFT_Z"
        print(f"\033[93m{message}\033[0m")
        return float(PICK_Z), float(LIFT_Z)

    def align_camera_to_block(start_pose, expected_category=""):
        """用方块视觉伺服把相机中心对准目标方块。

        start_pose 是 choose_block_rough_camera_pose 输出的观察位姿。
        这里正式委托 visual_servo_common.run_visual_servo_alignment 执行闭环：
        图像节点负责 get_visual_target_offset，公共函数负责像素误差到机械臂 XY 修正。
        返回格式固定为 success, camera_pose, message，供 down_pick_visual 统一处理。
        """
        if visual_target_offset is None:
            return False, list(start_pose), "视觉抓取未启用，未创建 get_visual_target_offset 服务代理"
        if run_visual_servo_alignment is None:
            return False, list(start_pose), "视觉伺服公共函数不可用，无法执行闭环对准"
        if not visual_servo_config:
            return False, list(start_pose), "视觉伺服配置为空，无法执行闭环对准"

        success, camera_pose, last_resp, message = run_visual_servo_alignment(
            arm_control,
            visual_target_offset,
            start_pose,
            visual_servo_config,
            expected_category=expected_category,
            speed=ARM_SPEED,
            error_threshold_px=VISUAL_ERROR_THRESHOLD_PX,
            max_step_mm=VISUAL_MAX_STEP_MM,
            max_iter=VISUAL_MAX_ITER,
            success_stable_frames=VISUAL_SUCCESS_STABLE_FRAMES,
            max_missed_frames=VISUAL_MAX_MISSED_FRAMES,
            settle_sec=VISUAL_SETTLE_SEC,
        )
        if not success:
            return False, list(camera_pose), message
        return True, list(camera_pose), message

    def make_sucker_pose_from_camera_pose(camera_pose):
        """相机中心对准方块后，换算吸盘中心应该到达的高位。

        输入 camera_pose 来自 align_camera_to_block。
        这里调用 visual_servo_common.apply_camera_to_sucker_offset，
        使用 visual_servo.yaml 中的 camera_to_sucker_offset_mm。
        该偏移是否适合当前吸取高度仍需要现场复核。
        """
        if apply_camera_to_sucker_offset is None:
            return False, list(camera_pose), "相机到吸盘偏移函数不可用"
        if not visual_servo_config:
            return False, list(camera_pose), "视觉伺服配置为空，无法应用相机到吸盘偏移"
        sucker_pose = apply_camera_to_sucker_offset(camera_pose, visual_servo_config)
        return True, list(sucker_pose), "已根据相机到吸盘偏移生成吸盘高位"

    def handle_block_servo_failed(index_cube, message):
        """处理方块视觉伺服失败。

        输入 index_cube 是当前方块序号，message 是失败原因。
        后续可以在这里实现停机、跳过该块、人工确认重试等策略。
        当前策略是保守停在当前位置，不下探，并抛出异常中断主流程。
        """
        error_message = f"第 {index_cube} 个方块视觉抓取失败: {message}"
        print(f"\033[91m{error_message}\033[0m")
        raise RuntimeError(error_message)

    def verify_pick_before_down(sucker_high_pose, index_cube):
        """下探吸取前的安全确认入口。

        输入 sucker_high_pose 是吸盘已经移动到方块上方后的高位。
        第一版视觉抓取建议先保留人工确认；后续确认安全后可以改成自动返回 True。
        返回 True 表示允许下探，False 表示本次不吸取。
        """
        print(f"第 {index_cube} 个方块吸盘高位: {sucker_high_pose}")
        answer = input("确认吸盘高位安全后按回车下探；输入 n 取消本次吸取: ")
        return answer.strip().lower() != "n"

    #################################           复位              ################################
    shooting_angle = [-250.4151306152343, 22.14801216125488, 380.3343505859375, -180, 0, 90]  # process那里还有一个
    send_arm_pose(shooting_angle, speed=ARM_SPEED)

    #################################           获取图像与摆放方法             ################################
    print("是否进行进阶任务")
    jinjie = input()
    if jinjie == "y":
        GetTargetPos_req.num = -2

    pick_list = [
        [1, 2, 90, "L_yellow"], [4, 1.5, 0, "T"], [7.5, 1, 0, "line"], [9.5, 2, -90, "z_green"], [2, 3, 90, "L_yellow"],
        [4, 3, 180, "L_blue"], [7, 2, 0, "L_yellow"], [1.5, 5, 90, "T"], [3.5, 5, 90, "z_blue"], [5, 4.5, 0, "z_green"],
        [6.5, 3.5, 0, "square"], [9, 5, -90, "L_blue"], [10, 4.5, 90, "line"], [7.5, 5.5, 0, "square"], [1.5, 7, -90, "z_green"],
        [3.5, 7, 90, "z_blue"], [6.5, 7, 90, "T"], [5, 7.5, 90, "line"], [9, 7, 0, "L_blue"], [3, 9, -90, "L_blue"], [7, 8.5, 180, "T"],
        [9.5, 8.5, 0, "square"], [1, 10, 90, "L_yellow"], [4.5, 10, 90, "T"], [6.5, 11, 90, "z_blue"], [8.5, 10, 0, "line"], [2.5, 11, 90, "z_blue"],
        [8, 12, 90, "L_yellow"], [9.5, 12, -90, "z_green"], [1.5, 12.5, 0, "square"], [3.5, 13, -90, "z_green"], [5, 12.5, 90, "line"], [6.5, 13, 90, "z_blue"], [9, 14, 180, "L_blue"],
    ]

    #################################           开始视觉识别             ################################
    rospy.wait_for_message("/camera/image_raw", Image, timeout=30)
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
        if input("识别结果满意扣1") == "1":
            break

    #################################           捡所有方块             ################################
    input("按回车后捡所有方块...")

    def down_pick(x, y, z, t, xuanzhuan_angle, index_cube):  # 根据位置过去捡方块
        """旧开环抓取流程，先保留为视觉抓取骨架的备用对照。"""
        global last_angle
        theta_pick, theta_place, wait_time = plan_pick_servo_angle(xuanzhuan_angle)

        # 先运动到方块上方再下去吸取。这里仍沿用旧九点预测位置。
        set_angle = make_pose_xy_z(shooting_angle, x, y, z + 7)
        send_arm_pose(set_angle, speed=ARM_SPEED, theta_deg=theta_pick, label="抓取上方")
        input("按空格继续...")

        # 等舵机转完再吸取，避免舵机还在转时下探。
        wait_motor_ready()
        set_angle[2] = z - 8
        arm_req.pose = set_angle
        arm_req.speed = PICK_SPEED
        # 旧代码当前没有实际下探 MoveL，保持原行为不改变。
        # send_arm_pose(set_angle, speed=PICK_SPEED, theta_deg=theta_pick, label="抓取下探")
        set_sucker_state(suck_in)

        # 吸到方块后抬起一点。
        set_angle[2] = z + 20
        send_arm_pose(set_angle, speed=ARM_SPEED, theta_deg=theta_pick, label="抓取抬起")

        # 开始设置摆放前的舵机角度。
        motor_req.angle = theta_place
        motor_control.call(motor_req)
        last_angle = motor_req.angle
        return theta_pick, theta_place, set_angle

    def down_pick_visual(x, y, z, t, xuanzhuan_angle, index_cube):
        """方块视觉伺服抓取骨架。

        这个函数只串联正式流程，不在这里写具体识别算法。
        当前多个关键函数仍是占位实现，因此 USE_VISUAL_PICK 默认 False。
        后续逐步填实 choose_block_rough_camera_pose、align_camera_to_block、
        make_sucker_pose_from_camera_pose 和 choose_block_pick_heights 后，再启用它。
        """
        global last_angle

        # 1. 规划抓取角度和摆放角度，逻辑沿用旧流程。
        theta_pick, theta_place, wait_time = plan_pick_servo_angle(xuanzhuan_angle)

        # 2. 选择视觉伺服开始前的相机粗定位观察位。
        rough_ok, rough_camera_pose, message = choose_block_rough_camera_pose(x, y, z, t, index_cube)
        if not rough_ok:
            handle_block_servo_failed(index_cube, message)

        # 3. 移动到粗定位观察位，让目标方块进入相机视野。
        send_arm_pose(rough_camera_pose, speed=ARM_SPEED, theta_deg=theta_pick, label="视觉抓取粗定位")

        # 4. 等待舵机避限位动作结束，再做闭环视觉伺服，避免画面持续变化。
        wait_motor_ready()

        # 5. 调用视觉伺服接口，让相机中心对准方块。
        success, camera_pose, message = align_camera_to_block(
            rough_camera_pose,
            expected_category=EXPECTED_BLOCK_CATEGORY,
        )
        if not success:
            handle_block_servo_failed(index_cube, message)

        # 6. 相机中心对准方块后，换算吸盘中心应该到达的高位。
        success, sucker_high_pose, message = make_sucker_pose_from_camera_pose(camera_pose)
        if not success:
            handle_block_servo_failed(index_cube, message)

        # 7. 选择本次吸取的下探高度和抬起高度。
        pick_z, lift_z = choose_block_pick_heights(x, y, z, t, index_cube)
        sucker_high_pose[2] = lift_z
        send_arm_pose(sucker_high_pose, speed=ARM_SPEED, theta_deg=theta_pick, label="视觉抓取吸盘高位")

        # 8. 下探前留出安全确认入口，避免未实测参数直接撞桌面。
        if not verify_pick_before_down(sucker_high_pose, index_cube):
            handle_block_servo_failed(index_cube, "人工取消下探吸取")

        # 9. 下探吸取。
        pick_pose = list(sucker_high_pose)
        pick_pose[2] = pick_z
        send_arm_pose(pick_pose, speed=PICK_SPEED, theta_deg=theta_pick, label="视觉抓取下探")
        set_sucker_state(suck_in)

        # 10. 抬起，准备进入摆放流程。
        lift_pose = list(sucker_high_pose)
        lift_pose[2] = lift_z
        send_arm_pose(lift_pose, speed=ARM_SPEED, theta_deg=theta_pick, label="视觉抓取抬起")

        # 11. 舵机转到摆放角度。
        motor_req.angle = theta_place
        motor_control.call(motor_req)
        last_angle = motor_req.angle
        return theta_pick, theta_place, lift_pose

    for i in range(cube_num):
        GetTargetPos_req.num = i
        # 先获取方块粗坐标和角度。视觉抓取启用后，这里只作为粗定位输入。
        resp = get_cube_location.call(GetTargetPos_req)
        x, y, z, t, xuanzhuan_angle = resp.array

        if USE_VISUAL_PICK:
            theta_pick, theta_place, set_angle = down_pick_visual(x, y, z, t, xuanzhuan_angle, i)
        else:
            theta_pick, theta_place, set_angle = down_pick(x, y, z, t, xuanzhuan_angle, i)

        # 获取这个方块摆放位置的姿态。托盘视觉伺服后续单独接入，这里暂时保留旧 get_put_pose。
        resp = get_put_pose.call(GetTargetPos_req)
        put_pose = list(resp.array)

        # 运动到放置位置上方较高处，气泵喷气。
        put_pose[2] = 197  # 高度稍微高一点，不然会撞到方块。
        put_pose[5] = shooting_angle[5]
        send_arm_pose(put_pose, speed=ARM_SPEED, theta_deg=theta_place, label="摆放上方")

        put_pose[2] = 188
        put_pose[5] = shooting_angle[5]
        send_arm_pose(put_pose, speed=PICK_SPEED, theta_deg=theta_place, label="摆放下放")
        input("放")
        set_sucker_state(suck_out)

        # 再上来，准备捡下一个方块。
        put_pose[2] = put_pose[2] + 20
        send_arm_pose(put_pose, speed=ARM_SPEED, theta_deg=theta_place, label="摆放抬起")
        i = i + 1
        if i == 34:  # 捡完34个方块结束。
            break

    set_sucker_state(sucker_off)
