#!/home/zhl/fr3env/fr3env/bin/python
import os
import sys
import time
import threading
import warnings

import rospy
from sensor_msgs.msg import Image

from control.srv import arm, armRequest, motor, motorRequest, suck, suckRequest
from image_process.srv import GetTargetPos, GetTargetPosRequest, VisualServoOffset, VisualServoOffsetRequest


warnings.filterwarnings("ignore", category=RuntimeWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from visual_servo.visual_servo_common import (
        apply_camera_to_sucker_offset,
        load_visual_servo_config,
        run_offset_visual_servo_alignment,
    )
except Exception as exc:
    # 视觉伺服骨架阶段允许公共库暂时不可用；真正启用视觉抓取前必须解决这里。
    apply_camera_to_sucker_offset = None
    load_visual_servo_config = None
    run_offset_visual_servo_alignment = None
    print(f"\033[91m视觉伺服公共库导入失败，视觉抓取暂不可用: {exc}\033[0m")


# 高位全场拍摄位姿。process.py 里也有同一份值，后面如果要彻底整理，应先统一接口再移动。
BASE_SHOOTING_ANGLE = [-250.4151306152343, 22.14801216125488, 380.3343505859375, -180, 0, 90]

# 与 image_process_lib.block_category.BLOCK_CATEGORY_NAMES 保持一致。
BLOCK_CATEGORY_NAMES = (
    "L_blue",
    "L_yellow",
    "z_blue",
    "z_green",
    "square",
    "T",
    "line",
)


def build_base_pick_list(include_index=False):
    """基础任务摆放表：col、row、目标角度、方块类别。"""
    pick_list = [
        [1, 2, 90, "L_yellow"], [4, 1.5, 0, "T"], [7.5, 1, 0, "line"], [9.5, 2, -90, "z_green"], [2, 3, 90, "L_yellow"],
        [4, 3, 180, "L_blue"], [7, 2, 0, "L_yellow"], [1.5, 5, 90, "T"], [3.5, 5, 90, "z_blue"], [5, 4.5, 0, "z_green"],
        [6.5, 3.5, 0, "square"], [9, 5, -90, "L_blue"], [10, 4.5, 90, "line"], [7.5, 5.5, 0, "square"], [1.5, 7, -90, "z_green"],
        [3.5, 7, 90, "z_blue"], [6.5, 7, 90, "T"], [5, 7.5, 90, "line"], [9, 7, 0, "L_blue"], [3, 9, -90, "L_blue"], [7, 8.5, 180, "T"],
        [9.5, 8.5, 0, "square"], [1, 10, 90, "L_yellow"], [4.5, 10, 90, "T"], [6.5, 11, 90, "z_blue"], [8.5, 10, 0, "line"], [2.5, 11, 90, "z_blue"],
        [8, 12, 90, "L_yellow"], [9.5, 12, -90, "z_green"], [1.5, 12.5, 0, "square"], [3.5, 13, -90, "z_green"], [5, 12.5, 90, "line"], [6.5, 13, 90, "z_blue"], [9, 14, 180, "L_blue"],
    ]
    if include_index:
        for index, item in enumerate(pick_list):
            item.append(index)
    return pick_list


def decode_category_code(category_code):
    """把图像节点返回的类别码转换成方块类别名。"""
    try:
        index = int(round(float(category_code)))
    except (TypeError, ValueError):
        return ""
    if 0 <= index < len(BLOCK_CATEGORY_NAMES):
        return BLOCK_CATEGORY_NAMES[index]
    return ""


def parse_cube_location_response(array, fallback_category=""):
    """解析 get_cube_location 响应，兼容旧版 5 项返回。"""
    values = list(array)
    if len(values) < 5:
        raise ValueError(f"get_cube_location 返回长度不足: {len(values)}")
    x, y, z, t, xuanzhuan_angle = [float(value) for value in values[:5]]
    high_category = decode_category_code(values[5]) if len(values) >= 6 else ""
    if not high_category:
        high_category = str(fallback_category or "").strip()
    return x, y, z, t, xuanzhuan_angle, high_category


if __name__ == "__main__":
    rospy.init_node("competition")

    # ========================= 主流程开关和待实测参数 =========================
    USE_VISUAL_PICK = True
    USE_VISUAL_PLACE = True

    EXPECTED_BLOCK_CATEGORY = ""#指定抓取某个类别方块，留空则正常运行
    BLOCK_TEMPLATE_PROFILE = "low"
    BLOCK_USE_ANGLE_PRIOR = True
    BLOCK_ANGLE_WINDOW_DEG = 10.0
    BLOCK_ANGLE_STEP_DEG = 1.0
    BLOCK_USE_POSITION_PRIOR = False
    BLOCK_SEARCH_RADIUS_PX = 0.0

    SERVO_LOOK_Z = 200.0
    PICK_Z = 170.0
    LIFT_Z = 205.0
    ARM_SPEED = 80
    PICK_SPEED = 40
    SERVO_SPEED = 25
    PLACE_HIGH_Z = 197.0
    PLACE_DOWN_Z = 188.0
    PLACE_LIFT_STEP_MM = 20.0

    VISUAL_ERROR_THRESHOLD_PX = 2.0
    VISUAL_MIN_STEP_MM = 0.05
    VISUAL_MAX_STEP_MM = 5.0
    VISUAL_MAX_ITER = 40
    VISUAL_SUCCESS_STABLE_FRAMES = 2
    VISUAL_MAX_MISSED_FRAMES = 5
    VISUAL_SETTLE_SEC = 0.0

    angle_velocity = 270
    last_angle = 180
    shooting_angle = list(BASE_SHOOTING_ANGLE)
    pick_list = build_base_pick_list(include_index=False)
    is_advanced_task = False

    # ========================= ROS 服务初始化 =========================
    rospy.wait_for_service("get_cube_pos")
    rospy.wait_for_service("get_board_pos")
    rospy.wait_for_service("get_cube_location")
    rospy.wait_for_service("get_put_pose")
    rospy.wait_for_service("suck_control")
    if USE_VISUAL_PICK or USE_VISUAL_PLACE:
        rospy.wait_for_service("get_visual_servo_offset")

    try:
        get_array = rospy.ServiceProxy("get_cube_pos", GetTargetPos)
        get_board_pos = rospy.ServiceProxy("get_board_pos", GetTargetPos)
        get_cube_location = rospy.ServiceProxy("get_cube_location", GetTargetPos)
        get_put_pose = rospy.ServiceProxy("get_put_pose", GetTargetPos)
        visual_servo_offset = (
            rospy.ServiceProxy("get_visual_servo_offset", VisualServoOffset)
            if USE_VISUAL_PICK or USE_VISUAL_PLACE
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
    SERVO_LOOK_Z = float(visual_servo_config.get("servo_look_z", SERVO_LOOK_Z))

    # rospy.set_param('/motor_done', 1)
    motor_done = 1

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

    def request_visual_servo_offset(
        target_type,
        expected_category="",
        row=0.0,
        col=0.0,
        template_profile="",
        use_angle_prior=False,
        angle_center_deg=0.0,
        angle_window_deg=0.0,
        angle_step_deg=1.0,
        use_position_prior=False,
        search_center_x=0.0,
        search_center_y=0.0,
        search_radius_px=0.0,
    ):
        """请求图像节点返回视觉伺服目标相对相机中心的像素偏差。

        这是 competition.py 和 image_process 节点之间的统一通信入口。
        方块请求使用 target_type="block"，托盘请求使用 target_type="board"。
        返回对象必须包含 found、dx_px、dy_px、message 等字段，供通用闭环函数使用。
        """
        if visual_servo_offset is None:
            raise RuntimeError("视觉伺服服务代理未创建，请先打开 USE_VISUAL_PICK 或 USE_VISUAL_PLACE")
        req = VisualServoOffsetRequest()
        req.target_type = str(target_type)
        req.expected_category = str(expected_category)
        req.row = float(row)
        req.col = float(col)
        req.template_profile = str(template_profile)
        req.use_angle_prior = bool(use_angle_prior)
        req.angle_center_deg = float(angle_center_deg)
        req.angle_window_deg = float(angle_window_deg)
        req.angle_step_deg = float(angle_step_deg)
        req.use_position_prior = bool(use_position_prior)
        req.search_center_x = float(search_center_x)
        req.search_center_y = float(search_center_y)
        req.search_radius_px = float(search_radius_px)
        return visual_servo_offset.call(req)

    def request_block_visual_offset(
        expected_category="",
        template_profile="low",
        use_angle_prior=False,
        angle_center_deg=0.0,
        angle_window_deg=0.0,
        angle_step_deg=1.0,
        use_position_prior=False,
        search_center_x=0.0,
        search_center_y=0.0,
        search_radius_px=0.0,
    ):
        """请求图像节点返回当前方块目标相对相机中心的像素偏差。"""
        return request_visual_servo_offset(
            target_type="block",
            expected_category=expected_category,
            template_profile=template_profile,
            use_angle_prior=use_angle_prior,
            angle_center_deg=angle_center_deg,
            angle_window_deg=angle_window_deg,
            angle_step_deg=angle_step_deg,
            use_position_prior=use_position_prior,
            search_center_x=search_center_x,
            search_center_y=search_center_y,
            search_radius_px=search_radius_px,
        )

    def request_board_visual_offset(row, col):
        """请求图像节点返回托盘目标格点相对相机中心的像素偏差。"""
        return request_visual_servo_offset(
            target_type="board",
            row=row,
            col=col,
        )

    def move_pose_for_visual_servo(pose, speed, wait_sec):
        """供通用视觉伺服闭环调用的机械臂移动函数。

        闭环模块只负责计算下一步 pose；真正发机械臂命令仍然通过 send_arm_pose，
        这样速度、等待、未来位姿补偿都集中在 competition.py 管理。
        """
        return send_arm_pose(pose, speed=speed, wait_sec=wait_sec, label="视觉伺服闭环移动")

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
                wait_time = abs(last_angle - motor_req.angle) / angle_velocity
                motor_control.call(motor_req)
                motor_done = 0
                last_angle = motor_req.angle
        else:
            if last_angle + xuanzhuan_angle < 0:
                motor_req.angle = 10 - xuanzhuan_angle
                wait_time = abs(last_angle - motor_req.angle) / angle_velocity
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

        输入 x/y/z/t 来自 get_cube_location，x/y/z 已经是图像节点生成的伺服观察位。
        这里保留姿态字段，真正的吸盘偏移和下探高度由后续函数处理。
        """
        rough_pose = make_pose_xy_z(shooting_angle, x, y, z)
        message = f"方块视觉抓取伺服观察位: x={x:.2f}, y={y:.2f}, z={z:.2f}"
        print(message)
        return True, rough_pose, message

    def choose_block_pick_heights(x, y, z, t, index_cube):
        """决定视觉抓取的下探高度和抬起高度。

        输入 x/y/z/t 来自 get_cube_location；视觉伺服对准后，相机中心会对准方块，
        但最终吸盘下探高度是否还能沿用旧 z-8，需要现场重新确认。
        返回 pick_z 和 lift_z，分别供下探吸取和吸后抬起使用。
        """
        message = f"抓方块高度{PICK_Z},抬起高度{LIFT_Z}"
        print(f"\033[93m{message}\033[0m")
        return float(PICK_Z), float(LIFT_Z)

    def align_camera_to_block(start_pose, expected_category="", high_angle_deg=0.0):
        """用方块视觉伺服把相机中心对准目标方块。

        start_pose 是 choose_block_rough_camera_pose 输出的观察位姿。
        这里委托 visual_servo_common.run_offset_visual_servo_alignment 执行闭环：
        图像节点负责 get_visual_servo_offset(target_type="block")，
        公共函数负责像素误差到机械臂 XY 修正。
        high_angle_deg 来自高位识别返回的模板角度 t，用于低位只生成 t 附近的少量模板。
        返回格式固定为 success, camera_pose, message，供 down_pick_visual 统一处理。
        """
        if visual_servo_offset is None:
            return False, list(start_pose), "视觉抓取未启用，未创建 get_visual_servo_offset 服务代理"
        if run_offset_visual_servo_alignment is None:
            return False, list(start_pose), "视觉伺服公共函数不可用，无法执行闭环对准"
        if not visual_servo_config:
            return False, list(start_pose), "视觉伺服配置为空，无法执行闭环对准"

        request_kwargs = {
            "expected_category": expected_category,
            "template_profile": BLOCK_TEMPLATE_PROFILE,
            "use_angle_prior": BLOCK_USE_ANGLE_PRIOR,
            "angle_center_deg": float(high_angle_deg),
            "angle_window_deg": BLOCK_ANGLE_WINDOW_DEG,
            "angle_step_deg": BLOCK_ANGLE_STEP_DEG,
            "use_position_prior": BLOCK_USE_POSITION_PRIOR,
            "search_center_x": 0.0,
            "search_center_y": 0.0,
            "search_radius_px": BLOCK_SEARCH_RADIUS_PX,
        }
        get_offset_func = lambda: request_block_visual_offset(**request_kwargs)
        success, camera_pose, last_resp, message = run_offset_visual_servo_alignment(
            get_offset_func,
            move_pose_for_visual_servo,
            start_pose,
            visual_servo_config,
            speed=SERVO_SPEED,
            error_threshold_px=VISUAL_ERROR_THRESHOLD_PX,
            max_step_mm=VISUAL_MAX_STEP_MM,
            min_step_mm=VISUAL_MIN_STEP_MM,
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

    def get_place_target_grid(index_cube):
        """获取当前方块要摆放到的托盘逻辑格点。

        基础任务中，本地 pick_list 的第 0 列是 col，第 1 列是 row。
        进阶任务时 process.py 内部可能根据规划结果改写 pick_list；后续如果要完整支持进阶任务，
        需要让图像/规划节点显式返回当前 index_cube 对应的 row/col，不能只依赖本地静态表。
        返回顺序固定为 row, col，供 request_board_visual_offset 使用。
        """
        if is_advanced_task:
            raise RuntimeError("进阶任务的摆放表由 process.py 动态生成，competition.py 不能使用本地静态 row/col")
        if index_cube < 0 or index_cube >= len(pick_list):
            raise IndexError(f"方块序号越界: {index_cube}，当前基础任务摆放表长度: {len(pick_list)}")
        place_item = pick_list[index_cube]
        col = float(place_item[0])
        row = float(place_item[1])
        return row, col

    def get_expected_block_category(index_cube):
        """获取视觉抓取时传给图像节点的方块类别先验。"""
        configured_category = EXPECTED_BLOCK_CATEGORY.strip()
        if configured_category:
            return configured_category
        if is_advanced_task:
            return ""
        if 0 <= index_cube < len(pick_list):
            return str(pick_list[index_cube][3])
        return ""

    def align_camera_to_board_target(start_pose, row, col):
        """用托盘视觉伺服把相机中心对准目标格点。

        图像节点通过 get_visual_servo_offset(target_type="board", row=row, col=col)
        返回目标格点相对相机中心的像素偏差；闭环运动逻辑与方块完全共用。
        """
        if visual_servo_offset is None:
            return False, list(start_pose), "视觉摆放未启用，未创建 get_visual_servo_offset 服务代理"
        if run_offset_visual_servo_alignment is None:
            return False, list(start_pose), "视觉伺服公共函数不可用，无法执行托盘闭环对准"
        if not visual_servo_config:
            return False, list(start_pose), "视觉伺服配置为空，无法执行托盘闭环对准"

        get_offset_func = lambda: request_board_visual_offset(row, col)
        success, camera_pose, last_resp, message = run_offset_visual_servo_alignment(
            get_offset_func,
            move_pose_for_visual_servo,
            start_pose,
            visual_servo_config,
            speed=SERVO_SPEED,
            error_threshold_px=VISUAL_ERROR_THRESHOLD_PX,
            max_step_mm=VISUAL_MAX_STEP_MM,
            min_step_mm=VISUAL_MIN_STEP_MM,
            max_iter=VISUAL_MAX_ITER,
            success_stable_frames=VISUAL_SUCCESS_STABLE_FRAMES,
            max_missed_frames=VISUAL_MAX_MISSED_FRAMES,
            settle_sec=VISUAL_SETTLE_SEC,
        )
        if not success:
            return False, list(camera_pose), message
        return True, list(camera_pose), message


    def choose_board_place_heights(index_cube, row, col):
        """集中管理托盘摆放高位、下放高度和抬起高度。

        当前先沿用旧开环摆放高度：高位 PLACE_HIGH_Z、下放 PLACE_DOWN_Z、释放后抬起 PLACE_LIFT_STEP_MM。
        后续如果不同层数、不同方块或不同 row/col 需要不同高度，只改这里。
        """
        print("\033[93摆放采用旧高度\033[0m")
        return float(PLACE_HIGH_Z), float(PLACE_DOWN_Z), float(PLACE_LIFT_STEP_MM)

    def verify_place_before_down(place_high_pose, index_cube):
        """托盘下放前的安全确认入口。"""
        print(f"第 {index_cube} 个方块托盘摆放高位: {place_high_pose}")
        answer = input("确认托盘摆放高位安全后按回车下放；输入 n 取消本次摆放: ")
        return answer.strip().lower() != "n"

    def handle_place_servo_failed(index_cube, message):
        """处理托盘视觉摆放失败。当前策略是保守中断，不下放。"""
        error_message = f"第 {index_cube} 个方块托盘视觉摆放失败: {message}"
        print(f"\033[91m{error_message}\033[0m")
        raise RuntimeError(error_message)

    def down_place_visual(put_pose, row, col, theta_place, index_cube):
        """托盘视觉伺服摆放骨架。

        流程：粗观察位 -> 对准托盘目标格点 -> 换算持块摆放位 -> 下放 -> 喷气 -> 抬起。
        put_pose 由 get_put_pose 返回，是基于高位托盘140格点计算出的相机粗观察位。
        当前持块中心偏移仍是占位函数，闭环成功后仍会保守中断，不自动下放。
        """
        rough_camera_pose = list(put_pose)
        print(f"第 {index_cube} 个方块托盘视觉摆放粗观察位: {rough_camera_pose}，目标行列=({row}, {col})")
        send_arm_pose(rough_camera_pose, speed=ARM_SPEED, theta_deg=theta_place, label="托盘视觉摆放粗定位")
        success, camera_pose, message = align_camera_to_board_target(rough_camera_pose, row, col)
        if not success:
            handle_place_servo_failed(index_cube, message)

        success, place_pose, message = make_sucker_pose_from_camera_pose(camera_pose)
        if not success:
            handle_place_servo_failed(index_cube, message)

        place_high_z, place_down_z, lift_step_mm = choose_board_place_heights(index_cube, row, col)
        place_high_pose = list(place_pose)
        place_high_pose[2] = place_high_z
        place_high_pose[5] = shooting_angle[5]
        send_arm_pose(place_high_pose, speed=ARM_SPEED, theta_deg=theta_place, label="托盘视觉摆放高位")

        # if not verify_place_before_down(place_high_pose, index_cube):
        #     handle_place_servo_failed(index_cube, "人工取消托盘下放")

        place_down_pose = list(place_high_pose)
        place_down_pose[2] = place_down_z
        send_arm_pose(place_down_pose, speed=PICK_SPEED, theta_deg=theta_place, label="托盘视觉摆放下放")
        # input("放")
        set_sucker_state(suck_out)

        place_lift_pose = list(place_down_pose)
        place_lift_pose[2] = place_lift_pose[2] + lift_step_mm
        send_arm_pose(place_lift_pose, speed=ARM_SPEED, theta_deg=theta_place, label="托盘视觉摆放抬起")

    def down_place_open_loop(put_pose, theta_place):
        """旧开环摆放流程，视觉摆放未启用时继续使用。"""
        put_pose = list(put_pose)
        put_pose[2] = PLACE_HIGH_Z
        put_pose[5] = shooting_angle[5]
        send_arm_pose(put_pose, speed=ARM_SPEED, theta_deg=theta_place, label="摆放上方")

        put_pose[2] = PLACE_DOWN_Z
        put_pose[5] = shooting_angle[5]
        send_arm_pose(put_pose, speed=PICK_SPEED, theta_deg=theta_place, label="摆放下放")
        input("放")
        set_sucker_state(suck_out)

        put_pose[2] = put_pose[2] + PLACE_LIFT_STEP_MM
        send_arm_pose(put_pose, speed=ARM_SPEED, theta_deg=theta_place, label="摆放抬起")

    #################################           复位              ################################
    send_arm_pose(shooting_angle, speed=ARM_SPEED)

    #################################           获取图像与摆放方法             ################################
    print("是否进行进阶任务")
    jinjie = input()
    if jinjie == "y":
        GetTargetPos_req.num = -2
        is_advanced_task = True

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
        if input("\033[93m识别结果满意扣1\033[0m") == "1":
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

    def down_pick_visual(x, y, z, t, xuanzhuan_angle, index_cube, expected_category=""):
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

        # 5. 闭环伺服相机中心对准方块。
        success, camera_pose, message = align_camera_to_block(
            rough_camera_pose,
            expected_category=expected_category,
            high_angle_deg=t,
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

        # 8. 下探前留出安全确认入口，避免未实测参数直接撞桌面。已经稳定实现，不需要暂停
        # if not verify_pick_before_down(sucker_high_pose, index_cube):
        #     handle_block_servo_failed(index_cube, "人工取消下探吸取")

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
        fallback_category = get_expected_block_category(i)
        x, y, z, t, xuanzhuan_angle, high_category = parse_cube_location_response(
            resp.array,
            fallback_category=fallback_category,
        )

        if USE_VISUAL_PICK:
            # 显式配置优先；正常比赛流程优先使用 process.py 当前任务返回的高位类别。
            expected_category = EXPECTED_BLOCK_CATEGORY.strip() or high_category
            theta_pick, theta_place, set_angle = down_pick_visual(
                x,
                y,
                z,
                t,
                xuanzhuan_angle,
                i,
                expected_category=expected_category,
            )
        else:
            theta_pick, theta_place, set_angle = down_pick(x, y, z, t, xuanzhuan_angle, i)

        if USE_VISUAL_PLACE:
            # get_put_pose 当前返回托盘目标格点粗观察位，是视觉摆放的相机起始观察位。
            resp = get_put_pose.call(GetTargetPos_req)
            if len(resp.array) < 6:
                raise RuntimeError(f"第 {i} 个方块未获取到有效托盘粗观察位")
            put_pose = list(resp.array)
            row, col = get_place_target_grid(i)
            down_place_visual(put_pose, row, col, theta_place, i)
        else:
            raise RuntimeError("旧开环摆放已停用：get_put_pose 当前返回托盘视觉粗观察位")

        i = i + 1
        if i == 34:  # 捡完34个方块结束。
            break

    set_sucker_state(sucker_off)
