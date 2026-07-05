#!/home/zhl/fr3env/fr3env/bin/python
import copy

import numpy as np
import rospy

from control.srv import arm, suck
from image_process.srv import VisualTargetOffset
from visual_servo_common import (
    apply_camera_to_sucker_offset,
    call_target_offset,
    load_visual_servo_config,
    move_arm,
    run_visual_servo_alignment,
    set_sucker,
)


# ===== 手动参数区：需要改参数时直接改这里 =====
# 期望识别的方块类别；空字符串表示不限制类别，画面里只有一个方块时建议保持为空。
EXPECTED_CATEGORY = ""

# 高位粗识别位姿 [x, y, z, rx, ry, rz]，单位 mm/deg。
# 用来模拟全场识别，要求视野能看到方块，不要求毫米级准确。
HIGH_LOOK_POSE = [-250.4151306152343, 22.14801216125488, 380.3343505859375, -180, 0, 90]

# 进入闭环视觉伺服时的相机高度，单位 mm。
# 需要和你人工记录像素/机械臂移动数据时的相机高度接近。
SERVO_LOOK_Z = 200.0

# 吸取下探高度，单位 mm；必须先确认不会撞桌面或压坏方块。
PICK_Z = 180

# 吸取后抬起高度，单位 mm；应高于方块和周边障碍。
LIFT_Z = 180

# 普通移动速度；首次调试建议保守。
ARM_SPEED = 30

# 下探吸取速度；建议低于普通移动速度。
PICK_SPEED = 30

# 高位粗定位比例，只用于把目标大致拉到画面中心；方向不对就改正负号。
# 机械臂 X 方向每 1 像素误差对应的开环移动量，单位 mm/px。
HIGH_ROUGH_X_MM_PER_PIXEL = 0.5

# 机械臂 Y 方向每 1 像素误差对应的开环移动量，单位 mm/px。
HIGH_ROUGH_Y_MM_PER_PIXEL = 0.5

# 高位开环粗定位单次最大移动距离，单位 mm；用于防止比例或方向填错后运动过大。
MAX_ROUGH_STEP_MM = 500.0

# 闭环对准误差阈值，单位 px；连续多帧都小于这个值才判定成功。
ERROR_THRESHOLD_PX = 2.0

# 闭环单次最大 XY 修正距离，单位 mm；限制识别异常时的运动幅度。
MAX_STEP_MM = 5.0

# 闭环最多迭代次数；每次迭代最多移动一次，也可能只是等待稳定帧。
MAX_ITER = 40

# 连续多少帧满足误差阈值才判定成功，避免单帧偶然抖动导致误判。
SUCCESS_STABLE_FRAMES = 5

# 连续多少帧未识别才判定失败；单帧未识别只跳过。
MAX_MISSED_FRAMES = 5

# 下探吸取前是否要求人工按回车确认；首次实机测试建议保持 True。
REQUIRE_ENTER_BEFORE_PICK = True

# 吸盘吸气状态编号，对应 control/scripts/controller.py。
SUCK_IN = 0

# 吸盘喷气状态编号，对应 control/scripts/controller.py。
SUCK_OUT = 1

# 吸盘关闭状态编号，对应 control/scripts/controller.py。
SUCKER_OFF = 2


def limit_rough_delta(delta_xy):
    """限制高位开环粗定位的移动距离，避免比例或方向填错时运动过大。"""
    delta_xy = np.array(delta_xy, dtype=float)
    norm = float(np.linalg.norm(delta_xy))
    if norm <= MAX_ROUGH_STEP_MM or norm <= 1e-9:
        return delta_xy
    return delta_xy / norm * MAX_ROUGH_STEP_MM


def main():
    rospy.init_node("test_single_block_visual_servo")
    rospy.wait_for_service("arm_control")
    rospy.wait_for_service("suck_control")
    rospy.wait_for_service("get_visual_target_offset")
    arm_client = rospy.ServiceProxy("arm_control", arm)
    suck_client = rospy.ServiceProxy("suck_control", suck)
    offset_client = rospy.ServiceProxy("get_visual_target_offset", VisualTargetOffset)
    config = load_visual_servo_config()

    input("确认机械臂运动区域安全、画面里只有一个方块后按回车开始单方块视觉伺服验证...")
    set_sucker(suck_client, SUCKER_OFF)

    high_pose = list(HIGH_LOOK_POSE)
    move_arm(arm_client, high_pose, speed=ARM_SPEED, wait_sec=0.8)
    first_resp = call_target_offset(offset_client, EXPECTED_CATEGORY)
    if not first_resp.found:
        raise RuntimeError(f"高位粗识别失败: {first_resp.message}")

    rough_delta = limit_rough_delta([
        first_resp.dy_px * HIGH_ROUGH_Y_MM_PER_PIXEL,
        first_resp.dx_px * HIGH_ROUGH_X_MM_PER_PIXEL,
    ])
    rough_pose = copy.deepcopy(high_pose)
    rough_pose[0] += float(rough_delta[0])
    rough_pose[1] += float(rough_delta[1])
    print(
        f"高位粗定位: 像素误差=({first_resp.dx_px:.2f},{first_resp.dy_px:.2f})px，"
        f"粗移动=({rough_delta[0]:.3f},{rough_delta[1]:.3f})mm"
    )
    move_arm(arm_client, rough_pose, speed=ARM_SPEED, wait_sec=0.7)

    servo_pose = list(rough_pose)
    servo_pose[2] = SERVO_LOOK_Z
    move_arm(arm_client, servo_pose, speed=ARM_SPEED, wait_sec=0.8)

    success, camera_pose, last_resp, message = run_visual_servo_alignment(
        arm_client,
        offset_client,
        servo_pose,
        config,
        expected_category=EXPECTED_CATEGORY,
        speed=ARM_SPEED,
        error_threshold_px=ERROR_THRESHOLD_PX,
        max_step_mm=MAX_STEP_MM,
        max_iter=MAX_ITER,
        success_stable_frames=SUCCESS_STABLE_FRAMES,
        max_missed_frames=MAX_MISSED_FRAMES,
    )
    if not success:
        print(f"视觉伺服失败: {message}，不下探吸取。")
        return

    sucker_high_pose = apply_camera_to_sucker_offset(camera_pose, config)
    sucker_high_pose[2] = LIFT_Z
    print(f"视觉伺服成功，移动到吸盘高位: {sucker_high_pose}")
    move_arm(arm_client, sucker_high_pose, speed=ARM_SPEED, wait_sec=0.5)

    pick_pose = list(sucker_high_pose)
    pick_pose[2] = PICK_Z
    if REQUIRE_ENTER_BEFORE_PICK:
        input("确认吸盘高位看起来安全后按回车下探吸取...")
    move_arm(arm_client, pick_pose, speed=PICK_SPEED, wait_sec=0.3)
    set_sucker(suck_client, SUCK_IN)

    lift_pose = list(sucker_high_pose)
    lift_pose[2] = LIFT_Z
    move_arm(arm_client, lift_pose, speed=ARM_SPEED, wait_sec=0.5)
    print("单方块视觉伺服抓取验证完成。")


if __name__ == "__main__":
    main()
