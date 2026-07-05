#!/home/zhl/fr3env/fr3env/bin/python
import time

import rospy

from control.srv import arm
from image_process.srv import VisualTargetOffset
from visual_servo_common import (
    VISUAL_SERVO_CONFIG_PATH,
    apply_camera_to_sucker_offset,
    load_visual_servo_config,
    move_arm,
    run_visual_servo_alignment,
    save_visual_servo_config,
)


# ===== 手动参数区：需要改参数时直接改这里 =====
# 期望识别的方块类别；空字符串表示不限制类别，画面里只有一个方块时建议保持为空。
EXPECTED_CATEGORY = ""

# 相机对准方块时使用的观察位姿 [x, y, z, rx, ry, rz]，单位 mm/deg。
# 这个高度应和最终视觉伺服验证高度接近，否则相机到吸盘偏移可能不一致。
SERVO_LOOK_POSE = [-251.979, 22.147, 200.723, -180, 0, 90]

# 人工修正相机到吸盘的 XY 偏移，单位 mm。
# 运行后观察吸盘偏差：吸盘需要往机械臂 +X/+Y 移动多少，就填多少。
MANUAL_OFFSET_ADJUST_MM = [0.0, 0.0]

# 视觉伺服对准和检查位移动速度，首次调试建议保守。
ARM_SPEED = 45

# 像素误差阈值，单位 px；连续多帧都小于这个值才认为相机已对准。
ERROR_THRESHOLD_PX = 2.0

# 单次视觉伺服最大 XY 修正距离，单位 mm；用于限制识别异常时的运动幅度。
MAX_STEP_MM = 5.0

# 闭环最多迭代次数；每次迭代最多移动一次，也可能只是等待稳定帧。
MAX_ITER = 40

# 连续多少帧满足误差阈值才判定成功，避免单帧偶然抖动导致误判。
SUCCESS_STABLE_FRAMES = 5

# 连续多少帧未识别才判定失败；单帧未识别只跳过。
MAX_MISSED_FRAMES = 5


def main():
    rospy.init_node("calibrate_camera_sucker_offset")
    rospy.wait_for_service("arm_control")
    rospy.wait_for_service("get_visual_target_offset")
    arm_client = rospy.ServiceProxy("arm_control", arm)
    offset_client = rospy.ServiceProxy("get_visual_target_offset", VisualTargetOffset)
    config = load_visual_servo_config()

    input("确认机械臂运动区域安全、画面里只有一个方块后按回车开始相机到吸盘偏移标定...")
    move_arm(arm_client, SERVO_LOOK_POSE, speed=ARM_SPEED, wait_sec=0.7)
    success, camera_pose, resp, message = run_visual_servo_alignment(
        arm_client,
        offset_client,
        SERVO_LOOK_POSE,
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
        raise RuntimeError(f"视觉伺服未对准，不能标定相机到吸盘偏移: {message}")

    old_offset = config.get("camera_to_sucker_offset_mm", [0.0, 0.0])
    new_offset = [
        float(old_offset[0]) + float(MANUAL_OFFSET_ADJUST_MM[0]),
        float(old_offset[1]) + float(MANUAL_OFFSET_ADJUST_MM[1]),
    ]
    config["camera_to_sucker_offset_mm"] = new_offset
    config.setdefault("last_calibration", {})
    config["last_calibration"]["camera_sucker_offset"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_visual_servo_config(config)

    sucker_pose = apply_camera_to_sucker_offset(camera_pose, config)
    print(f"旧偏移: {old_offset}, 本次人工修正: {MANUAL_OFFSET_ADJUST_MM}, 新偏移: {new_offset}")
    print(f"移动到吸盘标定检查位: {sucker_pose}")
    move_arm(arm_client, sucker_pose, speed=ARM_SPEED, wait_sec=0.7)
    print("请观察吸盘是否对准方块；不准就修改 MANUAL_OFFSET_ADJUST_MM 后重新运行。")
    print(f"配置文件: {VISUAL_SERVO_CONFIG_PATH}")


if __name__ == "__main__":
    main()
