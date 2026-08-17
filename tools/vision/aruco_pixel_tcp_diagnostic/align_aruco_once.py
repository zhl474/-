#!/home/zhl/fr3env/fr3env/bin/python
"""单次 ArUco 高位粗定位与低位视觉伺服对准。

本脚本只执行一次对准，除伺服过程录像外不保存图片、CSV 或标定数据。
高位识别帧与低位闭环每轮各写入一帧录像（叠加轮次/误差/标记/中心参考），
对准成功或失败都会落盘并打印路径；随后读取并打印 TCP 位姿后退出，退出前
不会返回高位，机械臂保持在最终对准位置。

运行前提：
- camera_node 与 controller 已启动；
- ArUco 板在高位和低位相机视野内；
- 字典为 DICT_6X6_50，标记 ID=0（可在下方参数区修改）。
"""

import argparse
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rospy
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# 项目根目录与包搜索路径必须在正式包导入前设置。
SRC_ROOT = Path(__file__).resolve().parents[3]
for _path in (SRC_ROOT, SRC_ROOT / "image_process"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from camera.srv import GetStableWorldPoints  # noqa: E402
from competition.competition_lib.config import (  # noqa: E402
    load_execution_config,
    load_visual_servo_config,
)
from competition.competition_lib.visual_servo import (  # noqa: E402
    run_offset_visual_servo_alignment,
)
from control.srv import (  # noqa: E402
    GetActualPose,
    GetActualPoseRequest,
    MoveArm,
    MoveArmRequest,
)
from image_process_lib.depth_rough_localization import DepthRoughLocalizer  # noqa: E402

from aruco_diagnostic_core import (  # noqa: E402
    CenterRefineConfig,
    OffsetResult,
    check_image_size_constant,
    create_aruco_detector,
    detect_aruco_center,
    draw_servo_overlay,
    median_center,
    open_video_writer,
    validate_motion_pose,
)

# ==================== 运行参数（直接修改后运行）====================
ARUCO_DICT_NAME = "DICT_6X6_50"  # ArUco 字典名称
ARUCO_MARKER_ID = 0  # 目标标记 ID
LOW_TCP_Z_MM = 166.26  # 低位固定 TCP Z = 方块观察 173.46 - 板面低于方块顶面的 7.2mm，配平相机到目标面的距离
ADAPTIVE_THRESH_MAX = 150  # 自适应阈值窗口上限；低位约 10cm 距离时码格约 82px，默认 63 会"找到外框但解码失败"
HIGH_SAMPLE_FRAMES = 7  # 高位有效识别帧数，中心逐轴取中位数
HIGH_DETECT_MAX_RETRIES = 30  # 高位连续失败上限
IMAGE_TIMEOUT_SEC = 1.0  # 获取一张新图像的超时时间（秒）
DEPTH_FRAME_COUNT = 15  # 稳定深度采集帧数
DEPTH_MIN_VALID_FRAMES = 10  # 深度最少有效帧数
DEPTH_CAPTURE_TIMEOUT_SEC = 2.0  # 稳定深度采集超时（秒）
DEPTH_MAX_MAD_MM = 1.0  # 深度跨帧 MAD 上限（mm）
SERVO_ERROR_THRESHOLD_PX = 1.0  # 闭环对准成功阈值（px）
IMAGE_TOPIC = "/camera/image_rect"  # 去畸变相机图像话题
REQUIRE_START_CONFIRMATION = True  # True 时，发送第一条运动命令前等待确认
SERVO_VIDEO_ENABLED = True  # 是否录制对准过程相机画面视频
SERVO_VIDEO_FPS = 10.0  # 录像帧率（每轮写入一帧，仅影响播放速度）
VIDEO_OUTPUT_ROOT = Path("/home/zhl/桌面/aruco单次对准")  # 录像输出根目录，与批量实验目录分开

CENTER_REFINE_CONFIG = CenterRefineConfig(
    window_ratio=0.025,
    min_window_px=3,
    max_window_px=15,
    max_shift_cell_ratio=0.12,
    min_contrast=5.0,
)

EXECUTION_CONFIG_PATH = SRC_ROOT / "competition" / "config" / "execution.yaml"
VISUAL_SERVO_CONFIG_PATH = SRC_ROOT / "competition" / "config" / "visual_servo.yaml"
PERCEPTION_CONFIG_PATH = SRC_ROOT / "image_process" / "config" / "perception.yaml"
HAND_EYE_PATH = SRC_ROOT / "camera" / "config" / "T_wrist2camera.npy"


class FreshImageReader:
    """读取严格晚于请求时刻的新相机帧。"""

    def __init__(self, topic=IMAGE_TOPIC):
        self._bridge = CvBridge()
        self._condition = threading.Condition()
        self._latest = {}
        self.expected_size = None
        self._subscriber = rospy.Subscriber(topic, Image, self._callback, queue_size=1)

    def _callback(self, message):
        image = self._bridge.imgmsg_to_cv2(message, "bgr8")
        stamp = getattr(message.header, "stamp", None)
        with self._condition:
            self._latest = {"image": image, "stamp": stamp}
            self._condition.notify_all()

    def fresh_image(self, timeout_sec):
        """返回 ``(图像, None)`` 或 ``(None, 失败原因)``。"""
        request_stamp = rospy.Time.now()
        with self._condition:
            deadline = time.monotonic() + float(timeout_sec)
            while True:
                image = self._latest.get("image")
                stamp = self._latest.get("stamp")
                if image is not None and stamp is not None and stamp > request_stamp:
                    height, width = image.shape[:2]
                    current_size = (int(width), int(height))
                    if self.expected_size is None:
                        self.expected_size = current_size
                    elif not check_image_size_constant(
                        width, height, self.expected_size
                    ):
                        return None, (
                            f"图像尺寸变化：期望 {self.expected_size}，"
                            f"实际 {current_size}"
                        )
                    return image.copy(), None
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None, "图像获取超时"
                self._condition.wait(remaining)


class RosServices:
    """单次对准需要的 ROS 服务客户端。"""

    def __init__(self):
        rospy.wait_for_service("/control/move_arm", timeout=10.0)
        rospy.wait_for_service("/control/get_actual_pose", timeout=10.0)
        rospy.wait_for_service("/camera/stable_world_points", timeout=10.0)
        self._move_arm = rospy.ServiceProxy("/control/move_arm", MoveArm)
        self._get_actual_pose = rospy.ServiceProxy(
            "/control/get_actual_pose", GetActualPose
        )
        self._stable_world_points = rospy.ServiceProxy(
            "/camera/stable_world_points", GetStableWorldPoints
        )

    def move_to(self, pose, speed, wait_until_stable=True):
        request = MoveArmRequest(
            pose=[float(value) for value in pose],
            speed=int(speed),
            wait_until_stable=bool(wait_until_stable),
            blend_enabled=False,
            blend_radius_mm=0.0,
        )
        response = self._move_arm(request)
        if not response.success:
            raise RuntimeError(f"move_arm 失败：{response.message}")

    def get_actual_pose(self):
        response = self._get_actual_pose(GetActualPoseRequest())
        if not response.success:
            raise RuntimeError(f"get_actual_pose 失败：{response.message}")
        return list(response.tcp_pose)

    def stable_world_point(self, x, y):
        response = self._stable_world_points(
            [int(round(x))],
            [int(round(y))],
            DEPTH_FRAME_COUNT,
            DEPTH_MIN_VALID_FRAMES,
            DEPTH_CAPTURE_TIMEOUT_SEC,
        )
        if not response.success:
            raise RuntimeError(f"稳定深度服务失败：{response.message}")
        valid_frames = (
            int(response.valid_frame_counts[0])
            if response.valid_frame_counts
            else 0
        )
        if not response.point_valid or not response.point_valid[0]:
            raise RuntimeError(f"深度点无效（有效帧 {valid_frames}）")
        depth_mad_mm = float(response.depth_mad_mm[0])
        if not np.isfinite(depth_mad_mm) or depth_mad_mm > DEPTH_MAX_MAD_MM:
            raise RuntimeError(
                f"深度 MAD={depth_mad_mm:.3f}mm 超过上限 "
                f"{DEPTH_MAX_MAD_MM:.3f}mm"
            )
        world_xyz = [float(value) for value in response.world_xyz[:3]]
        if len(world_xyz) != 3 or not np.all(np.isfinite(world_xyz)):
            raise RuntimeError("稳定深度服务返回的世界坐标无效")
        return world_xyz, valid_frames, depth_mad_mm


def detect_target_center(image, detector):
    """检测唯一目标 ArUco，并返回经过校验的中央精定位结果。"""
    return detect_aruco_center(
        image,
        detector,
        ARUCO_MARKER_ID,
        refine_config=CENTER_REFINE_CONFIG,
    )


def move_checked(
    services,
    pose,
    speed,
    minimum_tcp_z_mm,
    safe_x_range_mm,
    safe_y_range_mm,
    wait_until_stable=True,
):
    """安全检查通过后发送运动命令。"""
    ok, reason = validate_motion_pose(
        pose, minimum_tcp_z_mm, safe_x_range_mm, safe_y_range_mm
    )
    if not ok:
        raise RuntimeError(f"运动位姿安全检查未通过：{reason}")
    services.move_to(pose, speed, wait_until_stable=wait_until_stable)


def format_pose(pose):
    """以便于核对的固定格式输出六维 TCP 位姿。"""
    names = ("X", "Y", "Z", "R", "P", "YAW")
    return ", ".join(
        f"{name}={float(value):.3f}" for name, value in zip(names, pose)
    )


def build_video_path():
    """在独立输出根目录下生成本次对准的唯一录像文件路径。"""
    VIDEO_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    base_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_name = base_name
    suffix = 2
    while any(
        (VIDEO_OUTPUT_ROOT / f"{unique_name}_单次对准录像{ext}").exists()
        for ext in (".avi", ".mp4")
    ):
        unique_name = f"{base_name}-{suffix}"
        suffix += 1
    return VIDEO_OUTPUT_ROOT / f"{unique_name}_单次对准录像.avi"


def run_alignment(low_tcp_z_mm=None, assume_yes=False):
    """执行一次完整对准，成功后保持最终位置。

    ``low_tcp_z_mm`` 覆盖低位固定高度；``assume_yes=True`` 跳过控制台
    交互确认，供网页控制台等自动化调用方使用。
    """
    execution_config = load_execution_config(str(EXECUTION_CONFIG_PATH))
    visual_config = load_visual_servo_config(str(VISUAL_SERVO_CONFIG_PATH))
    with PERCEPTION_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        perception_config = yaml.safe_load(config_file) or {}

    high_localization = perception_config.get("high_tcp_localization", {})
    safe_x_range_mm = high_localization.get("safe_x_range_mm", [-1e9, 1e9])
    safe_y_range_mm = high_localization.get("safe_y_range_mm", [-1e9, 1e9])
    shooting_pose = list(execution_config.shooting_pose)
    minimum_tcp_z_mm = float(execution_config.minimum_tcp_z_mm)

    ok, reason = validate_motion_pose(
        shooting_pose, minimum_tcp_z_mm, safe_x_range_mm, safe_y_range_mm
    )
    if not ok:
        raise RuntimeError(f"高位拍摄位姿安全检查未通过：{reason}")
    low_tcp_z_mm = float(
        LOW_TCP_Z_MM if low_tcp_z_mm is None else low_tcp_z_mm
    )
    if not np.isfinite(low_tcp_z_mm) or low_tcp_z_mm < minimum_tcp_z_mm:
        raise RuntimeError(
            f"低位 TCP Z={low_tcp_z_mm}mm 低于安全下限 {minimum_tcp_z_mm}mm"
        )

    if REQUIRE_START_CONFIRMATION and not assume_yes:
        print("程序将运动机械臂到高位并执行一次 ArUco 对准。")
        answer = input("确认工作空间安全后按回车继续；输入 q 取消：").strip().lower()
        if answer == "q":
            print("已取消，未发送运动命令。")
            return
        if answer:
            raise RuntimeError("输入无效：仅接受空回车或 q")
    elif REQUIRE_START_CONFIRMATION:
        print("调用方已确认工作空间安全，跳过交互确认。")

    services = RosServices()
    reader = FreshImageReader()
    detector = create_aruco_detector(
        ARUCO_DICT_NAME, adaptive_thresh_max=ADAPTIVE_THRESH_MAX
    )
    localizer = DepthRoughLocalizer(shooting_pose, np.load(HAND_EYE_PATH))

    move_checked(
        services,
        shooting_pose,
        execution_config.arm_speed,
        minimum_tcp_z_mm,
        safe_x_range_mm,
        safe_y_range_mm,
    )
    print(f"已到达高位拍摄位姿：{format_pose(shooting_pose)}")

    video_writer = None
    video_actual_path = None
    video_open_failed = False

    def write_video_frame(image, round_no, result):
        """把一帧叠加诊断信息的相机画面写入本次对准录像。"""
        nonlocal video_writer, video_actual_path, video_open_failed
        if not SERVO_VIDEO_ENABLED or image is None or video_open_failed:
            return
        if video_writer is None:
            video_writer, video_actual_path = open_video_writer(
                build_video_path(),
                SERVO_VIDEO_FPS,
                (image.shape[1], image.shape[0]),
            )
            if video_writer is None:
                video_open_failed = True
                print("警告：对准录像无法打开，继续执行对准。")
                return
        video_writer.write(
            draw_servo_overlay(
                image,
                round_no=round_no,
                error_xy=(result.dx_px, result.dy_px) if result.found else None,
                center_uv=result.center_uv if result.found else None,
                rough_center_uv=result.rough_center_uv,
                refine_delta_uv=result.refine_delta_uv,
                center_contrast=result.center_contrast,
                corners=result.corners,
                rejected_corners=result.rejected_corners,
            )
        )

    try:
        # 高位连续识别多帧，使用中位数抑制单帧像素抖动。
        high_centers = []
        consecutive_failures = 0
        while len(high_centers) < HIGH_SAMPLE_FRAMES:
            image, error = reader.fresh_image(IMAGE_TIMEOUT_SEC)
            if image is None:
                consecutive_failures += 1
                failure_message = error
            else:
                result = detect_target_center(image, detector)
                write_video_frame(image, "高位", result)
                if result.found:
                    high_centers.append(result.center_uv)
                    consecutive_failures = 0
                    continue
                consecutive_failures += 1
                failure_message = result.message
            if consecutive_failures >= HIGH_DETECT_MAX_RETRIES:
                raise RuntimeError(f"高位识别失败：{failure_message}")

        high_center = median_center(high_centers)
        print(
            f"高位识别成功：中心像素=({high_center[0]:.3f}, "
            f"{high_center[1]:.3f})，有效帧={len(high_centers)}"
        )

        world_xyz, valid_frames, depth_mad_mm = services.stable_world_point(
            high_center[0], high_center[1]
        )
        print(
            "高位深度定位成功："
            f"世界坐标=({world_xyz[0]:.3f}, {world_xyz[1]:.3f}, "
            f"{world_xyz[2]:.3f})mm，有效帧={valid_frames}，"
            f"MAD={depth_mad_mm:.3f}mm"
        )

        tcp_xy = localizer.tcp_xy_from_world(world_xyz)
        low_pose = [
            float(tcp_xy[0]),
            float(tcp_xy[1]),
            low_tcp_z_mm,
            *[float(value) for value in shooting_pose[3:6]],
        ]
        move_checked(
            services,
            low_pose,
            execution_config.arm_speed,
            minimum_tcp_z_mm,
            safe_x_range_mm,
            safe_y_range_mm,
        )
        print(f"已到达低位粗定位位姿：{format_pose(low_pose)}")

        rounds_count = 0

        def get_offset():
            nonlocal rounds_count
            rounds_count += 1
            image, error = reader.fresh_image(IMAGE_TIMEOUT_SEC)
            if image is None:
                return OffsetResult(message=f"低位图像获取失败：{error}")
            result = detect_target_center(image, detector)
            write_video_frame(image, rounds_count, result)
            return result

        def move_pose_func(pose, speed, wait_sec=0.0, wait_until_stable=True):
            del wait_sec  # 兼容正式视觉伺服回调签名，稳定等待由正式逻辑处理。
            move_checked(
                services,
                pose,
                speed,
                minimum_tcp_z_mm,
                safe_x_range_mm,
                safe_y_range_mm,
                wait_until_stable=wait_until_stable,
            )

        success, final_command_pose, last_response, message = (
            run_offset_visual_servo_alignment(
                get_offset,
                move_pose_func,
                low_pose,
                visual_config,
                speed=int(execution_config.servo_speed),
                error_threshold_px=float(SERVO_ERROR_THRESHOLD_PX),
                max_step_mm=float(execution_config.max_step_mm),
                max_iter=int(execution_config.max_iter),
                success_stable_frames=int(execution_config.success_stable_frames),
                max_missed_frames=int(execution_config.max_missed_frames),
                settle_sec=float(execution_config.settle_sec),
                min_step_mm=float(execution_config.min_step_mm),
                log_label="ArUco 单次对准",
            )
        )
        if not success:
            raise RuntimeError(f"低位视觉伺服失败：{message}")

        # 对准成功后补一帧静止画面，方便确认最终对准状态。
        final_image, _final_error = reader.fresh_image(IMAGE_TIMEOUT_SEC)
        if final_image is not None:
            write_video_frame(
                final_image,
                "完成",
                detect_target_center(final_image, detector),
            )

        actual_tcp_pose = services.get_actual_pose()
        print("\nArUco 对准成功。")
        print(f"视觉伺服轮数：{rounds_count}")
        if last_response is not None:
            print(
                f"最终像素误差：X={float(last_response.dx_px):+.3f}px，"
                f"Y={float(last_response.dy_px):+.3f}px"
            )
        print(f"最终命令 TCP：{format_pose(final_command_pose)}")
        print(f"最终实测 TCP：{format_pose(actual_tcp_pose)}")
        print("脚本即将退出；不会返回高位，机械臂保持当前对准位置。")
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"对准录像已保存：{video_actual_path}")


def parse_args(argv=None):
    """解析命令行参数，保持交互式终端用法与网页调用兼容。"""
    parser = argparse.ArgumentParser(
        description="执行一次 ArUco 高位粗定位与低位视觉伺服对准。"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="跳过启动前的人工回车确认（仅用于网页控制台等已自行确认的调用方）",
    )
    parser.add_argument(
        "--low-tcp-z-mm",
        type=float,
        default=None,
        help=f"覆盖低位固定 TCP Z，默认 {LOW_TCP_Z_MM:g} mm",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rospy.init_node("aruco_align_once", anonymous=True)
    try:
        run_alignment(
            low_tcp_z_mm=args.low_tcp_z_mm,
            assume_yes=bool(args.yes),
        )
        return 0
    except KeyboardInterrupt:
        print("\n用户中断：停止发送新的运动命令，不执行自动复位。")
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"\nArUco 对准失败：{exc}", file=sys.stderr)
        print("未执行自动复位，请先确认机械臂当前实际位置。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
