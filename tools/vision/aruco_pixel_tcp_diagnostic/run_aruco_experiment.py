#!/home/zhl/fr3env/fr3env/bin/python
"""ArUco 高位像素—低位 TCP 独立诊断实验（实机入口）。

运行前提：
- 相机、深度、控制节点已启动（camera_node + controller）。
- ArUco 板（DICT_6X6_50，ID=0）平放在高、低位都能看到的区域。
- 只新增本工具目录，不修改任何正式 ROS 节点、服务、识别器或配置。

用法：直接运行本文件，按屏幕提示移动 ArUco 板；回车采样，输入 q 退出。
"""

import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import rospy
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# 项目根目录与包搜索路径（必须放在正式包导入之前，避免解析到 catkin devel 旧拷贝）。
SRC_ROOT = Path(__file__).resolve().parents[3]
for _path in (SRC_ROOT, SRC_ROOT / "image_process"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

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
    SetSuction,
    SetSuctionRequest,
)
from camera.srv import GetStableWorldPoints  # noqa: E402
from image_process_lib.depth_rough_localization import DepthRoughLocalizer  # noqa: E402

from aruco_diagnostic_core import (  # noqa: E402
    CenterRefineConfig,
    OffsetResult,
    check_image_size_constant,
    create_aruco_detector,
    detect_all_markers,
    detect_aruco_center,
    draw_aruco_debug,
    draw_rejected_candidates,
    draw_servo_overlay,
    image_center_uv,
    median_center,
    static_center_stats,
    validate_motion_pose,
    zero_error_tcp_xy,
)

# ==================== 运行参数（直接修改本文件后运行）====================
ARUCO_DICT_NAME = "DICT_6X6_50"  # ArUco 字典名称（与板子打印的字典一致）
ARUCO_MARKER_ID = 0  # 要追踪的标记 ID；只接受唯一匹配该 ID 的标记
CENTER_REFINE_WINDOW_RATIO = 0.025  # 中央角点搜索半窗口占 ArUco 平均边长比例
CENTER_REFINE_MIN_WINDOW_PX = 3  # 中央角点搜索半窗口下限（px）
CENTER_REFINE_MAX_WINDOW_PX = 15  # 中央角点搜索半窗口上限（px）
CENTER_REFINE_MAX_SHIFT_CELL_RATIO = 0.12  # 精中心相对粗中心最大允许修正（码格）
CENTER_REFINE_MIN_CONTRAST = 5.0  # 中央白格最暗值与黑格最亮值的最小灰度差
HIGH_SAMPLE_FRAMES = 7  # 高位有效采样帧数（逐轴取中位数得到高位中心）
LOW_TCP_Z_MM = 220.0  # 低位闭环时的固定 TCP Z（mm），不得低于 minimum_tcp_z_mm
STATIC_SAMPLE_FRAMES = 20  # 对准成功后静止采样帧数（不再移动机械臂）
STATIC_MIN_VALID_RATIO = 0.8  # 静止采样有效帧占比下限，低于此值不生成零误差等效 TCP
DEPTH_FRAME_COUNT = 15  # 稳定深度请求后需新采集的深度帧数
DEPTH_MIN_VALID_FRAMES = 10  # 深度点在 3x3 邻域至少多少帧有效才算该点有效
DEPTH_CAPTURE_TIMEOUT_SEC = 2.0  # 等待新深度帧的最大超时时间（秒）
DEPTH_MAX_MAD_MM = 1.0  # 深度跨帧 MAD 上限（mm），超过判定深度失败
IMAGE_TIMEOUT_SEC = 1.0  # 单次等待新相机帧的超时时间（秒）
SERVO_ERROR_THRESHOLD_PX = 1.0  # ArUco 低位闭环成功阈值（px），独立于方块和托盘配置
HIGH_DETECT_MAX_RETRIES = 30  # 高位连续取帧/识别失败多少次后放弃本样本
IMAGE_TOPIC = "/camera/image_rect"  # 去畸变相机图像话题（bgr8）
PROBE_DICT_NAMES = (  # 低位首帧诊断使用的多字典列表（第一个是正式检测字典）
    "DICT_6X6_50", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_5X5_50", "DICT_4X4_50", "DICT_ARUCO_ORIGINAL",
)
SERVO_VIDEO_ENABLED = True  # 是否录制低位伺服过程的相机画面视频
SERVO_VIDEO_FPS = 10.0  # 录像帧率（每伺服轮写入一帧，仅影响播放速度）
OUTPUT_ROOT = Path("/home/zhl/桌面/aruco诊断实验")  # 实验批次输出根目录
RESUME = False  # False 创建新批次；True 从 RESUME_OUTPUT_DIR 原目录继续
RESUME_OUTPUT_DIR = Path("/home/zhl/桌面/aruco诊断实验/20260808-220525")  # 续跑时改为已有批次目录，例如 Path("/home/zhl/桌面/aruco诊断实验/20260808-220525")

CENTER_REFINE_CONFIG = CenterRefineConfig(
    window_ratio=CENTER_REFINE_WINDOW_RATIO,
    min_window_px=CENTER_REFINE_MIN_WINDOW_PX,
    max_window_px=CENTER_REFINE_MAX_WINDOW_PX,
    max_shift_cell_ratio=CENTER_REFINE_MAX_SHIFT_CELL_RATIO,
    min_contrast=CENTER_REFINE_MIN_CONTRAST,
)

EXECUTION_CONFIG_PATH = SRC_ROOT / "competition" / "config" / "execution.yaml"  # 运动/伺服参数（shooting_pose、阈值、步长等）
VISUAL_SERVO_CONFIG_PATH = SRC_ROOT / "competition" / "config" / "visual_servo.yaml"  # 视觉伺服矩阵 pixel_to_robot_matrix
PERCEPTION_CONFIG_PATH = SRC_ROOT / "image_process" / "config" / "perception.yaml"  # 安全 XY 范围等感知配置

# ==================== CSV 列定义 ====================
CSV_COLUMNS = [
    "样本号", "事件", "时间戳",
    "高位检测像素X", "高位检测像素Y",
    "高位粗中心像素X", "高位粗中心像素Y",
    "高位中央修正量X", "高位中央修正量Y", "高位中心黑白分离度",
    "高位四角点",
    "低位四角点", "低位首帧检测ID", "低位首帧最大标记像素尺寸", "低位首帧rejected候选数",
    "高位世界坐标X", "高位世界坐标Y", "高位世界坐标Z",
    "深度有效帧数", "深度中位数毫米", "深度MAD毫米",
    "粗定位TCP位置X", "粗定位TCP位置Y",
    "最终命令TCP位置X", "最终命令TCP位置Y", "最终命令TCP位置Z",
    "实测TCP位置X", "实测TCP位置Y", "实测TCP位置Z",
    "实测TCP位置R", "实测TCP位置P", "实测TCP位置YAW",
    "实测相机位置X", "实测相机位置Y", "实测相机位置Z",
    "实测相机位置R", "实测相机位置P", "实测相机位置YAW",
    "最终像素误差X", "最终像素误差Y",
    "最终粗中心像素误差X", "最终粗中心像素误差Y",
    "静止像素误差均值X", "静止像素误差均值Y",
    "静止像素误差标准差X", "静止像素误差标准差Y",
    "静止粗中心均值X", "静止粗中心均值Y",
    "静止粗中心标准差X", "静止粗中心标准差Y",
    "静止中心黑白分离度均值", "静止中心黑白分离度最小值",
    "静止采样有效帧数", "静止采样请求帧数", "静止采样完整",
    "零误差等效TCP位置X", "零误差等效TCP位置Y",
    "视觉伺服轮数",
    "高位原图", "高位调试图", "低位首帧原图", "低位首帧调试图",
    "低位失败帧原图", "低位失败帧调试图", "低位伺服录像",
    "低位原图", "低位调试图", "消息",
]

SUCCESS_EVENT = "伺服成功"
FAILURE_EVENT = "伺服失败"
_SAMPLE_FILE_PATTERN = re.compile(r"^样本(\d+)_")


def _finite_text(value):
    """把数值转为 CSV 文本；非有限或缺失时留空。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number:.6f}" if np.isfinite(number) else ""


def _corners_json(corners):
    if corners is None:
        return ""
    return json.dumps([[round(float(value), 3) for value in point] for point in corners])


def detect_target_center(image, detector, ref_uv=None):
    """统一入口：ArUco 粗定位后只返回通过校验的中央棋盘角点。"""
    return detect_aruco_center(
        image,
        detector,
        ARUCO_MARKER_ID,
        ref_uv=ref_uv,
        refine_config=CENTER_REFINE_CONFIG,
    )


def _now_text():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class FreshImageReader:
    """按“快照必须新于请求时间戳”惯例读取最新相机帧。"""

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
        """返回 (图像, None) 或 (None, 失败原因)。"""
        request_stamp = rospy.Time.now()
        with self._condition:
            deadline = time.monotonic() + float(timeout_sec)
            while True:
                latest = self._latest.get("image")
                stamp = self._latest.get("stamp")
                if latest is not None and stamp is not None and stamp > request_stamp:
                    height, width = latest.shape[:2]
                    if self.expected_size is None:
                        self.expected_size = (int(width), int(height))
                    elif not check_image_size_constant(width, height, self.expected_size):
                        return None, f"图像尺寸变化: 期望 {self.expected_size}，实际 {(width, height)}"
                    return latest.copy(), None
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None, "图像获取超时"
                self._condition.wait(remaining)


class RosServices:
    """只依赖本实验需要的四个服务的轻量客户端。"""

    def __init__(self):
        rospy.wait_for_service("/control/move_arm", timeout=10.0)
        rospy.wait_for_service("/control/get_actual_pose", timeout=10.0)
        rospy.wait_for_service("/control/set_suction", timeout=10.0)
        rospy.wait_for_service("/camera/stable_world_points", timeout=10.0)
        self._move_arm = rospy.ServiceProxy("/control/move_arm", MoveArm)
        self._get_actual_pose = rospy.ServiceProxy("/control/get_actual_pose", GetActualPose)
        self._set_suction = rospy.ServiceProxy("/control/set_suction", SetSuction)
        self._stable_world_points = rospy.ServiceProxy("/camera/stable_world_points", GetStableWorldPoints)

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
            raise RuntimeError(f"move_arm 失败: {response.message}")

    def get_actual_pose(self):
        response = self._get_actual_pose(GetActualPoseRequest())
        if not response.success:
            raise RuntimeError(f"get_actual_pose 失败: {response.message}")
        return list(response.tcp_pose), list(response.camera_pose)

    def suction_off(self):
        response = self._set_suction(SetSuctionRequest(state=SetSuctionRequest.OFF))
        if not response.success:
            raise RuntimeError(f"set_suction 失败: {response.message}")

    def stable_world_point(self, x, y):
        response = self._stable_world_points(
            [int(round(x))], [int(round(y))],
            DEPTH_FRAME_COUNT, DEPTH_MIN_VALID_FRAMES, DEPTH_CAPTURE_TIMEOUT_SEC,
        )
        if not response.success:
            raise RuntimeError(f"稳定深度服务失败: {response.message}")
        if not response.point_valid or not response.point_valid[0]:
            return {
                "valid": False,
                "reason": f"深度点无效（有效帧 {response.valid_frame_counts[0] if response.valid_frame_counts else 0}）",
            }
        world = list(response.world_xyz[:3])
        return {
            "valid": True,
            "world_xyz": world,
            "valid_frames": int(response.valid_frame_counts[0]),
            "depth_median_mm": float(response.depth_median_mm[0]),
            "depth_mad_mm": float(response.depth_mad_mm[0]),
        }


def git_info():
    """记录 Git 提交与工作区状态，失败时返回空。"""
    try:
        commit = subprocess.run(
            ["git", "-C", str(SRC_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5.0,
        )
        status = subprocess.run(
            ["git", "-C", str(SRC_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5.0,
        )
        return {
            "commit": commit.stdout.strip(),
            "未提交文件数": len([line for line in status.stdout.splitlines() if line.strip()]),
        }
    except Exception as exc:  # noqa: BLE001
        return {"commit": None, "错误": str(exc)}


def write_json_atomic(path, data):
    """在同目录写临时文件后原子替换，避免中断时截断元数据。"""
    target = Path(path)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False, indent=2)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(target)


def _read_existing_csv(csv_path):
    """严格读取已有 CSV，返回数据行与正整数样本号集合。"""
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != CSV_COLUMNS:
            raise RuntimeError(
                "续跑 CSV 表头与当前代码不一致，禁止追加\n"
                f"批次表头: {reader.fieldnames}\n当前表头: {CSV_COLUMNS}"
            )
        rows = list(reader)
    sample_numbers = []
    for row_index, row in enumerate(rows, start=2):
        try:
            sample_no = int(row.get("样本号", ""))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"CSV 第 {row_index} 行样本号无效") from exc
        if sample_no <= 0:
            raise RuntimeError(f"CSV 第 {row_index} 行样本号必须为正整数")
        sample_numbers.append(sample_no)
    if len(sample_numbers) != len(set(sample_numbers)):
        raise RuntimeError("续跑 CSV 中存在重复样本号，禁止追加")
    return rows, set(sample_numbers)


def sample_numbers_from_files(batch_dir):
    """扫描批次根目录中的样本文件编号。"""
    numbers = set()
    for path in Path(batch_dir).iterdir():
        if not path.is_file():
            continue
        match = _SAMPLE_FILE_PATTERN.match(path.name)
        if match:
            numbers.add(int(match.group(1)))
    return numbers


def _value_differences(stored, current, prefix):
    """递归列出元数据与当前配置差异，供拒绝续跑时打印。"""
    if isinstance(stored, dict) and isinstance(current, dict):
        differences = []
        for key in sorted(set(stored) | set(current)):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if key not in stored:
                differences.append(f"{child_prefix}: 批次中缺失，当前={current[key]!r}")
            elif key not in current:
                differences.append(f"{child_prefix}: 批次={stored[key]!r}，当前缺失")
            else:
                differences.extend(
                    _value_differences(stored[key], current[key], child_prefix)
                )
        return differences
    if stored != current:
        return [f"{prefix}: 批次={stored!r}，当前={current!r}"]
    return []


def validate_resume_batch(output_root, resume_output_dir, parameters, configuration):
    """只读校验续跑目录、CSV、元数据和实验条件。"""
    if resume_output_dir is None:
        raise RuntimeError("RESUME=True 时必须设置 RESUME_OUTPUT_DIR")
    root = Path(output_root).expanduser().resolve()
    try:
        batch_dir = Path(resume_output_dir).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"续跑目录不存在: {resume_output_dir}") from exc
    if not batch_dir.is_dir():
        raise RuntimeError(f"续跑路径不是目录: {batch_dir}")
    try:
        relative = batch_dir.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"续跑目录必须位于 OUTPUT_ROOT 内: {root}") from exc
    if not relative.parts:
        raise RuntimeError("RESUME_OUTPUT_DIR 不能直接指向 OUTPUT_ROOT")

    csv_path = batch_dir / "aruco实验数据.csv"
    metadata_path = batch_dir / "实验元数据.json"
    if not csv_path.is_file():
        raise RuntimeError(f"续跑目录缺少 CSV: {csv_path}")
    if not metadata_path.is_file():
        raise RuntimeError(f"续跑目录缺少元数据: {metadata_path}")
    rows, csv_sample_numbers = _read_existing_csv(csv_path)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"续跑元数据无法读取: {metadata_path}: {exc}") from exc
    differences = []
    differences.extend(_value_differences(metadata.get("参数"), parameters, "参数"))
    differences.extend(_value_differences(metadata.get("配置"), configuration, "配置"))
    if differences:
        details = "\n".join(f"- {difference}" for difference in differences)
        raise RuntimeError(f"续跑实验条件与原批次不一致，禁止混合数据:\n{details}")

    file_sample_numbers = sample_numbers_from_files(batch_dir)
    all_numbers = csv_sample_numbers | file_sample_numbers
    max_sample_no = max(all_numbers, default=0)
    return {
        "batch_dir": batch_dir,
        "rows": rows,
        "csv_sample_numbers": csv_sample_numbers,
        "file_sample_numbers": file_sample_numbers,
        "orphan_sample_numbers": file_sample_numbers - csv_sample_numbers,
        "max_sample_no": max_sample_no,
        "metadata": metadata,
    }


class Experiment:
    """单个批次的实验上下文。"""

    def __init__(self, batch_dir, resume=False, resume_state=None):
        self.batch_dir = Path(batch_dir)
        self.csv_path = self.batch_dir / "aruco实验数据.csv"
        self.metadata_path = self.batch_dir / "实验元数据.json"
        if resume:
            if resume_state is None:
                raise ValueError("续跑模式缺少已校验状态")
            rows = list(resume_state["rows"])
            self.recorded_sample_numbers = set(resume_state["csv_sample_numbers"])
            self.sample_counter = int(resume_state["max_sample_no"])
            self.csv_file = self.csv_path.open(
                "a", encoding="utf-8-sig", newline=""
            )
        else:
            self.batch_dir.mkdir(parents=True, exist_ok=False)
            rows = []
            self.recorded_sample_numbers = set()
            self.sample_counter = 0
            self.csv_file = self.csv_path.open(
                "x", encoding="utf-8-sig", newline=""
            )
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=CSV_COLUMNS)
        if not resume:
            self.csv_writer.writeheader()
            self.csv_file.flush()
            os.fsync(self.csv_file.fileno())
        self.success_count = sum(row.get("事件") == SUCCESS_EVENT for row in rows)
        self.failure_count = sum(row.get("事件") == FAILURE_EVENT for row in rows)
        self.record_count = len(rows)
        self.session_start_record_count = self.record_count
        self.session_start_success_count = self.success_count
        self.session_start_failure_count = self.failure_count
        self.resume = bool(resume)

    def image_paths(self, sample_no):
        prefix = f"样本{sample_no:03d}"
        paths = {
            "high_original": self.batch_dir / f"{prefix}_高位_原图.png",
            "high_overlay": self.batch_dir / f"{prefix}_高位_调试图.png",
            "low_first_original": self.batch_dir / f"{prefix}_低位首帧_原图.png",
            "low_first_overlay": self.batch_dir / f"{prefix}_低位首帧_调试图.png",
            "low_failed_original": self.batch_dir / f"{prefix}_低位失败帧_原图.png",
            "low_failed_overlay": self.batch_dir / f"{prefix}_低位失败帧_调试图.png",
            "low_servo_video": self.batch_dir / f"{prefix}_低位伺服录像.avi",
            "low_original": self.batch_dir / f"{prefix}_低位_原图.png",
            "low_overlay": self.batch_dir / f"{prefix}_低位_调试图.png",
        }
        collisions = [path for path in paths.values() if path.exists()]
        mp4_path = paths["low_servo_video"].with_suffix(".mp4")
        if mp4_path.exists():
            collisions.append(mp4_path)
        if collisions:
            names = ", ".join(path.name for path in collisions)
            raise RuntimeError(f"样本 {sample_no} 输出文件已存在，禁止覆盖: {names}")
        return paths

    def record(self, row):
        sample_no = int(row.get("样本号", 0))
        if sample_no <= 0:
            raise ValueError("记录样本号必须为正整数")
        if sample_no in self.recorded_sample_numbers:
            raise RuntimeError(f"样本 {sample_no} 已存在于 CSV，禁止重复追加")
        self.csv_writer.writerow(row)
        self.csv_file.flush()
        os.fsync(self.csv_file.fileno())
        self.recorded_sample_numbers.add(sample_no)
        self.record_count += 1
        if row.get("事件") == SUCCESS_EVENT:
            self.success_count += 1
        elif row.get("事件") == FAILURE_EVENT:
            self.failure_count += 1

    def statistics(self):
        file_numbers = sample_numbers_from_files(self.batch_dir)
        return {
            "CSV记录数": self.record_count,
            "最大样本号": self.sample_counter,
            "成功数": self.success_count,
            "失败数": self.failure_count,
            "残留样本号": sorted(file_numbers - self.recorded_sample_numbers),
        }

    def session_statistics(self):
        return {
            "新增记录数": self.record_count - self.session_start_record_count,
            "新增成功数": self.success_count - self.session_start_success_count,
            "新增失败数": self.failure_count - self.session_start_failure_count,
        }

    def close(self):
        if not self.csv_file.closed:
            self.csv_file.close()


def create_or_resume_experiment(
    output_root,
    resume,
    resume_output_dir,
    parameters,
    configuration,
    batch_name=None,
):
    """创建新批次或以严格校验后的追加模式打开旧批次。"""
    output_root = Path(output_root)
    if resume:
        state = validate_resume_batch(
            output_root, resume_output_dir, parameters, configuration
        )
        return (
            Experiment(state["batch_dir"], resume=True, resume_state=state),
            state["metadata"],
            state,
        )
    base_name = batch_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_name = base_name
    suffix = 2
    while (output_root / unique_name).exists():
        unique_name = f"{base_name}-{suffix}"
        suffix += 1
    experiment = Experiment(output_root / unique_name, resume=False)
    return experiment, None, None


def start_experiment_metadata(experiment, previous_metadata, parameters, configuration):
    """建立本次运行记录；发现上次未结束时标记为异常中断。"""
    now = _now_text()
    if previous_metadata is None:
        metadata = {
            "批次": experiment.batch_dir.name,
            "创建时间": now,
            "参数": parameters,
            "配置": configuration,
            "git": git_info(),
            "运行历史": [],
        }
    else:
        # 通过 JSON 往返复制，避免修改校验阶段持有的原对象。
        metadata = json.loads(json.dumps(previous_metadata, ensure_ascii=False))
        history = metadata.setdefault("运行历史", [])
        if not history:
            legacy_end = metadata.get("结束时间")
            legacy_stats = metadata.get("统计", {})
            history.append({
                "模式": "历史运行",
                "启动时间": metadata.get("创建时间"),
                "结束时间": legacy_end,
                "状态": "已完成" if legacy_end else "异常中断",
                "起始样本号": 1,
                "结束最大样本号": legacy_stats.get(
                    "最大样本号", legacy_stats.get("总样本数")
                ),
                "本次统计": legacy_stats,
            })
        last_run = history[-1]
        if not last_run.get("结束时间") and last_run.get("状态") == "运行中":
            last_run["状态"] = "异常中断"
            last_run["结束原因"] = "下次续跑启动时检测到上次未正常结束"

    run_record = {
        "模式": "续跑" if experiment.resume else "新建",
        "启动时间": now,
        "结束时间": None,
        "状态": "运行中",
        "起始样本号": experiment.sample_counter + 1,
        "启动时CSV记录数": experiment.record_count,
        "启动时最大样本号": experiment.sample_counter,
        "git": git_info(),
    }
    metadata.setdefault("运行历史", []).append(run_record)
    run_index = len(metadata["运行历史"]) - 1
    metadata["状态"] = "运行中"
    metadata["最后启动时间"] = now
    metadata["统计"] = experiment.statistics()
    write_json_atomic(experiment.metadata_path, metadata)
    return metadata, run_index


def finish_experiment_metadata(experiment, metadata, run_index, status):
    """关闭本次运行记录并写入累计与本次统计。"""
    end_time = _now_text()
    run_record = metadata["运行历史"][run_index]
    run_record["结束时间"] = end_time
    run_record["状态"] = status
    run_record["结束最大样本号"] = experiment.sample_counter
    run_record["本次统计"] = experiment.session_statistics()
    metadata["状态"] = status
    metadata["结束时间"] = end_time
    metadata["最后结束时间"] = end_time
    metadata["统计"] = experiment.statistics()
    write_json_atomic(experiment.metadata_path, metadata)


def main():
    rospy.init_node("aruco_diagnostic_experiment", anonymous=True)
    print("ArUco 高位像素—低位 TCP 独立诊断实验启动")

    execution_config = load_execution_config(str(EXECUTION_CONFIG_PATH))
    visual_config = load_visual_servo_config(str(VISUAL_SERVO_CONFIG_PATH))
    with open(PERCEPTION_CONFIG_PATH, "r", encoding="utf-8") as config_file:
        perception_config = yaml.safe_load(config_file) or {}
    high_localization = perception_config.get("high_tcp_localization", {})
    safe_x_range_mm = high_localization.get("safe_x_range_mm", [-1e9, 1e9])
    safe_y_range_mm = high_localization.get("safe_y_range_mm", [-1e9, 1e9])

    shooting_pose = list(execution_config.shooting_pose)
    minimum_tcp_z_mm = float(execution_config.minimum_tcp_z_mm)
    pixel_to_robot_matrix = visual_config["pixel_to_robot_matrix"]

    ok, reason = validate_motion_pose(
        shooting_pose, minimum_tcp_z_mm, safe_x_range_mm, safe_y_range_mm
    )
    if not ok:
        raise RuntimeError(f"高位拍摄位姿安全检查未通过: {reason}")
    if not np.isfinite(LOW_TCP_Z_MM) or LOW_TCP_Z_MM < minimum_tcp_z_mm:
        raise RuntimeError(
            f"低位 TCP Z={LOW_TCP_Z_MM}mm 低于安全下限 {minimum_tcp_z_mm}mm"
        )

    parameters = {
        "aruco字典": ARUCO_DICT_NAME,
        "aruco标记ID": ARUCO_MARKER_ID,
        "中央精定位窗口比例": CENTER_REFINE_WINDOW_RATIO,
        "中央精定位窗口像素范围": [
            CENTER_REFINE_MIN_WINDOW_PX, CENTER_REFINE_MAX_WINDOW_PX,
        ],
        "中央最大修正码格比例": CENTER_REFINE_MAX_SHIFT_CELL_RATIO,
        "中央最小黑白分离度": CENTER_REFINE_MIN_CONTRAST,
        "高位采样帧数": HIGH_SAMPLE_FRAMES,
        "低位TCPZ毫米": LOW_TCP_Z_MM,
        "静止采样帧数": STATIC_SAMPLE_FRAMES,
        "静止最小有效比例": STATIC_MIN_VALID_RATIO,
        "深度帧数": DEPTH_FRAME_COUNT,
        "深度最小有效帧数": DEPTH_MIN_VALID_FRAMES,
        "深度采集超时秒": DEPTH_CAPTURE_TIMEOUT_SEC,
        "深度MAD上限毫米": DEPTH_MAX_MAD_MM,
        "图像超时秒": IMAGE_TIMEOUT_SEC,
    }
    configuration = {
        "shooting_pose": shooting_pose,
        "视觉伺服矩阵": pixel_to_robot_matrix,
        "安全X范围毫米": safe_x_range_mm,
        "安全Y范围毫米": safe_y_range_mm,
        "最低TCPZ毫米": minimum_tcp_z_mm,
        "误差阈值像素": SERVO_ERROR_THRESHOLD_PX,
        "最大步长毫米": execution_config.max_step_mm,
        "最小步长毫米": execution_config.min_step_mm,
        "最大轮数": execution_config.max_iter,
        "连续稳定帧数": execution_config.success_stable_frames,
        "最大丢失帧数": execution_config.max_missed_frames,
        "配置路径": {
            "execution": str(EXECUTION_CONFIG_PATH),
            "visual_servo": str(VISUAL_SERVO_CONFIG_PATH),
            "perception": str(PERCEPTION_CONFIG_PATH),
        },
    }
    # 必须先完成续跑校验，再创建服务或发送任何机械臂命令。
    experiment, previous_metadata, resume_state = create_or_resume_experiment(
        OUTPUT_ROOT,
        RESUME,
        RESUME_OUTPUT_DIR,
        parameters,
        configuration,
    )
    try:
        metadata, run_index = start_experiment_metadata(
            experiment, previous_metadata, parameters, configuration
        )
    except Exception:
        experiment.close()
        raise
    orphan_numbers = (
        sorted(resume_state["orphan_sample_numbers"]) if resume_state else []
    )
    print(f"运行模式: {'续跑已有批次' if RESUME else '创建新批次'}")
    print(f"实验目录: {experiment.batch_dir}")
    print(
        f"已有CSV记录={experiment.record_count}，最大样本号={experiment.sample_counter}，"
        f"残留编号={orphan_numbers if orphan_numbers else '无'}，"
        f"下一个样本=样本{experiment.sample_counter + 1:03d}"
    )

    finish_status = "异常结束"
    try:
        services = RosServices()
        reader = FreshImageReader()
        detector = create_aruco_detector(ARUCO_DICT_NAME)
        localizer = DepthRoughLocalizer(
            shooting_pose,
            np.load(SRC_ROOT / "camera" / "config" / "T_wrist2camera.npy"),
        )

        services.suction_off()
        print("吸盘已强制关闭")
        services.move_to(
            shooting_pose, execution_config.arm_speed, wait_until_stable=True
        )
        print(f"已回到高位拍摄位姿 {shooting_pose}")

        while True:
            print("\n请移动 ArUco 板到新位置，回车开始采样；输入 q 结束实验。")
            line = input("> ").strip().lower()
            if line == "q":
                break
            if line:
                print("仅接受空回车（采样）或 q（退出）")
                continue
            experiment.sample_counter += 1
            sample_no = experiment.sample_counter
            continue_run = run_one_sample(
                sample_no, experiment, services, reader, detector, localizer,
                execution_config, visual_config,
                shooting_pose, minimum_tcp_z_mm, safe_x_range_mm, safe_y_range_mm,
            )
            if not continue_run:
                print("控制服务异常，停止实验")
                finish_status = "控制服务异常停止"
                break
        if finish_status == "异常结束":
            finish_status = "正常结束"
    except KeyboardInterrupt:
        finish_status = "用户中断"
        print("\n检测到 Ctrl+C，已停止发送运动命令，保留已有数据")
    finally:
        experiment.close()
        finish_experiment_metadata(
            experiment, metadata, run_index, finish_status
        )
        print(f"实验结束，数据目录: {experiment.batch_dir}")


def run_one_sample(
    sample_no,
    experiment,
    services,
    reader,
    detector,
    localizer,
    execution_config,
    visual_config,
    shooting_pose,
    minimum_tcp_z_mm,
    safe_x_range_mm,
    safe_y_range_mm,
):
    """执行单个样本；返回是否继续实验（控制服务健康）。"""
    row = {column: "" for column in CSV_COLUMNS}
    row["样本号"] = sample_no
    row["时间戳"] = _now_text()
    row["事件"] = FAILURE_EVENT
    paths = experiment.image_paths(sample_no)
    rounds_count = 0
    try:
        # ---- 高位采样 ----
        samples = []
        retries = 0
        while len(samples) < HIGH_SAMPLE_FRAMES:
            image, error = reader.fresh_image(IMAGE_TIMEOUT_SEC)
            if image is None:
                retries += 1
                if retries >= HIGH_DETECT_MAX_RETRIES:
                    raise RuntimeError(f"高位图像获取失败: {error}")
                continue
            result = detect_target_center(image, detector)
            if not result.found:
                retries += 1
                if retries >= HIGH_DETECT_MAX_RETRIES:
                    raise RuntimeError(f"高位未持续识别到标记: {result.message}")
                continue
            retries = 0
            samples.append((result, image))
        high_center = median_center([sample[0].center_uv for sample in samples])
        high_rough_center = median_center([
            sample[0].rough_center_uv for sample in samples
        ])
        high_refine_delta = median_center([
            sample[0].refine_delta_uv for sample in samples
        ])
        high_contrast = float(np.median([
            sample[0].center_contrast for sample in samples
        ]))
        high_rep = min(samples, key=lambda sample: (
            (sample[0].center_uv[0] - high_center[0]) ** 2
            + (sample[0].center_uv[1] - high_center[1]) ** 2
        ))
        high_result, high_image = high_rep
        high_corners = high_result.corners
        row["高位检测像素X"] = _finite_text(high_center[0])
        row["高位检测像素Y"] = _finite_text(high_center[1])
        row["高位粗中心像素X"] = _finite_text(high_rough_center[0])
        row["高位粗中心像素Y"] = _finite_text(high_rough_center[1])
        row["高位中央修正量X"] = _finite_text(high_refine_delta[0])
        row["高位中央修正量Y"] = _finite_text(high_refine_delta[1])
        row["高位中心黑白分离度"] = _finite_text(high_contrast)
        row["高位四角点"] = _corners_json(high_corners)
        cv2.imwrite(str(paths["high_original"]), high_image)
        width, height = high_image.shape[1], high_image.shape[0]
        ref_uv = image_center_uv(width, height)
        draw_aruco_debug(
            high_image,
            high_corners,
            high_result.center_uv,
            ref_uv,
            paths["high_overlay"],
            rough_center_uv=high_result.rough_center_uv,
            refine_delta_uv=high_result.refine_delta_uv,
            center_contrast=high_result.center_contrast,
        )
        row["高位原图"] = paths["high_original"].relative_to(experiment.batch_dir).as_posix()
        row["高位调试图"] = paths["high_overlay"].relative_to(experiment.batch_dir).as_posix()

        # ---- 深度粗定位 ----
        depth_result = services.stable_world_point(high_center[0], high_center[1])
        if not depth_result["valid"]:
            raise RuntimeError(depth_result["reason"])
        if depth_result["depth_mad_mm"] > DEPTH_MAX_MAD_MM:
            raise RuntimeError(
                f"深度MAD={depth_result['depth_mad_mm']:.3f}mm 超过上限 {DEPTH_MAX_MAD_MM}mm"
            )
        world = depth_result["world_xyz"]
        row["高位世界坐标X"] = _finite_text(world[0])
        row["高位世界坐标Y"] = _finite_text(world[1])
        row["高位世界坐标Z"] = _finite_text(world[2])
        row["深度有效帧数"] = depth_result["valid_frames"]
        row["深度中位数毫米"] = _finite_text(depth_result["depth_median_mm"])
        row["深度MAD毫米"] = _finite_text(depth_result["depth_mad_mm"])

        tcp_xy = localizer.tcp_xy_from_world(world)
        low_pose = [
            float(tcp_xy[0]), float(tcp_xy[1]), LOW_TCP_Z_MM,
            *shooting_pose[3:6],
        ]
        row["粗定位TCP位置X"] = _finite_text(low_pose[0])
        row["粗定位TCP位置Y"] = _finite_text(low_pose[1])

        # ---- 移动到低位 ----
        move_checked(services, low_pose, execution_config.arm_speed,
                     minimum_tcp_z_mm, safe_x_range_mm, safe_y_range_mm)
        print(f"[{sample_no}] 已运动到低位粗定位位姿 {[round(v, 2) for v in low_pose]}")

        # ---- 低位首帧（无论成败都保存，用于诊断低位相机看到了什么）----
        first_low_image, first_low_error = reader.fresh_image(IMAGE_TIMEOUT_SEC)
        if first_low_image is not None:
            all_ids = []
            all_sizes = []
            first_rejected = 0
            for dict_name, found in detect_all_markers(
                first_low_image, PROBE_DICT_NAMES
            ).items():
                if found["ids"]:
                    all_ids.extend(found["ids"])
                    all_sizes.extend(found["sizes_px"])
                first_rejected = max(first_rejected, int(found["rejected_count"]))
            row["低位首帧检测ID"] = json.dumps(all_ids) if all_ids else ""
            if all_sizes:
                row["低位首帧最大标记像素尺寸"] = _finite_text(max(all_sizes))
            row["低位首帧rejected候选数"] = first_rejected
            cv2.imwrite(str(paths["low_first_original"]), first_low_image)
            first_width, first_height = first_low_image.shape[1], first_low_image.shape[0]
            first_ref_uv = image_center_uv(first_width, first_height)
            first_result = detect_target_center(first_low_image, detector)
            draw_aruco_debug(
                first_low_image,
                first_result.corners,
                first_result.center_uv if first_result.found else None,
                first_ref_uv,
                paths["low_first_overlay"],
                rough_center_uv=first_result.rough_center_uv,
                refine_delta_uv=first_result.refine_delta_uv,
                center_contrast=first_result.center_contrast,
            )
            row["低位首帧原图"] = paths["low_first_original"].relative_to(experiment.batch_dir).as_posix()
            row["低位首帧调试图"] = paths["low_first_overlay"].relative_to(experiment.batch_dir).as_posix()
            print(
                f"[{sample_no}] 低位首帧诊断: 检测ID={all_ids if all_ids else '无'}，"
                f"最大标记尺寸={max(all_sizes) if all_sizes else '无'}，"
                f"rejected候选={first_rejected}"
            )
        else:
            print(f"[{sample_no}] 低位首帧获取失败: {first_low_error}")

        # ---- 闭环对准（复用正式视觉伺服），全程录像 ----
        low_corners = None
        final_error = (float("nan"), float("nan"))
        static_rep = None
        last_missed_image = None
        last_missed_rejected = None
        last_missed_result = None
        servo_video = None
        servo_video_actual_path = None

        def write_servo_video_frame(image, round_no, result):
            if servo_video is None or image is None:
                return
            frame = draw_servo_overlay(
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
            servo_video.write(frame)

        try:
            if SERVO_VIDEO_ENABLED:
                servo_video, servo_video_actual_path = open_video_writer(
                    paths["low_servo_video"], SERVO_VIDEO_FPS, reader.expected_size
                )
                if servo_video is None:
                    print(f"[{sample_no}] 警告: 低位伺服录像无法打开，继续实验")

            def get_offset():
                nonlocal rounds_count, low_corners, last_missed_image
                nonlocal last_missed_rejected, last_missed_result
                rounds_count += 1
                image, error = reader.fresh_image(IMAGE_TIMEOUT_SEC)
                if image is None:
                    return OffsetResult(message=f"低位图像获取失败: {error}")
                result = detect_target_center(image, detector)
                write_servo_video_frame(image, rounds_count, result)
                if result.found:
                    low_corners = result.corners
                else:
                    last_missed_image = image
                    last_missed_rejected = result.rejected_corners
                    last_missed_result = result
                return result

            def move_pose_func(pose, speed, wait_sec=0.0, wait_until_stable=True):
                move_checked(
                    services, pose, speed, minimum_tcp_z_mm, safe_x_range_mm, safe_y_range_mm,
                    wait_until_stable=wait_until_stable,
                )

            success, final_pose, last_response, message = run_offset_visual_servo_alignment(
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
                log_label=f"样本{sample_no}低位",
            )
            row["视觉伺服轮数"] = rounds_count
            row["最终命令TCP位置X"] = _finite_text(final_pose[0])
            row["最终命令TCP位置Y"] = _finite_text(final_pose[1])
            row["最终命令TCP位置Z"] = _finite_text(final_pose[2])
            row["低位四角点"] = _corners_json(low_corners)
            if last_response is not None:
                final_error = (float(last_response.dx_px), float(last_response.dy_px))
                row["最终像素误差X"] = _finite_text(final_error[0])
                row["最终像素误差Y"] = _finite_text(final_error[1])
                if np.all(np.isfinite(last_response.rough_center_uv)):
                    rough_ref = image_center_uv(*reader.expected_size)
                    row["最终粗中心像素误差X"] = _finite_text(
                        last_response.rough_center_uv[0] - rough_ref[0]
                    )
                    row["最终粗中心像素误差Y"] = _finite_text(
                        last_response.rough_center_uv[1] - rough_ref[1]
                    )
            if not success:
                if last_missed_image is not None:
                    cv2.imwrite(str(paths["low_failed_original"]), last_missed_image)
                    failed_overlay = draw_rejected_candidates(
                        last_missed_image.copy(), last_missed_rejected or []
                    )
                    failed_height, failed_width = failed_overlay.shape[:2]
                    failed_ref_uv = image_center_uv(failed_width, failed_height)
                    failed_result = last_missed_result or OffsetResult()
                    draw_aruco_debug(
                        failed_overlay,
                        failed_result.corners,
                        failed_result.center_uv if failed_result.found else None,
                        failed_ref_uv,
                        paths["low_failed_overlay"],
                        rough_center_uv=failed_result.rough_center_uv,
                        refine_delta_uv=failed_result.refine_delta_uv,
                        center_contrast=failed_result.center_contrast,
                    )
                    row["低位失败帧原图"] = paths["low_failed_original"].relative_to(experiment.batch_dir).as_posix()
                    row["低位失败帧调试图"] = paths["low_failed_overlay"].relative_to(experiment.batch_dir).as_posix()
                    print(
                        f"[{sample_no}] 低位失败帧已保存，rejected候选={len(last_missed_rejected or [])}"
                    )
                raise RuntimeError(message)

            # ---- 对准成功后静止采样（不再移动，同步写入录像）----
            static_centers = []
            static_rough_centers = []
            static_contrasts = []
            static_records = []
            for _ in range(STATIC_SAMPLE_FRAMES):
                image, error = reader.fresh_image(IMAGE_TIMEOUT_SEC)
                if image is None:
                    static_centers.append((float("nan"), float("nan")))
                    static_rough_centers.append((float("nan"), float("nan")))
                    continue
                result = detect_target_center(image, detector)
                write_servo_video_frame(image, "静止", result)
                if result.found:
                    static_centers.append(result.center_uv)
                    static_contrasts.append(result.center_contrast)
                else:
                    static_centers.append((float("nan"), float("nan")))
                if np.all(np.isfinite(result.rough_center_uv)):
                    static_rough_centers.append(result.rough_center_uv)
                else:
                    static_rough_centers.append((float("nan"), float("nan")))
                static_records.append((result, image))
            stats = static_center_stats(static_centers, STATIC_SAMPLE_FRAMES, STATIC_MIN_VALID_RATIO)
            rough_stats = static_center_stats(
                static_rough_centers, STATIC_SAMPLE_FRAMES, STATIC_MIN_VALID_RATIO
            )
            row["静止采样有效帧数"] = stats["有效帧数"]
            row["静止采样请求帧数"] = stats["请求帧数"]
            row["静止采样完整"] = "是" if stats["完整"] else "否"
            row["静止像素误差均值X"] = _finite_text(stats["均值"][0])
            row["静止像素误差均值Y"] = _finite_text(stats["均值"][1])
            row["静止像素误差标准差X"] = _finite_text(stats["标准差"][0])
            row["静止像素误差标准差Y"] = _finite_text(stats["标准差"][1])
            row["静止粗中心均值X"] = _finite_text(rough_stats["均值"][0])
            row["静止粗中心均值Y"] = _finite_text(rough_stats["均值"][1])
            row["静止粗中心标准差X"] = _finite_text(rough_stats["标准差"][0])
            row["静止粗中心标准差Y"] = _finite_text(rough_stats["标准差"][1])
            if static_contrasts:
                row["静止中心黑白分离度均值"] = _finite_text(
                    np.mean(static_contrasts)
                )
                row["静止中心黑白分离度最小值"] = _finite_text(
                    np.min(static_contrasts)
                )

            # ---- 实测位姿与零误差等效 ----
            tcp_pose, camera_pose = services.get_actual_pose()
            for index, name in enumerate(["X", "Y", "Z", "R", "P", "YAW"]):
                row[f"实测TCP位置{name}"] = _finite_text(tcp_pose[index])
                row[f"实测相机位置{name}"] = _finite_text(camera_pose[index])
            static_mean_error = [
                stats["均值"][0] - (reader.expected_size[0] / 2.0),
                stats["均值"][1] - (reader.expected_size[1] / 2.0),
            ]
            equivalent = zero_error_tcp_xy(
                tcp_pose[:2], static_mean_error,
                visual_config["pixel_to_robot_matrix"], stats["完整"],
            )
            if equivalent is not None:
                row["零误差等效TCP位置X"] = _finite_text(equivalent[0])
                row["零误差等效TCP位置Y"] = _finite_text(equivalent[1])

            # ---- 保存低位代表帧与调试图 ----
            if stats["有效帧数"] >= 1:
                static_rep = min(
                    (
                        record for record in static_records if record[0].found
                    ),
                    key=lambda record: (
                        (record[0].center_uv[0] - stats["均值"][0]) ** 2
                        + (record[0].center_uv[1] - stats["均值"][1]) ** 2
                    ),
                )
                static_result, static_image = static_rep
                low_width, low_height = static_image.shape[1], static_image.shape[0]
                low_ref_uv = image_center_uv(low_width, low_height)
                cv2.imwrite(str(paths["low_original"]), static_image)
                draw_aruco_debug(
                    static_image,
                    static_result.corners,
                    static_result.center_uv,
                    low_ref_uv,
                    paths["low_overlay"],
                    rough_center_uv=static_result.rough_center_uv,
                    refine_delta_uv=static_result.refine_delta_uv,
                    center_contrast=static_result.center_contrast,
                )
                row["低位原图"] = paths["low_original"].relative_to(experiment.batch_dir).as_posix()
                row["低位调试图"] = paths["low_overlay"].relative_to(experiment.batch_dir).as_posix()

        finally:
            if servo_video is not None:
                servo_video.release()
                row["低位伺服录像"] = Path(servo_video_actual_path).relative_to(experiment.batch_dir).as_posix()

        row["事件"] = SUCCESS_EVENT
    except Exception as exc:  # noqa: BLE001
        row["消息"] = str(exc)
        print(f"[{sample_no}] 样本失败: {exc}")

    experiment.record(row)
    print(f"[{sample_no}] 已记录样本，返回高位")
    try:
        services.move_to(shooting_pose, execution_config.arm_speed, wait_until_stable=True)
    except Exception as exc:  # noqa: BLE001
        print(f"回高位失败，控制服务可能异常: {exc}")
        return False
    return True


def open_video_writer(video_path, fps, frame_size):
    """按正式代码惯例依次尝试 avi+MJPG、mp4+mp4v，返回 (writer, 实际路径)。"""
    candidates = [
        (video_path.with_suffix(".avi"), "MJPG"),
        (video_path.with_suffix(".mp4"), "mp4v"),
    ]
    for path, codec_name in candidates:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec_name),
            float(fps),
            (int(frame_size[0]), int(frame_size[1])),
        )
        if writer.isOpened():
            return writer, path
        writer.release()
    return None, None


def move_checked(
    services, pose, speed, minimum_tcp_z_mm, safe_x_range_mm, safe_y_range_mm,
    wait_until_stable=True,
):
    """发送运动前执行安全检查；每次伺服修正也经过此函数。

    wait_until_stable 默认 True：MoveArm 服务端会等机械臂停稳
    （controller.yaml arm_stability：速度≤3mm/s、位姿误差≤1mm/0.5°）后才返回，
    视觉伺服每轮修正前必须确认机械臂真正停稳。
    """
    ok, reason = validate_motion_pose(
        pose, minimum_tcp_z_mm, safe_x_range_mm, safe_y_range_mm
    )
    if not ok:
        raise RuntimeError(f"运动位姿安全检查未通过: {reason}")
    services.move_to(pose, speed, wait_until_stable=bool(wait_until_stable))


if __name__ == "__main__":
    main()
