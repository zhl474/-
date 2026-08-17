"""运行配置的白名单读取、校验、原子保存、历史和预设管理。"""

from copy import deepcopy
from datetime import datetime
import difflib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading

import numpy as np
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from .constants import (
    HAND_EYE_MATRIX_PATH,
    READ_ONLY_CONFIG_FILES,
    SRC_DIR,
    WRITABLE_CONFIG_FILES,
)


class ConfigError(ValueError):
    """配置内容或配置操作不符合约束。"""


class ConfigConflict(ConfigError):
    """文件已被其它程序修改，禁止静默覆盖。"""


class DangerousChangeRequired(ConfigError):
    """危险参数修改尚未得到页面二次确认。"""


def _revision(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return str(value)


def _finite_number(value, label, *, positive=False, nonnegative=False):
    if isinstance(value, bool):
        raise ConfigError(f"{label} 必须是有限数值")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{label} 必须是有限数值") from None
    if not math.isfinite(number):
        raise ConfigError(f"{label} 不能是 NaN 或无穷值")
    if positive and number <= 0:
        raise ConfigError(f"{label} 必须大于 0")
    if nonnegative and number < 0:
        raise ConfigError(f"{label} 必须大于等于 0")
    return number


def _strict_integer(value, label, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} 必须是整数")
    if positive and value <= 0:
        raise ConfigError(f"{label} 必须大于 0")
    if nonnegative and value < 0:
        raise ConfigError(f"{label} 必须大于等于 0")
    return int(value)


def _require_mapping(value, label):
    if not isinstance(value, dict):
        raise ConfigError(f"{label} 必须是 YAML 对象")
    return value


def _require_sequence(value, length, label):
    if not isinstance(value, list) or len(value) != int(length):
        raise ConfigError(f"{label} 必须固定包含 {length} 项")
    return value


def _nested(data, path):
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ConfigError(f"配置缺少字段 {path}")
        current = current[part]
    return current


FIELD_OVERRIDES = {
    "camera.rgb_camera.auto_exposure": {"label": "彩色自动曝光"},
    "camera.rgb_camera.exposure": {"label": "彩色曝光", "risk": "warning"},
    "camera.rgb_camera.gain": {"label": "彩色增益", "risk": "warning"},
    "camera.depth_camera.auto_exposure": {"label": "深度自动曝光"},
    "camera.depth_camera.exposure": {"label": "深度曝光(μs)", "risk": "warning"},
    "camera.depth_camera.gain": {"label": "深度增益", "risk": "warning"},
    "execution.shooting_pose": {
        "label": "高位拍摄位姿", "unit": "mm / °", "risk": "danger",
        "description": "顺序固定为 X、Y、Z、R、P、YAW，保存前必须确认现场安全。",
    },
    "execution.motion.arm_speed": {
        "label": "普通运动速度", "risk": "warning", "unit": "SDK 速度参数",
    },
    "execution.motion.pick_speed": {
        "label": "抓放下探速度", "risk": "danger", "unit": "SDK 速度参数",
    },
    "execution.motion.servo_speed": {
        "label": "视觉伺服速度", "risk": "danger", "unit": "SDK 速度参数",
    },
    "execution.motion.pick_approach_speed": {
        "label": "预抓取接近速度", "risk": "danger", "unit": "SDK 速度参数",
    },
    "execution.motion.minimum_tcp_z_mm": {
        "label": "TCP 最低安全高度", "risk": "danger", "unit": "mm",
    },
    "execution.motion.pick_surface_offset_mm": {
        "label": "抓取表面偏移", "risk": "danger", "unit": "mm",
    },
    "execution.motion.pick_approach_clearance_mm": {
        "label": "预抓取间隙", "risk": "danger", "unit": "mm",
    },
    "execution.motion.pick_retreat_blend_radius_mm": {
        "label": "抓后抬升圆滑半径", "risk": "danger", "unit": "mm",
    },
    "execution.motion.place_descent_offset_mm": {
        "label": "摆放下探深度", "risk": "danger", "unit": "mm",
    },
    "execution.motion.place_descent_blend_radius_mm": {
        "label": "摆放下探圆滑半径", "risk": "danger", "unit": "mm",
    },
    "execution.motion.place_lift_blend_radius_mm": {
        "label": "摆放后抬升圆滑半径", "risk": "danger", "unit": "mm",
    },
    "execution.motion.final_blow_hold_sec": {
        "label": "最后喷气保持时长", "risk": "warning", "unit": "s", "min": 0,
    },
    "execution.servo.sample_complete_ratio": {
        "label": "静止采样完整比例", "risk": "warning", "min": 0, "max": 1,
    },
    "execution.tool_motor.initial_angle_deg": {
        "label": "舵机初始角度", "unit": "°", "risk": "warning", "min": 0, "max": 360,
    },
    "execution.tool_motor.lower_margin_deg": {
        "label": "舵机安全下界", "unit": "°", "risk": "danger", "min": 0, "max": 360,
    },
    "execution.tool_motor.upper_margin_deg": {
        "label": "舵机安全上界", "unit": "°", "risk": "danger", "min": 0, "max": 360,
    },
    "visual_servo.pixel_to_robot_matrix": {
        "label": "像素到机器人映射矩阵", "unit": "mm/px", "risk": "danger",
    },
    "visual_servo.camera_to_sucker_offset_mm": {
        "label": "相机到吸盘偏移", "unit": "mm", "risk": "danger",
    },
    # --- 图像识别参数（perception）分组与字段大字号中文名 ---
    "perception.models": {
        "label": "推理模型", "risk": "danger",
        "description": "模型路径修改后必须重新启动感知节点。",
    },
    "perception.models.detection": {
        "label": "方块检测模型", "risk": "danger",
    },
    "perception.models.board": {
        "label": "托盘外框识别模型", "risk": "danger",
    },
    "perception.models.segmentation": {
        "label": "方块上表面分割模型", "risk": "danger",
    },
    "perception.task_sequence_optimizer": {
        "label": "固定盘面任务序列优化",
    },
    "perception.task_sequence_optimizer.mode": {
        "label": "固定盘面优化模式", "options": ["legacy", "shadow", "execute"],
        "risk": "warning",
    },
    "perception.task_sequence_optimizer.beam_width": {
        "label": "束搜索宽度",
    },
    "perception.task_sequence_optimizer.report_top_candidates": {
        "label": "记录候选方案数",
    },
    "perception.dynamic_board_selection": {
        "label": "动态盘面选择",
    },
    "perception.dynamic_board_selection.mode": {
        "label": "动态盘面模式", "options": ["disabled", "shadow", "execute"],
        "risk": "danger",
    },
    "perception.dynamic_board_selection.library_path": {
        "label": "动态盘面库路径", "risk": "danger",
    },
    "perception.dynamic_board_selection.coarse_top_k": {
        "label": "粗筛候选数",
    },
    "perception.dynamic_board_selection.final_candidate_k": {
        "label": "最终候选数",
    },
    "perception.dynamic_board_selection.keep_coarse_boundary_ties": {
        "label": "粗筛边界并列保留",
    },
    "perception.dynamic_board_selection.comparison_beam_width": {
        "label": "比较束搜索宽度",
    },
    "perception.dynamic_board_selection.comparison_returned_candidates": {
        "label": "比较返回候选数",
    },
    "perception.dynamic_board_selection.comparison_worker_count": {
        "label": "比较并行进程数",
    },
    "perception.dynamic_board_selection.confirmation_candidate_k": {
        "label": "确认候选数",
    },
    "perception.dynamic_board_selection.confirmation_beam_width": {
        "label": "确认束搜索宽度",
    },
    "perception.dynamic_board_selection.confirmation_returned_candidates": {
        "label": "确认返回候选数",
    },
    "perception.dynamic_board_selection.confirmation_worker_count": {
        "label": "确认并行进程数",
    },
    "perception.dynamic_board_selection.soft_time_budget_sec": {
        "label": "软时间预算",
    },
    "perception.dynamic_board_selection.failure_prompt_timeout_sec": {
        "label": "失败提示超时",
    },
    "perception.high_mask_manual_editor": {
        "label": "高位 Mask 人工编辑",
    },
    "perception.high_mask_manual_editor.enabled": {
        "label": "启用人工 Mask 编辑",
    },
    "perception.high_mask_manual_editor.script": {
        "label": "Mask 编辑器脚本", "risk": "danger",
    },
    "perception.high_mask_manual_editor.preview_device": {
        "label": "Mask 预览设备", "options": ["cpu", "cuda"],
    },
    "perception.yolo_manual_correction": {
        "label": "YOLO 检测框人工修正",
    },
    "perception.yolo_manual_correction.enabled": {
        "label": "启用检测框人工修正",
    },
    "perception.yolo_manual_correction.script": {
        "label": "检测框修正脚本", "risk": "danger",
    },
    "perception.calibration": {
        "label": "标定文件", "risk": "danger",
    },
    "perception.calibration.block_pixel_to_tcp": {
        "label": "方块像素-TCP 标定文件", "risk": "danger",
    },
    "perception.calibration.tray_pixel_to_tcp": {
        "label": "托盘像素-TCP 标定文件", "risk": "danger",
    },
    "perception.calibration.hand_eye_matrix": {
        "label": "手眼矩阵文件", "risk": "danger",
    },
    "perception.high_template_match": {
        "label": "高位模板匹配",
    },
    "perception.high_template_match.enabled": {
        "label": "启用高位快速匹配",
    },
    "perception.high_template_match.size_tolerance_px": {
        "label": "尺寸筛选容差",
    },
    "perception.high_template_match.relaxed_size_tolerance_px": {
        "label": "放宽尺寸容差",
    },
    "perception.high_template_match.min_candidate_angles": {
        "label": "最少候选角度数",
    },
    "perception.high_template_match.kernel_safety_margin_px": {
        "label": "核安全边距",
    },
    "perception.high_template_match.minimum_translation_margin_px": {
        "label": "最小平移余量",
    },
    "perception.high_template_match.legacy_fallback_enabled": {
        "label": "尺寸失败回退慢匹配",
    },
    "perception.block_recognition": {
        "label": "高位识别链选择",
    },
    "perception.block_recognition.mode": {
        "label": "识别模式",
        "options": ["v1", "v2", "shadow"],
    },
    "perception.block_recognition.fallback_to_v1": {
        "label": "V2 失败自动回退 V1",
    },
    "perception.block_recognition.v2": {
        "label": "V2 边缘模板匹配",
    },
    "perception.block_recognition.v2.angle_step_deg": {
        "label": "角度采样步进", "unit": "°",
    },
    "perception.block_recognition.v2.canny_low": {
        "label": "Canny 低阈值",
    },
    "perception.block_recognition.v2.canny_high": {
        "label": "Canny 高阈值",
    },
    "perception.block_recognition.v2.gaussian_ksize": {
        "label": "高斯核尺寸",
    },
    "perception.block_recognition.v2.distance_cap_px": {
        "label": "距离场截断", "unit": "px",
    },
    "perception.block_recognition.v2.crop_margin_px": {
        "label": "检测框外扩", "unit": "px",
    },
    "perception.block_recognition.v2.kernel_margin_px": {
        "label": "核画布边距", "unit": "px",
    },
    "perception.block_recognition.v2.search_margin_px": {
        "label": "平移搜索余量", "unit": "px",
    },
    "perception.calibration_depth": {
        "label": "深度标定",
    },
    "perception.calibration_depth.frame_count": {
        "label": "深度采集帧数",
    },
    "perception.calibration_depth.min_valid_frames": {
        "label": "最少有效帧数",
    },
    "perception.calibration_depth.capture_timeout_sec": {
        "label": "采集超时",
    },
    "perception.calibration_depth.block_max_mad_mm": {
        "label": "方块深度 MAD 上限",
    },
    "perception.calibration_depth.block_plane_max_rmse_mm": {
        "label": "方块平面 RMSE 上限",
    },
    "perception.calibration_depth.tray_tcp_below_block_observation_mm": {
        "label": "托盘相对方块观察有符号高度差", "unit": "mm", "risk": "danger",
    },
    "perception.high_tcp_localization": {
        "label": "高位 TCP 定位",
    },
    "perception.high_tcp_localization.safe_x_range_mm": {
        "label": "高位定位 X 安全范围", "unit": "mm", "risk": "danger",
    },
    "perception.high_tcp_localization.safe_y_range_mm": {
        "label": "高位定位 Y 安全范围", "unit": "mm", "risk": "danger",
    },
    "perception.high_tcp_localization.fixed_tcp_z": {
        "label": "固定 TCP Z",
    },
    "perception.high_tcp_localization.fixed_tcp_z.enabled": {
        "label": "固定 TCP Z 开关",
    },
    "perception.high_tcp_localization.fixed_tcp_z.block_observation_z_mm": {
        "label": "方块观察固定 Z", "unit": "mm", "risk": "danger",
    },
    "perception.high_tcp_localization.fixed_tcp_z.tray_z_mm": {
        "label": "托盘固定 Z", "unit": "mm", "risk": "danger",
    },
    "perception.pick_height": {
        "label": "抓取高度",
    },
    "perception.pick_height.block_observation_height_mm": {
        "label": "方块观察高度", "unit": "mm", "risk": "danger",
    },
    "perception.fallback": {
        "label": "回退兜底",
    },
    "perception.fallback.color_segmentation": {
        "label": "颜色分割回退",
    },
    "perception.block_servo": {
        "label": "方块视觉伺服",
    },
    "perception.block_servo.search_radius_px": {
        "label": "搜索半径",
    },
    "perception.block_servo.fallback_search_radius_px": {
        "label": "回退搜索半径",
    },
    "perception.block_servo.boundary_guard_px": {
        "label": "边界保护像素",
    },
    "perception.block_servo.kernel_safety_margin_px": {
        "label": "核安全边距",
    },
    "perception.block_servo.legacy_fallback_enabled": {
        "label": "异常回退旧慢速匹配",
    },
    "perception.block_servo.angle_window_deg": {
        "label": "角度窗口",
    },
    "perception.block_servo.angle_step_deg": {
        "label": "角度步长",
    },
    "perception.block_servo.white_s_max": {
        "label": "白色饱和度上限",
    },
    "perception.block_servo.white_v_min": {
        "label": "白色亮度下限",
    },
    "perception.block_servo.min_foreground_area": {
        "label": "最小前景面积",
    },
    "perception.board_servo": {
        "label": "托盘视觉伺服",
    },
    "perception.board_servo.roi_half_size": {
        "label": "ROI 半尺寸",
    },
    "perception.board_servo.blackhat_kernel_size": {
        "label": "黑帽核尺寸",
    },
    "perception.board_servo.min_dot_area": {
        "label": "最小圆点面积",
    },
    "perception.board_servo.max_dot_area": {
        "label": "最大圆点面积",
    },
    "perception.board_servo.min_dot_circularity": {
        "label": "最小圆点圆度",
    },
    "perception.board_servo.max_dot_aspect_ratio": {
        "label": "最大圆点长宽比",
    },
    "perception.visual_servo_debug": {
        "label": "视觉伺服调试",
    },
    "perception.visual_servo_debug.enabled": {
        "label": "启用视觉伺服调试",
    },
    "perception.visual_servo_debug.output_dir": {
        "label": "调试视频输出目录",
    },
    "perception.visual_servo_debug.video_fps": {
        "label": "调试视频帧率",
    },
    "controller.stop_motion": {
        "label": "软件停止确认参数", "risk": "danger", "unit": "s",
    },
    "template.template_sizes": {
        "label": "模板尺寸", "risk": "warning", "unit": "px",
    },
    "camera.depth_camera.depth_work_mode": {
        "label": "深度工作模式", "options": [0, 1, 2, 3, 4, 5],
        "description": "0=Default，1=Hand，2=High Accuracy，3=High Density，4=Medium Density，5=Factory Calib。",
    },
}


def _metadata_override(file_id, path):
    full = f"{file_id}.{path}" if path else file_id
    best = None
    for key, value in FIELD_OVERRIDES.items():
        if full == key or full.startswith(key + ".") or full.startswith(key + "["):
            if best is None or len(key) > len(best[0]):
                best = (key, value)
    return dict(best[1]) if best else {}


def _infer_unit(path):
    if path.endswith("_mm") or "_mm." in path:
        return "mm"
    if path.endswith("_deg") or "_deg." in path:
        return "°"
    if path.endswith("_px") or "_px." in path:
        return "px"
    if path.endswith("_sec") or path.endswith("_seconds"):
        return "s"
    if path.endswith("_fps"):
        return "FPS"
    return ""


class ConfigManager:
    """只允许访问固定 ID，并把全部持久化操作串行化。"""

    def __init__(self, store, state_dir=None):
        self.store = store
        self.state_dir = Path(state_dir or store.path.parent)
        self.backup_dir = self.state_dir / "calibration_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.yaml = YAML(typ="rt")
        self.yaml.preserve_quotes = True
        self.yaml.allow_duplicate_keys = False
        self.yaml.width = 4096
        self._ensure_initial_preset()

    def _entry(self, file_id):
        try:
            return WRITABLE_CONFIG_FILES[str(file_id)]
        except KeyError:
            raise ConfigError("未知配置文件 ID") from None

    @staticmethod
    def _read(path):
        return Path(path).read_text(encoding="utf-8")

    def _load_text(self, text):
        try:
            document = self.yaml.load(str(text))
        except Exception as exc:
            raise ConfigError(f"YAML 解析失败：{exc}") from None
        if not isinstance(document, CommentedMap):
            raise ConfigError("配置文件根节点必须是 YAML 对象")
        return document

    def _dump(self, document):
        output = io.StringIO()
        self.yaml.dump(document, output)
        return output.getvalue()

    @staticmethod
    def _extract_comments(text):
        """提取键前中文注释，作为前端字段说明的主要来源。"""
        result = {}
        stack = []
        pending = []
        key_pattern = re.compile(r"^(\s*)([^#\s][^:]*?):(?:\s*(.*))?$")
        for raw_line in str(text).splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("#"):
                pending.append(stripped.lstrip("#").strip())
                continue
            match = key_pattern.match(raw_line)
            if not match:
                if stripped:
                    pending = []
                continue
            indent = len(match.group(1).replace("\t", "  "))
            key = match.group(2).strip().strip("'\"")
            while stack and stack[-1][0] >= indent:
                stack.pop()
            path = ".".join([item[1] for item in stack] + [key])
            inline = ""
            value_part = match.group(3) or ""
            if "#" in value_part:
                inline = value_part.split("#", 1)[1].strip()
            description = " ".join(item for item in pending if item)
            if inline:
                description = (description + " " + inline).strip()
            if description:
                result[path] = description
            pending = []
            if not value_part.strip() or value_part.lstrip().startswith("#"):
                stack.append((indent, key))
        return result

    @staticmethod
    def _field_type(value):
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, str):
            lowered = value.lower()
            if "/" in value or lowered.endswith((".yaml", ".yml", ".npy", ".pt", ".engine", ".npz")):
                return "path"
            return "string"
        if value is None:
            return "null"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "string"

    def _build_schema(self, file_id, document, comments):
        schema = {}

        def visit(value, path):
            if path:
                metadata = {
                    "path": path,
                    "type": self._field_type(value),
                    "label": path.split(".")[-1],
                    "description": comments.get(path, ""),
                    "unit": _infer_unit(path),
                    "risk": "normal",
                    "expert": False,
                    "restart_scope": self._entry(file_id)["restart_scope"],
                }
                metadata.update(_metadata_override(file_id, path))
                if isinstance(value, list):
                    metadata["fixed_length"] = len(value)
                    if value and all(isinstance(item, list) for item in value):
                        metadata["matrix_shape"] = [len(value), len(value[0])]
                if metadata["risk"] == "normal" and isinstance(value, (int, float)) and not isinstance(value, bool):
                    # 没有可靠物理范围的数值只做类型与有限值校验。
                    metadata["expert"] = "min" not in metadata and "max" not in metadata
                schema[path] = metadata
            if isinstance(value, dict):
                for key, item in value.items():
                    child = f"{path}.{key}" if path else str(key)
                    visit(item, child)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    visit(item, f"{path}[{index}]")

        visit(document, "")
        if str(file_id) == "template":
            active = schema.get("template_sizes.active_profile")
            profiles = document.get("template_sizes", {}).get("profiles", {})
            if active is not None and isinstance(profiles, dict):
                active["options"] = [str(name) for name in profiles]
        return schema

    def get_config(self, file_id):
        entry = self._entry(file_id)
        text = self._read(entry["path"])
        document = self._load_text(text)
        comments = self._extract_comments(text)
        return {
            "file_id": str(file_id),
            "label": entry["label"],
            "description": entry["description"],
            "restart_scope": entry["restart_scope"],
            "revision": _revision(text),
            "data": _plain(document),
            "schema": self._build_schema(str(file_id), document, comments),
        }

    def list_configs(self):
        result = []
        for file_id, entry in WRITABLE_CONFIG_FILES.items():
            text = self._read(entry["path"])
            result.append({
                "file_id": file_id,
                "label": entry["label"],
                "description": entry["description"],
                "restart_scope": entry["restart_scope"],
                "revision": _revision(text),
            })
        return result

    def _merge(self, current, incoming, path=""):
        if isinstance(current, dict):
            if not isinstance(incoming, dict):
                raise ConfigError(f"{path or '根节点'} 类型不能改变")
            if set(current.keys()) != set(incoming.keys()):
                missing = [str(key) for key in current if key not in incoming]
                extra = [str(key) for key in incoming if key not in current]
                details = []
                if missing:
                    details.append("缺少：" + "、".join(missing))
                if extra:
                    details.append("新增：" + "、".join(extra))
                raise ConfigError(f"{path or '根节点'} 不允许新增或删除键（{'；'.join(details)}）")
            for key in current:
                child = f"{path}.{key}" if path else str(key)
                self._merge(current[key], incoming[key], child)
            return
        if isinstance(current, list):
            if not isinstance(incoming, list) or len(incoming) != len(current):
                raise ConfigError(f"{path} 固定数组长度不能改变")
            for index, item in enumerate(current):
                self._merge(item, incoming[index], f"{path}[{index}]")
            return
        if isinstance(current, bool):
            if not isinstance(incoming, bool):
                raise ConfigError(f"{path} 必须是布尔值")
            replacement = bool(incoming)
        elif isinstance(current, int):
            if isinstance(incoming, bool) or not isinstance(incoming, int):
                raise ConfigError(f"{path} 必须是整数")
            replacement = int(incoming)
        elif isinstance(current, float):
            if isinstance(incoming, bool) or not isinstance(incoming, (int, float)):
                raise ConfigError(f"{path} 必须是数值")
            replacement = float(incoming)
        elif isinstance(current, str):
            if not isinstance(incoming, str):
                raise ConfigError(f"{path} 必须是文本")
            replacement = str(incoming)
        elif current is None:
            if incoming is not None:
                raise ConfigError(f"{path} 当前为 null，V1 不允许改变类型")
            replacement = None
        else:
            raise ConfigError(f"{path} 使用了 V1 不支持的 YAML 类型")

        parent, key = self._resolve_parent(self._merge_root, path)
        parent[key] = replacement

    @staticmethod
    def _resolve_parent(root, path):
        tokens = re.findall(r"(?:^|\.)([^.\[]+)|\[(\d+)\]", path)
        parts = [int(index) if index else key for key, index in tokens]
        current = root
        for part in parts[:-1]:
            current = current[part]
        return current, parts[-1]

    def _merge_document(self, current, incoming):
        self._merge_root = current
        try:
            self._merge(current, incoming)
        finally:
            self._merge_root = None
        return current

    @staticmethod
    def _changed_paths(before, after, path=""):
        changed = []
        if isinstance(before, dict) and isinstance(after, dict):
            for key in before:
                child = f"{path}.{key}" if path else str(key)
                changed.extend(ConfigManager._changed_paths(before[key], after[key], child))
            return changed
        if isinstance(before, list) and isinstance(after, list):
            for index, item in enumerate(before):
                changed.extend(ConfigManager._changed_paths(item, after[index], f"{path}[{index}]"))
            return changed
        if before != after:
            changed.append(path)
        return changed

    @staticmethod
    def _is_dangerous(file_id, paths):
        for path in paths:
            metadata = _metadata_override(file_id, path)
            if metadata.get("risk") == "danger":
                return True
        return False

    def _validate_finite_tree(self, value, path=""):
        if isinstance(value, dict):
            for key, item in value.items():
                self._validate_finite_tree(item, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                self._validate_finite_tree(item, f"{path}[{index}]")
        elif isinstance(value, float) and not math.isfinite(value):
            raise ConfigError(f"{path} 不能是 NaN 或无穷值")

    def _validate_execution(self, data):
        pose = _require_sequence(_nested(data, "shooting_pose"), 6, "shooting_pose")
        for index, value in enumerate(pose):
            _finite_number(value, f"shooting_pose[{index}]")
        motion = _require_mapping(_nested(data, "motion"), "motion")
        for key in ("arm_speed", "pick_speed", "servo_speed", "pick_approach_speed"):
            _strict_integer(motion[key], f"motion.{key}", positive=True)
        _finite_number(motion["pick_surface_offset_mm"], "motion.pick_surface_offset_mm")
        _finite_number(motion["pick_approach_clearance_mm"], "motion.pick_approach_clearance_mm", positive=True)
        blend = _finite_number(motion["pick_retreat_blend_radius_mm"], "motion.pick_retreat_blend_radius_mm", nonnegative=True)
        if blend > 1000:
            raise ConfigError("motion.pick_retreat_blend_radius_mm 必须小于等于 1000")
        descent_offset = _finite_number(motion["place_descent_offset_mm"], "motion.place_descent_offset_mm", nonnegative=True)
        descent_blend = _finite_number(motion["place_descent_blend_radius_mm"], "motion.place_descent_blend_radius_mm", nonnegative=True)
        if descent_blend > 1000:
            raise ConfigError("motion.place_descent_blend_radius_mm 必须小于等于 1000")
        if descent_offset > 0 and descent_blend >= descent_offset:
            raise ConfigError("motion.place_descent_blend_radius_mm 必须小于 motion.place_descent_offset_mm")
        lift_blend = _finite_number(motion["place_lift_blend_radius_mm"], "motion.place_lift_blend_radius_mm", nonnegative=True)
        if lift_blend > 1000:
            raise ConfigError("motion.place_lift_blend_radius_mm 必须小于等于 1000")
        if descent_offset > 0 and lift_blend >= descent_offset:
            raise ConfigError("motion.place_lift_blend_radius_mm 必须小于 motion.place_descent_offset_mm")
        minimum_tcp_z_mm = _finite_number(
            motion["minimum_tcp_z_mm"], "motion.minimum_tcp_z_mm", positive=True
        )
        if float(pose[2]) < minimum_tcp_z_mm:
            raise ConfigError("shooting_pose 的 Z 低于 minimum_tcp_z_mm")
        servo = _require_mapping(_nested(data, "servo"), "servo")
        for key in ("enabled", "timing_debug"):
            if not isinstance(servo[key], bool):
                raise ConfigError(f"servo.{key} 必须是布尔值")
        for key in ("block_error_threshold_px", "tray_error_threshold_px", "min_step_mm", "max_step_mm", "settle_sec"):
            _finite_number(servo[key], f"servo.{key}", nonnegative=True)
        if float(servo["min_step_mm"]) > float(servo["max_step_mm"]):
            raise ConfigError("servo.min_step_mm 不能大于 servo.max_step_mm")
        for key in ("max_iter", "success_stable_frames", "max_missed_frames"):
            _strict_integer(servo[key], f"servo.{key}", positive=True)
        _strict_integer(servo["post_success_sample_frames"], "servo.post_success_sample_frames", nonnegative=True)
        if "sample_complete_ratio" in servo:
            ratio = _finite_number(servo["sample_complete_ratio"], "servo.sample_complete_ratio", positive=True)
            if ratio > 1:
                raise ConfigError("servo.sample_complete_ratio 必须在 (0, 1] 之间")
        if "final_blow_hold_sec" in motion:
            _finite_number(motion["final_blow_hold_sec"], "motion.final_blow_hold_sec", nonnegative=True)
        motor = _require_mapping(_nested(data, "tool_motor"), "tool_motor")
        initial = _finite_number(motor["initial_angle_deg"], "tool_motor.initial_angle_deg")
        lower = _finite_number(motor["lower_margin_deg"], "tool_motor.lower_margin_deg")
        upper = _finite_number(motor["upper_margin_deg"], "tool_motor.upper_margin_deg")
        _finite_number(motor["velocity_deg_per_sec"], "tool_motor.velocity_deg_per_sec", positive=True)
        if not 0 <= initial <= 360 or not 0 <= lower < upper <= 360:
            raise ConfigError("舵机初始角度或安全边界必须位于 0～360°，且下界小于上界")
    def _validate_visual_servo(self, data):
        matrix = _require_sequence(_nested(data, "pixel_to_robot_matrix"), 2, "pixel_to_robot_matrix")
        for row_index, row in enumerate(matrix):
            for col_index, value in enumerate(_require_sequence(row, 2, f"pixel_to_robot_matrix[{row_index}]")):
                _finite_number(value, f"pixel_to_robot_matrix[{row_index}][{col_index}]")
        offset = _require_sequence(_nested(data, "camera_to_sucker_offset_mm"), 2, "camera_to_sucker_offset_mm")
        for index, value in enumerate(offset):
            _finite_number(value, f"camera_to_sucker_offset_mm[{index}]")

    def _validate_perception(self, data):
        models = _require_mapping(_nested(data, "models"), "models")
        for name in ("detection", "board", "segmentation"):
            if not isinstance(models[name], str) or not models[name].strip():
                raise ConfigError(f"models.{name} 必须是非空路径")
            self._require_existing_source_file(models[name], f"models.{name}")
        enum_rules = (
            ("task_sequence_optimizer.mode", {"legacy", "shadow", "execute"}),
            ("dynamic_board_selection.mode", {"disabled", "shadow", "execute"}),
        )
        for path, options in enum_rules:
            if _nested(data, path) not in options:
                raise ConfigError(f"{path} 只能是：{'、'.join(sorted(options))}")
        optimizer = _require_mapping(_nested(data, "task_sequence_optimizer"), "task_sequence_optimizer")
        for key in ("beam_width", "report_top_candidates"):
            _strict_integer(optimizer[key], f"task_sequence_optimizer.{key}", positive=True)
        dynamic = _require_mapping(_nested(data, "dynamic_board_selection"), "dynamic_board_selection")
        for key in (
            "coarse_top_k", "final_candidate_k", "comparison_beam_width",
            "comparison_returned_candidates", "confirmation_beam_width",
            "confirmation_returned_candidates",
        ):
            _strict_integer(dynamic[key], f"dynamic_board_selection.{key}", positive=True)
        if not isinstance(dynamic["keep_coarse_boundary_ties"], bool):
            raise ConfigError("dynamic_board_selection.keep_coarse_boundary_ties 必须是布尔值")
        _finite_number(
            dynamic["soft_time_budget_sec"],
            "dynamic_board_selection.soft_time_budget_sec",
            nonnegative=True,
        )
        _finite_number(
            dynamic["failure_prompt_timeout_sec"],
            "dynamic_board_selection.failure_prompt_timeout_sec",
            positive=True,
        )
        self._require_existing_source_file(
            dynamic["library_path"], "dynamic_board_selection.library_path"
        )
        manual = _require_mapping(_nested(data, "high_mask_manual_editor"), "high_mask_manual_editor")
        if not isinstance(manual["enabled"], bool):
            raise ConfigError("high_mask_manual_editor.enabled 必须是布尔值")
        if not isinstance(manual["script"], str) or not manual["script"].strip():
            raise ConfigError("high_mask_manual_editor.script 必须是非空路径")
        if manual["preview_device"] not in ("cpu", "cuda"):
            raise ConfigError("high_mask_manual_editor.preview_device 只能是 cpu 或 cuda")
        if manual["enabled"]:
            self._require_existing_source_file(
                manual["script"], "high_mask_manual_editor.script"
            )
        calibration = _require_mapping(_nested(data, "calibration"), "calibration")
        for key in ("block_pixel_to_tcp", "tray_pixel_to_tcp", "hand_eye_matrix"):
            self._require_existing_source_file(calibration[key], f"calibration.{key}")
        for path in (
            "high_tcp_localization.safe_x_range_mm",
            "high_tcp_localization.safe_y_range_mm",
        ):
            values = _require_sequence(_nested(data, path), 2, path)
            low = _finite_number(values[0], path + "[0]")
            high = _finite_number(values[1], path + "[1]")
            if low >= high:
                raise ConfigError(f"{path} 下界必须小于上界")
        _finite_number(
            _nested(data, "pick_height.block_observation_height_mm"),
            "pick_height.block_observation_height_mm",
            positive=True,
        )
        depth = _require_mapping(_nested(data, "calibration_depth"), "calibration_depth")
        frame_count = _strict_integer(
            depth["frame_count"], "calibration_depth.frame_count", positive=True
        )
        minimum_frames = _strict_integer(
            depth["min_valid_frames"],
            "calibration_depth.min_valid_frames",
            positive=True,
        )
        if minimum_frames > frame_count:
            raise ConfigError("calibration_depth.min_valid_frames 不能大于 frame_count")
        for key in (
            "capture_timeout_sec", "block_max_mad_mm", "block_plane_max_rmse_mm",
        ):
            _finite_number(depth[key], f"calibration_depth.{key}", positive=True)
        _finite_number(
            depth["tray_tcp_below_block_observation_mm"],
            "calibration_depth.tray_tcp_below_block_observation_mm",
        )

        high_match = _require_mapping(_nested(data, "high_template_match"), "high_template_match")
        for key in ("enabled", "legacy_fallback_enabled"):
            if not isinstance(high_match[key], bool):
                raise ConfigError(f"high_template_match.{key} 必须是布尔值")
        for key in (
            "size_tolerance_px", "relaxed_size_tolerance_px",
            "kernel_safety_margin_px", "minimum_translation_margin_px",
        ):
            _strict_integer(high_match[key], f"high_template_match.{key}", nonnegative=True)
        _strict_integer(
            high_match["min_candidate_angles"],
            "high_template_match.min_candidate_angles",
            positive=True,
        )
        if high_match["relaxed_size_tolerance_px"] < high_match["size_tolerance_px"]:
            raise ConfigError(
                "high_template_match.relaxed_size_tolerance_px 必须大于等于 size_tolerance_px"
            )

        recognition = _require_mapping(
            _nested(data, "block_recognition"), "block_recognition"
        )
        if recognition["mode"] not in ("v1", "v2", "shadow"):
            raise ConfigError("block_recognition.mode 只能是 v1 / v2 / shadow")
        if not isinstance(recognition["fallback_to_v1"], bool):
            raise ConfigError("block_recognition.fallback_to_v1 必须是布尔值")
        edge_v2 = _require_mapping(
            recognition.get("v2", {}), "block_recognition.v2"
        )
        _finite_number(
            edge_v2["angle_step_deg"],
            "block_recognition.v2.angle_step_deg",
            positive=True,
        )
        _finite_number(
            edge_v2["distance_cap_px"],
            "block_recognition.v2.distance_cap_px",
            positive=True,
        )
        for key in ("canny_low", "canny_high"):
            _finite_number(edge_v2[key], f"block_recognition.v2.{key}", nonnegative=True)
        if edge_v2["canny_high"] <= edge_v2["canny_low"]:
            raise ConfigError("block_recognition.v2.canny_high 必须大于 canny_low")
        gaussian_ksize = _strict_integer(
            edge_v2["gaussian_ksize"],
            "block_recognition.v2.gaussian_ksize",
            positive=True,
        )
        if gaussian_ksize % 2 == 0:
            raise ConfigError("block_recognition.v2.gaussian_ksize 必须是奇数")
        for key in ("crop_margin_px", "kernel_margin_px", "search_margin_px"):
            _strict_integer(edge_v2[key], f"block_recognition.v2.{key}", nonnegative=True)

        block_servo = _require_mapping(_nested(data, "block_servo"), "block_servo")
        for key in (
            "search_radius_px", "fallback_search_radius_px", "boundary_guard_px",
            "kernel_safety_margin_px",
        ):
            _strict_integer(block_servo[key], f"block_servo.{key}", nonnegative=True)
        if block_servo["fallback_search_radius_px"] < block_servo["search_radius_px"]:
            raise ConfigError("block_servo.fallback_search_radius_px 必须大于等于 search_radius_px")
        if not isinstance(block_servo["legacy_fallback_enabled"], bool):
            raise ConfigError("block_servo.legacy_fallback_enabled 必须是布尔值")
        for key in ("angle_window_deg", "angle_step_deg"):
            _finite_number(block_servo[key], f"block_servo.{key}", positive=True)
        for key in ("white_s_max", "white_v_min"):
            value = _strict_integer(block_servo[key], f"block_servo.{key}", nonnegative=True)
            if value > 255:
                raise ConfigError(f"block_servo.{key} 必须位于 0～255")
        _strict_integer(
            block_servo["min_foreground_area"],
            "block_servo.min_foreground_area",
            positive=True,
        )

        board_servo = _require_mapping(_nested(data, "board_servo"), "board_servo")
        for key in ("roi_half_size", "blackhat_kernel_size", "min_dot_area", "max_dot_area"):
            _strict_integer(board_servo[key], f"board_servo.{key}", positive=True)
        if board_servo["min_dot_area"] > board_servo["max_dot_area"]:
            raise ConfigError("board_servo.min_dot_area 不能大于 max_dot_area")
        circularity = _finite_number(
            board_servo["min_dot_circularity"],
            "board_servo.min_dot_circularity",
            nonnegative=True,
        )
        if circularity > 1:
            raise ConfigError("board_servo.min_dot_circularity 必须位于 0～1")
        _finite_number(
            board_servo["max_dot_aspect_ratio"],
            "board_servo.max_dot_aspect_ratio",
            positive=True,
        )
        fallback = _require_mapping(_nested(data, "fallback"), "fallback")
        if not isinstance(fallback["color_segmentation"], bool):
            raise ConfigError("fallback.color_segmentation 必须是布尔值")
        debug = _require_mapping(_nested(data, "visual_servo_debug"), "visual_servo_debug")
        if not isinstance(debug["enabled"], bool):
            raise ConfigError("visual_servo_debug.enabled 必须是布尔值")
        if not isinstance(debug["output_dir"], str) or not debug["output_dir"].strip():
            raise ConfigError("visual_servo_debug.output_dir 必须是非空路径")
        _finite_number(debug["video_fps"], "visual_servo_debug.video_fps", positive=True)

    @staticmethod
    def _require_existing_source_file(value, label):
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{label} 必须是非空路径")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = SRC_DIR / path
        if not path.is_file():
            raise ConfigError(f"{label} 指向的文件不存在：{path}")

    def _validate_controller(self, data):
        if not isinstance(_nested(data, "move_arm_timing_debug"), bool):
            raise ConfigError("move_arm_timing_debug 必须是布尔值")
        for section in ("stop_motion", "arm_stability"):
            mapping = _require_mapping(_nested(data, section), section)
            for key, value in mapping.items():
                _finite_number(value, f"{section}.{key}", positive=True)

    def _validate_camera(self, data):
        for section in ("rgb_camera", "depth_camera"):
            values = _require_mapping(_nested(data, section), section)
            _strict_integer(values["fps"], f"{section}.fps", positive=True)
            for key, value in values.items():
                if key.startswith("auto_") and not isinstance(value, bool):
                    raise ConfigError(f"{section}.{key} 必须是布尔值")
                if isinstance(value, float):
                    _finite_number(value, f"{section}.{key}")
        depth_mode = _nested(data, "depth_camera.depth_work_mode")
        if isinstance(depth_mode, bool) or not isinstance(depth_mode, int) or depth_mode not in range(6):
            raise ConfigError("depth_camera.depth_work_mode 必须是已有枚举 0～5")

    def _validate_template(self, data):
        sizes = _nested(data, "template_sizes")
        if sizes["active_profile"] not in sizes["profiles"]:
            raise ConfigError("template_sizes.active_profile 必须指向已有 profile")
        for profile, values in sizes["profiles"].items():
            _strict_integer(values["block_px"], f"template_sizes.profiles.{profile}.block_px", positive=True)
            _strict_integer(values["connector_px"], f"template_sizes.profiles.{profile}.connector_px", positive=True)
            overrides = values.get("overrides")
            if overrides is None:
                continue
            _require_mapping(overrides, f"template_sizes.profiles.{profile}.overrides")
            for category, override in overrides.items():
                context = f"template_sizes.profiles.{profile}.overrides.{category}"
                _require_mapping(override, context)
                for key in ("x_runs", "y_runs"):
                    runs = override.get(key)
                    if runs is None:
                        continue
                    if not isinstance(runs, list) or not runs or len(runs) % 2 == 0:
                        raise ConfigError(f"{context}.{key} 必须是奇数长度的非空整数列表")
                    for index, value in enumerate(runs):
                        _strict_integer(value, f"{context}.{key}[{index}]", positive=True)
        categories = _nested(data, "color_segmentation.categories")
        color = _nested(data, "color_segmentation")
        for key in ("seed_search_half_size", "seed_patch_size", "seed_stride"):
            _strict_integer(color[key], f"color_segmentation.{key}", positive=True)
        _finite_number(
            color["local_dist_thresh"],
            "color_segmentation.local_dist_thresh",
            positive=True,
        )
        for category, values in categories.items():
            rgb = _require_sequence(values["rgb"], 3, f"color_segmentation.categories.{category}.rgb")
            for index, value in enumerate(rgb):
                channel = _strict_integer(value, f"{category}.rgb[{index}]", nonnegative=True)
                if channel > 255:
                    raise ConfigError(f"{category}.rgb[{index}] 必须位于 0～255")

    def validate_document(self, file_id, document):
        data = _plain(document)
        self._validate_finite_tree(data)
        validators = {
            "execution": self._validate_execution,
            "visual_servo": self._validate_visual_servo,
            "perception": self._validate_perception,
            "controller": self._validate_controller,
            "camera": self._validate_camera,
            "template": self._validate_template,
        }
        validators[str(file_id)](data)
        return data

    @staticmethod
    def _validate_cross_documents(documents):
        """执行已有运行代码明确要求的跨文件约束，不推测新的物理范围。"""
        execution = documents.get("execution")
        perception = documents.get("perception")
        if execution is None or perception is None:
            return
        dynamic_mode = _nested(perception, "dynamic_board_selection.mode")
        servo_enabled = _nested(execution, "servo.enabled")
        if dynamic_mode == "execute" and servo_enabled is not False:
            raise ConfigError(
                "跨文件校验失败：dynamic_board_selection.mode=execute "
                "要求 execution.yaml 中 servo.enabled=false"
            )

    def _validate_cross_for_candidate(self, file_id, candidate_data):
        documents = {}
        for current_id, entry in WRITABLE_CONFIG_FILES.items():
            if current_id == str(file_id):
                documents[current_id] = _plain(candidate_data)
            else:
                document = self._load_text(self._read(entry["path"]))
                documents[current_id] = self.validate_document(current_id, document)
        self._validate_cross_documents(documents)

    @staticmethod
    def _atomic_replace(path, text):
        path = Path(path)
        mode = path.stat().st_mode & 0o777
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_name, mode)
            os.replace(temporary_name, str(path))
            temporary_name = None
            # rename 已完成后文件内容已经原子提交。目录 fsync 只用于增强断电
            # 持久性，失败时不能再把一次成功提交误报成“未保存”，否则会留下
            # 已改文件却没有历史记录的不一致状态。
            try:
                directory_fd = os.open(str(path.parent), os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        except Exception:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
            raise

    @staticmethod
    def _git_revision():
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(SRC_DIR),
                check=False, capture_output=True, text=True, timeout=2.0,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""

    @staticmethod
    def _diff(file_id, before_text, after_text):
        return "".join(difflib.unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile=f"{file_id}:修改前", tofile=f"{file_id}:修改后",
        ))

    def _prepare_save_locked(self, file_id, data, expected_revision, confirm_dangerous):
        entry = self._entry(file_id)
        before_text = self._read(entry["path"])
        before_revision = _revision(before_text)
        if str(expected_revision) != before_revision:
            raise ConfigConflict("文件已被其它程序修改，请重新读取后再保存")
        current = self._load_text(before_text)
        before_data = _plain(current)
        self._merge_document(current, deepcopy(data))
        after_data = self.validate_document(str(file_id), current)
        self._validate_cross_for_candidate(file_id, after_data)
        changed_paths = self._changed_paths(before_data, after_data)
        dangerous = self._is_dangerous(str(file_id), changed_paths)
        if dangerous and not bool(confirm_dangerous):
            raise DangerousChangeRequired("包含危险位姿、高度、安全范围、映射矩阵或模型路径修改，需要二次确认")
        after_text = self._dump(current) if changed_paths else before_text
        # 重新解析最终文本，避免序列化后出现不能加载的文档。
        self.validate_document(str(file_id), self._load_text(after_text))
        return {
            "entry": entry,
            "before_text": before_text,
            "before_revision": before_revision,
            "after_text": after_text,
            "after_revision": _revision(after_text),
            "changed_paths": changed_paths,
            "dangerous": dangerous,
            "diff": self._diff(file_id, before_text, after_text),
        }

    def preflight_save(self, file_id, data, expected_revision, confirm_dangerous=False):
        """在创建异步操作前完成结构、revision 和危险确认校验。"""
        with self._lock:
            prepared = self._prepare_save_locked(
                file_id, data, expected_revision, confirm_dangerous
            )
            return {
                "changed": bool(prepared["changed_paths"]),
                "changed_paths": list(prepared["changed_paths"]),
                "dangerous": bool(prepared["dangerous"]),
                "restart_scope": prepared["entry"]["restart_scope"],
                "diff": prepared["diff"],
            }

    def save_config(self, file_id, data, expected_revision, confirm_dangerous=False, reason="网页保存"):
        with self._lock:
            prepared = self._prepare_save_locked(
                file_id, data, expected_revision, confirm_dangerous
            )
            entry = prepared["entry"]
            before_text = prepared["before_text"]
            before_revision = prepared["before_revision"]
            changed_paths = prepared["changed_paths"]
            if not changed_paths:
                return {
                    "changed": False, "revision": before_revision,
                    "changed_paths": [], "restart_scope": entry["restart_scope"],
                }
            after_text = prepared["after_text"]
            after_revision = prepared["after_revision"]
            diff_text = prepared["diff"]
            self._atomic_replace(entry["path"], after_text)
            try:
                history_id = self.store.add_history(
                    file_id, before_revision, after_revision, before_text,
                    after_text, diff_text, self._git_revision(), reason,
                )
            except Exception:
                # 历史写入失败时恢复原文，不能留下无审计的配置修改。
                self._atomic_replace(entry["path"], before_text)
                raise
            return {
                "changed": True,
                "revision": after_revision,
                "history_id": history_id,
                "diff": diff_text,
                "changed_paths": changed_paths,
                "dangerous": prepared["dangerous"],
                "restart_scope": entry["restart_scope"],
            }

    def restore_history(self, history_id, expected_revision, confirm_dangerous=False):
        history = self.store.get_history(history_id)
        if history is None:
            raise ConfigError("历史版本不存在")
        file_id = history["file_id"]
        document = self._load_text(history["before_text"])
        return self.save_config(
            file_id, _plain(document), expected_revision,
            confirm_dangerous=confirm_dangerous,
            reason=f"恢复历史 #{int(history_id)}",
        )

    def _snapshot(self):
        result = {}
        for file_id, entry in WRITABLE_CONFIG_FILES.items():
            text = self._read(entry["path"])
            result[file_id] = {"text": text, "revision": _revision(text)}
        return result

    def _ensure_initial_preset(self):
        if any(item["name"] == "当前稳定配置" for item in self.store.list_presets()):
            return
        try:
            self.store.save_preset(
                "当前稳定配置", self._snapshot(), self._git_revision(), protected=True
            )
        except ValueError:
            pass

    def save_preset(self, name, overwrite=False):
        clean_name = str(name or "").strip()
        if not clean_name or len(clean_name) > 60:
            raise ConfigError("预设名称长度必须为 1～60 个字符")
        with self._lock:
            preset_id = self.store.save_preset(
                clean_name,
                self._snapshot(),
                self._git_revision(),
                overwrite=bool(overwrite),
            )
        return self.store.get_preset(preset_id)

    def preset_diff(self, preset_id):
        preset = self.store.get_preset(preset_id)
        if preset is None:
            raise ConfigError("预设不存在")
        diffs = {}
        for file_id, entry in WRITABLE_CONFIG_FILES.items():
            target = preset["snapshot"].get(file_id, {}).get("text")
            if target is None:
                raise ConfigError(f"预设缺少 {file_id} 配置")
            current = self._read(entry["path"])
            diff = self._diff(file_id, current, target)
            if diff:
                diffs[file_id] = diff
        return diffs

    def restore_preset(self, preset_id, expected_revisions, confirm_dangerous=False):
        preset = self.store.get_preset(preset_id)
        if preset is None:
            raise ConfigError("预设不存在")
        expected_revisions = expected_revisions or {}
        with self._lock:
            before = {}
            targets = {}
            target_documents = {}
            changed = []
            dangerous = False
            for file_id, entry in WRITABLE_CONFIG_FILES.items():
                current_text = self._read(entry["path"])
                current_revision = _revision(current_text)
                if expected_revisions.get(file_id) != current_revision:
                    raise ConfigConflict(f"{entry['label']} 已被其它程序修改，请刷新后重试")
                target_text = preset["snapshot"].get(file_id, {}).get("text")
                if target_text is None:
                    raise ConfigError(f"预设缺少 {entry['label']}")
                target_document = self._load_text(target_text)
                target_documents[file_id] = self.validate_document(file_id, target_document)
                current_document = self._load_text(current_text)
                paths = self._changed_paths(_plain(current_document), _plain(target_document))
                if paths:
                    changed.append(file_id)
                    dangerous = dangerous or self._is_dangerous(file_id, paths)
                before[file_id] = current_text
                targets[file_id] = target_text
            self._validate_cross_documents(target_documents)
            if dangerous and not bool(confirm_dangerous):
                raise DangerousChangeRequired("预设包含危险参数变化，需要二次确认")
            replaced = []
            try:
                for file_id in changed:
                    self._atomic_replace(WRITABLE_CONFIG_FILES[file_id]["path"], targets[file_id])
                    replaced.append(file_id)
                git_revision = self._git_revision()
                history_ids = self.store.add_history_batch([
                    {
                        "file_id": file_id,
                        "before_revision": _revision(before[file_id]),
                        "after_revision": _revision(targets[file_id]),
                        "before_text": before[file_id],
                        "after_text": targets[file_id],
                        "diff_text": self._diff(file_id, before[file_id], targets[file_id]),
                        "git_revision": git_revision,
                        "reason": f"恢复预设：{preset['name']}",
                    }
                    for file_id in changed
                ])
            except Exception:
                for file_id in reversed(replaced):
                    self._atomic_replace(WRITABLE_CONFIG_FILES[file_id]["path"], before[file_id])
                raise
            scopes = sorted({WRITABLE_CONFIG_FILES[item]["restart_scope"] for item in changed})
            return {"changed_files": changed, "history_ids": history_ids, "restart_scopes": scopes}

    def inspect_read_only(self):
        result = []
        for file_id, entry in READ_ONLY_CONFIG_FILES.items():
            path = Path(entry["path"])
            item = {
                "file_id": file_id,
                "label": entry["label"],
                "kind": entry["kind"],
                "path": str(path),
                "exists": path.is_file(),
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds") if path.exists() else "",
            }
            if path.is_file():
                text = self._read(path)
                document = self._load_text(text)
                data = _plain(document)
                item["revision"] = _revision(text)
                if entry["kind"] == "layout":
                    item["data"] = data
                    item["target_count"] = len(data.get("targets", []))
                    item["board"] = {"columns": 10, "rows": 14}
                else:
                    item["summary"] = {
                        "schema_version": data.get("schema_version", ""),
                        "generation_id": data.get("generation_id", ""),
                        "subject": data.get("calibration_subject", ""),
                        "description": data.get("description", ""),
                        "metrics": data.get("metrics", {}),
                        "coverage": data.get("coverage", {}),
                    }
            result.append(item)
        hand_eye = {
            "file_id": "hand_eye",
            "label": "手眼矩阵",
            "kind": "matrix",
            "path": str(HAND_EYE_MATRIX_PATH),
            "exists": HAND_EYE_MATRIX_PATH.is_file(),
        }
        if HAND_EYE_MATRIX_PATH.is_file():
            matrix = np.load(str(HAND_EYE_MATRIX_PATH), allow_pickle=False)
            hand_eye.update({
                "modified_at": datetime.fromtimestamp(HAND_EYE_MATRIX_PATH.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
                "valid": bool(matrix.shape == (4, 4) and np.all(np.isfinite(matrix))),
                "shape": list(matrix.shape),
                "matrix": matrix.tolist() if matrix.shape == (4, 4) else [],
            })
        result.append(hand_eye)
        return result

    def deploy_calibration_pair(self, block_text, tray_text):
        """校验方块与托盘标定主体及 generation_id 后成对原子部署。"""
        with self._lock:
            block = _plain(self._load_text(block_text))
            tray = _plain(self._load_text(tray_text))
            for label, data, subject in (("方块", block, "block"), ("托盘", tray, "tray")):
                if data.get("schema_version") != 2:
                    raise ConfigError(f"{label}标定 schema_version 必须为 2")
                if data.get("calibration_subject") != subject:
                    raise ConfigError(f"{label}标定主体必须是 {subject}")
                if not data.get("generation_id"):
                    raise ConfigError(f"{label}标定缺少 generation_id")
                self._validate_finite_tree(data)
            if block["generation_id"] != tray["generation_id"]:
                raise ConfigError("方块与托盘标定 generation_id 不一致，禁止混合部署")
            paths = [
                Path(READ_ONLY_CONFIG_FILES["block_calibration"]["path"]),
                Path(READ_ONLY_CONFIG_FILES["tray_calibration"]["path"]),
            ]
            texts = [str(block_text), str(tray_text)]
            # 复用正式感知链的完整加载器，继续校验坐标语义、模型阶数、
            # 系数形状、Z 平面和非退化覆盖凸包，避免网页只检查顶层字段。
            from image_process_lib.pixel_to_tcp_calibration import (
                load_pixel_to_tcp_calibration,
            )

            validation_files = []
            try:
                for subject, text in zip(("block", "tray"), texts):
                    descriptor, temporary_name = tempfile.mkstemp(
                        prefix=f".网页标定校验-{subject}-",
                        suffix=".yaml",
                        dir=str(paths[0].parent),
                        text=True,
                    )
                    validation_files.append(temporary_name)
                    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                        handle.write(text)
                    load_pixel_to_tcp_calibration(
                        temporary_name,
                        expected_subject=subject,
                    )
            except ValueError as exc:
                raise ConfigError(f"像素标定完整 schema 校验失败：{exc}") from None
            finally:
                for temporary_name in validation_files:
                    try:
                        os.unlink(temporary_name)
                    except FileNotFoundError:
                        pass
            before = [self._read(path) for path in paths]
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            for path, original in zip(paths, before):
                (self.backup_dir / f"{stamp}_{path.name}").write_text(original, encoding="utf-8")
            replaced = []
            try:
                for path, text in zip(paths, texts):
                    self._atomic_replace(path, text)
                    replaced.append(path)
            except Exception:
                for path, original in zip(replaced, before):
                    self._atomic_replace(path, original)
                raise
            return {"generation_id": block["generation_id"], "backup_batch": stamp}

    def deploy_hand_eye(self, matrix):
        """整文件替换手眼矩阵，不提供网页逐项编辑入口。"""
        array = np.asarray(matrix, dtype=float)
        if array.shape != (4, 4) or not np.all(np.isfinite(array)):
            raise ConfigError("手眼矩阵必须是有限的 4×4 数组")
        with self._lock:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup = self.backup_dir / f"{stamp}_{HAND_EYE_MATRIX_PATH.name}"
            shutil.copy2(str(HAND_EYE_MATRIX_PATH), str(backup))
            mode = HAND_EYE_MATRIX_PATH.stat().st_mode & 0o777
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{HAND_EYE_MATRIX_PATH.name}.", suffix=".tmp",
                dir=str(HAND_EYE_MATRIX_PATH.parent),
            )
            os.close(descriptor)
            try:
                with open(temporary_name, "wb") as handle:
                    np.save(handle, array, allow_pickle=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary_name, mode)
                os.replace(temporary_name, str(HAND_EYE_MATRIX_PATH))
                temporary_name = None
                try:
                    directory_fd = os.open(str(HAND_EYE_MATRIX_PATH.parent), os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                except OSError:
                    pass
            except Exception:
                if temporary_name is not None:
                    try:
                        os.unlink(temporary_name)
                    except FileNotFoundError:
                        pass
                raise
            return {"backup": str(backup), "matrix": array.tolist()}
