"""方块和托盘像素到 TCP 标定数据记录器。"""

import csv
from datetime import datetime
import json
from pathlib import Path
import re


# 标定采集和主标定分析脚本共用同一目录，避免分析到旧的手动复制数据。
DEFAULT_SERVO_CSV_OUTPUT_DIR = Path("/home/zhl/桌面/标定数据")
EXPERIMENT_ARCHIVE_DIRNAME = "实验日志"
LOG_SCHEMA_VERSION = 3


LEGACY_CSV_FIELDNAMES = (
    "方块类别",
    "事件",
    "高位检测像素X",
    "高位检测像素Y",
    "深度采样像素X",
    "深度采样像素Y",
    "高位世界坐标X",
    "高位世界坐标Y",
    "高位世界坐标Z",
    "深度有效帧数",
    "深度中位数毫米",
    "深度MAD毫米",
    "粗定位TCP位置X",
    "粗定位TCP位置Y",
    "粗定位TCP位置Z",
    "标定目标TCP位置Z",
    "粗定位来源",
    "失败信息",
    "实测TCP位置X",
    "实测TCP位置Y",
    "实测TCP位置Z",
)


# 旧字段保持原顺序；新增诊断字段只能追加，避免破坏已有分析脚本和人工表格。
CSV_FIELDNAMES = LEGACY_CSV_FIELDNAMES + (
    "日志模式版本",
    "实验批次ID",
    "任务序号",
    "目标类型",
    "托盘行",
    "托盘列",
    "高位图像中心X",
    "高位图像中心Y",
    "高位检测角度deg",
    "目标旋转增量deg",
    "伺服开始时间",
    "伺服结束时间",
    "伺服总轮数",
    "执行修正次数",
    "目标丢失次数",
    "稳定帧数",
    "静止采样请求帧数",
    "静止采样有效帧数",
    "静止采样完整",
    "最终低位目标像素X",
    "最终低位目标像素Y",
    "最终低位图像中心X",
    "最终低位图像中心Y",
    "最终像素误差X",
    "最终像素误差Y",
    "最终低位检测角度deg",
    "最终低位匹配得分",
    "静止像素误差均值X",
    "静止像素误差均值Y",
    "静止像素误差标准差X",
    "静止像素误差标准差Y",
    "静止像素误差P95",
    "最终命令TCP位置X",
    "最终命令TCP位置Y",
    "最终命令TCP位置Z",
    "最终命令TCP姿态R",
    "最终命令TCP姿态P",
    "最终命令TCP姿态YAW",
    "实测TCP姿态R",
    "实测TCP姿态P",
    "实测TCP姿态YAW",
    "实测相机位置X",
    "实测相机位置Y",
    "实测相机位置Z",
    "实测相机姿态R",
    "实测相机姿态P",
    "实测相机姿态YAW",
    "实测减命令TCP位置X",
    "实测减命令TCP位置Y",
    "实测减命令TCP位置Z",
    "零误差等效TCP位置X",
    "零误差等效TCP位置Y",
)


ROUND_CSV_FIELDNAMES = (
    "日志模式版本",
    "实验批次ID",
    "任务序号",
    "目标类型",
    "方块类别",
    "托盘行",
    "托盘列",
    "高位检测像素X",
    "高位检测像素Y",
    "高位检测角度deg",
    "试次开始时间",
    "记录时间",
    "全试次记录序号",
    "事件",
    "伺服轮次",
    "连续稳定帧序号",
    "静止采样序号",
    "识别成功",
    "消息",
    "低位目标像素X",
    "低位目标像素Y",
    "低位图像中心X",
    "低位图像中心Y",
    "像素误差X",
    "像素误差Y",
    "最大像素误差",
    "低位检测角度deg",
    "低位匹配得分",
    "XY修正X毫米",
    "XY修正Y毫米",
    "末端命令X",
    "末端命令Y",
    "末端命令Z",
    "末端命令R",
    "末端命令P",
    "末端命令YAW",
    "本轮实测TCP位置X",
    "本轮实测TCP位置Y",
    "本轮实测TCP位置Z",
    "本轮实测TCP姿态R",
    "本轮实测TCP姿态P",
    "本轮实测TCP姿态YAW",
    "本轮实测相机位置X",
    "本轮实测相机位置Y",
    "本轮实测相机位置Z",
    "本轮实测相机姿态R",
    "本轮实测相机姿态P",
    "本轮实测相机姿态YAW",
    "图像服务耗时毫秒",
    "控制计算耗时毫秒",
    "机械臂到位耗时毫秒",
    "稳定等待耗时毫秒",
    "本轮总耗时毫秒",
)


ALLOWED_EVENTS = frozenset({"伺服成功", "伺服失败"})
ALLOWED_ROUND_EVENTS = frozenset(
    {
        "目标丢失",
        "执行修正",
        "稳定帧",
        "成功后静止帧",
        "成功后静止丢失",
        "成功后静止异常",
    }
)


def _now_iso():
    """生成带本地时区的毫秒时间，便于跨节点核对录像和 CSV。"""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _default_session_id():
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")


def _safe_session_id(value):
    session_id = str(value or "").strip() or _default_session_id()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", session_id):
        raise ValueError("实验批次 ID 只能包含字母、数字、点、下划线和连字符")
    return session_id


class ServoCsvLogger:
    """同时写入最新汇总和不可覆盖的实验批次归档。"""

    _TARGETS = {
        "block": "方块视觉伺服.csv",
        "board": "托盘视觉伺服.csv",
    }
    _ROUND_TARGETS = {
        "block": "方块视觉伺服逐轮.csv",
        "board": "托盘视觉伺服逐轮.csv",
    }

    def __init__(self, output_dir=DEFAULT_SERVO_CSV_OUTPUT_DIR, session_id=None):
        self.output_dir = Path(output_dir)
        self._explicit_session_id = bool(str(session_id or "").strip())
        self.session_id = _safe_session_id(session_id)
        self.archive_dir = (
            self.output_dir / EXPERIMENT_ARCHIVE_DIRNAME / self.session_id
        )
        self.paths = {}
        self._summary_files = {}
        self._summary_writers = {}
        self._round_files = {}
        self._round_writers = {}
        self._metadata = {}
        self._summary_counts = {
            "block": {"success": 0, "failure": 0},
            "board": {"success": 0, "failure": 0},
        }

    @property
    def is_open(self):
        """最新、归档及逐轮文件全部打开时才允许写入。"""
        return (
            set(self._summary_files) == set(self._TARGETS)
            and set(self._round_files) == set(self._TARGETS)
        )

    @property
    def metadata_path(self):
        return self.archive_dir / "实验元数据.json"

    def _open_csv(self, path, fieldnames):
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handle = path.open("w", encoding="utf-8-sig", newline="")
        writer = csv.DictWriter(
            file_handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        file_handle.flush()
        return file_handle, writer

    def open(self, metadata=None):
        """创建最新文件和批次目录；允许图像节点先写入调试图片。"""
        self.close()
        protected_names = {
            *self._TARGETS.values(),
            *self._ROUND_TARGETS.values(),
            "实验元数据.json",
        }
        collisions = [
            self.archive_dir / name
            for name in protected_names
            if (self.archive_dir / name).exists()
        ]
        if collisions and not self._explicit_session_id:
            self.session_id = _default_session_id()
            self.archive_dir = (
                self.output_dir / EXPERIMENT_ARCHIVE_DIRNAME / self.session_id
            )
            collisions = []
        if collisions:
            raise FileExistsError(f"实验批次日志已存在，拒绝覆盖: {collisions}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.paths = {}
        self._summary_counts = {
            "block": {"success": 0, "failure": 0},
            "board": {"success": 0, "failure": 0},
        }
        try:
            for target_key, filename in self._TARGETS.items():
                latest_path = self.output_dir / filename
                archive_path = self.archive_dir / filename
                latest_file, latest_writer = self._open_csv(
                    latest_path, CSV_FIELDNAMES
                )
                archive_file, archive_writer = self._open_csv(
                    archive_path, CSV_FIELDNAMES
                )
                self._summary_files[target_key] = [latest_file, archive_file]
                self._summary_writers[target_key] = [latest_writer, archive_writer]
                round_path = self.archive_dir / self._ROUND_TARGETS[target_key]
                round_file, round_writer = self._open_csv(
                    round_path, ROUND_CSV_FIELDNAMES
                )
                self._round_files[target_key] = round_file
                self._round_writers[target_key] = round_writer
                self.paths[target_key] = latest_path
                self.paths[f"{target_key}_archive"] = archive_path
                self.paths[f"{target_key}_round"] = round_path
            self.paths["archive_dir"] = self.archive_dir
            self._metadata = {
                "日志模式版本": LOG_SCHEMA_VERSION,
                "实验批次ID": self.session_id,
                "开始时间": _now_iso(),
                "结束时间": None,
                "输出文件": {key: str(value) for key, value in self.paths.items()},
            }
            if metadata:
                self._metadata.update(metadata)
            self._write_metadata_file()
        except Exception:
            self.close()
            raise
        return dict(self.paths)

    def _write_metadata_file(self):
        if not self.archive_dir.exists():
            return
        self.metadata_path.write_text(
            json.dumps(self._metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def update_metadata(self, values):
        """合并实验元数据并立即写盘。"""
        self._metadata.update(dict(values or {}))
        self._write_metadata_file()

    def write(self, target_key, row):
        """写入一条最终结果，并同步刷新最新文件和归档文件。"""
        if target_key not in self._TARGETS:
            raise ValueError(f"未知标定对象: {target_key}")
        writers = self._summary_writers.get(target_key)
        files = self._summary_files.get(target_key)
        if writers is None or files is None:
            return False

        event = row.get("事件")
        if event not in ALLOWED_EVENTS:
            raise ValueError(f"标定 CSV 事件只允许伺服成功或伺服失败，当前为: {event}")
        normalized = {field: row.get(field, "") for field in CSV_FIELDNAMES}
        normalized["日志模式版本"] = LOG_SCHEMA_VERSION
        normalized["实验批次ID"] = self.session_id
        for writer, file_handle in zip(writers, files):
            writer.writerow(normalized)
            file_handle.flush()
        count_key = "success" if event == "伺服成功" else "failure"
        self._summary_counts[target_key][count_key] += 1
        return True

    def write_round(self, target_key, row):
        """写入并立即刷新一条低位检测事件。"""
        if target_key not in self._TARGETS:
            raise ValueError(f"未知标定对象: {target_key}")
        writer = self._round_writers.get(target_key)
        file_handle = self._round_files.get(target_key)
        if writer is None or file_handle is None:
            return False
        event = row.get("事件")
        if event not in ALLOWED_ROUND_EVENTS:
            raise ValueError(f"未知逐轮日志事件: {event}")
        normalized = {field: row.get(field, "") for field in ROUND_CSV_FIELDNAMES}
        normalized["日志模式版本"] = LOG_SCHEMA_VERSION
        normalized["实验批次ID"] = self.session_id
        normalized["记录时间"] = normalized["记录时间"] or _now_iso()
        writer.writerow(normalized)
        file_handle.flush()
        return True

    def close(self):
        """关闭全部文件，并把结束时间及最终数量写入批次元数据。"""
        had_open_files = bool(self._summary_files or self._round_files)
        for files in self._summary_files.values():
            for file_handle in files:
                file_handle.close()
        for file_handle in self._round_files.values():
            file_handle.close()
        self._summary_files = {}
        self._summary_writers = {}
        self._round_files = {}
        self._round_writers = {}
        if had_open_files and self.archive_dir.exists():
            self._metadata["结束时间"] = _now_iso()
            self._metadata["汇总数量"] = self._summary_counts
            self._write_metadata_file()
