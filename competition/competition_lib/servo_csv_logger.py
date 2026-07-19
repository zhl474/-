"""方块和托盘视觉伺服的覆盖式 CSV 诊断日志。"""

import csv
from datetime import datetime
from pathlib import Path


DEFAULT_SERVO_CSV_OUTPUT_DIR = Path("/home/zhl/桌面/logs")

CSV_FIELDNAMES = (
    "运行编号",
    "记录时间",
    "对象类型",
    "任务序号",
    "方块类别",
    "托盘行",
    "托盘列",
    "事件",
    "伺服轮次",
    "识别成功",
    "消息",
    "高位检测像素X",
    "高位检测像素Y",
    "深度采样像素X",
    "深度采样像素Y",
    "高位图像中心X",
    "高位图像中心Y",
    "高位世界坐标X",
    "高位世界坐标Y",
    "高位世界坐标Z",
    "高位世界坐标有效",
    "粗定位来源",
    "粗定位末端命令X",
    "粗定位末端命令Y",
    "粗定位末端命令Z",
    "粗定位末端命令R",
    "粗定位末端命令P",
    "粗定位末端命令YAW",
    "低位目标像素X",
    "低位目标像素Y",
    "低位图像中心X",
    "低位图像中心Y",
    "像素误差X",
    "像素误差Y",
    "最大像素误差",
    "XY修正X毫米",
    "XY修正Y毫米",
    "末端命令X",
    "末端命令Y",
    "末端命令Z",
    "末端命令R",
    "末端命令P",
    "末端命令YAW",
    "实测TCP位置X",
    "实测TCP位置Y",
    "实测TCP位置Z",
    "实测TCP姿态R",
    "实测TCP姿态P",
    "实测TCP姿态YAW",
    "实测相机光心位置X",
    "实测相机光心位置Y",
    "实测相机光心位置Z",
    "实测相机光心姿态R",
    "实测相机光心姿态P",
    "实测相机光心姿态YAW",
    "实测位姿读取信息",
    "图像服务耗时毫秒",
    "控制计算耗时毫秒",
    "机械臂到位耗时毫秒",
    "稳定等待耗时毫秒",
    "本轮总耗时毫秒",
)


class ServoCsvLogger:
    """一次任务覆盖写入一对 CSV，逐行刷新以保留异常前的数据。"""

    _TARGETS = {
        "block": ("方块", "方块视觉伺服.csv"),
        "board": ("托盘", "托盘视觉伺服.csv"),
    }

    def __init__(self, output_dir=DEFAULT_SERVO_CSV_OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.run_id = ""
        self.paths = {}
        self._files = {}
        self._writers = {}

    def open(self):
        """创建目录并覆盖旧文件；两个文件共享本次任务的运行编号。"""
        self.close()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.paths = {}
        try:
            for target_key, (_target_name, filename) in self._TARGETS.items():
                path = self.output_dir / filename
                file_handle = path.open("w", encoding="utf-8-sig", newline="")
                writer = csv.DictWriter(file_handle, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
                writer.writeheader()
                file_handle.flush()
                self.paths[target_key] = path
                self._files[target_key] = file_handle
                self._writers[target_key] = writer
        except Exception:
            self.close()
            raise
        return dict(self.paths)

    def write(self, target_key, row):
        """写入一条事件；未开启记录时静默跳过，方便单元测试直接调用子流程。"""
        writer = self._writers.get(target_key)
        file_handle = self._files.get(target_key)
        if writer is None or file_handle is None:
            return False
        if target_key not in self._TARGETS:
            raise ValueError(f"未知视觉伺服对象: {target_key}")
        target_name, _filename = self._TARGETS[target_key]
        normalized = {field: "" for field in CSV_FIELDNAMES}
        normalized.update(row)
        normalized["运行编号"] = self.run_id
        normalized["记录时间"] = datetime.now().isoformat(timespec="milliseconds")
        normalized["对象类型"] = target_name
        writer.writerow(normalized)
        file_handle.flush()
        return True

    def close(self):
        """关闭当前一对 CSV，确保异常退出前已写入的数据落盘。"""
        for file_handle in self._files.values():
            file_handle.close()
        self._files = {}
        self._writers = {}
