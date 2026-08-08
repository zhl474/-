"""方块和托盘像素到 TCP 标定数据记录器。"""

import csv
from pathlib import Path


# 标定采集和主标定分析脚本共用同一目录，避免分析到旧的手动复制数据。
DEFAULT_SERVO_CSV_OUTPUT_DIR = Path("/home/zhl/桌面/标定数据")

CSV_FIELDNAMES = (
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

ALLOWED_EVENTS = frozenset({"伺服成功", "伺服失败"})


class ServoCsvLogger:
    """一次标定任务覆盖写入一对 CSV，每行立即刷新。"""

    _TARGETS = {
        "block": "方块视觉伺服.csv",
        "board": "托盘视觉伺服.csv",
    }

    def __init__(self, output_dir=DEFAULT_SERVO_CSV_OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.paths = {}
        self._files = {}
        self._writers = {}

    @property
    def is_open(self):
        """只有两份标定 CSV 都成功打开时才认为记录器可用。"""
        return set(self._files) == set(self._TARGETS)

    def open(self):
        """创建目录并覆盖旧文件。"""
        self.close()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths = {}
        try:
            for target_key, filename in self._TARGETS.items():
                path = self.output_dir / filename
                file_handle = path.open("w", encoding="utf-8-sig", newline="")
                writer = csv.DictWriter(
                    file_handle,
                    fieldnames=CSV_FIELDNAMES,
                    extrasaction="ignore",
                )
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
        """写入一条最终标定结果；未打开时静默跳过。"""
        if target_key not in self._TARGETS:
            raise ValueError(f"未知标定对象: {target_key}")
        writer = self._writers.get(target_key)
        file_handle = self._files.get(target_key)
        if writer is None or file_handle is None:
            return False

        event = row.get("事件")
        if event not in ALLOWED_EVENTS:
            raise ValueError(f"标定 CSV 事件只允许伺服成功或伺服失败，当前为: {event}")

        normalized = {field: row.get(field, "") for field in CSV_FIELDNAMES}
        writer.writerow(normalized)
        file_handle.flush()
        return True

    def close(self):
        """关闭当前一对 CSV，确保异常前的数据已落盘。"""
        for file_handle in self._files.values():
            file_handle.close()
        self._files = {}
        self._writers = {}
