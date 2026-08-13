#!/home/zhl/fr3env/fr3env/bin/python
"""将 V5 JSONL 转换为现场快速加载的未压缩 NPZ。

使用时直接修改下方参数，无需命令行传参。
"""

from pathlib import Path
import os
import shutil
import sys
import tempfile
import time


# =========================
# 可直接修改的参数
# =========================
脚本目录 = Path(__file__).resolve().parent
项目根目录 = 脚本目录.parents[1]
输入目录 = 脚本目录 / "layouts_260_v5_final"
输出文件 = 项目根目录 / "image_process" / "config" / "v5_board_library_v1.npz"
# 同步保留一份在原始盘面目录，便于离线回放且避免误用旧库。
盘面目录副本 = 输入目录 / "v5_board_library_v1.npz"


sys.path.insert(0, str(项目根目录 / "image_process"))

from image_process_lib.v5_board_library import convert_v5_board_library  # noqa: E402


def _原子复制文件(源文件: Path, 目标文件: Path) -> None:
    """先完整落盘临时文件，再原子替换目标文件。"""
    目标文件.parent.mkdir(parents=True, exist_ok=True)
    临时文件路径 = None
    try:
        with 源文件.open("rb") as 输入流, tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{目标文件.name}.",
            suffix=".tmp",
            dir=目标文件.parent,
            delete=False,
        ) as 输出流:
            临时文件路径 = Path(输出流.name)
            shutil.copyfileobj(输入流, 输出流)
            输出流.flush()
            os.fsync(输出流.fileno())
        os.chmod(临时文件路径, 0o644)
        os.replace(临时文件路径, 目标文件)
        临时文件路径 = None
    finally:
        if 临时文件路径 is not None:
            try:
                临时文件路径.unlink()
            except FileNotFoundError:
                pass


def main() -> None:
    start = time.perf_counter()
    summary = convert_v5_board_library(输入目录, 输出文件)
    if 盘面目录副本.resolve() != 输出文件.resolve():
        _原子复制文件(输出文件, 盘面目录副本)
    elapsed = time.perf_counter() - start
    print("V5 盘面库转换完成")
    print(f"输入 JSONL 数 = {summary.input_file_count}")
    print(f"盘面数 = {summary.board_count}")
    print(f"V5 signature 数 = {summary.signature_count}")
    print(f"placement 数 = {summary.placement_count}")
    print(f"输出文件 = {summary.output_path}")
    print(f"盘面目录副本 = {盘面目录副本}")
    print(f"文件大小 = {summary.output_bytes / 1024 / 1024:.3f} MiB")
    print(f"转换耗时 = {elapsed:.3f} 秒")


if __name__ == "__main__":
    main()
