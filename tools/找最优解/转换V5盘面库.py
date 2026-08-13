#!/home/zhl/fr3env/fr3env/bin/python
"""将 V5 JSONL 转换为现场快速加载的未压缩 NPZ。

使用时直接修改下方参数，无需命令行传参。
"""

from pathlib import Path
import sys
import time


# =========================
# 可直接修改的参数
# =========================
脚本目录 = Path(__file__).resolve().parent
输入目录 = 脚本目录 / "layouts_260_v5_final"
输出文件 = 输入目录 / "v5_board_library_v1.npz"


项目根目录 = 脚本目录.parents[1]
sys.path.insert(0, str(项目根目录 / "image_process"))

from image_process_lib.v5_board_library import convert_v5_board_library  # noqa: E402


def main() -> None:
    start = time.perf_counter()
    summary = convert_v5_board_library(输入目录, 输出文件)
    elapsed = time.perf_counter() - start
    print("V5 盘面库转换完成")
    print(f"输入 JSONL 数 = {summary.input_file_count}")
    print(f"盘面数 = {summary.board_count}")
    print(f"V5 signature 数 = {summary.signature_count}")
    print(f"placement 数 = {summary.placement_count}")
    print(f"输出文件 = {summary.output_path}")
    print(f"文件大小 = {summary.output_bytes / 1024 / 1024:.3f} MiB")
    print(f"转换耗时 = {elapsed:.3f} 秒")


if __name__ == "__main__":
    main()
