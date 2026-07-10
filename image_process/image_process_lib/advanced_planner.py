"""进阶任务 IDBS 动态库的隔离包装。"""

from ctypes import CDLL, c_char_p
from typing import List, Sequence, Tuple

from image_process_lib.block_category import normalize_category_name


class AdvancedPlanner:
    def __init__(self, library_path: str):
        self.library_path = library_path

    def build_layout(self, cube_counts: Sequence[int], place_order: Sequence[int]) -> Tuple[List[dict], str]:
        if len(cube_counts) != 7 or len(place_order) != 7:
            raise ValueError("进阶任务需要 7 类方块数量和 7 项摆放顺序")
        library = CDLL(self.library_path)
        library.IDBS.restype = c_char_p
        raw_result = library.IDBS(*[int(value) for value in cube_counts], *[int(value) for value in place_order])
        if not raw_result:
            raise RuntimeError("IDBS 未返回规划结果")
        fields = raw_result.decode("gbk").split(",")
        if len(fields) < 5 or (len(fields) - 1) % 4 != 0:
            raise ValueError(f"IDBS 返回格式错误: {fields}")

        layout = []
        for index in range((len(fields) - 1) // 4):
            offset = index * 4
            layout.append({
                "index": index,
                "category": normalize_category_name(fields[offset]),
                "angle_deg": float(fields[offset + 1]),
                "col": float(fields[offset + 2]) + 0.5,
                "row": float(fields[offset + 3]) + 0.5,
            })
        return layout, fields[-1]
