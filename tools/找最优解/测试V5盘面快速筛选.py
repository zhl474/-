#!/home/zhl/fr3env/fr3env/bin/python
"""V5 盘面快速筛选器的固定合成数据离线演示与基准。"""

from pathlib import Path
import sys
import time

import numpy as np


# =========================
# 可直接修改的参数
# =========================
脚本目录 = Path(__file__).resolve().parent
盘面库文件 = 脚本目录 / "layouts_260_v5_final" / "v5_board_library_v1.npz"
粗筛候选数 = 300
最终候选数 = 20
保留粗筛边界同分 = True
单格尺寸毫米 = 30.0
托盘中心像素 = (640.0, 360.0)
盘面角度 = 0.0
筛选耗时目标秒 = 5.0
超时视为失败 = True


项目根目录 = 脚本目录.parents[1]
sys.path.insert(0, str(项目根目录 / "image_process"))

from image_process_lib.block_category import BLOCK_CATEGORY_NAMES  # noqa: E402
from image_process_lib.board_candidate_selector import (  # noqa: E402
    BoardCandidateSelector,
    BoardCandidateSelectorConfig,
    format_candidate_selection_report,
)
from image_process_lib.task_planner import ObservedBlock  # noqa: E402
from image_process_lib.v5_board_library import load_v5_board_library  # noqa: E402


def _make_synthetic_observations():
    """构造稳定的 7×5 合成现场实体，只用于测试与性能对比。"""
    nominal_positions = (
        (2.0, 3.0),
        (4.0, 6.0),
        (6.5, 4.0),
        (8.0, 10.5),
        (5.0, 8.0),
    )
    detected_angles = (0.0, 86.0, -92.0, 178.0, 4.0)
    blocks = []
    for category_index, category in enumerate(BLOCK_CATEGORY_NAMES):
        for local_index, (col, row) in enumerate(nominal_positions):
            adjusted_col = col + (category_index - 3) * 0.06
            adjusted_row = row + ((category_index + local_index) % 3 - 1) * 0.08
            pixel_u = 托盘中心像素[0] + (adjusted_col - 5.5) * 48.0
            pixel_v = 托盘中心像素[1] + (adjusted_row - 7.5) * 34.0
            x_mm = adjusted_col * 单格尺寸毫米
            y_mm = adjusted_row * 单格尺寸毫米
            blocks.append(
                ObservedBlock(
                    category=category,
                    observation_pose=(x_mm, y_mm, 200.0, -180.0, 0.0, 90.0),
                    detected_angle_deg=detected_angles[local_index],
                    source_id=category_index * 5 + local_index,
                    high_detected_pixel_xy=(pixel_u, pixel_v),
                )
            )
    return blocks


def _make_required_placement_xy(library, required_pids):
    """模拟调用方对候选 PID 执行“格点插值→TCP”，同行列只计算一次。"""
    placement_xy = np.full((library.placement_count, 2), np.nan, dtype=np.float64)
    xy_by_grid_point = {}
    for pid in required_pids:
        key = (float(library.placement_row[pid]), float(library.placement_col[pid]))
        xy = xy_by_grid_point.get(key)
        if xy is None:
            # 演示中用等间距坐标代替现场的标定插值链。
            xy = (key[1] * 单格尺寸毫米, key[0] * 单格尺寸毫米)
            xy_by_grid_point[key] = xy
        placement_xy[pid] = xy
    return placement_xy, len(xy_by_grid_point)


def main() -> None:
    load_start = time.perf_counter()
    library = load_v5_board_library(盘面库文件)
    load_elapsed = time.perf_counter() - load_start
    observations = _make_synthetic_observations()
    selector = BoardCandidateSelector(
        library,
        BoardCandidateSelectorConfig(
            coarse_top_k=粗筛候选数,
            final_candidate_k=最终候选数,
            keep_coarse_boundary_ties=保留粗筛边界同分,
        ),
    )

    coarse_result = selector.select_coarse(observations, 托盘中心像素)
    required_pids = selector.required_placement_ids(coarse_result)
    placement_xy, unique_point_count = _make_required_placement_xy(
        library, required_pids
    )
    final_result = selector.select_relaxed(
        coarse_result,
        observations,
        placement_xy,
        盘面角度,
    )

    print(f"盘面库加载耗时 = {load_elapsed:.4f} 秒")
    print(f"relaxed 候选共需 {len(required_pids)} 个 PID")
    print(f"去重后只需转换 {unique_point_count} 个目标中心点")
    print(format_candidate_selection_report(coarse_result, final_result))

    selection_elapsed = (
        final_result.coarse_elapsed_seconds + final_result.relaxed_elapsed_seconds
    )
    print(f"\n粗筛 + relaxed 总耗时 = {selection_elapsed:.4f} 秒")
    if selection_elapsed > 筛选耗时目标秒:
        message = (
            f"筛选耗时 {selection_elapsed:.4f} 秒超过目标 "
            f"{筛选耗时目标秒:.4f} 秒"
        )
        if 超时视为失败:
            raise RuntimeError(message)
        print(f"警告：{message}")


if __name__ == "__main__":
    main()
