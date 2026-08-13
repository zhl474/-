"""V5 高速盘面库的转换、角度和格式校验。"""

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pytest

from image_process_lib.block_category import BLOCK_CATEGORY_NAMES
from image_process_lib.task_planner import load_task_layout, normalize_rotation_delta
from image_process_lib.v5_board_library import (
    V5_CLOCKWISE_ANGLE_OUTPUT_MAP,
    V5_REGION_MASK_ORDER,
    convert_v5_board_library,
    load_v5_board_library,
)


V5_CATEGORY_ORDER = (
    "L_yellow",
    "L_blue",
    "z_green",
    "z_blue",
    "T",
    "square",
    "line",
)
PACKAGE_DIRECTORY = Path(__file__).resolve().parents[1]
OFFICIAL_LIBRARY_PATH = (
    PACKAGE_DIRECTORY
    / "config"
    / "v5_board_library_v1.npz"
)


def _write_small_v5_source(directory: Path):
    placements = []
    internal_angles = (0, 90, 180, -90, 0)
    masks = (1, 2, 4, 8, 15)
    # 四种形态分别覆盖 L 长边朝横向/纵向及参考点的正负半格修正。
    l_cells_by_orientation = (
        [[1, 1], [2, 1], [3, 1], [3, 2]],
        [[1, 1], [1, 2], [1, 3], [2, 1]],
        [[1, 1], [1, 2], [2, 2], [3, 2]],
        [[1, 3], [2, 1], [2, 2], [2, 3]],
    )
    for category in V5_CATEGORY_ORDER:
        for local_index in range(5):
            pid = len(placements)
            if category in ("L_yellow", "L_blue"):
                cells = l_cells_by_orientation[local_index % 4]
            else:
                col_offset = local_index * 2
                cells = [
                    [1 + col_offset, 1],
                    [2 + col_offset, 1],
                    [1 + col_offset, 2],
                    [2 + col_offset, 2],
                ]
            cell_array = np.asarray(cells, dtype=float)
            placements.append(
                {
                    "id": pid,
                    "category": category,
                    "angle_deg_internal": internal_angles[local_index],
                    # 故意写入旧的错误输出，转换器不应信任该字段。
                    "angle_deg": internal_angles[local_index],
                    # 与正式 catalog 一样保存包围盒中心；L 的运行参考点由 cells 恢复。
                    "row": float(
                        (np.min(cell_array[:, 1]) + np.max(cell_array[:, 1])) / 2.0
                    ),
                    "col": float(
                        (np.min(cell_array[:, 0]) + np.max(cell_array[:, 0])) / 2.0
                    ),
                    "cells": cells,
                    "region_mask": masks[local_index],
                }
            )
    signature_order = [
        {"category": category, "region_mask": mask}
        for category in V5_CATEGORY_ORDER
        for mask in V5_REGION_MASK_ORDER
    ]
    catalog = {
        "version": 5,
        "spatial_signature_order": signature_order,
        "placements": placements,
    }
    (directory / "placement_catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False), encoding="utf-8"
    )

    ids_by_category = {
        category: [item["id"] for item in placements if item["category"] == category]
        for category in V5_CATEGORY_ORDER
    }

    def make_record(global_id, rank, missing_category):
        selected_ids = []
        for category in V5_CATEGORY_ORDER:
            category_ids = ids_by_category[category]
            selected_ids.extend(
                category_ids[:4] if category == missing_category else category_ids
            )
        # 故意打乱 PID，NPZ 内部仍应生成稳定顺序。
        selected_ids.reverse()
        counts = Counter(
            (placements[pid]["category"], placements[pid]["region_mask"])
            for pid in selected_ids
        )
        return {
            "version": 5,
            "global_id": global_id,
            "rank": rank,
            "layout_index": 0,
            "ids": selected_ids,
            "missing_category": missing_category,
            "spatial_signature_vector": [
                counts[(item["category"], item["region_mask"])]
                for item in signature_order
            ],
        }

    # 文件顺序与期望库内顺序相反，用于验证稳定排序。
    records = [
        make_record(global_id=12, rank=2, missing_category="square"),
        make_record(global_id=11, rank=1, missing_category="T"),
    ]
    layout_path = directory / "library_shard00of01.jsonl"
    layout_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return placements, layout_path


def _save_library_arrays(path: Path, library, **updates):
    arrays = {
        "format_version": np.asarray(library.format_version, dtype=np.uint16),
        "category_names": np.asarray(library.category_names, dtype=np.str_),
        "region_mask_order": library.region_mask_order,
        "board_global_id": library.board_global_id,
        "board_layout_index": library.board_layout_index,
        "board_rank": library.board_rank,
        "board_missing_category": library.board_missing_category,
        "board_target_pid": library.board_target_pid,
        "board_region_counts": library.board_region_counts,
        "placement_category": library.placement_category,
        "placement_row": library.placement_row,
        "placement_col": library.placement_col,
        "placement_yaw_clockwise_deg": library.placement_yaw_clockwise_deg,
        "placement_region_mask": library.placement_region_mask,
        "placement_cells": library.placement_cells,
    }
    arrays.update(updates)
    np.savez(path, **arrays)


def test_conversion_reorders_categories_derives_clockwise_angles_and_sorts(tmp_path):
    placements, _ = _write_small_v5_source(tmp_path)
    output_path = tmp_path / "v5_board_library_v1.npz"

    summary = convert_v5_board_library(tmp_path, output_path)
    library = load_v5_board_library(output_path)

    assert summary.board_count == 2
    assert summary.signature_count == 2
    assert summary.placement_count == 35
    assert library.category_names == BLOCK_CATEGORY_NAMES
    assert library.source_path == output_path.resolve()
    assert len(library.source_sha256) == 64
    assert library.board_global_id.tolist() == [11, 12]
    assert np.sum(library.board_target_pid >= 0, axis=(1, 2)).tolist() == [34, 34]
    for placement in placements:
        pid = placement["id"]
        expected = V5_CLOCKWISE_ANGLE_OUTPUT_MAP[placement["angle_deg_internal"]]
        assert int(library.placement_yaw_clockwise_deg[pid]) == expected
    assert int(library.placement_yaw_clockwise_deg[1]) == -90
    assert int(library.placement_yaw_clockwise_deg[3]) == 90

    expected_l_references = (
        (1.0, 2.0),
        (2.0, 1.0),
        (2.0, 2.0),
        (2.0, 2.0),
        (1.0, 2.0),
    )
    for category in ("L_yellow", "L_blue"):
        category_pids = [
            item["id"] for item in placements if item["category"] == category
        ]
        actual_references = tuple(
            (
                float(library.placement_row[pid]),
                float(library.placement_col[pid]),
            )
            for pid in category_pids
        )
        assert actual_references == expected_l_references


def test_loader_rejects_wrong_version_and_bad_pid(tmp_path):
    _write_small_v5_source(tmp_path)
    valid_path = tmp_path / "valid.npz"
    convert_v5_board_library(tmp_path, valid_path)
    library = load_v5_board_library(valid_path)

    wrong_version_path = tmp_path / "wrong_version.npz"
    _save_library_arrays(
        wrong_version_path,
        library,
        format_version=np.asarray(99, dtype=np.uint16),
    )
    with pytest.raises(ValueError, match="版本不支持"):
        load_v5_board_library(wrong_version_path)

    bad_pids = np.array(library.board_target_pid, copy=True)
    bad_pids[0, 0, 0] = library.placement_count
    bad_pid_path = tmp_path / "bad_pid.npz"
    _save_library_arrays(bad_pid_path, library, board_target_pid=bad_pids)
    with pytest.raises(ValueError, match="board_target_pid 越界"):
        load_v5_board_library(bad_pid_path)

    l_category_index = library.category_names.index("L_yellow")
    l_pid = int(np.flatnonzero(library.placement_category == l_category_index)[0])
    wrong_l_rows = np.array(library.placement_row, copy=True)
    wrong_l_rows[l_pid] += 0.5
    wrong_l_reference_path = tmp_path / "wrong_l_reference.npz"
    _save_library_arrays(
        wrong_l_reference_path,
        library,
        placement_row=wrong_l_rows,
    )
    with pytest.raises(ValueError, match="L 方块摆放参考点错误"):
        load_v5_board_library(wrong_l_reference_path)


@pytest.mark.skipif(not OFFICIAL_LIBRARY_PATH.exists(), reason="当前工作区没有正式 V5 NPZ")
def test_current_library_counts_and_task_layout_angle_equivalence():
    library = load_v5_board_library(OFFICIAL_LIBRARY_PATH)
    assert library.board_count == 8460
    assert len(np.unique(library.board_global_id)) == 7000
    assert library.placement_count == 2021
    unique_centers = {
        (float(row), float(col))
        for row, col in zip(library.placement_row, library.placement_col)
    }
    # 旧库的 365 是包围盒几何中心数量；L 改用实际长边中间格后，
    # 现场需要转换的不同摆放参考点共有 501 个，规模仍然很小。
    assert len(unique_centers) == 501

    targets = load_task_layout(str(PACKAGE_DIRECTORY / "config" / "task_layout.yaml"))
    for target in targets:
        expected_cells = tuple(sorted(tuple(cell) for cell in target["cells"]))
        matches = []
        for pid in range(library.placement_count):
            category = library.category_names[int(library.placement_category[pid])]
            cells = tuple(
                sorted(tuple(int(value) for value in cell) for cell in library.placement_cells[pid])
            )
            if category == target["category"] and cells == expected_cells:
                matches.append(pid)
        assert len(matches) == 1
        pid = matches[0]
        delta = normalize_rotation_delta(
            target["category"],
            float(library.placement_yaw_clockwise_deg[pid]),
            float(target["angle_deg"]),
            0.0,
        )
        assert delta == pytest.approx(0.0)
        assert float(library.placement_row[pid]) == pytest.approx(target["row"])
        assert float(library.placement_col[pid]) == pytest.approx(target["col"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
