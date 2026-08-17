import pytest
from pathlib import Path
from ctypes import CDLL, POINTER, c_char_p, c_int

import image_process_lib.advanced_planner as advanced_module
from image_process_lib.advanced_planner import (
    AdvancedPlanner,
    parse_idbs_config_result,
    parse_idbs_with_cells_result,
)


def test_advanced_planner_validates_seven_inputs_before_loading_library(monkeypatch):
    monkeypatch.setattr(advanced_module, "CDLL", lambda _path: (_ for _ in ()).throw(AssertionError("不应加载")))
    with pytest.raises(ValueError, match="7 类"):
        AdvancedPlanner("missing.so").build_layout([1], [1])


def test_advanced_planner_load_failure_only_affects_advanced_mode(monkeypatch):
    monkeypatch.setattr(advanced_module, "CDLL", lambda _path: (_ for _ in ()).throw(OSError("动态库不存在")))
    with pytest.raises(OSError, match="动态库不存在"):
        AdvancedPlanner("missing.so").build_layout([1] * 7, list(range(7)))


def test_idbs_with_cells_parser_strictly_reads_version_count_and_cells():
    raw = (
        b"IDBS_WITH_CELLS_V1,1,"
        b"T,0,2.0,1.0,1,1,2,1,2,2,3,1,0"
    )

    layout, full_rows = parse_idbs_with_cells_result(raw, expected_count=1)

    assert full_rows == "0"
    assert layout == [{
        "index": 0,
        "category": "T",
        "angle_deg": 0.0,
        "col": 2.0,
        "row": 1.0,
        "cells": ((1, 1), (2, 1), (2, 2), (3, 1)),
    }]
    with pytest.raises(ValueError, match="协议版本"):
        parse_idbs_with_cells_result(raw.replace(b"V1", b"V2"), 1)
    with pytest.raises(ValueError, match="数量错误"):
        parse_idbs_with_cells_result(raw, 2)


def test_idbs_config_parser_reconstructs_cells_and_keeps_block_index():
    # 新格式：名称,角度,x,y,方块索引, ... ,满行数
    raw = b"LR,-90,9.5,1.5,2,LL,-90,8.5,2.5,0,0"
    layout, full_rows = parse_idbs_config_result(raw, expected_count=2)

    assert full_rows == "0"
    assert len(layout) == 2
    assert layout[0]["category"] == "L_blue"
    assert layout[0]["angle_deg"] == -90.0
    assert layout[0]["col"] == 10.0
    assert layout[0]["row"] == 2.0
    assert layout[0]["block_index"] == 2
    assert len(layout[0]["cells"]) == 4
    assert len(set(layout[0]["cells"])) == 4
    assert layout[1]["block_index"] == 0


def test_real_advanced_library_returns_dynamic_cells_and_counts():
    library_path = Path(__file__).resolve().parents[2] / "jinjie" / "IDBSA.so"

    layout, _full_rows = AdvancedPlanner(str(library_path)).build_layout(
        [1, 1, 0, 0, 0, 0, 0],
        [0, 1, 2, 3, 4, 5, 6],
    )

    assert len(layout) == 2
    assert [item["category"] for item in layout] == ["L_blue", "L_yellow"]
    cells = [cell for item in layout for cell in item["cells"]]
    assert len(cells) == 8
    assert len(set(cells)) == 8
    assert all(len(item["cells"]) == 4 for item in layout)
    assert all("block_index" in item for item in layout)

    # 新版 IDBS 与 IDBS_Config 使用同一输出格式；不带坐标时走内置默认坐标。
    library = CDLL(str(library_path))
    library.IDBS.restype = c_char_p
    library.IDBS.argtypes = [POINTER(c_int), POINTER(c_int)]
    counts = (c_int * 7)(1, 1, 0, 0, 0, 0, 0)
    orders = (c_int * 7)(0, 1, 2, 3, 4, 5, 6)
    raw_fields = library.IDBS(counts, orders).decode("gbk").split(",")
    for index, item in enumerate(layout):
        offset = index * 5
        assert item["category"] == advanced_module.normalize_category_name(
            raw_fields[offset]
        )
        assert item["angle_deg"] == float(raw_fields[offset + 1])
        assert item["col"] == float(raw_fields[offset + 2]) + 0.5
        assert item["row"] == float(raw_fields[offset + 3]) + 0.5
        assert item["block_index"] == int(raw_fields[offset + 4])
