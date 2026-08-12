import pytest
from pathlib import Path
from ctypes import CDLL, c_char_p

import image_process_lib.advanced_planner as advanced_module
from image_process_lib.advanced_planner import (
    AdvancedPlanner,
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


def test_real_advanced_library_returns_dynamic_cells_and_counts():
    library_path = Path(__file__).resolve().parents[2] / "jinjie" / "jinjie_libtetris.so"

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

    # 旧符号必须继续存在，且新版中心必须与旧接口经 Python +0.5 后完全一致。
    library = CDLL(str(library_path))
    library.IDBS.restype = c_char_p
    old_fields = library.IDBS(
        1, 1, 0, 0, 0, 0, 0,
        0, 1, 2, 3, 4, 5, 6,
    ).decode("gbk").split(",")
    for index, item in enumerate(layout):
        offset = index * 4
        assert item["category"] == old_fields[offset]
        assert item["angle_deg"] == float(old_fields[offset + 1])
        assert item["col"] == float(old_fields[offset + 2]) + 0.5
        assert item["row"] == float(old_fields[offset + 3]) + 0.5
