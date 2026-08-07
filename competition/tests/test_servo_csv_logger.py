import csv

import pytest

from competition_lib.servo_csv_logger import CSV_FIELDNAMES, ServoCsvLogger


EXPECTED_FIELDS = (
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


def _read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        return tuple(reader.fieldnames), list(reader)


def _success_row(category="T"):
    return {
        "方块类别": category,
        "事件": "伺服成功",
        "高位检测像素X": 120.5,
        "高位检测像素Y": 220.5,
        "实测TCP位置X": 1.0,
        "实测TCP位置Y": 2.0,
        "实测TCP位置Z": 3.0,
    }


def test_logger_creates_two_csv_files_with_exact_calibration_headers(tmp_path):
    logger = ServoCsvLogger(tmp_path)
    paths = logger.open()
    logger.write("block", _success_row("T"))
    logger.write("board", {**_success_row("square"), "事件": "伺服失败"})
    logger.close()

    assert paths["block"].name == "方块视觉伺服.csv"
    assert paths["board"].name == "托盘视觉伺服.csv"
    block_fields, block_rows = _read_rows(paths["block"])
    board_fields, board_rows = _read_rows(paths["board"])
    assert CSV_FIELDNAMES == EXPECTED_FIELDS
    assert block_fields == EXPECTED_FIELDS
    assert board_fields == EXPECTED_FIELDS
    assert block_rows[0]["方块类别"] == "T"
    assert board_rows[0]["事件"] == "伺服失败"


def test_logger_overwrites_rows_from_previous_execution(tmp_path):
    logger = ServoCsvLogger(tmp_path)
    paths = logger.open()
    logger.write("block", _success_row("old"))
    logger.close()

    logger.open()
    logger.write("block", _success_row("new"))
    logger.close()

    _fields, rows = _read_rows(paths["block"])
    assert [row["方块类别"] for row in rows] == ["new"]


def test_logger_rejects_nonfinal_event(tmp_path):
    logger = ServoCsvLogger(tmp_path)
    logger.open()
    with pytest.raises(ValueError, match="只允许伺服成功或伺服失败"):
        logger.write("block", {**_success_row(), "事件": "执行修正"})
    logger.close()
