import csv

import pytest

from competition_lib.servo_csv_logger import (
    CSV_FIELDNAMES,
    LEGACY_CSV_FIELDNAMES,
    LOG_SCHEMA_VERSION,
    ROUND_CSV_FIELDNAMES,
    ServoCsvLogger,
)


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
    assert LEGACY_CSV_FIELDNAMES == EXPECTED_FIELDS
    assert CSV_FIELDNAMES[:len(EXPECTED_FIELDS)] == EXPECTED_FIELDS
    assert block_fields == CSV_FIELDNAMES
    assert board_fields == CSV_FIELDNAMES
    assert block_rows[0]["方块类别"] == "T"
    assert block_rows[0]["日志模式版本"] == str(LOG_SCHEMA_VERSION)
    assert board_rows[0]["事件"] == "伺服失败"
    archive_fields, archive_rows = _read_rows(paths["block_archive"])
    assert archive_fields == CSV_FIELDNAMES
    assert archive_rows == block_rows


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


def test_logger_writes_round_events_and_metadata(tmp_path):
    logger = ServoCsvLogger(tmp_path, session_id="test_session")
    paths = logger.open(metadata={"测试元数据": "存在"})
    logger.write_round(
        "block",
        {
            "任务序号": 1,
            "目标类型": "方块",
            "事件": "成功后静止帧",
            "静止采样序号": 1,
            "像素误差X": 0.25,
            "像素误差Y": -0.5,
        },
    )
    logger.close()

    fields, rows = _read_rows(paths["block_round"])
    assert fields == ROUND_CSV_FIELDNAMES
    assert rows[0]["实验批次ID"] == "test_session"
    assert rows[0]["静止采样序号"] == "1"
    metadata = __import__("json").loads(logger.metadata_path.read_text(encoding="utf-8"))
    assert metadata["测试元数据"] == "存在"
    assert metadata["结束时间"]


def test_explicit_session_refuses_to_overwrite_archive(tmp_path):
    first = ServoCsvLogger(tmp_path, session_id="fixed")
    first.open()
    first.close()
    second = ServoCsvLogger(tmp_path, session_id="fixed")
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        second.open()
