import csv

from competition_lib.servo_csv_logger import CSV_FIELDNAMES, ServoCsvLogger


def _read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file_handle:
        return list(csv.DictReader(file_handle))


def test_logger_creates_two_csv_files_with_chinese_headers(tmp_path):
    logger = ServoCsvLogger(tmp_path)

    paths = logger.open()
    logger.write("block", {"事件": "伺服开始", "任务序号": 1, "方块类别": "T"})
    logger.write("board", {"事件": "伺服开始", "任务序号": 1, "托盘行": 2, "托盘列": 3})
    logger.close()

    assert paths["block"].name == "方块视觉伺服.csv"
    assert paths["board"].name == "托盘视觉伺服.csv"
    block_rows = _read_rows(paths["block"])
    board_rows = _read_rows(paths["board"])
    assert block_rows[0]["对象类型"] == "方块"
    assert block_rows[0]["运行编号"]
    assert block_rows[0]["方块类别"] == "T"
    assert board_rows[0]["对象类型"] == "托盘"
    assert board_rows[0]["托盘行"] == "2"
    assert CSV_FIELDNAMES[0] == "运行编号"


def test_logger_overwrites_rows_from_previous_execution(tmp_path):
    logger = ServoCsvLogger(tmp_path)
    paths = logger.open()
    logger.write("block", {"事件": "旧记录"})
    logger.close()

    logger.open()
    logger.write("block", {"事件": "新记录"})
    logger.close()

    rows = _read_rows(paths["block"])
    assert [row["事件"] for row in rows] == ["新记录"]
