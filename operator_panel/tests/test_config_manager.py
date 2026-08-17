from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from ruamel.yaml import YAML

from operator_panel_lib.config_manager import (
    ConfigConflict,
    ConfigError,
    ConfigManager,
    DangerousChangeRequired,
)
from operator_panel_lib.constants import READ_ONLY_CONFIG_FILES, WRITABLE_CONFIG_FILES
from operator_panel_lib.state_store import StateStore


@pytest.fixture
def manager(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for file_id, entry in WRITABLE_CONFIG_FILES.items():
        copied = config_dir / f"{file_id}.yaml"
        copied.write_text(Path(entry["path"]).read_text(encoding="utf-8"), encoding="utf-8")
        monkeypatch.setitem(entry, "path", copied)
    store = StateStore(tmp_path / "state" / "panel.sqlite3")
    return ConfigManager(store, tmp_path / "state")


def test六份YAML完整读取且建立初始稳定预设(manager):
    files = manager.list_configs()

    assert len(files) == 6
    assert all(len(item["revision"]) == 64 for item in files)
    assert manager.store.list_presets()[0]["name"] == "当前稳定配置"
    for item in files:
        document = manager.get_config(item["file_id"])
        assert document["data"]
        assert document["schema"]

    camera = manager.get_config("camera")
    assert camera["schema"]["depth_camera.depth_work_mode"]["options"] == list(range(6))
    template = manager.get_config("template")
    assert template["schema"]["template_sizes.active_profile"]["options"] == ["high", "low"]


def test模板尺寸按类别overrides结构校验(manager):
    document = manager.get_config("template")

    good = deepcopy(document["data"])
    good["template_sizes"]["profiles"]["high"]["overrides"] = {
        "L_blue": {"x_runs": [36, 5, 36, 5, 37], "y_runs": [36, 5, 36]},
        "line": {"y_runs": [37]},
    }
    assert manager.validate_document("template", good)

    for bad in (
        {"L_blue": {"x_runs": [36, 5, 36, 5]}},
        {"L_blue": {"x_runs": [36, 5, 36, 5, 0]}},
        {"L_blue": {"x_runs": [36, 5, 36, 5, 36.5]}},
        {"L_blue": "bad"},
    ):
        broken = deepcopy(document["data"])
        broken["template_sizes"]["profiles"]["high"]["overrides"] = bad
        with pytest.raises(ConfigError):
            manager.validate_document("template", broken)


def test最低安全高度只保留在执行配置(manager):
    execution = manager.get_config("execution")
    assert execution["data"]["motion"]["minimum_tcp_z_mm"] == 162.0
    assert execution["restart_scope"] == "hardware"

    launch_path = Path(__file__).parents[2] / "competition" / "launch" / "hardware.launch"
    panel_path = Path(__file__).parents[2] / "operator_panel" / "config" / "panel.yaml"
    assert 'name="minimum_z"' not in launch_path.read_text(encoding="utf-8")
    assert "minimum_z_mm" not in panel_path.read_text(encoding="utf-8")


def test图像识别参数所有大字号标签均为中文(manager):
    schema = manager.get_config("perception")["schema"]

    def has_cjk(text):
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    for path, metadata in schema.items():
        assert has_cjk(metadata["label"]), f"{path} 的大字号标签仍是英文：{metadata['label']}"

    # 锁定若干关键映射与危险标记（models/calibration 子项风险不得因补标签而丢失）
    assert schema["models.detection"]["label"] == "方块检测模型"
    assert schema["models.detection"]["risk"] == "danger"
    assert schema["dynamic_board_selection"]["label"] == "动态盘面选择"
    assert schema["high_template_match.size_tolerance_px"]["label"] == "尺寸筛选容差"
    assert schema["calibration.block_pixel_to_tcp"]["label"] == "方块像素-TCP 标定文件"
    assert schema["calibration.block_pixel_to_tcp"]["risk"] == "danger"
    assert schema["board_servo.min_dot_circularity"]["label"] == "最小圆点圆度"


def test保存保留中文注释并记录差异(manager):
    document = manager.get_config("execution")
    document["data"]["servo"]["timing_debug"] = not document["data"]["servo"]["timing_debug"]
    result = manager.save_config(
        "execution", document["data"], document["revision"], confirm_dangerous=False
    )
    saved_text = Path(WRITABLE_CONFIG_FILES["execution"]["path"]).read_text(encoding="utf-8")

    assert result["changed"] is True
    assert "是否输出任务步骤及视觉伺服分阶段耗时" in saved_text
    assert "timing_debug" in result["diff"]
    assert manager.store.get_history(result["history_id"])["after_text"] == saved_text


def test并发修改返回revision冲突(manager):
    document = manager.get_config("execution")
    path = Path(WRITABLE_CONFIG_FILES["execution"]["path"])
    path.write_text(path.read_text(encoding="utf-8") + "\n# 外部修改\n", encoding="utf-8")

    with pytest.raises(ConfigConflict, match="其它程序修改"):
        manager.save_config("execution", document["data"], document["revision"])


def test禁止新增键删除键和改变固定数组长度(manager):
    document = manager.get_config("visual_servo")
    extra = deepcopy(document["data"])
    extra["new_key"] = 1
    with pytest.raises(ConfigError, match="不允许新增"):
        manager.save_config("visual_servo", extra, document["revision"])

    shortened = deepcopy(document["data"])
    shortened["camera_to_sucker_offset_mm"].pop()
    with pytest.raises(ConfigError, match="固定数组长度"):
        manager.save_config("visual_servo", shortened, document["revision"])


def test客户端JSON键顺序变化不改变原YAML顺序(manager):
    document = manager.get_config("camera")
    reordered = dict(reversed(list(document["data"].items())))
    reordered["rgb_camera"] = dict(reversed(list(reordered["rgb_camera"].items())))
    reordered["rgb_camera"]["brightness"] += 1

    manager.save_config("camera", reordered, document["revision"])
    text = Path(WRITABLE_CONFIG_FILES["camera"]["path"]).read_text(encoding="utf-8")

    assert text.index("rgb_camera:") < text.index("depth_camera:")
    assert text.index("  fps:") < text.index("  auto_exposure:")


def test危险映射矩阵必须二次确认(manager):
    document = manager.get_config("visual_servo")
    document["data"]["pixel_to_robot_matrix"][0][0] += 0.01

    with pytest.raises(DangerousChangeRequired):
        manager.save_config("visual_servo", document["data"], document["revision"])
    result = manager.save_config(
        "visual_servo", document["data"], document["revision"], confirm_dangerous=True
    )
    assert result["dangerous"] is True


def test吸盘偏移标定策略非法值被拒绝(manager):
    document = manager.get_config("visual_servo")
    invalid = deepcopy(document["data"])
    invalid["sucker_offset_strategy"] = "invented"
    with pytest.raises(ConfigError, match="sucker_offset_strategy 只能是"):
        manager.validate_document("visual_servo", invalid)


def test吸盘偏移左右只配一个被拒绝(manager):
    document = manager.get_config("visual_servo")
    broken = deepcopy(document["data"])
    del broken["camera_to_sucker_offset_left_mm"]
    with pytest.raises(ConfigError, match="必须成对配置"):
        manager.validate_document("visual_servo", broken)


def test三参数策略缺少左右偏移被拒绝(manager):
    document = manager.get_config("visual_servo")
    broken = deepcopy(document["data"])
    broken["sucker_offset_strategy"] = "THREE_CALIBRATION"
    del broken["camera_to_sucker_offset_left_mm"]
    del broken["camera_to_sucker_offset_right_mm"]
    with pytest.raises(ConfigError, match="THREE_CALIBRATION 策略必须同时配置"):
        manager.validate_document("visual_servo", broken)


def test切换到三参数策略且左右齐全可保存(manager):
    document = manager.get_config("visual_servo")
    updated = deepcopy(document["data"])
    updated["sucker_offset_strategy"] = "THREE_CALIBRATION"
    result = manager.save_config(
        "visual_servo", updated, document["revision"], confirm_dangerous=True
    )
    assert result["changed"] is True
    saved = manager.get_config("visual_servo")
    assert saved["data"]["sucker_offset_strategy"] == "THREE_CALIBRATION"


def test左右方块标定文件必须成对(manager):
    document = manager.get_config("perception")
    broken = deepcopy(document["data"])
    del broken["calibration"]["block_pixel_to_tcp_left"]
    with pytest.raises(ConfigError, match="必须成对配置"):
        manager.validate_document("perception", broken)


def test左右方块标定文件路径必须存在(manager):
    document = manager.get_config("perception")
    broken = deepcopy(document["data"])
    broken["calibration"]["block_pixel_to_tcp_left"] = (
        "image_process/config/不存在的标定.yaml"
    )
    with pytest.raises(ConfigError, match="指向的文件不存在"):
        manager.validate_document("perception", broken)


def test左右方块标定文件路径合法可保存(manager):
    """左右标定指向真实存在的单模型文件时（占位/同文件）校验与保存正常。"""
    document = manager.get_config("perception")
    updated = deepcopy(document["data"])
    updated["calibration"]["block_pixel_to_tcp_left"] = (
        document["data"]["calibration"]["block_pixel_to_tcp"]
    )
    updated["calibration"]["block_pixel_to_tcp_right"] = (
        document["data"]["calibration"]["block_pixel_to_tcp"]
    )
    assert manager.validate_document("perception", updated) is not None


def test数值有限值枚举和跨字段校验(manager):
    document = manager.get_config("execution")
    invalid = deepcopy(document["data"])
    invalid["servo"]["max_step_mm"] = float("nan")
    with pytest.raises(ConfigError, match="NaN"):
        manager.save_config("execution", invalid, document["revision"], confirm_dangerous=True)

    perception = manager.get_config("perception")
    invalid_mode = deepcopy(perception["data"])
    invalid_mode["task_sequence_optimizer"]["mode"] = "invented"
    with pytest.raises(ConfigError, match="只能是"):
        manager.save_config("perception", invalid_mode, perception["revision"])

    invalid_depth = deepcopy(perception["data"])
    invalid_depth["calibration_depth"]["min_valid_frames"] = (
        invalid_depth["calibration_depth"]["frame_count"] + 1
    )
    with pytest.raises(ConfigError, match="min_valid_frames 不能大于 frame_count"):
        manager.save_config("perception", invalid_depth, perception["revision"])

    camera = manager.get_config("camera")
    invalid_camera = deepcopy(camera["data"])
    invalid_camera["depth_camera"]["depth_work_mode"] = 6
    with pytest.raises(ConfigError, match="已有枚举 0～5"):
        manager.save_config("camera", invalid_camera, camera["revision"])


def test托盘相对观察高度差允许负值(manager):
    perception = manager.get_config("perception")
    data = deepcopy(perception["data"])
    data["calibration_depth"]["tray_tcp_below_block_observation_mm"] = -7.0

    result = manager.save_config(
        "perception",
        data,
        perception["revision"],
        confirm_dangerous=True,
    )

    assert result["changed"] is True
    assert (
        manager.get_config("perception")["data"]["calibration_depth"]
        ["tray_tcp_below_block_observation_mm"]
        == -7.0
    )


def test动态盘面execute与视觉伺服执行跨文件校验(manager):
    execution = manager.get_config("execution")
    if execution["data"]["servo"]["enabled"]:
        execution["data"]["servo"]["enabled"] = False
        manager.save_config("execution", execution["data"], execution["revision"])
        execution = manager.get_config("execution")
    perception = manager.get_config("perception")
    if perception["data"]["dynamic_board_selection"]["mode"] != "execute":
        perception["data"]["dynamic_board_selection"]["mode"] = "execute"
        manager.save_config("perception", perception["data"], perception["revision"])
    execution["data"]["servo"]["enabled"] = True

    with pytest.raises(ConfigError, match="跨文件校验.*servo.enabled=false"):
        manager.save_config(
            "execution",
            execution["data"],
            execution["revision"],
        )


def test历史可以一键恢复且恢复本身也有历史(manager):
    document = manager.get_config("controller")
    before = document["data"]["move_arm_timing_debug"]
    document["data"]["move_arm_timing_debug"] = not before
    changed = manager.save_config("controller", document["data"], document["revision"])
    current = manager.get_config("controller")
    restored = manager.restore_history(changed["history_id"], current["revision"])

    assert restored["changed"] is True
    assert manager.get_config("controller")["data"]["move_arm_timing_debug"] is before
    assert len(manager.store.list_history("controller")) == 2


def test命名预设保存差异和恢复六份快照(manager):
    preset = manager.save_preset("实验前")
    camera = manager.get_config("camera")
    camera["data"]["rgb_camera"]["brightness"] += 1
    manager.save_config("camera", camera["data"], camera["revision"])

    assert "camera" in manager.preset_diff(preset["id"])
    revisions = {item["file_id"]: item["revision"] for item in manager.list_configs()}
    restored = manager.restore_preset(preset["id"], revisions)
    assert restored["changed_files"] == ["camera"]
    assert manager.preset_diff(preset["id"]) == {}


def test原子替换失败时原文件不变(manager, monkeypatch):
    document = manager.get_config("camera")
    before_text = Path(WRITABLE_CONFIG_FILES["camera"]["path"]).read_text(encoding="utf-8")
    document["data"]["rgb_camera"]["brightness"] += 1

    def fail(_path, _text):
        raise OSError("模拟磁盘失败")

    monkeypatch.setattr(manager, "_atomic_replace", fail)
    with pytest.raises(OSError, match="模拟磁盘失败"):
        manager.save_config("camera", document["data"], document["revision"])
    assert Path(WRITABLE_CONFIG_FILES["camera"]["path"]).read_text(encoding="utf-8") == before_text


def test多文件预设恢复中途失败会回滚全部文件(manager, monkeypatch):
    preset = manager.save_preset("回滚基线")
    controller = manager.get_config("controller")
    controller["data"]["move_arm_timing_debug"] = not controller["data"]["move_arm_timing_debug"]
    manager.save_config("controller", controller["data"], controller["revision"])
    camera = manager.get_config("camera")
    camera["data"]["rgb_camera"]["brightness"] += 1
    manager.save_config("camera", camera["data"], camera["revision"])
    modified = {
        file_id: Path(WRITABLE_CONFIG_FILES[file_id]["path"]).read_text(encoding="utf-8")
        for file_id in ("controller", "camera")
    }
    revisions = {item["file_id"]: item["revision"] for item in manager.list_configs()}
    original_replace = manager._atomic_replace
    calls = []

    def fail_second(path, text):
        calls.append(str(path))
        if len(calls) == 2:
            raise OSError("模拟第二份文件替换失败")
        return original_replace(path, text)

    monkeypatch.setattr(manager, "_atomic_replace", fail_second)
    with pytest.raises(OSError, match="第二份文件替换失败"):
        manager.restore_preset(preset["id"], revisions)

    for file_id, expected_text in modified.items():
        assert Path(WRITABLE_CONFIG_FILES[file_id]["path"]).read_text(encoding="utf-8") == expected_text


def test方块托盘标定必须成对且generation一致(manager, tmp_path, monkeypatch):
    block_original = Path(READ_ONLY_CONFIG_FILES["block_calibration"]["path"])
    tray_original = Path(READ_ONLY_CONFIG_FILES["tray_calibration"]["path"])
    block_path = tmp_path / "block.yaml"
    tray_path = tmp_path / "tray.yaml"
    block_path.write_text(block_original.read_text(encoding="utf-8"), encoding="utf-8")
    tray_path.write_text(tray_original.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setitem(READ_ONLY_CONFIG_FILES["block_calibration"], "path", block_path)
    monkeypatch.setitem(READ_ONLY_CONFIG_FILES["tray_calibration"], "path", tray_path)
    yaml = YAML(typ="safe")
    block_text = block_path.read_text(encoding="utf-8")
    tray_data = yaml.load(tray_path.read_text(encoding="utf-8"))
    tray_data["generation_id"] = "不一致"
    from io import StringIO
    output = StringIO(); yaml.dump(tray_data, output)

    with pytest.raises(ConfigError, match="generation_id 不一致"):
        manager.deploy_calibration_pair(block_text, output.getvalue())

    result = manager.deploy_calibration_pair(block_text, tray_path.read_text(encoding="utf-8"))
    assert result["generation_id"]
    assert list(manager.backup_dir.glob("*.yaml"))


def test方块托盘标定部署复用正式加载器完整校验(manager, tmp_path, monkeypatch):
    block_original = Path(READ_ONLY_CONFIG_FILES["block_calibration"]["path"])
    tray_original = Path(READ_ONLY_CONFIG_FILES["tray_calibration"]["path"])
    block_path = tmp_path / "block_full.yaml"
    tray_path = tmp_path / "tray_full.yaml"
    block_path.write_text(block_original.read_text(encoding="utf-8"), encoding="utf-8")
    tray_path.write_text(tray_original.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setitem(READ_ONLY_CONFIG_FILES["block_calibration"], "path", block_path)
    monkeypatch.setitem(READ_ONLY_CONFIG_FILES["tray_calibration"], "path", tray_path)
    yaml = YAML(typ="safe")
    block_data = yaml.load(block_path.read_text(encoding="utf-8"))
    block_data["xy_model"]["parameters"]["uv_scale"] = [0.0, 1.0]
    from io import StringIO
    output = StringIO(); yaml.dump(block_data, output)

    with pytest.raises(ConfigError, match="完整 schema.*uv_scale"):
        manager.deploy_calibration_pair(
            output.getvalue(),
            tray_path.read_text(encoding="utf-8"),
        )


def test手眼矩阵必须有限4乘4并整文件备份部署(manager, tmp_path, monkeypatch):
    import operator_panel_lib.config_manager as config_module

    matrix_path = tmp_path / "T_wrist2camera.npy"
    np.save(matrix_path, np.eye(4), allow_pickle=False)
    matrix_path.chmod(0o644)
    monkeypatch.setattr(config_module, "HAND_EYE_MATRIX_PATH", matrix_path)

    with pytest.raises(ConfigError, match="有限的 4×4"):
        manager.deploy_hand_eye([[1, 0], [0, 1]])
    target = np.eye(4)
    target[0, 3] = 12.5
    result = manager.deploy_hand_eye(target.tolist())

    assert np.array_equal(np.load(matrix_path, allow_pickle=False), target)
    assert Path(result["backup"]).is_file()
    assert matrix_path.stat().st_mode & 0o777 == 0o644
