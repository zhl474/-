from operator_panel_lib.state_store import StateStore


def test历史预设和运行摘要可以持久化(tmp_path):
    store = StateStore(tmp_path / "panel.sqlite3")
    history_id = store.add_history(
        "execution", "a", "b", "before", "after", "diff", "git", "网页保存"
    )
    preset_id = store.save_preset(
        "低速方案", {"execution": {"text": "x", "revision": "r"}}
    )
    store.add_run_summary("start", "finish", "formal", "完成", 34, "")

    assert store.get_history(history_id)["before_text"] == "before"
    assert store.get_preset(preset_id)["snapshot"]["execution"]["text"] == "x"
    assert store.list_presets()[0]["name"] == "低速方案"


def test受保护初始预设不能覆盖或删除(tmp_path):
    store = StateStore(tmp_path / "panel.sqlite3")
    preset_id = store.save_preset("当前稳定配置", {}, protected=True)

    try:
        store.save_preset("当前稳定配置", {}, overwrite=True)
        assert False, "受保护预设不应允许覆盖"
    except PermissionError:
        pass
    try:
        store.delete_preset(preset_id)
        assert False, "受保护预设不应允许删除"
    except PermissionError:
        pass
