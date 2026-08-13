import pytest

from operator_panel_lib import runtime


def test单实例锁阻止第二个后台并在释放后可再次获取(tmp_path):
    lock_path = tmp_path / "panel.lock"

    with runtime._single_instance(lock_path):
        with pytest.raises(runtime.AlreadyRunning, match="已经运行"):
            with runtime._single_instance(lock_path):
                pass

    with runtime._single_instance(lock_path):
        assert lock_path.read_text(encoding="utf-8").strip()


def test运行入口拒绝局域网监听地址(monkeypatch):
    monkeypatch.setattr(
        runtime,
        "_load_panel_config",
        lambda: {"server": {"host": "0.0.0.0", "port": 8765}},
    )

    with pytest.raises(RuntimeError, match="只允许监听 127.0.0.1"):
        runtime.run_panel()


def test第二次启动只重新打开已有页面(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(
        runtime,
        "_load_panel_config",
        lambda: {"server": {"host": "127.0.0.1", "port": 8765}},
    )
    monkeypatch.setattr(runtime, "_state_dir", lambda: tmp_path)
    monkeypatch.setattr(
        runtime.webbrowser,
        "open",
        lambda url, new=0: opened.append((url, new)) or True,
    )

    with runtime._single_instance(tmp_path / "panel.lock"):
        assert runtime.run_panel() == 0

    assert opened == [("http://127.0.0.1:8765", 2)]
