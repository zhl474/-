from pathlib import Path

import pytest

from operator_panel_lib.config_manager import ConfigError
from operator_panel_lib.event_bus import EventBus
from operator_panel_lib.web_app import create_app


class FakeStore:
    def list_history(self, *_args):
        return []

    def list_presets(self):
        return [{"id": 1, "name": "当前稳定配置", "protected": 1}]

class FakeConfigManager:
    def __init__(self):
        self.store = FakeStore()

    def list_configs(self):
        return [{"file_id": "execution", "label": "执行", "revision": "r"}]

    def get_config(self, file_id):
        if file_id != "execution":
            raise ConfigError("未知配置文件 ID")
        return {"file_id": file_id, "data": {}, "schema": {}, "revision": "r"}

    def inspect_read_only(self):
        return []

    def preset_diff(self, _preset_id):
        return {}


class FakeRos:
    def image_bytes(self):
        return b"\xff\xd8\xff\xd9", "now"


class FakeCoordinator:
    def __init__(self):
        self.calls = []

    def snapshot(self):
        return {"task": {"state": "空闲"}}

    def operation(self, operation_id):
        return {"operation_id": operation_id, "status": "success"} if operation_id == "known" else None

    def _accepted(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        return {"operation_id": name, "status": "accepted"}

    def start_hardware(self): return self._accepted("start_hardware")
    def stop_hardware(self): return self._accepted("stop_hardware")
    def start_runtime(self, *args): return self._accepted("start_runtime", *args)
    def stop_runtime(self): return self._accepted("stop_runtime")
    def prepare_task(self, **kwargs): return self._accepted("prepare_task", **kwargs)
    def confirm_task(self): return {"confirmed": True}
    def execute_task(self): return self._accepted("execute_task")
    def emergency_stop(self): return self._accepted("emergency_stop")
    def clear_stop(self): return self._accepted("clear_stop")
    def control_suction(self, *args, **kwargs): return self._accepted("suction", *args, **kwargs)
    def control_servo(self, *args, **kwargs): return self._accepted("servo", *args, **kwargs)
    def reset_arm(self, *args, **kwargs): return self._accepted("reset", *args, **kwargs)
    def get_pose(self): return {"tcp_pose": [0] * 6, "camera_pose": [0] * 6}
    def save_config(self, *args, **kwargs): return self._accepted("save_config", *args, **kwargs)
    def restore_history(self, *args): return self._accepted("restore_history", *args)
    def save_preset(self, name, overwrite=False): return self._accepted("save_preset", name, overwrite=overwrite)
    def delete_preset(self, preset_id): return self._accepted("delete_preset", preset_id)
    def restore_preset(self, payload): return self._accepted("restore_preset", payload)
    def deploy_calibration_pair(self, *args, **kwargs): return self._accepted("pair", *args, **kwargs)
    def deploy_hand_eye(self, *args, **kwargs): return self._accepted("hand_eye", *args, **kwargs)
    def request_exit(self): return {"exiting": True}


@pytest.fixture
def web(tmp_path):
    config = {
        "server": {"host": "127.0.0.1", "port": 8765},
        "output": {"debug_output_dir": str(tmp_path)},
    }
    coordinator = FakeCoordinator()
    bus = EventBus()
    app = create_app(
        coordinator, bus, FakeConfigManager(), FakeRos(), config, page_token="测试令牌"
    )
    app.config["TESTING"] = True
    return app.test_client(), coordinator, bus, tmp_path


def url(path):
    return {"base_url": "http://127.0.0.1:8765"}


def write_headers():
    return {"Origin": "http://127.0.0.1:8765", "X-Operator-Token": "测试令牌"}


def test首页和状态接口只接受本机Host(web):
    client, _coordinator, _bus, _tmp = web
    assert client.get("/", **url("/")).status_code == 200
    assert client.get("/api/state", **url("/api/state")).get_json()["task"]["state"] == "空闲"
    assert client.get("/api/state", base_url="http://evil.example").status_code == 403


def test所有写接口要求本机Origin和页面令牌(web):
    client, _coordinator, _bus, _tmp = web
    assert client.post("/api/process/hardware/start", json={}, **url("x")).status_code == 403
    assert client.post(
        "/api/process/hardware/start", json={},
        headers={"Origin": "http://127.0.0.1:8765", "X-Operator-Token": "错误"},
        **url("x"),
    ).status_code == 403
    response = client.post(
        "/api/process/hardware/start", json={}, headers=write_headers(), **url("x")
    )
    assert response.status_code == 202
    assert response.get_json()["operation_id"] == "start_hardware"


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/api/process/hardware/stop", {}),
        ("/api/process/runtime/start", {"mode": "formal"}),
        ("/api/process/runtime/stop", {}),
        ("/api/task/prepare", {"advanced": False}),
        ("/api/task/confirm", {}),
        ("/api/task/start", {}),
        ("/api/task/abort", {}),
        ("/api/control/stop", {}),
        ("/api/control/clear-stop", {}),
        ("/api/control/suction", {"action": "off"}),
        ("/api/control/servo", {"angle_deg": 180}),
        ("/api/control/reset", {"confirmed_pose": True}),
        ("/api/system/exit", {}),
    ],
)
def test主要写接口可由合法页面调用(web, path, payload):
    client, _coordinator, _bus, _tmp = web
    response = client.post(path, json=payload, headers=write_headers(), **url(path))
    assert response.status_code in (200, 202)


def test配置ID和图片ID均为固定白名单(web):
    client, _coordinator, _bus, tmp_path = web
    assert client.get("/api/config/execution", **url("x")).status_code == 200
    assert client.get("/api/config/../../etc/passwd", **url("x")).status_code == 404
    assert client.get("/api/images/camera", **url("x")).status_code == 200
    assert client.get("/api/images/../../etc/passwd", **url("x")).status_code == 404
    assert client.get("/api/images/not-allowlisted", **url("x")).status_code == 404

    (tmp_path / "方块上表面掩码.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    assert client.get("/api/images/block_mask", **url("x")).status_code == 200


def testSSE发送重连间隔和状态事件(web):
    client, _coordinator, _bus, _tmp = web
    response = client.get("/api/events", buffered=False, **url("x"))
    iterator = iter(response.response)
    assert b"retry: 1500" in next(iterator)
    event_chunk = next(iterator)
    assert b"event: state" in event_chunk
    response.close()
