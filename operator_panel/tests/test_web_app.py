from pathlib import Path

import pytest

from operator_panel_lib.config_manager import ConfigError
from operator_panel_lib.coordinator import OperationRejected
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

    def ros_system_snapshot(self):
        return {
            "master_online": True,
            "distro": "noetic",
            "master_uri": "http://127.0.0.1:11311",
            "node_count": 1,
            "topic_count": 1,
            "service_count": 0,
            "nodes": ["/operator_panel"],
            "topics": [{
                "name": "/camera/image_rect",
                "type": "sensor_msgs/Image",
                "publishers": ["/camera_node"],
                "subscribers": ["/operator_panel"],
                "hz": 30.0,
            }],
            "services": [],
            "updated_at": "now",
        }


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
    def usb_occupancy(self): return {"camera": {"status": "free"}, "servo": {"status": "free"}}
    def start_runtime(self, *args): return self._accepted("start_runtime", *args)
    def stop_runtime(self): return self._accepted("stop_runtime")
    def prepare_task(self, **kwargs): return self._accepted("prepare_task", **kwargs)
    def confirm_task(self): return {"confirmed": True}
    def respond_interaction(self, *args, **kwargs): return self._accepted("interaction", *args, **kwargs)
    def discard_task(self): return {"discarded": True}
    def execute_task(self): return self._accepted("execute_task")
    def emergency_stop(self): return self._accepted("emergency_stop")
    def clear_stop(self): return self._accepted("clear_stop")
    def control_suction(self, *args, **kwargs): return self._accepted("suction", *args, **kwargs)
    def control_servo(self, *args, **kwargs): return self._accepted("servo", *args, **kwargs)
    def servo_sweep_start(self, *args, **kwargs): return self._accepted("servo_sweep_start", *args, **kwargs)
    def servo_sweep_stop(self): return self._accepted("servo_sweep_stop")
    def reset_arm(self, *args, **kwargs): return self._accepted("reset", *args, **kwargs)
    def move_arm_relative(self, *args, **kwargs): return self._accepted("move_relative", *args, **kwargs)
    def aruco_align(self, *args, **kwargs): return self._accepted("aruco_align", *args, **kwargs)
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
        coordinator, bus, FakeConfigManager(), FakeRos(), config,
        page_token="测试令牌", launch_log_dir=tmp_path / "运行日志",
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


def testUSB占用接口返回只读探测结果(web):
    client, _coordinator, _bus, _tmp = web
    response = client.get("/api/hardware/usb-occupancy", **url("x"))

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["camera"]["status"] == "free"
    assert payload["servo"]["status"] == "free"


def testROS系统接口返回Master实时图(web):
    client, _coordinator, _bus, _tmp = web
    response = client.get("/api/ros-system", **url("/api/ros-system"))

    assert response.status_code == 200
    assert response.get_json()["nodes"] == ["/operator_panel"]
    assert response.get_json()["topics"][0]["type"] == "sensor_msgs/Image"


def test首页使用简洁标题并展示ROS身份(web):
    client, _coordinator, _bus, _tmp = web
    html = client.get("/", **url("/")).get_data(as_text=True)

    for title in ("运行控制台", "手动控制", "参数中心", "只读配置与标定", "ROS 系统", "运行日志"):
        assert f'<h1 class="page-title">{title}</h1>' in html
    assert "ROS 单臂俄罗斯方块控制台" in html
    assert "/camera/image_rect" in html
    assert "sensor_msgs/Image" in html
    assert 'id="image-zoom-dialog"' in html
    assert 'id="task-interaction-dialog"' in html
    assert "从识别到抓放，一条清晰的操作链" not in html
    assert "把常用工具动作放在伸手可及的位置" not in html


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


def test人工选择接口完整转发提示编号和二次确认(web):
    client, coordinator, _bus, _tmp = web
    response = client.post(
        "/api/task/interaction/respond",
        json={
            "prompt_id": "prompt-1",
            "choice": "continue_dynamic",
            "confirm_speed_mismatch": True,
        },
        headers=write_headers(),
        **url("x"),
    )

    assert response.status_code == 200
    name, args, kwargs = coordinator.calls[-1]
    assert name == "interaction"
    assert args == ("prompt-1", "continue_dynamic")
    assert kwargs == {"confirm_speed_mismatch": True}


def test人工选择非法选项返回400且过期提示返回409(web):
    client, coordinator, _bus, _tmp = web

    def reject_invalid(*_args, **_kwargs):
        raise ValueError("当前提示不允许此选项")

    coordinator.respond_interaction = reject_invalid
    invalid = client.post(
        "/api/task/interaction/respond",
        json={"prompt_id": "prompt-1", "choice": "任意输入"},
        headers=write_headers(),
        **url("x"),
    )

    def reject_stale(*_args, **_kwargs):
        raise OperationRejected("人工选择提示已过期")

    coordinator.respond_interaction = reject_stale
    stale = client.post(
        "/api/task/interaction/respond",
        json={"prompt_id": "old-prompt", "choice": "stop"},
        headers=write_headers(),
        **url("x"),
    )

    assert invalid.status_code == 400
    assert invalid.get_json()["code"] == "invalid_request"
    assert stale.status_code == 409
    assert stale.get_json()["code"] == "operation_rejected"


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/api/process/hardware/stop", {}),
        ("/api/process/runtime/start", {"mode": "formal"}),
        ("/api/process/runtime/stop", {}),
        ("/api/task/prepare", {"advanced": False}),
        ("/api/task/confirm", {}),
        ("/api/task/interaction/respond", {"prompt_id": "p", "choice": "stop"}),
        ("/api/task/discard", {}),
        ("/api/task/start", {}),
        ("/api/task/abort", {}),
        ("/api/control/stop", {}),
        ("/api/control/clear-stop", {}),
        ("/api/control/suction", {"action": "off"}),
        ("/api/control/servo", {"angle_deg": 180}),
        ("/api/control/servo-sweep/start", {"min_deg": 0, "max_deg": 360, "wait_seconds": 2, "repeat_count": 0}),
        ("/api/control/servo-sweep/stop", {}),
        ("/api/control/reset", {"confirmed_pose": True}),
        ("/api/control/move-relative", {"dx": 1, "dy": 2, "dz": 3}),
        ("/api/tools/aruco-align", {"low_tcp_z_mm": 220.0, "confirmed": True}),
        ("/api/system/exit", {}),
    ],
)
def test主要写接口可由合法页面调用(web, path, payload):
    client, _coordinator, _bus, _tmp = web
    response = client.post(path, json=payload, headers=write_headers(), **url(path))
    assert response.status_code in (200, 202)


def testArUco对准接口转发低位Z和确认标记(web):
    client, coordinator, _bus, _tmp = web
    response = client.post(
        "/api/tools/aruco-align",
        json={"low_tcp_z_mm": 220.0, "confirmed": True},
        headers=write_headers(),
        **url("/api/tools/aruco-align"),
    )

    assert response.status_code == 202
    name, args, kwargs = coordinator.calls[-1]
    assert name == "aruco_align"
    assert args == (220.0,)
    assert kwargs == {"confirmed": True}


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


def testlaunch日志接口列出文件并返回尾部内容(web):
    client, _coordinator, _bus, tmp_path = web
    log_dir = tmp_path / "运行日志"
    log_dir.mkdir()
    (log_dir / "perception.launch.log").write_text("行一\n行二\n行三\n", encoding="utf-8")

    listed = client.get("/api/launch-log", **url("x"))
    assert listed.status_code == 200
    files = {item["kind"]: item for item in listed.get_json()["files"]}
    assert files["hardware"]["exists"] is False
    assert files["runtime"]["exists"] is True
    assert files["runtime"]["path"].endswith("perception.launch.log")
    assert files["task"]["exists"] is False

    content = client.get("/api/launch-log/runtime?lines=2", **url("x"))
    assert content.status_code == 200
    assert content.get_data(as_text=True) == "行二\n行三\n"

    download = client.get("/api/launch-log/runtime?download=1", **url("x"))
    assert download.status_code == 200
    assert download.get_data(as_text=True) == "行一\n行二\n行三\n"

    assert client.get("/api/launch-log/runtime?lines=abc", **url("x")).status_code == 400
    assert client.get("/api/launch-log/任意类别", **url("x")).status_code == 404
    assert client.get("/api/launch-log/hardware", **url("x")).status_code == 404


@pytest.fixture
def wildcard_web(tmp_path):
    config = {
        "server": {"host": "0.0.0.0", "port": 8765},
        "output": {"debug_output_dir": str(tmp_path)},
    }
    coordinator = FakeCoordinator()
    bus = EventBus()
    app = create_app(
        coordinator, bus, FakeConfigManager(), FakeRos(), config, page_token="测试令牌"
    )
    app.config["TESTING"] = True
    return app.test_client(), coordinator, bus


def test通配监听接受任意主机名访问(wildcard_web):
    client, _coordinator, _bus = wildcard_web
    response = client.get("/api/state", base_url="http://10.42.0.1:8765")
    assert response.status_code == 200
    assert client.get("/", base_url="http://10.42.0.1:8765").status_code == 200


def test通配监听写操作要求Origin与访问地址一致(wildcard_web):
    client, coordinator, _bus = wildcard_web
    remote = {"base_url": "http://10.42.0.1:8765"}
    assert client.post(
        "/api/process/hardware/start", json={}, headers={
            "Origin": "http://10.42.0.1:8765", "X-Operator-Token": "测试令牌",
        }, **remote,
    ).status_code == 202
    # 不同 Origin 仍被拒绝，避免被其他网页跨站借用
    assert client.post(
        "/api/process/hardware/start", json={}, headers={
            "Origin": "http://evil.example:8765", "X-Operator-Token": "测试令牌",
        }, **remote,
    ).status_code == 403
    # 错误端口的主机名仍被拒绝
    assert client.get("/api/state", base_url="http://10.42.0.1:9999").status_code == 403
