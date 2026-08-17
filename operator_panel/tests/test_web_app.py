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
    def __init__(self):
        self.exposure_calls = []
        self.yolo_preview_calls = []
        self.exposure_error = None

    def image_bytes(self):
        return b"\xff\xd8\xff\xd9", "now"

    def yolo_preview_snapshot(self):
        return b"\xff\xd8\xff\xd9-yolo", "识别到 2 个: squarex2", "now"

    def get_exposure_state(self):
        if self.exposure_error:
            raise self.exposure_error
        return {
            "rgb": {
                "auto_exposure": True, "exposure": 156,
                "exposure_min": 1, "exposure_max": 1665,
                "gain": 16, "gain_min": 0, "gain_max": 255,
            },
            "depth": {
                "auto_exposure": True, "exposure": 3000,
                "exposure_min": 1, "exposure_max": 100000,
                "gain": 1000, "gain_min": 16, "gain_max": 255,
            },
            "message": "查询成功",
        }

    def set_exposure_param(self, sensor, key, value):
        self.exposure_calls.append((sensor, key, value))
        if self.exposure_error:
            raise self.exposure_error
        return {"success": True, "applied": value, "message": "设置成功"}

    def set_yolo_preview_enabled(self, enabled):
        self.yolo_preview_calls.append(bool(enabled))
        return {"enabled": bool(enabled)}

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


@pytest.fixture
def web_ros(tmp_path):
    """同 web，但把 FakeRos 实例也交出来，供相机调参接口断言使用。"""
    config = {
        "server": {"host": "127.0.0.1", "port": 8765},
        "output": {"debug_output_dir": str(tmp_path)},
    }
    ros = FakeRos()
    app = create_app(
        FakeCoordinator(), EventBus(), FakeConfigManager(), ros, config,
        page_token="测试令牌", launch_log_dir=tmp_path / "运行日志",
    )
    app.config["TESTING"] = True
    return app.test_client(), ros


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

    for title in ("运行控制台", "手动控制", "参数中心", "只读配置与标定", "ROS 系统", "运行日志", "相机调参"):
        assert f'<h1 class="page-title">{title}</h1>' in html
    assert 'data-view="camera-tune"' in html
    assert 'id="ct-yolo-image"' in html
    assert "ROS 单臂俄罗斯方块控制台" in html
    assert "/camera/image_rect" in html
    assert "sensor_msgs/Image" in html
    assert 'id="image-zoom-dialog"' in html
    assert 'id="task-interaction-dialog"' in html
    assert 'Line 模板像素尺寸' in html
    assert 'id="line-template-calc"' in html
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


def testLine模板像素尺寸接口返回计算结果(web):
    client, _coordinator, _bus, _tmp = web
    response = client.get(
        "/api/tools/line-template-size",
        query_string={
            "p1_x": 436, "p1_y": 442,
            "p2_x": 843, "p2_y": 370,
            "short_side_mm": 17.8, "long_side_mm": 78.2,
        },
        **url("/api/tools/line-template-size"),
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["long_side_px"] > 0
    assert payload["block_px"] > 0
    assert payload["connector_px"] > 0
    assert set(payload) == {"long_side_px", "block_px", "connector_px"}


def testLine模板像素尺寸接口缺少参数返回400(web):
    client, _coordinator, _bus, _tmp = web
    response = client.get(
        "/api/tools/line-template-size",
        query_string={"p1_x": 1, "p1_y": 2, "p2_x": 3},
        **url("/api/tools/line-template-size"),
    )

    assert response.status_code == 400
    assert response.get_json()["code"] == "invalid_request"


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


def test定位Z链路端点返回实时计算结构(web):
    client, _coordinator, _bus, _tmp_path = web
    response = client.get("/api/localization-z-chain", **url("/api/localization-z-chain"))

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["mode"] in {"fixed_constant", "calibration_z_plane"}
    assert payload["mode_label"]
    assert {item["symbol"] for item in payload["constants"]} >= {"B", "T", "h", "o", "z_min"}
    assert payload["rows"]
    assert all({"stage", "formula", "substitution", "value"} <= set(row) for row in payload["rows"])
    assert all({"name", "detail", "ok"} <= set(check) for check in payload["checks"])
    assert payload["notes"]


def test相机曝光状态接口返回当前值和范围(web_ros):
    client, ros = web_ros
    response = client.get("/api/camera/exposure", **url("/api/camera/exposure"))

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["rgb"]["exposure"] == 156
    assert payload["rgb"]["exposure_max"] == 1665
    assert payload["depth"]["gain"] == 1000


def test相机曝光状态接口在相机不可用时返回503(web_ros):
    client, ros = web_ros
    ros.exposure_error = RuntimeError("service [/camera/get_exposure_state] unavailable")
    response = client.get("/api/camera/exposure", **url("/api/camera/exposure"))

    assert response.status_code == 503
    assert response.get_json()["code"] == "camera_unavailable"


def test相机曝光设置接口校验参数并透传(web_ros):
    client, ros = web_ros
    ok = client.post(
        "/api/camera/exposure", json={"sensor": "rgb", "key": "exposure", "value": 300},
        headers=write_headers(), **url("/api/camera/exposure"),
    )
    assert ok.status_code == 200
    assert ok.get_json()["applied"] == 300
    assert ros.exposure_calls == [("rgb", "exposure", 300)]

    for payload in (
        {"sensor": "color", "key": "exposure", "value": 1},
        {"sensor": "rgb", "key": "brightness", "value": 1},
        {"sensor": "rgb", "key": "exposure", "value": True},
        {"sensor": "rgb", "key": "exposure"},
    ):
        rejected = client.post(
            "/api/camera/exposure", json=payload,
            headers=write_headers(), **url("/api/camera/exposure"),
        )
        assert rejected.status_code == 400
    assert ros.exposure_calls == [("rgb", "exposure", 300)]


def test相机曝光设置接口在相机不可用时返回503(web_ros):
    client, ros = web_ros
    ros.exposure_error = RuntimeError("相机节点未启动")
    response = client.post(
        "/api/camera/exposure", json={"sensor": "depth", "key": "gain", "value": 32},
        headers=write_headers(), **url("/api/camera/exposure"),
    )

    assert response.status_code == 503
    assert response.get_json()["code"] == "camera_unavailable"


def testYOLO预览开关接口透传启停(web_ros):
    client, ros = web_ros
    response = client.post(
        "/api/camera/yolo-preview", json={"enabled": True},
        headers=write_headers(), **url("/api/camera/yolo-preview"),
    )

    assert response.status_code == 200
    assert response.get_json() == {"enabled": True}
    assert ros.yolo_preview_calls == [True]


def testYOLO识别图接口返回缓存JPEG(web_ros):
    client, _ros = web_ros
    response = client.get("/api/images/yolo", **url("/api/images/yolo"))

    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert response.data == b"\xff\xd8\xff\xd9-yolo"
