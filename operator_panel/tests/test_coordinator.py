import threading
import time

import pytest

from operator_panel_lib.coordinator import OperationCoordinator, OperationRejected
from operator_panel_lib.event_bus import EventBus


PANEL_CONFIG = {
    "manual_control": {"timed_blow_seconds": 0.01, "reset_speed": 50},
    "timeouts": {
        "hardware_start_seconds": 0.1,
        "perception_start_seconds": 0.1,
        "process_stop_seconds": 0.1,
    },
    "output": {"servo_csv_output_dir": "/tmp"},
}


class FakeSupervisor:
    def __init__(self):
        self.runtime_stopped = threading.Event()
        self.state = {
            "hardware": {"running": True, "owned": True, "external": False},
            "runtime": {"running": True, "owned": True, "external": False, "mode": "formal"},
        }

    def snapshot(self):
        return self.state

    def stop_runtime(self):
        self.runtime_stopped.set()
        self.state["runtime"]["running"] = False
        return True

    def start_hardware(self):
        self.state["hardware"].update({
            "running": True, "owned": True, "external": False
        })
        return 1234

    def stop_hardware(self):
        self.state["hardware"]["running"] = False
        return True


class FakeRos:
    def __init__(self, stop_success=True):
        self.stop_success = stop_success
        self.stop_called = threading.Event()
        self.suction_calls = []
        self.status = {
            "ros_master": True,
            "camera_node": True,
            "control_node": True,
            "perception_node": True,
            "camera_frame_fresh": True,
            "last_frame_at": "now",
            "control_services_ready": True,
            "perception_services_ready": True,
            "control": {
                "available": True,
                "stop_latched": False,
                "commanded_suction_state": 0,
                "servo_target_known": False,
                "servo_target_angle_deg": 0,
                "motion_state_known": True,
                "motion_done": True,
            },
        }

    def health_snapshot(self):
        return self.status

    def stop_arm(self):
        self.stop_called.set()
        return {
            "success": self.stop_success,
            "stop_latched": True,
            "message": "已停止" if self.stop_success else "无法确认停稳",
        }

    def wait_hardware_ready(self, _timeout, frame_after=0.0):
        return dict(self.status, frame_after=frame_after)

    def get_control_status(self):
        return {
            "motion_state_known": self.status["control"]["motion_state_known"],
            "motion_done": self.status["control"]["motion_done"],
        }

    def clear_arm_stop(self):
        return {"success": True, "stop_latched": False, "message": "已解除"}

    def set_suction(self, state):
        self.suction_calls.append(state)
        return {"success": True, "state": state, "message": "完成"}


class FakeConfigManager:
    def list_configs(self):
        return []

    def get_config(self, file_id):
        assert file_id == "execution"
        return {
            "data": {
                "shooting_pose": [0, 0, 300, -180, 0, 90],
                "tool_motor": {
                    "initial_angle_deg": 180,
                    "lower_margin_deg": 10,
                    "upper_margin_deg": 350,
                },
            }
        }


class FakeStore:
    def add_run_summary(self, *_args, **_kwargs):
        pass


def make_coordinator(stop_success=True):
    bus = EventBus()
    supervisor = FakeSupervisor()
    ros = FakeRos(stop_success=stop_success)
    coordinator = OperationCoordinator(
        bus, supervisor, ros, FakeConfigManager(), FakeStore(), PANEL_CONFIG
    )
    return coordinator, supervisor, ros


def wait_operation(coordinator, operation_id, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        operation = coordinator.operation(operation_id)
        if operation["status"] != "running":
            return operation
        time.sleep(0.005)
    raise AssertionError("异步操作未在测试时限内完成")


def test停止通道不等待正在阻塞的普通操作且保持吸盘():
    coordinator, supervisor, ros = make_coordinator()
    release = threading.Event()
    normal = coordinator.submit("阻塞运动", lambda: release.wait(1.0))

    started = time.monotonic()
    stop = coordinator.emergency_stop()
    assert time.monotonic() - started < 0.1
    assert ros.stop_called.wait(0.2)
    assert supervisor.runtime_stopped.wait(0.2)
    completed = wait_operation(coordinator, stop["operation_id"])
    release.set()

    assert completed["status"] == "success"
    assert ros.suction_calls == []
    assert coordinator.snapshot()["hardware"]["stop_latched"] is True
    assert coordinator.snapshot()["task"]["state"] == "已中止"
    wait_operation(coordinator, normal["operation_id"])


def test停止令牌在网络StopMotion调用前立即生效():
    coordinator, _supervisor, ros = make_coordinator()

    class Token:
        requested = False

        def request(self):
            self.requested = True

    token = Token()
    coordinator._abort_token = token
    original_stop = ros.stop_arm

    def stop_arm():
        assert token.requested is True
        return original_stop()

    ros.stop_arm = stop_arm
    operation = coordinator.emergency_stop()

    assert wait_operation(coordinator, operation["operation_id"])["status"] == "success"


def test无法确认停止时持续全屏告警语义():
    coordinator, _supervisor, _ros = make_coordinator(stop_success=False)
    stop = coordinator.emergency_stop()
    completed = wait_operation(coordinator, stop["operation_id"])
    hardware = coordinator.snapshot()["hardware"]

    assert completed["status"] == "error"
    assert hardware["stop_latched"] is True
    assert hardware["stop_confirmed"] is False
    assert "物理急停" in hardware["emergency_warning"]


def test滞后的远端状态不能清除本地停止锁且解锁后可恢复():
    coordinator, _supervisor, ros = make_coordinator()
    stop = coordinator.emergency_stop()
    wait_operation(coordinator, stop["operation_id"])
    stale_health = dict(ros.status)
    stale_health["control"] = dict(ros.status["control"], stop_latched=False)

    coordinator.on_ros_health(stale_health)
    assert coordinator.snapshot()["hardware"]["stop_latched"] is True

    clear = coordinator.clear_stop()
    wait_operation(coordinator, clear["operation_id"])
    coordinator.on_ros_health(stale_health)
    assert coordinator.snapshot()["hardware"]["stop_latched"] is False


def test停止锁期间拒绝普通控制但解除后允许():
    coordinator, _supervisor, _ros = make_coordinator()
    with coordinator._lock:
        coordinator._state["hardware"]["stop_latched"] = True

    with pytest.raises(OperationRejected, match="停止锁"):
        coordinator._assert_manual_allowed()

    operation = coordinator.clear_stop()
    assert wait_operation(coordinator, operation["operation_id"])["status"] == "success"
    assert coordinator.snapshot()["hardware"]["stop_latched"] is False


def test停止锁期间允许重建硬件并立即同步远端停止锁():
    coordinator, supervisor, ros = make_coordinator()
    supervisor.state["hardware"].update({"running": False, "owned": False})
    with coordinator._lock:
        coordinator._local_stop_latched = True
        coordinator._state["hardware"]["stop_latched"] = True

    operation = coordinator.start_hardware()
    completed = wait_operation(coordinator, operation["operation_id"])

    assert completed["status"] == "success"
    assert completed["result"]["stop_confirmed"] is True
    assert ros.stop_called.is_set()
    assert coordinator.snapshot()["hardware"]["stop_latched"] is True


def test退出拒绝正在执行的普通操作():
    coordinator, _supervisor, _ros = make_coordinator()
    release = threading.Event()
    operation = coordinator.submit("机械臂复位", lambda: release.wait(1.0))

    with pytest.raises(OperationRejected, match="机械臂复位"):
        coordinator.request_exit()

    release.set()
    wait_operation(coordinator, operation["operation_id"])


def test退出自有硬件前必须确认机械臂停稳():
    coordinator, _supervisor, ros = make_coordinator()
    ros.status["control"].update({
        "motion_state_known": True,
        "motion_done": False,
    })

    with pytest.raises(OperationRejected, match="未明确确认停稳"):
        coordinator.request_exit()


def test停止自有硬件前必须确认机械臂停稳():
    coordinator, supervisor, ros = make_coordinator()
    ros.status["control"].update({
        "motion_state_known": True,
        "motion_done": False,
    })

    operation = coordinator.stop_hardware()
    completed = wait_operation(coordinator, operation["operation_id"])

    assert completed["status"] == "error"
    assert "未明确确认停稳" in completed["error"]
    assert supervisor.state["hardware"]["running"] is True


def test进阶顺序必须为0到6无重复排列():
    assert OperationCoordinator._validate_place_order(True, [6, 5, 4, 3, 2, 1, 0]) == [6, 5, 4, 3, 2, 1, 0]
    with pytest.raises(OperationRejected):
        OperationCoordinator._validate_place_order(True, [0, 1, 2, 3, 4, 5, 5])
    with pytest.raises(OperationRejected):
        OperationCoordinator._validate_place_order(True, [0, 1])


def test不会停止外部硬件进程():
    coordinator, supervisor, _ros = make_coordinator()
    supervisor.state["hardware"] = {
        "running": True, "owned": False, "external": True
    }
    operation = coordinator.stop_hardware()
    completed = wait_operation(coordinator, operation["operation_id"])

    assert completed["status"] == "error"
    assert "外部进程" in completed["error"]
    assert supervisor.state["hardware"]["running"] is True


def test定时喷气会自动关闭但停止锁会取消自动关闭():
    coordinator, _supervisor, ros = make_coordinator()
    operation = coordinator.control_suction("blow")
    wait_operation(coordinator, operation["operation_id"])
    time.sleep(0.03)
    assert ros.suction_calls == [1, 2]

    ros.suction_calls.clear()
    operation = coordinator.control_suction("blow")
    wait_operation(coordinator, operation["operation_id"])
    coordinator.emergency_stop()
    time.sleep(0.03)
    assert ros.suction_calls == [1]
