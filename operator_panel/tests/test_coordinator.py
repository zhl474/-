import threading
import time

import pytest

from operator_panel_lib.coordinator import OperationCoordinator, OperationRejected
from operator_panel_lib.event_bus import EventBus


PANEL_CONFIG = {
    "manual_control": {
        "timed_blow_seconds": 0.01,
        "reset_speed": 50,
        "relative_move_speed": 50,
    },
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
        self.prompt_responses = []
        self.prompt = {
            "pending": False,
            "prompt_id": "",
            "prompt_type": "",
            "message": "",
            "allow_fixed_yaml": False,
            "allow_continue_dynamic": False,
            "remaining_seconds": 0.0,
        }
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
        return dict(self.status, operator_prompt=dict(self.prompt))

    def operator_prompt_snapshot(self):
        return dict(self.prompt)

    def respond_operator_prompt(self, prompt_id, choice):
        self.prompt_responses.append((prompt_id, choice))
        if not self.prompt["pending"] or prompt_id != self.prompt["prompt_id"]:
            return {"success": False, "code": "conflict", "message": "提示已过期"}
        self.prompt["pending"] = False
        return {"success": True, "code": "accepted", "message": "已接受"}

    def cancel_operator_prompt(self, timeout=1.0):
        if not self.prompt["pending"]:
            return False
        return self.respond_operator_prompt(
            self.prompt["prompt_id"], "stop"
        )["success"]

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


def test停止运动会回答并取消正在等待的网页提示():
    coordinator, _supervisor, ros = make_coordinator()
    ros.prompt.update({
        "pending": True,
        "prompt_id": "prompt-stop",
        "prompt_type": "dynamic_board_failure",
        "message": "等待选择",
        "allow_fixed_yaml": True,
        "remaining_seconds": 60.0,
    })
    coordinator.on_ros_health(ros.health_snapshot())

    operation = coordinator.emergency_stop()

    assert wait_operation(coordinator, operation["operation_id"])["status"] == "success"
    assert ros.prompt_responses == [("prompt-stop", "stop")]


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


def test人工选择可绕过活动识别队列且继续动态盘面必须二次确认():
    coordinator, _supervisor, ros = make_coordinator()
    ros.prompt = {
        "pending": True,
        "prompt_id": "prompt-1",
        "prompt_type": "dynamic_board_failure",
        "message": "速度不一致",
        "allow_fixed_yaml": True,
        "allow_continue_dynamic": True,
        "remaining_seconds": 42.1,
    }
    coordinator.on_ros_health(ros.health_snapshot())
    with coordinator._lock:
        coordinator._active_operation = {
            "operation_id": "running",
            "kind": "识别与规划",
            "status": "running",
        }

    interaction_events = coordinator.event_bus.snapshot("interaction")
    assert interaction_events[-1]["data"]["prompt_id"] == "prompt-1"

    with pytest.raises(OperationRejected, match="二次确认"):
        coordinator.respond_interaction(
            "prompt-1",
            "continue_dynamic",
        )
    response = coordinator.respond_interaction(
        "prompt-1",
        "continue_dynamic",
        confirm_speed_mismatch=True,
    )

    assert response["accepted"] is True
    assert ros.prompt_responses == [("prompt-1", "continue_dynamic")]
    assert coordinator.snapshot()["interaction"]["pending"] is False


def test结束本轮只丢弃识别数据且不调用硬件():
    coordinator, _supervisor, ros = make_coordinator()
    with coordinator._lock:
        coordinator._state["task"].update({
            "state": "等待确认",
            "recognition_valid": True,
            "confirmed": False,
            "task_count": 34,
            "total": 34,
        })
        coordinator._runner = object()
        coordinator._prepare_response = object()

    response = coordinator.discard_task()
    task = coordinator.snapshot()["task"]

    assert response == {"discarded": True}
    assert task["state"] == "空闲"
    assert task["task_count"] == 0
    assert task["recognition_valid"] is False
    assert ros.stop_called.is_set() is False
    assert ros.suction_calls == []


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


class FakeDirect:
    """记录调用的直连硬件替身。"""

    def __init__(self):
        self.calls = []

    def set_suction(self, state):
        self.calls.append(("set_suction", state))
        return {"success": True, "message": "直连吸盘", "state": state}

    def rotate_tool(self, angle):
        self.calls.append(("rotate_tool", angle))
        return {"success": True, "message": "直连舵机", "angle_deg": angle}

    def move_arm(self, pose, speed, wait_until_stable=True):
        self.calls.append(("move_arm", pose, speed, wait_until_stable))
        return {"success": True, "message": "直连复位", "pose": pose}

    def get_pose(self):
        self.calls.append(("get_pose",))
        return {"success": True, "tcp_pose": [0] * 6, "camera_pose": None, "message": "ok"}

    def stop_motion(self):
        self.calls.append(("stop_motion",))
        return {"success": True, "stop_latched": True, "message": "已停止"}

    def is_stopped(self):
        self.calls.append(("is_stopped",))
        return True


def make_coordinator_with_direct():
    bus = EventBus()
    supervisor = FakeSupervisor()
    ros = FakeRos()
    direct = FakeDirect()
    coordinator = OperationCoordinator(
        bus, supervisor, ros, FakeConfigManager(), FakeStore(), PANEL_CONFIG,
        direct_hardware=direct,
    )
    return coordinator, supervisor, ros, direct


def test手动后端在control_node运行时走ROS():
    coordinator, _supervisor, _ros, direct = make_coordinator_with_direct()
    assert coordinator._control_node_running() is True
    assert coordinator._manual_backend() is coordinator.ros
    assert direct.calls == []


def test手动后端在control_node缺失时走直连():
    coordinator, _supervisor, ros, direct = make_coordinator_with_direct()
    ros.status["control_node"] = False
    assert coordinator._control_node_running() is False
    assert coordinator._manual_backend() is direct


def test手动后端无直连且无control_node时报错():
    coordinator, _supervisor, ros = make_coordinator()
    ros.status["control_node"] = False
    with pytest.raises(OperationRejected, match="直连硬件层未配置"):
        coordinator._manual_backend()


def test手动控制不再要求控制服务与相机画面():
    coordinator, _supervisor, ros = make_coordinator()
    ros.status["control_services_ready"] = False
    ros.status["camera_frame_fresh"] = False
    # 手动控制已脱离 ROS，不应再因控制服务/相机未就绪而被拒绝。
    coordinator._assert_manual_allowed()


def test吸盘在control_node缺失时走直连():
    coordinator, _supervisor, ros, direct = make_coordinator_with_direct()
    ros.status["control_node"] = False
    operation = coordinator.control_suction("suck")
    assert wait_operation(coordinator, operation["operation_id"])["status"] == "success"
    assert direct.calls[0] == ("set_suction", 0)


def test相对移动读取TCP并叠加增量后运动():
    coordinator, _supervisor, ros, direct = make_coordinator_with_direct()
    ros.status["control_node"] = False
    operation = coordinator.move_arm_relative(5.0, -3.0, 2.5, speed=30)
    assert wait_operation(coordinator, operation["operation_id"])["status"] == "success"
    assert direct.calls == [
        ("get_pose",),
        ("move_arm", [5.0, -3.0, 2.5, 0.0, 0.0, 0.0], 30, True),
    ]


def test相对移动拒绝非法增量与速度():
    coordinator, _supervisor, _ros, _direct = make_coordinator_with_direct()
    with pytest.raises(OperationRejected, match="增量"):
        coordinator.move_arm_relative("x", 0, 0)
    with pytest.raises(OperationRejected, match="速度"):
        coordinator.move_arm_relative(1, 0, 0, speed=0)


class FakePauseToken:
    def __init__(self):
        self._paused = False

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    @property
    def paused(self):
        return self._paused


class FakeAbortToken:
    def __init__(self):
        self._requested = False

    def request(self):
        self._requested = True

    @property
    def requested(self):
        return self._requested


def test暂停和恢复切换任务状态():
    coordinator, _supervisor, _ros = make_coordinator()
    coordinator._pause_token = FakePauseToken()
    with coordinator._lock:
        coordinator._state["task"]["state"] = "执行中"

    assert coordinator.pause_task()["paused"] is True
    assert coordinator._state["task"]["state"] == "已暂停"
    assert coordinator._pause_token.paused is True

    assert coordinator.resume_task()["resumed"] is True
    assert coordinator._state["task"]["state"] == "执行中"
    assert coordinator._pause_token.paused is False


def test暂停要求处于执行中():
    coordinator, _supervisor, _ros = make_coordinator()
    coordinator._pause_token = FakePauseToken()
    with pytest.raises(OperationRejected, match="不在执行中"):
        coordinator.pause_task()


def test停止执行请求中止令牌():
    coordinator, _supervisor, _ros = make_coordinator()
    with coordinator._lock:
        coordinator._state["task"]["state"] = "执行中"
    abort = FakeAbortToken()
    coordinator._abort_token = abort

    coordinator.stop_execution()
    assert abort.requested is True


def test停止执行要求处于执行中或已暂停():
    coordinator, _supervisor, _ros = make_coordinator()
    with pytest.raises(OperationRejected, match="没有正在执行"):
        coordinator.stop_execution()


def test暂停释放操作队列且继续恢复():
    coordinator, _supervisor, _ros = make_coordinator()
    coordinator._pause_token = FakePauseToken()
    with coordinator._lock:
        coordinator._state["task"]["state"] = "执行中"
        coordinator._active_operation = {
            "operation_id": "exec", "kind": "执行任务", "status": "running",
        }

    coordinator.pause_task()
    assert coordinator._active_operation is None
    assert coordinator._execution_operation["kind"] == "执行任务"

    coordinator.resume_task()
    assert coordinator._active_operation["kind"] == "执行任务"
    assert coordinator._execution_operation is None


def test继续在手動操作进行中时拒绝():
    coordinator, _supervisor, _ros = make_coordinator()
    coordinator._pause_token = FakePauseToken()
    with coordinator._lock:
        coordinator._state["task"]["state"] = "已暂停"
        coordinator._active_operation = {
            "operation_id": "manual", "kind": "吸盘控制", "status": "running",
        }

    with pytest.raises(OperationRejected, match="手动操作进行中"):
        coordinator.resume_task()
