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

    def service_names(self):
        return {"/camera/stable_world_points", "/control/move_arm"}

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


def _free_usb_occupancy():
    return {
        "camera": {
            "key": "camera",
            "label": "Orbbec Gemini 335 相机",
            "status": "free",
            "present": True,
            "nodes": ["/dev/bus/usb/002/006"],
            "occupants": [],
            "message": "设备在线，当前未发现占用进程",
        },
        "servo": {
            "key": "servo",
            "label": "HL-340 USB 转串口（舵机）",
            "status": "free",
            "present": True,
            "nodes": ["/dev/servo_motor"],
            "occupants": [],
            "message": "设备在线，当前未发现占用进程",
        },
    }


def make_coordinator(stop_success=True, usb_probe=None):
    bus = EventBus()
    supervisor = FakeSupervisor()
    ros = FakeRos(stop_success=stop_success)
    coordinator = OperationCoordinator(
        bus, supervisor, ros, FakeConfigManager(), FakeStore(), PANEL_CONFIG,
        usb_probe=usb_probe or _free_usb_occupancy,
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


def make_coordinator_with_direct(usb_probe=None):
    bus = EventBus()
    supervisor = FakeSupervisor()
    ros = FakeRos()
    direct = FakeDirect()
    coordinator = OperationCoordinator(
        bus, supervisor, ros, FakeConfigManager(), FakeStore(), PANEL_CONFIG,
        direct_hardware=direct,
        usb_probe=usb_probe or _free_usb_occupancy,
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


def wait_sweep_idle(coordinator, timeout=2.0):
    deadline = time.monotonic() + timeout
    while coordinator.snapshot()["hardware"]["servo_sweep_running"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert coordinator.snapshot()["hardware"]["servo_sweep_running"] is False


def test舵机往复测试在control_node缺失时走直连并按序发送角度():
    coordinator, _supervisor, ros, direct = make_coordinator_with_direct()
    ros.status["control_node"] = False

    response = coordinator.servo_sweep_start(10, 350, 0.05, 1)
    assert response["started"] is True
    wait_sweep_idle(coordinator)

    assert coordinator.snapshot()["hardware"]["servo_sweep_cycle"] == 1
    assert direct.calls == [
        ("rotate_tool", 10.0),
        ("rotate_tool", 350.0),
        ("rotate_tool", 10.0),
    ]


def test舵机往复测试超出安全边界需要二次确认():
    coordinator, _supervisor, _ros, _direct = make_coordinator_with_direct()
    with pytest.raises(OperationRejected, match="安全边界"):
        coordinator.servo_sweep_start(0, 360, 1.0, 0)
    assert coordinator.servo_sweep_start(0, 360, 0.05, 1, confirm_outside_safe=True)["started"] is True
    wait_sweep_idle(coordinator)


def test舵机往复测试拒绝非法参数():
    coordinator, _supervisor, _ros, _direct = make_coordinator_with_direct()
    with pytest.raises(OperationRejected, match="起始角必须小于终止角"):
        coordinator.servo_sweep_start(350, 10, 1.0, 0)
    with pytest.raises(OperationRejected, match="等待时间"):
        coordinator.servo_sweep_start(10, 350, 0, 0)
    with pytest.raises(OperationRejected, match="次数"):
        coordinator.servo_sweep_start(10, 350, 1.0, -1)


def test停止往复会停止无限循环():
    coordinator, _supervisor, ros, _direct = make_coordinator_with_direct()
    ros.status["control_node"] = False

    coordinator.servo_sweep_start(10, 350, 0.05, 0)
    assert coordinator.snapshot()["hardware"]["servo_sweep_running"] is True
    assert coordinator.servo_sweep_stop()["stopping"] is True
    wait_sweep_idle(coordinator)


def test往复测试进行中拒绝其它手动控制():
    coordinator, _supervisor, ros, _direct = make_coordinator_with_direct()
    ros.status["control_node"] = False
    coordinator.servo_sweep_start(10, 350, 0.05, 0)

    with pytest.raises(OperationRejected, match="往复"):
        coordinator._assert_manual_allowed()

    coordinator.servo_sweep_stop()
    wait_sweep_idle(coordinator)


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


def _with_execution_motion_config(coordinator, minimum_tcp_z_mm=163.0):
    """给 FakeConfigManager 的 execution 数据补齐 motion 段。"""
    base_get_config = coordinator.config_manager.get_config

    def get_config(file_id):
        result = base_get_config(file_id)
        result["data"]["motion"] = {"minimum_tcp_z_mm": minimum_tcp_z_mm}
        return result

    coordinator.config_manager.get_config = get_config


def testArUco对准未确认时拒绝():
    coordinator, _supervisor, _ros = make_coordinator()
    with pytest.raises(OperationRejected, match="确认"):
        coordinator.aruco_align(220.0)


def testArUco对准拒绝低于安全下限的Z():
    coordinator, _supervisor, _ros = make_coordinator()
    _with_execution_motion_config(coordinator, minimum_tcp_z_mm=163.0)

    with pytest.raises(OperationRejected, match="安全下限"):
        coordinator.aruco_align(100.0, confirmed=True)


def testArUco对准作为串行操作执行并使识别失效(monkeypatch):
    coordinator, _supervisor, _ros = make_coordinator()
    _with_execution_motion_config(coordinator, minimum_tcp_z_mm=163.0)
    with coordinator._lock:
        coordinator._state["task"].update({
            "state": "等待确认",
            "recognition_valid": True,
            "confirmed": True,
        })

    captured = {}

    def fake_execute(low_z):
        captured["low_z"] = low_z
        return {"success": True, "message": "完成"}

    monkeypatch.setattr(coordinator, "_execute_aruco_align", fake_execute)
    operation = coordinator.aruco_align(220.0, confirmed=True)

    assert operation["low_tcp_z_mm"] == 220.0
    completed = wait_operation(coordinator, operation["operation_id"])
    assert completed["status"] == "success"
    assert captured == {"low_z": 220.0}
    assert coordinator.snapshot()["task"]["state"] == "空闲"
    assert coordinator.snapshot()["task"]["recognition_valid"] is False


def testArUco对准子进程可被终止():
    import subprocess

    coordinator, _supervisor, _ros = make_coordinator()
    process = subprocess.Popen(
        ["sleep", "30"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    with coordinator._lock:
        coordinator._aruco_align_process = process

    coordinator._terminate_aruco_align_process()

    assert process.poll() is not None
    assert coordinator._aruco_align_abort.is_set()


def test停止通道立即请求终止ArUco对准():
    coordinator, _supervisor, _ros = make_coordinator()
    coordinator._aruco_align_abort.clear()

    operation = coordinator.emergency_stop()

    assert coordinator._aruco_align_abort.is_set()
    wait_operation(coordinator, operation["operation_id"])


def testArUco对准子进程输出被解析并转发日志(tmp_path, monkeypatch):
    import operator_panel_lib.coordinator as coordinator_module

    fake_script = tmp_path / "fake_align.py"
    fake_script.write_text(
        "import sys\n"
        "print('已到达高位拍摄位姿')\n"
        "print('ArUco 对准成功。')\n"
        "print('最终命令 TCP：X=1.000, Y=2.000, Z=220.000, "
        "R=-180.000, P=0.000, YAW=90.000')\n"
        "print('最终实测 TCP：X=1.100, Y=2.200, Z=220.000, "
        "R=-180.000, P=0.000, YAW=90.000')\n"
        "print('对准录像已保存：/home/zhl/桌面/aruco单次对准/20260816-120000_单次对准录像.avi')\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(coordinator_module, "ARUCO_ALIGN_SCRIPT_PATH", fake_script)
    coordinator, _supervisor, _ros = make_coordinator()

    result = coordinator._execute_aruco_align(220.0)

    assert result["actual_tcp_pose"] == [1.1, 2.2, 220.0, -180.0, 0.0, 90.0]
    assert result["final_command_pose"] == [1.0, 2.0, 220.0, -180.0, 0.0, 90.0]
    assert result["video_path"] == (
        "/home/zhl/桌面/aruco单次对准/20260816-120000_单次对准录像.avi"
    )
    log_messages = [
        item["data"]["message"]
        for item in coordinator.event_bus.snapshot("log")
    ]
    assert "已到达高位拍摄位姿" in log_messages
    assert "ArUco 对准成功。" in log_messages


def test启动硬件前相机被外部进程占用时快速拒绝(monkeypatch):
    occupancy = _free_usb_occupancy()
    occupancy["camera"].update({
        "status": "occupied",
        "message": "被占用",
        "occupants": [{
            "pid": 12345,
            "cmdline": "python camera_profile_probe.py",
            "is_self": False,
        }],
    })
    coordinator, supervisor, _ros = make_coordinator(usb_probe=lambda: occupancy)
    started = threading.Event()
    monkeypatch.setattr(supervisor, "start_hardware", lambda: started.set())

    operation = coordinator.start_hardware()
    completed = wait_operation(coordinator, operation["operation_id"])

    assert completed["status"] == "error"
    assert "相机" in completed["error"]
    assert "PID 12345" in completed["error"]
    assert not started.is_set()


def test启动硬件前设备缺失时快速拒绝():
    occupancy = _free_usb_occupancy()
    occupancy["camera"].update({
        "status": "missing",
        "present": False,
        "message": "本机 USB 总线上未发现相机",
    })
    coordinator, _supervisor, _ros = make_coordinator(usb_probe=lambda: occupancy)

    operation = coordinator.start_hardware()
    completed = wait_operation(coordinator, operation["operation_id"])

    assert completed["status"] == "error"
    assert "本机 USB 总线上未发现相机" in completed["error"]


def testUSB占用结果会标注自身与控制台硬件节点():
    occupancy = _free_usb_occupancy()
    occupancy["camera"].update({
        "status": "occupied",
        "occupants": [{
            "pid": 200,
            "cmdline": "python camera_node.py __name:=camera_node",
            "is_self": False,
        }],
    })
    occupancy["servo"].update({
        "status": "occupied",
        "occupants": [{
            "pid": 300,
            "cmdline": "python controller.py __name:=control_node",
            "is_self": False,
        }],
    })
    coordinator, _supervisor, _ros = make_coordinator(usb_probe=lambda: occupancy)

    result = coordinator.usb_occupancy()

    assert result["camera"]["occupants"][0]["owner"] == "panel"
    assert result["servo"]["occupants"][0]["owner"] == "panel"


def test执行配置应用硬件范围会重启硬件和感知(monkeypatch):
    coordinator, supervisor, ros = make_coordinator()
    events = []

    # execution.yaml 由硬件 controller 和 perception 同时读取，验证两套节点都被重启。
    for method_name in ("stop_runtime", "stop_hardware", "start_hardware"):
        original = getattr(supervisor, method_name)

        def wrapped(*args, _original=original, _name=method_name, **kwargs):
            events.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(supervisor, method_name, wrapped)

    def start_runtime(mode):
        events.append(f"start_runtime:{mode}")
        supervisor.state["runtime"]["running"] = True
        supervisor.state["runtime"]["mode"] = mode

    monkeypatch.setattr(supervisor, "start_runtime", start_runtime, raising=False)
    monkeypatch.setattr(
        ros,
        "wait_perception_ready",
        lambda _timeout: events.append("perception_ready"),
        raising=False,
    )

    coordinator._apply_restart({"hardware"})

    assert events == [
        "stop_runtime",
        "stop_hardware",
        "start_hardware",
        "start_runtime:formal",
        "perception_ready",
    ]
