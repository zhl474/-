import io
import threading

import pytest

from operator_panel_lib.event_bus import EventBus
from operator_panel_lib.process_supervisor import ProcessConflict, ProcessSupervisor


class FakeProcess:
    _next_pid = 1000

    def __init__(self):
        self.pid = FakeProcess._next_pid
        FakeProcess._next_pid += 1
        self.stdout = io.StringIO("")
        self.returncode = None
        self.release = threading.Event()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.release.wait(timeout or 0.1)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def test只会构造固定硬件和感知launch命令():
    commands = []

    def popen(command, **_kwargs):
        commands.append(command)
        return FakeProcess()

    supervisor = ProcessSupervisor(
        EventBus(), "/tmp/调试", "/tmp/标定", node_provider=lambda: set(), popen_factory=popen
    )
    supervisor.start_hardware()
    supervisor.start_runtime("calibration")

    assert commands[0] == ["roslaunch", "competition", "hardware.launch"]
    assert commands[1][:3] == ["roslaunch", "competition", "perception.launch"]
    assert "calibration_mode:=true" in commands[1]
    with pytest.raises(ValueError):
        supervisor.start_runtime("任意launch")


def test外部节点冲突时不启动也不结束外部节点():
    called = []
    supervisor = ProcessSupervisor(
        EventBus(), "/tmp", "/tmp",
        node_provider=lambda: {"/camera_node"},
        popen_factory=lambda *args, **kwargs: called.append((args, kwargs)),
    )

    with pytest.raises(ProcessConflict, match="外部 ROS 节点"):
        supervisor.start_hardware()
    assert called == []
