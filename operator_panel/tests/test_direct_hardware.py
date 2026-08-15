"""直连硬件层单元测试：用替身逐调用核对与现有程序一致。

对照来源：
- 复位/move_arm、吸盘/set_suction、读位姿/get_pose → control/scripts/controller.py
- 舵机/rotate_tool → tools/hardware/测试舵机.py
"""

import numpy as np
import pytest

from operator_panel_lib import direct_hardware as dh
from operator_panel_lib.direct_hardware import DirectHardware, DirectHardwareError


PANEL_CONFIG = {
    "manual_control": {
        "arm_ip": "192.168.58.2",
        "arm_port": 20003,
        "servo_port": "/dev/servo_motor",
        "servo_baudrate": 115200,
    }
}

RESET_POSE = [-250.415, 22.148, 380.0, -180.0, 0.0, 90.0]


class FakeArmSdk:
    """AkaiFr().arm 的替身，记录 MoveL / Get* 调用。"""

    def __init__(self):
        self.calls = []
        self.motion_done = True
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.pose = [0.0, 0.0, 380.0, -180.0, 0.0, 90.0]
        self.move_result = 0

    def MoveL(self, pose, tool, user, vel, blendR):
        self.calls.append(("MoveL", list(pose), tool, user, vel, blendR))
        return self.move_result

    def GetRobotMotionDone(self):
        self.calls.append(("GetRobotMotionDone",))
        return [0, int(self.motion_done)]

    def GetActualTCPCompositeSpeed(self):
        self.calls.append(("GetActualTCPCompositeSpeed",))
        return [0, [self.linear_speed, self.angular_speed]]

    def GetActualTCPPose(self):
        self.calls.append(("GetActualTCPPose",))
        return [0, list(self.pose)]


class FakeAkaiFr:
    instances = []

    def __init__(self):
        self.arm = FakeArmSdk()
        self.tcf_calls = []
        self.speed_calls = []
        self.tmat_calls = []
        FakeAkaiFr.instances.append(self)

    def set_tmat_wrist2camera(self, matrix):
        self.tmat_calls.append(matrix)

    def set_tcf(self, id_, coord):
        self.tcf_calls.append((id_, coord))

    def set_speed(self, speed):
        self.speed_calls.append(speed)

    def get_camera_pose(self):
        return (True, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0])


class FakeSucker:
    instances = []

    def __init__(self, arm):
        self.arm = arm
        self.solenoid_calls = []
        self.pump_calls = []
        FakeSucker.instances.append(self)

    def set_solenoid_valve(self, value):
        self.solenoid_calls.append(value)

    def set_pump_motor(self, value):
        self.pump_calls.append(value)


class FakeSerial:
    def __init__(self, *args, **kwargs):
        self.written = []
        self.closed = False
        self.flushed = False

    def write(self, data):
        self.written.append(data)

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


class FakeRpc:
    def __init__(self, motion_done=True, linear=0.0, angular=0.0):
        self.motion_done = motion_done
        self.linear = linear
        self.angular = angular
        self.stop_calls = []

    def StopMotion(self):
        self.stop_calls.append(True)
        return 0

    def GetRobotMotionDone(self):
        return [0, int(self.motion_done)]

    def GetActualTCPCompositeSpeed(self, flag=1):
        return [0, self.linear, self.angular]


@pytest.fixture
def hardware(monkeypatch):
    monkeypatch.setattr(dh, "AkaiFr", FakeAkaiFr)
    monkeypatch.setattr(dh, "AkaiElectricSucker", FakeSucker)
    monkeypatch.setattr(dh.np, "load", lambda path: np.eye(4))
    FakeAkaiFr.instances.clear()
    FakeSucker.instances.clear()
    return DirectHardware(PANEL_CONFIG)


def test吸盘照抄controller先阀后泵(hardware):
    direct = hardware

    direct.set_suction(0)  # 吸气：阀关 + 泵开
    sucker = FakeSucker.instances[0]
    assert sucker.solenoid_calls == [False]
    assert sucker.pump_calls == [True]

    sucker.solenoid_calls.clear()
    sucker.pump_calls.clear()
    direct.set_suction(1)  # 喷气：阀开 + 泵开
    assert sucker.solenoid_calls == [True]
    assert sucker.pump_calls == [True]

    sucker.solenoid_calls.clear()
    sucker.pump_calls.clear()
    direct.set_suction(2)  # 关闭：阀关 + 泵关
    assert sucker.solenoid_calls == [False]
    assert sucker.pump_calls == [False]


def test吸盘拒绝非法状态(hardware):
    with pytest.raises(DirectHardwareError, match="吸气、喷气或关闭"):
        hardware.set_suction(9)


def test复位照抄controller_move_arm(hardware):
    direct = hardware

    direct.move_arm(RESET_POSE, 50, wait_until_stable=False)

    arm = FakeAkaiFr.instances[0]
    sdk = arm.arm
    # 初始化照抄 controller.py __init__：set_tcf(1, 全零) + set_tmat_wrist2camera。
    assert arm.tcf_calls == [(1, [0, 0, 0, 0, 0, 0])]
    assert len(arm.tmat_calls) == 1
    # move_arm 照抄：set_speed → MoveL(位姿, tool=0, user=0, vel=speed, blendR=-1)。
    assert arm.speed_calls == [50]
    move = [call for call in sdk.calls if call[0] == "MoveL"][0]
    assert move[1] == RESET_POSE
    assert move[2:] == (0, 0, 50, -1.0)


def test复位MoveL失败时翻译错误码(hardware):
    direct = hardware
    direct._ensure_arm()
    FakeAkaiFr.instances[0].arm.move_result = 14

    with pytest.raises(DirectHardwareError, match="关节命令点错误"):
        direct.move_arm(RESET_POSE, 50, wait_until_stable=False)


def test低于统一安全高度时抬高但操作失败(hardware):
    direct = hardware

    with pytest.raises(DirectHardwareError, match="本次运动判定失败"):
        direct.move_arm(
            [-250.415, 22.148, 162.0, -180.0, 0.0, 90.0],
            50,
            wait_until_stable=False,
        )

    arm = FakeAkaiFr.instances[0]
    move = [call for call in arm.arm.calls if call[0] == "MoveL"][0]
    assert move[1][2] == pytest.approx(163.0)


def test复位前运动未停稳时拒绝(hardware):
    direct = hardware
    direct._ensure_arm()
    sdk = FakeAkaiFr.instances[0].arm
    sdk.motion_done = False
    sdk.linear_speed = 50.0

    with pytest.raises(DirectHardwareError, match="仍在运动"):
        direct.move_arm(RESET_POSE, 50, wait_until_stable=False)


def test舵机长驻串口write并flush复用(monkeypatch):
    direct = DirectHardware(PANEL_CONFIG)
    fake = FakeSerial()
    monkeypatch.setattr(dh.serial, "Serial", lambda *a, **k: fake)

    direct.rotate_tool(180)
    # 照 controller.py / 测试舵机往复转.py：长驻串口，write → flush，不在命令间关串口。
    assert fake.written == [b"180.0E"]
    assert fake.flushed is True
    assert fake.closed is False

    direct.rotate_tool(90)
    assert fake.written == [b"180.0E", b"90.0E"]
    assert fake.closed is False

    direct.release_servo()
    assert fake.closed is True


def test读位姿返回TCP与相机(hardware):
    direct = hardware
    result = direct.get_pose()
    assert result["tcp_pose"] == FakeAkaiFr.instances[0].arm.pose
    assert result["camera_pose"] == [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]


def test急停调用StopMotion并确认停稳(hardware, monkeypatch):
    direct = hardware
    fake = FakeRpc(motion_done=True, linear=0.0, angular=0.0)
    monkeypatch.setattr(direct, "_create_stop_rpc", lambda: fake)

    result = direct.stop_motion(timeout_seconds=0.02)
    assert fake.stop_calls == [True]
    assert result["success"] is True
    assert result["stop_latched"] is True


def test急停未停稳时报错(hardware, monkeypatch):
    direct = hardware
    fake = FakeRpc(motion_done=False, linear=50.0, angular=10.0)
    monkeypatch.setattr(direct, "_create_stop_rpc", lambda: fake)

    with pytest.raises(DirectHardwareError, match="未在限定时间内确认停稳"):
        direct.stop_motion(timeout_seconds=0.02)
