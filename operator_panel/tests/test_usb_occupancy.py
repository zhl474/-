import os

from operator_panel_lib.usb_occupancy import probe_usb_occupancy


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(str(text), encoding="utf-8")


def _add_usb_device(sys_root, devpath, vid, pid, busnum, devnum):
    device = sys_root / "bus" / "usb" / "devices" / devpath
    _write(device / "idVendor", vid)
    _write(device / "idProduct", pid)
    _write(device / "busnum", busnum)
    _write(device / "devnum", devnum)
    return device


def test探测相机占用进程并识别空闲舵机(tmp_path):
    sys_root = tmp_path / "sys"
    proc_root = tmp_path / "proc"
    dev_root = tmp_path / "dev"

    _add_usb_device(sys_root, "2-1", "2bc5", "0800", "2", "6")
    _add_usb_device(sys_root, "1-3", "1a86", "7523", "1", "6")

    # 相机同时拥有 /dev/bus/usb/002/006 与 /dev/video0 两个候选节点。
    usb_node = dev_root / "bus" / "usb" / "002" / "006"
    _write(usb_node, "usbfs-placeholder")
    video_node = dev_root / "video0"
    _write(video_node, "video-placeholder")

    video_class = sys_root / "class" / "video4linux" / "video0"
    video_class.mkdir(parents=True, exist_ok=True)
    resolved_target = tmp_path / "devices" / "usb2" / "2-1" / "2-1:1.0"
    resolved_target.mkdir(parents=True)
    os.symlink(resolved_target, video_class / "device")

    # 舵机 udev 别名指向 ttyUSB0。
    tty_node = dev_root / "ttyUSB0"
    _write(tty_node, "tty-placeholder")
    os.symlink(tty_node, dev_root / "servo_motor")
    tty_class = sys_root / "class" / "tty" / "ttyUSB0"
    tty_class.mkdir(parents=True, exist_ok=True)
    resolved_tty = tmp_path / "devices" / "usb1" / "1-3" / "1-3:1.0"
    resolved_tty.mkdir(parents=True)
    os.symlink(resolved_tty, tty_class / "device")

    # PID 100 打开了相机的 /dev/video0。
    pid_dir = proc_root / "100"
    fd_dir = pid_dir / "fd"
    fd_dir.mkdir(parents=True)
    os.symlink(video_node, fd_dir / "3")
    _write(
        pid_dir / "cmdline",
        b"/home/zhl/fr3env/fr3env/bin/python\x00camera_profile_probe.py\x00",
    )

    result = probe_usb_occupancy(
        sys_root=sys_root, proc_root=proc_root, dev_root=dev_root
    )

    camera = result["camera"]
    assert camera["present"] is True
    assert camera["status"] == "occupied"
    assert camera["occupants"][0]["pid"] == 100
    assert "camera_profile_probe.py" in camera["occupants"][0]["cmdline"]
    assert str(video_node) in camera["occupants"][0]["opened_nodes"]
    assert str(usb_node) in camera["nodes"]
    assert str(video_node) in camera["nodes"]

    servo = result["servo"]
    assert servo["present"] is True
    assert servo["status"] == "free"
    assert str(dev_root / "servo_motor") in servo["nodes"]
    assert str(tty_node) in servo["nodes"]


def test设备未连接到本机时状态为missing(tmp_path):
    sys_root = tmp_path / "sys"
    proc_root = tmp_path / "proc"
    dev_root = tmp_path / "dev"

    result = probe_usb_occupancy(
        sys_root=sys_root, proc_root=proc_root, dev_root=dev_root
    )

    assert result["camera"]["status"] == "missing"
    assert "未发现" in result["camera"]["message"]
    assert result["servo"]["status"] == "missing"


def test其他用户fd不可读时仍判free并标记检查受限(tmp_path):
    sys_root = tmp_path / "sys"
    proc_root = tmp_path / "proc"
    dev_root = tmp_path / "dev"

    _add_usb_device(sys_root, "1-3", "1a86", "7523", "1", "6")
    tty_node = dev_root / "ttyUSB0"
    _write(tty_node, "tty-placeholder")
    tty_class = sys_root / "class" / "tty" / "ttyUSB0"
    tty_class.mkdir(parents=True, exist_ok=True)
    resolved_tty = tmp_path / "devices" / "usb1" / "1-3" / "1-3:1.0"
    resolved_tty.mkdir(parents=True)
    os.symlink(resolved_tty, tty_class / "device")

    locked_fd = proc_root / "200" / "fd"
    locked_fd.mkdir(parents=True)
    locked_fd.chmod(0)
    try:
        result = probe_usb_occupancy(
            sys_root=sys_root, proc_root=proc_root, dev_root=dev_root
        )
    finally:
        locked_fd.chmod(0o700)

    servo = result["servo"]
    assert servo["status"] == "free"
    assert servo["inspection_limited"] is True
    assert "未纳入检查" in servo["message"]
