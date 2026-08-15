"""只读探测 Orbbec 相机与舵机 USB 设备的占用进程。

本模块不结束任何进程，不写任何配置。它通过 /sys 识别固定 VID/PID 的
USB 设备，通过 /proc/<pid>/fd 找出哪些进程打开了这些设备对应的节点。

设计约束：
- 相机与舵机都是 USB 设备，机械臂/吸盘走网络并支持多客户端，不在这里检查。
- 相机是 Orbbec Gemini 335（2bc5:0800）。
- 舵机是 HL-340 USB 转串口（1a86:7523），udev 规则会创建 /dev/servo_motor。
"""

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import pwd


@dataclass(frozen=True)
class UsbDeviceSpec:
    """一种需要探测占用的 USB 设备。"""

    key: str
    label: str
    usb_id: str
    vid: str
    pid: str
    classes: tuple = ()
    serial_alias: str = ""


CAMERA_SPEC = UsbDeviceSpec(
    key="camera",
    label="Orbbec Gemini 335 相机",
    usb_id="2bc5:0800",
    vid="2bc5",
    pid="0800",
    classes=("video4linux", "media"),
)

SERVO_SPEC = UsbDeviceSpec(
    key="servo",
    label="HL-340 USB 转串口（舵机）",
    usb_id="1a86:7523",
    vid="1a86",
    pid="7523",
    classes=("tty",),
    serial_alias="/dev/servo_motor",
)


def _read_text(path, default=""):
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return default


def _matching_usb_devices(spec, sys_root):
    """在 /sys/bus/usb/devices 下查找 VID/PID 匹配的 USB 设备。"""
    devices = []
    base = sys_root / "bus" / "usb" / "devices"
    if not base.exists():
        return devices
    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        vid = _read_text(entry / "idVendor")
        pid = _read_text(entry / "idProduct")
        if vid.lower() != spec.vid.lower() or pid.lower() != spec.pid.lower():
            continue
        busnum = _read_text(entry / "busnum")
        devnum = _read_text(entry / "devnum")
        if not busnum or not devnum:
            continue
        devices.append({
            "busnum": busnum,
            "devnum": devnum,
            "devpath": entry.name,
            "sysfs_path": str(entry),
        })
    devices.sort(key=lambda item: (item["busnum"], item["devnum"]))
    return devices


def _candidate_nodes(spec, usb_devices, sys_root, dev_root):
    """收集设备可能被进程打开的全部 /dev 节点。"""
    nodes = set()
    for device in usb_devices:
        try:
            bus_dir = f"{int(device['busnum']):03d}"
            dev_name = f"{int(device['devnum']):03d}"
        except (KeyError, TypeError, ValueError):
            continue
        usb_node = dev_root / "bus" / "usb" / bus_dir / dev_name
        if usb_node.exists():
            nodes.add(str(usb_node))

        devpath = device["devpath"]
        for class_name in spec.classes:
            class_dir = sys_root / "class" / class_name
            if not class_dir.exists():
                continue
            for entry in class_dir.iterdir():
                if not entry.is_dir():
                    continue
                if class_name == "tty" and not (
                    entry.name.startswith("ttyUSB")
                    or entry.name.startswith("ttyACM")
                ):
                    continue
                device_link = entry / "device"
                try:
                    resolved = device_link.resolve()
                except OSError:
                    continue
                if devpath not in resolved.parts:
                    continue
                node = dev_root / entry.name
                if node.exists():
                    nodes.add(str(node))

    if spec.serial_alias:
        alias = Path(spec.serial_alias)
        if alias.is_absolute():
            try:
                alias = dev_root / alias.relative_to("/dev")
            except ValueError:
                pass
        try:
            if alias.exists():
                nodes.add(str(alias))
                resolved = alias.resolve()
                if resolved.exists():
                    nodes.add(str(resolved))
        except OSError:
            pass
    return sorted(nodes)


def _read_cmdline(proc_root, pid):
    try:
        raw = (proc_root / str(pid) / "cmdline").read_bytes()
    except OSError:
        return ""
    if not raw:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()


def _process_user(pid_dir):
    try:
        uid = pid_dir.stat().st_uid
        return pwd.getpwuid(uid).pw_name
    except (OSError, KeyError):
        return ""


def _process_started_at(pid_dir):
    """从 /proc/<pid>/stat 计算进程启动时间；解析失败返回空字符串。"""
    try:
        stat_text = (pid_dir / "stat").read_text(
            encoding="utf-8", errors="replace"
        )
        rest = stat_text.rsplit(")", 1)[1].split()
        start_ticks = int(rest[19])  # comm 后的第 20 项是原始第 22 字段
        clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        boot_text = (pid_dir.parent / "stat").read_text(
            encoding="utf-8", errors="replace"
        )
        boot_seconds = None
        for field in boot_text.split():
            if field.startswith("btime"):
                boot_seconds = int(field.split("=", 1)[1])
                break
        if boot_seconds is None or clock_ticks <= 0:
            return ""
        return datetime.fromtimestamp(
            boot_seconds + start_ticks / clock_ticks
        ).astimezone().isoformat(timespec="seconds")
    except (OSError, ValueError, IndexError):
        return ""


def _scan_proc(candidate_nodes, proc_root):
    """扫描 /proc 中打开候选设备节点的进程。"""
    occupants = []
    denied_pids = 0
    if not candidate_nodes:
        return occupants, denied_pids

    node_set = set(candidate_nodes)
    real_map = {}
    for node in candidate_nodes:
        try:
            real = os.path.realpath(node)
            if real and real not in real_map:
                real_map[real] = node
        except OSError:
            pass

    current_pid = os.getpid()
    for pid_dir in proc_root.iterdir():
        pid_name = pid_dir.name
        if not pid_name.isdigit():
            continue
        try:
            fd_entries = list((pid_dir / "fd").iterdir())
        except OSError:
            denied_pids += 1
            continue

        opened_nodes = []
        for fd_entry in fd_entries:
            try:
                target = os.readlink(str(fd_entry))
            except OSError:
                continue
            matched = target if target in node_set else None
            if matched is None and target.startswith("/dev/"):
                try:
                    real = os.path.realpath(target)
                except OSError:
                    real = ""
                matched = real_map.get(real)
            if matched is not None and matched not in opened_nodes:
                opened_nodes.append(matched)

        if not opened_nodes:
            continue
        pid = int(pid_name)
        occupants.append({
            "pid": pid,
            "cmdline": _read_cmdline(proc_root, pid),
            "user": _process_user(pid_dir),
            "started_at": _process_started_at(pid_dir),
            "opened_nodes": sorted(opened_nodes),
            "is_self": pid == current_pid,
        })

    occupants.sort(key=lambda item: item["pid"])
    return occupants, denied_pids


def _format_occupants(occupants, limit=120):
    parts = []
    for occupant in occupants:
        command = occupant.get("cmdline") or f"PID {occupant.get('pid')}"
        if len(command) > limit:
            command = command[: limit - 1] + "…"
        parts.append(f"PID {occupant.get('pid')}（{command}）")
    return "；".join(parts)


def _probe_device(spec, sys_root, proc_root, dev_root):
    usb_devices = _matching_usb_devices(spec, sys_root)
    base = {
        "key": spec.key,
        "label": spec.label,
        "usb_id": spec.usb_id,
        "present": bool(usb_devices),
        "devices": usb_devices,
        "nodes": [],
        "status": "missing",
        "occupants": [],
        "message": "",
        "inspection_limited": False,
    }
    if not usb_devices:
        base["message"] = (
            f"本机 USB 总线上未发现 {spec.label}（{spec.usb_id}）；"
            "可能被上位机独占、USB 切换器切走、设备掉线或未授权给本机"
        )
        return base

    nodes = _candidate_nodes(spec, usb_devices, sys_root, dev_root)
    base["nodes"] = nodes
    occupants, denied_pids = _scan_proc(nodes, proc_root)
    base["occupants"] = occupants
    base["inspection_limited"] = bool(denied_pids)

    if occupants:
        base["status"] = "occupied"
        base["message"] = "被 " + _format_occupants(occupants) + " 占用"
    elif not nodes:
        base["status"] = "unknown"
        base["message"] = (
            "设备已在本机 USB 总线枚举，但没有生成可检查的 /dev 节点"
            "（/dev/bus/usb、/dev/video*、/dev/media* 或串口节点）"
        )
    elif denied_pids:
        # 系统里总会有一部分 root/其他用户的 /proc/<pid>/fd 不可读。
        # 这些进程通常不是本项目脚本，不应因此把空闲设备误判为 unknown。
        base["status"] = "free"
        base["message"] = (
            "当前用户可见进程未发现占用；"
            f"另有 {denied_pids} 个进程的 fd 不可读，未纳入检查"
        )
    else:
        base["status"] = "free"
        base["message"] = "设备在线，当前未发现占用进程"
    return base


def probe_usb_occupancy(
    sys_root=Path("/sys"),
    proc_root=Path("/proc"),
    dev_root=Path("/dev"),
):
    """执行一次相机与舵机 USB 占用探测，返回网页可用的结构化结果。"""
    return {
        "checked_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "camera": _probe_device(CAMERA_SPEC, sys_root, proc_root, dev_root),
        "servo": _probe_device(SERVO_SPEC, sys_root, proc_root, dev_root),
    }
