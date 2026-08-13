"""仅允许固定 roslaunch 的后台进程管理。"""

from datetime import datetime
import os
import re
import signal
import subprocess
import threading
import time


ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


class ProcessConflict(RuntimeError):
    """目标 ROS 节点由外部进程占用。"""


class ProcessSupervisor:
    """管理控制台自己启动的硬件和感知进程，不接收任意命令。"""

    HARDWARE_NODES = {"/camera_node", "/control_node"}
    RUNTIME_NODES = {"/image_process_node"}

    def __init__(
        self,
        event_bus,
        debug_output_dir,
        servo_csv_output_dir,
        process_stop_seconds=8.0,
        node_provider=None,
        popen_factory=None,
        state_callback=None,
    ):
        self.event_bus = event_bus
        self.debug_output_dir = str(debug_output_dir)
        self.servo_csv_output_dir = str(servo_csv_output_dir)
        self.process_stop_seconds = max(0.1, float(process_stop_seconds))
        self.node_provider = node_provider or (lambda: set())
        self.popen_factory = popen_factory or subprocess.Popen
        self.state_callback = state_callback
        self._lock = threading.RLock()
        self._processes = {"hardware": None, "runtime": None}
        self._requested_stop = {"hardware": False, "runtime": False}
        self._modes = {"runtime": None}
        self._started_at = {"hardware": "", "runtime": ""}

    @staticmethod
    def _now():
        return datetime.now().astimezone().isoformat(timespec="seconds")

    @staticmethod
    def _level(line):
        upper = str(line).upper()
        if "FATAL" in upper or "ERROR" in upper or "[ERR" in upper:
            return "error"
        if "WARN" in upper:
            return "warning"
        if "DEBUG" in upper:
            return "debug"
        return "info"

    def _emit_state(self):
        if self.state_callback is not None:
            try:
                self.state_callback(self.snapshot())
            except Exception:
                pass

    def _read_output(self, kind, process):
        try:
            for raw_line in iter(process.stdout.readline, ""):
                line = ANSI_PATTERN.sub("", raw_line.rstrip())
                if line:
                    self.event_bus.publish("log", {
                        "source": f"launch:{kind}",
                        "level": self._level(line),
                        "message": line,
                    })
                if process.poll() is not None and not raw_line:
                    break
        except Exception as exc:
            self.event_bus.publish("log", {
                "source": f"launch:{kind}", "level": "warning",
                "message": f"读取 launch 输出失败：{exc}",
            })

    def _watch(self, kind, process):
        return_code = process.wait()
        with self._lock:
            requested = bool(self._requested_stop.get(kind))
            if self._processes.get(kind) is process:
                self._processes[kind] = None
                if kind == "runtime":
                    self._modes["runtime"] = None
            self._requested_stop[kind] = False
        self.event_bus.publish("process", {
            "kind": kind,
            "running": False,
            "return_code": int(return_code),
            "unexpected": return_code != 0 and not requested,
        })
        if return_code != 0 and not requested:
            self.event_bus.publish("log", {
                "source": f"launch:{kind}", "level": "error",
                "message": f"{kind} launch 异常退出，返回码 {return_code}",
            })
        self._emit_state()

    def _start(self, kind, command, mode=None):
        with self._lock:
            current = self._processes.get(kind)
            if current is not None and current.poll() is None:
                raise ProcessConflict(f"{kind} 已由控制台启动")
            occupied = set(self.node_provider() or set())
            targets = self.HARDWARE_NODES if kind == "hardware" else self.RUNTIME_NODES
            conflicts = sorted(occupied.intersection(targets))
            if conflicts:
                raise ProcessConflict(
                    "检测到外部 ROS 节点占用：" + "、".join(conflicts)
                    + "。控制台不会结束外部节点。"
                )
            process = self.popen_factory(
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
                env=os.environ.copy(),
            )
            self._processes[kind] = process
            self._requested_stop[kind] = False
            self._started_at[kind] = self._now()
            if kind == "runtime":
                self._modes["runtime"] = mode
        threading.Thread(
            target=self._read_output, args=(kind, process),
            name=f"{kind}-launch-output", daemon=True,
        ).start()
        threading.Thread(
            target=self._watch, args=(kind, process),
            name=f"{kind}-launch-watch", daemon=True,
        ).start()
        self.event_bus.publish("process", {
            "kind": kind, "running": True, "pid": process.pid, "mode": mode,
        })
        self._emit_state()
        return process.pid

    def start_hardware(self):
        """启动唯一白名单 hardware.launch。"""
        return self._start(
            "hardware", ["roslaunch", "competition", "hardware.launch"]
        )

    def start_runtime(self, mode):
        """启动只含图像处理节点的 perception.launch。"""
        normalized = str(mode)
        if normalized not in ("formal", "calibration"):
            raise ValueError("感知模式只能是 formal 或 calibration")
        calibration = "true" if normalized == "calibration" else "false"
        command = [
            "roslaunch", "competition", "perception.launch",
            f"calibration_mode:={calibration}",
            f"debug_output_dir:={self.debug_output_dir}",
            f"servo_csv_output_dir:={self.servo_csv_output_dir}",
        ]
        return self._start("runtime", command, mode=normalized)

    def _stop(self, kind):
        with self._lock:
            process = self._processes.get(kind)
        if process is None or process.poll() is not None:
            return False
        with self._lock:
            self._requested_stop[kind] = True
        try:
            group_id = os.getpgid(process.pid)
            os.killpg(group_id, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + self.process_stop_seconds
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2.0)
        with self._lock:
            if self._processes.get(kind) is process:
                self._processes[kind] = None
                if kind == "runtime":
                    self._modes["runtime"] = None
        self.event_bus.publish("process", {
            "kind": kind, "running": False, "requested": True,
            "return_code": process.returncode,
        })
        self._emit_state()
        return True

    def stop_runtime(self):
        return self._stop("runtime")

    def stop_hardware(self):
        return self._stop("hardware")

    def stop_all_owned(self):
        """按感知、硬件顺序只清理控制台拥有的进程。"""
        self.stop_runtime()
        self.stop_hardware()

    def owns(self, kind):
        with self._lock:
            process = self._processes.get(kind)
            return process is not None and process.poll() is None

    def snapshot(self):
        nodes = set(self.node_provider() or set())
        with self._lock:
            result = {}
            for kind, target_nodes in (
                ("hardware", self.HARDWARE_NODES),
                ("runtime", self.RUNTIME_NODES),
            ):
                process = self._processes.get(kind)
                owned = process is not None and process.poll() is None
                present = bool(nodes.intersection(target_nodes))
                result[kind] = {
                    "running": bool(owned or present),
                    "owned": bool(owned),
                    "external": bool(present and not owned),
                    "pid": process.pid if owned else None,
                    "started_at": self._started_at[kind] if owned else "",
                }
            result["runtime"]["mode"] = self._modes["runtime"]
            return result
