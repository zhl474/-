"""仅允许固定 roslaunch 的后台进程管理。"""

from collections import deque
from datetime import datetime
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time


ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")

LAUNCH_LOG_FILES = {
    "hardware": "hardware.launch.log",
    "runtime": "perception.launch.log",
}


def tail_lines(path, count):
    """读取文件最后 count 行；大文件也不会整个载入内存。"""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return "".join(deque(handle, maxlen=max(1, int(count))))


class LaunchLogFile:
    """把 launch 原始输出逐行追加到文件；失败只告警一次，不影响进程。"""

    def __init__(self, path, event_bus, kind):
        self.event_bus = event_bus
        self.kind = kind
        self._handle = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("a", encoding="utf-8", errors="replace")
        except Exception as exc:
            self._warn(f"无法打开 launch 日志文件 {path}：{exc}；本次运行不落盘。")

    def _warn(self, message):
        self.event_bus.publish("log", {
            "source": f"launch:{self.kind}", "level": "warning", "message": message,
        })

    def append(self, line):
        handle = self._handle
        if handle is None:
            return
        try:
            handle.write(line + "\n")
            handle.flush()
        except Exception as exc:
            self._handle = None
            try:
                handle.close()
            except Exception:
                pass
            self._warn(f"写入 launch 日志失败：{exc}；已停止落盘。")

    def finish(self, note):
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            handle.write(note + "\n")
            handle.flush()
        except Exception as exc:
            self._warn(f"写入 launch 日志结束行失败：{exc}。")
        try:
            handle.close()
        except Exception:
            pass


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
        launch_log_dir=None,
    ):
        self.event_bus = event_bus
        self.debug_output_dir = str(debug_output_dir)
        self.servo_csv_output_dir = str(servo_csv_output_dir)
        self.process_stop_seconds = max(0.1, float(process_stop_seconds))
        self.node_provider = node_provider or (lambda: set())
        self.popen_factory = popen_factory or subprocess.Popen
        self.state_callback = state_callback
        self.launch_log_dir = Path(launch_log_dir) if launch_log_dir else None
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

    def _open_launch_log(self, kind, command, pid):
        """为一次 launch 打开追加日志文件并写头部分隔行；未配置则返回 None。"""
        if self.launch_log_dir is None:
            return None
        log_file = LaunchLogFile(
            self.launch_log_dir / LAUNCH_LOG_FILES[kind], self.event_bus, kind
        )
        log_file.append(f"===== {self._now()} 启动 {' '.join(command)} (pid {pid}) =====")
        return log_file

    def _read_output(self, kind, process, log_file):
        try:
            for raw_line in iter(process.stdout.readline, ""):
                line = ANSI_PATTERN.sub("", raw_line.rstrip())
                if line:
                    if log_file is not None:
                        log_file.append(line)
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

    def _watch(self, kind, process, log_file, reader):
        return_code = process.wait()
        # 先等读线程把管道里剩余输出写完，结束行才不会插到输出前面。
        if reader is not None:
            reader.join(timeout=2.0)
        with self._lock:
            requested = bool(self._requested_stop.get(kind))
            if self._processes.get(kind) is process:
                self._processes[kind] = None
                if kind == "runtime":
                    self._modes["runtime"] = None
            self._requested_stop[kind] = False
        if log_file is not None:
            note = "控制台主动停止" if requested else "自行退出"
            log_file.finish(
                f"===== {self._now()} 进程{note}，返回码 {int(return_code)} ====="
            )
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
        log_file = self._open_launch_log(kind, command, process.pid)
        reader = threading.Thread(
            target=self._read_output, args=(kind, process, log_file),
            name=f"{kind}-launch-output", daemon=True,
        )
        reader.start()
        threading.Thread(
            target=self._watch, args=(kind, process, log_file, reader),
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
            "interaction_mode:=web",
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
