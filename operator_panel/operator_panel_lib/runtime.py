"""控制台启动、单实例、ROS Master 所有权和清理入口。"""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import webbrowser

from ruamel.yaml import YAML
from werkzeug.serving import make_server

from .config_manager import ConfigManager
from .constants import PANEL_CONFIG_PATH
from .coordinator import OperationCoordinator
from .event_bus import EventBus
from .process_supervisor import ProcessSupervisor
from .ros_gateway import RosGateway
from .state_store import StateStore
from .web_app import create_app


class AlreadyRunning(RuntimeError):
    """同一个用户已经启动过控制台。"""


def _state_dir():
    root = os.environ.get("XDG_STATE_HOME")
    if root:
        return Path(root) / "single-arm-tetris"
    return Path.home() / ".local" / "state" / "single-arm-tetris"


def _load_panel_config():
    yaml = YAML(typ="safe")
    with PANEL_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.load(handle)


@contextmanager
def _single_instance(lock_path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise AlreadyRunning("控制台已经运行") from None
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


class PanelRuntime:
    def __init__(self, panel_config):
        self.config = panel_config
        self.state_dir = _state_dir()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.event_bus = EventBus(capacity=3000)
        self.store = StateStore(self.state_dir / "panel.sqlite3")
        self.config_manager = ConfigManager(self.store, self.state_dir)
        self.ros = RosGateway(
            self.event_bus,
            preview_fps=self.config["preview"]["fps"],
            jpeg_quality=self.config["preview"]["jpeg_quality"],
        )
        self.supervisor = ProcessSupervisor(
            self.event_bus,
            debug_output_dir=self.config["output"]["debug_output_dir"],
            servo_csv_output_dir=self.config["output"]["servo_csv_output_dir"],
            process_stop_seconds=self.config["timeouts"]["process_stop_seconds"],
            node_provider=self.ros.node_names,
        )
        self.coordinator = OperationCoordinator(
            self.event_bus, self.supervisor, self.ros,
            self.config_manager, self.store, self.config,
            exit_callback=self.request_exit,
        )
        self.ros.state_callback = self.coordinator.on_ros_health
        self.supervisor.state_callback = self.coordinator.on_process_state
        self.app = create_app(
            self.coordinator, self.event_bus, self.config_manager,
            self.ros, self.config,
        )
        self.server = None
        self.roscore_process = None
        self._exiting = threading.Event()

    @property
    def url(self):
        return f"http://{self.config['server']['host']}:{int(self.config['server']['port'])}"

    def _start_ros_master_if_needed(self):
        import rosgraph

        if rosgraph.is_master_online():
            self.event_bus.publish("log", {
                "source": "启动器", "level": "info",
                "message": "检测到现有 ROS Master，将复用且不会在退出时结束它。",
            })
            return
        self.roscore_process = subprocess.Popen(
            ["roscore"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        def read_roscore():
            for line in iter(self.roscore_process.stdout.readline, ""):
                if line.strip():
                    self.event_bus.publish("log", {
                        "source": "roscore", "level": "info", "message": line.rstrip(),
                    })

        threading.Thread(target=read_roscore, name="roscore-output", daemon=True).start()
        deadline = time.monotonic() + float(self.config["timeouts"]["ros_master_seconds"])
        while time.monotonic() < deadline:
            if self.roscore_process.poll() is not None:
                raise RuntimeError(f"roscore 启动失败，返回码 {self.roscore_process.returncode}")
            if rosgraph.is_master_online():
                self.event_bus.publish("log", {
                    "source": "启动器", "level": "info",
                    "message": "已启动控制台自有 ROS Master。",
                })
                return
            time.sleep(0.1)
        raise TimeoutError("等待 ROS Master 启动超时")

    def request_exit(self):
        if self._exiting.is_set():
            return
        self._exiting.set()
        # 给 HTTP 响应留出发送时间，再结束 make_server 循环。
        time.sleep(0.2)
        if self.server is not None:
            self.server.shutdown()

    def cleanup(self):
        self.supervisor.stop_all_owned()
        self.ros.shutdown()
        if self.roscore_process is not None and self.roscore_process.poll() is None:
            try:
                os.killpg(os.getpgid(self.roscore_process.pid), signal.SIGTERM)
                self.roscore_process.wait(timeout=5.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(self.roscore_process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def run(self):
        try:
            self._start_ros_master_if_needed()
            self.ros.start()
            self.server = make_server(
                self.config["server"]["host"],
                int(self.config["server"]["port"]),
                self.app,
                threaded=True,
            )
            threading.Timer(0.5, lambda: webbrowser.open(self.url, new=2)).start()
            self.event_bus.publish("log", {
                "source": "启动器", "level": "info", "message": f"控制台已启动：{self.url}",
            })
            self.server.serve_forever()
        finally:
            self.cleanup()


def run_panel():
    config = _load_panel_config()
    host = str(config["server"]["host"])
    if host not in ("127.0.0.1", "localhost"):
        raise RuntimeError("V1 只允许监听 127.0.0.1，禁止局域网远程访问")
    state_dir = _state_dir()
    url = f"http://{host}:{int(config['server']['port'])}"
    try:
        with _single_instance(state_dir / "panel.lock"):
            runtime = PanelRuntime(config)

            def signal_handler(_number, _frame):
                threading.Thread(target=runtime.request_exit, daemon=True).start()

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            runtime.run()
    except AlreadyRunning:
        webbrowser.open(url, new=2)
        return 0
    return 0
