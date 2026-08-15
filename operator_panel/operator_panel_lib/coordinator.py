"""网页控制台所有可变操作的单一协调器。"""

from collections import deque
from copy import deepcopy
from datetime import datetime
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import uuid

from .constants import (
    ARUCO_ALIGN_DEFAULT_LOW_TCP_Z_MM,
    ARUCO_ALIGN_SCRIPT_PATH,
    BLOCK_CATEGORY_NAMES,
    WRITABLE_CONFIG_FILES,
)
from .usb_occupancy import probe_usb_occupancy


class OperationBusy(RuntimeError):
    """已有常规操作正在执行。"""


class OperationRejected(RuntimeError):
    """当前状态不允许所请求操作。"""


class OperationCoordinator:
    """常规写操作串行化，停止运动使用不受常规队列阻塞的独立通道。"""

    FIXED_TASK_STATES = {
        "空闲", "准备识别", "等待确认", "可以执行",
        "执行中", "完成", "失败", "已中止",
    }

    def __init__(
        self,
        event_bus,
        supervisor,
        ros_gateway,
        config_manager,
        state_store,
        panel_config,
        exit_callback=None,
        direct_hardware=None,
        usb_probe=None,
    ):
        self.event_bus = event_bus
        self.supervisor = supervisor
        self.ros = ros_gateway
        self.config_manager = config_manager
        self.store = state_store
        self.panel_config = panel_config
        self.exit_callback = exit_callback
        self.direct = direct_hardware
        self.usb_probe = usb_probe or probe_usb_occupancy
        self._lock = threading.RLock()
        self._stop_channel_lock = threading.Lock()
        self._active_operation = None
        self._operations = {}
        self._local_stop_latched = False
        self._runner = None
        self._prepare_response = None
        self._abort_token = None
        self._pause_token = None
        self._execution_operation = None
        self._timed_blow_timer = None
        self._timed_blow_generation = 0
        self._aruco_align_process = None
        self._aruco_align_abort = threading.Event()
        self._aruco_align_lines = deque(maxlen=50)
        self._servo_sweep_stop = threading.Event()
        self._servo_sweep_thread = None
        self._state = {
            "system": {
                "started_at": self._now(),
                "exiting": False,
                "local_only": True,
                "software_stop_disclaimer": "软件停止不替代物理急停。停止结果不确定时请立即使用物理急停。",
            },
            "process": {
                "hardware": {"running": False, "owned": False, "external": False},
                "runtime": {"running": False, "owned": False, "external": False, "mode": None},
            },
            "health": {
                "ros_master": False,
                "camera_node": False,
                "control_node": False,
                "perception_node": False,
                "camera_frame_fresh": False,
                "control_services_ready": False,
                "perception_services_ready": False,
                "last_frame_at": "",
            },
            "task": {
                "state": "空闲",
                "phase": "",
                "mode": None,
                "advanced": False,
                "place_order": list(range(7)),
                "recognition_valid": False,
                "confirmed": False,
                "task_count": 0,
                "block_count": 0,
                "tray_count": 0,
                "current": 0,
                "total": 0,
                "category": "",
                "target_type": "",
                "message": "",
                "error": "",
                "board": None,
            },
            "hardware": {
                "stop_latched": False,
                "stop_confirmed": False,
                "emergency_warning": "",
                "holding_block": False,
                "suction_state": -1,
                "servo_target_known": False,
                "servo_target_angle_deg": 0.0,
                "motion_state_known": False,
                "motion_done": False,
                "servo_sweep_running": False,
                "servo_sweep_cycle": 0,
                "servo_sweep_repeat": 0,
                "servo_sweep_target_angle_deg": None,
            },
            "config": {
                "pending_restart": [],
                "files": self.config_manager.list_configs(),
            },
            "interaction": self._empty_interaction(),
            "operation": {"active": None, "latest": None, "stop": None},
        }

    @staticmethod
    def _now():
        return datetime.now().astimezone().isoformat(timespec="milliseconds")

    @staticmethod
    def _empty_interaction():
        return {
            "pending": False,
            "prompt_id": "",
            "type": "",
            "title": "",
            "message": "",
            "choices": [],
            "remaining_seconds": 0,
        }

    @classmethod
    def _interaction_from_prompt(cls, prompt):
        if not isinstance(prompt, dict) or not prompt.get("pending"):
            return cls._empty_interaction()
        choices = [
            {"id": "stop", "label": "停止本轮", "tone": "danger"},
        ]
        if prompt.get("allow_fixed_yaml"):
            choices.append({
                "id": "fixed_yaml",
                "label": "回退固定 task_layout.yaml",
                "tone": "secondary",
            })
        if prompt.get("allow_continue_dynamic"):
            choices.append({
                "id": "continue_dynamic",
                "label": "忽略速度不一致并继续动态盘面",
                "tone": "warning",
                "requires_confirmation": True,
            })
        return {
            "pending": True,
            "prompt_id": str(prompt.get("prompt_id", "")),
            "type": str(prompt.get("prompt_type", "")),
            "title": "V5 动态盘面需要选择",
            "message": str(prompt.get("message", "")),
            "choices": choices,
            "remaining_seconds": max(
                0,
                int(math.ceil(float(prompt.get("remaining_seconds", 0.0)))),
            ),
        }

    def _new_operation(self, kind):
        return {
            "operation_id": uuid.uuid4().hex,
            "kind": str(kind),
            "status": "running",
            "started_at": self._now(),
            "finished_at": "",
            "result": None,
            "error": "",
        }

    def _publish_state(self):
        self.event_bus.publish("state", self.snapshot())

    def _publish_operation(self, operation):
        self.event_bus.publish("operation", deepcopy(operation))

    def snapshot(self):
        with self._lock:
            state = deepcopy(self._state)
            state["operation"]["active"] = deepcopy(self._active_operation)
            local_stop_latched = bool(self._local_stop_latched)
        try:
            process_state = self.supervisor.snapshot()
            state["process"].update(process_state)
        except Exception:
            pass
        try:
            health = self.ros.health_snapshot()
            state["health"].update({
                key: value for key, value in health.items()
                if key not in {"control", "operator_prompt"}
            })
            control = health.get("control", {})
            if control:
                state["hardware"].update({
                    # 本地刚设置的停止锁不能被一次滞后的状态轮询清掉。
                    "stop_latched": local_stop_latched or bool(control.get("stop_latched", False)),
                    "suction_state": int(control.get("commanded_suction_state", -1)),
                    "servo_target_known": bool(control.get("servo_target_known", False)),
                    "servo_target_angle_deg": float(control.get("servo_target_angle_deg", 0.0)),
                    "motion_state_known": bool(control.get("motion_state_known", False)),
                    "motion_done": bool(control.get("motion_done", False)),
                })
            prompt = health.get("operator_prompt")
            if prompt is not None:
                state["interaction"] = self._interaction_from_prompt(prompt)
        except Exception:
            pass
        state["metadata"] = {
            "block_categories": list(BLOCK_CATEGORY_NAMES),
            "suction_states": {"-1": "未知", "0": "吸气", "1": "喷气", "2": "关闭"},
        }
        return state

    def on_ros_health(self, health):
        with self._lock:
            interaction = deepcopy(self._state["interaction"])
            self._state["health"].update({
                key: value for key, value in health.items()
                if key not in {"control", "operator_prompt"}
            })
            control = health.get("control", {})
            if control:
                self._state["hardware"].update({
                    # 远端锁状态单独记录；snapshot 再与本地高优先级锁合并。
                    "stop_latched": bool(control.get("stop_latched", False)),
                    "suction_state": int(control.get("commanded_suction_state", -1)),
                    "servo_target_known": bool(control.get("servo_target_known", False)),
                    "servo_target_angle_deg": float(control.get("servo_target_angle_deg", 0.0)),
                    "motion_state_known": bool(control.get("motion_state_known", False)),
                    "motion_done": bool(control.get("motion_done", False)),
                })
            prompt = health.get("operator_prompt")
            if prompt is not None:
                interaction = self._interaction_from_prompt(prompt)
                self._state["interaction"] = interaction
                if (
                    interaction["pending"]
                    and self._state["task"]["state"] == "准备识别"
                ):
                    self._state["task"]["phase"] = "等待人工选择"
        if interaction.get("pending"):
            self.event_bus.publish("interaction", deepcopy(interaction))
        self._publish_state()

    def on_process_state(self, process_state):
        with self._lock:
            self._state["process"].update(deepcopy(process_state))
            runtime = process_state.get("runtime", {})
            if not runtime.get("running") and self._state["task"]["state"] in {"等待确认", "可以执行"}:
                self._invalidate_recognition_locked("感知进程已停止，识别结果已失效")
            if not runtime.get("running"):
                self._state["interaction"] = self._empty_interaction()
            # 节点停止运行即离开待重启名单：不再有节点持有旧参数。
            pending = set(self._state["config"]["pending_restart"])
            if not process_state.get("hardware", {}).get("running"):
                pending.discard("hardware")
            if not runtime.get("running"):
                pending.discard("perception")
            self._state["config"]["pending_restart"] = sorted(pending)
        self._publish_state()

    def _log(self, level, message, source="控制台"):
        self.event_bus.publish("log", {
            "source": source, "level": str(level), "message": str(message),
        })

    def submit(self, kind, operation, source="控制台"):
        """提交一个常规异步操作；忙时直接拒绝，不积压现场指令。"""
        with self._lock:
            if self._state["system"]["exiting"]:
                raise OperationRejected("控制台正在退出，不再接受新的操作")
            if self._active_operation is not None:
                raise OperationBusy(
                    f"正在执行 {self._active_operation['kind']}，请等待其结束"
                )
            record = self._new_operation(kind)
            self._active_operation = record
            self._operations[record["operation_id"]] = record
            self._state["operation"]["latest"] = deepcopy(record)
        self._publish_operation(record)
        self._publish_state()

        def run():
            try:
                result = operation()
                with self._lock:
                    record["status"] = "success"
                    record["result"] = result
            except Exception as exc:
                with self._lock:
                    record["status"] = "aborted" if self._is_aborting() else "error"
                    record["error"] = str(exc)
                self._log(
                    "warning" if record["status"] == "aborted" else "error",
                    f"{kind}：{exc}",
                    source=source,
                )
            finally:
                with self._lock:
                    record["finished_at"] = self._now()
                    if self._active_operation is record:
                        self._active_operation = None
                    self._state["operation"]["latest"] = deepcopy(record)
                self._publish_operation(record)
                self._publish_state()

        threading.Thread(
            target=run, name=f"operator-{kind}-{record['operation_id'][:8]}", daemon=True
        ).start()
        return {"operation_id": record["operation_id"], "status": "accepted"}

    def operation(self, operation_id):
        with self._lock:
            value = self._operations.get(str(operation_id))
            return deepcopy(value) if value else None

    def _is_aborting(self):
        token = self._abort_token
        with self._lock:
            tool_abort = self._aruco_align_abort.is_set()
            active_kind = (
                self._active_operation.get("kind")
                if self._active_operation
                else None
            )
        return bool(
            (token is not None and token.requested)
            or (tool_abort and active_kind == "ArUco 单次对准")
        )

    def _require_not_stopped(self):
        with self._lock:
            if self._local_stop_latched or self._state["hardware"]["stop_latched"]:
                raise OperationRejected("停止锁已锁定，请人工检查现场后先解除停止锁")

    def _require_hardware(self):
        health = self.ros.health_snapshot()
        if not health.get("control_services_ready") or not health.get("camera_frame_fresh"):
            raise OperationRejected("硬件未就绪，需要控制服务和新相机画面")

    def _require_task_idle(self):
        with self._lock:
            if self._state["task"]["state"] in {"准备识别", "执行中"}:
                raise OperationRejected("识别或任务执行期间不允许此操作")

    def _invalidate_recognition_locked(self, reason):
        task = self._state["task"]
        task.update({
            "recognition_valid": False,
            "confirmed": False,
            "task_count": 0,
            "block_count": 0,
            "tray_count": 0,
            "current": 0,
            "total": 0,
            "category": "",
            "target_type": "",
        })
        if task["state"] in {"等待确认", "可以执行", "完成", "失败"}:
            task["state"] = "空闲"
        task["message"] = str(reason)
        self._runner = None
        self._prepare_response = None
        task["board"] = None

    def invalidate_recognition(self, reason):
        with self._lock:
            self._invalidate_recognition_locked(reason)
        self._publish_state()

    def usb_occupancy(self):
        """返回相机与舵机 USB 占用探测结果，并标注占用者归属。"""
        occupancy = deepcopy(self.usb_probe())
        try:
            hardware_owned = bool(
                self.supervisor.snapshot().get("hardware", {}).get("owned")
            )
        except Exception:
            hardware_owned = False
        for device in occupancy.values():
            if not isinstance(device, dict):
                continue
            for occupant in device.get("occupants", []):
                if not isinstance(occupant, dict):
                    continue
                command = str(occupant.get("cmdline", ""))
                if occupant.get("is_self"):
                    owner = "self"
                elif hardware_owned and (
                    (
                        device.get("key") == "camera"
                        and "camera_node.py" in command
                    )
                    or (
                        device.get("key") == "servo"
                        and "controller.py" in command
                    )
                ):
                    owner = "panel"
                else:
                    owner = "external"
                occupant["owner"] = owner
        return occupancy

    @staticmethod
    def _format_usb_occupants(occupants):
        parts = []
        for occupant in occupants:
            command = str(occupant.get("cmdline") or "")
            if not command:
                command = f"PID {occupant.get('pid')}"
            elif len(command) > 160:
                command = command[:159] + "…"
            parts.append(f"PID {occupant.get('pid')}（{command}）")
        return "；".join(parts)

    def _preflight_usb_devices(self, occupancy):
        """启动硬件前快速拒绝明确的 USB 设备缺失或外部占用。"""
        blockers = []
        for key in ("camera", "servo"):
            device = occupancy.get(key) or {}
            label = str(device.get("label") or key)
            status = str(device.get("status") or "unknown")
            if status in {"missing", "unknown"}:
                blockers.append(
                    f"{label}：{device.get('message') or '状态未知'}"
                )
                continue
            if status != "occupied":
                continue
            external = [
                occupant
                for occupant in device.get("occupants", [])
                if occupant.get("owner") == "external"
            ]
            if external:
                blockers.append(
                    f"{label}被 {self._format_usb_occupants(external)} 占用"
                )
        if blockers:
            raise OperationRejected(
                "硬件 USB 设备检查未通过：" + "；".join(blockers)
            )

    def start_hardware(self):
        def action():
            with self._lock:
                if self._state["hardware"]["servo_sweep_running"]:
                    raise OperationRejected("舵机往复测试进行中，请先停止往复")
            with self._lock:
                # 停止时控制节点若已离线，必须允许只重建硬件服务，否则既无法
                # 调用 clear-stop，也无法把本地停止锁同步到新控制节点。
                stopped = bool(
                    self._local_stop_latched
                    or self._state["hardware"]["stop_latched"]
                )
            frame_before = getattr(self.ros, "_last_frame_monotonic", 0.0)
            # 启动 hardware.launch 前先释放直连层长驻的舵机串口，否则 control_node 打不开串口。
            if self.direct is not None:
                self.direct.release_servo()
            # 相机/舵机被外部占用时快速失败，避免等待 hardware_start_seconds 超时。
            self._preflight_usb_devices(self.usb_occupancy())
            self.supervisor.start_hardware()
            try:
                timeout = self.panel_config["timeouts"]["hardware_start_seconds"]
                result = self.ros.wait_hardware_ready(timeout, frame_after=frame_before)
            except Exception:
                self.supervisor.stop_hardware()
                raise
            if stopped:
                try:
                    response = self.ros.stop_arm()
                    confirmed = bool(
                        response.get("success")
                        and response.get("stop_latched", True)
                    )
                    message = str(response.get("message", ""))
                except Exception as exc:
                    confirmed = False
                    message = str(exc)
                warning = "" if confirmed else (
                    "硬件已重新连接，但无法确认机械臂停止锁已经同步："
                    + (message or "StopMotion 未确认成功")
                    + "。请立即使用物理急停。"
                )
                with self._lock:
                    self._state["hardware"].update({
                        "stop_latched": True,
                        "stop_confirmed": confirmed,
                        "emergency_warning": warning,
                    })
                result = dict(result)
                result.update({
                    "stop_latched": True,
                    "stop_confirmed": confirmed,
                    "stop_message": message,
                })
                if confirmed:
                    self._log("warning", "硬件重新连接后已同步停止锁；仍需人工检查并解除")
                else:
                    self._log("fatal", warning)
                    raise RuntimeError(warning)
                return result
            self._log("info", "硬件已就绪：控制服务正常并收到新相机画面")
            return result

        return self.submit("启动硬件", action)

    def stop_hardware(self):
        def action():
            self._require_task_idle()
            self._request_sweep_stop()
            hardware = self.supervisor.snapshot().get("hardware", {})
            if hardware.get("external") and not hardware.get("owned"):
                raise OperationRejected("硬件节点由外部进程启动，控制台不会结束它")
            with self._lock:
                stop_confirmed = bool(
                    self._state["hardware"]["stop_latched"]
                    and self._state["hardware"]["stop_confirmed"]
                )
            if hardware.get("owned") and not stop_confirmed:
                # 结束控制节点前必须明确没有仍由机器人控制器执行的运动。
                self._require_motion_idle()
            with self._lock:
                stop_operation = self._state["operation"].get("stop")
                if stop_operation and stop_operation.get("status") == "running":
                    raise OperationRejected("停止请求仍在确认中，暂不能停止硬件")
            self.supervisor.stop_runtime()
            self.supervisor.stop_hardware()
            with self._lock:
                self._invalidate_recognition_locked("硬件已停止")
                self._state["task"].update({"state": "空闲", "mode": None, "phase": ""})
            return {"stopped": True}

        return self.submit("停止硬件", action)

    def start_runtime(self, mode):
        normalized = str(mode or "formal")
        if normalized not in ("formal", "calibration"):
            raise OperationRejected("模式只能是正式或标定")

        def action():
            self._require_not_stopped()
            self._require_hardware()
            self._require_task_idle()
            process = self.supervisor.snapshot().get("runtime", {})
            current_mode = process.get("mode")
            if process.get("running"):
                if process.get("external"):
                    raise OperationRejected("感知节点由外部进程启动，控制台不会切换或结束它")
                if current_mode == normalized:
                    raise OperationRejected("当前感知模式已经启动")
                self.supervisor.stop_runtime()
            with self._lock:
                self._invalidate_recognition_locked("切换感知模式后必须重新识别")
            self.supervisor.start_runtime(normalized)
            try:
                timeout = self.panel_config["timeouts"]["perception_start_seconds"]
                result = self.ros.wait_perception_ready(timeout)
            except Exception:
                self.supervisor.stop_runtime()
                raise
            with self._lock:
                self._state["task"].update({
                    "state": "空闲", "mode": normalized,
                    "phase": "感知已就绪", "message": "可以开始识别", "error": "",
                })
            self._log("info", f"{'标定' if normalized == 'calibration' else '正式'}感知模式已就绪")
            return result

        return self.submit("启动感知", action)

    def stop_runtime(self):
        def action():
            self._require_task_idle()
            runtime = self.supervisor.snapshot().get("runtime", {})
            if runtime.get("external") and not runtime.get("owned"):
                raise OperationRejected("感知节点由外部进程启动，控制台不会结束它")
            self.supervisor.stop_runtime()
            with self._lock:
                self._invalidate_recognition_locked("感知已停止，识别结果失效")
                self._state["task"].update({"state": "空闲", "mode": None, "phase": ""})
            return {"stopped": True}

        return self.submit("停止感知", action)

    @staticmethod
    def _validate_place_order(advanced, place_order):
        if not bool(advanced):
            return []
        if not isinstance(place_order, list) or len(place_order) != 7:
            raise OperationRejected("进阶顺序必须正好包含 7 项")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in place_order):
            raise OperationRejected("进阶顺序必须使用整数 0～6")
        if sorted(place_order) != list(range(7)):
            raise OperationRejected("进阶顺序必须是 0～6 的无重复排列")
        return list(place_order)

    def _task_state_callback(self, payload):
        with self._lock:
            self._state["task"]["phase"] = str(payload.get("state", ""))
            self._state["hardware"]["holding_block"] = bool(payload.get("holding_block", False))
        self.event_bus.publish("progress", self.snapshot()["task"])

    def _task_progress_callback(self, payload):
        with self._lock:
            self._state["task"].update({
                "current": int(payload.get("current", 0)),
                "total": int(payload.get("total", 0)),
                "category": str(payload.get("category", "")),
                "target_type": str(payload.get("target_type", "")),
                "phase": str(payload.get("state", self._state["task"]["phase"])),
            })
        self.event_bus.publish("progress", self.snapshot()["task"])

    def _holding_callback(self, payload):
        with self._lock:
            self._state["hardware"]["holding_block"] = bool(payload.get("holding_block", False))
        self._publish_state()

    def prepare_task(self, advanced=False, place_order=None):
        place_order = self._validate_place_order(advanced, place_order or [])

        def action():
            self._require_not_stopped()
            self._require_hardware()
            process = self.supervisor.snapshot().get("runtime", {})
            mode = process.get("mode")
            if not process.get("running") or mode not in ("formal", "calibration"):
                raise OperationRejected("请先启动正式或标定感知模式")
            if not self.ros.health_snapshot().get("perception_services_ready"):
                raise OperationRejected("感知服务尚未全部就绪")
            from competition_lib.task_runner import (
                CalibrationTaskRunner,
                TaskAbortToken,
                TaskPauseToken,
                TaskRunner,
            )

            token = TaskAbortToken()
            pause_token = TaskPauseToken()
            clients = self.ros.create_robot_clients()
            runner_class = CalibrationTaskRunner if mode == "calibration" else TaskRunner
            output_dir = self.panel_config["output"]["servo_csv_output_dir"]
            runner = runner_class(
                clients=clients,
                servo_csv_output_dir=output_dir,
                experiment_session_id=f"panel-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                state_callback=self._task_state_callback,
                progress_callback=self._task_progress_callback,
                holding_callback=self._holding_callback,
                abort_token=token,
                pause_token=pause_token,
            )
            with self._lock:
                self._runner = runner
                self._abort_token = token
                self._pause_token = pause_token
                self._execution_operation = None
                self._prepare_response = None
                self._state["task"].update({
                    "state": "准备识别", "phase": "移动到拍摄位并识别",
                    "mode": mode, "advanced": bool(advanced),
                    "place_order": place_order if advanced else list(range(7)),
                    "recognition_valid": False, "confirmed": False,
                    "message": "", "error": "", "current": 0, "total": 0,
                })
            self._publish_state()
            try:
                response = runner.prepare(
                    advanced=bool(advanced and mode == "formal"),
                    place_order=place_order,
                )
            except Exception as exc:
                with self._lock:
                    if token.requested:
                        self._state["task"].update({"state": "已中止", "error": str(exc)})
                    else:
                        self._state["task"].update({"state": "失败", "error": str(exc)})
                raise
            success = bool(getattr(response, "success", False))
            message = str(getattr(response, "message", ""))
            counts = {
                "task_count": int(getattr(response, "task_count", 0)),
                "block_count": int(getattr(response, "block_count", 0)),
                "tray_count": int(getattr(response, "tray_count", 0)),
            }
            board = None
            if success:
                layout_json = str(getattr(response, "layout_json", "") or "")
                if layout_json:
                    try:
                        targets = json.loads(layout_json)
                        if isinstance(targets, list) and targets:
                            if "V5 动态盘面" in message:
                                mode_label = "V5 动态盘面"
                            elif advanced:
                                mode_label = "进阶任务布局"
                            else:
                                mode_label = "基础任务布局"
                            board = {"mode_label": mode_label, "targets": targets}
                    except (ValueError, TypeError):
                        board = None
            with self._lock:
                self._prepare_response = response if success else None
                self._state["task"]["board"] = board
                self._state["task"].update({
                    "state": "等待确认" if success else "失败",
                    "phase": "识别完成" if success else "识别失败，可调整后重试",
                    "recognition_valid": success,
                    "confirmed": False,
                    "message": message,
                    "error": "" if success else message,
                    **counts,
                    "total": counts["task_count"],
                })
            if not success:
                raise OperationRejected(message or "识别失败")
            return {"success": True, "message": message, **counts}

        return self.submit("识别与规划", action)

    def confirm_task(self):
        with self._lock:
            if self._state["system"]["exiting"]:
                raise OperationRejected("控制台正在退出，不能确认任务")
            if self._local_stop_latched or self._state["hardware"]["stop_latched"]:
                raise OperationRejected("停止锁已锁定，不能确认任务")
            if self._active_operation is not None:
                raise OperationBusy("识别仍在执行，请等待完成")
            task = self._state["task"]
            if task["state"] != "等待确认" or not task["recognition_valid"] or self._runner is None:
                raise OperationRejected("当前没有可确认的识别结果")
            task.update({"state": "可以执行", "confirmed": True, "phase": "等待开始执行"})
        self._publish_state()
        return {"confirmed": True}

    def respond_interaction(
        self,
        prompt_id,
        choice,
        confirm_speed_mismatch=False,
    ):
        """回答阻塞中的感知提示；该通道不占用常规操作队列。"""
        normalized_id = str(prompt_id or "")
        normalized_choice = str(choice or "")
        if normalized_choice not in {"stop", "fixed_yaml", "continue_dynamic"}:
            raise ValueError("人工选择只能是停止、固定盘面或继续动态盘面")
        with self._lock:
            interaction = deepcopy(self._state["interaction"])
        if not interaction.get("pending"):
            # 健康轮询可能刚好还未写入本地状态，最后读取一次网关缓存。
            try:
                interaction = self._interaction_from_prompt(
                    self.ros.operator_prompt_snapshot()
                )
            except Exception:
                interaction = self._empty_interaction()
        if not interaction.get("pending"):
            raise OperationRejected("当前没有等待回答的人工选择")
        if normalized_id != interaction.get("prompt_id"):
            raise OperationRejected("人工选择提示已过期，请刷新页面状态")
        allowed = {item["id"] for item in interaction.get("choices", [])}
        if normalized_choice not in allowed:
            raise ValueError("当前提示不允许此选项")
        if (
            normalized_choice == "continue_dynamic"
            and not bool(confirm_speed_mismatch)
        ):
            raise OperationRejected(
                "忽略速度差异继续动态盘面前必须完成二次确认"
            )
        response = self.ros.respond_operator_prompt(
            normalized_id,
            normalized_choice,
        )
        if not response.get("success"):
            if response.get("code") == "invalid_choice":
                raise ValueError(response.get("message") or "当前提示不允许此选项")
            raise OperationRejected(
                response.get("message") or "人工选择已经失效"
            )
        with self._lock:
            self._state["interaction"] = self._empty_interaction()
            if self._state["task"]["state"] == "准备识别":
                self._state["task"]["phase"] = "正在处理人工选择"
        self._log("warning", f"已提交人工选择：{normalized_choice}", source="人工选择")
        self._publish_state()
        return {
            "accepted": True,
            "choice": normalized_choice,
            "message": str(response.get("message", "")),
        }

    def discard_task(self):
        """只丢弃本轮识别数据，不发出任何硬件命令。"""
        with self._lock:
            if self._active_operation is not None:
                raise OperationBusy(
                    f"正在执行 {self._active_operation['kind']}，暂不能结束本轮"
                )
            task = self._state["task"]
            if task["state"] not in {"等待确认", "可以执行", "失败"}:
                raise OperationRejected("当前没有可以结束的识别轮次")
            self._invalidate_recognition_locked("本轮已结束，可以重新识别")
            self._abort_token = None
            self._pause_token = None
            self._execution_operation = None
            task.update({
                "state": "空闲",
                "phase": "本轮已结束",
                "error": "",
            })
        self._publish_state()
        return {"discarded": True}

    def execute_task(self):
        with self._lock:
            task = self._state["task"]
            if task["state"] != "可以执行" or not task["confirmed"]:
                raise OperationRejected("请先完成识别并确认结果")
            runner = self._runner
            response = self._prepare_response
            mode = task["mode"]
            counts = {
                "task_count": task["task_count"],
                "block_count": task["block_count"],
                "tray_count": task["tray_count"],
            }
        self._require_not_stopped()

        def action():
            started_at = self._now()
            with self._lock:
                self._state["task"].update({
                    "state": "执行中", "phase": "开始执行", "error": "",
                    "current": 0, "total": counts["task_count"],
                })
            self._publish_state()
            runner.execution_start_time = time.monotonic()
            status = "完成"
            message = ""
            try:
                if mode == "calibration":
                    runner.execute_all(
                        counts["task_count"],
                        block_count=counts["block_count"],
                        tray_count=counts["tray_count"],
                    )
                else:
                    runner.execute_all(counts["task_count"])
                with self._lock:
                    self._state["task"].update({
                        "state": "完成", "phase": "任务完成",
                        "recognition_valid": False, "confirmed": False,
                    })
                return {"completed": True, **counts}
            except Exception as exc:
                message = str(exc)
                if self._is_aborting():
                    status = "已中止"
                else:
                    status = "失败"
                with self._lock:
                    self._state["task"].update({
                        "state": status, "phase": status,
                        "error": message, "recognition_valid": False, "confirmed": False,
                    })
                raise
            finally:
                self.store.add_run_summary(
                    started_at, self._now(), str(mode), status,
                    counts["task_count"], message,
                )

        return self.submit("执行任务", action)

    def pause_task(self):
        """暂停正在执行的任务：置暂停标志，并释放操作队列让手动控制可用。"""
        with self._lock:
            if self._pause_token is None:
                raise OperationRejected("当前没有可暂停的任务")
            if self._state["task"]["state"] != "执行中":
                raise OperationRejected("当前不在执行中，无法暂停")
            self._pause_token.pause()
            self._state["task"].update({"state": "已暂停", "phase": "已暂停"})
            # 挂起“执行任务”操作，释放操作队列，手动控制可正常提交。
            self._execution_operation = self._active_operation
            self._active_operation = None
        self._publish_state()
        self._log("info", "任务已暂停：当前运动完成后停止，可进行手动控制，点击“继续”恢复", source="手动控制")
        return {"paused": True}

    def resume_task(self):
        """恢复已暂停的任务。"""
        with self._lock:
            if self._pause_token is None:
                raise OperationRejected("当前没有可恢复的任务")
            if self._state["task"]["state"] != "已暂停":
                raise OperationRejected("当前不在暂停状态，无法恢复")
            if self._active_operation is not None:
                raise OperationRejected("手动操作进行中，请等它结束后再继续")
            self._pause_token.resume()
            self._state["task"].update({"state": "执行中", "phase": "继续执行"})
            self._active_operation = self._execution_operation
            self._execution_operation = None
        self._publish_state()
        self._log("info", "任务已恢复执行", source="手动控制")
        return {"resumed": True}

    def stop_execution(self):
        """软停止：中止任务 + 停止感知（等价 Ctrl+C），不停臂、不加停止锁。"""
        with self._lock:
            if self._state["task"]["state"] not in {"执行中", "已暂停"}:
                raise OperationRejected("当前没有正在执行的任务")
            if self._abort_token is not None:
                self._abort_token.request()
            self._execution_operation = None

        def stop_runtime_async():
            try:
                runtime_stopped = self.supervisor.stop_runtime()
                if not runtime_stopped:
                    runtime = self.supervisor.snapshot().get("runtime", {})
                    if runtime.get("external"):
                        self._log(
                            "warning",
                            "感知节点由外部进程管理，停止执行不会结束它；任务令牌已中止。",
                        )
                self._log("info", "任务已中止，感知已停止；机械臂走完当前运动后停止", source="手动控制")
            except Exception as exc:
                self._log("error", f"停止执行时停感知失败：{exc}", source="手动控制")

        threading.Thread(
            target=stop_runtime_async,
            name=f"operator-stop-exec-{uuid.uuid4().hex[:8]}",
            daemon=True,
        ).start()
        return {"stopping": True}

    def emergency_stop(self):
        """高优先级停止通道：不等待常规操作锁，也不自动改变吸盘。"""
        with self._stop_channel_lock:
            record = self._new_operation("停止运动")
            with self._lock:
                existing = self._state["operation"].get("stop")
                if existing and existing.get("status") == "running":
                    raise OperationBusy("停止请求已经在执行")
                self._state["operation"]["stop"] = deepcopy(record)
                self._operations[record["operation_id"]] = record
                self._local_stop_latched = True
                self._aruco_align_abort.set()
                self._state["hardware"].update({
                    "stop_latched": True,
                    "stop_confirmed": False,
                    "emergency_warning": "正在请求软件停止，请观察设备；若运动未立即停止请使用物理急停。",
                })
                self._timed_blow_generation += 1
                if self._timed_blow_timer is not None:
                    self._timed_blow_timer.cancel()
                    self._timed_blow_timer = None
                # 本地令牌必须立即生效，不能等待 StopMotion 网络调用返回；
                # 这样任务线程不会在停止确认期间再发出吸盘等后续命令。
                if self._abort_token is not None:
                    self._abort_token.request()
            self._publish_operation(record)
            self._publish_state()

        def run_stop():
            response = None
            error = ""
            try:
                # 先结束 ArUco 等一次性工具子进程，避免停止确认期间继续发送运动命令。
                self._terminate_aruco_align_process()
                # control_node 在跑时经 ROS 服务停止；否则直连独立 XML-RPC 调 StopMotion。
                if self._control_node_running():
                    response = self.ros.stop_arm()
                elif self.direct is not None:
                    response = self.direct.stop_motion()
                else:
                    raise OperationRejected("直连硬件层未配置，无法在脱离 ROS 时停止机械臂")
                try:
                    cancel_prompt = getattr(self.ros, "cancel_operator_prompt", None)
                    if cancel_prompt is not None:
                        cancel_prompt(timeout=1.0)
                except Exception as prompt_exc:
                    self._log("warning", f"停止本轮人工选择失败：{prompt_exc}")
                runtime_stopped = self.supervisor.stop_runtime()
                if not runtime_stopped:
                    runtime = self.supervisor.snapshot().get("runtime", {})
                    if runtime.get("external"):
                        self._log(
                            "warning",
                            "感知节点由外部进程管理，停止通道不会结束它；任务令牌已中止。",
                        )
                confirmed = bool(response.get("success"))
                message = str(response.get("message", ""))
                if not response.get("stop_latched", True):
                    confirmed = False
                    message = message or "控制节点未保持停止锁"
                with self._lock:
                    self._state["hardware"].update({
                        "stop_latched": True,
                        "stop_confirmed": confirmed,
                        "emergency_warning": "" if confirmed else (
                            "无法确认机械臂已经停止：" + message + "。请立即使用物理急停。"
                        ),
                    })
                    self._state["task"].update({
                        "state": "已中止", "phase": "已中止",
                        "recognition_valid": False, "confirmed": False,
                        "message": "停止后保持吸盘当前状态，等待人工检查。",
                    })
                if not confirmed:
                    raise RuntimeError(message or "StopMotion 未确认成功")
                record["status"] = "success"
                record["result"] = response
                self._log("warning", "机械臂已确认停止，停止锁保持；吸盘状态未自动改变")
            except Exception as exc:
                error = str(exc)
                if self._abort_token is not None:
                    self._abort_token.request()
                try:
                    self.supervisor.stop_runtime()
                except Exception as stop_exc:
                    error += f"；停止感知失败：{stop_exc}"
                with self._lock:
                    self._state["hardware"].update({
                        "stop_latched": True,
                        "stop_confirmed": False,
                        "emergency_warning": "无法确认机械臂已经停止：" + error + "。请立即使用物理急停。",
                    })
                    self._state["task"].update({
                        "state": "已中止", "phase": "停止状态不确定",
                        "recognition_valid": False, "confirmed": False,
                    })
                record["status"] = "error"
                record["error"] = error
                self._log("fatal", self._state["hardware"]["emergency_warning"])
            finally:
                record["finished_at"] = self._now()
                with self._lock:
                    self._state["operation"]["stop"] = deepcopy(record)
                self._publish_operation(record)
                self._publish_state()

        threading.Thread(
            target=run_stop, name=f"operator-stop-{record['operation_id'][:8]}", daemon=True
        ).start()
        return {"operation_id": record["operation_id"], "status": "accepted"}

    def clear_stop(self):
        def action():
            if self._control_node_running():
                response = self.ros.clear_arm_stop()
            elif self.direct is not None:
                if not self.direct.is_stopped():
                    raise OperationRejected("机械臂尚未确认停稳，不能解除停止锁")
                response = {"success": True, "stop_latched": False, "message": "停止锁已解除（直连）"}
            else:
                raise OperationRejected("直连硬件层未配置，无法在脱离 ROS 时解除停止锁")
            if not response.get("success") or response.get("stop_latched"):
                raise OperationRejected(response.get("message") or "停止锁解除失败")
            with self._lock:
                self._local_stop_latched = False
                self._state["hardware"].update({
                    "stop_latched": False,
                    "stop_confirmed": False,
                    "emergency_warning": "",
                })
                if self._state["task"]["state"] == "已中止":
                    self._state["task"]["message"] = "停止锁已解除；重新启动感知并识别后才能继续。"
            self._log("warning", "停止锁已由人工确认解除")
            return response

        return self.submit("解除停止锁", action)

    def _control_node_running(self):
        """control_node 在 ROS 里存在时返回 True（此时走 ROS 服务，不直连抢硬件）。"""
        return bool(self.ros.health_snapshot().get("control_node"))

    def _manual_backend(self):
        """手动控制后端分流：control_node 在跑走 ROS，否则走直连硬件。"""
        if self._control_node_running():
            return self.ros
        if self.direct is None:
            raise OperationRejected("直连硬件层未配置，无法在脱离 ROS 时手动控制")
        return self.direct

    def _assert_manual_allowed(self):
        self._require_not_stopped()
        self._require_task_idle()
        with self._lock:
            if self._state["hardware"]["servo_sweep_running"]:
                raise OperationRejected("舵机往复测试进行中，请先停止往复")

    def control_suction(self, action, continuous=False, confirm_continuous=False):
        normalized = str(action)
        if normalized not in ("suck", "off", "blow"):
            raise OperationRejected("吸盘操作只能是吸气、关闭或喷气")
        if normalized == "blow" and bool(continuous) and not bool(confirm_continuous):
            raise OperationRejected("持续喷气需要二次确认")

        def operation():
            self._assert_manual_allowed()
            self.invalidate_recognition("手动控制后识别结果已失效")
            values = {"suck": 0, "blow": 1, "off": 2}
            result = self._manual_backend().set_suction(values[normalized])
            self._log("info", f"吸盘命令完成：{result.get('message', '')}", source="手动控制")
            with self._lock:
                self._state["hardware"]["suction_state"] = values[normalized]
                self._timed_blow_generation += 1
                generation = self._timed_blow_generation
                if self._timed_blow_timer is not None:
                    self._timed_blow_timer.cancel()
                    self._timed_blow_timer = None
            if normalized == "blow" and not continuous:
                duration = float(self.panel_config["manual_control"]["timed_blow_seconds"])

                def turn_off():
                    with self._lock:
                        if (
                            generation != self._timed_blow_generation
                            or self._local_stop_latched
                            or self._state["hardware"]["stop_latched"]
                        ):
                            return
                    try:
                        self._manual_backend().set_suction(2)
                        with self._lock:
                            self._state["hardware"]["suction_state"] = 2
                            self._timed_blow_timer = None
                        self._publish_state()
                    except Exception as exc:
                        self._log("error", f"定时喷气后自动关闭失败：{exc}", source="手动控制")

                timer = threading.Timer(duration, turn_off)
                timer.daemon = True
                with self._lock:
                    self._timed_blow_timer = timer
                timer.start()
                result["auto_off_seconds"] = duration
            return result

        return self.submit("吸盘控制", operation, source="手动控制")

    def control_servo(self, angle_deg, confirm_outside_safe=False):
        try:
            angle = float(angle_deg)
        except (TypeError, ValueError):
            raise OperationRejected("舵机角度必须是数值") from None
        if not math.isfinite(angle) or not 0.0 <= angle <= 360.0:
            raise OperationRejected("舵机角度必须位于 0～360°")
        execution = self.config_manager.get_config("execution")["data"]
        motor = execution["tool_motor"]
        outside = not float(motor["lower_margin_deg"]) <= angle <= float(motor["upper_margin_deg"])
        if outside and not bool(confirm_outside_safe):
            raise OperationRejected("目标角度超出正式安全边界，需要二次确认")

        def operation():
            self._assert_manual_allowed()
            self.invalidate_recognition("手动舵机运动后识别结果已失效")
            result = self._manual_backend().rotate_tool(angle)
            self._log("info", f"舵机命令完成：{result.get('message', '')}（目标 {angle}°）", source="手动控制")
            with self._lock:
                self._state["hardware"].update({
                    "servo_target_known": True,
                    "servo_target_angle_deg": angle,
                })
            return result

        return self.submit("舵机旋转", operation, source="手动控制")

    # ------------------------------------------------------------ 舵机往复测试
    def _sweep_should_stop(self, stop_event):
        """往复循环的统一停止判定：显式停止、停止锁或控制台退出。"""
        if stop_event.is_set():
            return True
        with self._lock:
            if self._local_stop_latched or self._state["hardware"]["stop_latched"]:
                return True
            if self._state["system"]["exiting"]:
                return True
        return False

    def _sweep_wait(self, stop_event, seconds):
        """分片等待，让“停止往复”与停止锁能在等待期间及时生效。"""
        deadline = time.monotonic() + max(0.0, float(seconds))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if self._sweep_should_stop(stop_event):
                return True
            time.sleep(min(0.05, remaining))

    def _publish_sweep_target(self, angle):
        with self._lock:
            self._state["hardware"]["servo_sweep_target_angle_deg"] = float(angle)
        self._publish_state()

    def _run_servo_sweep(self, backend, min_angle, max_angle, wait, repeat, stop_event):
        try:
            # 先转到起始位置，之后每轮都是一次完整往返（照 tools/hardware/测试舵机往复转.py）。
            if self._sweep_should_stop(stop_event):
                return
            backend.rotate_tool(min_angle)
            self._publish_sweep_target(min_angle)
            if self._sweep_wait(stop_event, wait):
                return
            completed = 0
            while repeat == 0 or completed < repeat:
                if self._sweep_should_stop(stop_event):
                    return
                backend.rotate_tool(max_angle)
                self._publish_sweep_target(max_angle)
                if self._sweep_wait(stop_event, wait):
                    return
                if self._sweep_should_stop(stop_event):
                    return
                backend.rotate_tool(min_angle)
                self._publish_sweep_target(min_angle)
                if self._sweep_wait(stop_event, wait):
                    return
                completed += 1
                with self._lock:
                    self._state["hardware"]["servo_sweep_cycle"] = completed
                self._publish_state()
        except Exception as exc:
            self._log("error", f"舵机往复测试异常结束：{exc}", source="手动控制")
        finally:
            with self._lock:
                self._state["hardware"].update({
                    "servo_sweep_running": False,
                    "servo_sweep_target_angle_deg": None,
                })
            self._publish_state()

    def _request_sweep_stop(self):
        """请求停止往复循环；不等待线程结束，线程会尽快自行退出。"""
        self._servo_sweep_stop.set()

    def servo_sweep_start(
        self, min_deg, max_deg, wait_seconds, repeat_count, confirm_outside_safe=False
    ):
        try:
            min_angle = float(min_deg)
            max_angle = float(max_deg)
            wait = float(wait_seconds)
        except (TypeError, ValueError):
            raise OperationRejected("往复参数必须是数值") from None
        try:
            repeat = int(repeat_count)
        except (TypeError, ValueError):
            raise OperationRejected("往返次数必须是整数") from None
        if not all(math.isfinite(value) for value in (min_angle, max_angle, wait)):
            raise OperationRejected("往复参数必须是有限数值")
        if not 0.0 <= min_angle <= 360.0 or not 0.0 <= max_angle <= 360.0:
            raise OperationRejected("往复角度必须位于 0～360°")
        if min_angle >= max_angle:
            raise OperationRejected("起始角必须小于终止角")
        if wait <= 0:
            raise OperationRejected("每次发送后的等待时间必须大于 0")
        if repeat < 0:
            raise OperationRejected("往返次数不能小于 0")

        execution = self.config_manager.get_config("execution")["data"]
        motor = execution["tool_motor"]
        outside = not (
            float(motor["lower_margin_deg"]) <= min_angle
            and max_angle <= float(motor["upper_margin_deg"])
        )
        if outside and not bool(confirm_outside_safe):
            raise OperationRejected("往复范围超出正式安全边界，需要二次确认")

        with self._lock:
            already_running = bool(self._state["hardware"]["servo_sweep_running"])
        if already_running:
            raise OperationRejected("往复测试已经在进行，请先停止往复")
        self._assert_manual_allowed()
        self.invalidate_recognition("舵机往复测试后识别结果已失效")

        backend = self._manual_backend()
        stop_event = threading.Event()
        with self._lock:
            self._servo_sweep_stop = stop_event
            self._state["hardware"].update({
                "servo_sweep_running": True,
                "servo_sweep_cycle": 0,
                "servo_sweep_repeat": repeat,
                "servo_sweep_target_angle_deg": min_angle,
            })
        self._publish_state()
        self._log(
            "info",
            f"舵机往复测试开始：{min_angle:g}° ↔ {max_angle:g}°，"
            f"间隔 {wait:g} 秒，次数 {'无限' if repeat == 0 else repeat}",
            source="手动控制",
        )
        self._servo_sweep_thread = threading.Thread(
            target=self._run_servo_sweep,
            args=(backend, min_angle, max_angle, wait, repeat, stop_event),
            name="operator-servo-sweep",
            daemon=True,
        )
        self._servo_sweep_thread.start()
        return {"started": True, "min_deg": min_angle, "max_deg": max_angle}

    def servo_sweep_stop(self):
        with self._lock:
            running = bool(self._state["hardware"]["servo_sweep_running"])
        if not running:
            raise OperationRejected("往复测试当前没有在运行")
        self._request_sweep_stop()
        self._log("info", "已请求停止舵机往复测试", source="手动控制")
        return {"stopping": True}

    def reset_arm(self, confirmed_pose=False):
        if not bool(confirmed_pose):
            raise OperationRejected("复位前必须确认页面显示的完整拍摄位姿")
        execution = self.config_manager.get_config("execution")["data"]
        pose = [float(value) for value in execution["shooting_pose"]]
        speed = int(self.panel_config["manual_control"]["reset_speed"])

        def operation():
            self._assert_manual_allowed()
            self.invalidate_recognition("机械臂复位后识别结果已失效")
            result = self._manual_backend().move_arm(pose, speed, wait_until_stable=True)
            self._log("info", f"机械臂复位完成：{result.get('message', '')}", source="手动控制")
            return result

        result = self.submit("机械臂复位", operation, source="手动控制")
        result["pose"] = pose
        result["speed"] = speed
        return result

    def move_arm_relative(self, dx, dy, dz, speed=None):
        """以当前 TCP 位姿为基准平移 dx/dy/dz（mm），姿态保持不变。"""
        try:
            delta = [float(dx), float(dy), float(dz)]
        except (TypeError, ValueError):
            raise OperationRejected("增量必须是数值") from None
        if not all(math.isfinite(value) for value in delta):
            raise OperationRejected("增量必须是有限数值")
        if speed is None:
            speed = int(self.panel_config["manual_control"].get("relative_move_speed", 50))
        else:
            try:
                speed = int(speed)
            except (TypeError, ValueError):
                raise OperationRejected("速度必须是数值") from None
        if speed <= 0:
            raise OperationRejected("速度必须大于 0")

        def operation():
            self._assert_manual_allowed()
            self.invalidate_recognition("手动增量移动后识别结果已失效")
            pose = self._manual_backend().get_pose()
            tcp = [float(value) for value in pose["tcp_pose"]]
            target = list(tcp)
            target[0] += delta[0]
            target[1] += delta[1]
            target[2] += delta[2]
            result = self._manual_backend().move_arm(target, speed, wait_until_stable=True)
            self._log(
                "info",
                f"增量移动完成：{result.get('message', '')}（Δ={delta}）",
                source="手动控制",
            )
            result["from_pose"] = tcp
            result["target_pose"] = target
            result["delta"] = delta
            return result

        result = self.submit("增量移动", operation, source="手动控制")
        result["delta"] = delta
        result["speed"] = speed
        return result

    def get_pose(self):
        self._assert_manual_allowed()
        return self._manual_backend().get_pose()

    def _require_aruco_align_hardware(self):
        """ArUco 单次对准不依赖感知节点，但必须有控制服务、相机帧与稳定深度服务。"""
        health = self.ros.health_snapshot()
        if not health.get("control_services_ready") or not health.get("camera_frame_fresh"):
            raise OperationRejected(
                "ArUco 对准需要控制服务就绪并持续收到新相机画面"
            )
        services = set(getattr(self.ros, "service_names", lambda: ())() or ())
        if "/camera/stable_world_points" not in services:
            raise OperationRejected(
                "ArUco 对准需要 /camera/stable_world_points 稳定深度服务"
            )

    def aruco_align(self, low_tcp_z_mm=None, confirmed=False):
        """启动一次 ArUco 高位粗定位 + 低位视觉伺服对准。"""
        if not bool(confirmed):
            raise OperationRejected("ArUco 对准会运动机械臂，必须先在页面完成安全确认")
        if low_tcp_z_mm is None:
            low_z = float(ARUCO_ALIGN_DEFAULT_LOW_TCP_Z_MM)
        else:
            try:
                low_z = float(low_tcp_z_mm)
            except (TypeError, ValueError):
                raise OperationRejected("低位 TCP Z 必须是数值") from None
        if not math.isfinite(low_z):
            raise OperationRejected("低位 TCP Z 必须是有限数值")

        execution = self.config_manager.get_config("execution")["data"]
        minimum_tcp_z_mm = float(execution["motion"]["minimum_tcp_z_mm"])
        if low_z < minimum_tcp_z_mm:
            raise OperationRejected(
                f"低位 TCP Z={low_z:g}mm 低于安全下限 {minimum_tcp_z_mm:g}mm"
            )

        def action():
            self._assert_manual_allowed()
            self._require_aruco_align_hardware()
            self.invalidate_recognition("ArUco 对准后识别结果已失效")
            result = self._execute_aruco_align(low_z)
            self._log(
                "info",
                "ArUco 单次对准完成，机械臂保持最终对准位置",
                source="手动控制",
            )
            return result

        result = self.submit("ArUco 单次对准", action, source="手动控制")
        result["low_tcp_z_mm"] = low_z
        return result

    @staticmethod
    def _parse_aruco_pose(lines, prefix):
        """从脚本固定格式输出中解析六维位姿。"""
        for line in reversed(lines):
            if prefix not in line:
                continue
            try:
                return [
                    float(part.split("=", 1)[1].strip())
                    for part in line.split(prefix, 1)[1].split(",")
                ]
            except (ValueError, IndexError):
                return None
        return None

    @staticmethod
    def _parse_aruco_video_path(lines):
        """从脚本固定格式输出中解析对准录像保存路径。"""
        prefix = "对准录像已保存："
        for line in reversed(lines):
            if prefix in line:
                path = line.split(prefix, 1)[1].strip()
                return path or None
        return None

    def _execute_aruco_align(self, low_z_mm):
        """以独立 ROS 节点子进程执行 align_aruco_once.py，并实时转发输出。"""
        if not ARUCO_ALIGN_SCRIPT_PATH.is_file():
            raise OperationRejected(
                f"ArUco 对准脚本不存在：{ARUCO_ALIGN_SCRIPT_PATH}"
            )
        command = [
            sys.executable,
            "-u",
            str(ARUCO_ALIGN_SCRIPT_PATH),
            "--yes",
            "--low-tcp-z-mm",
            f"{low_z_mm:.3f}",
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        self._require_not_stopped()
        with self._lock:
            if (
                self._aruco_align_process is not None
                and self._aruco_align_process.poll() is None
            ):
                raise OperationRejected("ArUco 对准子进程已经在运行")
            self._aruco_align_abort.clear()
            self._aruco_align_lines.clear()

        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
            env=env,
        )
        with self._lock:
            self._aruco_align_process = process

        if self._aruco_align_abort.is_set():
            self._terminate_aruco_align_process()
            raise OperationRejected("ArUco 对准在启动阶段被停止")

        def pump(stream, level):
            try:
                for raw_line in iter(stream.readline, ""):
                    line = raw_line.rstrip()
                    if not line:
                        continue
                    with self._lock:
                        self._aruco_align_lines.append(line)
                    self.event_bus.publish("log", {
                        "source": "ArUco对准",
                        "level": level,
                        "message": line,
                    })
            except Exception as exc:
                self._log(
                    "warning",
                    f"读取 ArUco 对准输出失败：{exc}",
                    source="ArUco对准",
                )

        stdout_thread = threading.Thread(
            target=pump,
            args=(process.stdout, "info"),
            name="aruco-align-stdout",
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=pump,
            args=(process.stderr, "error"),
            name="aruco-align-stderr",
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        timeout = float(
            self.panel_config.get("timeouts", {}).get("aruco_align_seconds", 300.0)
        )
        try:
            return_code = process.wait(timeout=max(1.0, timeout))
        except subprocess.TimeoutExpired:
            self._aruco_align_abort.set()
            self._terminate_aruco_align_process()
            raise RuntimeError(
                f"ArUco 单次对准超过 {timeout:g} 秒，已终止子进程"
            ) from None

        stdout_thread.join(timeout=1.0)
        stderr_thread.join(timeout=1.0)

        with self._lock:
            self._aruco_align_process = None
            tail = list(self._aruco_align_lines)

        if self._aruco_align_abort.is_set():
            raise OperationRejected("ArUco 单次对准已被停止通道结束")
        if return_code != 0:
            detail = "\n".join(tail[-8:]) or f"退出码 {return_code}"
            raise RuntimeError(f"ArUco 单次对准失败：{detail}")

        return {
            "success": True,
            "message": next(
                (
                    line
                    for line in reversed(tail)
                    if "ArUco 对准成功" in line
                ),
                "ArUco 单次对准完成",
            ),
            "actual_tcp_pose": self._parse_aruco_pose(
                tail, "最终实测 TCP："
            ),
            "final_command_pose": self._parse_aruco_pose(
                tail, "最终命令 TCP："
            ),
            "video_path": self._parse_aruco_video_path(tail),
        }

    def _terminate_aruco_align_process(self):
        """结束 ArUco 对准子进程组，防止它继续发送运动命令。"""
        with self._lock:
            process = self._aruco_align_process
        if process is None or process.poll() is not None:
            return

        self._aruco_align_abort.set()
        self._log(
            "warning",
            "正在结束 ArUco 对准子进程，不再发送新的运动命令",
            source="手动控制",
        )
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return

        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._log(
                    "error",
                    "ArUco 对准子进程未能及时退出",
                    source="手动控制",
                )

    def stop_tools(self):
        """控制台退出时统一清理一次性工具子进程。"""
        self._aruco_align_abort.set()
        self._terminate_aruco_align_process()

    def _config_operation_allowed(self):
        self._require_not_stopped()
        with self._lock:
            if self._state["system"]["exiting"]:
                raise OperationRejected("控制台正在退出，禁止修改配置")
            if self._state["task"]["state"] in {"准备识别", "执行中"}:
                raise OperationRejected("识别或任务执行期间禁止保存参数")

    def _require_motion_idle(self):
        """整文件标定部署时，运行中的硬件必须能明确确认机械臂已停稳。"""
        hardware = self.supervisor.snapshot().get("hardware", {})
        if not hardware.get("running"):
            return
        if not self.ros.health_snapshot().get("control_services_ready"):
            raise OperationRejected("硬件状态未知，不能确认空闲，禁止部署标定文件")
        status = self.ros.get_control_status()
        if not status.get("motion_state_known") or not status.get("motion_done"):
            raise OperationRejected("机械臂未明确确认停稳，禁止部署标定文件")

    def _running_scopes(self):
        """返回当前正在运行、可能持有旧参数的节点作用域。"""
        process = self.supervisor.snapshot()
        scopes = set()
        if process.get("hardware", {}).get("running"):
            scopes.add("hardware")
        if process.get("runtime", {}).get("running"):
            scopes.add("perception")
        return scopes

    def _apply_restart(self, scopes):
        scopes = set(scopes)
        process = self.supervisor.snapshot()
        hardware_info = process.get("hardware", {})
        hardware_running = hardware_info.get("running", False)
        runtime_info = process.get("runtime", {})
        runtime_running = runtime_info.get("running", False)
        runtime_mode = runtime_info.get("mode")
        if "hardware" in scopes:
            if hardware_info.get("external") or runtime_info.get("external"):
                raise OperationRejected(
                    "硬件或感知节点由外部进程启动，已保存但控制台不会重启外部节点"
                )
            if runtime_running:
                self.supervisor.stop_runtime()
            if hardware_running:
                self.supervisor.stop_hardware()
                frame_before = getattr(self.ros, "_last_frame_monotonic", 0.0)
                self.supervisor.start_hardware()
                self.ros.wait_hardware_ready(
                    self.panel_config["timeouts"]["hardware_start_seconds"],
                    frame_after=frame_before,
                )
            if runtime_running and runtime_mode:
                self.supervisor.start_runtime(runtime_mode)
                self.ros.wait_perception_ready(
                    self.panel_config["timeouts"]["perception_start_seconds"]
                )
        elif "perception" in scopes:
            if runtime_info.get("external"):
                raise OperationRejected(
                    "感知节点由外部进程启动，已保存但控制台不会重启外部节点"
                )
            if runtime_running:
                self.supervisor.stop_runtime()
                self.supervisor.start_runtime(runtime_mode)
                self.ros.wait_perception_ready(
                    self.panel_config["timeouts"]["perception_start_seconds"]
                )

    def save_config(self, file_id, payload):
        # 结构、revision 与危险确认在 HTTP 返回前先校验一次，冲突可立即以
        # 4xx 告知页面；异步落盘时仍会再次校验以覆盖校验后的竞态修改。
        self._config_operation_allowed()
        self.config_manager.preflight_save(
            file_id,
            payload.get("data"),
            payload.get("revision"),
            confirm_dangerous=payload.get("confirm_dangerous", False),
        )

        def operation():
            self._config_operation_allowed()
            result = self.config_manager.save_config(
                file_id,
                payload.get("data"),
                payload.get("revision"),
                confirm_dangerous=payload.get("confirm_dangerous", False),
            )
            scope = result["restart_scope"]
            if result.get("changed"):
                running = self._running_scopes()
                with self._lock:
                    pending = set(self._state["config"]["pending_restart"])
                    if scope in running:
                        pending.add(scope)
                    self._state["config"]["pending_restart"] = sorted(pending)
                    self._state["config"]["files"] = self.config_manager.list_configs()
                    self._invalidate_recognition_locked("配置修改后识别结果已失效")
            if payload.get("apply"):
                try:
                    self._apply_restart({scope})
                except Exception as exc:
                    # 文件已成功原子保存，应用失败不能误报为“未保存”。
                    result["applied"] = False
                    result["apply_error"] = str(exc)
                    self._log("warning", f"配置已保存，但自动重启失败：{exc}")
                else:
                    with self._lock:
                        pending = set(self._state["config"]["pending_restart"])
                        pending.discard(scope)
                        self._state["config"]["pending_restart"] = sorted(pending)
                    result["applied"] = True
            else:
                result["applied"] = False
            return result

        return self.submit("保存配置", operation)

    def restore_history(self, payload):
        def operation():
            self._config_operation_allowed()
            result = self.config_manager.restore_history(
                payload.get("history_id"), payload.get("revision"),
                confirm_dangerous=payload.get("confirm_dangerous", False),
            )
            scope = result["restart_scope"]
            running = self._running_scopes()
            with self._lock:
                pending = set(self._state["config"]["pending_restart"])
                if scope in running:
                    pending.add(scope)
                self._state["config"]["pending_restart"] = sorted(pending)
                self._state["config"]["files"] = self.config_manager.list_configs()
                self._invalidate_recognition_locked("恢复配置后识别结果已失效")
            if payload.get("apply"):
                try:
                    self._apply_restart({scope})
                except Exception as exc:
                    result["applied"] = False
                    result["apply_error"] = str(exc)
                    self._log("warning", f"历史已恢复，但自动重启失败：{exc}")
                else:
                    with self._lock:
                        self._state["config"]["pending_restart"] = [
                            item for item in self._state["config"]["pending_restart"] if item != scope
                        ]
                    result["applied"] = True
            return result

        return self.submit("恢复配置历史", operation)

    def save_preset(self, name, overwrite=False):
        def operation():
            self._config_operation_allowed()
            return self.config_manager.save_preset(name, overwrite=overwrite)

        return self.submit("保存配置预设", operation)

    def delete_preset(self, preset_id):
        try:
            normalized_id = int(preset_id)
        except (TypeError, ValueError):
            raise OperationRejected("预设 ID 必须是整数") from None

        def operation():
            self._config_operation_allowed()
            try:
                deleted = self.store.delete_preset(normalized_id)
            except PermissionError as exc:
                raise OperationRejected(str(exc)) from None
            if not deleted:
                raise OperationRejected("预设不存在")
            return {"deleted": True, "preset_id": normalized_id}

        return self.submit("删除配置预设", operation)

    def restore_preset(self, payload):
        def operation():
            self._config_operation_allowed()
            result = self.config_manager.restore_preset(
                payload.get("preset_id"), payload.get("revisions", {}),
                confirm_dangerous=payload.get("confirm_dangerous", False),
            )
            scopes = set(result["restart_scopes"])
            running = self._running_scopes()
            with self._lock:
                pending = set(self._state["config"]["pending_restart"])
                pending.update(scopes & running)
                self._state["config"]["pending_restart"] = sorted(pending)
                self._state["config"]["files"] = self.config_manager.list_configs()
                self._invalidate_recognition_locked("恢复预设后识别结果已失效")
            if payload.get("apply"):
                try:
                    self._apply_restart(scopes)
                except Exception as exc:
                    result["applied"] = False
                    result["apply_error"] = str(exc)
                    self._log("warning", f"预设已恢复，但自动重启失败：{exc}")
                else:
                    with self._lock:
                        self._state["config"]["pending_restart"] = [
                            item for item in self._state["config"]["pending_restart"] if item not in scopes
                        ]
                    result["applied"] = True
            return result

        return self.submit("恢复配置预设", operation)

    def deploy_calibration_pair(self, block_text, tray_text, confirmed=False):
        if not bool(confirmed):
            raise OperationRejected("部署整套标定前必须二次确认")

        def operation():
            self._config_operation_allowed()
            self._require_task_idle()
            self._require_motion_idle()
            result = self.config_manager.deploy_calibration_pair(block_text, tray_text)
            running = self._running_scopes()
            with self._lock:
                pending = set(self._state["config"]["pending_restart"])
                if "perception" in running:
                    pending.add("perception")
                self._state["config"]["pending_restart"] = sorted(pending)
                self._invalidate_recognition_locked("标定替换后识别结果已失效")
            return result

        return self.submit("部署像素标定", operation)

    def deploy_hand_eye(self, matrix, confirmed=False):
        if not bool(confirmed):
            raise OperationRejected("部署手眼矩阵前必须二次确认")

        def operation():
            self._config_operation_allowed()
            self._require_task_idle()
            self._require_motion_idle()
            result = self.config_manager.deploy_hand_eye(matrix)
            running = self._running_scopes()
            with self._lock:
                pending = set(self._state["config"]["pending_restart"])
                if "hardware" in running:
                    pending.add("hardware")
                self._state["config"]["pending_restart"] = sorted(pending)
                self._invalidate_recognition_locked("手眼矩阵替换后识别结果已失效")
            return result

        return self.submit("部署手眼矩阵", operation)

    def request_exit(self):
        with self._lock:
            if self._state["system"]["exiting"]:
                raise OperationRejected("控制台已经在退出")
            if self._active_operation is not None:
                raise OperationRejected(
                    f"正在执行 {self._active_operation['kind']}，结束后才能退出控制台"
                )
            stop_operation = self._state["operation"].get("stop")
            if stop_operation and stop_operation.get("status") == "running":
                raise OperationRejected("停止请求仍在确认中，暂不能退出控制台")
            if self._state["task"]["state"] in {"准备识别", "执行中"}:
                raise OperationRejected("任务正在运动，请先使用停止运动并确认现场")
            stop_confirmed = bool(
                self._state["hardware"]["stop_latched"]
                and self._state["hardware"]["stop_confirmed"]
            )
            # 先封住常规操作入口，避免停稳检查和真正退出之间又开始新运动。
            self._state["system"]["exiting"] = True
        try:
            hardware = self.supervisor.snapshot().get("hardware", {})
            if hardware.get("owned") and not stop_confirmed:
                # 退出会结束自有控制节点，因此必须先确认当前没有尚在执行的运动。
                self._require_motion_idle()
            with self._lock:
                stop_operation = self._state["operation"].get("stop")
                if stop_operation and stop_operation.get("status") == "running":
                    raise OperationRejected("停止请求仍在确认中，暂不能退出控制台")
        except Exception:
            with self._lock:
                self._state["system"]["exiting"] = False
            raise
        self._publish_state()
        if self.exit_callback is not None:
            threading.Thread(target=self.exit_callback, name="operator-exit", daemon=True).start()
        return {"exiting": True}
