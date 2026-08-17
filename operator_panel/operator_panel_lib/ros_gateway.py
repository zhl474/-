"""控制台与 ROS 话题、服务之间的集中适配层。"""

from collections import deque
from copy import deepcopy
from datetime import datetime
import math
import os
import threading
import time


class RosGateway:
    """只暴露控制台需要的固定服务和低帧率 JPEG 缓存。"""

    CONTROL_SERVICES = {
        "/control/move_arm",
        "/control/get_actual_pose",
        "/control/rotate_tool",
        "/control/set_suction",
        "/control/stop_arm",
        "/control/clear_arm_stop",
        "/control/get_status",
    }
    PERCEPTION_SERVICES = {
        "/perception/prepare_task",
        "/perception/get_task_target",
        "/perception/block_offset",
        "/perception/board_offset",
        "/perception/get_operator_prompt",
        "/perception/respond_operator_prompt",
    }

    def __init__(self, event_bus, preview_fps=2.0, jpeg_quality=78, state_callback=None):
        self.event_bus = event_bus
        self.preview_interval = 1.0 / max(0.1, float(preview_fps))
        self.jpeg_quality = max(20, min(95, int(jpeg_quality)))
        self.state_callback = state_callback
        self._lock = threading.RLock()
        self._jpeg = None
        self._last_frame_monotonic = 0.0
        self._last_frame_at = ""
        self._last_encoded_monotonic = 0.0
        self._camera_frame_samples = deque(maxlen=180)
        self._nodes = set()
        self._services = set()
        self._master_online = False
        self._ros_system = self._empty_ros_system()
        self._control_status = {
            "available": False,
            "stop_latched": False,
            "motion_state_known": False,
            "motion_done": False,
            "commanded_suction_state": -1,
            "servo_target_known": False,
            "servo_target_angle_deg": 0.0,
            "message": "控制服务未连接",
        }
        self._operator_prompt = self._empty_operator_prompt()
        self._stop_event = threading.Event()
        self._subscribers = []
        self._started = False
        self._rospy = None
        # YOLO 识别预览缓存（相机调参视图用，按需启停的 1Hz 拉取循环）
        self._yolo_jpeg = None
        self._yolo_summary = ""
        self._yolo_updated_at = ""
        self._yolo_preview_enabled = False
        self._yolo_preview_started = False
        self._yolo_last_error = ""

    def start(self):
        """ROS Master 就绪后启动单一 rospy 节点。"""
        if self._started:
            return
        import cv2
        from cv_bridge import CvBridge
        import rospy
        from rosgraph_msgs.msg import Log
        from sensor_msgs.msg import Image

        self._cv2 = cv2
        self._bridge = CvBridge()
        self._rospy = rospy
        if not rospy.core.is_initialized():
            rospy.init_node("operator_panel", anonymous=False, disable_signals=True)
        self._subscribers = [
            rospy.Subscriber(
                "/camera/image_rect", Image, self._on_image,
                queue_size=1, buff_size=16 * 1024 * 1024,
            ),
            rospy.Subscriber("/rosout_agg", Log, self._on_ros_log, queue_size=500),
        ]
        self._started = True
        threading.Thread(
            target=self._health_loop, name="operator-panel-ros-health", daemon=True
        ).start()

    @staticmethod
    def _now():
        return datetime.now().astimezone().isoformat(timespec="milliseconds")

    @staticmethod
    def _empty_operator_prompt():
        return {
            "pending": False,
            "prompt_id": "",
            "prompt_type": "",
            "message": "",
            "allow_fixed_yaml": False,
            "allow_continue_dynamic": False,
            "remaining_seconds": 0.0,
        }

    @staticmethod
    def _empty_ros_system(master_online=False):
        """返回离线时也可直接展示的 ROS 系统结构。"""
        return {
            "master_online": bool(master_online),
            "distro": str(os.environ.get("ROS_DISTRO", "未知") or "未知"),
            "master_uri": str(
                os.environ.get("ROS_MASTER_URI", "http://127.0.0.1:11311")
            ),
            "updated_at": "",
            "node_count": 0,
            "topic_count": 0,
            "service_count": 0,
            "nodes": [],
            "topics": [],
            "services": [],
        }

    @classmethod
    def _build_ros_system(cls, master_online, nodes, system_state, topic_types, camera_hz):
        """把 ROS Master 原始图数据整理成稳定、只读的网页结构。"""
        if not master_online:
            return cls._empty_ros_system(False)
        publishers, subscribers, service_providers = system_state
        publisher_map = {str(name): sorted(map(str, owners)) for name, owners in publishers}
        subscriber_map = {str(name): sorted(map(str, owners)) for name, owners in subscribers}
        type_map = {str(name): str(type_name) for name, type_name in topic_types}
        topic_names = sorted(set(type_map) | set(publisher_map) | set(subscriber_map))
        topics = []
        for name in topic_names:
            topics.append({
                "name": name,
                "type": type_map.get(name, "类型未知"),
                "publishers": publisher_map.get(name, []),
                "subscribers": subscriber_map.get(name, []),
                "hz": round(float(camera_hz), 1) if name == "/camera/image_rect" else None,
            })
        services = [
            {"name": str(name), "providers": sorted(map(str, providers))}
            for name, providers in sorted(service_providers, key=lambda item: str(item[0]))
        ]
        result = cls._empty_ros_system(True)
        result.update({
            "updated_at": cls._now(),
            "node_count": len(nodes),
            "topic_count": len(topics),
            "service_count": len(services),
            "nodes": sorted(map(str, nodes)),
            "topics": topics,
            "services": services,
        })
        return result

    def _camera_hz_locked(self, now=None):
        """按最近五秒收到的 ROS 图像帧计算实际话题频率。"""
        current = time.monotonic() if now is None else float(now)
        recent = [stamp for stamp in self._camera_frame_samples if current - stamp <= 5.0]
        if len(recent) < 2 or recent[-1] <= recent[0]:
            return 0.0
        return (len(recent) - 1) / (recent[-1] - recent[0])

    def _on_image(self, message):
        now = time.monotonic()
        with self._lock:
            self._last_frame_monotonic = now
            self._last_frame_at = self._now()
            self._camera_frame_samples.append(now)
            if now - self._last_encoded_monotonic < self.preview_interval:
                return
            self._last_encoded_monotonic = now
        try:
            frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            success, encoded = self._cv2.imencode(
                ".jpg", frame,
                [int(self._cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not success:
                raise RuntimeError("OpenCV JPEG 编码失败")
            with self._lock:
                self._jpeg = encoded.tobytes()
                timestamp = self._last_frame_at
            self.event_bus.publish("image", {"image_id": "camera", "updated_at": timestamp})
        except Exception as exc:
            self.event_bus.publish("log", {
                "source": "相机预览", "level": "warning",
                "message": f"低帧率预览编码失败：{exc}",
            })

    def _on_ros_log(self, message):
        levels = {1: "debug", 2: "info", 4: "warning", 8: "error", 16: "fatal"}
        self.event_bus.publish("log", {
            "source": getattr(message, "name", "rosout"),
            "level": levels.get(int(getattr(message, "level", 2)), "info"),
            "message": str(getattr(message, "msg", "")),
            "file": str(getattr(message, "file", "")),
            "line": int(getattr(message, "line", 0)),
        })

    def _refresh_graph(self):
        import rosgraph
        import rosnode
        import rosservice

        master_online = bool(rosgraph.is_master_online())
        nodes = set(rosnode.get_node_names()) if master_online else set()
        services = set(rosservice.get_service_list()) if master_online else set()
        if master_online:
            master = rosgraph.Master("/operator_panel")
            system_state = master.getSystemState()
            topic_types = master.getTopicTypes()
            with self._lock:
                camera_hz = self._camera_hz_locked()
            ros_system = self._build_ros_system(
                True,
                nodes,
                system_state,
                topic_types,
                camera_hz,
            )
        else:
            ros_system = self._empty_ros_system(False)
        with self._lock:
            self._master_online = master_online
            self._nodes = nodes
            self._services = services
            self._ros_system = ros_system
        if "/control/get_status" in services:
            try:
                status = self.get_control_status(timeout=0.8)
            except Exception as exc:
                status = dict(self._control_status)
                status.update({"available": False, "message": str(exc)})
            with self._lock:
                self._control_status = status
        else:
            with self._lock:
                self._control_status = {
                    "available": False,
                    "stop_latched": False,
                    "motion_state_known": False,
                    "motion_done": False,
                    "commanded_suction_state": -1,
                    "servo_target_known": False,
                    "servo_target_angle_deg": 0.0,
                    "message": "控制服务未连接",
                }
        if "/perception/get_operator_prompt" in services:
            try:
                prompt = self.get_operator_prompt(timeout=0.8)
            except Exception as exc:
                self.event_bus.publish("log", {
                    "source": "人工选择", "level": "warning",
                    "message": f"读取待处理提示失败：{exc}",
                })
                prompt = self._empty_operator_prompt()
        else:
            prompt = self._empty_operator_prompt()
        with self._lock:
            self._operator_prompt = prompt

    def _health_loop(self):
        last_snapshot = None
        while not self._stop_event.is_set():
            try:
                self._refresh_graph()
                snapshot = self.health_snapshot()
                if snapshot != last_snapshot:
                    self.event_bus.publish("health", snapshot)
                    if self.state_callback is not None:
                        self.state_callback(snapshot)
                    last_snapshot = snapshot
            except Exception as exc:
                self.event_bus.publish("log", {
                    "source": "ROS 状态", "level": "warning",
                    "message": f"读取 ROS 图失败：{exc}",
                })
            self._stop_event.wait(1.0)

    def node_names(self):
        with self._lock:
            return set(self._nodes)

    def service_names(self):
        with self._lock:
            return set(self._services)

    def health_snapshot(self):
        with self._lock:
            frame_age = (
                time.monotonic() - self._last_frame_monotonic
                if self._last_frame_monotonic else None
            )
            return {
                "ros_master": self._master_online,
                "camera_node": "/camera_node" in self._nodes,
                "control_node": "/control_node" in self._nodes,
                "perception_node": "/image_process_node" in self._nodes,
                "camera_frame_fresh": frame_age is not None and frame_age <= 3.0,
                "last_frame_at": self._last_frame_at,
                "control_services_ready": self.CONTROL_SERVICES.issubset(self._services),
                "perception_services_ready": self.PERCEPTION_SERVICES.issubset(self._services),
                "ros_node_count": self._ros_system["node_count"],
                "ros_topic_count": self._ros_system["topic_count"],
                "ros_service_count": self._ros_system["service_count"],
                "camera_topic_hz": next((
                    topic["hz"] for topic in self._ros_system["topics"]
                    if topic["name"] == "/camera/image_rect"
                ), 0.0),
                "control": deepcopy(self._control_status),
                "operator_prompt": deepcopy(self._operator_prompt),
            }

    def ros_system_snapshot(self):
        """返回从 ROS Master 实时采集的节点、话题与服务图。"""
        with self._lock:
            return deepcopy(self._ros_system)

    def image_bytes(self):
        with self._lock:
            return self._jpeg, self._last_frame_at

    def get_exposure_state(self, timeout=3.0):
        """读回彩色/深度相机当前曝光与增益状态及可调范围。"""
        from camera.srv import GetExposureState, GetExposureStateRequest

        self._wait_service("/camera/get_exposure_state", timeout)
        proxy = self._rospy.ServiceProxy(
            "/camera/get_exposure_state", GetExposureState, persistent=False
        )
        response = proxy(GetExposureStateRequest())
        if not response.ok:
            raise RuntimeError(str(response.message))
        return {
            "rgb": {
                "auto_exposure": bool(response.rgb_auto_exposure),
                "exposure": int(response.rgb_exposure),
                "exposure_min": int(response.rgb_exposure_min),
                "exposure_max": int(response.rgb_exposure_max),
                "gain": int(response.rgb_gain),
                "gain_min": int(response.rgb_gain_min),
                "gain_max": int(response.rgb_gain_max),
            },
            "depth": {
                "auto_exposure": bool(response.depth_auto_exposure),
                "exposure": int(response.depth_exposure),
                "exposure_min": int(response.depth_exposure_min),
                "exposure_max": int(response.depth_exposure_max),
                "gain": int(response.depth_gain),
                "gain_min": int(response.depth_gain_min),
                "gain_max": int(response.depth_gain_max),
            },
            "message": str(response.message),
        }

    def set_exposure_param(self, sensor, key, value, timeout=3.0):
        """运行时写单个相机曝光/增益参数，流保持运行。"""
        from camera.srv import SetExposureParam, SetExposureParamRequest

        self._wait_service("/camera/set_exposure_param", timeout)
        proxy = self._rospy.ServiceProxy(
            "/camera/set_exposure_param", SetExposureParam, persistent=False
        )
        response = proxy(SetExposureParamRequest(
            sensor=str(sensor),
            key=str(key),
            value=int(value),
        ))
        return {
            "success": bool(response.ok),
            "applied": int(response.applied),
            "message": str(response.message),
        }

    def yolo_preview_snapshot(self):
        """返回最近一次 YOLO 预览的 JPEG、摘要与时间戳。"""
        with self._lock:
            return self._yolo_jpeg, self._yolo_summary, self._yolo_updated_at

    def is_yolo_preview_enabled(self):
        with self._lock:
            return self._yolo_preview_enabled

    def set_yolo_preview_enabled(self, enabled):
        """按需启停 YOLO 预览拉取（相机调参视图手动开启，离开时停止）。"""
        enabled = bool(enabled)
        with self._lock:
            self._yolo_preview_enabled = enabled
            self._yolo_last_error = ""
        if enabled:
            with self._lock:
                already_started = self._yolo_preview_started
                self._yolo_preview_started = True
            if not already_started:
                threading.Thread(
                    target=self._yolo_preview_loop,
                    name="operator-panel-yolo-preview",
                    daemon=True,
                ).start()
        return {"enabled": enabled}

    def _fetch_yolo_preview(self, timeout=1.0):
        from image_process.srv import YoloPreview, YoloPreviewRequest

        # 先看健康轮询维护的服务表：感知未启动时立即失败，绝不阻塞等超时。
        if "/perception/yolo_preview" not in self.service_names():
            raise PerceptionNotReady("感知节点未启动")
        self._wait_service("/perception/yolo_preview", timeout)
        proxy = self._rospy.ServiceProxy(
            "/perception/yolo_preview", YoloPreview, persistent=False
        )
        response = proxy(YoloPreviewRequest(conf=0.0))
        if not response.ok and "让路" in str(response.message):
            # 正式识别/任务占用 GPU，属正常让路，不算错误。
            raise PerceptionBusy(str(response.message))
        if not response.ok:
            raise RuntimeError(str(response.message))
        return bytes(response.jpeg), str(response.summary)

    def _yolo_preview_loop(self):
        """按需调感知节点跑 YOLO 并缓存标注图；感知未启动时静默退避，不刷日志。"""
        while not self._stop_event.is_set():
            backoff_sec = 1.0
            with self._lock:
                enabled = self._yolo_preview_enabled
            if enabled:
                try:
                    jpeg, summary = self._fetch_yolo_preview()
                    with self._lock:
                        self._yolo_jpeg = jpeg
                        self._yolo_summary = summary
                        self._yolo_updated_at = self._now()
                        self._yolo_last_error = ""
                    self.event_bus.publish("image", {
                        "image_id": "yolo",
                        "updated_at": self._yolo_updated_at,
                        "summary": summary,
                    })
                except PerceptionBusy:
                    pass  # 正在识别/执行，静默让路
                except PerceptionNotReady:
                    backoff_sec = 5.0  # 感知未启动，退避慢查
                except Exception as exc:
                    message = f"YOLO 预览不可用：{exc}"
                    with self._lock:
                        if self._yolo_last_error != message:
                            self._yolo_last_error = message
                            should_log = True
                        else:
                            should_log = False
                    if should_log:
                        self.event_bus.publish("log", {
                            "source": "YOLO 预览", "level": "warning",
                            "message": message,
                        })
                    backoff_sec = 5.0
            self._stop_event.wait(backoff_sec)

    def wait_hardware_ready(self, timeout, frame_after=0.0):
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            snapshot = self.health_snapshot()
            with self._lock:
                new_frame = self._last_frame_monotonic > float(frame_after)
            if snapshot["control_services_ready"] and snapshot["camera_frame_fresh"] and new_frame:
                return snapshot
            if self._stop_event.wait(0.2):
                break
        raise TimeoutError("硬件启动超时：需要控制服务全部就绪并收到一帧新相机画面")

    def wait_perception_ready(self, timeout):
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            snapshot = self.health_snapshot()
            if snapshot["perception_node"] and snapshot["perception_services_ready"]:
                return snapshot
            if self._stop_event.wait(0.2):
                break
        raise TimeoutError("感知启动超时：模型或感知服务尚未全部就绪")

    def _wait_service(self, name, timeout):
        if self._rospy is None:
            raise RuntimeError("ROS 网关尚未启动")
        self._rospy.wait_for_service(name, timeout=max(0.1, float(timeout)))

    def stop_arm(self, timeout=5.0):
        from control.srv import StopArm, StopArmRequest

        self._wait_service("/control/stop_arm", timeout)
        proxy = self._rospy.ServiceProxy("/control/stop_arm", StopArm, persistent=False)
        response = proxy(StopArmRequest())
        return {
            "success": bool(response.success),
            "stop_latched": bool(response.stop_latched),
            "message": str(response.message),
        }

    def clear_arm_stop(self, timeout=5.0):
        from control.srv import ClearArmStop, ClearArmStopRequest

        self._wait_service("/control/clear_arm_stop", timeout)
        proxy = self._rospy.ServiceProxy("/control/clear_arm_stop", ClearArmStop, persistent=False)
        response = proxy(ClearArmStopRequest())
        return {
            "success": bool(response.success),
            "stop_latched": bool(response.stop_latched),
            "message": str(response.message),
        }

    def get_control_status(self, timeout=2.0):
        from control.srv import GetControlStatus, GetControlStatusRequest

        self._wait_service("/control/get_status", timeout)
        proxy = self._rospy.ServiceProxy("/control/get_status", GetControlStatus, persistent=False)
        response = proxy(GetControlStatusRequest())
        return {
            "available": bool(response.success),
            "stop_latched": bool(response.stop_latched),
            "motion_state_known": bool(response.motion_state_known),
            "motion_done": bool(response.motion_done),
            "commanded_suction_state": int(response.commanded_suction_state),
            "servo_target_known": bool(response.servo_target_known),
            "servo_target_angle_deg": float(response.servo_target_angle_deg),
            "message": str(response.message),
        }

    def get_operator_prompt(self, timeout=2.0):
        """读取感知节点当前等待网页回答的固定选项提示。"""
        from image_process.srv import GetOperatorPrompt, GetOperatorPromptRequest

        self._wait_service("/perception/get_operator_prompt", timeout)
        proxy = self._rospy.ServiceProxy(
            "/perception/get_operator_prompt",
            GetOperatorPrompt,
            persistent=False,
        )
        response = proxy(GetOperatorPromptRequest())
        return {
            "pending": bool(response.pending),
            "prompt_id": str(response.prompt_id),
            "prompt_type": str(response.prompt_type),
            "message": str(response.message),
            "allow_fixed_yaml": bool(response.allow_fixed_yaml),
            "allow_continue_dynamic": bool(response.allow_continue_dynamic),
            "remaining_seconds": max(0.0, float(response.remaining_seconds)),
        }

    def operator_prompt_snapshot(self):
        with self._lock:
            return deepcopy(self._operator_prompt)

    def respond_operator_prompt(self, prompt_id, choice, timeout=2.0):
        """向感知节点提交经过控制台校验的人工选择。"""
        from image_process.srv import (
            RespondOperatorPrompt,
            RespondOperatorPromptRequest,
        )

        self._wait_service("/perception/respond_operator_prompt", timeout)
        proxy = self._rospy.ServiceProxy(
            "/perception/respond_operator_prompt",
            RespondOperatorPrompt,
            persistent=False,
        )
        response = proxy(RespondOperatorPromptRequest(
            prompt_id=str(prompt_id),
            choice=str(choice),
        ))
        result = {
            "success": bool(response.success),
            "code": str(response.code),
            "message": str(response.message),
        }
        if result["success"]:
            with self._lock:
                self._operator_prompt = self._empty_operator_prompt()
        return result

    def cancel_operator_prompt(self, timeout=1.0):
        """若存在网页提示，则按安全默认值停止本轮。"""
        prompt = self.operator_prompt_snapshot()
        if not prompt.get("pending"):
            return False
        response = self.respond_operator_prompt(
            prompt.get("prompt_id", ""),
            "stop",
            timeout=timeout,
        )
        return bool(response.get("success"))

    def set_suction(self, state, timeout=5.0):
        from control.srv import SetSuction, SetSuctionRequest

        value = int(state)
        if value not in (0, 1, 2):
            raise ValueError("吸盘状态只能是吸气、喷气或关闭")
        self._wait_service("/control/set_suction", timeout)
        proxy = self._rospy.ServiceProxy("/control/set_suction", SetSuction, persistent=False)
        response = proxy(SetSuctionRequest(state=value))
        if not response.success:
            raise RuntimeError(response.message)
        return {"success": True, "message": str(response.message), "state": value}

    def rotate_tool(self, angle_deg, timeout=5.0):
        from control.srv import RotateTool, RotateToolRequest

        angle = float(angle_deg)
        self._wait_service("/control/rotate_tool", timeout)
        proxy = self._rospy.ServiceProxy("/control/rotate_tool", RotateTool, persistent=False)
        response = proxy(RotateToolRequest(angle_deg=angle))
        if not response.success:
            raise RuntimeError(response.message)
        return {"success": True, "message": str(response.message), "angle_deg": angle}

    def move_arm(self, pose, speed, wait_until_stable=True, timeout=5.0):
        from control.srv import MoveArm, MoveArmRequest

        values = [float(value) for value in pose]
        if len(values) != 6 or not all(math.isfinite(value) for value in values):
            raise ValueError("机械臂位姿必须是 6 个有限数值")
        self._wait_service("/control/move_arm", timeout)
        request = MoveArmRequest(
            pose=values,
            speed=int(speed),
            wait_until_stable=bool(wait_until_stable),
            blend_enabled=False,
            blend_radius_mm=0.0,
        )
        proxy = self._rospy.ServiceProxy("/control/move_arm", MoveArm, persistent=False)
        response = proxy(request)
        if not response.success:
            raise RuntimeError(response.message)
        return {"success": True, "message": str(response.message), "pose": values}

    def get_pose(self, timeout=5.0):
        from control.srv import GetActualPose, GetActualPoseRequest

        self._wait_service("/control/get_actual_pose", timeout)
        proxy = self._rospy.ServiceProxy("/control/get_actual_pose", GetActualPose, persistent=False)
        response = proxy(GetActualPoseRequest())
        if not response.success:
            raise RuntimeError(response.message)
        return {
            "success": True,
            "tcp_pose": [float(value) for value in response.tcp_pose],
            "camera_pose": [float(value) for value in response.camera_pose],
            "message": str(response.message),
        }

    @staticmethod
    def create_robot_clients():
        from competition_lib.ros_clients import RobotClients

        return RobotClients(service_wait_timeout=10.0)

    def shutdown(self):
        self._stop_event.set()
        with self._lock:
            self._yolo_preview_enabled = False
        for subscriber in self._subscribers:
            try:
                subscriber.unregister()
            except Exception:
                pass
