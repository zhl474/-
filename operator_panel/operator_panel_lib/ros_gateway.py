"""控制台与 ROS 话题、服务之间的集中适配层。"""

from copy import deepcopy
from datetime import datetime
import math
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
        self._nodes = set()
        self._services = set()
        self._master_online = False
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
        self._stop_event = threading.Event()
        self._subscribers = []
        self._started = False
        self._rospy = None

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

    def _on_image(self, message):
        now = time.monotonic()
        with self._lock:
            self._last_frame_monotonic = now
            self._last_frame_at = self._now()
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
        with self._lock:
            self._master_online = master_online
            self._nodes = nodes
            self._services = services
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
                "control": deepcopy(self._control_status),
            }

    def image_bytes(self):
        with self._lock:
            return self._jpeg, self._last_frame_at

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
        for subscriber in self._subscribers:
            try:
                subscriber.unregister()
            except Exception:
                pass
