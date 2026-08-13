"""正式任务使用的 ROS 服务客户端。"""

import time

import rospy

from control.srv import (
    ClearArmStop,
    ClearArmStopRequest,
    GetActualPose,
    GetActualPoseRequest,
    GetControlStatus,
    GetControlStatusRequest,
    MoveArm,
    MoveArmRequest,
    RotateTool,
    RotateToolRequest,
    SetSuction,
    SetSuctionRequest,
    StopArm,
    StopArmRequest,
)
from image_process.srv import (
    DetectBlockOffset,
    DetectBlockOffsetRequest,
    DetectBoardOffset,
    DetectBoardOffsetRequest,
    GetTaskTarget,
    GetTaskTargetRequest,
    PrepareTask,
    PrepareTaskRequest,
)


SERVICE_NAMES = (
    "/perception/prepare_task",
    "/perception/get_task_target",
    "/perception/block_offset",
    "/perception/board_offset",
    "/control/move_arm",
    "/control/get_actual_pose",
    "/control/rotate_tool",
    "/control/set_suction",
    "/control/stop_arm",
    "/control/clear_arm_stop",
    "/control/get_status",
)


class RobotClients:
    SUCK = 0
    BLOW = 1
    OFF = 2

    def __init__(self, service_wait_timeout=None):
        for service_name in SERVICE_NAMES:
            if service_wait_timeout is None:
                # 原终端入口保持历史行为：等待操作人员启动所需节点。
                rospy.wait_for_service(service_name)
            else:
                # 网页入口已经做过就绪检查，这里使用有限超时覆盖节点竞态退出。
                rospy.wait_for_service(
                    service_name, timeout=max(0.1, float(service_wait_timeout))
                )
        self._prepare_task = rospy.ServiceProxy("/perception/prepare_task", PrepareTask)
        self._get_task_target = rospy.ServiceProxy("/perception/get_task_target", GetTaskTarget)
        self._block_offset = rospy.ServiceProxy("/perception/block_offset", DetectBlockOffset)
        self._board_offset = rospy.ServiceProxy("/perception/board_offset", DetectBoardOffset)
        self._move_arm = rospy.ServiceProxy("/control/move_arm", MoveArm)
        self._get_actual_pose = rospy.ServiceProxy("/control/get_actual_pose", GetActualPose)
        self._rotate_tool = rospy.ServiceProxy("/control/rotate_tool", RotateTool)
        self._set_suction = rospy.ServiceProxy("/control/set_suction", SetSuction)
        self._stop_arm = rospy.ServiceProxy("/control/stop_arm", StopArm)
        self._clear_arm_stop = rospy.ServiceProxy(
            "/control/clear_arm_stop", ClearArmStop
        )
        self._get_control_status = rospy.ServiceProxy(
            "/control/get_status", GetControlStatus
        )

    @staticmethod
    def _require_success(response, action):
        if not response.success:
            raise RuntimeError(f"{action}失败: {response.message}")
        return response

    def prepare_task(self, advanced=False, place_order=()):
        """请求高位粗定位与任务规划。

        识别失败属于可恢复的现场状态，因此这里保留 success=false 响应，
        交给交互层提示用户调整后重试；只有服务通信异常才会直接抛出。
        """
        request = PrepareTaskRequest()
        request.advanced = bool(advanced)
        request.place_order = [int(value) for value in place_order]
        return self._prepare_task(request)

    def get_task_target(self, index):
        request = GetTaskTargetRequest(index=int(index))
        return self._require_success(self._get_task_target(request), f"读取任务 {index}")

    def detect_block_offset(self, category, high_angle_deg):
        request = DetectBlockOffsetRequest(category=str(category), high_angle_deg=float(high_angle_deg))
        return self._block_offset(request)

    def detect_board_offset(self, row, col):
        request = DetectBoardOffsetRequest(row=float(row), col=float(col))
        return self._board_offset(request)

    def move_arm(
        self,
        pose,
        speed,
        wait_sec=0.0,
        wait_until_stable=False,
        blend_radius_mm=None,
    ):
        blend_enabled = blend_radius_mm is not None
        request = MoveArmRequest(
            pose=[float(value) for value in pose],
            speed=int(speed),
            wait_until_stable=bool(wait_until_stable),
            blend_enabled=blend_enabled,
            blend_radius_mm=(float(blend_radius_mm) if blend_enabled else 0.0),
        )
        response = self._require_success(self._move_arm(request), "机械臂运动")
        if wait_sec > 0:
            time.sleep(wait_sec)
        return response

    def get_actual_pose(self):
        """读取控制器当前实测 TCP 与相机光心位姿，仅用于定位诊断。"""
        return self._get_actual_pose(GetActualPoseRequest())

    def rotate_tool(self, angle_deg):
        request = RotateToolRequest(angle_deg=float(angle_deg))
        return self._require_success(self._rotate_tool(request), "末端舵机旋转")

    def set_suction(self, state):
        request = SetSuctionRequest(state=int(state))
        return self._require_success(self._set_suction(request), "吸盘控制")

    def stop_arm(self):
        """立即请求控制器终止运动；失败响应交给网页显示物理急停提示。"""
        return self._stop_arm(StopArmRequest())

    def clear_arm_stop(self):
        """在人工检查现场后请求解除控制节点停止锁。"""
        return self._clear_arm_stop(ClearArmStopRequest())

    def get_control_status(self):
        """读取停止锁及最近一次吸盘、舵机命令状态。"""
        return self._get_control_status(GetControlStatusRequest())
