#!/home/zhl/fr3env/fr3env/bin/python
"""独立测量 MoveL 返回后，机械臂何时真正满足停稳条件。"""

import math
import time

from akai_fr import AkaiFr


# ========================= 直接修改的实机参数 =========================
# 目标工具位姿：[X, Y, Z, Rx, Ry, Rz]，单位分别为 mm 和 °。
# 默认值沿用项目现有复位测试点；运行前仍须根据现场确认其安全、可达。
TARGET_POSE = [-250.4151306152343, -100, 200, -180.0, 0.0, 90.0]
MOVE_SPEED = 50
TOOL_ID = 0
USER_ID = 0

# MoveL 返回后，最多再等待多少秒。
STABLE_TIMEOUT_SECONDS = 5.0
# 期望采样周期，以及所有条件需要连续保持的时间。
SAMPLE_INTERVAL_SECONDS = 0.005
STABLE_CONFIRM_SECONDS = 0.20

# TCP 实际速度阈值。
LINEAR_SPEED_THRESHOLD_MM_S = 1.0
ANGULAR_SPEED_THRESHOLD_DEG_S = 1.0
# TCP 到达目标点的误差阈值。
POSITION_TOLERANCE_MM = 1.0
ORIENTATION_TOLERANCE_DEG = 0.5
# =====================================================================


def angle_error_deg(actual, target):
    """计算考虑 ±180° 环绕后的最小角度误差。"""
    return abs((float(actual) - float(target) + 180.0) % 360.0 - 180.0)


def position_error_mm(actual_pose, target_pose):
    """计算 TCP 三维位置与目标点之间的欧氏距离。"""
    return math.dist(actual_pose[:3], target_pose[:3])


def orientation_error_deg(actual_pose, target_pose):
    """返回 Rx、Ry、Rz 三个分量中的最大姿态误差。"""
    return max(angle_error_deg(actual_pose[index], target_pose[index]) for index in range(3, 6))


def unpack_rpc_result(name, result, value_length=None):
    """统一校验法奥 XML-RPC 查询结果，并返回有效数据。"""
    if not isinstance(result, (tuple, list)) or len(result) < 2:
        raise RuntimeError(f"{name} 返回格式异常：{result!r}")

    error_code = result[0]
    if error_code != 0:
        raise RuntimeError(f"{name} 查询失败，错误码：{error_code}")

    value = result[1]
    if value_length is not None:
        if not isinstance(value, (tuple, list)) or len(value) != value_length:
            raise RuntimeError(f"{name} 数据长度异常：{value!r}")
        return [float(item) for item in value]
    return value


def read_robot_sample(xmlrpc_arm):
    """读取运动完成信号、TCP 实际合速度及实际位姿。"""
    motion_done = bool(
        unpack_rpc_result("GetRobotMotionDone", xmlrpc_arm.GetRobotMotionDone())
    )
    composite_speed = unpack_rpc_result(
        "GetActualTCPCompositeSpeed",
        xmlrpc_arm.GetActualTCPCompositeSpeed(),
        value_length=2,
    )
    actual_pose = unpack_rpc_result(
        "GetActualTCPPose",
        xmlrpc_arm.GetActualTCPPose(),
        value_length=6,
    )
    return motion_done, composite_speed[0], composite_speed[1], actual_pose


def format_event_time(event_time, move_return_time):
    """格式化相对测试起点以及相对 MoveL 返回的时间。"""
    if event_time is None:
        return "未观测到"
    return f"{event_time:.3f} 秒（MoveL 返回后 {event_time - move_return_time:+.3f} 秒）"


def print_timeout_diagnosis(last_sample, stable_since):
    """超时时指出尚未满足的具体条件。"""
    print("\n========== 超时诊断 ==========")
    if last_sample is None:
        print("没有取得有效采样。")
        return

    motion_done, linear_speed, angular_speed, actual_pose = last_sample
    position_error = position_error_mm(actual_pose, TARGET_POSE)
    orientation_error = orientation_error_deg(actual_pose, TARGET_POSE)
    checks = {
        "控制器运动完成": motion_done,
        f"线速度 ≤ {LINEAR_SPEED_THRESHOLD_MM_S:.3f} mm/s": (
            abs(linear_speed) <= LINEAR_SPEED_THRESHOLD_MM_S
        ),
        f"姿态速度 ≤ {ANGULAR_SPEED_THRESHOLD_DEG_S:.3f} °/s": (
            abs(angular_speed) <= ANGULAR_SPEED_THRESHOLD_DEG_S
        ),
        f"位置误差 ≤ {POSITION_TOLERANCE_MM:.3f} mm": (
            position_error <= POSITION_TOLERANCE_MM
        ),
        f"姿态误差 ≤ {ORIENTATION_TOLERANCE_DEG:.3f}°": (
            orientation_error <= ORIENTATION_TOLERANCE_DEG
        ),
    }

    print(f"最后一次运动完成信号：{motion_done}")
    print(f"最后一次 TCP 实际合速度：线速度 {linear_speed:.6f} mm/s，姿态速度 {angular_speed:.6f} °/s")
    print(f"最后一次实际 TCP 位姿：{[round(value, 6) for value in actual_pose]}")
    print(f"最后一次目标误差：位置 {position_error:.6f} mm，姿态 {orientation_error:.6f}°")
    for description, passed in checks.items():
        print(f"{'通过' if passed else '未通过'}：{description}")
    if stable_since is not None:
        print("所有瞬时条件已经满足，但连续保持时间尚未达到确认时长。")


def wait_until_stable(xmlrpc_arm, test_start_time, move_return_time):
    """持续采样，返回停稳事件和最终状态；超时则抛出异常。"""
    deadline = time.perf_counter() + STABLE_TIMEOUT_SECONDS
    next_sample_time = time.perf_counter()
    sample_times = []
    last_sample = None

    first_motion_time = None
    first_done_after_motion_time = None
    first_linear_speed_ok_time = None
    first_angular_speed_ok_time = None
    first_pose_ok_time = None
    first_all_ok_time = None
    stable_since = None

    max_linear_speed = 0.0
    max_angular_speed = 0.0

    while True:
        now = time.perf_counter()
        if now >= deadline:
            print_timeout_diagnosis(last_sample, stable_since)
            raise TimeoutError(f"MoveL 返回后等待停稳超过 {STABLE_TIMEOUT_SECONDS:.1f} 秒")

        if now < next_sample_time:
            time.sleep(next_sample_time - now)

        sample_started = time.perf_counter()
        elapsed = sample_started - test_start_time
        motion_done, linear_speed, angular_speed, actual_pose = read_robot_sample(xmlrpc_arm)
        last_sample = (motion_done, linear_speed, angular_speed, actual_pose)
        sample_times.append(elapsed)

        linear_speed = abs(linear_speed)
        angular_speed = abs(angular_speed)
        max_linear_speed = max(max_linear_speed, linear_speed)
        max_angular_speed = max(max_angular_speed, angular_speed)

        position_error = position_error_mm(actual_pose, TARGET_POSE)
        orientation_error = orientation_error_deg(actual_pose, TARGET_POSE)
        linear_speed_ok = linear_speed <= LINEAR_SPEED_THRESHOLD_MM_S
        angular_speed_ok = angular_speed <= ANGULAR_SPEED_THRESHOLD_DEG_S
        pose_ok = (
            position_error <= POSITION_TOLERANCE_MM
            and orientation_error <= ORIENTATION_TOLERANCE_DEG
        )
        all_ok = motion_done and linear_speed_ok and angular_speed_ok and pose_ok

        if not motion_done and first_motion_time is None:
            first_motion_time = elapsed
        if motion_done and first_motion_time is not None and first_done_after_motion_time is None:
            first_done_after_motion_time = elapsed
        if linear_speed_ok and first_linear_speed_ok_time is None:
            first_linear_speed_ok_time = elapsed
        if angular_speed_ok and first_angular_speed_ok_time is None:
            first_angular_speed_ok_time = elapsed
        if pose_ok and first_pose_ok_time is None:
            first_pose_ok_time = elapsed
        if all_ok:
            if first_all_ok_time is None:
                first_all_ok_time = elapsed
            if stable_since is None:
                stable_since = elapsed
        else:
            stable_since = None

        if stable_since is not None and elapsed - stable_since >= STABLE_CONFIRM_SECONDS:
            return {
                "first_motion_time": first_motion_time,
                "first_done_after_motion_time": first_done_after_motion_time,
                "first_linear_speed_ok_time": first_linear_speed_ok_time,
                "first_angular_speed_ok_time": first_angular_speed_ok_time,
                "first_pose_ok_time": first_pose_ok_time,
                "first_all_ok_time": first_all_ok_time,
                "confirmed_stable_start_time": stable_since,
                "confirmed_stable_time": elapsed,
                "sample_times": sample_times,
                "max_linear_speed": max_linear_speed,
                "max_angular_speed": max_angular_speed,
                "last_sample": last_sample,
            }

        # 按采样开始时刻推进，查询耗时过长时不额外忙等追赶。
        next_sample_time = max(
            sample_started + SAMPLE_INTERVAL_SECONDS,
            time.perf_counter(),
        )


def print_report(events, move_return_time, move_call_seconds):
    """打印完整的中文停稳时间测试报告。"""
    motion_done, linear_speed, angular_speed, actual_pose = events["last_sample"]
    position_error = position_error_mm(actual_pose, TARGET_POSE)
    orientation_error = orientation_error_deg(actual_pose, TARGET_POSE)
    sample_times = events["sample_times"]
    intervals = [later - earlier for earlier, later in zip(sample_times, sample_times[1:])]

    print("\n========== MoveL 后实际速度停稳测试报告 ==========")
    print(f"MoveL 调用耗时：{move_call_seconds:.3f} 秒")
    print(f"MoveL 返回时刻：{move_return_time:.3f} 秒")
    print(f"首次观测到运动中：{format_event_time(events['first_motion_time'], move_return_time)}")
    print(
        "运动中之后首次观测到完成："
        f"{format_event_time(events['first_done_after_motion_time'], move_return_time)}"
    )
    print(
        f"线速度首次 ≤ {LINEAR_SPEED_THRESHOLD_MM_S:.3f} mm/s："
        f"{format_event_time(events['first_linear_speed_ok_time'], move_return_time)}"
    )
    print(
        f"姿态速度首次 ≤ {ANGULAR_SPEED_THRESHOLD_DEG_S:.3f} °/s："
        f"{format_event_time(events['first_angular_speed_ok_time'], move_return_time)}"
    )
    print(f"TCP 首次进入目标容差：{format_event_time(events['first_pose_ok_time'], move_return_time)}")
    print(f"所有条件首次同时满足：{format_event_time(events['first_all_ok_time'], move_return_time)}")
    print(
        "本次最终连续稳定区间起点："
        f"{format_event_time(events['confirmed_stable_start_time'], move_return_time)}"
    )
    print(
        f"连续稳定 {STABLE_CONFIRM_SECONDS:.3f} 秒后的确认时刻："
        f"{format_event_time(events['confirmed_stable_time'], move_return_time)}"
    )
    print(
        "MoveL 返回 → 首次满足全部条件："
        f"{events['first_all_ok_time'] - move_return_time:.3f} 秒"
    )
    print(
        "MoveL 返回 → 本次最终稳定区间起点："
        f"{events['confirmed_stable_start_time'] - move_return_time:.3f} 秒"
    )
    print(
        "MoveL 返回 → 确认连续稳定："
        f"{events['confirmed_stable_time'] - move_return_time:.3f} 秒"
    )
    if intervals:
        print(
            "实际采样周期："
            f"平均 {sum(intervals) / len(intervals):.4f} 秒，"
            f"最小 {min(intervals):.4f} 秒，最大 {max(intervals):.4f} 秒"
        )
    print(
        "采样期间最大实际合速度："
        f"线速度 {events['max_linear_speed']:.6f} mm/s，"
        f"姿态速度 {events['max_angular_speed']:.6f} °/s"
    )
    print(f"确认稳定时运动完成信号：{motion_done}")
    print(f"确认稳定时实际速度：线速度 {linear_speed:.6f} mm/s，姿态速度 {angular_speed:.6f} °/s")
    print(f"确认稳定时实际 TCP 位姿：{[round(value, 6) for value in actual_pose]}")
    print(f"确认稳定时目标误差：位置 {position_error:.6f} mm，姿态 {orientation_error:.6f}°")


def move_result_succeeded(result):
    """兼容布尔返回值和法奥 XML-RPC 整数错误码。"""
    if isinstance(result, bool):
        return result
    return result == 0


def main():
    print("警告：本脚本会驱动真实机械臂运动。")
    print(f"目标 TCP 位姿：{TARGET_POSE}")
    print(f"运动速度：{MOVE_SPEED}")
    confirmation = input("确认目标点和运动路径安全后，输入 yes 开始测试：").strip().lower()
    if confirmation != "yes":
        print("已取消测试，未发送运动指令。")
        return

    arm = AkaiFr()
    xmlrpc_arm = arm.arm
    try:
        arm.set_speed(MOVE_SPEED)
        arm.set_tcf(1, [0, 0, 0, 0, 0, 0])

        initial_pose = unpack_rpc_result(
            "GetActualTCPPose",
            xmlrpc_arm.GetActualTCPPose(),
            value_length=6,
        )
        print(f"运动前实际 TCP 位姿：{[round(value, 6) for value in initial_pose]}")
        print(f"运动前到目标的位置距离：{position_error_mm(initial_pose, TARGET_POSE):.6f} mm")

        test_start = time.perf_counter()
        move_call_start = time.perf_counter()
        result = xmlrpc_arm.MoveL(
            TARGET_POSE,
            tool=TOOL_ID,
            user=USER_ID,
            vel=MOVE_SPEED,
            blendR=-1.0,
        )
        move_return_time = time.perf_counter() - test_start
        move_call_seconds = time.perf_counter() - move_call_start
        if not move_result_succeeded(result):
            raise RuntimeError(f"MoveL 执行失败，返回值：{result!r}")

        print(f"MoveL 已返回，调用耗时 {move_call_seconds:.3f} 秒；开始采集实际速度和位姿……")
        events = wait_until_stable(xmlrpc_arm, test_start, move_return_time)
        print_report(events, move_return_time, move_call_seconds)
    finally:
        close_rpc = getattr(xmlrpc_arm, "CloseRPC", None)
        if callable(close_rpc):
            close_rpc()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, TimeoutError) as error:
        print(f"测试失败：{error}")
