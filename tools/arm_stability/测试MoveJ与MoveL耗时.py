#!/home/zhl/fr3env/fr3env/bin/python
"""在相同 TCP 起终点和关节构型下，单次测试 MoveJ 或 MoveL 的耗时。"""

import math
import time

from akai_fr import AkaiFr


# ========================= 直接修改的实机参数 =========================
# 每次只测试一种运动，可填写 "MoveJ" 或 "MoveL"。
MOTION_TYPE = "MoveJ"

# 固定测试起点 A 和终点 B：[X, Y, Z, Rx, Ry, Rz]，单位分别为 mm 和 °。
POINT_A = [-337.4151306152343, -100.0, 330.0, -180.0, 0.0, 90.0]
POINT_B = [-337.4151306152343, 200.0, 330.0, -180.0, 0.0, 90.0]

# 首次必须用 20% 验证 MoveJ 完整轨迹，确认安全后再改成 100% 做正式对比。
MOVE_SPEED = 100
# 每次测试前使用 MoveL 回到 A 点，该动作不计入测试时间。
PREPOSITION_SPEED = 50
# 全局加速度百分比；当前 SDK 的合法范围为 1～100。
ACCELERATION_SCALE = 100

TOOL_ID = 0
USER_ID = 0
# 项目正式运动允许的最低 TCP 绝对高度，单位 mm。
MINIMUM_TCP_Z_MM = 165.0

# MoveJ/MoveL 阻塞调用返回后，最多等待多少秒确认真正停稳。
STABLE_TIMEOUT_SECONDS = 5.0
SAMPLE_INTERVAL_SECONDS = 0.005
STABLE_CONFIRM_SECONDS = 0.20

# TCP 停稳和到位判定阈值。
LINEAR_SPEED_THRESHOLD_MM_S = 3.0
ANGULAR_SPEED_THRESHOLD_DEG_S = 1.0
POSITION_TOLERANCE_MM = 1.0
ORIENTATION_TOLERANCE_DEG = 0.5
# =====================================================================


def normalized_motion_type():
    """返回规范化后的运动类型，并拒绝不支持的配置。"""
    value = str(MOTION_TYPE).strip().lower()
    if value == "movej":
        return "MoveJ"
    if value == "movel":
        return "MoveL"
    raise ValueError('MOTION_TYPE 只能填写 "MoveJ" 或 "MoveL"')


def validate_percentage(name, value):
    """校验机械臂百分比参数，避免将越界值发送给控制器。"""
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是 1～100 之间的数值")
    number = float(value)
    if not math.isfinite(number) or not 1.0 <= number <= 100.0:
        raise ValueError(f"{name} 必须是 1～100 之间的数值，当前为 {value!r}")
    return number


def validate_pose(name, pose):
    """校验 TCP 位姿的长度、有限性和最低安全高度。"""
    if not isinstance(pose, (tuple, list)) or len(pose) != 6:
        raise ValueError(f"{name} 必须包含 6 个数值")
    values = [float(item) for item in pose]
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} 必须全部为有限数值")
    if values[2] < MINIMUM_TCP_Z_MM:
        raise ValueError(
            f"{name} 的 Z={values[2]:.3f} mm 低于安全下限 {MINIMUM_TCP_Z_MM:.3f} mm"
        )
    return values


def validate_parameters():
    """在连接机械臂之前完成全部本地参数校验。"""
    motion_type = normalized_motion_type()
    point_a = validate_pose("POINT_A", POINT_A)
    point_b = validate_pose("POINT_B", POINT_B)
    move_speed = validate_percentage("MOVE_SPEED", MOVE_SPEED)
    preposition_speed = validate_percentage("PREPOSITION_SPEED", PREPOSITION_SPEED)
    acceleration_scale = validate_percentage("ACCELERATION_SCALE", ACCELERATION_SCALE)

    if not 0 <= int(TOOL_ID) <= 14 or int(TOOL_ID) != TOOL_ID:
        raise ValueError("TOOL_ID 必须是 0～14 之间的整数")
    if not 0 <= int(USER_ID) <= 14 or int(USER_ID) != USER_ID:
        raise ValueError("USER_ID 必须是 0～14 之间的整数")
    if math.dist(point_a[:3], point_b[:3]) <= 0.0:
        raise ValueError("POINT_A 和 POINT_B 的位置不能相同")
    if STABLE_TIMEOUT_SECONDS <= 0.0:
        raise ValueError("STABLE_TIMEOUT_SECONDS 必须大于 0")
    if SAMPLE_INTERVAL_SECONDS <= 0.0:
        raise ValueError("SAMPLE_INTERVAL_SECONDS 必须大于 0")
    if STABLE_CONFIRM_SECONDS < 0.0:
        raise ValueError("STABLE_CONFIRM_SECONDS 不能小于 0")

    return {
        "motion_type": motion_type,
        "point_a": point_a,
        "point_b": point_b,
        "move_speed": move_speed,
        "preposition_speed": preposition_speed,
        "acceleration_scale": acceleration_scale,
    }


def angle_error_deg(actual, target):
    """计算考虑 ±180° 环绕后的最小角度误差。"""
    return abs((float(actual) - float(target) + 180.0) % 360.0 - 180.0)


def position_error_mm(actual_pose, target_pose):
    """计算 TCP 三维位置误差。"""
    return math.dist(actual_pose[:3], target_pose[:3])


def orientation_error_deg(actual_pose, target_pose):
    """返回 Rx、Ry、Rz 三个分量中的最大姿态误差。"""
    return max(angle_error_deg(actual_pose[index], target_pose[index]) for index in range(3, 6))


def unpack_rpc_result(name, result, value_length=None):
    """统一校验法奥 XML-RPC 查询结果，并返回有效数据。"""
    if not isinstance(result, (tuple, list)) or len(result) < 2:
        raise RuntimeError(f"{name} 返回格式异常：{result!r}")
    if result[0] != 0:
        raise RuntimeError(f"{name} 查询失败，错误码：{result[0]}")

    value = result[1]
    if value_length is not None:
        if not isinstance(value, (tuple, list)) or len(value) != value_length:
            raise RuntimeError(f"{name} 数据长度异常：{value!r}")
        return [float(item) for item in value]
    return value


def ensure_move_succeeded(name, result):
    """兼容布尔返回值和法奥 XML-RPC 整数错误码。"""
    succeeded = result if isinstance(result, bool) else result == 0
    if not succeeded:
        raise RuntimeError(f"{name} 执行失败，返回值：{result!r}")


def read_robot_sample(xmlrpc_arm):
    """读取控制器完成信号、TCP 实际合速度和实际位姿。"""
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
    return motion_done, abs(composite_speed[0]), abs(composite_speed[1]), actual_pose


def sample_checks(sample, target_pose):
    """计算一次采样的速度、到位和综合停稳判定。"""
    motion_done, linear_speed, angular_speed, actual_pose = sample
    position_error = position_error_mm(actual_pose, target_pose)
    orientation_error = orientation_error_deg(actual_pose, target_pose)
    all_ok = (
        motion_done
        and linear_speed <= LINEAR_SPEED_THRESHOLD_MM_S
        and angular_speed <= ANGULAR_SPEED_THRESHOLD_DEG_S
        and position_error <= POSITION_TOLERANCE_MM
        and orientation_error <= ORIENTATION_TOLERANCE_DEG
    )
    return all_ok, position_error, orientation_error


def print_timeout_diagnosis(sample, target_pose):
    """停稳等待超时时输出最后一次反馈，便于判断失败原因。"""
    print("\n========== 停稳超时诊断 ==========")
    if sample is None:
        print("没有取得有效采样。")
        return

    motion_done, linear_speed, angular_speed, actual_pose = sample
    _, position_error, orientation_error = sample_checks(sample, target_pose)
    print(f"控制器运动完成信号：{motion_done}")
    print(f"TCP 线速度：{linear_speed:.6f} mm/s")
    print(f"TCP 姿态速度：{angular_speed:.6f} °/s")
    print(f"实际 TCP 位姿：{[round(value, 6) for value in actual_pose]}")
    print(f"位置误差：{position_error:.6f} mm")
    print(f"姿态误差：{orientation_error:.6f}°")


def wait_until_stable(xmlrpc_arm, target_pose):
    """从阻塞运动返回后持续采样，直到所有条件连续满足指定时长。"""
    wait_started = time.perf_counter()
    deadline = wait_started + STABLE_TIMEOUT_SECONDS
    next_sample_time = wait_started
    stable_since = None
    first_all_ok_time = None
    last_sample = None
    sample_count = 0

    while True:
        now = time.perf_counter()
        if now >= deadline:
            print_timeout_diagnosis(last_sample, target_pose)
            raise TimeoutError(f"等待机械臂停稳超过 {STABLE_TIMEOUT_SECONDS:.1f} 秒")
        if now < next_sample_time:
            time.sleep(next_sample_time - now)

        sample_started = time.perf_counter()
        last_sample = read_robot_sample(xmlrpc_arm)
        sample_count += 1
        all_ok, position_error, orientation_error = sample_checks(last_sample, target_pose)

        if all_ok:
            if first_all_ok_time is None:
                first_all_ok_time = sample_started
            if stable_since is None:
                stable_since = sample_started
            if sample_started - stable_since >= STABLE_CONFIRM_SECONDS:
                return {
                    "wait_seconds": sample_started - wait_started,
                    "first_all_ok_seconds": first_all_ok_time - wait_started,
                    "sample_count": sample_count,
                    "last_sample": last_sample,
                    "position_error_mm": position_error,
                    "orientation_error_deg": orientation_error,
                }
        else:
            stable_since = None

        next_sample_time = max(
            sample_started + SAMPLE_INTERVAL_SECONDS,
            time.perf_counter(),
        )


def move_to_point_a(xmlrpc_arm, point_a, preposition_speed):
    """使用不计时的低速 MoveL 回到固定起点，并等待真正停稳。"""
    print("\n正在低速 MoveL 回到 A 点，此动作不计入测试时间……")
    result = xmlrpc_arm.MoveL(
        point_a,
        tool=int(TOOL_ID),
        user=int(USER_ID),
        vel=preposition_speed,
        blendR=-1.0,
    )
    ensure_move_succeeded("预定位 MoveL", result)
    stable_result = wait_until_stable(xmlrpc_arm, point_a)
    print(
        "A 点已确认停稳："
        f"位置误差 {stable_result['position_error_mm']:.6f} mm，"
        f"姿态误差 {stable_result['orientation_error_deg']:.6f}°"
    )


def read_actual_joint_position(xmlrpc_arm):
    """读取 A 点的六轴实际关节角。"""
    return unpack_rpc_result(
        "GetActualJointPosDegree",
        xmlrpc_arm.GetActualJointPosDegree(),
        value_length=6,
    )


def solve_target_joint_position(xmlrpc_arm, point_b, joint_reference):
    """以 A 点关节角为参考求 B 点逆解，固定关节构型分支。"""
    return unpack_rpc_result(
        "GetInverseKinRef",
        xmlrpc_arm.GetInverseKinRef(0, point_b, joint_reference),
        value_length=6,
    )


def execute_measured_motion(xmlrpc_arm, parameters, target_joint):
    """执行一次选定运动并返回阻塞调用和停稳计时结果。"""
    motion_type = parameters["motion_type"]
    point_b = parameters["point_b"]
    move_speed = parameters["move_speed"]

    move_started = time.perf_counter()
    if motion_type == "MoveL":
        result = xmlrpc_arm.MoveL(
            point_b,
            tool=int(TOOL_ID),
            user=int(USER_ID),
            joint_pos=target_joint,
            vel=move_speed,
            blendR=-1.0,
        )
    else:
        result = xmlrpc_arm.MoveJ(
            target_joint,
            tool=int(TOOL_ID),
            user=int(USER_ID),
            desc_pos=point_b,
            vel=move_speed,
            blendT=-1.0,
        )
    move_returned = time.perf_counter()
    ensure_move_succeeded(motion_type, result)

    stable_result = wait_until_stable(xmlrpc_arm, point_b)
    return {
        "controller_result": result,
        "move_call_seconds": move_returned - move_started,
        "stable_wait_seconds": stable_result["wait_seconds"],
        "total_seconds": time.perf_counter() - move_started,
        "stable_result": stable_result,
    }


def joint_change_statistics(point_a_joint, point_b_joint):
    """返回六轴关节变化量、最大单轴变化量和欧氏范数。"""
    changes = [
        abs(float(target) - float(start))
        for start, target in zip(point_a_joint, point_b_joint)
    ]
    return changes, max(changes), math.sqrt(sum(value * value for value in changes))


def print_report(parameters, point_a_joint, point_b_joint, measurement):
    """打印便于复制比较的中文单次测试报告。"""
    changes, maximum_change, joint_norm = joint_change_statistics(
        point_a_joint,
        point_b_joint,
    )
    stable_result = measurement["stable_result"]
    motion_done, linear_speed, angular_speed, actual_pose = stable_result["last_sample"]

    print("\n========== MoveJ / MoveL 单次耗时测试报告 ==========")
    print(f"运动类型：{parameters['motion_type']}")
    print(f"运动速度：{parameters['move_speed']:.1f}%")
    print(f"全局加速度：{parameters['acceleration_scale']:.1f}%")
    print(f"A/B 直线位置距离：{math.dist(parameters['point_a'][:3], parameters['point_b'][:3]):.6f} mm")
    print(f"A 点实际关节角：{[round(value, 6) for value in point_a_joint]}")
    print(f"B 点目标关节角：{[round(value, 6) for value in point_b_joint]}")
    print(f"各关节绝对变化量：{[round(value, 6) for value in changes]}°")
    print(f"最大单轴关节变化量：{maximum_change:.6f}°")
    print(f"关节变化量欧氏范数：{joint_norm:.6f}°")
    print(f"控制器运动返回值：{measurement['controller_result']!r}")
    print(f"{parameters['motion_type']} 阻塞调用耗时：{measurement['move_call_seconds']:.6f} 秒")
    print(f"调用返回后确认停稳耗时：{measurement['stable_wait_seconds']:.6f} 秒")
    print(f"下发运动到确认停稳总耗时：{measurement['total_seconds']:.6f} 秒")
    print(f"返回后首次满足全部停稳条件：{stable_result['first_all_ok_seconds']:.6f} 秒")
    print(f"停稳等待采样次数：{stable_result['sample_count']}")
    print(f"最终控制器运动完成信号：{motion_done}")
    print(f"最终 TCP 速度：线速度 {linear_speed:.6f} mm/s，姿态速度 {angular_speed:.6f} °/s")
    print(f"最终 TCP 位姿：{[round(value, 6) for value in actual_pose]}")
    print(f"最终位置误差：{stable_result['position_error_mm']:.6f} mm")
    print(f"最终姿态误差：{stable_result['orientation_error_deg']:.6f}°")
    print(
        "复制对比："
        f"{parameters['motion_type']}, "
        f"速度={parameters['move_speed']:.1f}%, "
        f"调用={measurement['move_call_seconds']:.6f}s, "
        f"停稳={measurement['stable_wait_seconds']:.6f}s, "
        f"总计={measurement['total_seconds']:.6f}s, "
        f"位置误差={stable_result['position_error_mm']:.6f}mm"
    )


def main():
    parameters = validate_parameters()
    print("警告：本脚本会驱动真实机械臂运动。")
    print(f"测试类型：{parameters['motion_type']}")
    print(f"A 点：{parameters['point_a']}")
    print(f"B 点：{parameters['point_b']}")
    print(f"A/B 直线位置距离：{math.dist(parameters['point_a'][:3], parameters['point_b'][:3]):.3f} mm")
    print(f"测试速度：{parameters['move_speed']:.1f}%")
    print(f"预定位速度：{parameters['preposition_speed']:.1f}%")
    print(f"全局加速度：{parameters['acceleration_scale']:.1f}%")
    if parameters["motion_type"] == "MoveJ":
        print("安全提醒：MoveJ 的 TCP 中间轨迹不是直线，可能下沉或绕行。")

    confirmation = input("确认 A 点、B 点及预定位路径安全后，输入 yes 开始预定位：").strip().lower()
    if confirmation != "yes":
        print("已取消测试，未连接机械臂。")
        return

    arm = AkaiFr()
    xmlrpc_arm = arm.arm
    try:
        # 全局速度固定为 100%，实际测试速度由每条运动指令的 vel 参数控制。
        arm.set_speed(100)
        arm.set_tcf(1, [0, 0, 0, 0, 0, 0])
        acceleration_result = xmlrpc_arm.SetOaccScale(parameters["acceleration_scale"])
        ensure_move_succeeded("设置全局加速度", acceleration_result)

        move_to_point_a(
            xmlrpc_arm,
            parameters["point_a"],
            parameters["preposition_speed"],
        )
        point_a_joint = read_actual_joint_position(xmlrpc_arm)
        point_b_joint = solve_target_joint_position(
            xmlrpc_arm,
            parameters["point_b"],
            point_a_joint,
        )

        changes, maximum_change, joint_norm = joint_change_statistics(
            point_a_joint,
            point_b_joint,
        )
        print(f"A 点实际关节角：{[round(value, 6) for value in point_a_joint]}")
        print(f"B 点逆解关节角：{[round(value, 6) for value in point_b_joint]}")
        print(f"各关节绝对变化量：{[round(value, 6) for value in changes]}°")
        print(f"最大单轴变化量：{maximum_change:.6f}°，关节变化范数：{joint_norm:.6f}°")

        confirmation = input(
            f"确认将以 {parameters['move_speed']:.1f}% 执行 "
            f"{parameters['motion_type']} 从 A 到 B，输入 yes 开始计时："
        ).strip().lower()
        if confirmation != "yes":
            print("已取消计时运动，机械臂停留在 A 点。")
            return

        measurement = execute_measured_motion(
            xmlrpc_arm,
            parameters,
            point_b_joint,
        )
        print_report(parameters, point_a_joint, point_b_joint, measurement)
    finally:
        close_rpc = getattr(xmlrpc_arm, "CloseRPC", None)
        if callable(close_rpc):
            close_rpc()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, TimeoutError, ValueError) as error:
        print(f"测试失败：{error}")
