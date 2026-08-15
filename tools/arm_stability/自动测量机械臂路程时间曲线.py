#!/home/zhl/fr3env/fr3env/bin/python
"""批量测量固定路径上的机械臂路程—时间关系。

每次测量都先使用低速 MoveL 回到 A 点，再从 A 点运动到指定距离的
目标点。脚本保存每次原始数据、按距离汇总的统计结果、JSON 插值表和中文曲线图。
"""

import csv
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path


# 将项目 src 加入导入路径，使实机工具和正式节点使用同一份执行配置。
SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from akai_fr import AkaiFr
from competition_lib.config import load_execution_config


# ========================= 直接修改的实机参数 =========================
# 默认测 MoveL；如果要测 MoveJ，必须先低速确认所有中间轨迹安全。
MOTION_TYPE = "MoveL"

# A 点是每次测量的固定起点，PATH_END 定义路径方向和 450 mm 标定终点。
# 位姿格式：[X, Y, Z, Rx, Ry, Rz]，单位分别为 mm 和 °。
POINT_A = [-337.4151306152343, -225.0, 330.0, -180.0, 0.0, 90.0]
PATH_END = [-337.4151306152343, 225.0, 330.0, -180.0, 0.0, 90.0]

# 短距离区间加密采样，长距离数据用于后续判断外推方式。
# 脚本会自动生成 A 点沿 PATH_END 方向移动这些距离后的目标点。
DISTANCES_MM = [
    2,
    5,
    10,
    15,
    20,
    30,
    40,
    60,
    80,
    100,
    150,
    200,
    250,
    300,
    350,
    400,
    450,
]
# 每个距离重复测量次数，建议至少 5 次。
REPEAT_COUNT = 5
# 奇数轮从短到长，偶数轮从长到短，减少温度和时间漂移造成的顺序偏差。
ALTERNATE_DISTANCE_ORDER = True

MOVE_SPEED = 100
PREPOSITION_SPEED = 50
ACCELERATION_SCALE = 100

TOOL_ID = 0
USER_ID = 0
# 项目正式运动允许的最低 TCP 绝对高度，单位 mm；只从唯一执行配置读取。
MINIMUM_TCP_Z_MM = load_execution_config().minimum_tcp_z_mm

# 测试全部完成后是否使用低速 MoveL 回到 A 点。
RETURN_TO_A_AFTER_TEST = True

# 停稳判定参数。
STABLE_TIMEOUT_SECONDS = 5.0
SAMPLE_INTERVAL_SECONDS = 0.005
STABLE_CONFIRM_SECONDS = 0.01
LINEAR_SPEED_THRESHOLD_MM_S = 3.0
ANGULAR_SPEED_THRESHOLD_DEG_S = 1.0
POSITION_TOLERANCE_MM = 1.0
ORIENTATION_TOLERANCE_DEG = 0.5

# 结果保存到脚本同级的“路程时间标定结果”目录。
RESULT_DIRECTORY = Path(__file__).resolve().parent / "路程时间标定结果"
# =====================================================================


RAW_CSV_FIELDS = [
    "测量时间",
    "运动类型",
    "轮次",
    "本轮序号",
    "距离_mm",
    "目标_X_mm",
    "目标_Y_mm",
    "目标_Z_mm",
    "运动速度_%",
    "加速度_%",
    "控制器返回值",
    "运动调用耗时_s",
    "首次满足停稳条件总耗时_s",
    "确认停稳总耗时_s",
    "调用返回后等待_s",
    "采样次数",
    "最终位置误差_mm",
    "最终姿态误差_deg",
    "最终线速度_mm_s",
    "最终姿态速度_deg_s",
]


def normalize_motion_type(value):
    """返回规范化后的运动类型。"""
    normalized = str(value).strip().lower()
    if normalized == "movel":
        return "MoveL"
    if normalized == "movej":
        return "MoveJ"
    raise ValueError('MOTION_TYPE 只能填写 "MoveL" 或 "MoveJ"')


def validate_percentage(name, value):
    """校验速度和加速度百分比。"""
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是 1～100 之间的数值")
    number = float(value)
    if not math.isfinite(number) or not 1.0 <= number <= 100.0:
        raise ValueError(f"{name} 必须在 1～100 之间，当前为 {value!r}")
    return number


def validate_pose(name, pose):
    """校验 TCP 位姿和最低高度。"""
    if not isinstance(pose, (list, tuple)) or len(pose) != 6:
        raise ValueError(f"{name} 必须包含 6 个数值")
    values = [float(item) for item in pose]
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} 不能包含无限大或非数值")
    if values[2] < MINIMUM_TCP_Z_MM:
        raise ValueError(
            f"{name} 的 Z={values[2]:.3f} mm 低于安全下限 "
            f"{MINIMUM_TCP_Z_MM:.3f} mm"
        )
    return values


def validate_integer(name, value, minimum, maximum):
    """校验整数配置。"""
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是 {minimum}～{maximum} 之间的整数")
    try:
        integer = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} 必须是 {minimum}～{maximum} 之间的整数"
        ) from error
    if numeric != integer or not minimum <= integer <= maximum:
        raise ValueError(f"{name} 必须是 {minimum}～{maximum} 之间的整数")
    return integer


def build_target_pose(point_a, path_end, distance_mm):
    """按 A 点到路径末端的方向，生成指定距离的目标位姿。"""
    maximum_distance = math.dist(point_a[:3], path_end[:3])
    ratio = float(distance_mm) / maximum_distance
    position = [
        point_a[index] + ratio * (path_end[index] - point_a[index])
        for index in range(3)
    ]
    return position + list(point_a[3:])


def validate_parameters():
    """在连接机械臂之前校验全部配置，并生成目标点。"""
    motion_type = normalize_motion_type(MOTION_TYPE)
    point_a = validate_pose("POINT_A", POINT_A)
    path_end = validate_pose("PATH_END", PATH_END)
    maximum_distance = math.dist(point_a[:3], path_end[:3])
    if maximum_distance <= 0.0:
        raise ValueError("POINT_A 和 PATH_END 的位置不能相同")
    if any(
        angle_error_deg(point_a[index], path_end[index]) > 1e-9
        for index in range(3, 6)
    ):
        raise ValueError("本标定要求 POINT_A 和 PATH_END 的姿态完全相同")

    if not isinstance(DISTANCES_MM, (list, tuple)) or not DISTANCES_MM:
        raise ValueError("DISTANCES_MM 必须是非空距离列表")
    distances = []
    for value in DISTANCES_MM:
        if isinstance(value, bool):
            raise ValueError("测量距离必须是有限正数")
        distance = float(value)
        if not math.isfinite(distance) or distance <= 0.0:
            raise ValueError(f"测量距离必须是有限正数，当前为 {value!r}")
        if distance > maximum_distance + 1e-9:
            raise ValueError(
                f"测量距离 {distance:.3f} mm 超过路径最大长度 "
                f"{maximum_distance:.3f} mm"
            )
        distances.append(distance)
    if len(set(distances)) != len(distances):
        raise ValueError("DISTANCES_MM 中不能有重复距离")
    distances.sort()

    repeat_count = validate_integer("REPEAT_COUNT", REPEAT_COUNT, 1, 100)
    tool_id = validate_integer("TOOL_ID", TOOL_ID, 0, 14)
    user_id = validate_integer("USER_ID", USER_ID, 0, 14)
    move_speed = validate_percentage("MOVE_SPEED", MOVE_SPEED)
    preposition_speed = validate_percentage("PREPOSITION_SPEED", PREPOSITION_SPEED)
    acceleration_scale = validate_percentage("ACCELERATION_SCALE", ACCELERATION_SCALE)

    if STABLE_TIMEOUT_SECONDS <= 0.0:
        raise ValueError("STABLE_TIMEOUT_SECONDS 必须大于 0")
    if SAMPLE_INTERVAL_SECONDS <= 0.0:
        raise ValueError("SAMPLE_INTERVAL_SECONDS 必须大于 0")
    if STABLE_CONFIRM_SECONDS < 0.0:
        raise ValueError("STABLE_CONFIRM_SECONDS 不能小于 0")
    for name, value in (
        ("LINEAR_SPEED_THRESHOLD_MM_S", LINEAR_SPEED_THRESHOLD_MM_S),
        ("ANGULAR_SPEED_THRESHOLD_DEG_S", ANGULAR_SPEED_THRESHOLD_DEG_S),
        ("POSITION_TOLERANCE_MM", POSITION_TOLERANCE_MM),
        ("ORIENTATION_TOLERANCE_DEG", ORIENTATION_TOLERANCE_DEG),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} 必须是有限非负数")

    targets = [
        {
            "distance_mm": distance,
            "pose": build_target_pose(point_a, path_end, distance),
        }
        for distance in distances
    ]
    for index, target in enumerate(targets, start=1):
        validate_pose(f"目标点 {index}", target["pose"])

    return {
        "motion_type": motion_type,
        "point_a": point_a,
        "path_end": path_end,
        "maximum_distance_mm": maximum_distance,
        "distances_mm": distances,
        "targets": targets,
        "repeat_count": repeat_count,
        "tool_id": tool_id,
        "user_id": user_id,
        "move_speed": move_speed,
        "preposition_speed": preposition_speed,
        "acceleration_scale": acceleration_scale,
    }


def angle_error_deg(actual, target):
    """计算考虑 ±180° 环绕后的最小角度误差。"""
    return abs((float(actual) - float(target) + 180.0) % 360.0 - 180.0)


def pose_errors(actual_pose, target_pose):
    """计算 TCP 位置误差和最大姿态分量误差。"""
    position_error = math.dist(actual_pose[:3], target_pose[:3])
    orientation_error = max(
        angle_error_deg(actual_pose[index], target_pose[index])
        for index in range(3, 6)
    )
    return position_error, orientation_error


def unpack_rpc_result(name, result, value_length=None):
    """校验法奥 XML-RPC 查询结果并返回数据。"""
    if not isinstance(result, (tuple, list)) or len(result) < 2:
        raise RuntimeError(f"{name} 返回格式异常：{result!r}")
    if result[0] != 0:
        raise RuntimeError(f"{name} 失败，错误码：{result[0]}")
    value = result[1]
    if value_length is not None:
        if not isinstance(value, (tuple, list)) or len(value) != value_length:
            raise RuntimeError(f"{name} 数据长度异常：{value!r}")
        return [float(item) for item in value]
    return value


def ensure_command_succeeded(name, result):
    """校验运动或设置指令的返回值。"""
    succeeded = result if isinstance(result, bool) else result == 0
    if not succeeded:
        raise RuntimeError(f"{name} 执行失败，返回值：{result!r}")


def read_robot_sample(xmlrpc_arm):
    """读取运动完成信号、TCP 合速度和实际位姿。"""
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


def wait_until_stable(xmlrpc_arm, target_pose):
    """等待所有停稳条件连续满足指定时长。"""
    wait_started = time.perf_counter()
    deadline = wait_started + STABLE_TIMEOUT_SECONDS
    stable_since = None
    first_all_ok_time = None
    sample_count = 0
    last_sample = None

    while True:
        sample_time = time.perf_counter()
        if sample_time >= deadline:
            if last_sample is None:
                diagnosis = "未取得有效采样"
            else:
                done, linear_speed, angular_speed, actual_pose = last_sample
                position_error, orientation_error = pose_errors(actual_pose, target_pose)
                diagnosis = (
                    f"完成信号={done}, 线速度={linear_speed:.6f} mm/s, "
                    f"姿态速度={angular_speed:.6f} °/s, "
                    f"位置误差={position_error:.6f} mm, "
                    f"姿态误差={orientation_error:.6f}°"
                )
            raise TimeoutError(
                f"等待停稳超过 {STABLE_TIMEOUT_SECONDS:.1f} 秒；{diagnosis}"
            )

        last_sample = read_robot_sample(xmlrpc_arm)
        sample_count += 1
        done, linear_speed, angular_speed, actual_pose = last_sample
        position_error, orientation_error = pose_errors(actual_pose, target_pose)
        all_ok = (
            done
            and linear_speed <= LINEAR_SPEED_THRESHOLD_MM_S
            and angular_speed <= ANGULAR_SPEED_THRESHOLD_DEG_S
            and position_error <= POSITION_TOLERANCE_MM
            and orientation_error <= ORIENTATION_TOLERANCE_DEG
        )

        if all_ok:
            if first_all_ok_time is None:
                first_all_ok_time = sample_time
            if stable_since is None:
                stable_since = sample_time
            if sample_time - stable_since >= STABLE_CONFIRM_SECONDS:
                return {
                    "wait_seconds": sample_time - wait_started,
                    "first_all_ok_seconds": first_all_ok_time - wait_started,
                    "sample_count": sample_count,
                    "last_sample": last_sample,
                    "position_error_mm": position_error,
                    "orientation_error_deg": orientation_error,
                }
        else:
            stable_since = None
        time.sleep(SAMPLE_INTERVAL_SECONDS)


def move_to_point_a(xmlrpc_arm, parameters):
    """使用低速 MoveL 回到 A 点，此时间不计入标定。"""
    result = xmlrpc_arm.MoveL(
        parameters["point_a"],
        tool=parameters["tool_id"],
        user=parameters["user_id"],
        vel=parameters["preposition_speed"],
        blendR=-1.0,
    )
    ensure_command_succeeded("回 A 点 MoveL", result)
    wait_until_stable(xmlrpc_arm, parameters["point_a"])


def get_actual_joints(xmlrpc_arm):
    """读取当前六轴实际关节角。"""
    return unpack_rpc_result(
        "GetActualJointPosDegree",
        xmlrpc_arm.GetActualJointPosDegree(),
        value_length=6,
    )


def preflight_inverse_kinematics(xmlrpc_arm, parameters):
    """在批量运动前一次性验证全部目标点可达。"""
    reference_joints = get_actual_joints(xmlrpc_arm)
    prepared_targets = []
    for target in parameters["targets"]:
        joints = unpack_rpc_result(
            f"GetInverseKinRef({target['distance_mm']:.3f} mm)",
            xmlrpc_arm.GetInverseKinRef(0, target["pose"], reference_joints),
            value_length=6,
        )
        prepared_targets.append({**target, "joints": joints})
    return reference_joints, prepared_targets


def execute_measured_motion(xmlrpc_arm, parameters, target):
    """执行一次计时运动。"""
    move_started = time.perf_counter()
    if parameters["motion_type"] == "MoveL":
        controller_result = xmlrpc_arm.MoveL(
            target["pose"],
            tool=parameters["tool_id"],
            user=parameters["user_id"],
            joint_pos=target["joints"],
            vel=parameters["move_speed"],
            blendR=-1.0,
        )
    else:
        controller_result = xmlrpc_arm.MoveJ(
            target["joints"],
            tool=parameters["tool_id"],
            user=parameters["user_id"],
            desc_pos=target["pose"],
            vel=parameters["move_speed"],
            blendT=-1.0,
        )
    move_returned = time.perf_counter()
    ensure_command_succeeded(parameters["motion_type"], controller_result)
    stable_result = wait_until_stable(xmlrpc_arm, target["pose"])
    finished = time.perf_counter()
    return {
        "controller_result": controller_result,
        "move_call_seconds": move_returned - move_started,
        "first_all_ok_total_seconds": (
            move_returned - move_started + stable_result["first_all_ok_seconds"]
        ),
        "stable_total_seconds": finished - move_started,
        "stable_result": stable_result,
    }


def measurement_to_row(parameters, target, round_index, sequence_index, measurement):
    """将一次测量结果转换为 CSV 记录。"""
    stable = measurement["stable_result"]
    _, linear_speed, angular_speed, _ = stable["last_sample"]
    return {
        "测量时间": datetime.now().isoformat(timespec="seconds"),
        "运动类型": parameters["motion_type"],
        "轮次": round_index,
        "本轮序号": sequence_index,
        "距离_mm": target["distance_mm"],
        "目标_X_mm": target["pose"][0],
        "目标_Y_mm": target["pose"][1],
        "目标_Z_mm": target["pose"][2],
        "运动速度_%": parameters["move_speed"],
        "加速度_%": parameters["acceleration_scale"],
        "控制器返回值": measurement["controller_result"],
        "运动调用耗时_s": measurement["move_call_seconds"],
        "首次满足停稳条件总耗时_s": measurement[
            "first_all_ok_total_seconds"
        ],
        "确认停稳总耗时_s": measurement["stable_total_seconds"],
        "调用返回后等待_s": stable["wait_seconds"],
        "采样次数": stable["sample_count"],
        "最终位置误差_mm": stable["position_error_mm"],
        "最终姿态误差_deg": stable["orientation_error_deg"],
        "最终线速度_mm_s": linear_speed,
        "最终姿态速度_deg_s": angular_speed,
    }


def percentile(values, percentage):
    """使用线性插值计算百分位数。"""
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(percentage) / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    ratio = position - lower
    return ordered[lower] * (1.0 - ratio) + ordered[upper] * ratio


def build_summary(rows):
    """按距离汇总均值、中位数、分位数和范围。"""
    grouped = {}
    for row in rows:
        grouped.setdefault(float(row["距离_mm"]), []).append(row)

    summary = []
    time_fields = (
        "运动调用耗时_s",
        "首次满足停稳条件总耗时_s",
        "确认停稳总耗时_s",
    )
    for distance in sorted(grouped):
        distance_rows = grouped[distance]
        item = {"距离_mm": distance, "有效次数": len(distance_rows)}
        for field in time_fields:
            values = [float(row[field]) for row in distance_rows]
            # 指定虚拟环境使用 Python 3.8，不使用 str.removesuffix。
            prefix = field[:-2] if field.endswith("_s") else field
            item[f"{prefix}均值_s"] = statistics.fmean(values)
            item[f"{prefix}中位数_s"] = statistics.median(values)
            item[f"{prefix}P10_s"] = percentile(values, 10.0)
            item[f"{prefix}P90_s"] = percentile(values, 90.0)
            item[f"{prefix}最小值_s"] = min(values)
            item[f"{prefix}最大值_s"] = max(values)
        summary.append(item)
    return summary


def write_summary_csv(path, summary):
    """写入按距离汇总的 CSV。"""
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def write_interpolation_json(path, parameters, summary):
    """写入后续优化代码可直接读取的插值表。"""
    table = []
    for item in summary:
        table.append(
            {
                "distance_mm": item["距离_mm"],
                "move_call_median_s": item["运动调用耗时中位数_s"],
                "move_call_p10_s": item["运动调用耗时P10_s"],
                "first_all_ok_median_s": item[
                    "首次满足停稳条件总耗时中位数_s"
                ],
                "first_all_ok_p10_s": item[
                    "首次满足停稳条件总耗时P10_s"
                ],
                "stable_total_median_s": item["确认停稳总耗时中位数_s"],
                "stable_total_p90_s": item["确认停稳总耗时P90_s"],
            }
        )
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "motion_type": parameters["motion_type"],
        "move_speed_percent": parameters["move_speed"],
        "acceleration_percent": parameters["acceleration_scale"],
        "point_a": parameters["point_a"],
        "path_end": parameters["path_end"],
        "stable_confirm_seconds": STABLE_CONFIRM_SECONDS,
        "recommended_interpolation": "linear",
        "distance_time_table": table,
    }
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)


def configure_chinese_font(plt, font_manager):
    """为保存的图表选择可用中文字体。"""
    preferred_names = (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
        "SimHei",
    )
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred_names:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def save_curve_plot(path, summary, parameters):
    """保存中文路程—时间曲线图。"""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/arm_distance_time_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    configure_chinese_font(plt, font_manager)
    distances = [item["距离_mm"] for item in summary]
    move_medians = [item["运动调用耗时中位数_s"] for item in summary]
    first_ok_medians = [
        item["首次满足停稳条件总耗时中位数_s"]
        for item in summary
    ]
    stable_medians = [item["确认停稳总耗时中位数_s"] for item in summary]
    stable_p90 = [item["确认停稳总耗时P90_s"] for item in summary]

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(distances, move_medians, "o-", label="运动调用耗时中位数")
    axis.plot(distances, first_ok_medians, "s-", label="首次满足停稳条件中位数")
    axis.plot(distances, stable_medians, "^-", label="确认停稳总耗时中位数")
    axis.plot(distances, stable_p90, "--", color="tab:red", label="确认停稳总耗时 P90")
    axis.set_title(
        f"{parameters['motion_type']} 路程—时间标定"
        f"（速度 {parameters['move_speed']:.0f}%，"
        f"加速度 {parameters['acceleration_scale']:.0f}%）"
    )
    axis.set_xlabel("路程（mm）")
    axis.set_ylabel("时间（s）")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def print_configuration(parameters):
    """打印自动测量的路径、数量和所有目标点。"""
    total_measurements = len(parameters["targets"]) * parameters["repeat_count"]
    print("警告：本脚本会连续驱动真实机械臂。")
    print(f"运动类型：{parameters['motion_type']}")
    print(f"A 点：{parameters['point_a']}")
    print(f"路径末端：{parameters['path_end']}")
    print(f"路径最大长度：{parameters['maximum_distance_mm']:.3f} mm")
    print(f"测量距离：{parameters['distances_mm']} mm")
    print(f"每个距离重复：{parameters['repeat_count']} 次")
    print(f"计时运动总数：{total_measurements} 次")
    print(f"测试速度：{parameters['move_speed']:.1f}%")
    print(f"回 A 点速度：{parameters['preposition_speed']:.1f}%")
    print(f"全局加速度：{parameters['acceleration_scale']:.1f}%")
    print("自动生成的目标点：")
    for target in parameters["targets"]:
        rounded_pose = [round(value, 6) for value in target["pose"]]
        print(f"  {target['distance_mm']:8.3f} mm -> {rounded_pose}")
    if parameters["motion_type"] == "MoveJ":
        print("安全提醒：MoveJ 只保证终点，TCP 中间轨迹可能下沉或绕行。")


def print_summary(summary):
    """在终端打印方便复制的中位数和 P90 结果。"""
    print("\n========== 路程—时间标定汇总 ==========")
    print("距离(mm) | 运动调用中位数(s) | 首次满足条件(s) | 确认停稳中位数(s) | 停稳P90(s)")
    for item in summary:
        print(
            f"{item['距离_mm']:9.3f} | "
            f"{item['运动调用耗时中位数_s']:18.6f} | "
            f"{item['首次满足停稳条件总耗时中位数_s']:16.6f} | "
            f"{item['确认停稳总耗时中位数_s']:20.6f} | "
            f"{item['确认停稳总耗时P90_s']:11.6f}"
        )
    print("\n后续优化默认可用“确认停稳总耗时中位数”做线性插值。")
    print("如果舅机只需在机械臂到位前完成，可改用“首次满足条件”一列。")


def run_measurements(xmlrpc_arm, parameters, prepared_targets, raw_csv_path):
    """执行全部自动测量，每次成功后立即落盘。"""
    rows = []
    total = len(prepared_targets) * parameters["repeat_count"]
    completed = 0
    with raw_csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RAW_CSV_FIELDS)
        writer.writeheader()
        csv_file.flush()

        for round_index in range(1, parameters["repeat_count"] + 1):
            targets_this_round = list(prepared_targets)
            if ALTERNATE_DISTANCE_ORDER and round_index % 2 == 0:
                targets_this_round.reverse()

            for sequence_index, target in enumerate(targets_this_round, start=1):
                completed += 1
                print(
                    f"\n[{completed}/{total}] 第 {round_index} 轮，"
                    f"距离 {target['distance_mm']:.3f} mm：正在回 A 点……"
                )
                move_to_point_a(xmlrpc_arm, parameters)
                print(f"开始 {parameters['motion_type']} 计时运动……")
                measurement = execute_measured_motion(xmlrpc_arm, parameters, target)
                row = measurement_to_row(
                    parameters,
                    target,
                    round_index,
                    sequence_index,
                    measurement,
                )
                rows.append(row)
                writer.writerow(row)
                csv_file.flush()
                print(
                    f"本次结果：运动调用 "
                    f"{row['运动调用耗时_s']:.6f} s，"
                    f"首次满足条件 "
                    f"{row['首次满足停稳条件总耗时_s']:.6f} s，"
                    f"确认停稳 "
                    f"{row['确认停稳总耗时_s']:.6f} s"
                )
    return rows


def main():
    parameters = validate_parameters()
    print_configuration(parameters)
    confirmation = input(
        "确认 A 点、全部目标点和往返路径安全后，输入 yes 开始预检："
    ).strip().lower()
    if confirmation != "yes":
        print("已取消，未连接机械臂。")
        return

    arm = AkaiFr()
    xmlrpc_arm = arm.arm
    try:
        # 全局速度固定为 100%，单条指令速度由 vel 控制。
        arm.set_speed(100)
        arm.set_tcf(1, [0, 0, 0, 0, 0, 0])
        acceleration_result = xmlrpc_arm.SetOaccScale(
            parameters["acceleration_scale"]
        )
        ensure_command_succeeded("设置全局加速度", acceleration_result)

        print("\n正在回 A 点并对全部目标点做逆解预检……")
        move_to_point_a(xmlrpc_arm, parameters)
        reference_joints, prepared_targets = preflight_inverse_kinematics(
            xmlrpc_arm,
            parameters,
        )
        print(f"A 点实际关节角：{[round(value, 6) for value in reference_joints]}")
        print(f"逆解预检通过：{len(prepared_targets)} 个目标点全部可达。")

        total = len(prepared_targets) * parameters["repeat_count"]
        confirmation = input(
            f"即将自动执行 {total} 次计时运动和对应的回 A 点运动，"
            "请保持急停可操作，输入 yes 开始："
        ).strip().lower()
        if confirmation != "yes":
            print("已取消批量测量，机械臂停留在 A 点。")
            return

        RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{timestamp}_{parameters['motion_type']}_{parameters['move_speed']:.0f}速度"
        raw_csv_path = RESULT_DIRECTORY / f"{prefix}_原始数据.csv"
        summary_csv_path = RESULT_DIRECTORY / f"{prefix}_汇总.csv"
        interpolation_json_path = RESULT_DIRECTORY / f"{prefix}_插值表.json"
        plot_path = RESULT_DIRECTORY / f"{prefix}_曲线.png"

        rows = run_measurements(
            xmlrpc_arm,
            parameters,
            prepared_targets,
            raw_csv_path,
        )
        if RETURN_TO_A_AFTER_TEST:
            print("\n全部测量完成，正在回 A 点……")
            move_to_point_a(xmlrpc_arm, parameters)

        summary = build_summary(rows)
        write_summary_csv(summary_csv_path, summary)
        write_interpolation_json(interpolation_json_path, parameters, summary)
        save_curve_plot(plot_path, summary, parameters)
        print_summary(summary)
        print("\n结果文件：")
        print(f"  原始数据：{raw_csv_path}")
        print(f"  汇总数据：{summary_csv_path}")
        print(f"  插值表：{interpolation_json_path}")
        print(f"  曲线图：{plot_path}")
    finally:
        close_rpc = getattr(xmlrpc_arm, "CloseRPC", None)
        if callable(close_rpc):
            close_rpc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n测试被用户中断。已成功的原始记录仍保留在 CSV 中。")
    except (RuntimeError, TimeoutError, ValueError, OSError) as error:
        print(f"\n测试失败：{error}")
