#!/home/zhl/fr3env/fr3env/bin/python
"""测量吸盘喷气服务及底层 SDK 调用的耗时。

使用方法：
1. 先在文件开头设置 TEST_MODE，并确认机械臂周围无人、吸盘下方没有需要保持的方块。
2. 将 ENABLE_HARDWARE_COMMANDS 改为 True。
3. 使用指定虚拟环境直接运行本文件；不使用命令行参数。

模式说明：
- ros_service：推荐。控制节点保持运行，测量一次 /control/set_suction(BLOW) 的端到端耗时。
- direct_sdk：必须先停止 control_node，直接测量电磁阀、气泵两条 SDK 调用的各自耗时。
  不要在 control_node 仍连接机器人时使用该模式，避免两个进程同时控制同一台机器人。
"""

import csv
import os
import statistics
import time


# 测试模式只能是 "ros_service" 或 "direct_sdk"。
TEST_MODE = "direct_sdk"

# 安全开关：默认 False 时不连接机器人、不发送任何吸盘指令。
ENABLE_HARDWARE_COMMANDS = True

# 仅 direct_sdk 模式使用：确认 control_node 已停止后，才改为 True。
DIRECT_SDK_CONTROL_NODE_STOPPED_CONFIRMED = True

# 每个正式样本前先切回吸气状态，复现“抓住方块后喷气释放”的调用顺序。
PREPARE_SUCTION_BEFORE_EACH_SAMPLE = True

# 切到吸气后保持多久再测喷气，确保每次电磁阀都从稳定吸气状态切换。
# 此等待发生在计时开始前，不计入 BLOW 耗时；设为 0 会退回连续切换测试。
PREPARE_SUCTION_HOLD_SECONDS = 1.0

# 预热样本不计入统计，避免首次 RPC/SDK 调用的初始化耗时干扰结论。
WARMUP_COUNT = 3
SAMPLE_COUNT = 5

# 样本之间的间隔不计入耗时统计。设为 0 可测试连续调用，设为正数可降低频繁切换负担。
INTERVAL_SECONDS = 0.10

# 测试结束后把吸盘关闭，避免遗留喷气或吸气状态。
RESTORE_OFF_AFTER_TEST = True

# ROS 服务等待上限，单位秒。
SERVICE_WAIT_TIMEOUT_SECONDS = 5.0

# 测试结果 CSV 的保存位置；设为 None 则只打印结果。
OUTPUT_CSV_PATH = "/tmp/吸盘喷气耗时测试.csv"


SUCK = 0
BLOW = 1
OFF = 2


def _require_valid_config():
    """在连接硬件前校验配置，防止因拼写错误进入错误模式。"""
    if TEST_MODE not in {"ros_service", "direct_sdk"}:
        raise ValueError('TEST_MODE 只能是 "ros_service" 或 "direct_sdk"')
    if WARMUP_COUNT < 0 or SAMPLE_COUNT <= 0:
        raise ValueError("WARMUP_COUNT 必须不小于 0，SAMPLE_COUNT 必须大于 0")
    if INTERVAL_SECONDS < 0.0:
        raise ValueError("INTERVAL_SECONDS 不能小于 0")
    if PREPARE_SUCTION_HOLD_SECONDS < 0.0:
        raise ValueError("PREPARE_SUCTION_HOLD_SECONDS 不能小于 0")


def _summary(values_ms):
    """返回一组毫秒数据的稳定性摘要。"""
    if not values_ms:
        return None
    return {
        "样本数": len(values_ms),
        "平均值_ms": statistics.mean(values_ms),
        "中位数_ms": statistics.median(values_ms),
        "最小值_ms": min(values_ms),
        "最大值_ms": max(values_ms),
        "跨度_ms": max(values_ms) - min(values_ms),
        "标准差_ms": statistics.pstdev(values_ms) if len(values_ms) > 1 else 0.0,
    }


def _print_summary(label, values_ms):
    """以中文打印汇总，方便直接与任务日志中的间隔比较。"""
    summary = _summary(values_ms)
    if summary is None:
        return
    print(
        f"{label}：样本={summary['样本数']}，"
        f"平均={summary['平均值_ms']:.1f} ms，"
        f"中位={summary['中位数_ms']:.1f} ms，"
        f"最小={summary['最小值_ms']:.1f} ms，"
        f"最大={summary['最大值_ms']:.1f} ms，"
        f"跨度={summary['跨度_ms']:.1f} ms，"
        f"标准差={summary['标准差_ms']:.1f} ms"
    )


def _save_csv(rows):
    """保存逐样本耗时，后续可用表格软件查看是否存在固定延时。"""
    if OUTPUT_CSV_PATH is None:
        return
    output_dir = os.path.dirname(OUTPUT_CSV_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fieldnames = ["样本序号", "模式", "喷气服务总耗时_ms", "电磁阀耗时_ms", "气泵耗时_ms"]
    with open(OUTPUT_CSV_PATH, "w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"逐样本结果已保存到: {OUTPUT_CSV_PATH}")


def _sleep_between_samples():
    """样本间隔不属于被测调用，刻意放在记录结束之后。"""
    if INTERVAL_SECONDS > 0.0:
        time.sleep(INTERVAL_SECONDS)


def _prepare_suction_before_blow(set_suck_func):
    """把吸盘切到稳定吸气状态；保持时间不属于待测喷气调用。"""
    if not PREPARE_SUCTION_BEFORE_EACH_SAMPLE:
        return
    set_suck_func()
    if PREPARE_SUCTION_HOLD_SECONDS > 0.0:
        time.sleep(PREPARE_SUCTION_HOLD_SECONDS)


def _run_ros_service_test():
    """测量 competition 实际使用的 BLOW ROS 服务端到端耗时。"""
    import rospy
    from control.srv import SetSuction, SetSuctionRequest

    rospy.init_node("suction_latency_test", anonymous=True)
    print("等待 /control/set_suction 服务……")
    rospy.wait_for_service("/control/set_suction", timeout=SERVICE_WAIT_TIMEOUT_SECONDS)
    set_suction = rospy.ServiceProxy("/control/set_suction", SetSuction)

    def set_state(state):
        response = set_suction(SetSuctionRequest(state=state))
        if not response.success:
            raise RuntimeError(f"吸盘服务失败: {response.message}")

    try:
        for _ in range(WARMUP_COUNT):
            _prepare_suction_before_blow(lambda: set_state(SUCK))
            set_state(BLOW)
            _sleep_between_samples()

        rows = []
        for index in range(1, SAMPLE_COUNT + 1):
            _prepare_suction_before_blow(lambda: set_state(SUCK))
            started_at = time.perf_counter()
            set_state(BLOW)
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            rows.append(
                {
                    "样本序号": index,
                    "模式": "ROS服务端到端",
                    "喷气服务总耗时_ms": f"{elapsed_ms:.3f}",
                    "电磁阀耗时_ms": "",
                    "气泵耗时_ms": "",
                }
            )
            print(f"第 {index:02d} 次 BLOW 服务总耗时: {elapsed_ms:.1f} ms")
            _sleep_between_samples()
        return rows
    finally:
        if RESTORE_OFF_AFTER_TEST:
            try:
                set_state(OFF)
                print("测试结束，吸盘已关闭。")
            except Exception as exc:
                print(f"测试结束时关闭吸盘失败，请人工确认吸盘状态: {exc}")


def _run_direct_sdk_test():
    """逐项测量控制节点中 BLOW 对应的两条底层 SDK 调用。"""
    from akai_fr import AkaiElectricSucker, AkaiFr

    print("直连 SDK 模式：请确认 control_node 已停止，避免并发控制机器人。")
    arm = AkaiFr()
    sucker = AkaiElectricSucker(arm)

    def set_suck_state():
        sucker.set_solenoid_valve(False)
        sucker.set_pump_motor(True)

    def set_off_state():
        sucker.set_solenoid_valve(False)
        sucker.set_pump_motor(False)

    def measure_blow():
        total_started_at = time.perf_counter()
        valve_started_at = time.perf_counter()
        sucker.set_solenoid_valve(True)
        valve_elapsed_ms = (time.perf_counter() - valve_started_at) * 1000.0
        pump_started_at = time.perf_counter()
        sucker.set_pump_motor(True)
        pump_elapsed_ms = (time.perf_counter() - pump_started_at) * 1000.0
        total_elapsed_ms = (time.perf_counter() - total_started_at) * 1000.0
        return total_elapsed_ms, valve_elapsed_ms, pump_elapsed_ms

    try:
        for _ in range(WARMUP_COUNT):
            _prepare_suction_before_blow(set_suck_state)
            measure_blow()
            _sleep_between_samples()

        rows = []
        for index in range(1, SAMPLE_COUNT + 1):
            _prepare_suction_before_blow(set_suck_state)
            total_ms, valve_ms, pump_ms = measure_blow()
            rows.append(
                {
                    "样本序号": index,
                    "模式": "直连SDK",
                    "喷气服务总耗时_ms": f"{total_ms:.3f}",
                    "电磁阀耗时_ms": f"{valve_ms:.3f}",
                    "气泵耗时_ms": f"{pump_ms:.3f}",
                }
            )
            print(
                f"第 {index:02d} 次：总计={total_ms:.1f} ms，"
                f"电磁阀={valve_ms:.1f} ms，气泵={pump_ms:.1f} ms"
            )
            _sleep_between_samples()
        return rows
    finally:
        if RESTORE_OFF_AFTER_TEST:
            try:
                set_off_state()
                print("测试结束，吸盘已关闭。")
            except Exception as exc:
                print(f"测试结束时关闭吸盘失败，请人工确认吸盘状态: {exc}")


def main():
    _require_valid_config()
    if not ENABLE_HARDWARE_COMMANDS:
        print("安全开关未开启：未连接机器人、未发送吸盘指令。")
        print("确认现场安全后，将 ENABLE_HARDWARE_COMMANDS 改为 True 再运行。")
        return
    if TEST_MODE == "direct_sdk" and not DIRECT_SDK_CONTROL_NODE_STOPPED_CONFIRMED:
        raise RuntimeError(
            "直连 SDK 前必须先停止 control_node，并将 "
            "DIRECT_SDK_CONTROL_NODE_STOPPED_CONFIRMED 改为 True"
        )

    if TEST_MODE == "ros_service":
        rows = _run_ros_service_test()
    else:
        rows = _run_direct_sdk_test()

    total_values_ms = [float(row["喷气服务总耗时_ms"]) for row in rows]
    _print_summary("喷气总耗时", total_values_ms)
    if TEST_MODE == "direct_sdk":
        _print_summary("电磁阀调用耗时", [float(row["电磁阀耗时_ms"]) for row in rows])
        _print_summary("气泵调用耗时", [float(row["气泵耗时_ms"]) for row in rows])
    _save_csv(rows)


if __name__ == "__main__":
    main()
