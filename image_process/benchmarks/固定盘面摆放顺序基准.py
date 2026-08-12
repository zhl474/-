#!/home/zhl/fr3env/fr3env/bin/python
"""对同一份基础盘面输入比较 V1 不同束宽；参数直接在文件开头修改。"""

# ===== 直接修改以下参数 =====
束宽列表 = [50, 100, 500, 1000, 5000]
结果文件名 = "固定盘面摆放顺序基准结果.json"
# ============================

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
import time


脚本目录 = Path(__file__).resolve().parent
图像包目录 = 脚本目录.parent
sys.path.insert(0, str(图像包目录))

from image_process_lib.arm_motion_time import get_default_arm_motion_time_model
from image_process_lib.task_planner import (
    ObservedBlock,
    PlacementTarget,
    load_task_layout,
)
from image_process_lib.task_sequence_optimizer import (
    TaskSequenceOptimizerConfig,
    optimize_task_sequence,
)


def 位姿(x, y, z):
    return (float(x), float(y), float(z), -180.0, 0.0, 90.0)


def 构造同一份规划输入():
    """使用基础盘面几何和固定的合成开环位姿，确保各束宽输入完全相同。"""
    布局 = load_task_layout(str(图像包目录 / "config" / "task_layout.yaml"))
    类别计数 = Counter(item["category"] for item in 布局)
    方块 = []
    实体编号 = 0
    for 类别序号, 类别 in enumerate(sorted(类别计数)):
        for 类内序号 in range(类别计数[类别]):
            方块.append(ObservedBlock(
                category=类别,
                observation_pose=位姿(
                    -450.0 + 45.0 * 类内序号,
                    -220.0 + 65.0 * 类别序号,
                    350.0,
                ),
                detected_angle_deg=13.0 * 类内序号 - 20.0,
                source_id=实体编号,
                pick_surface_z_mm=20.0,
                pick_surface_z_valid=True,
            ))
            实体编号 += 1
    目标 = [
        PlacementTarget(
            index=item["index"],
            row=item["row"],
            col=item["col"],
            desired_angle_deg=item["angle_deg"],
            category=item["category"],
            observation_pose=位姿(
                -500.0 + 30.0 * item["col"],
                -230.0 + 30.0 * item["row"],
                185.0,
            ),
            cells=item["cells"],
        )
        for item in 布局
    ]
    return 方块, 目标


def 运行基准():
    方块, 目标 = 构造同一份规划输入()
    时间模型 = get_default_arm_motion_time_model()
    结果 = []
    for 束宽 in 束宽列表:
        配置 = TaskSequenceOptimizerConfig(
            shooting_pose=位姿(-250.4151306152343, 22.14801216125488, 380.3343505859375),
            camera_to_sucker_offset_mm=(-94.1, -13.8),
            pick_surface_offset_mm=165.0,
            pick_approach_clearance_mm=3.0,
            motor_velocity_deg_per_sec=270.0,
            initial_motor_angle_deg=180.0,
            motor_lower_margin_deg=10.0,
            motor_upper_margin_deg=350.0,
            beam_width=束宽,
            report_top_candidates=20,
        )
        开始时间 = time.perf_counter()
        规划结果 = optimize_task_sequence(
            方块,
            目标,
            board_angle_deg=0.0,
            config=配置,
            motion_model=时间模型,
        )
        端到端耗时 = time.perf_counter() - 开始时间
        结果.append({
            "束宽": 束宽,
            "预测摆放时间秒": 规划结果.simplified_cost_seconds,
            "搜索耗时秒": 规划结果.statistics.elapsed_seconds,
            "规划端到端耗时秒": 端到端耗时,
            "展开父节点数": 规划结果.statistics.expanded_parent_count,
            "生成子节点数": 规划结果.statistics.generated_child_count,
            "峰值保留节点数": 规划结果.statistics.peak_retained_node_count,
            "目标序列": list(规划结果.target_sequence),
            "实体序列": list(规划结果.source_sequence),
        })
        print(
            f"束宽={束宽:5d}  预测={规划结果.simplified_cost_seconds:8.3f} 秒  "
            f"搜索={规划结果.statistics.elapsed_seconds:8.3f} 秒  "
            f"节点={规划结果.statistics.generated_child_count}"
        )

    参考 = next((item for item in 结果 if item["束宽"] == 5000), 结果[-1])
    for item in 结果:
        item["相对K5000预测时间差秒"] = (
            item["预测摆放时间秒"] - 参考["预测摆放时间秒"]
        )
        item["相对K5000预测时间差百分比"] = (
            100.0
            * item["相对K5000预测时间差秒"]
            / 参考["预测摆放时间秒"]
        )
    文档 = {
        "生成时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "说明": "基础盘面、同一组合成开环位姿；仅比较 Python V1 束宽",
        "束宽结果说明": (
            "普通 Beam 只按当前前缀成本剪枝，前缀成本不是剩余成本下界；"
            "因此更宽束的最终结果不保证单调优于窄束，K=5000 仅作为本次对照。"
        ),
        "束宽列表": 束宽列表,
        "结果": 结果,
    }
    输出路径 = 脚本目录 / 结果文件名
    with 输出路径.open("w", encoding="utf-8") as 文件:
        json.dump(文档, 文件, ensure_ascii=False, indent=2)
        文件.write("\n")
    print(f"中文基准结果已写入：{输出路径}")


if __name__ == "__main__":
    运行基准()
