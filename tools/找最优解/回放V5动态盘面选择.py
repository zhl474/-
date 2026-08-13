#!/home/zhl/fr3env/fr3env/bin/python
"""回放旧任务规划报告和新 V5 动态盘面报告。

使用时直接修改下方参数，无需命令行传参。
"""

from pathlib import Path
import json
import sys
import time

import numpy as np
import yaml


# =========================
# 可直接修改的参数
# =========================
数据根目录 = Path("/home/zhl/桌面/标定数据")
回放旧任务规划报告 = True
精确回放动态盘面报告 = True
旧报告结构回放Beam = 1000
旧报告重复次数 = 2
最多回放旧报告数 = None
最多回放动态报告数 = None


脚本目录 = Path(__file__).resolve().parent
项目根目录 = 脚本目录.parents[1]
部署盘面库 = 项目根目录 / "image_process" / "config" / "v5_board_library_v1.npz"
执行配置 = 项目根目录 / "competition" / "config" / "execution.yaml"
视觉伺服配置 = 项目根目录 / "competition" / "config" / "visual_servo.yaml"
sys.path.insert(0, str(项目根目录 / "image_process"))

from image_process_lib.arm_motion_time import (  # noqa: E402
    get_default_arm_motion_time_model,
    load_arm_motion_time_model,
)
from image_process_lib.board_candidate_selector import (  # noqa: E402
    BoardCandidateSelector,
    BoardCandidateSelectorConfig,
)
from image_process_lib.dynamic_board_report import (  # noqa: E402
    deserialize_observed_block,
)
from image_process_lib.dynamic_board_runtime import sha256_file  # noqa: E402
from image_process_lib.final_board_selector import (  # noqa: E402
    FinalBoardSelector,
    FinalBoardSelectorConfig,
)
from image_process_lib.task_geometry import (  # noqa: E402
    build_stable_legal_target_sequence,
    build_support_graph,
    validate_dense_target_sequence,
)
from image_process_lib.task_planner import (  # noqa: E402
    ObservedBlock,
    PlacementTarget,
)
from image_process_lib.task_sequence_optimizer import (  # noqa: E402
    TaskSequenceOptimizerConfig,
    optimize_task_sequence,
    validate_motion_model_speeds,
)
from image_process_lib.v5_board_library import load_v5_board_library  # noqa: E402


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _limit(paths, maximum):
    paths = sorted(paths)
    return paths if maximum is None else paths[: int(maximum)]


def _old_report_optimizer_config():
    with 执行配置.open("r", encoding="utf-8") as input_file:
        execution = yaml.safe_load(input_file) or {}
    with 视觉伺服配置.open("r", encoding="utf-8") as input_file:
        servo = yaml.safe_load(input_file) or {}
    motion = execution["motion"]
    motor = execution["tool_motor"]
    return TaskSequenceOptimizerConfig(
        shooting_pose=tuple(float(value) for value in execution["shooting_pose"]),
        camera_to_sucker_offset_mm=tuple(
            float(value) for value in servo["camera_to_sucker_offset_mm"]
        ),
        pick_surface_offset_mm=float(motion["pick_surface_offset_mm"]),
        pick_approach_clearance_mm=float(motion["pick_approach_clearance_mm"]),
        motor_velocity_deg_per_sec=float(motor["velocity_deg_per_sec"]),
        initial_motor_angle_deg=float(motor["initial_angle_deg"]),
        motor_lower_margin_deg=float(motor["lower_margin_deg"]),
        motor_upper_margin_deg=float(motor["upper_margin_deg"]),
        beam_width=int(旧报告结构回放Beam),
        returned_candidate_limit=1,
        report_top_candidates=1,
    )


def _old_report_inputs(document):
    blocks = tuple(ObservedBlock(
        category=item["类别"],
        observation_pose=tuple(float(value) for value in item["观察位"]),
        detected_angle_deg=float(item["识别角度"]),
        source_id=int(item["实体ID"]),
        pick_surface_z_mm=float(item["抓取表面Z毫米"]),
        pick_surface_z_valid=True,
    ) for item in document["实体"])
    targets = tuple(PlacementTarget(
        index=int(item["目标ID"]),
        row=float(item["行"]),
        col=float(item["列"]),
        desired_angle_deg=float(item["期望角度"]),
        category=item["类别"],
        observation_pose=tuple(float(value) for value in item["观察位"]),
        cells=tuple(tuple(int(value) for value in cell) for cell in item["占用格"]),
    ) for item in document["目标"])
    return blocks, targets


def replay_old_report(path, motion_model, optimizer_config):
    """旧报告不伪造像素、格点和盘面角，只做明确标注的结构回放。"""
    document = _read_json(path)
    counts = document.get("数量", {})
    entity_count = int(counts.get("实体数", len(document.get("实体", ()))))
    target_count = int(counts.get("目标数", len(document.get("目标", ()))))
    if (entity_count, target_count) != (35, 34):
        print(
            f"[跳过] {path}：实体/目标={entity_count}/{target_count}，"
            "不是完整 35/34 报告"
        )
        return "skipped"
    blocks, raw_targets = _old_report_inputs(document)
    graph = build_support_graph(raw_targets)
    selected_target_ids = tuple(document["V1选中方案"]["目标序列"])
    target_lookup = {int(target.index): index for index, target in enumerate(raw_targets)}
    validate_dense_target_sequence(
        tuple(target_lookup[int(value)] for value in selected_target_ids),
        graph,
    )
    dense_order = build_stable_legal_target_sequence(raw_targets)
    targets = tuple(raw_targets[index] for index in dense_order)
    signatures = []
    started_at = time.perf_counter()
    for _repeat in range(int(旧报告重复次数)):
        result = optimize_task_sequence(
            blocks,
            targets,
            # 旧协议没有盘面角。0 度只用于结构/确定性回放，不宣称重现旧成本。
            board_angle_deg=0.0,
            config=optimizer_config,
            motion_model=motion_model,
        )
        signatures.append((
            result.target_sequence,
            result.source_sequence,
            result.simplified_cost_seconds,
            result.selected_servo_replay.replay_total_seconds,
        ))
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RuntimeError(f"旧报告结构回放不确定：{path}")
    print(
        f"[结构回放] {path}：Beam={旧报告结构回放Beam}，"
        f"耗时={time.perf_counter() - started_at:.3f}s，顺序确定"
    )
    return "ok"


def _dynamic_optimizer_config(runtime):
    item = runtime["task_sequence_optimizer"]
    return TaskSequenceOptimizerConfig(
        shooting_pose=tuple(item["shooting_pose"]),
        camera_to_sucker_offset_mm=tuple(item["camera_to_sucker_offset_mm"]),
        pick_surface_offset_mm=float(item["pick_surface_offset_mm"]),
        pick_approach_clearance_mm=float(item["pick_approach_clearance_mm"]),
        motor_velocity_deg_per_sec=float(item["motor_velocity_deg_per_sec"]),
        initial_motor_angle_deg=float(item["initial_motor_angle_deg"]),
        motor_lower_margin_deg=float(item["motor_lower_margin_deg"]),
        motor_upper_margin_deg=float(item["motor_upper_margin_deg"]),
        beam_width=int(runtime["comparison_beam_width"]),
        report_top_candidates=int(runtime["comparison_returned_candidates"]),
    )


def _dynamic_target_cache(document, library):
    placement_xy = np.full((library.placement_count, 2), np.nan, dtype=np.float64)
    center_by_key = {}
    for record in document["实际转换的目标中心缓存"]:
        row = float(record["行"])
        col = float(record["列"])
        pose = tuple(float(value) for value in record["TCP观察位姿"])
        center_by_key[(row, col)] = (record, pose)
        for pid in record["placement_PID"]:
            placement_xy[int(pid)] = pose[:2]
    return placement_xy, center_by_key


def _dynamic_targets(library, relaxed_result, center_by_key):
    pids = sorted({
        int(pid)
        for candidate in relaxed_result.candidates
        for pid in library.board_target_pid[candidate.board_index].flat
        if int(pid) >= 0
    })
    targets = {}
    for pid in pids:
        row = float(library.placement_row[pid])
        col = float(library.placement_col[pid])
        record, pose = center_by_key[(row, col)]
        diagnostic = record.get("高位定位诊断", {})
        targets[pid] = PlacementTarget(
            index=pid,
            row=row,
            col=col,
            desired_angle_deg=float(library.placement_yaw_clockwise_deg[pid]),
            category=library.category_names[int(library.placement_category[pid])],
            observation_pose=pose,
            cells=tuple(
                tuple(int(value) for value in cell)
                for cell in library.placement_cells[pid]
            ),
            **diagnostic,
        )
    return targets


def replay_dynamic_report(path, library):
    """使用新报告保存的原始像素/TCP/格点数据精确回放全链路。"""
    document = _read_json(path)
    if document.get("协议版本") != 1:
        raise ValueError(f"不支持的动态报告版本：{path}")
    expected_library_hash = document["文件身份"]["V5盘面库SHA256"]
    if library.source_sha256 != expected_library_hash:
        raise RuntimeError(
            f"盘面库 SHA256 不一致，拒绝伪精确回放：{path}"
        )
    manifest = document.get("唯一盘面清单")
    if manifest is None:
        print(f"[跳过] {path}：本轮没有唯一盘面结果")
        return "skipped"
    snapshot = document["高位输入快照"]
    blocks = tuple(
        deserialize_observed_block(item)
        for item in snapshot["实体"]
    )
    runtime = document["运行参数"]
    selector = BoardCandidateSelector(
        library,
        BoardCandidateSelectorConfig(
            coarse_top_k=int(runtime["coarse_top_k"]),
            final_candidate_k=int(runtime["final_candidate_k"]),
            keep_coarse_boundary_ties=bool(runtime["keep_coarse_boundary_ties"]),
        ),
    )
    started_at = time.perf_counter()
    coarse_result = selector.select_coarse(
        blocks,
        snapshot["托盘中心像素XY"],
    )
    placement_xy, center_by_key = _dynamic_target_cache(document, library)
    required = selector.required_placement_ids(coarse_result)
    if not np.all(np.isfinite(placement_xy[np.asarray(required, dtype=np.intp)])):
        raise RuntimeError(f"动态报告缺少粗筛必需的 PID TCP：{path}")
    relaxed_result = selector.select_relaxed(
        coarse_result,
        blocks,
        placement_xy,
        float(snapshot["盘面角度度"]),
    )
    reported_coarse = [
        (item["盘面ID"], int(item["N_LR"]), int(item["N_UD"]))
        for item in document["四区粗筛"]["候选"]
    ]
    actual_coarse = [
        (item.board_id, item.n_lr, item.n_ud)
        for item in coarse_result.candidates
    ]
    if reported_coarse != actual_coarse:
        raise RuntimeError(f"精确回放的粗筛排名不一致：{path}")
    reported_relaxed = [
        (item["盘面ID"], tuple(item["分数元组"]))
        for item in document["relaxed筛选"]["候选"]
    ]
    actual_relaxed = [
        (item.board_id, tuple(item.score))
        for item in relaxed_result.candidates
    ]
    if reported_relaxed != actual_relaxed:
        raise RuntimeError(f"精确回放的 relaxed 排名或分数不一致：{path}")
    motion_path = str(runtime.get("motion_model_path", "")).strip()
    motion_model = (
        load_arm_motion_time_model(motion_path)
        if motion_path
        else get_default_arm_motion_time_model()
    )
    expected_motion_hash = document["文件身份"]["标定文件SHA256"].get(
        "机械臂运动时间", ""
    )
    if expected_motion_hash and sha256_file(motion_model.source_path) != expected_motion_hash:
        raise RuntimeError(f"运动时间模型 SHA256 不一致：{path}")
    validate_motion_model_speeds(
        motion_model,
        float(runtime["arm_speed"]),
        float(runtime["pick_approach_speed"]),
    )
    final_selector = FinalBoardSelector(
        library,
        FinalBoardSelectorConfig(
            comparison_beam_width=int(runtime["comparison_beam_width"]),
            comparison_returned_candidates=int(
                runtime["comparison_returned_candidates"]
            ),
            confirmation_beam_width=int(runtime["confirmation_beam_width"]),
            confirmation_returned_candidates=int(
                runtime["confirmation_returned_candidates"]
            ),
            soft_time_budget_sec=float(runtime["soft_time_budget_sec"]),
        ),
    )
    decision = final_selector.select(
        relaxed_result,
        blocks,
        _dynamic_targets(library, relaxed_result, center_by_key),
        float(snapshot["盘面角度度"]),
        _dynamic_optimizer_config(runtime),
        motion_model,
    )
    reported_attempts = [
        (
            item["盘面ID"],
            bool(item["成功"]),
            item["简化成本秒"],
            item["舵机重放总时间秒"],
            tuple(item["PID执行顺序"]),
            tuple(item["source执行顺序"]),
        )
        for item in document["20盘比较"]
    ]
    actual_attempts = [
        (
            item.board_id,
            item.succeeded,
            item.simplified_cost_seconds,
            item.servo_replay_total_seconds,
            item.target_pid_sequence,
            item.source_id_sequence,
        )
        for item in decision.comparison_attempts
    ]
    if reported_attempts != actual_attempts:
        raise RuntimeError(f"精确回放的跨盘面比较结果不一致：{path}")
    exact_checks = {
        "board_id": decision.board_id,
        "decision_fingerprint": decision.decision_fingerprint,
        "placement_pids": list(decision.placement_pids),
        "final_pid_sequence": list(decision.final_pid_sequence),
        "final_source_to_pid": [
            {"source_id": source_id, "placement_pid": pid}
            for source_id, pid in decision.final_source_to_pid
        ],
        "comparison_simplified_cost_seconds": (
            decision.comparison_attempt.simplified_cost_seconds
        ),
        "confirmation_simplified_cost_seconds": (
            decision.confirmation_result.simplified_cost_seconds
        ),
    }
    for key, actual in exact_checks.items():
        if manifest.get(key) != actual:
            raise RuntimeError(
                f"精确回放字段 {key} 不一致："
                f"报告={manifest.get(key)!r}，回放={actual!r}"
            )
    print(
        f"[精确回放] {path}：{decision.board_id}，"
        f"耗时={time.perf_counter() - started_at:.3f}s，指纹一致"
    )
    return "ok"


def main():
    if not 数据根目录.is_dir():
        raise FileNotFoundError(f"数据根目录不存在：{数据根目录}")
    old_paths = _limit(
        数据根目录.rglob("任务规划报告.json"),
        最多回放旧报告数,
    )
    dynamic_paths = _limit(
        数据根目录.rglob("动态盘面选择报告.json"),
        最多回放动态报告数,
    )
    print(f"找到旧任务规划报告 {len(old_paths)} 份")
    print(f"找到动态盘面报告 {len(dynamic_paths)} 份")
    failures = []
    if 回放旧任务规划报告:
        motion_model = get_default_arm_motion_time_model()
        optimizer_config = _old_report_optimizer_config()
        print(
            "旧协议缺少当时速度与盘面角：结构回放使用当前运动模型"
            "和 0 度盘面，不比较历史成本数值。"
        )
        for path in old_paths:
            try:
                replay_old_report(path, motion_model, optimizer_config)
            except Exception as exc:
                failures.append((path, exc))
                print(f"[失败] {path}：{type(exc).__name__}: {exc}")
    if 精确回放动态盘面报告 and dynamic_paths:
        library = load_v5_board_library(部署盘面库)
        for path in dynamic_paths:
            try:
                replay_dynamic_report(path, library)
            except Exception as exc:
                failures.append((path, exc))
                print(f"[失败] {path}：{type(exc).__name__}: {exc}")
    if failures:
        print(f"\n回放完成，失败 {len(failures)} 份")
        return 1
    print("\n回放完成，没有失败")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
