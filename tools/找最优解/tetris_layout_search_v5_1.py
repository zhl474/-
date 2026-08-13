#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14x10 俄罗斯方块 260 分代表性盘面库生成器 V5.1

V5.1 是 V5 的性能修正版，求解模型和输出数据格式不变。
主要修复：V5 在每个 signature 返回、每次 heartbeat、以及构造 progress snapshot 时
反复扫描全部 signature 的 .json/.meta/.done 文件；当 top-signatures=7000 时会产生
近似 O(N^2) 的大量小文件读取。

V5.1 改为：
- 启动时完整扫描一次已有结果；
- 运行期间只根据 future 返回的 meta 在主进程内存中增量维护统计；
- heartbeat 完全不扫描 signature 文件；
- 结束时完整扫描一次，用磁盘真实状态校正最终汇总；
- library 导出仍在结束时单次读取全部已找到结果。

求解逻辑完全复用同目录下的 tetris_layout_search_v5.py，因此：
- CP-SAT 模型、260 分硬约束、support 剪枝不变；
- signature 定义与概率排序不变；
- K 个四区粗分布不同的布局策略不变；
- resume/shard/输出目录结构与 V5 兼容；
- 已有 V5 输出目录可直接使用 --resume 继续。

使用前请把本文件与 tetris_layout_search_v5.py 放在同一目录。

冒烟测试：
    python3 tetris_layout_search_v5_1.py \
        --output-dir v5_1_smoke \
        --top-signatures 20 \
        --layouts-per-signature 1 \
        --jobs 2 --cp-workers 2 \
        --task-time 20 --heartbeat 5

注意：V5.1 只修运行期统计 I/O 性能；关于 K=1/K=3、多轮 restart、jobs 数量等
长跑策略不在本次修改范围内。
"""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

try:
    import tetris_layout_search_v5 as v5
except ImportError as e:
    raise SystemExit(
        "找不到 tetris_layout_search_v5.py。请把 V5.1 与 V5 放在同一目录后再运行。"
    ) from e


PROGRAM_VERSION = "5.1"


# =========================
# 1. 运行期内存状态
# =========================


def state_from_disk(task: v5.SignatureTask) -> Dict[str, object]:
    """启动/最终校正时读取一次磁盘状态。运行中不要调用。"""
    st = v5.scan_task_state(task)
    return {
        "done": bool(st["done"]),
        "layouts": int(st["layouts"]),
        "covered": bool(st["covered"]),
        "status": str(st["status"]),
        "elapsed_s": st.get("elapsed_s"),
    }


def state_from_meta(
    old: Dict[str, object],
    meta: Dict[str, object],
) -> Dict[str, object]:
    """用一个已返回 future 的 meta 更新该 signature 的内存状态。"""
    layouts = int(meta.get("layouts", old.get("layouts", 0)))
    complete = bool(meta.get("complete", False)) or bool(meta.get("skipped", False))
    return {
        "done": complete,
        "layouts": layouts,
        "covered": layouts > 0,
        "status": str(meta.get("status", old.get("status", "UNKNOWN"))),
        "elapsed_s": meta.get("elapsed_s", old.get("elapsed_s")),
    }


def aggregate_states(
    tasks: Sequence[v5.SignatureTask],
    states: Dict[int, Dict[str, object]],
) -> Tuple[int, int, int, int]:
    """只遍历内存字典，不做文件 I/O。"""
    done_total = 0
    covered_total = 0
    layouts_total = 0
    covered_weight = 0
    for task in tasks:
        st = states[task.global_id]
        if bool(st["done"]):
            done_total += 1
        n = int(st["layouts"])
        layouts_total += n
        if n > 0:
            covered_total += 1
            covered_weight += task.weight
    return done_total, covered_total, layouts_total, covered_weight


def apply_returned_meta(
    task: v5.SignatureTask,
    meta: Dict[str, object],
    states: Dict[int, Dict[str, object]],
    *,
    done_total: int,
    covered_total: int,
    layouts_total: int,
    covered_weight: int,
) -> Tuple[int, int, int, int]:
    """O(1) 增量更新全局计数。"""
    old = states[task.global_id]
    new = state_from_meta(old, meta)

    old_done = 1 if bool(old["done"]) else 0
    new_done = 1 if bool(new["done"]) else 0
    done_total += new_done - old_done

    old_layouts = int(old["layouts"])
    new_layouts = int(new["layouts"])
    layouts_total += new_layouts - old_layouts

    old_covered = old_layouts > 0
    new_covered = new_layouts > 0
    if old_covered != new_covered:
        if new_covered:
            covered_total += 1
            covered_weight += task.weight
        else:
            covered_total -= 1
            covered_weight -= task.weight

    states[task.global_id] = new
    return done_total, covered_total, layouts_total, covered_weight


# =========================
# 2. 无磁盘 I/O 的 progress snapshot
# =========================


def build_fast_progress_snapshot(
    *,
    selected_total: int,
    selected_weight: int,
    done_total: int,
    covered_total: int,
    layouts_total: int,
    covered_weight: int,
    elapsed_run: float,
    new_done: int,
    running: Dict[Future, Tuple[v5.SignatureTask, float]],
    total_probability_weight: int,
) -> Dict[str, object]:
    remaining = max(0, selected_total - done_total)
    pct = 100.0 if selected_total == 0 else 100.0 * done_total / selected_total
    rate = new_done / elapsed_run if elapsed_run > 1e-9 else 0.0
    eta = remaining / rate if rate > 1e-12 else None

    running_items = []
    now = time.monotonic()
    for task, start_mono in running.values():
        running_items.append(
            {
                "global_id": task.global_id,
                "rank": task.rank,
                "left_counts": list(task.left_counts),
                "elapsed_s": max(0.0, now - start_mono),
            }
        )
    running_items.sort(key=lambda x: float(x["elapsed_s"]), reverse=True)

    return {
        "selected_total": selected_total,
        "done_total": done_total,
        "remaining": remaining,
        "progress_pct": pct,
        "covered_signatures": covered_total,
        "layouts_total": layouts_total,
        "elapsed_run_s": elapsed_run,
        "new_done_this_run": new_done,
        "throughput_tasks_per_s": rate,
        "eta_s": eta,
        "selected_probability_mass_global": selected_weight / total_probability_weight,
        "covered_probability_mass_global": covered_weight / total_probability_weight,
        "running": running_items,
    }


def write_progress(
    path: Path,
    snapshot: Dict[str, object],
    *,
    shard_count: int,
    shard_index: int,
) -> None:
    v5.atomic_write_json(
        path,
        {
            "version": 5,
            "program_version": PROGRAM_VERSION,
            "progress_mode": "in_memory_incremental",
            "shard_count": shard_count,
            "shard_index": shard_index,
            **snapshot,
        },
    )


# =========================
# 3. main：只重写 V5 的调度/统计层
# =========================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="14x10、34块、13个四色完整行的260分代表性盘面库生成器 V5.1"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("layouts_260_v5"),
        help="输出目录，默认 layouts_260_v5；与V5格式兼容",
    )
    parser.add_argument(
        "--left-count",
        type=int,
        default=17,
        help="目标盘 LEFT 方块数。V5/V5.1 当前要求左右相等，因此默认/建议17",
    )
    parser.add_argument(
        "--top-signatures",
        type=int,
        default=7000,
        help="按随机组合权重只跑前 N 个 signature；0=全部。默认7000",
    )
    parser.add_argument(
        "--signature",
        type=v5.parse_signature,
        default=None,
        help="仅调试指定 signature，例如 2,3,2,3,2,3,2；设置后忽略 --top-signatures",
    )
    parser.add_argument(
        "--layouts-per-signature",
        type=int,
        default=3,
        help="每个左右 signature 保存多少个粗空间分布不同的盘面，默认3",
    )
    parser.add_argument(
        "--task-time",
        type=float,
        default=60.0,
        help="每个 signature 本次运行总时间上限秒；0=不限时。默认60",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="外层并行 signature 进程数，默认8",
    )
    parser.add_argument(
        "--cp-workers",
        type=int,
        default=1,
        help="每个 CP-SAT Solve 内部 worker 数，默认1；总并行约为 jobs*cp-workers",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="按稳定 global_id 模分成多少个互斥 shard，默认1",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="当前机器负责哪个 shard，0-based，默认0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="CP-SAT 基础随机种子，每个 signature 自动派生，默认1",
    )
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=30.0,
        help="全局进度打印周期秒，默认30",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="跳过已完成 signature；timeout 任务保留已有布局并继续补",
    )
    parser.add_argument(
        "--store-order-hint",
        action="store_true",
        help="每个布局额外保存一个合法 order_hint；默认不保存",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="打印子进程 OR-Tools 搜索日志；并行时会交错",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="只扫描当前筛选/shard已有结果并打印状态，不启动求解",
    )
    args = parser.parse_args()

    if args.left_count * 2 != v5.TOTAL_PIECES:
        raise SystemExit(
            f"V5.1 当前实现要求目标左右数量相等，所以 --left-count 必须为 "
            f"{v5.TOTAL_PIECES // 2}"
        )
    if args.top_signatures < 0:
        raise SystemExit("--top-signatures 不能为负")
    if args.layouts_per_signature < 1:
        raise SystemExit("--layouts-per-signature 必须 >=1")
    if args.task_time < 0:
        raise SystemExit("--task-time 不能为负")
    if args.jobs < 1:
        raise SystemExit("--jobs 必须 >=1")
    if args.cp_workers < 1:
        raise SystemExit("--cp-workers 必须 >=1")
    if args.shard_count < 1:
        raise SystemExit("--shard-count 必须 >=1")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index 必须满足 0 <= index < shard-count")
    if args.heartbeat <= 0:
        raise SystemExit("--heartbeat 必须 >0")

    specs, total_probability_weight = v5.generate_signature_specs(args.left_count)
    total_signature_count = len(specs)

    if args.signature is not None:
        if sum(args.signature) != args.left_count:
            raise SystemExit(
                f"--signature 总和必须等于 --left-count={args.left_count}"
            )
        matches = [s for s in specs if s.left_counts == args.signature]
        if not matches:
            raise SystemExit("指定 signature 不在理论空间中")
        selected_specs = matches
    else:
        selected_specs = (
            specs if args.top_signatures == 0 else specs[: args.top_signatures]
        )

    # 与 V5 完全一致：top-N 在 shard 前执行。
    selected_mass_before_shard = (
        sum(s.weight for s in selected_specs) / total_probability_weight
    )
    selected_specs = [
        s
        for s in selected_specs
        if s.global_id % args.shard_count == args.shard_index
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    placements = v5.build_placements()
    v5.write_catalog(args.output_dir / "placement_catalog.json", placements)

    tasks = [
        v5.SignatureTask(
            global_id=s.global_id,
            rank=s.rank,
            left_counts=s.left_counts,
            weight=s.weight,
            probability=s.probability,
            total_probability_weight=total_probability_weight,
            output_dir=str(args.output_dir),
            task_time=args.task_time,
            layouts_per_signature=args.layouts_per_signature,
            cp_workers=args.cp_workers,
            seed=args.seed + s.global_id * 1009,
            log_search=args.log,
            resume=args.resume,
            store_order_hint=args.store_order_hint,
        )
        for s in selected_specs
    ]
    tasks.sort(key=lambda t: t.rank)

    selected_total = len(tasks)
    selected_weight = sum(t.weight for t in tasks)

    shard_tag = f"shard{args.shard_index:02d}of{args.shard_count:02d}"
    progress_path = args.output_dir / f"progress_{shard_tag}.json"
    summary_path = args.output_dir / f"summary_{shard_tag}.json"
    library_path = args.output_dir / f"library_{shard_tag}.jsonl"

    print("V5.1：V5求解逻辑 + 内存增量进度统计；运行中不再全目录扫描。", flush=True)
    print(f"候选 placement 数量：{len(placements)}", flush=True)
    print(
        f"理论左右 signature：{total_signature_count}（sum(left)={args.left_count}）",
        flush=True,
    )
    print(
        f"top-N 在 shard 前理论概率覆盖：{100*selected_mass_before_shard:.3f}%",
        flush=True,
    )
    print(
        f"当前 shard：{len(tasks)} signatures, shard={args.shard_index}/{args.shard_count}",
        flush=True,
    )
    print(
        f"每 signature 目标布局数 K={args.layouts_per_signature}; "
        f"外层 jobs={args.jobs}; CP workers={args.cp_workers}",
        flush=True,
    )
    print(
        f"单 signature 本次时间："
        f"{'不限时' if args.task_time == 0 else f'{args.task_time:g}s'}; "
        f"heartbeat={args.heartbeat:g}s",
        flush=True,
    )

    # ---------- 唯一一次启动扫描 ----------
    print("[SCAN] 启动时扫描已有结果……", flush=True)
    states: Dict[int, Dict[str, object]] = {}
    pending: List[v5.SignatureTask] = []
    for task in tasks:
        st = state_from_disk(task)
        states[task.global_id] = st
        if not bool(st["done"]):
            pending.append(task)

    done_total, covered_total, layouts_total, covered_weight = aggregate_states(
        tasks, states
    )

    print(
        f"启动前：done={done_total}/{len(tasks)}, covered={covered_total}, "
        f"layouts={layouts_total}, pending={len(pending)}",
        flush=True,
    )

    initial_done_total = done_total
    start_run = time.monotonic()
    initial_snapshot = build_fast_progress_snapshot(
        selected_total=selected_total,
        selected_weight=selected_weight,
        done_total=done_total,
        covered_total=covered_total,
        layouts_total=layouts_total,
        covered_weight=covered_weight,
        elapsed_run=0.0,
        new_done=0,
        running={},
        total_probability_weight=total_probability_weight,
    )
    v5.print_snapshot(initial_snapshot, args.jobs)
    write_progress(
        progress_path,
        initial_snapshot,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )

    if args.status_only:
        exported = v5.export_library(tasks, library_path)
        print(f"--status-only：不启动求解；当前导出 layouts={exported}", flush=True)
        print(f"library: {library_path.resolve()}", flush=True)
        return

    errors = 0
    returned = 0
    queue: Deque[v5.SignatureTask] = deque(pending)
    running: Dict[Future, Tuple[v5.SignatureTask, float]] = {}

    def submit_one(executor: ProcessPoolExecutor) -> None:
        if not queue:
            return
        task = queue.popleft()
        fut = executor.submit(v5.solve_signature_task, task)
        running[fut] = (task, time.monotonic())

    next_heartbeat = time.monotonic() + args.heartbeat

    if pending:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(pending))) as ex:
            for _ in range(min(args.jobs, len(pending))):
                submit_one(ex)

            while running:
                now = time.monotonic()
                timeout = max(0.0, next_heartbeat - now)
                done_futs, _ = wait(
                    list(running.keys()),
                    timeout=timeout,
                    return_when=FIRST_COMPLETED,
                )

                for fut in done_futs:
                    task, _start = running.pop(fut)
                    returned += 1
                    try:
                        meta = fut.result()
                        n = int(meta.get("layouts", 0))
                        complete = bool(meta.get("complete", False))
                        status = str(meta.get("status", "?"))
                        elapsed_s = float(meta.get("elapsed_s", 0.0))

                        # V5.1 核心：O(1) 内存增量更新，不再扫描所有 task 文件。
                        (
                            done_total,
                            covered_total,
                            layouts_total,
                            covered_weight,
                        ) = apply_returned_meta(
                            task,
                            meta,
                            states,
                            done_total=done_total,
                            covered_total=covered_total,
                            layouts_total=layouts_total,
                            covered_weight=covered_weight,
                        )

                        print(
                            f"[RETURN id={task.global_id:05d} rank={task.rank:05d}] "
                            f"left={','.join(map(str, task.left_counts))} "
                            f"status={status} layouts={n}/{task.layouts_per_signature} "
                            f"complete={complete} time={v5.fmt_duration(elapsed_s)}",
                            flush=True,
                        )
                    except Exception as e:
                        errors += 1
                        print(
                            f"[ERROR id={task.global_id:05d} rank={task.rank}] {e!r}",
                            flush=True,
                        )

                    submit_one(ex)

                now = time.monotonic()
                if now >= next_heartbeat or not running:
                    elapsed_run = now - start_run
                    snapshot = build_fast_progress_snapshot(
                        selected_total=selected_total,
                        selected_weight=selected_weight,
                        done_total=done_total,
                        covered_total=covered_total,
                        layouts_total=layouts_total,
                        covered_weight=covered_weight,
                        elapsed_run=elapsed_run,
                        new_done=max(0, done_total - initial_done_total),
                        running=running,
                        total_probability_weight=total_probability_weight,
                    )
                    v5.print_snapshot(snapshot, args.jobs)
                    write_progress(
                        progress_path,
                        snapshot,
                        shard_count=args.shard_count,
                        shard_index=args.shard_index,
                    )
                    while next_heartbeat <= now:
                        next_heartbeat += args.heartbeat

    elapsed_run = time.monotonic() - start_run

    # ---------- 唯一一次结束扫描 ----------
    # 用磁盘真实结果覆盖内存缓存，处理极少数异常/中断边界情况。
    print("[SCAN] 运行结束，做一次最终磁盘校正……", flush=True)
    status_counts = Counter()
    for task in tasks:
        st = state_from_disk(task)
        states[task.global_id] = st
        status_counts[str(st["status"])] += 1

    done_total, covered_total, layouts_total, covered_weight = aggregate_states(
        tasks, states
    )

    final_snapshot = build_fast_progress_snapshot(
        selected_total=selected_total,
        selected_weight=selected_weight,
        done_total=done_total,
        covered_total=covered_total,
        layouts_total=layouts_total,
        covered_weight=covered_weight,
        elapsed_run=elapsed_run,
        new_done=max(0, done_total - initial_done_total),
        running={},
        total_probability_weight=total_probability_weight,
    )

    exported = v5.export_library(tasks, library_path)
    summary = {
        "version": 5,
        "program_version": PROGRAM_VERSION,
        "progress_mode": "in_memory_incremental",
        "left_count": args.left_count,
        "theoretical_signatures": total_signature_count,
        "top_signatures": args.top_signatures,
        "selected_probability_mass_before_shard": selected_mass_before_shard,
        "layouts_per_signature": args.layouts_per_signature,
        "task_time": args.task_time,
        "jobs": args.jobs,
        "cp_workers": args.cp_workers,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "errors_this_run": errors,
        "returned_this_run": returned,
        "status_counts": dict(status_counts),
        "library_records": exported,
        **final_snapshot,
    }
    v5.atomic_write_json(summary_path, summary)
    write_progress(
        progress_path,
        final_snapshot,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )

    print("\n===== V5.1 汇总 =====", flush=True)
    v5.print_snapshot(final_snapshot, args.jobs)
    print(f"status_counts={dict(status_counts)}", flush=True)
    print(f"library : {library_path.resolve()} ({exported} layouts)", flush=True)
    print(f"progress: {progress_path.resolve()}", flush=True)
    print(f"summary : {summary_path.resolve()}", flush=True)
    print(f"errors={errors}", flush=True)

    if done_total == len(tasks):
        print("当前筛选/shard 的代表盘面搜索已全部结束。", flush=True)
    else:
        print(
            "仍有 timeout/未完成 signature；可增大 --task-time 后使用 --resume 继续。",
            flush=True,
        )


if __name__ == "__main__":
    main()
