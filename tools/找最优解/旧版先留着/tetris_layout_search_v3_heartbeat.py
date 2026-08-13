#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14x10 俄罗斯方块基础任务 260 分版型枚举器 V3 + Heartbeat

硬约束：
- 共放 34 块；
- 七类库存各最多 5 块：固定一个类别用 4 块，其余 6 类各用 5 块；
- 14 行中恰好 13 行完整；唯一不完整行恰好占 6 格、空 4 格；
- 13 个完整行每行都至少出现 4 个类别/颜色；
- 不重叠、不越界；
- 每个非底层方块至少有一个底部格在下一行受到已选方块支撑。

V3 不优化分数、不 fallback：只枚举满足上述硬约束的 260 分版型。

全集拆分：
- gap_row: 14 种
- missing_category: 7 种
- 共 14*7=98 个互斥分区

因为 OR-Tools CP-SAT 的 enumerate_all_solutions 模式要求单个 solver 使用 1 worker，
所以本脚本采用“外层多进程 + 内层单 worker”的方式并行枚举。

新增心跳：
- --heartbeat 30：默认每 30 秒打印一次每个正在运行分区的状态；
- 分区开始时打印 [START]；
- 运行中打印 [HEARTBEAT]：elapsed / solutions / 输出文件大小；
- 分区结束时打印 [DONE]；
- 主进程同时打印 [GLOBAL]：完成分区数 / 总分区数 / 总运行时间。

输出：
- placement_catalog.json
- partitions/*.jsonl                完整枚举完成的分区
- partitions/*.partial.jsonl        尚未完整结束的分区
- partitions/*.meta.json
- partitions/*.done                 只有完整枚举完成或证明无解时才存在
- summary.json

只有目标范围内所有分区均有 .done，且 --max-solutions-per-partition=0，
才可以声称该范围已完整枚举。

依赖：
    python3 -m pip install ortools

双机示例：
电脑 A：
    python3 tetris_layout_search_v3_heartbeat.py \
      --gap-row 1,3,5,7,9,11,13 --missing-category all \
      --jobs 8 --partition-time 0 --max-solutions-per-partition 0 \
      --heartbeat 30 --output-dir layouts_260_A --resume

电脑 B：
    python3 tetris_layout_search_v3_heartbeat.py \
      --gap-row 2,4,6,8,10,12,14 --missing-category all \
      --jobs 8 --partition-time 0 --max-solutions-per-partition 0 \
      --heartbeat 30 --output-dir layouts_260_B --resume
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ortools.sat.python import cp_model


ROWS = 14
COLS = 10
TOTAL_PIECES = 34
COPIES_PER_CATEGORY = 5

CATEGORIES = [
    "L_yellow",
    "L_blue",
    "z_green",
    "z_blue",
    "T",
    "square",
    "line",
]

BASE_SHAPES: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "L_yellow": ((0, 0), (1, 0), (2, 0), (2, 1)),
    "L_blue": ((0, 0), (1, 0), (2, 0), (0, 1)),
    "z_green": ((0, 0), (1, 0), (1, 1), (2, 1)),
    "z_blue": ((1, 0), (2, 0), (0, 1), (1, 1)),
    "T": ((0, 0), (1, 0), (2, 0), (1, 1)),
    "square": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "line": ((0, 0), (1, 0), (2, 0), (3, 0)),
}

ANGLE_OUTPUT_MAP = {
    0: 0,
    90: 90,
    180: 180,
    -90: -90,
}


@dataclass(frozen=True)
class Orientation:
    category: str
    angle_deg: int
    cells: Tuple[Tuple[int, int], ...]
    width: int
    height: int


@dataclass(frozen=True)
class Placement:
    pid: int
    category: str
    angle_deg: int
    x0: int
    y0: int
    width: int
    height: int
    cells: Tuple[Tuple[int, int], ...]

    @property
    def center_col(self) -> float:
        return self.x0 + (self.width - 1) / 2.0

    @property
    def center_row(self) -> float:
        return self.y0 + (self.height - 1) / 2.0

    @property
    def min_row(self) -> int:
        return min(r for _, r in self.cells)

    @property
    def max_row(self) -> int:
        return max(r for _, r in self.cells)


@dataclass(frozen=True)
class PartitionTask:
    gap_row: int
    missing_category: str
    output_dir: str
    partition_time: float
    max_solutions: int
    seed: int
    log_search: bool
    flush_every: int
    resume: bool
    heartbeat: float


def normalize(cells: Iterable[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    cells = list(cells)
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    return tuple(sorted((x - min_x, y - min_y) for x, y in cells))


def rotate_ccw_90(cells: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    return normalize((-y, x) for x, y in cells)


def build_orientations() -> List[Orientation]:
    out: List[Orientation] = []
    for category in CATEGORIES:
        cur = normalize(BASE_SHAPES[category])
        seen = set()
        for angle in (0, 90, 180, -90):
            key = tuple(cur)
            if key not in seen:
                seen.add(key)
                out.append(
                    Orientation(
                        category=category,
                        angle_deg=angle,
                        cells=key,
                        width=max(x for x, _ in key) + 1,
                        height=max(y for _, y in key) + 1,
                    )
                )
            cur = rotate_ccw_90(cur)
    return out


def build_placements() -> List[Placement]:
    out: List[Placement] = []
    pid = 0
    for ori in build_orientations():
        for y0 in range(1, ROWS - ori.height + 2):
            for x0 in range(1, COLS - ori.width + 2):
                cells = tuple((x0 + dx, y0 + dy) for dx, dy in ori.cells)
                out.append(
                    Placement(
                        pid=pid,
                        category=ori.category,
                        angle_deg=ori.angle_deg,
                        x0=x0,
                        y0=y0,
                        width=ori.width,
                        height=ori.height,
                        cells=cells,
                    )
                )
                pid += 1
    return out


def bottom_profile_cells(p: Placement) -> List[Tuple[int, int]]:
    min_row_by_col: Dict[int, int] = {}
    for col, row in p.cells:
        if col not in min_row_by_col or row < min_row_by_col[col]:
            min_row_by_col[col] = row
    return sorted(min_row_by_col.items())


def build_indices(placements: Sequence[Placement]):
    cell_to_ids = defaultdict(list)
    category_to_ids = defaultdict(list)
    row_category_to_ids = defaultdict(list)

    for p in placements:
        category_to_ids[p.category].append(p.pid)
        touched_rows = set()
        for cell in p.cells:
            cell_to_ids[cell].append(p.pid)
            touched_rows.add(cell[1])
        for row in touched_rows:
            row_category_to_ids[(row, p.category)].append(p.pid)

    return cell_to_ids, category_to_ids, row_category_to_ids


def build_partition_model(
    placements: Sequence[Placement],
    gap_row: int,
    missing_category: str,
):
    if not 1 <= gap_row <= ROWS:
        raise ValueError(f"gap_row 必须在 1..{ROWS}")
    if missing_category not in CATEGORIES:
        raise ValueError(f"未知类别 {missing_category}")

    model = cp_model.CpModel()
    x = [model.new_bool_var(f"x_{p.pid}") for p in placements]
    cell_to_ids, category_to_ids, row_category_to_ids = build_indices(placements)

    # 1) 13 个完整行每格恰好覆盖一次；gap row 每格最多一次且整行总占用恰好 6 格。
    gap_occupancies = []
    for row in range(1, ROWS + 1):
        for col in range(1, COLS + 1):
            ids = cell_to_ids[(col, row)]
            occ = sum(x[i] for i in ids)
            if row == gap_row:
                model.add(occ <= 1)
                gap_occupancies.append(occ)
            else:
                model.add(occ == 1)
    model.add(sum(gap_occupancies) == 6)

    # 2) 34 块；固定本分区缺哪一类的一块。
    model.add(sum(x) == TOTAL_PIECES)
    for category in CATEGORIES:
        required = 4 if category == missing_category else 5
        model.add(sum(x[i] for i in category_to_ids[category]) == required)

    # 3) 13 个完整行每行至少 4 个类别/颜色。
    # present 变量使用双向约束，避免辅助变量自由取值导致同一几何版型重复枚举。
    for row in range(1, ROWS + 1):
        if row == gap_row:
            continue
        present_vars = []
        for category in CATEGORIES:
            y = model.new_bool_var(f"present_r{row}_{category}")
            ids = row_category_to_ids[(row, category)]
            selected_touching_row = sum(x[i] for i in ids)
            model.add(selected_touching_row >= y)
            model.add(selected_touching_row <= COPIES_PER_CATEGORY * y)
            present_vars.append(y)
        model.add(sum(present_vars) >= 4)

    # 4) 支撑硬约束：任意不接触第 1 行的已选方块，其底部轮廓至少有一格
    # 的正下方必须属于另一个已选方块。
    for p in placements:
        if p.min_row == 1:
            continue
        supporter_ids = set()
        for col, row in bottom_profile_cells(p):
            below = (col, row - 1)
            for qid in cell_to_ids.get(below, []):
                if qid != p.pid:
                    supporter_ids.add(qid)
        if supporter_ids:
            model.add(sum(x[qid] for qid in supporter_ids) >= x[p.pid])
        else:
            model.add(x[p.pid] == 0)

    return model, x


def validate_solution(
    placements: Sequence[Placement],
    selected_ids: Sequence[int],
    gap_row: int,
    missing_category: str,
) -> None:
    if len(selected_ids) != 34:
        raise RuntimeError(f"方块数错误: {len(selected_ids)} != 34")

    selected = [placements[i] for i in selected_ids]
    counts = Counter(p.category for p in selected)
    for c in CATEGORIES:
        expected = 4 if c == missing_category else 5
        if counts[c] != expected:
            raise RuntimeError(f"{c} 数量错误: {counts[c]} != {expected}")

    owner = {}
    for p in selected:
        for cell in p.cells:
            if cell in owner:
                raise RuntimeError(f"重叠: {cell}")
            owner[cell] = p.pid

    if len(owner) != 136:
        raise RuntimeError(f"覆盖格数错误: {len(owner)} != 136")

    for row in range(1, ROWS + 1):
        occupied = sum((col, row) in owner for col in range(1, COLS + 1))
        expected = 6 if row == gap_row else 10
        if occupied != expected:
            raise RuntimeError(f"row {row} 占用 {occupied} != {expected}")

        if row != gap_row:
            cats = {
                p.category
                for p in selected
                if any(r == row for _, r in p.cells)
            }
            if len(cats) < 4:
                raise RuntimeError(f"row {row} 只有 {len(cats)} 类")

    for p in selected:
        if p.min_row == 1:
            continue
        supported = False
        for col, row in bottom_profile_cells(p):
            qid = owner.get((col, row - 1))
            if qid is not None and qid != p.pid:
                supported = True
                break
        if not supported:
            raise RuntimeError(
                f"支撑失败: pid={p.pid} {p.category} "
                f"center=({p.center_col},{p.center_row})"
            )


def partition_stem(gap_row: int, missing_category: str) -> str:
    return f"gap{gap_row:02d}_missing_{missing_category.replace('/', '_')}"


def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def human_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class JsonlSolutionWriter(cp_model.CpSolverSolutionCallback):
    def __init__(
        self,
        x: Sequence[cp_model.IntVar],
        file_obj,
        placements: Sequence[Placement],
        gap_row: int,
        missing_category: str,
        max_solutions: int,
        flush_every: int,
    ) -> None:
        super().__init__()
        self._x = x
        self._f = file_obj
        self._placements = placements
        self._gap_row = gap_row
        self._missing_category = missing_category
        self._max_solutions = max_solutions
        self._flush_every = max(1, flush_every)
        self.solution_count = 0

    def on_solution_callback(self) -> None:
        ids = [i for i, var in enumerate(self._x) if self.boolean_value(var)]

        # 第一解和之后每 10000 解独立校验一次，避免 callback 每解都做重活。
        if self.solution_count == 0 or self.solution_count % 10000 == 0:
            validate_solution(
                self._placements,
                ids,
                self._gap_row,
                self._missing_category,
            )

        self._f.write(json.dumps({"ids": ids}, separators=(",", ":")) + "\n")
        self.solution_count += 1

        if self.solution_count % self._flush_every == 0:
            self._f.flush()

        if self._max_solutions > 0 and self.solution_count >= self._max_solutions:
            self._f.flush()
            self.stop_search()


def write_catalog(path: Path, placements: Sequence[Placement]) -> None:
    payload = {
        "version": "3-heartbeat",
        "rows": ROWS,
        "cols": COLS,
        "angle_output_map": ANGLE_OUTPUT_MAP,
        "placements": [
            {
                "id": p.pid,
                "category": p.category,
                "angle_deg_internal": p.angle_deg,
                "angle_deg": ANGLE_OUTPUT_MAP[p.angle_deg],
                "col": p.center_col,
                "row": p.center_row,
                "x0": p.x0,
                "y0": p.y0,
                "width": p.width,
                "height": p.height,
                "cells": [list(c) for c in p.cells],
            }
            for p in placements
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def solve_partition(task: PartitionTask) -> Dict[str, object]:
    out_dir = Path(task.output_dir)
    part_dir = out_dir / "partitions"
    part_dir.mkdir(parents=True, exist_ok=True)

    stem = partition_stem(task.gap_row, task.missing_category)
    final_path = part_dir / f"{stem}.jsonl"
    partial_path = part_dir / f"{stem}.partial.jsonl"
    meta_path = part_dir / f"{stem}.meta.json"
    done_path = part_dir / f"{stem}.done"

    if task.resume and done_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {
                "gap_row": task.gap_row,
                "missing_category": task.missing_category,
                "status": "SKIPPED_DONE",
                "complete": True,
                "solutions": 0,
            }
        meta["skipped"] = True
        print(
            f"[SKIP  {stem}] 已有 .done，跳过完整分区",
            flush=True,
        )
        return meta

    placements = build_placements()
    model, x = build_partition_model(
        placements,
        gap_row=task.gap_row,
        missing_category=task.missing_category,
    )

    solver = cp_model.CpSolver()
    # all-solutions 模式下单个 solver 固定 1 worker；并行由外层分区完成。
    solver.parameters.num_workers = 1
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.random_seed = task.seed
    solver.parameters.log_search_progress = task.log_search
    if task.partition_time > 0:
        solver.parameters.max_time_in_seconds = task.partition_time

    if partial_path.exists():
        partial_path.unlink()

    start = time.monotonic()
    stop_heartbeat = threading.Event()
    heartbeat_thread: Optional[threading.Thread] = None

    print(
        f"[START {stem}] pid={os.getpid()} seed={task.seed} "
        f"limit={'∞' if task.partition_time == 0 else f'{task.partition_time:g}s'}",
        flush=True,
    )

    with partial_path.open("w", encoding="utf-8", buffering=1024 * 1024) as f:
        cb = JsonlSolutionWriter(
            x=x,
            file_obj=f,
            placements=placements,
            gap_row=task.gap_row,
            missing_category=task.missing_category,
            max_solutions=task.max_solutions,
            flush_every=task.flush_every,
        )

        if task.heartbeat > 0:
            def heartbeat_loop() -> None:
                # Event.wait() 让结束时无需等满一个 heartbeat 周期。
                while not stop_heartbeat.wait(task.heartbeat):
                    elapsed = time.monotonic() - start
                    try:
                        size = partial_path.stat().st_size
                    except FileNotFoundError:
                        size = 0
                    print(
                        f"[HEARTBEAT {stem}] "
                        f"elapsed={human_duration(elapsed)} "
                        f"solutions={cb.solution_count:,} "
                        f"file={human_bytes(size)}",
                        flush=True,
                    )

            heartbeat_thread = threading.Thread(
                target=heartbeat_loop,
                name=f"heartbeat-{stem}",
                daemon=True,
            )
            heartbeat_thread.start()

        try:
            status = solver.solve(model, cb)
            f.flush()
        finally:
            stop_heartbeat.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=max(1.0, min(task.heartbeat, 2.0)))

    elapsed = time.monotonic() - start
    status_name = solver.status_name(status)
    complete = status in (cp_model.OPTIMAL, cp_model.INFEASIBLE)

    if complete:
        os.replace(partial_path, final_path)
        done_path.write_text(status_name + "\n", encoding="utf-8")
    else:
        # 未完整枚举时不要留下没有 .done 的旧 final，避免误判为完整分区。
        if final_path.exists() and not done_path.exists():
            final_path.unlink()

    output_path = final_path if complete else partial_path
    try:
        output_size = output_path.stat().st_size
    except FileNotFoundError:
        output_size = 0

    meta = {
        "gap_row": task.gap_row,
        "missing_category": task.missing_category,
        "status": status_name,
        "complete": complete,
        "solutions": cb.solution_count,
        "elapsed_s": elapsed,
        "seed": task.seed,
        "partition_time": task.partition_time,
        "max_solutions": task.max_solutions,
        "output": str(output_path),
        "output_bytes": output_size,
        "skipped": False,
    }
    tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_meta, meta_path)

    print(
        f"[DONE  {stem}] status={status_name} complete={complete} "
        f"solutions={cb.solution_count:,} "
        f"elapsed={human_duration(elapsed)} file={human_bytes(output_size)}",
        flush=True,
    )
    return meta


def parse_missing_categories(value: str) -> List[str]:
    if value.lower() == "all":
        return list(CATEGORIES)
    items = [x.strip() for x in value.split(",") if x.strip()]
    bad = [x for x in items if x not in CATEGORIES]
    if bad:
        raise argparse.ArgumentTypeError(
            f"未知类别 {bad}; 可选: {', '.join(CATEGORIES)} 或 all"
        )
    return items


def parse_gap_rows(value: str) -> List[int]:
    if value.lower() == "all":
        return list(range(1, ROWS + 1))
    try:
        rows = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError("--gap-row 必须是 all 或逗号分隔整数") from e
    bad = [r for r in rows if not 1 <= r <= ROWS]
    if bad:
        raise argparse.ArgumentTypeError(f"gap row 越界: {bad}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="14x10、34块、13个四色完整行的260分版型全集枚举器 V3 + heartbeat"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("layouts_260"),
        help="输出目录，默认 layouts_260",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="外层并行分区进程数；每个分区内部固定1个CP-SAT worker。默认8",
    )
    parser.add_argument(
        "--partition-time",
        type=float,
        default=0.0,
        help="每个(gap_row,missing_category)分区时间上限秒；0=不限时",
    )
    parser.add_argument(
        "--max-solutions-per-partition",
        type=int,
        default=0,
        help="每分区最多保存多少解；0=不限制。非0时不能声称完成全集",
    )
    parser.add_argument(
        "--gap-row",
        default="all",
        help="只枚举指定不完整行，如 14 或 1,3,5；默认 all",
    )
    parser.add_argument(
        "--missing-category",
        default="all",
        help="只枚举指定缺1块类别，如 square 或 square,T；默认 all",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="基础随机种子；不同分区自动偏移",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=100,
        help="每多少个解 flush 一次 jsonl，默认100；心跳查看文件增长更及时",
    )
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=30.0,
        help="心跳输出周期秒，默认30；0=关闭心跳",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="跳过已有 .done 的完整分区；未完成分区会从头重新枚举",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="打印各子进程 OR-Tools 日志；多进程时会交错，正式长跑通常不建议",
    )
    args = parser.parse_args()

    if args.jobs < 1:
        raise SystemExit("--jobs 必须 >=1")
    if args.partition_time < 0:
        raise SystemExit("--partition-time 不能为负")
    if args.max_solutions_per_partition < 0:
        raise SystemExit("--max-solutions-per-partition 不能为负")
    if args.flush_every < 1:
        raise SystemExit("--flush-every 必须 >=1")
    if args.heartbeat < 0:
        raise SystemExit("--heartbeat 不能为负")

    gap_rows = parse_gap_rows(args.gap_row)
    missing_categories = parse_missing_categories(args.missing_category)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    placements = build_placements()
    write_catalog(args.output_dir / "placement_catalog.json", placements)

    tasks: List[PartitionTask] = []
    k = 0
    for gap_row in gap_rows:
        for missing in missing_categories:
            tasks.append(
                PartitionTask(
                    gap_row=gap_row,
                    missing_category=missing,
                    output_dir=str(args.output_dir),
                    partition_time=args.partition_time,
                    max_solutions=args.max_solutions_per_partition,
                    seed=args.seed + k,
                    log_search=args.log,
                    flush_every=args.flush_every,
                    resume=args.resume,
                    heartbeat=args.heartbeat,
                )
            )
            k += 1

    print("V3：只枚举 260 分硬约束方案，不存在 fallback。", flush=True)
    print(f"候选 placement 数量：{len(placements)}", flush=True)
    print(f"分区数量：{len(tasks)} / 98", flush=True)
    print(f"外层并行 jobs：{args.jobs}", flush=True)
    print("每个 CP-SAT 分区内部 workers：1", flush=True)
    print(
        f"心跳：{'关闭' if args.heartbeat == 0 else f'每 {args.heartbeat:g} 秒'}",
        flush=True,
    )
    print(
        f"单分区时间限制：{'不限时' if args.partition_time == 0 else f'{args.partition_time:g} s'}",
        flush=True,
    )
    if args.max_solutions_per_partition > 0:
        print(
            f"警告：每分区最多 {args.max_solutions_per_partition} 解；"
            "这是候选池模式，不是全集枚举。",
            flush=True,
        )

    results: List[Dict[str, object]] = []
    start_all = time.monotonic()
    stop_global_heartbeat = threading.Event()
    progress_counter = {"returned": 0, "complete": 0}
    counter_lock = threading.Lock()

    def global_heartbeat_loop() -> None:
        while not stop_global_heartbeat.wait(args.heartbeat):
            with counter_lock:
                returned = progress_counter["returned"]
                complete = progress_counter["complete"]
            elapsed = time.monotonic() - start_all
            print(
                f"[GLOBAL] done={complete}/{len(tasks)} "
                f"returned={returned}/{len(tasks)} "
                f"not_returned={len(tasks) - returned} "
                f"elapsed={human_duration(elapsed)}",
                flush=True,
            )

    global_thread: Optional[threading.Thread] = None
    if args.heartbeat > 0:
        global_thread = threading.Thread(
            target=global_heartbeat_loop,
            name="global-heartbeat",
            daemon=True,
        )
        global_thread.start()

    try:
        if len(tasks) == 1 or args.jobs == 1:
            for task in tasks:
                meta = solve_partition(task)
                results.append(meta)
                with counter_lock:
                    progress_counter["returned"] += 1
                    if bool(meta.get("complete")) or bool(meta.get("skipped")):
                        progress_counter["complete"] += 1
        else:
            with ProcessPoolExecutor(max_workers=min(args.jobs, len(tasks))) as ex:
                future_to_task = {ex.submit(solve_partition, t): t for t in tasks}
                for fut in as_completed(future_to_task):
                    task = future_to_task[fut]
                    try:
                        meta = fut.result()
                    except Exception as e:
                        print(
                            f"[ERROR gap={task.gap_row} missing={task.missing_category}] {e}",
                            flush=True,
                        )
                        meta = {
                            "gap_row": task.gap_row,
                            "missing_category": task.missing_category,
                            "status": "ERROR",
                            "complete": False,
                            "solutions": 0,
                            "elapsed_s": 0.0,
                            "error": repr(e),
                        }
                    results.append(meta)
                    with counter_lock:
                        progress_counter["returned"] += 1
                        if bool(meta.get("complete")) or bool(meta.get("skipped")):
                            progress_counter["complete"] += 1
                        returned = progress_counter["returned"]
                        complete = progress_counter["complete"]
                    print(
                        f"[PROGRESS] done={complete}/{len(tasks)} "
                        f"returned={returned}/{len(tasks)} "
                        f"last={int(meta['gap_row']):02d}/{meta['missing_category']} "
                        f"status={meta['status']} solutions={int(meta.get('solutions', 0)):,}",
                        flush=True,
                    )
    finally:
        stop_global_heartbeat.set()
        if global_thread is not None:
            global_thread.join(timeout=max(1.0, min(args.heartbeat, 2.0)))

    elapsed_all = time.monotonic() - start_all
    results.sort(
        key=lambda m: (
            int(m["gap_row"]),
            CATEGORIES.index(str(m["missing_category"])),
        )
    )

    summary = {
        "version": "3-heartbeat",
        "requested_partitions": len(tasks),
        "complete_partitions": sum(
            bool(m.get("complete")) or bool(m.get("skipped")) for m in results
        ),
        "solutions_this_run": sum(
            int(m.get("solutions", 0)) for m in results if not m.get("skipped")
        ),
        "elapsed_s": elapsed_all,
        "all_requested_partitions_complete": all(
            bool(m.get("complete")) or bool(m.get("skipped")) for m in results
        ),
        "results": results,
    }
    summary_path = args.output_dir / "summary.json"
    tmp = summary_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, summary_path)

    print("\n===== V3 汇总 =====", flush=True)
    print(f"本次耗时：{human_duration(elapsed_all)}", flush=True)
    print(f"本次新枚举解数：{summary['solutions_this_run']:,}", flush=True)
    print(
        f"请求分区完整结束：{summary['complete_partitions']}/{summary['requested_partitions']}",
        flush=True,
    )
    print(f"汇总：{summary_path.resolve()}", flush=True)

    if summary["all_requested_partitions_complete"] and args.max_solutions_per_partition == 0:
        print("当前请求范围已被完整枚举。", flush=True)
        if len(gap_rows) == ROWS and len(missing_categories) == len(CATEGORIES):
            print("98/98 全部分区完成：可以声称所有 260 分版型已枚举完毕。", flush=True)
    else:
        print(
            "当前结果只是 260 分候选库的一部分；只有所有目标分区完整结束后，"
            "才能声称枚举了全集。",
            flush=True,
        )


if __name__ == "__main__":
    main()
