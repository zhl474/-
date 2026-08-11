#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14x10 俄罗斯方块基础任务 260 分版型全集枚举器 V4

V4 相比 V3 的核心变化：
1) 不再只按 (gap_row, missing_category) 分成 98 个“大分区”；
2) 再把“不完整行的 4 个空格具体在哪 4 列”固定下来：C(10,4)=210 种；
3) 因此全集被严格分成 14 * 7 * 210 = 20580 个互斥微分区；
4) 每个微分区完成后立即写 .done，可真正断点续跑；
5) 主进程实时显示 completed/total、吞吐率、ETA、耗时分位数和最长运行任务；
6) 支持 --shard-count / --shard-index，把 20580 个任务无重叠分给多台电脑。

每个保存的几何版型都满足：
- 共放 34 块；
- 恰好一个类别用 4 块，其余 6 类各用 5 块；
- 14 行中恰好 13 行完整；
- 唯一不完整行恰好有 4 个指定空格；
- 13 个完整行每行至少 4 种类别/颜色；
- 不重叠、不越界；
- 存在符合比赛规则的自下而上有效拼接顺序。

全集规模：
    14 gap rows * 7 missing categories * C(10,4) = 20580 micro tasks

两台电脑推荐：
    电脑A：--shard-count 2 --shard-index 0
    电脑B：--shard-count 2 --shard-index 1

二者微分区严格互斥。以后把两个 output-dir 下的 micro/ 目录合并即可。

依赖：
    python3 -m pip install ortools

第二台电脑先开始实际枚举并估算 ETA：
    python3 tetris_layout_search_v4.py \
        --output-dir layouts_260_v4_shard1 \
        --jobs 8 --shard-count 2 --shard-index 1 \
        --task-time 0 --heartbeat 30 --resume

若随后停掉旧 V3，让第一台电脑承担另一半：
    python3 tetris_layout_search_v4.py \
        --output-dir layouts_260_v4_shard0 \
        --jobs 8 --shard-count 2 --shard-index 0 \
        --task-time 0 --heartbeat 30 --resume

说明：
- row=1 为最底行，col=1 为最左列。
- 0°/±90° 几何定义沿用 V3。
- 运行时版型选择若要真正最短运动时间，不应被这里保存的某一个固定摆放顺序绑死。
  因此 V4 默认只保存几何 ids；合法顺序只用于验证。可用 --store-order-hint 保存一个顺序提示。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from ortools.sat.python import cp_model


ROWS = 14
COLS = 10
TOTAL_PIECES = 34
COPIES_PER_CATEGORY = 5
TOTAL_EMPTY_PER_GAP_ROW = 4
EMPTY_COMBINATIONS_PER_ROW = math.comb(COLS, TOTAL_EMPTY_PER_GAP_ROW)  # 210
TOTAL_MICRO_TASKS = ROWS * 7 * EMPTY_COMBINATIONS_PER_ROW  # 20580

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
class MicroTask:
    global_id: int
    gap_row: int
    missing_category: str
    empty_cols: Tuple[int, int, int, int]
    output_dir: str
    task_time: float
    max_solutions: int
    seed: int
    log_search: bool
    flush_every: int
    resume: bool
    store_order_hint: bool


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


def build_micro_model(
    placements: Sequence[Placement],
    gap_row: int,
    missing_category: str,
    empty_cols: Sequence[int],
):
    """构造一个“空格位置已完全固定”的微分区模型。"""
    if not 1 <= gap_row <= ROWS:
        raise ValueError(f"gap_row 必须在 1..{ROWS}")
    if missing_category not in CATEGORIES:
        raise ValueError(f"未知类别 {missing_category}")
    empty_cols = tuple(sorted(empty_cols))
    if len(empty_cols) != 4 or len(set(empty_cols)) != 4:
        raise ValueError("empty_cols 必须恰好包含 4 个不同列")
    if any(c < 1 or c > COLS for c in empty_cols):
        raise ValueError("empty_cols 列号必须在 1..10")

    empty_set = set(empty_cols)
    model = cp_model.CpModel()
    x = [model.new_bool_var(f"x_{p.pid}") for p in placements]
    cell_to_ids, category_to_ids, row_category_to_ids = build_indices(placements)

    # 1) 140 个格子的占用状态完全固定：
    #    - gap_row 的 4 个指定列必须为空；
    #    - 其余 136 个格子必须恰好被覆盖一次。
    for row in range(1, ROWS + 1):
        for col in range(1, COLS + 1):
            ids = cell_to_ids[(col, row)]
            occ = sum(x[i] for i in ids)
            if row == gap_row and col in empty_set:
                model.add(occ == 0)
            else:
                model.add(occ == 1)

    # 2) 恰好 34 块；当前 missing_category 用 4 块，其余类别各 5 块。
    model.add(sum(x) == TOTAL_PIECES)
    for category in CATEGORIES:
        required = 4 if category == missing_category else 5
        model.add(sum(x[i] for i in category_to_ids[category]) == required)

    # 3) 13 个完整行都必须 >=4 类/色。
    # present 变量由“是否存在该类方块触及该行”双向唯一确定，
    # 不会让同一几何版型因辅助变量不同而被重复枚举。
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

    return model, x


def compute_legal_order(
    placements: Sequence[Placement],
    selected_ids: Sequence[int],
) -> Optional[List[int]]:
    """判断是否存在符合“底层先放、上层至少有一处已支撑”的合法顺序。"""
    selected = {pid: placements[pid] for pid in selected_ids}
    owner: Dict[Tuple[int, int], int] = {}
    for pid, p in selected.items():
        for cell in p.cells:
            owner[cell] = pid

    placed = {pid for pid, p in selected.items() if p.min_row == 1}
    order = sorted(
        placed,
        key=lambda pid: (
            selected[pid].max_row,
            selected[pid].center_col,
            selected[pid].category,
        ),
    )
    remaining = set(selected) - placed

    while remaining:
        newly_placeable = []
        for pid in remaining:
            p = selected[pid]
            ok = False
            for col, row in bottom_profile_cells(p):
                if row <= 1:
                    ok = True
                    break
                supporter = owner.get((col, row - 1))
                if supporter is not None and supporter != pid and supporter in placed:
                    ok = True
                    break
            if ok:
                newly_placeable.append(pid)

        if not newly_placeable:
            return None

        newly_placeable.sort(
            key=lambda pid: (
                selected[pid].min_row,
                selected[pid].max_row,
                selected[pid].center_col,
                selected[pid].category,
            )
        )
        for pid in newly_placeable:
            placed.add(pid)
            remaining.remove(pid)
            order.append(pid)

    return order


def validate_solution(
    placements: Sequence[Placement],
    selected_ids: Sequence[int],
    gap_row: int,
    missing_category: str,
    empty_cols: Sequence[int],
) -> None:
    if len(selected_ids) != 34:
        raise RuntimeError(f"方块数错误: {len(selected_ids)} != 34")

    selected = [placements[i] for i in selected_ids]
    counts = Counter(p.category for p in selected)
    for c in CATEGORIES:
        expected = 4 if c == missing_category else 5
        if counts[c] != expected:
            raise RuntimeError(f"{c} 数量错误: {counts[c]} != {expected}")

    owner: Dict[Tuple[int, int], int] = {}
    for p in selected:
        for cell in p.cells:
            if cell in owner:
                raise RuntimeError(f"重叠: {cell}")
            owner[cell] = p.pid

    if len(owner) != 136:
        raise RuntimeError(f"覆盖格数错误: {len(owner)} != 136")

    expected_empty = {(c, gap_row) for c in empty_cols}
    actual_empty = {
        (c, r)
        for r in range(1, ROWS + 1)
        for c in range(1, COLS + 1)
        if (c, r) not in owner
    }
    if actual_empty != expected_empty:
        raise RuntimeError(
            f"空格位置错误: actual={sorted(actual_empty)} expected={sorted(expected_empty)}"
        )

    for row in range(1, ROWS + 1):
        if row == gap_row:
            continue
        cats = {
            p.category
            for p in selected
            if any(r == row for _, r in p.cells)
        }
        if len(cats) < 4:
            raise RuntimeError(f"row {row} 只有 {len(cats)} 类")

    if compute_legal_order(placements, selected_ids) is None:
        raise RuntimeError("不存在符合规则的有效拼接顺序")


def empty_code(empty_cols: Sequence[int]) -> str:
    return "-".join(f"{c:02d}" for c in empty_cols)


def task_rel_dir(task: MicroTask) -> Path:
    safe_cat = task.missing_category.replace("/", "_")
    return Path(f"gap{task.gap_row:02d}") / f"missing_{safe_cat}"


def task_stem(task: MicroTask) -> str:
    return f"empty_{empty_code(task.empty_cols)}"


def task_paths(task: MicroTask):
    base = Path(task.output_dir) / "micro" / task_rel_dir(task)
    stem = task_stem(task)
    return {
        "dir": base,
        "final": base / f"{stem}.jsonl",
        "partial": base / f"{stem}.partial.jsonl",
        "meta": base / f"{stem}.meta.json",
        "done": base / f"{stem}.done",
    }


class JsonlSolutionWriter(cp_model.CpSolverSolutionCallback):
    def __init__(
        self,
        x: Sequence[cp_model.IntVar],
        file_obj,
        placements: Sequence[Placement],
        task: MicroTask,
    ) -> None:
        super().__init__()
        self._x = x
        self._f = file_obj
        self._placements = placements
        self._task = task
        self.raw_solution_count = 0
        self.solution_count = 0

    def on_solution_callback(self) -> None:
        self.raw_solution_count += 1
        ids = [i for i, var in enumerate(self._x) if self.value(var) == 1]

        legal_order = compute_legal_order(self._placements, ids)
        if legal_order is None:
            return

        if self.solution_count == 0:
            validate_solution(
                self._placements,
                ids,
                self._task.gap_row,
                self._task.missing_category,
                self._task.empty_cols,
            )

        record = {"ids": ids}
        if self._task.store_order_hint:
            # 这里只是一个合法顺序提示，不代表实际识别场景下运动时间最优。
            record["order_hint"] = legal_order
        self._f.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.solution_count += 1

        if self.solution_count % max(1, self._task.flush_every) == 0:
            self._f.flush()

        if (
            self._task.max_solutions > 0
            and self.solution_count >= self._task.max_solutions
        ):
            self._f.flush()
            self.stop_search()


def write_catalog(path: Path, placements: Sequence[Placement]) -> None:
    payload = {
        "version": 4,
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


def solve_micro_task(task: MicroTask) -> Dict[str, object]:
    paths = task_paths(task)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    if task.resume and paths["done"].exists():
        try:
            meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        except Exception:
            meta = {
                "global_id": task.global_id,
                "gap_row": task.gap_row,
                "missing_category": task.missing_category,
                "empty_cols": list(task.empty_cols),
                "status": "SKIPPED_DONE",
                "complete": True,
            }
        meta["skipped"] = True
        return meta

    placements = build_placements()
    model, x = build_micro_model(
        placements,
        gap_row=task.gap_row,
        missing_category=task.missing_category,
        empty_cols=task.empty_cols,
    )

    solver = cp_model.CpSolver()
    # 单个微分区使用 all-solutions 枚举，因此内部固定单 worker。
    try:
        solver.parameters.num_workers = 1
    except (AttributeError, ValueError):
        solver.parameters.num_search_workers = 1
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.random_seed = task.seed
    solver.parameters.log_search_progress = task.log_search
    if task.task_time > 0:
        solver.parameters.max_time_in_seconds = task.task_time

    # 未完成微分区无法恢复 CP-SAT 内部搜索树，因此 resume 时从头枚举该微分区。
    # 旧 partial 不是“已完成进度”，直接覆盖，避免重复解混入最终文件。
    if paths["partial"].exists():
        paths["partial"].unlink()

    start = time.monotonic()
    with paths["partial"].open("w", encoding="utf-8", buffering=1024 * 1024) as f:
        cb = JsonlSolutionWriter(x, f, placements, task)
        status = solver.solve(model, cb)
        f.flush()

    elapsed = time.monotonic() - start
    status_name = solver.status_name(status)
    complete = status in (cp_model.OPTIMAL, cp_model.INFEASIBLE)

    if complete:
        os.replace(paths["partial"], paths["final"])
    else:
        if paths["final"].exists() and not paths["done"].exists():
            paths["final"].unlink()

    meta = {
        "version": 4,
        "global_id": task.global_id,
        "gap_row": task.gap_row,
        "missing_category": task.missing_category,
        "empty_cols": list(task.empty_cols),
        "status": status_name,
        "complete": complete,
        "solutions": cb.solution_count,
        "raw_geometric_solutions": cb.raw_solution_count,
        "elapsed_s": elapsed,
        "seed": task.seed,
        "task_time": task.task_time,
        "max_solutions": task.max_solutions,
        "output": str(paths["final"] if complete else paths["partial"]),
        "skipped": False,
    }

    tmp_meta = paths["meta"].with_suffix(paths["meta"].suffix + ".tmp")
    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_meta, paths["meta"])

    # done 必须最后写；它是“这个微分区已被完整证明结束”的原子标志。
    if complete:
        paths["done"].write_text(status_name + "\n", encoding="utf-8")

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


def parse_empty_cols(value: str) -> Optional[Tuple[int, int, int, int]]:
    if value.lower() == "all":
        return None
    try:
        cols = tuple(sorted(int(x.strip()) for x in value.split(",") if x.strip()))
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "--empty-cols 必须是 all 或 4 个逗号分隔列号，例如 1,2,3,4"
        ) from e
    if len(cols) != 4 or len(set(cols)) != 4 or any(c < 1 or c > COLS for c in cols):
        raise argparse.ArgumentTypeError(
            "--empty-cols 必须恰好是 1..10 中 4 个不同列，例如 1,2,3,4"
        )
    return cols  # type: ignore[return-value]


def generate_task_specs(
    gap_rows: Sequence[int],
    missing_categories: Sequence[str],
    exact_empty_cols: Optional[Tuple[int, int, int, int]],
) -> List[Tuple[int, int, str, Tuple[int, int, int, int]]]:
    """
    返回 (global_id, gap_row, missing_category, empty_cols)。

    global_id 始终基于完整 20580 空间的固定顺序，与筛选条件无关，
    因而 --shard-count/--shard-index 在不同机器上始终一致且互斥。
    """
    wanted_rows = set(gap_rows)
    wanted_cats = set(missing_categories)
    wanted_empty = exact_empty_cols

    specs = []
    gid = 0
    all_empty = list(combinations(range(1, COLS + 1), 4))
    for gap_row in range(1, ROWS + 1):
        for missing in CATEGORIES:
            for empty_cols in all_empty:
                if (
                    gap_row in wanted_rows
                    and missing in wanted_cats
                    and (wanted_empty is None or empty_cols == wanted_empty)
                ):
                    specs.append((gid, gap_row, missing, empty_cols))
                gid += 1
    if gid != TOTAL_MICRO_TASKS:
        raise RuntimeError(f"内部任务总数错误: {gid} != {TOTAL_MICRO_TASKS}")
    return specs


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "?"
    seconds = int(round(seconds))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d{h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def safe_read_meta(task: MicroTask) -> Optional[Dict[str, object]]:
    p = task_paths(task)["meta"]
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def is_done(task: MicroTask) -> bool:
    return task_paths(task)["done"].exists()


def progress_bar(done: int, total: int, width: int = 28) -> str:
    frac = 1.0 if total == 0 else done / total
    n = min(width, max(0, int(frac * width)))
    return "[" + "#" * n + "-" * (width - n) + "]"


def build_progress_snapshot(
    *,
    selected_total: int,
    done_total: int,
    feasible_total: int,
    infeasible_total: int,
    solutions_total: int,
    elapsed_run: float,
    new_done: int,
    durations: Sequence[float],
    running: Dict[Future, Tuple[MicroTask, float]],
) -> Dict[str, object]:
    remaining = max(0, selected_total - done_total)
    pct = 100.0 if selected_total == 0 else 100.0 * done_total / selected_total

    # ETA 1：当前这次运行的真实“微分区完成吞吐率”。
    rate = new_done / elapsed_run if elapsed_run > 1e-9 else 0.0
    eta_rate = remaining / rate if rate > 1e-12 else None

    # ETA 2：历史/本次已完成任务的中位耗时 / jobs，只作为结构性参考。
    med = percentile(durations, 0.50)
    p90 = percentile(durations, 0.90)
    p99 = percentile(durations, 0.99)
    max_d = max(durations) if durations else None

    running_items = []
    now = time.monotonic()
    for task, start_mono in running.values():
        running_items.append(
            {
                "global_id": task.global_id,
                "gap_row": task.gap_row,
                "missing_category": task.missing_category,
                "empty_cols": list(task.empty_cols),
                "elapsed_s": max(0.0, now - start_mono),
            }
        )
    running_items.sort(key=lambda x: float(x["elapsed_s"]), reverse=True)

    return {
        "selected_total": selected_total,
        "done_total": done_total,
        "remaining": remaining,
        "progress_pct": pct,
        "feasible_tasks": feasible_total,
        "infeasible_tasks": infeasible_total,
        "solutions_total": solutions_total,
        "elapsed_run_s": elapsed_run,
        "new_done_this_run": new_done,
        "throughput_tasks_per_s": rate,
        "eta_by_current_rate_s": eta_rate,
        "duration_median_s": med,
        "duration_p90_s": p90,
        "duration_p99_s": p99,
        "duration_max_s": max_d,
        "running": running_items,
    }


def print_snapshot(s: Dict[str, object], jobs: int) -> None:
    done = int(s["done_total"])
    total = int(s["selected_total"])
    pct = float(s["progress_pct"])
    remaining = int(s["remaining"])
    elapsed = float(s["elapsed_run_s"])
    rate = float(s["throughput_tasks_per_s"])
    eta = s["eta_by_current_rate_s"]

    print(
        f"[GLOBAL] {progress_bar(done, total)} "
        f"done={done}/{total} ({pct:.2f}%) "
        f"feasible={s['feasible_tasks']} infeasible={s['infeasible_tasks']} "
        f"running={len(s['running'])}/{jobs} remaining={remaining} "
        f"elapsed={fmt_duration(elapsed)} "
        f"rate={rate*60:.2f} task/min ETA={fmt_duration(eta if isinstance(eta, (int, float)) else None)}"
    )

    med = s["duration_median_s"]
    p90 = s["duration_p90_s"]
    p99 = s["duration_p99_s"]
    mx = s["duration_max_s"]
    if isinstance(med, (int, float)):
        print(
            f"[TIMING] completed-task duration: median={fmt_duration(float(med))} "
            f"p90={fmt_duration(float(p90)) if isinstance(p90, (int, float)) else '?'} "
            f"p99={fmt_duration(float(p99)) if isinstance(p99, (int, float)) else '?'} "
            f"max={fmt_duration(float(mx)) if isinstance(mx, (int, float)) else '?'} "
            f"layouts={s['solutions_total']}"
        )

    running = s["running"]
    if running:
        longest = running[0]
        print(
            f"[LONGEST] id={longest['global_id']} "
            f"gap={int(longest['gap_row']):02d} "
            f"missing={longest['missing_category']} "
            f"empty={','.join(map(str, longest['empty_cols']))} "
            f"elapsed={fmt_duration(float(longest['elapsed_s']))}"
        )


def write_progress(path: Path, snapshot: Dict[str, object], extra: Dict[str, object]) -> None:
    payload = dict(extra)
    payload.update(snapshot)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="14x10、34块、13个四色完整行的260分版型微分区全集枚举器 V4"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("layouts_260_v4"),
        help="输出目录，默认 layouts_260_v4",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="外层并行微分区进程数；每个微分区内部固定1个CP-SAT worker。默认8",
    )
    parser.add_argument(
        "--task-time",
        type=float,
        default=0.0,
        help="每个微分区时间上限秒；0=不限时。全集枚举建议0",
    )
    parser.add_argument(
        "--max-solutions-per-task",
        type=int,
        default=0,
        help="每个微分区最多保存多少解；0=不限。非0时不能声称全集完成",
    )
    parser.add_argument(
        "--gap-row",
        default="all",
        help="只枚举指定不完整行，如 14 或 1,14；默认 all",
    )
    parser.add_argument(
        "--missing-category",
        default="all",
        help="只枚举指定缺1块类别，如 square 或 square,T；默认 all",
    )
    parser.add_argument(
        "--empty-cols",
        default="all",
        help="只枚举指定4个空列，如 1,2,3,4；默认 all=210种",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="把完整任务按 global_id 模分成多少个互斥 shard；默认1",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="当前机器负责哪个 shard，0-based；默认0",
    )
    parser.add_argument(
        "--schedule-seed",
        type=int,
        default=20260811,
        help="仅用于打乱微分区执行顺序，让早期ETA样本更有代表性；不改变任务归属",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="CP-SAT基础随机种子；每个微分区按global_id自动派生",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1000,
        help="每多少个解 flush 一次 jsonl，默认1000",
    )
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=30.0,
        help="全局进度/ETA打印周期秒，默认30",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="跳过已有 .done 的微分区；未完成微分区从头重跑",
    )
    parser.add_argument(
        "--store-order-hint",
        action="store_true",
        help="每个布局额外保存一个合法order_hint；默认不保存以减小文件并避免误当最优顺序",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="打印子进程 OR-Tools 日志；并行时会交错，正式枚举通常不要开",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="只扫描当前筛选/shard已有.done并打印状态，不启动求解",
    )
    args = parser.parse_args()

    if args.jobs < 1:
        raise SystemExit("--jobs 必须 >=1")
    if args.task_time < 0:
        raise SystemExit("--task-time 不能为负")
    if args.max_solutions_per_task < 0:
        raise SystemExit("--max-solutions-per-task 不能为负")
    if args.heartbeat <= 0:
        raise SystemExit("--heartbeat 必须 >0")
    if args.shard_count < 1:
        raise SystemExit("--shard-count 必须 >=1")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index 必须满足 0 <= index < shard-count")

    gap_rows = parse_gap_rows(args.gap_row)
    missing_categories = parse_missing_categories(args.missing_category)
    exact_empty_cols = parse_empty_cols(args.empty_cols)

    specs = generate_task_specs(gap_rows, missing_categories, exact_empty_cols)
    specs = [s for s in specs if s[0] % args.shard_count == args.shard_index]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    placements = build_placements()
    write_catalog(args.output_dir / "placement_catalog.json", placements)

    tasks: List[MicroTask] = []
    for gid, gap_row, missing, empty_cols in specs:
        tasks.append(
            MicroTask(
                global_id=gid,
                gap_row=gap_row,
                missing_category=missing,
                empty_cols=empty_cols,
                output_dir=str(args.output_dir),
                task_time=args.task_time,
                max_solutions=args.max_solutions_per_task,
                seed=args.seed + gid * 1009,
                log_search=args.log,
                flush_every=args.flush_every,
                resume=args.resume,
                store_order_hint=args.store_order_hint,
            )
        )

    # 执行顺序打乱，但 global_id / shard 归属完全不变。
    rng = random.Random(args.schedule_seed)
    rng.shuffle(tasks)

    shard_tag = f"shard{args.shard_index:02d}of{args.shard_count:02d}"
    progress_path = args.output_dir / f"progress_{shard_tag}.json"
    summary_path = args.output_dir / f"summary_{shard_tag}.json"

    print("V4：260分全集微分区枚举，不存在 fallback。")
    print(f"候选 placement 数量：{len(placements)}")
    print(f"完整理论微分区：{TOTAL_MICRO_TASKS} = 14*7*C(10,4)")
    print(
        f"当前筛选 + shard：{len(tasks)} tasks, "
        f"shard={args.shard_index}/{args.shard_count}"
    )
    print(f"外层并行 jobs：{args.jobs}；每个CP-SAT内部 workers=1")
    print(f"任务执行顺序：固定随机打散，schedule-seed={args.schedule_seed}")
    print(f"进度心跳：每 {args.heartbeat:g} s")
    if args.task_time == 0:
        print("单微分区时间限制：不限时")
    else:
        print(f"单微分区时间限制：{args.task_time:g} s")
    if args.max_solutions_per_task > 0:
        print(
            f"警告：每微分区最多 {args.max_solutions_per_task} 解；"
            "这是候选池/测试模式，不是全集枚举。"
        )

    # 启动前扫描已完成任务。
    completed_tasks: List[MicroTask] = []
    pending_tasks: List[MicroTask] = []
    durations: List[float] = []
    feasible_total = 0
    infeasible_total = 0
    solutions_total = 0

    for task in tasks:
        if is_done(task):
            completed_tasks.append(task)
            meta = safe_read_meta(task)
            if meta:
                dur = meta.get("elapsed_s")
                if isinstance(dur, (int, float)) and dur >= 0:
                    durations.append(float(dur))
                sols = int(meta.get("solutions", 0))
                solutions_total += sols
                if sols > 0:
                    feasible_total += 1
                else:
                    infeasible_total += 1
        else:
            pending_tasks.append(task)

    selected_total = len(tasks)
    done_total = len(completed_tasks)
    print(
        f"启动前已完成：{done_total}/{selected_total}; "
        f"待运行：{len(pending_tasks)}"
    )

    initial_snapshot = build_progress_snapshot(
        selected_total=selected_total,
        done_total=done_total,
        feasible_total=feasible_total,
        infeasible_total=infeasible_total,
        solutions_total=solutions_total,
        elapsed_run=0.0,
        new_done=0,
        durations=durations,
        running={},
    )
    print_snapshot(initial_snapshot, args.jobs)

    extra_progress = {
        "version": 4,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "global_micro_tasks": TOTAL_MICRO_TASKS,
        "schedule_seed": args.schedule_seed,
        "output_dir": str(args.output_dir),
    }
    write_progress(progress_path, initial_snapshot, extra_progress)

    if args.status_only:
        print("--status-only：不启动求解。")
        return

    if not pending_tasks:
        print("当前筛选/shard 已全部完成。")
        return

    # 用 deque 保存已经随机打散后的待执行队列。
    queue: Deque[MicroTask] = deque(pending_tasks)
    start_run = time.monotonic()
    next_heartbeat = start_run + args.heartbeat
    new_done = 0
    errors = 0
    incomplete_returned = 0
    completion_events: List[float] = []

    # 只维持 jobs 个已提交 future，这样主进程能准确知道“当前正在跑谁、跑了多久”。
    running: Dict[Future, Tuple[MicroTask, float]] = {}

    def submit_one(executor: ProcessPoolExecutor) -> None:
        if not queue:
            return
        task = queue.popleft()
        fut = executor.submit(solve_micro_task, task)
        running[fut] = (task, time.monotonic())

    with ProcessPoolExecutor(max_workers=min(args.jobs, len(pending_tasks))) as ex:
        for _ in range(min(args.jobs, len(pending_tasks))):
            submit_one(ex)

        while running:
            now = time.monotonic()
            timeout = max(0.0, next_heartbeat - now)
            done_futures, _ = wait(
                list(running.keys()),
                timeout=timeout,
                return_when=FIRST_COMPLETED,
            )

            for fut in done_futures:
                task, _start = running.pop(fut)
                try:
                    meta = fut.result()
                    complete = bool(meta.get("complete")) or bool(meta.get("skipped"))
                    sols = int(meta.get("solutions", 0))
                    elapsed_s = float(meta.get("elapsed_s", 0.0))
                    status = str(meta.get("status", "?"))

                    if complete:
                        done_total += 1
                        new_done += 1
                        completion_events.append(time.monotonic())
                        if elapsed_s >= 0:
                            durations.append(elapsed_s)
                        solutions_total += sols
                        if sols > 0:
                            feasible_total += 1
                        else:
                            infeasible_total += 1
                    else:
                        incomplete_returned += 1

                    print(
                        f"[DONE id={task.global_id:05d}] "
                        f"gap={task.gap_row:02d} missing={task.missing_category} "
                        f"empty={','.join(map(str, task.empty_cols))} "
                        f"status={status} solutions={sols} time={fmt_duration(elapsed_s)}"
                    )
                except Exception as e:
                    errors += 1
                    print(
                        f"[ERROR id={task.global_id:05d}] gap={task.gap_row:02d} "
                        f"missing={task.missing_category} empty={task.empty_cols}: {e!r}"
                    )

                submit_one(ex)

            now = time.monotonic()
            if now >= next_heartbeat or not running:
                elapsed_run = now - start_run
                snapshot = build_progress_snapshot(
                    selected_total=selected_total,
                    done_total=done_total,
                    feasible_total=feasible_total,
                    infeasible_total=infeasible_total,
                    solutions_total=solutions_total,
                    elapsed_run=elapsed_run,
                    new_done=new_done,
                    durations=durations,
                    running=running,
                )
                print_snapshot(snapshot, args.jobs)
                write_progress(progress_path, snapshot, extra_progress)
                while next_heartbeat <= now:
                    next_heartbeat += args.heartbeat

    elapsed_run = time.monotonic() - start_run
    all_selected_complete = done_total == selected_total
    exhaustive_mode = args.task_time == 0 and args.max_solutions_per_task == 0

    final_snapshot = build_progress_snapshot(
        selected_total=selected_total,
        done_total=done_total,
        feasible_total=feasible_total,
        infeasible_total=infeasible_total,
        solutions_total=solutions_total,
        elapsed_run=elapsed_run,
        new_done=new_done,
        durations=durations,
        running={},
    )
    write_progress(progress_path, final_snapshot, extra_progress)

    summary = dict(extra_progress)
    summary.update(final_snapshot)
    summary.update(
        {
            "all_selected_complete": all_selected_complete,
            "exhaustive_mode": exhaustive_mode,
            "errors_this_run": errors,
            "incomplete_tasks_returned_this_run": incomplete_returned,
            "task_time": args.task_time,
            "max_solutions_per_task": args.max_solutions_per_task,
            "gap_rows": gap_rows,
            "missing_categories": missing_categories,
            "empty_cols_filter": list(exact_empty_cols) if exact_empty_cols else "all",
        }
    )
    tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, summary_path)

    print("\n===== V4 汇总 =====")
    print_snapshot(final_snapshot, args.jobs)
    print(f"progress: {progress_path.resolve()}")
    print(f"summary : {summary_path.resolve()}")
    print(f"errors={errors}, incomplete_returned={incomplete_returned}")

    if all_selected_complete and exhaustive_mode:
        print("当前筛选/shard 已被完整枚举。")
        if (
            args.shard_count == 1
            and len(gap_rows) == ROWS
            and len(missing_categories) == len(CATEGORIES)
            and exact_empty_cols is None
        ):
            print("20580/20580 全部微分区完成：可以声称所有260分版型已枚举完毕。")
        elif args.shard_count > 1:
            print(
                "注意：这里只证明当前 shard 完成。所有 shard 都完成并合并后，"
                "才是完整 20580 微分区全集。"
            )
    else:
        print(
            "当前仍不是全集完成状态；只有目标微分区全部 .done，且未设置时间/解数截断，"
            "才能声称完整枚举。"
        )


if __name__ == "__main__":
    main()
