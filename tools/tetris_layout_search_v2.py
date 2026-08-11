#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14x10 俄罗斯方块基础任务版型搜索器 V2

V2 核心思路：
1. 阶段 1 只做“可行性搜索”：找到任意一个 280 分满分版型后立即保存并停止；
2. 阶段 2 固定仍然必须 280 分，在满分版型中长时间优化工程风险；
3. 阶段 2 使用阶段 1 的满分解作为 solution hint，并在找到更优解时立即覆盖保存 best_280.yaml；
4. 若阶段 1 在时间限制内没有找到 280 分解，才进入 fallback：完整铺满仍是硬约束，最大化四色完整行数；
5. 默认增加一个安全的 180° 对称性破除约束，减少等价搜索。

比赛约束：
- 盘面 14x10；
- 7 类俄罗斯方块，每类恰好 5 个，共 35 个；
- 每块 4 格，因此 35*4=140，要求完整覆盖盘面；
- 基础任务中完整行 10 分，每行若 >=4 种颜色/类别再 +10 分；
- 因七种形状与七种颜色一一对应，这里以 category 统计颜色种类。

依赖：
    python3 -m pip install ortools

推荐运行：
    python3 tetris_layout_search_v2.py \
        --find-time 600 \
        --optimize-time 10800 \
        --fallback-time 1800 \
        --workers 8 \
        --log

说明：
- row=1 视为盘面最底行，row 向上增加；col=1 为最左列。
- 0° 采用 BASE_SHAPES 定义的标准姿态；+90° 表示在该坐标系中逆时针旋转。
- 若机械臂程序的 angle_deg 正负方向/零位定义不同，只改 ANGLE_OUTPUT_MAP。
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ortools.sat.python import cp_model


# =========================
# 1. 比赛盘面与方块定义
# =========================

ROWS = 14
COLS = 10
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
CATEGORY_CODE = {category: i + 1 for i, category in enumerate(CATEGORIES)}

# 单个小格中心使用整数格点坐标。
# angle_deg=0 时，坐标原点位于形状包围盒左下角。
BASE_SHAPES: Dict[str, Tuple[Tuple[int, int], ...]] = {
    # 黄色 L：底边 3 格，右端向上 1 格
    "L_yellow": ((0, 0), (1, 0), (2, 0), (2, 1)),
    # 紫/蓝色 L：底边 3 格，左端向上 1 格
    "L_blue": ((0, 0), (1, 0), (2, 0), (0, 1)),
    # 绿色 Z/S：下层左两格，上层右两格
    "z_green": ((0, 0), (1, 0), (1, 1), (2, 1)),
    # 蓝色 Z：下层右两格，上层左两格
    "z_blue": ((1, 0), (2, 0), (0, 1), (1, 1)),
    # T：底边 3 格，中间向上 1 格
    "T": ((0, 0), (1, 0), (2, 0), (1, 1)),
    "square": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "line": ((0, 0), (1, 0), (2, 0), (3, 0)),
}

# 搜索器内部角度 -> 比赛程序输出角度。
ANGLE_OUTPUT_MAP = {
    0: 0,
    90: 90,
    180: 180,
    -90: -90,
}

# =========================
# 2. 工程优化指标
# =========================
#
# 阶段 2 的硬约束已经锁死为 280 分，因此这里只负责“满分里面挑更稳的”。
# 目标优先级用整数权重近似成：
#   1) 强烈减少跨 4 行的块（典型就是竖直 line）；
#   2) 其次减少跨 3 行的块；
#   3) 再减少一般跨行程度。
#
# 这不是比赛官方分数，只是工程鲁棒性的次级指标。

SPAN4_PENALTY = 10_000
SPAN3_PENALTY = 200
EXTRA_ROW_PENALTY = 1


@dataclass(frozen=True)
class Orientation:
    category: str
    angle_deg: int
    cells: Tuple[Tuple[int, int], ...]
    width: int
    height: int


@dataclass(frozen=True)
class Placement:
    category: str
    angle_deg: int
    x0: int  # 包围盒左下角列，1-based
    y0: int  # 包围盒左下角行，1-based
    width: int
    height: int
    cells: Tuple[Tuple[int, int], ...]  # (col,row), 1-based

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

    @property
    def row_span(self) -> int:
        return self.max_row - self.min_row + 1


@dataclass
class ModelData:
    model: cp_model.CpModel
    x: List[cp_model.IntVar]
    bonus_row: Dict[int, cp_model.IntVar]


# =========================
# 3. 几何生成
# =========================

def normalize(cells: Iterable[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    cells = list(cells)
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    return tuple(sorted((x - min_x, y - min_y) for x, y in cells))


def rotate_ccw_90(cells: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    # row 轴向上，标准二维坐标逆时针 90°：(x,y) -> (-y,x)
    return normalize((-y, x) for x, y in cells)


def build_orientations() -> List[Orientation]:
    all_orientations: List[Orientation] = []

    for category in CATEGORIES:
        cur = normalize(BASE_SHAPES[category])
        seen = set()

        for angle in (0, 90, 180, -90):
            key = tuple(cur)
            if key not in seen:
                seen.add(key)
                width = max(x for x, _ in cur) + 1
                height = max(y for _, y in cur) + 1
                all_orientations.append(
                    Orientation(
                        category=category,
                        angle_deg=angle,
                        cells=cur,
                        width=width,
                        height=height,
                    )
                )
            cur = rotate_ccw_90(cur)

    return all_orientations


def build_placements() -> List[Placement]:
    placements: List[Placement] = []

    for ori in build_orientations():
        for y0 in range(1, ROWS - ori.height + 2):
            for x0 in range(1, COLS - ori.width + 2):
                cells = tuple((x0 + dx, y0 + dy) for dx, dy in ori.cells)
                placements.append(
                    Placement(
                        category=ori.category,
                        angle_deg=ori.angle_deg,
                        x0=x0,
                        y0=y0,
                        width=ori.width,
                        height=ori.height,
                        cells=cells,
                    )
                )

    return placements


# =========================
# 4. 搜索目标与建模
# =========================

def engineering_risk(p: Placement) -> int:
    """满分版型之间用于比较的工程风险指标；不是官方比赛分数。"""
    risk = EXTRA_ROW_PENALTY * max(0, p.row_span - 1)
    if p.row_span == 3:
        risk += SPAN3_PENALTY
    elif p.row_span >= 4:
        risk += SPAN4_PENALTY
    return risk


def risk_metrics(selected: Sequence[Placement]) -> Dict[str, int]:
    return {
        "risk": sum(engineering_risk(p) for p in selected),
        "span4_count": sum(p.row_span >= 4 for p in selected),
        "span3_count": sum(p.row_span == 3 for p in selected),
        "span2_count": sum(p.row_span == 2 for p in selected),
        "total_extra_rows": sum(max(0, p.row_span - 1) for p in selected),
    }


def build_model(
    placements: Sequence[Placement],
    mode: str,
    symmetry_break: bool,
    hint_solution: Optional[Sequence[Placement]] = None,
) -> ModelData:
    """
    mode:
      - 'first_280'   : 14 行全部 >=4 色，无目标函数，只找第一个可行解；
      - 'optimize_280': 14 行全部 >=4 色，最小化工程风险；
      - 'fallback'    : 完整铺满不变，最大化四色行数，再次级最小化工程风险。
    """
    if mode not in {"first_280", "optimize_280", "fallback"}:
        raise ValueError(f"未知 mode: {mode}")

    model = cp_model.CpModel()
    x = [model.new_bool_var(f"x_{i}") for i in range(len(placements))]

    cell_to_indices = defaultdict(list)
    category_to_indices = defaultdict(list)
    row_category_to_indices = defaultdict(list)

    for i, p in enumerate(placements):
        category_to_indices[p.category].append(i)
        touched_rows = set()
        for cell in p.cells:
            cell_to_indices[cell].append(i)
            touched_rows.add(cell[1])
        for row in touched_rows:
            row_category_to_indices[(row, p.category)].append(i)

    # A. 每个盘面格子恰好覆盖一次 => 无空格、无重叠、完整 14x10 满铺。
    for row in range(1, ROWS + 1):
        for col in range(1, COLS + 1):
            ids = cell_to_indices[(col, row)]
            model.add(sum(x[i] for i in ids) == 1)

    # B. 七类方块各恰好 5 个。
    for category in CATEGORIES:
        ids = category_to_indices[category]
        model.add(sum(x[i] for i in ids) == COPIES_PER_CATEGORY)

    # C. 统计每一行是否出现每个类别，从而判断该行是否 >=4 色。
    bonus_row: Dict[int, cp_model.IntVar] = {}
    for row in range(1, ROWS + 1):
        present_vars = []

        for category in CATEGORIES:
            y = model.new_bool_var(f"present_r{row}_{category}")
            ids = row_category_to_indices[(row, category)]
            selected_in_row = sum(x[i] for i in ids)

            # y == 1 <=> 这一行至少有一个该类别方块。
            model.add(selected_in_row >= y)
            model.add(selected_in_row <= COPIES_PER_CATEGORY * y)
            present_vars.append(y)

        color_count = sum(present_vars)

        if mode in {"first_280", "optimize_280"}:
            model.add(color_count >= 4)
        else:
            b = model.new_bool_var(f"bonus_row_{row}")
            bonus_row[row] = b
            model.add(color_count >= 4).only_enforce_if(b)
            model.add(color_count <= 3).only_enforce_if(b.Not())

    # D. 安全的 180° 对称性破除。
    # 任一完整解做 180° 旋转后仍是同分、同计数的合法解，且类别本身不发生镜像交换。
    # 因此可规定：左下角所属类别编码 <= 右上角所属类别编码。
    # 对任一对 180° 等价解，总有至少一个满足该条件，不会删掉全部等价解。
    if symmetry_break:
        bottom_left_expr = sum(
            CATEGORY_CODE[placements[i].category] * x[i]
            for i in cell_to_indices[(1, 1)]
        )
        top_right_expr = sum(
            CATEGORY_CODE[placements[i].category] * x[i]
            for i in cell_to_indices[(COLS, ROWS)]
        )
        model.add(bottom_left_expr <= top_right_expr)

    risk_expr = sum(engineering_risk(p) * x[i] for i, p in enumerate(placements))

    if mode == "optimize_280":
        model.minimize(risk_expr)
    elif mode == "fallback":
        # 确保“多一条四色行”永远比任何风险改善更重要。
        max_single_risk = max(engineering_risk(p) for p in placements)
        risk_upper_bound = 35 * max_single_risk
        bonus_dominance_weight = risk_upper_bound + 1
        model.maximize(
            bonus_dominance_weight * sum(bonus_row.values()) - risk_expr
        )

    # 阶段 2 用阶段 1 的满分解做 hint。hint 只是搜索起点，不限制最终答案。
    if hint_solution is not None:
        hint_set = set(hint_solution)
        for i, var in enumerate(x):
            model.add_hint(var, 1 if placements[i] in hint_set else 0)

    return ModelData(model=model, x=x, bonus_row=bonus_row)


# =========================
# 5. 校验与有效拼接顺序
# =========================

def bottom_profile_cells(p: Placement) -> List[Tuple[int, int]]:
    """返回该方块每一列中最低的小格。"""
    min_row_by_col = {}
    for col, row in p.cells:
        min_row_by_col[col] = min(row, min_row_by_col.get(col, row))
    return sorted((col, row) for col, row in min_row_by_col.items())


def validate_and_order(selected: Sequence[Placement]) -> Tuple[List[Placement], Counter]:
    if len(selected) != 35:
        raise RuntimeError(f"解中方块数量不是 35，而是 {len(selected)}")

    counts = Counter(p.category for p in selected)
    for category in CATEGORIES:
        if counts[category] != 5:
            raise RuntimeError(f"{category} 数量错误：{counts[category]} != 5")

    owner = {}
    for pid, p in enumerate(selected):
        for cell in p.cells:
            if cell in owner:
                raise RuntimeError(f"发生重叠：格子 {cell}")
            owner[cell] = pid

    expected_cells = {
        (col, row)
        for row in range(1, ROWS + 1)
        for col in range(1, COLS + 1)
    }
    if set(owner) != expected_cells:
        missing = sorted(expected_cells - set(owner))
        extra = sorted(set(owner) - expected_cells)
        raise RuntimeError(f"盘面没有被完整覆盖。missing={missing}, extra={extra}")

    # 自下而上排执行顺序。
    order = sorted(
        range(len(selected)),
        key=lambda i: (
            selected[i].min_row,
            selected[i].max_row,
            selected[i].center_col,
            selected[i].category,
        ),
    )

    placed = set()
    for pid in order:
        p = selected[pid]
        if p.min_row == 1:
            placed.add(pid)
            continue

        supporters = set()
        for col, row in bottom_profile_cells(p):
            if row <= 1:
                continue
            other_pid = owner.get((col, row - 1))
            if other_pid is not None and other_pid != pid:
                supporters.add(other_pid)

        if not (supporters & placed):
            raise RuntimeError(
                "支撑顺序校验失败："
                f"{p.category} center=({p.center_col},{p.center_row}) "
                "没有已放置的下层支撑块"
            )

        placed.add(pid)

    return [selected[i] for i in order], counts


def row_color_statistics(selected: Sequence[Placement]):
    stats = []
    for row in range(1, ROWS + 1):
        categories = sorted(
            {
                p.category
                for p in selected
                if any(cell_row == row for _, cell_row in p.cells)
            }
        )
        stats.append((row, categories))
    return stats


def static_score(selected: Sequence[Placement]) -> int:
    bonus_rows = sum(len(categories) >= 4 for _, categories in row_color_statistics(selected))
    return ROWS * 10 + bonus_rows * 10


# =========================
# 6. 输出
# =========================

def fmt_num(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.1f}"


def target_line(p: Placement) -> str:
    out_angle = ANGLE_OUTPUT_MAP[p.angle_deg]
    return (
        f"- {{col: {fmt_num(p.center_col)}, row: {fmt_num(p.center_row)}, "
        f"angle_deg: {out_angle}, category: {p.category}}}"
    )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_targets(
    path: Path,
    ordered: Sequence[Placement],
    label: str,
) -> None:
    metrics = risk_metrics(ordered)
    score = static_score(ordered)
    lines = [
        "# 基础任务唯一摆放表。行列坐标允许使用半格。",
        f"# {label}",
        f"# 静态得分={score}/280，工程风险={metrics['risk']}，"
        f"span4={metrics['span4_count']}，span3={metrics['span3_count']}，"
        f"total_extra_rows={metrics['total_extra_rows']}",
        "# targets 已按自下而上的有效拼接顺序排序。",
        "",
        "targets:",
        "",
    ]
    lines.extend(target_line(p) for p in ordered)
    atomic_write_text(path, "\n".join(lines) + "\n")


def print_board(selected: Sequence[Placement]) -> None:
    char = {
        "L_yellow": "Y",
        "L_blue": "B",
        "z_green": "G",
        "z_blue": "Z",
        "T": "T",
        "square": "O",
        "line": "I",
    }

    board = [["." for _ in range(COLS)] for _ in range(ROWS)]
    for p in selected:
        for col, row in p.cells:
            board[row - 1][col - 1] = char[p.category]

    print("\n盘面预览（14 行在上，1 行在下）：")
    print("    " + "".join(str(c % 10) for c in range(1, COLS + 1)))
    for row in range(ROWS, 0, -1):
        print(f"{row:>2}: " + "".join(board[row - 1]))
    print("图例：Y=L_yellow, B=L_blue, G=z_green, Z=z_blue, T=T, O=square, I=line")


def print_report(selected: Sequence[Placement], title: str) -> None:
    counts = Counter(p.category for p in selected)
    stats = row_color_statistics(selected)
    bonus_rows = [row for row, categories in stats if len(categories) >= 4]
    metrics = risk_metrics(selected)

    print(f"\n===== {title} =====")
    print("类别数量：")
    for category in CATEGORIES:
        print(f"  {category:10s}: {counts[category]}")

    print("\n逐行颜色/类别统计：")
    for row, categories in stats:
        mark = "BONUS" if len(categories) >= 4 else "----"
        print(f"  row {row:2d}: {len(categories)} 类  {mark}  {categories}")

    print("\n完整行：14/14")
    print(f">=4 色完整行：{len(bonus_rows)}/14")
    print(f"基础任务静态布局得分：{static_score(selected)}/280")
    print(
        "工程风险："
        f"risk={metrics['risk']}, "
        f"span4={metrics['span4_count']}, "
        f"span3={metrics['span3_count']}, "
        f"span2={metrics['span2_count']}, "
        f"total_extra_rows={metrics['total_extra_rows']}"
    )


# =========================
# 7. Solver 工具与 callback
# =========================

def configure_solver(
    time_limit_s: float,
    workers: int,
    seed: int,
    log_search: bool,
) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.log_search_progress = log_search
    return solver


def selected_from_solver(
    placements: Sequence[Placement],
    x: Sequence[cp_model.IntVar],
    solver: cp_model.CpSolver,
) -> List[Placement]:
    return [p for p, var in zip(placements, x) if solver.value(var) == 1]


class FirstFeasibleSaver(cp_model.CpSolverSolutionCallback):
    """阶段 1：拿到第一个 280 分解就保存并主动停止搜索。"""

    def __init__(
        self,
        placements: Sequence[Placement],
        x: Sequence[cp_model.IntVar],
        output_path: Path,
    ) -> None:
        super().__init__()
        self._placements = placements
        self._x = x
        self._output_path = output_path
        self.selected: List[Placement] = []

    def on_solution_callback(self) -> None:
        selected = [
            p for p, var in zip(self._placements, self._x)
            if self.value(var) == 1
        ]
        ordered, _ = validate_and_order(selected)
        write_targets(
            self._output_path,
            ordered,
            label="阶段1：找到的第一个 280 分可行解",
        )
        self.selected = list(selected)
        print(
            f"\n[阶段1] 找到第一个 280 分解，"
            f"已立即保存到 {self._output_path.resolve()}"
        )
        self.stop_search()


class BestRiskSaver(cp_model.CpSolverSolutionCallback):
    """阶段 2：每当找到工程风险更低的 280 分解，就立即覆盖保存。"""

    def __init__(
        self,
        placements: Sequence[Placement],
        x: Sequence[cp_model.IntVar],
        output_path: Path,
        initial_best_risk: int,
    ) -> None:
        super().__init__()
        self._placements = placements
        self._x = x
        self._output_path = output_path
        self.best_risk = initial_best_risk
        self.best_selected: List[Placement] = []
        self.improvement_count = 0

    def on_solution_callback(self) -> None:
        selected = [
            p for p, var in zip(self._placements, self._x)
            if self.value(var) == 1
        ]
        risk = risk_metrics(selected)["risk"]

        # callback 可能报告当前 incumbent；这里只在真的更好时落盘。
        if risk >= self.best_risk:
            return

        ordered, _ = validate_and_order(selected)
        self.best_risk = risk
        self.best_selected = list(selected)
        self.improvement_count += 1
        write_targets(
            self._output_path,
            ordered,
            label=f"阶段2：280 分版型工程优化，第 {self.improvement_count} 次改进",
        )
        print(
            f"[阶段2] 工程风险改进到 {risk}，"
            f"已保存 {self._output_path.resolve()}"
        )


class BestFallbackSaver(cp_model.CpSolverSolutionCallback):
    """fallback：优先保存四色行更多的解，同分时保存工程风险更低的解。"""

    def __init__(
        self,
        placements: Sequence[Placement],
        x: Sequence[cp_model.IntVar],
        output_path: Path,
    ) -> None:
        super().__init__()
        self._placements = placements
        self._x = x
        self._output_path = output_path
        self.best_key: Optional[Tuple[int, int]] = None  # (bonus_rows, -risk)
        self.best_selected: List[Placement] = []

    def on_solution_callback(self) -> None:
        selected = [
            p for p, var in zip(self._placements, self._x)
            if self.value(var) == 1
        ]
        bonus_rows = sum(
            len(categories) >= 4
            for _, categories in row_color_statistics(selected)
        )
        risk = risk_metrics(selected)["risk"]
        key = (bonus_rows, -risk)

        if self.best_key is not None and key <= self.best_key:
            return

        ordered, _ = validate_and_order(selected)
        self.best_key = key
        self.best_selected = list(selected)
        write_targets(
            self._output_path,
            ordered,
            label=f"fallback 当前最好：四色完整行={bonus_rows}/14",
        )
        print(
            f"[fallback] 当前最好：四色行={bonus_rows}/14, risk={risk}，"
            f"已保存 {self._output_path.resolve()}"
        )


def status_name(solver: cp_model.CpSolver, status) -> str:
    try:
        return solver.status_name(status)
    except Exception:
        return str(status)


# =========================
# 8. 三阶段流程
# =========================

def search_first_280(
    placements: Sequence[Placement],
    time_limit_s: float,
    workers: int,
    seed: int,
    log_search: bool,
    symmetry_break: bool,
    output_path: Path,
):
    data = build_model(
        placements,
        mode="first_280",
        symmetry_break=symmetry_break,
    )
    solver = configure_solver(time_limit_s, workers, seed, log_search)
    callback = FirstFeasibleSaver(placements, data.x, output_path)
    status = solver.solve(data.model, callback)

    selected = callback.selected
    if not selected and status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        # 理论上 callback 已经拿到；保留兜底。
        selected = selected_from_solver(placements, data.x, solver)
        if selected:
            ordered, _ = validate_and_order(selected)
            write_targets(output_path, ordered, "阶段1：280 分可行解")

    return status, solver, selected


def optimize_280(
    placements: Sequence[Placement],
    initial_solution: Sequence[Placement],
    time_limit_s: float,
    workers: int,
    seed: int,
    log_search: bool,
    symmetry_break: bool,
    output_path: Path,
):
    data = build_model(
        placements,
        mode="optimize_280",
        symmetry_break=symmetry_break,
        hint_solution=initial_solution,
    )

    initial_ordered, _ = validate_and_order(initial_solution)
    initial_risk = risk_metrics(initial_solution)["risk"]
    write_targets(
        output_path,
        initial_ordered,
        label="阶段2初始值：直接继承阶段1的 280 分解",
    )

    solver = configure_solver(time_limit_s, workers, seed, log_search)
    callback = BestRiskSaver(
        placements,
        data.x,
        output_path,
        initial_best_risk=initial_risk,
    )
    status = solver.solve(data.model, callback)

    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        final_selected = selected_from_solver(placements, data.x, solver)
        final_risk = risk_metrics(final_selected)["risk"]
        if final_risk <= callback.best_risk:
            ordered, _ = validate_and_order(final_selected)
            callback.best_risk = final_risk
            callback.best_selected = list(final_selected)
            write_targets(
                output_path,
                ordered,
                label=(
                    "阶段2最终结果："
                    + ("已证明工程风险全局最优" if status == cp_model.OPTIMAL else "当前时间内最好")
                ),
            )

    if callback.best_selected:
        best_selected = callback.best_selected
    else:
        best_selected = list(initial_solution)

    return status, solver, best_selected


def fallback_search(
    placements: Sequence[Placement],
    time_limit_s: float,
    workers: int,
    seed: int,
    log_search: bool,
    symmetry_break: bool,
    output_path: Path,
):
    data = build_model(
        placements,
        mode="fallback",
        symmetry_break=symmetry_break,
    )
    solver = configure_solver(time_limit_s, workers, seed, log_search)
    callback = BestFallbackSaver(placements, data.x, output_path)
    status = solver.solve(data.model, callback)

    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        final_selected = selected_from_solver(placements, data.x, solver)
        ordered, _ = validate_and_order(final_selected)
        write_targets(
            output_path,
            ordered,
            label=(
                "fallback 最终结果："
                + ("已证明目标最优" if status == cp_model.OPTIMAL else "当前时间内最好")
            ),
        )
        return status, solver, final_selected

    return status, solver, callback.best_selected


# =========================
# 9. 主程序
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="14x10、35 块、每类 5 块的俄罗斯方块基础任务搜索器 V2"
    )
    parser.add_argument(
        "--find-time",
        "--full-time",
        dest="find_time",
        type=float,
        default=600.0,
        help="阶段1：只找第一个 280 分可行解的时间上限，默认 600 秒；--full-time 为兼容旧参数名",
    )
    parser.add_argument(
        "--optimize-time",
        type=float,
        default=3600.0,
        help="阶段2：在 280 分解中优化工程风险的时间上限，默认 3600 秒；设为 0 可跳过",
    )
    parser.add_argument(
        "--fallback-time",
        type=float,
        default=1800.0,
        help="阶段1没找到 280 时，fallback 最大化四色行数的时间上限，默认 1800 秒",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="CP-SAT 并行 worker 数，默认 8",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="随机种子，默认 1；阶段2自动使用 seed+1",
    )
    parser.add_argument(
        "--first-output",
        type=Path,
        default=Path("first_280.yaml"),
        help="第一个 280 分解输出文件，默认 first_280.yaml",
    )
    parser.add_argument(
        "--best-output",
        "--output",
        dest="best_output",
        type=Path,
        default=Path("best_280.yaml"),
        help="阶段2当前最好 280 分解输出文件，默认 best_280.yaml；--output 为兼容旧参数名",
    )
    parser.add_argument(
        "--fallback-output",
        type=Path,
        default=Path("fallback_targets.yaml"),
        help="fallback 输出文件，默认 fallback_targets.yaml",
    )
    parser.add_argument(
        "--no-symmetry-break",
        action="store_true",
        help="关闭 180° 对称性破除；默认开启",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="打印 OR-Tools 搜索日志",
    )
    args = parser.parse_args()

    symmetry_break = not args.no_symmetry_break
    placements = build_placements()

    print(f"候选 placement 数量：{len(placements)}")
    print(f"180° 对称性破除：{'开启' if symmetry_break else '关闭'}")
    print(
        "阶段1：只找第一个 280 分版型；找到后立即保存并停止，"
        "不在这一阶段证明工程风险最优。"
    )

    status1, solver1, first_280 = search_first_280(
        placements=placements,
        time_limit_s=args.find_time,
        workers=args.workers,
        seed=args.seed,
        log_search=args.log,
        symmetry_break=symmetry_break,
        output_path=args.first_output,
    )

    print(f"[阶段1] solver status = {status_name(solver1, status1)}")

    if first_280:
        ordered_first, _ = validate_and_order(first_280)
        print_board(first_280)
        print_report(first_280, "阶段1：第一个 280 分解")

        if args.optimize_time > 0:
            print(
                f"\n阶段2：保持 280 分硬约束，继续优化工程风险，"
                f"时间上限 {args.optimize_time:g} 秒。"
            )
            print(
                f"阶段1解会作为 hint；{args.best_output} 会先写入阶段1解，"
                "之后每发现更优版型就立即覆盖。"
            )

            status2, solver2, best_280 = optimize_280(
                placements=placements,
                initial_solution=first_280,
                time_limit_s=args.optimize_time,
                workers=args.workers,
                seed=args.seed + 1,
                log_search=args.log,
                symmetry_break=symmetry_break,
                output_path=args.best_output,
            )
            print(f"[阶段2] solver status = {status_name(solver2, status2)}")
            print_board(best_280)
            print_report(best_280, "阶段2：当前最好 280 分解")
            print(f"\n最终建议使用：{args.best_output.resolve()}")
        else:
            print("\n--optimize-time=0，已跳过阶段2。")
            print(f"最终建议使用：{args.first_output.resolve()}")

    else:
        print(
            "\n阶段1在当前时间限制内没有找到 280 分解。"
            "如果状态是 UNKNOWN，只表示时间内没找到，并不等于证明不存在。"
        )
        print(
            f"进入 fallback：保持 35 块完整铺满，最大化四色完整行数量，"
            f"时间上限 {args.fallback_time:g} 秒。"
        )

        status3, solver3, fallback = fallback_search(
            placements=placements,
            time_limit_s=args.fallback_time,
            workers=args.workers,
            seed=args.seed + 2,
            log_search=args.log,
            symmetry_break=symmetry_break,
            output_path=args.fallback_output,
        )
        print(f"[fallback] solver status = {status_name(solver3, status3)}")

        if not fallback:
            raise SystemExit(
                "没有找到可用版型。建议增大 --find-time / --fallback-time，"
                "或更换 --seed 后重试。"
            )

        print_board(fallback)
        print_report(fallback, "fallback 当前最好解")
        print(f"\n已输出：{args.fallback_output.resolve()}")

    print(
        "\n重要：第一次上实机前，仍需确认 BASE_SHAPES 的 0° 定义以及 "
        "ANGLE_OUTPUT_MAP 与机械臂实际角度正负方向完全一致。"
    )


if __name__ == "__main__":
    main()
