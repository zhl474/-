#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14x10 俄罗斯方块基础任务 260 分代表性盘面库生成器 V5

设计目标
========
V5 不再像 V4 那样枚举 (gap_row, missing_category, empty_cols) 下的全部几何解。
它直接服务于比赛现场的“盘面匹配”思路：

1. 以“目标盘左侧 7 类方块数量”作为离线搜索任务 signature；
2. signature 按随机现场出现的组合权重从高到低排序；
3. 对每个 signature，只找少量 K 个“粗空间分布不同”的 260 分合法盘面；
4. gap row、4 个空格位置、缺失类别全部交给 CP-SAT 自己决定；
5. 中线目标不硬分左右：center_col == 5.5 时可灵活归到 LEFT/RIGHT；
6. 四区只作为代表性/在线匹配特征：LU/RU/LD/RD，中线目标保存 allowed regions；
7. 每找到一个布局立即原子写盘，timeout 后可 --resume 继续；
8. 保留 V4 的多进程、shard、heartbeat、status-only、合法摆放顺序检查等基础设施。

260 分硬约束
============
- 共放 34 块；
- 7 类中恰好一类用 4 块，其余 6 类各 5 块；
- 14 行中恰好 13 行完整，唯一 gap row 恰好 6/10 格被占；
- 13 个完整行每行至少 4 种类别/颜色；
- 不重叠、不越界；
- 存在符合比赛规则的自下而上合法拼接顺序。

默认 signature
==============
35 个现场方块、每类 5 个。默认假设用于匹配的“小侧”有 17 个源方块，
因此目标盘也约束为 17 个目标归 LEFT、17 个归 RIGHT。

理论 signature 数：
    sum(s_c)=17, 0<=s_c<=5, c=1..7  -> 24017 种

随机现场（固定 17 个槽在左侧）下某 signature 的组合权重：
    weight(s) = Π C(5, s_c)

默认只跑权重最高的前 7000 种，可用 --top-signatures 0 跑全部。

四区与中线
==========
盘中心：col=5.5, row=7.5。
- 普通目标：唯一属于 LU/RU/LD/RD；
- 压竖中线：上半区允许 LU|RU，下半区允许 LD|RD；
- 压横中线：左半区允许 LU|LD，右半区允许 RU|RD；
- 正中心：允许 LU|RU|LD|RD。

V5 用“每类别 × allowed-region-mask 的计数向量”定义粗空间 signature。
同一个左右 signature 下保存的 K 个布局必须拥有不同的粗空间 signature，
避免只得到几何细节不同、整体分布几乎相同的盘面。

依赖
====
    python3 -m pip install ortools

建议先冒烟测试：
    python3 tetris_layout_search_v5.py \
        --output-dir v5_smoke \
        --top-signatures 20 \
        --layouts-per-signature 1 \
        --jobs 2 --cp-workers 2 \
        --task-time 20 --heartbeat 5

正式先跑高概率约 95% 的前 7000 种：
    python3 tetris_layout_search_v5.py \
        --output-dir layouts_260_v5 \
        --top-signatures 7000 \
        --layouts-per-signature 3 \
        --jobs 8 --cp-workers 1 \
        --task-time 60 --heartbeat 30 --resume

两台电脑：
    # A
    python3 tetris_layout_search_v5.py \
        --output-dir layouts_260_v5_A \
        --top-signatures 7000 --layouts-per-signature 3 \
        --jobs 8 --cp-workers 1 --task-time 60 \
        --shard-count 2 --shard-index 0 --heartbeat 30 --resume

    # B
    python3 tetris_layout_search_v5.py \
        --output-dir layouts_260_v5_B \
        --top-signatures 7000 --layouts-per-signature 3 \
        --jobs 8 --cp-workers 1 --task-time 60 \
        --shard-count 2 --shard-index 1 --heartbeat 30 --resume

说明
====
- row=1 为最底行，col=1 为最左列；
- V5 默认保存一个合法 order_hint 仅作验证/参考；真正比赛执行顺序仍应由在线路径优化器重新搜索；
- --task-time 是“一个左右 signature 本次运行的总墙钟时间”，不是每次 Solve 的时间；
- timeout 的任务不会写 .done，但已经找到的布局会保留；下次 --resume 会从这些布局继续；
- 同一个 output-dir 最好保持相同 --left-count。--layouts-per-signature 可以后续增大，V5 会继续补齐。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from ortools.sat.python import cp_model


# =========================
# 1. 常量与几何定义（沿用 V4）
# =========================

ROWS = 14
COLS = 10
TOTAL_PIECES = 34
COPIES_PER_CATEGORY = 5

CENTER_COL = (COLS + 1) / 2.0  # 5.5
CENTER_ROW = (ROWS + 1) / 2.0  # 7.5
EPS = 1e-9

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

# bit mask：LU=1, RU=2, LD=4, RD=8
REGION_BITS = {
    "LU": 1,
    "RU": 2,
    "LD": 4,
    "RD": 8,
}
REGION_MASK_ORDER = (1, 2, 4, 8, 3, 12, 5, 10, 15)
REGION_MASK_NAMES = {
    1: ("LU",),
    2: ("RU",),
    4: ("LD",),
    8: ("RD",),
    3: ("LU", "RU"),
    12: ("LD", "RD"),
    5: ("LU", "LD"),
    10: ("RU", "RD"),
    15: ("LU", "RU", "LD", "RD"),
}

SPATIAL_SIGNATURE_ORDER: Tuple[Tuple[str, int], ...] = tuple(
    (category, mask)
    for category in CATEGORIES
    for mask in REGION_MASK_ORDER
)


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
class SignatureSpec:
    global_id: int
    rank: int
    left_counts: Tuple[int, ...]
    weight: int
    probability: float


@dataclass(frozen=True)
class SignatureTask:
    global_id: int
    rank: int
    left_counts: Tuple[int, ...]
    weight: int
    probability: float
    total_probability_weight: int
    output_dir: str
    task_time: float
    layouts_per_signature: int
    cp_workers: int
    seed: int
    log_search: bool
    resume: bool
    store_order_hint: bool


@dataclass
class V5ModelData:
    model: cp_model.CpModel
    x: List[cp_model.IntVar]
    gap_vars: Dict[int, cp_model.IntVar]
    missing_vars: Dict[str, cp_model.IntVar]
    spatial_component_ids: Dict[Tuple[str, int], Tuple[int, ...]]


# =========================
# 2. 几何生成
# =========================

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


def placement_region_mask(p: Placement) -> int:
    """返回 placement 中心点允许归属的四区 bit mask。"""
    x = p.center_col
    y = p.center_row

    on_v = abs(x - CENTER_COL) < EPS
    on_h = abs(y - CENTER_ROW) < EPS

    if on_v and on_h:
        return 15
    if on_v:
        return 3 if y > CENTER_ROW else 12
    if on_h:
        return 5 if x < CENTER_COL else 10
    if x < CENTER_COL and y > CENTER_ROW:
        return 1
    if x > CENTER_COL and y > CENTER_ROW:
        return 2
    if x < CENTER_COL and y < CENTER_ROW:
        return 4
    return 8


def allowed_regions(p: Placement) -> Tuple[str, ...]:
    return REGION_MASK_NAMES[placement_region_mask(p)]


def build_indices(placements: Sequence[Placement]):
    cell_to_ids = defaultdict(list)
    category_to_ids = defaultdict(list)
    row_category_to_ids = defaultdict(list)
    spatial_component_ids = defaultdict(list)

    for p in placements:
        category_to_ids[p.category].append(p.pid)
        spatial_component_ids[(p.category, placement_region_mask(p))].append(p.pid)
        touched_rows = set()
        for cell in p.cells:
            cell_to_ids[cell].append(p.pid)
            touched_rows.add(cell[1])
        for row in touched_rows:
            row_category_to_ids[(row, p.category)].append(p.pid)

    return (
        cell_to_ids,
        category_to_ids,
        row_category_to_ids,
        {k: tuple(v) for k, v in spatial_component_ids.items()},
    )


# =========================
# 3. signature 生成与概率排序
# =========================

def generate_signature_specs(left_count: int) -> Tuple[List[SignatureSpec], int]:
    if not 0 <= left_count <= 35:
        raise ValueError("left_count 必须在 0..35")

    raw: List[Tuple[int, Tuple[int, ...], int]] = []
    gid = 0
    for counts in product(range(COPIES_PER_CATEGORY + 1), repeat=len(CATEGORIES)):
        if sum(counts) != left_count:
            continue
        weight = math.prod(math.comb(COPIES_PER_CATEGORY, n) for n in counts)
        raw.append((gid, tuple(counts), weight))
        gid += 1

    total_weight = math.comb(35, left_count)
    ranked = sorted(raw, key=lambda item: (-item[2], item[1]))

    specs: List[SignatureSpec] = []
    for rank0, (global_id, counts, weight) in enumerate(ranked):
        specs.append(
            SignatureSpec(
                global_id=global_id,
                rank=rank0 + 1,
                left_counts=counts,
                weight=weight,
                probability=weight / total_weight,
            )
        )
    return specs, total_weight


def parse_signature(value: str) -> Tuple[int, ...]:
    try:
        vals = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "--signature 必须是 7 个逗号分隔整数，例如 2,3,2,3,2,3,2"
        ) from e
    if len(vals) != len(CATEGORIES):
        raise argparse.ArgumentTypeError("--signature 必须恰好包含 7 个整数")
    if any(v < 0 or v > COPIES_PER_CATEGORY for v in vals):
        raise argparse.ArgumentTypeError("signature 每一项必须在 0..5")
    return vals


def signature_dict(counts: Sequence[int]) -> Dict[str, int]:
    return {c: int(counts[i]) for i, c in enumerate(CATEGORIES)}


# =========================
# 4. V5 CP-SAT 模型
# =========================

def build_v5_model(
    placements: Sequence[Placement],
    target_left_counts: Sequence[int],
) -> V5ModelData:
    if len(target_left_counts) != len(CATEGORIES):
        raise ValueError("target_left_counts 长度必须为 7")
    if any(v < 0 or v > COPIES_PER_CATEGORY for v in target_left_counts):
        raise ValueError("target_left_counts 每项必须在 0..5")
    if sum(target_left_counts) * 2 != TOTAL_PIECES:
        raise ValueError(
            f"V5 当前要求目标左右数量相等，因此 sum(left) 必须为 {TOTAL_PIECES // 2}"
        )

    model = cp_model.CpModel()
    x = [model.new_bool_var(f"x_{p.pid}") for p in placements]
    (
        cell_to_ids,
        category_to_ids,
        row_category_to_ids,
        spatial_component_ids,
    ) = build_indices(placements)

    # 1) 所有格子最多覆盖一次；恰好一行是 gap row，行占用为 6，其余行占用为 10。
    gap_vars = {row: model.new_bool_var(f"gap_r{row}") for row in range(1, ROWS + 1)}
    model.add(sum(gap_vars.values()) == 1)

    for row in range(1, ROWS + 1):
        row_occupancies = []
        for col in range(1, COLS + 1):
            ids = cell_to_ids[(col, row)]
            occ = sum(x[i] for i in ids)
            model.add(occ <= 1)
            row_occupancies.append(occ)
        model.add(sum(row_occupancies) == COLS - 4 * gap_vars[row])

    # 2) 34 块；missing category 由模型自行决定。
    model.add(sum(x) == TOTAL_PIECES)
    missing_vars = {
        c: model.new_bool_var(f"missing_{c}")
        for c in CATEGORIES
    }
    model.add(sum(missing_vars.values()) == 1)

    for category in CATEGORIES:
        model.add(
            sum(x[i] for i in category_to_ids[category])
            == COPIES_PER_CATEGORY - missing_vars[category]
        )

    # 3) 13 个完整行都必须 >=4 类/色；gap row 不要求四色。
    for row in range(1, ROWS + 1):
        present_vars = []
        for category in CATEGORIES:
            y = model.new_bool_var(f"present_r{row}_{category}")
            ids = row_category_to_ids[(row, category)]
            selected_touching_row = sum(x[i] for i in ids)
            model.add(selected_touching_row >= y)
            model.add(selected_touching_row <= COPIES_PER_CATEGORY * y)
            present_vars.append(y)
        model.add(sum(present_vars) >= 4).only_enforce_if(gap_vars[row].Not())

    # 4) 必要支撑硬约束（取自 V3-heartbeat 的有效剪枝）：
    #    不接触 row=1 的方块，最终盘面上底部轮廓至少一格正下方必须存在另一个已选方块。
    #    这是合法执行顺序的必要条件；最终仍用 compute_legal_order 做精确判定。
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

    # 5) 左侧类别 signature。
    #    center_col < 5.5 固定 LEFT；>5.5 固定 RIGHT；==5.5 可灵活 L/R。
    for ci, category in enumerate(CATEGORIES):
        fixed_left_ids = [
            p.pid
            for p in placements
            if p.category == category and p.center_col < CENTER_COL - EPS
        ]
        flex_ids = [
            p.pid
            for p in placements
            if p.category == category and abs(p.center_col - CENTER_COL) < EPS
        ]

        flex_left_vars = []
        for pid in flex_ids:
            z = model.new_bool_var(f"flex_left_{pid}")
            model.add(z <= x[pid])
            flex_left_vars.append(z)

        model.add(
            sum(x[i] for i in fixed_left_ids) + sum(flex_left_vars)
            == int(target_left_counts[ci])
        )

    return V5ModelData(
        model=model,
        x=x,
        gap_vars=gap_vars,
        missing_vars=missing_vars,
        spatial_component_ids=spatial_component_ids,
    )


def spatial_signature_vector(
    placements: Sequence[Placement],
    selected_ids: Sequence[int],
) -> Tuple[int, ...]:
    counts = Counter(
        (placements[pid].category, placement_region_mask(placements[pid]))
        for pid in selected_ids
    )
    return tuple(int(counts[(category, mask)]) for category, mask in SPATIAL_SIGNATURE_ORDER)


def add_spatial_signature_nogood(
    data: V5ModelData,
    signature_vector: Sequence[int],
    tag: str,
) -> None:
    """要求下一解至少有一个“类别×allowed-region-mask 计数”不同。"""
    if len(signature_vector) != len(SPATIAL_SIGNATURE_ORDER):
        raise ValueError("spatial signature vector 长度错误")

    diff_lits = []
    for j, ((category, mask), old_value) in enumerate(
        zip(SPATIAL_SIGNATURE_ORDER, signature_vector)
    ):
        ids = data.spatial_component_ids.get((category, mask), ())
        if not ids:
            # 该组件永远为 0，不可能承担“不同”。
            continue
        d = data.model.new_bool_var(f"diff_{tag}_{j}")
        expr = sum(data.x[i] for i in ids)
        data.model.add(expr != int(old_value)).only_enforce_if(d)
        diff_lits.append(d)

    if not diff_lits:
        # 理论上不会发生；为安全起见直接禁止模型。
        data.model.add(0 == 1)
    else:
        data.model.add_bool_or(diff_lits)


def add_exact_geometry_nogood(data: V5ModelData, selected_ids: Sequence[int]) -> None:
    """仅排除当前 34 个 placement 的精确几何集合。"""
    data.model.add(sum(data.x[i] for i in selected_ids) <= TOTAL_PIECES - 1)


# =========================
# 5. 合法顺序与独立校验（沿用 V4）
# =========================

def compute_legal_order(
    placements: Sequence[Placement],
    selected_ids: Sequence[int],
) -> Optional[List[int]]:
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


def derive_layout_properties(
    placements: Sequence[Placement],
    selected_ids: Sequence[int],
) -> Tuple[int, str, Tuple[int, int, int, int]]:
    selected = [placements[i] for i in selected_ids]
    counts = Counter(p.category for p in selected)
    missing = [c for c in CATEGORIES if counts[c] == COPIES_PER_CATEGORY - 1]
    if len(missing) != 1:
        raise RuntimeError(f"无法唯一确定 missing category: counts={dict(counts)}")

    owner: Dict[Tuple[int, int], int] = {}
    for p in selected:
        for cell in p.cells:
            if cell in owner:
                raise RuntimeError(f"重叠: {cell}")
            owner[cell] = p.pid

    row_occ = {
        row: sum((col, row) in owner for col in range(1, COLS + 1))
        for row in range(1, ROWS + 1)
    }
    gap_rows = [row for row, n in row_occ.items() if n == 6]
    if len(gap_rows) != 1:
        raise RuntimeError(f"无法唯一确定 gap row: row_occ={row_occ}")
    gap_row = gap_rows[0]

    empty_cols = tuple(
        col for col in range(1, COLS + 1) if (col, gap_row) not in owner
    )
    if len(empty_cols) != 4:
        raise RuntimeError(f"gap row 空格数不是4: {empty_cols}")

    return gap_row, missing[0], empty_cols  # type: ignore[return-value]


def validate_v5_solution(
    placements: Sequence[Placement],
    selected_ids: Sequence[int],
    target_left_counts: Sequence[int],
) -> Tuple[int, str, Tuple[int, int, int, int], List[int]]:
    if len(selected_ids) != TOTAL_PIECES:
        raise RuntimeError(f"方块数错误: {len(selected_ids)} != {TOTAL_PIECES}")

    gap_row, missing_category, empty_cols = derive_layout_properties(
        placements, selected_ids
    )
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

    # 验证左右 signature 在中线 flex 允许下可实现。
    for ci, category in enumerate(CATEGORIES):
        fixed_left = sum(
            p.category == category and p.center_col < CENTER_COL - EPS
            for p in selected
        )
        flex_lr = sum(
            p.category == category and abs(p.center_col - CENTER_COL) < EPS
            for p in selected
        )
        desired = int(target_left_counts[ci])
        if not fixed_left <= desired <= fixed_left + flex_lr:
            raise RuntimeError(
                f"{category} 左侧 signature 不可实现: fixed_left={fixed_left}, "
                f"flex={flex_lr}, desired={desired}"
            )

    legal_order = compute_legal_order(placements, selected_ids)
    if legal_order is None:
        raise RuntimeError("不存在符合规则的有效拼接顺序")

    return gap_row, missing_category, empty_cols, legal_order


# =========================
# 6. 输出格式 / resume
# =========================

def task_stem(task: SignatureTask) -> str:
    return f"sig_{task.global_id:05d}"


def task_paths(task: SignatureTask) -> Dict[str, Path]:
    base = Path(task.output_dir) / "signatures"
    stem = task_stem(task)
    return {
        "dir": base,
        "data": base / f"{stem}.json",
        "meta": base / f"{stem}.meta.json",
        "done": base / f"{stem}.done",
    }


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def empty_task_payload(task: SignatureTask) -> Dict[str, object]:
    return {
        "version": 5,
        "global_id": task.global_id,
        "rank": task.rank,
        "target_left_counts": list(task.left_counts),
        "target_left_by_category": signature_dict(task.left_counts),
        "weight": task.weight,
        "probability": task.probability,
        "layouts": [],
    }


def load_task_payload(task: SignatureTask) -> Dict[str, object]:
    p = task_paths(task)["data"]
    if not p.exists():
        return empty_task_payload(task)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return empty_task_payload(task)

    if payload.get("version") != 5:
        raise RuntimeError(f"{p} 不是 V5 数据文件")
    if tuple(payload.get("target_left_counts", [])) != task.left_counts:
        raise RuntimeError(f"{p} 的 signature 与当前任务不一致")
    if not isinstance(payload.get("layouts"), list):
        payload["layouts"] = []
    return payload


def safe_read_meta(task: SignatureTask) -> Optional[Dict[str, object]]:
    p = task_paths(task)["meta"]
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def layout_count(task: SignatureTask) -> int:
    try:
        payload = load_task_payload(task)
        return len(payload.get("layouts", []))  # type: ignore[arg-type]
    except Exception:
        return 0


def is_done_for_current_quota(task: SignatureTask) -> bool:
    # 实际已保存的布局数达到当前配额时，不再依赖 .done 标记。
    # 这样从 K=3 切回 K=1 时，已有 1～2 个布局的任务也能正确跳过。
    if layout_count(task) >= task.layouts_per_signature:
        return True

    paths = task_paths(task)
    if not paths["done"].exists():
        return False
    meta = safe_read_meta(task)
    if not meta:
        return False
    status = str(meta.get("status", ""))
    return status == "EXHAUSTED"


def write_catalog(path: Path, placements: Sequence[Placement]) -> None:
    payload = {
        "version": 5,
        "rows": ROWS,
        "cols": COLS,
        "center_col": CENTER_COL,
        "center_row": CENTER_ROW,
        "angle_output_map": ANGLE_OUTPUT_MAP,
        "region_bits": REGION_BITS,
        "region_masks": {
            str(mask): list(REGION_MASK_NAMES[mask]) for mask in REGION_MASK_ORDER
        },
        "spatial_signature_order": [
            {
                "category": category,
                "region_mask": mask,
                "allowed_regions": list(REGION_MASK_NAMES[mask]),
            }
            for category, mask in SPATIAL_SIGNATURE_ORDER
        ],
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
                "region_mask": placement_region_mask(p),
                "allowed_regions": list(allowed_regions(p)),
            }
            for p in placements
        ],
    }
    atomic_write_json(path, payload)


def make_layout_record(
    task: SignatureTask,
    placements: Sequence[Placement],
    selected_ids: Sequence[int],
    gap_row: int,
    missing_category: str,
    empty_cols: Sequence[int],
    legal_order: Sequence[int],
) -> Dict[str, object]:
    vec = spatial_signature_vector(placements, selected_ids)
    record: Dict[str, object] = {
        "ids": list(selected_ids),
        "missing_category": missing_category,
        "gap_row": gap_row,
        "empty_cols": list(empty_cols),
        "spatial_signature_vector": list(vec),
    }
    if task.store_order_hint:
        record["order_hint"] = list(legal_order)
    return record


# =========================
# 7. 单 signature 求解
# =========================

def configure_solver(
    solver: cp_model.CpSolver,
    *,
    workers: int,
    seed: int,
    log_search: bool,
    max_time_s: Optional[float],
) -> None:
    try:
        solver.parameters.num_workers = workers
    except (AttributeError, ValueError):
        solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.log_search_progress = log_search
    if max_time_s is not None and max_time_s > 0:
        solver.parameters.max_time_in_seconds = max_time_s


def solve_signature_task(task: SignatureTask) -> Dict[str, object]:
    paths = task_paths(task)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    if task.resume and is_done_for_current_quota(task):
        meta = safe_read_meta(task) or {
            "version": 5,
            "global_id": task.global_id,
            "rank": task.rank,
            "status": "SKIPPED_DONE",
            "complete": True,
            "layouts": layout_count(task),
        }
        meta["skipped"] = True
        return meta

    if not task.resume:
        for key in ("data", "meta", "done"):
            try:
                paths[key].unlink()
            except FileNotFoundError:
                pass

    payload = load_task_payload(task)
    layouts: List[Dict[str, object]] = list(payload.get("layouts", []))  # type: ignore[arg-type]

    # 若之前 quota 较小写过 .done，而本次 quota 增大，则继续求解。
    if paths["done"].exists() and len(layouts) < task.layouts_per_signature:
        try:
            paths["done"].unlink()
        except FileNotFoundError:
            pass

    placements = build_placements()
    data = build_v5_model(placements, task.left_counts)

    # resume：先排除已经保存过的粗空间 signature。
    seen_vectors = set()
    dedup_layouts: List[Dict[str, object]] = []
    for rec in layouts:
        vec = tuple(int(v) for v in rec.get("spatial_signature_vector", []))
        if len(vec) != len(SPATIAL_SIGNATURE_ORDER):
            continue
        if vec in seen_vectors:
            continue
        seen_vectors.add(vec)
        dedup_layouts.append(rec)
        add_spatial_signature_nogood(data, vec, f"resume{len(dedup_layouts)}")
    layouts = dedup_layouts
    payload["layouts"] = layouts
    atomic_write_json(paths["data"], payload)

    start = time.monotonic()
    deadline = None if task.task_time == 0 else start + task.task_time
    solve_calls = 0
    illegal_geometries = 0
    status_text = "UNKNOWN"
    complete = False

    if len(layouts) >= task.layouts_per_signature:
        status_text = "QUOTA_REACHED"
        complete = True
    else:
        while len(layouts) < task.layouts_per_signature:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    status_text = "TIMEOUT"
                    break
            else:
                remaining = None

            solver = cp_model.CpSolver()
            configure_solver(
                solver,
                workers=task.cp_workers,
                seed=task.seed + solve_calls * 7919,
                log_search=task.log_search,
                max_time_s=remaining,
            )
            solve_calls += 1
            status = solver.solve(data.model)
            status_name = solver.status_name(status)

            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                selected_ids = [
                    i for i, var in enumerate(data.x) if solver.value(var) == 1
                ]
                legal_order = compute_legal_order(placements, selected_ids)
                if legal_order is None:
                    # 必要支撑约束仍不足以保证全局顺序可达；只排除当前几何解继续找。
                    illegal_geometries += 1
                    add_exact_geometry_nogood(data, selected_ids)
                    continue

                # 完整独立校验，并从几何解直接推导 gap/missing/empty。
                gap_row, missing_category, empty_cols, legal_order = validate_v5_solution(
                    placements,
                    selected_ids,
                    task.left_counts,
                )
                rec = make_layout_record(
                    task,
                    placements,
                    selected_ids,
                    gap_row,
                    missing_category,
                    empty_cols,
                    legal_order,
                )
                vec = tuple(int(v) for v in rec["spatial_signature_vector"])  # type: ignore[arg-type]

                if vec in seen_vectors:
                    # 理论上前面的 no-good 已避免；保险处理。
                    add_spatial_signature_nogood(data, vec, f"dup{solve_calls}")
                    continue

                seen_vectors.add(vec)
                layouts.append(rec)
                payload["layouts"] = layouts
                atomic_write_json(paths["data"], payload)

                add_spatial_signature_nogood(data, vec, f"found{len(layouts)}")

                if len(layouts) >= task.layouts_per_signature:
                    status_text = "QUOTA_REACHED"
                    complete = True
                    break
                continue

            if status == cp_model.INFEASIBLE:
                # 在已排除已有 coarse signatures 后，证明没有更多满足条件的布局。
                status_text = "EXHAUSTED"
                complete = True
                break

            # UNKNOWN / 其他非可行状态通常就是本次 time limit 内没找到。
            status_text = "TIMEOUT" if deadline is not None else status_name
            break

    elapsed = time.monotonic() - start
    meta = {
        "version": 5,
        "global_id": task.global_id,
        "rank": task.rank,
        "target_left_counts": list(task.left_counts),
        "target_left_by_category": signature_dict(task.left_counts),
        "weight": task.weight,
        "probability": task.probability,
        "status": status_text,
        "complete": complete,
        "layouts": len(layouts),
        "requested_layouts": task.layouts_per_signature,
        "covered": len(layouts) > 0,
        "elapsed_s": elapsed,
        "solve_calls": solve_calls,
        "illegal_geometries_rejected": illegal_geometries,
        "task_time": task.task_time,
        "cp_workers": task.cp_workers,
        "seed": task.seed,
        "skipped": False,
    }
    atomic_write_json(paths["meta"], meta)

    if complete:
        atomic_write_text(paths["done"], status_text + "\n")
    else:
        try:
            paths["done"].unlink()
        except FileNotFoundError:
            pass

    return meta


# =========================
# 8. 进度 / 汇总
# =========================

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


def progress_bar(done: int, total: int, width: int = 28) -> str:
    frac = 1.0 if total == 0 else done / total
    n = min(width, max(0, int(frac * width)))
    return "[" + "#" * n + "-" * (width - n) + "]"


def scan_task_state(task: SignatureTask) -> Dict[str, object]:
    meta = safe_read_meta(task) or {}
    n = layout_count(task)
    done = is_done_for_current_quota(task)
    return {
        "done": done,
        "layouts": n,
        "covered": n > 0,
        "status": str(meta.get("status", "PENDING")),
        "elapsed_s": meta.get("elapsed_s"),
    }


def build_progress_snapshot(
    *,
    tasks: Sequence[SignatureTask],
    done_total: int,
    covered_total: int,
    layouts_total: int,
    elapsed_run: float,
    new_done: int,
    running: Dict[Future, Tuple[SignatureTask, float]],
    total_probability_weight: int,
) -> Dict[str, object]:
    selected_total = len(tasks)
    remaining = max(0, selected_total - done_total)
    pct = 100.0 if selected_total == 0 else 100.0 * done_total / selected_total
    rate = new_done / elapsed_run if elapsed_run > 1e-9 else 0.0
    eta = remaining / rate if rate > 1e-12 else None

    # covered mass：当前 shard 已经至少有一个布局的 signature 对全体随机 signature 概率的贡献。
    covered_weight = 0
    selected_weight = sum(t.weight for t in tasks)
    for t in tasks:
        if layout_count(t) > 0:
            covered_weight += t.weight

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


def print_snapshot(s: Dict[str, object], jobs: int) -> None:
    done = int(s["done_total"])
    total = int(s["selected_total"])
    pct = float(s["progress_pct"])
    elapsed = float(s["elapsed_run_s"])
    rate = float(s["throughput_tasks_per_s"])
    eta = s["eta_s"]

    print(
        f"[GLOBAL] {progress_bar(done, total)} "
        f"done={done}/{total} ({pct:.2f}%) "
        f"covered={s['covered_signatures']} layouts={s['layouts_total']} "
        f"running={len(s['running'])}/{jobs} remaining={s['remaining']} "
        f"elapsed={fmt_duration(elapsed)} rate={rate*60:.2f} sig/min "
        f"ETA={fmt_duration(eta if isinstance(eta, (int, float)) else None)}",
        flush=True,
    )
    print(
        f"[COVERAGE] selected_mass(global)={100*float(s['selected_probability_mass_global']):.3f}% "
        f"covered_mass(global contribution)={100*float(s['covered_probability_mass_global']):.3f}%",
        flush=True,
    )

    running = s["running"]
    if running:
        longest = running[0]
        print(
            f"[LONGEST] id={longest['global_id']} rank={longest['rank']} "
            f"left={','.join(map(str, longest['left_counts']))} "
            f"elapsed={fmt_duration(float(longest['elapsed_s']))}",
            flush=True,
        )


def export_library(tasks: Sequence[SignatureTask], path: Path) -> int:
    """把当前 shard / 筛选范围里已经找到的布局扁平导出成 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as f:
        for task in sorted(tasks, key=lambda t: t.rank):
            payload = load_task_payload(task)
            layouts = payload.get("layouts", [])
            if not isinstance(layouts, list):
                continue
            for li, layout in enumerate(layouts):
                if not isinstance(layout, dict):
                    continue
                rec = {
                    "version": 5,
                    "global_id": task.global_id,
                    "rank": task.rank,
                    "layout_index": li,
                    "target_left_counts": list(task.left_counts),
                    "target_left_by_category": signature_dict(task.left_counts),
                    "weight": task.weight,
                    "probability": task.probability,
                }
                rec.update(layout)
                f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
    os.replace(tmp, path)
    return count


# =========================
# 9. main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="14x10、34块、13个四色完整行的260分代表性盘面库生成器 V5"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("layouts_260_v5"),
        help="输出目录，默认 layouts_260_v5",
    )
    parser.add_argument(
        "--left-count",
        type=int,
        default=17,
        help="目标盘 LEFT 方块数。V5 当前要求左右相等，因此默认/建议17",
    )
    parser.add_argument(
        "--top-signatures",
        type=int,
        default=7000,
        help="按随机组合权重只跑前 N 个 signature；0=全部。默认7000",
    )
    parser.add_argument(
        "--signature",
        type=parse_signature,
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

    if args.left_count * 2 != TOTAL_PIECES:
        raise SystemExit(
            f"V5 当前实现要求目标左右数量相等，所以 --left-count 必须为 {TOTAL_PIECES // 2}"
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

    specs, total_probability_weight = generate_signature_specs(args.left_count)
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
        selected_specs = specs if args.top_signatures == 0 else specs[: args.top_signatures]

    # top-N 过滤在 shard 前执行；不同机器使用相同 top-N 时严格互斥且并集等于 top-N。
    selected_mass_before_shard = sum(s.weight for s in selected_specs) / total_probability_weight
    selected_specs = [
        s for s in selected_specs
        if s.global_id % args.shard_count == args.shard_index
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    placements = build_placements()
    write_catalog(args.output_dir / "placement_catalog.json", placements)

    tasks = [
        SignatureTask(
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

    shard_tag = f"shard{args.shard_index:02d}of{args.shard_count:02d}"
    progress_path = args.output_dir / f"progress_{shard_tag}.json"
    summary_path = args.output_dir / f"summary_{shard_tag}.json"
    library_path = args.output_dir / f"library_{shard_tag}.jsonl"

    print("V5：260分代表性盘面库生成；不做全集枚举。", flush=True)
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
        f"单 signature 本次时间：{'不限时' if args.task_time == 0 else f'{args.task_time:g}s'}; "
        f"heartbeat={args.heartbeat:g}s",
        flush=True,
    )

    # 启动前扫描。
    done_total = 0
    covered_total = 0
    layouts_total = 0
    pending: List[SignatureTask] = []
    for task in tasks:
        st = scan_task_state(task)
        if bool(st["done"]):
            done_total += 1
        else:
            pending.append(task)
        n = int(st["layouts"])
        layouts_total += n
        if n > 0:
            covered_total += 1

    print(
        f"启动前：done={done_total}/{len(tasks)}, covered={covered_total}, "
        f"layouts={layouts_total}, pending={len(pending)}",
        flush=True,
    )

    initial_done_total = done_total
    start_run = time.monotonic()
    initial_snapshot = build_progress_snapshot(
        tasks=tasks,
        done_total=done_total,
        covered_total=covered_total,
        layouts_total=layouts_total,
        elapsed_run=0.0,
        new_done=0,
        running={},
        total_probability_weight=total_probability_weight,
    )
    print_snapshot(initial_snapshot, args.jobs)
    atomic_write_json(
        progress_path,
        {
            "version": 5,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            **initial_snapshot,
        },
    )

    if args.status_only:
        exported = export_library(tasks, library_path)
        print(f"--status-only：不启动求解；当前导出 layouts={exported}", flush=True)
        print(f"library: {library_path.resolve()}", flush=True)
        return

    new_done = 0
    errors = 0
    returned = 0
    queue: Deque[SignatureTask] = deque(pending)
    running: Dict[Future, Tuple[SignatureTask, float]] = {}

    def submit_one(executor: ProcessPoolExecutor) -> None:
        if not queue:
            return
        task = queue.popleft()
        fut = executor.submit(solve_signature_task, task)
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
                    # 这里 task 已完成写盘，直接读取返回 meta / 文件。
                    try:
                        meta = fut.result()
                        n = int(meta.get("layouts", 0))
                        complete = bool(meta.get("complete", False))
                        status = str(meta.get("status", "?"))
                        elapsed_s = float(meta.get("elapsed_s", 0.0))

                        # 重新汇总全局计数，避免 timeout/resume 导致增量计数出错。
                        done_total = 0
                        covered_total = 0
                        layouts_total = 0
                        for t in tasks:
                            st = scan_task_state(t)
                            if bool(st["done"]):
                                done_total += 1
                            tn = int(st["layouts"])
                            layouts_total += tn
                            if tn > 0:
                                covered_total += 1
                        new_done = max(0, done_total - initial_done_total)

                        print(
                            f"[RETURN id={task.global_id:05d} rank={task.rank:05d}] "
                            f"left={','.join(map(str, task.left_counts))} "
                            f"status={status} layouts={n}/{task.layouts_per_signature} "
                            f"complete={complete} time={fmt_duration(elapsed_s)}",
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
                    # 精确重扫状态；虽然是 O(N) 文件读，但 heartbeat 频率低，N~几千可接受。
                    done_total = 0
                    covered_total = 0
                    layouts_total = 0
                    for t in tasks:
                        st = scan_task_state(t)
                        if bool(st["done"]):
                            done_total += 1
                        tn = int(st["layouts"])
                        layouts_total += tn
                        if tn > 0:
                            covered_total += 1

                    elapsed_run = now - start_run
                    snapshot = build_progress_snapshot(
                        tasks=tasks,
                        done_total=done_total,
                        covered_total=covered_total,
                        layouts_total=layouts_total,
                        elapsed_run=elapsed_run,
                        new_done=max(0, done_total - initial_done_total),
                        running=running,
                        total_probability_weight=total_probability_weight,
                    )
                    print_snapshot(snapshot, args.jobs)
                    atomic_write_json(
                        progress_path,
                        {
                            "version": 5,
                            "shard_count": args.shard_count,
                            "shard_index": args.shard_index,
                            **snapshot,
                        },
                    )
                    while next_heartbeat <= now:
                        next_heartbeat += args.heartbeat

    elapsed_run = time.monotonic() - start_run

    # 最终精确扫描。
    done_total = 0
    covered_total = 0
    layouts_total = 0
    status_counts = Counter()
    for task in tasks:
        st = scan_task_state(task)
        if bool(st["done"]):
            done_total += 1
        n = int(st["layouts"])
        layouts_total += n
        if n > 0:
            covered_total += 1
        status_counts[str(st["status"])] += 1

    final_snapshot = build_progress_snapshot(
        tasks=tasks,
        done_total=done_total,
        covered_total=covered_total,
        layouts_total=layouts_total,
        elapsed_run=elapsed_run,
        new_done=max(0, done_total - initial_done_total),
        running={},
        total_probability_weight=total_probability_weight,
    )

    exported = export_library(tasks, library_path)
    summary = {
        "version": 5,
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
    atomic_write_json(summary_path, summary)
    atomic_write_json(
        progress_path,
        {
            "version": 5,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            **final_snapshot,
        },
    )

    print("\n===== V5 汇总 =====", flush=True)
    print_snapshot(final_snapshot, args.jobs)
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
