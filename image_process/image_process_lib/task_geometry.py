"""正式任务盘面的四格几何与单点支撑关系。"""

from dataclasses import dataclass
import math
from typing import Iterable, Sequence, Tuple


TASK_BOARD_ROW_COUNT = 14
TASK_BOARD_COL_COUNT = 10


def _item_value(item, name):
    if isinstance(item, dict):
        return item[name]
    return getattr(item, name)


def _normalize_target_id(value) -> int:
    if isinstance(value, bool):
        raise ValueError("target ID 必须是非负整数")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("target ID 必须是非负整数") from exc
    if not math.isfinite(number) or not number.is_integer() or number < 0.0:
        raise ValueError("target ID 必须是非负整数")
    return int(number)


def normalize_cells(raw_cells, label="cells") -> Tuple[Tuple[int, int], ...]:
    """把四个占用格严格规范为排序后的 ``(列, 行)`` 元组。"""
    if isinstance(raw_cells, (str, bytes)):
        raise ValueError(f"{label} 必须包含四个 (列, 行) 格子")
    try:
        values = list(raw_cells)
    except TypeError as exc:
        raise ValueError(f"{label} 必须包含四个 (列, 行) 格子") from exc
    if len(values) != 4:
        raise ValueError(f"{label} 必须恰好包含四个格子")

    cells = []
    for offset, cell in enumerate(values):
        try:
            col_value, row_value = cell
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} 第 {offset + 1} 个格子必须是 (列, 行)") from exc
        if isinstance(col_value, bool) or isinstance(row_value, bool):
            raise ValueError(f"{label} 只能使用整数行列")
        try:
            col_float = float(col_value)
            row_float = float(row_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} 只能使用整数行列") from exc
        if (
            not math.isfinite(col_float)
            or not math.isfinite(row_float)
            or not col_float.is_integer()
            or not row_float.is_integer()
        ):
            raise ValueError(f"{label} 只能使用整数行列")
        col = int(col_float)
        row = int(row_float)
        if not 1 <= col <= TASK_BOARD_COL_COUNT or not 1 <= row <= TASK_BOARD_ROW_COUNT:
            raise ValueError(
                f"{label} 格子 ({col}, {row}) 超出 "
                f"{TASK_BOARD_COL_COUNT}×{TASK_BOARD_ROW_COUNT} 托盘范围"
            )
        cells.append((col, row))
    if len(set(cells)) != 4:
        raise ValueError(f"{label} 内不能包含重复格子")
    return tuple(sorted(cells))


def bottom_profile_cells(
    cells: Iterable[Tuple[int, int]],
) -> Tuple[Tuple[int, int], ...]:
    """返回方块在每个占用列中的最低格。"""
    minimum_row_by_col = {}
    for col, row in cells:
        if col not in minimum_row_by_col or row < minimum_row_by_col[col]:
            minimum_row_by_col[col] = row
    return tuple(sorted(minimum_row_by_col.items()))


@dataclass(frozen=True)
class SupportGraph:
    """按传入 target 顺序使用稠密位编号的支撑图。"""

    target_ids: Tuple[int, ...]
    cells: Tuple[Tuple[Tuple[int, int], ...], ...]
    first_layer_mask: int
    lower_masks: Tuple[int, ...]
    unlock_masks: Tuple[int, ...]

    @property
    def all_targets_mask(self) -> int:
        return (1 << len(self.target_ids)) - 1

    def is_available(self, dense_index: int, placed_mask: int) -> bool:
        bit = 1 << dense_index
        return bool(
            self.first_layer_mask & bit
            or self.lower_masks[dense_index] & int(placed_mask)
        )


def build_support_graph(targets: Sequence) -> SupportGraph:
    """校验完整盘面并建立比赛规则所需的 OR 支撑位掩码。"""
    if not targets:
        raise ValueError("正式任务盘面不能为空")

    target_ids = []
    normalized_cells = []
    owner = {}
    for dense_index, target in enumerate(targets):
        try:
            target_id = _normalize_target_id(_item_value(target, "index"))
            cells = normalize_cells(
                _item_value(target, "cells"),
                f"目标 {target_id} cells",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"目标 {dense_index} 的四格几何无效") from exc
        if target_id in target_ids:
            raise ValueError(f"摆放目标序号不能重复：{target_id}")
        target_ids.append(target_id)
        normalized_cells.append(cells)
        for cell in cells:
            previous = owner.get(cell)
            if previous is not None:
                raise ValueError(
                    f"目标 {target_id} 与目标 {target_ids[previous]} 在格子 {cell} 重叠"
                )
            owner[cell] = dense_index

    first_layer_mask = 0
    lower_masks = [0] * len(targets)
    unlock_masks = [0] * len(targets)
    for dense_index, cells in enumerate(normalized_cells):
        if min(row for _, row in cells) == 1:
            first_layer_mask |= 1 << dense_index
            continue
        supporter_mask = 0
        for col, row in bottom_profile_cells(cells):
            supporter = owner.get((col, row - 1))
            if supporter is not None and supporter != dense_index:
                supporter_mask |= 1 << supporter
        if supporter_mask == 0:
            raise ValueError(f"目标 {target_ids[dense_index]} 不在第一层且没有下方支撑候选")
        lower_masks[dense_index] = supporter_mask
        bits = supporter_mask
        while bits:
            least_bit = bits & -bits
            supporter = least_bit.bit_length() - 1
            unlock_masks[supporter] |= 1 << dense_index
            bits ^= least_bit

    # 支撑关系是单调 OR 条件；反复加入所有已解锁目标即可验证整盘可达。
    placed = 0
    available = first_layer_mask
    all_mask = (1 << len(targets)) - 1
    while available & ~placed:
        newly_placed = available & ~placed
        placed |= newly_placed
        bits = newly_placed
        while bits:
            least_bit = bits & -bits
            supporter = least_bit.bit_length() - 1
            available |= unlock_masks[supporter]
            bits ^= least_bit
    if placed != all_mask:
        unreachable = [
            target_ids[index]
            for index in range(len(targets))
            if not placed & (1 << index)
        ]
        raise ValueError(f"盘面不存在完整合法摆放顺序，不可达目标：{unreachable}")

    return SupportGraph(
        target_ids=tuple(target_ids),
        cells=tuple(normalized_cells),
        first_layer_mask=first_layer_mask,
        lower_masks=tuple(lower_masks),
        unlock_masks=tuple(unlock_masks),
    )


def validate_dense_target_sequence(
    sequence: Sequence[int],
    support_graph: SupportGraph,
) -> None:
    """校验一条使用稠密 target 下标的完整摆放顺序。"""
    if len(sequence) != len(support_graph.target_ids):
        raise ValueError("摆放顺序长度与目标数量不一致")
    placed = 0
    for step, dense_index in enumerate(sequence, start=1):
        if not 0 <= int(dense_index) < len(support_graph.target_ids):
            raise ValueError(f"第 {step} 步目标下标越界：{dense_index}")
        bit = 1 << int(dense_index)
        if placed & bit:
            raise ValueError(f"第 {step} 步重复摆放目标 {dense_index}")
        if not support_graph.is_available(int(dense_index), placed):
            raise ValueError(f"第 {step} 步目标 {dense_index} 尚未获得支撑")
        placed |= bit


def build_stable_legal_target_sequence(targets: Sequence) -> Tuple[int, ...]:
    """生成稳定的合法稠密下标序列。

    每一步在已解锁且未摆放的目标中选最小 target ID，使没有
    ``order_hint`` 的 V5 盘面也能为固定顺序对照项提供可重现输入。
    """
    graph = build_support_graph(targets)
    placed = 0
    sequence = []
    while len(sequence) < len(graph.target_ids):
        available = [
            dense_index
            for dense_index, target_id in enumerate(graph.target_ids)
            if not placed & (1 << dense_index)
            and graph.is_available(dense_index, placed)
        ]
        if not available:
            raise RuntimeError("构造稳定合法顺序时没有已解锁目标")
        selected = min(
            available,
            key=lambda dense_index: (
                graph.target_ids[dense_index],
                dense_index,
            ),
        )
        sequence.append(selected)
        placed |= 1 << selected
    result = tuple(sequence)
    validate_dense_target_sequence(result, graph)
    return result
