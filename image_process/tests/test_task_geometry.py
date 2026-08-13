import pytest

from image_process_lib.task_geometry import (
    build_stable_legal_target_sequence,
    build_support_graph,
    validate_dense_target_sequence,
)


def _or_support_layout():
    return [
        {
            "index": 10,
            "cells": ((1, 1), (1, 2), (2, 1), (2, 2)),
        },
        {
            "index": 11,
            "cells": ((3, 1), (3, 2), (4, 1), (4, 2)),
        },
        {
            "index": 12,
            "cells": ((2, 3), (2, 4), (3, 3), (3, 4)),
        },
    ]


def test_or支撑任一候选完成即可解锁上层目标():
    graph = build_support_graph(_or_support_layout())

    assert graph.lower_masks[2] == (1 << 0) | (1 << 1)
    assert graph.is_available(2, 1 << 0)
    assert graph.is_available(2, 1 << 1)
    validate_dense_target_sequence((0, 2, 1), graph)


def test无支撑与重叠盘面在加载阶段明确失败():
    unsupported = [
        {"index": 0, "cells": ((1, 2), (1, 3), (2, 2), (2, 3))},
    ]
    with pytest.raises(ValueError, match="没有下方支撑"):
        build_support_graph(unsupported)

    overlapping = _or_support_layout()
    overlapping[1] = {"index": 11, "cells": overlapping[0]["cells"]}
    with pytest.raises(ValueError, match="重叠"):
        build_support_graph(overlapping)


def test摆放顺序会拒绝未解锁和重复目标():
    graph = build_support_graph(_or_support_layout())
    with pytest.raises(ValueError, match="尚未获得支撑"):
        validate_dense_target_sequence((2, 0, 1), graph)
    with pytest.raises(ValueError, match="重复"):
        validate_dense_target_sequence((0, 0, 1), graph)


def test稳定合法顺序每轮选已解锁的最小目标ID():
    layout = _or_support_layout()
    layout[0]["index"] = 30
    layout[1]["index"] = 10
    layout[2]["index"] = 20

    sequence = build_stable_legal_target_sequence(layout)

    assert sequence == (1, 2, 0)
    graph = build_support_graph(layout)
    validate_dense_target_sequence(sequence, graph)
