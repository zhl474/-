#!/home/zhl/fr3env/fr3env/bin/python
"""生成“所有方块均有支撑，但允许后来填洞”的最小规则示意图。"""

# ============================== 可直接修改的参数 ==============================

输出目录名 = "洞内摆放规则示意"
图片分辨率 = 220

# =============================================================================

import json
import os
from pathlib import Path
import sys


脚本目录 = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tetris_hole_demo_matplotlib")

# 复用原可视化脚本的中文字体和类别配色。
sys.path.insert(0, str(脚本目录))
from tetris_layout_visual_check import 类别样式, 配置中文字体

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


# 这是专门说明规则的最小盘面，不对应某一轮识别结果。
# 所有已摆方块均为标准四格方块：左右为竖直 line，顶部为水平 line，底部和待填块为 square。
已摆方块 = [
    {
        "名称": "左侧立柱",
        "类别": "line",
        "颜色": "#5E35B1",
        "cells": ((4, 1), (4, 2), (4, 3), (4, 4)),
        "支撑说明": "接触第一行",
    },
    {
        "名称": "右侧立柱",
        "类别": "line",
        "颜色": "#3949AB",
        "cells": ((7, 1), (7, 2), (7, 3), (7, 4)),
        "支撑说明": "接触第一行",
    },
    {
        "名称": "底部支撑",
        "类别": "square",
        "颜色": "#FDD835",
        "cells": ((5, 1), (6, 1), (5, 2), (6, 2)),
        "支撑说明": "接触第一行",
    },
    {
        "名称": "顶部横梁",
        "类别": "line",
        "颜色": "#43A047",
        "cells": ((4, 5), (5, 5), (6, 5), (7, 5)),
        "支撑说明": "由左右立柱支撑",
    },
]

待填方块 = {
    "名称": "后来填入",
    "类别": "square",
    "颜色": 类别样式["square"]["颜色"],
    "cells": ((5, 3), (6, 3), (5, 4), (6, 4)),
    "支撑说明": "由底部方块支撑",
}


def 每列最低格(cells):
    最低行 = {}
    for col, row in cells:
        最低行[col] = min(row, 最低行.get(col, row))
    return sorted(最低行.items())


def 校验顺序不悬空(placements):
    """按项目的单点 OR 支撑规则逐块校验示意顺序。"""
    occupied = set()
    results = []
    for placement in placements:
        cells = set(placement["cells"])
        if cells & occupied:
            raise ValueError(f"{placement['名称']} 与已有方块重叠")
        touches_first_row = min(row for _, row in cells) == 1
        support_cells = [
            (col, row - 1)
            for col, row in 每列最低格(cells)
            if (col, row - 1) in occupied
        ]
        legal = touches_first_row or bool(support_cells)
        if not legal:
            raise ValueError(f"{placement['名称']} 悬空，示意图无效")
        results.append({
            "名称": placement["名称"],
            "接触第一行": touches_first_row,
            "实际支撑格": support_cells,
            "合法": legal,
        })
        occupied.update(cells)
    return results


def 绘制方块(ax, placement, zorder=3):
    cells = set(placement["cells"])
    color = placement["颜色"]
    for col, row in cells:
        ax.add_patch(Rectangle(
            (col - 0.5, row - 0.5),
            1,
            1,
            facecolor=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=zorder,
        ))
        for start, end, neighbor in (
            ((col - 0.5, row - 0.5), (col - 0.5, row + 0.5), (col - 1, row)),
            ((col + 0.5, row - 0.5), (col + 0.5, row + 0.5), (col + 1, row)),
            ((col - 0.5, row - 0.5), (col + 0.5, row - 0.5), (col, row - 1)),
            ((col - 0.5, row + 0.5), (col + 0.5, row + 0.5), (col, row + 1)),
        ):
            if neighbor not in cells:
                ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    color="#202020",
                    linewidth=1.8,
                    zorder=zorder + 1,
                )
    center_col = sum(col for col, _ in cells) / 4.0
    center_row = sum(row for _, row in cells) / 4.0
    ax.text(
        center_col,
        center_row,
        placement["名称"],
        ha="center",
        va="center",
        fontsize=8.5,
        color="white",
        weight="bold",
        bbox={"boxstyle": "round,pad=0.16", "facecolor": "#111111", "alpha": 0.58},
        zorder=zorder + 3,
    )


def 绘制一张(path, 填入后):
    figure, ax = plt.subplots(figsize=(7.8, 7.2))
    ax.set_xlim(2.5, 8.5)
    ax.set_ylim(0.5, 6.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(3, 9))
    ax.set_yticks(range(1, 7))
    ax.set_xlabel("托盘列")
    ax.set_ylabel("托盘行（第一行在最下方）")
    ax.grid(color="#B8B8B8", linewidth=0.7)
    ax.set_axisbelow(True)

    # 强调第 1 行是地基，避免局部截取造成悬空误解。
    ax.axhspan(0.5, 1.5, color="#E8F5E9", alpha=0.7, zorder=0)
    ax.text(2.72, 1.0, "第一行", color="#1B5E20", fontsize=9, weight="bold")

    for placement in 已摆方块:
        绘制方块(ax, placement)

    hole_cells = set(待填方块["cells"])
    if 填入后:
        绘制方块(ax, 待填方块, zorder=8)
        标题 = "填入后：方块有下方支撑，合法填入洞中"
        说明 = (
            "待填方块的下方两格由“底部支撑”承托；"
            "其他方块也都接触第一行或已有明确支撑，因此没有任何悬空方块。"
        )
    else:
        for col, row in hole_cells:
            ax.add_patch(Rectangle(
                (col - 0.5, row - 0.5),
                1,
                1,
                facecolor="#FFF7E6",
                edgecolor="#D32F2F",
                hatch="///",
                linewidth=1.4,
                zorder=2,
            ))
        ax.text(
            5.5,
            3.5,
            "待填 2×2 洞位",
            ha="center",
            va="center",
            fontsize=10.5,
            color="#B71C1C",
            weight="bold",
            zorder=7,
        )
        ax.annotate(
            "洞底已经有支撑",
            xy=(5.5, 2.45),
            xytext=(7.7, 2.0),
            arrowprops={"arrowstyle": "->", "color": "#1565C0", "lw": 1.8},
            fontsize=9,
            color="#0D47A1",
            ha="center",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#1565C0"},
            zorder=10,
        )
        标题 = "填入前：从第一行开始搭建，中央留下可支撑洞位"
        说明 = (
            "左右立柱和底部方块均直接接触第一行；顶部横梁由左右立柱承托。"
            "所有已摆方块都不悬空。"
        )

    figure.suptitle(标题, fontsize=13, weight="bold", y=0.91)
    figure.legend(
        handles=[
            Patch(facecolor=item["颜色"], edgecolor="#202020", label=item["名称"])
            for item in 已摆方块
        ] + [
            Patch(facecolor=待填方块["颜色"], edgecolor="#202020", label=待填方块["名称"])
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        fontsize=8,
    )
    figure.text(
        0.5,
        0.025,
        说明 + "\n本图只用于说明：规则禁止悬空，但不禁止在有支撑的洞中后放方块。",
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#333333",
    )
    figure.tight_layout(rect=(0.03, 0.11, 0.97, 0.84))
    figure.savefig(path, dpi=图片分辨率, bbox_inches="tight")
    plt.close(figure)


def main():
    配置中文字体()
    填入前校验 = 校验顺序不悬空(已摆方块)
    填入后校验 = 校验顺序不悬空([*已摆方块, 待填方块])

    输出目录 = 脚本目录 / "可视化检查" / 输出目录名
    输出目录.mkdir(parents=True, exist_ok=True)
    # 覆盖上一版局部截取图，避免目录里同时保留可能被误解为悬空的旧图片。
    填入前路径 = 输出目录 / "01_周围已摆_中央留洞.png"
    填入后路径 = 输出目录 / "02_向洞内填入目标方块.png"
    绘制一张(填入前路径, False)
    绘制一张(填入后路径, True)

    元数据 = {
        "用途": "只说明不悬空条件下允许后来填洞，不对应某一轮识别结果",
        "已摆方块": 已摆方块,
        "待填方块": 待填方块,
        "填入前逐块支撑校验": 填入前校验,
        "填入后逐块支撑校验": 填入后校验,
        "结论": "所有方块均合法受支撑；后来填洞符合当前单点 OR 支撑规则",
        "图片": [str(填入前路径), str(填入后路径)],
    }
    (输出目录 / "示意图说明.json").write_text(
        json.dumps(元数据, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"填入前：{填入前路径}")
    print(f"填入后：{填入后路径}")
    for item in 填入后校验:
        print(
            f"{item['名称']}：合法={item['合法']}，"
            f"接触第一行={item['接触第一行']}，支撑格={item['实际支撑格']}"
        )


if __name__ == "__main__":
    main()
