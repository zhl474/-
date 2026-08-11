#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只读抽样检查尚未完成的 260 分版型，并生成中文盘面图片。"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ============================== 可直接修改的参数 ==============================

脚本目录 = Path(__file__).resolve().parent
结果目录 = 脚本目录 / "layouts_260_A"
可视化输出根目录 = 脚本目录 / "可视化检查"

# 每个非空分区抽取几盘；默认抽取较早、中间和较新的结果。
每分区抽样数 = 3
最多绘制盘数 = 36
每页盘数 = 6

# 防止将来单个结果文件非常大时，检查程序本身运行过久。
单文件最多检查记录数 = 20000

显示方块编号 = True
显示求解器中心点 = True
图片分辨率 = 180

# ============================================================================


# Matplotlib 默认配置目录不可写时会产生警告，这里只把缓存放到临时目录。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tetris_layout_visual_matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402


行数 = 14
列数 = 10
类别顺序 = [
    "L_blue",
    "z_green",
    "L_yellow",
    "square",
    "z_blue",
    "T",
    "line",
]

类别样式 = {
    "L_blue": {"名称": "四格L字形右", "简称": "L右", "颜色": "#7B2CBF"},
    "z_green": {"名称": "四格Z字形右", "简称": "Z右", "颜色": "#19C86B"},
    "L_yellow": {"名称": "四格L字形左", "简称": "L左", "颜色": "#FFE500"},
    "square": {"名称": "四格田字形", "简称": "田", "颜色": "#F58220"},
    "z_blue": {"名称": "四格Z字形左", "简称": "Z左", "颜色": "#087FC3"},
    "T": {"名称": "四格山字形", "简称": "山", "颜色": "#8B3F00"},
    "line": {"名称": "四格一字形", "简称": "一", "颜色": "#F04444"},
}


def 配置中文字体() -> None:
    """选择当前系统中可用的中文字体。"""
    # 当前环境的 Matplotlib 字体缓存没有自动索引 Noto CJK，按绝对路径注册。
    noto_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if noto_path.is_file():
        font_manager.fontManager.addfont(str(noto_path))
        family = font_manager.FontProperties(fname=str(noto_path)).get_name()
        plt.rcParams["font.family"] = family
    else:
        plt.rcParams["font.sans-serif"] = ["Droid Sans Fallback", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def 解析分区文件名(path: Path) -> Tuple[Optional[int], Optional[str]]:
    match = re.match(
        r"gap(?P<row>\d+)_missing_(?P<category>.+?)(?:\.partial)?\.jsonl$",
        path.name,
    )
    if match is None:
        return None, None
    return int(match.group("row")), match.group("category")


def 方块底部格(cells: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """返回每一列中属于该方块的最低格。"""
    最低行: Dict[int, int] = {}
    for col, row in cells:
        if col not in 最低行 or row < 最低行[col]:
            最低行[col] = row
    return sorted(最低行.items())


def 顺序可以执行(
    order: Sequence[int],
    placements_by_id: Dict[int, dict],
) -> bool:
    """按搜索器的单点支撑规则检查一个摆放顺序。"""
    已占用 = set()
    for pid in order:
        placement = placements_by_id.get(pid)
        if placement is None:
            return False
        cells = [tuple(cell) for cell in placement["cells"]]
        if min(row for _, row in cells) > 1:
            if not any((col, row - 1) in 已占用 for col, row in 方块底部格(cells)):
                return False
        已占用.update(cells)
    return True


def 计算支撑优先顺序(placements: Sequence[dict]) -> Optional[List[int]]:
    """只在内存中计算一个可执行顺序，不改写搜索结果。"""
    剩余 = {int(item["id"]): item for item in placements}
    已占用 = set()
    order: List[int] = []

    while 剩余:
        可放置 = []
        for pid, placement in 剩余.items():
            cells = [tuple(cell) for cell in placement["cells"]]
            接触底层 = min(row for _, row in cells) == 1
            已有支撑 = any(
                (col, row - 1) in 已占用 for col, row in 方块底部格(cells)
            )
            if 接触底层 or 已有支撑:
                可放置.append((min(row for _, row in cells), pid))

        if not 可放置:
            return None

        _, pid = min(可放置)
        placement = 剩余.pop(pid)
        order.append(pid)
        已占用.update(tuple(cell) for cell in placement["cells"])

    return order


def 检查一条记录(
    payload: dict,
    placement_catalog: Dict[int, dict],
    gap_row: Optional[int],
    missing_category: Optional[str],
) -> dict:
    """检查一条解，并返回绘图和汇总需要的信息。"""
    errors: List[str] = []
    warnings: List[str] = []
    ids = payload.get("ids")
    if not isinstance(ids, list):
        return {"可绘制": False, "错误": ["缺少 ids 列表"], "警告": warnings}
    if len(ids) != 34:
        errors.append(f"方块数量为 {len(ids)}，应为 34")
    if any(not isinstance(pid, int) for pid in ids):
        return {"可绘制": False, "错误": errors + ["ids 中存在非整数"], "警告": warnings}
    if len(set(ids)) != len(ids):
        errors.append("ids 中存在重复方块")

    unknown_ids = [pid for pid in ids if pid not in placement_catalog]
    if unknown_ids:
        return {
            "可绘制": False,
            "错误": errors + [f"目录中找不到 ID：{unknown_ids[:5]}"],
            "警告": warnings,
        }

    placements = [placement_catalog[pid] for pid in ids]
    owner: Dict[Tuple[int, int], int] = {}
    overlap_cells = []
    out_of_range_cells = []
    piece_size_errors = []

    for placement in placements:
        cells = [tuple(cell) for cell in placement["cells"]]
        if len(cells) != 4 or len(set(cells)) != 4:
            piece_size_errors.append(int(placement["id"]))
        for cell in cells:
            col, row = cell
            if not (1 <= col <= 列数 and 1 <= row <= 行数):
                out_of_range_cells.append(cell)
            if cell in owner:
                overlap_cells.append(cell)
            else:
                owner[cell] = int(placement["id"])

    if piece_size_errors:
        errors.append(f"存在非四格方块：{piece_size_errors[:5]}")
    if out_of_range_cells:
        errors.append(f"存在越界格：{out_of_range_cells[:5]}")
    if overlap_cells:
        errors.append(f"存在重叠格：{overlap_cells[:5]}")
    if len(owner) != 136:
        errors.append(f"实际覆盖 {len(owner)} 格，应为 136 格")

    counts = Counter(item["category"] for item in placements)
    if missing_category in 类别顺序:
        for category in 类别顺序:
            expected = 4 if category == missing_category else 5
            if counts[category] != expected:
                errors.append(f"{category} 数量 {counts[category]}，应为 {expected}")
    else:
        warnings.append("无法从文件名确定缺少类别")

    row_occupancies = {
        row: sum((col, row) in owner for col in range(1, 列数 + 1))
        for row in range(1, 行数 + 1)
    }
    detected_gap_rows = [row for row, count in row_occupancies.items() if count == 6]
    abnormal_rows = [
        row
        for row, count in row_occupancies.items()
        if count != (6 if row == gap_row else 10)
    ]
    if gap_row is None:
        warnings.append("无法从文件名确定缺口行")
    elif abnormal_rows:
        errors.append(f"行占用错误：{[(row, row_occupancies[row]) for row in abnormal_rows]}")

    row_category_counts = {}
    for row in range(1, 行数 + 1):
        categories = {
            item["category"]
            for item in placements
            if any(int(cell[1]) == row for cell in item["cells"])
        }
        row_category_counts[row] = len(categories)
        if row != gap_row and len(categories) < 4:
            errors.append(f"第 {row} 行只有 {len(categories)} 类方块")

    unsupported = []
    for placement in placements:
        cells = [tuple(cell) for cell in placement["cells"]]
        if min(row for _, row in cells) == 1:
            continue
        supported = any(
            (col, row - 1) in owner and owner[(col, row - 1)] != placement["id"]
            for col, row in 方块底部格(cells)
        )
        if not supported:
            unsupported.append(int(placement["id"]))
    if unsupported:
        errors.append(f"存在无支撑方块：{unsupported[:5]}")

    placements_by_id = {int(item["id"]): item for item in placements}
    source_order = payload.get("order")
    source_order_valid = False
    if source_order is None:
        warnings.append("原结果没有 order，将只计算诊断顺序")
    elif not isinstance(source_order, list):
        errors.append("order 不是列表")
    elif len(source_order) != len(ids) or set(source_order) != set(ids):
        errors.append("order 不是 ids 的完整排列")
    else:
        source_order_valid = 顺序可以执行(source_order, placements_by_id)
        if not source_order_valid:
            errors.append("原始 order 不满足支撑优先顺序")

    computed_order = 计算支撑优先顺序(placements)
    if computed_order is None:
        errors.append("无法计算支撑优先摆放顺序")

    return {
        "可绘制": True,
        "通过": not errors,
        "错误": errors,
        "警告": warnings,
        "placements": placements,
        "owner": owner,
        "类别数量": dict(counts),
        "各行占用": row_occupancies,
        "各行类别数": row_category_counts,
        "检测缺口行": detected_gap_rows,
        "原始顺序有效": source_order_valid,
        "诊断顺序": computed_order,
        "记录字段": sorted(payload.keys()),
    }


def 均匀抽样(records: Sequence[dict], count: int) -> List[dict]:
    """从较早、中间和较新位置均匀抽样，异常记录优先。"""
    if len(records) <= count:
        return list(records)

    abnormal = [record for record in records if not record["检查"].get("通过", False)]
    selected = abnormal[:count]
    selected_lines = {record["行号"] for record in selected}
    remaining_count = count - len(selected)
    if remaining_count <= 0:
        return selected

    normal = [record for record in records if record["行号"] not in selected_lines]
    if remaining_count == 1:
        indices = [len(normal) - 1]
    else:
        indices = [
            round(i * (len(normal) - 1) / (remaining_count - 1))
            for i in range(remaining_count)
        ]
    selected.extend(normal[index] for index in dict.fromkeys(indices))
    return selected


def 读取并检查结果(placement_catalog: Dict[int, dict]) -> Tuple[List[dict], dict]:
    """读取当前磁盘快照，验证记录并产生跨分区样本。"""
    partition_dir = 结果目录 / "partitions"
    files = sorted(
        path
        for path in partition_dir.glob("*.jsonl")
        if path.is_file()
    )

    file_summaries = []
    samples = []
    total_valid_json = 0
    total_incomplete_lines = 0
    total_parse_errors = 0
    total_constraint_errors = 0
    total_with_order = 0

    for path in files:
        gap_row, missing_category = 解析分区文件名(path)
        records = []
        incomplete_lines = 0
        parse_errors = 0
        constraint_errors = 0
        checked = 0
        reached_limit = False

        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if checked >= 单文件最多检查记录数:
                    reached_limit = True
                    break
                if not raw_line.endswith("\n"):
                    incomplete_lines += 1
                    continue
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue
                if not isinstance(payload, dict):
                    parse_errors += 1
                    continue

                checked += 1
                check = 检查一条记录(
                    payload,
                    placement_catalog,
                    gap_row,
                    missing_category,
                )
                if not check.get("通过", False):
                    constraint_errors += 1
                if "order" in payload:
                    total_with_order += 1
                records.append(
                    {
                        "文件": path.name,
                        "路径": str(path),
                        "行号": line_number,
                        "缺口行": gap_row,
                        "缺少类别": missing_category,
                        "payload": payload,
                        "检查": check,
                    }
                )

        file_samples = 均匀抽样(records, 每分区抽样数)
        samples.extend(file_samples)
        file_summaries.append(
            {
                "文件": path.name,
                "文件大小字节": path.stat().st_size,
                "已检查完整记录": checked,
                "跳过未完成行": incomplete_lines,
                "JSON解析错误": parse_errors,
                "硬约束异常记录": constraint_errors,
                "达到单文件检查上限": reached_limit,
                "抽样行号": [item["行号"] for item in file_samples],
            }
        )
        total_valid_json += checked
        total_incomplete_lines += incomplete_lines
        total_parse_errors += parse_errors
        total_constraint_errors += constraint_errors

    # 先保证异常样本进入图片，其余样本按分区和行号稳定排列。
    samples.sort(
        key=lambda item: (
            bool(item["检查"].get("通过", False)),
            item["文件"],
            item["行号"],
        )
    )
    samples = samples[:最多绘制盘数]

    summary = {
        "检查时间": datetime.now().isoformat(timespec="seconds"),
        "结果目录": str(结果目录),
        "发现结果文件数": len(files),
        "检查完整JSON记录数": total_valid_json,
        "跳过未完成行数": total_incomplete_lines,
        "JSON解析错误数": total_parse_errors,
        "硬约束异常记录数": total_constraint_errors,
        "包含原始order的记录数": total_with_order,
        "本次绘制盘数": len(samples),
        "说明": "这是读取开始时至读取结束时的只读快照；搜索进程可能仍在继续写入。",
        "文件汇总": file_summaries,
    }
    return samples, summary


def 绘制单盘(ax, record: dict) -> None:
    check = record["检查"]
    placements = check["placements"]
    owner = check["owner"]
    order = check.get("诊断顺序") or []
    order_index = {pid: index + 1 for index, pid in enumerate(order)}

    ax.set_xlim(0.5, 列数 + 0.5)
    ax.set_ylim(0.5, 行数 + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(1, 列数 + 1))
    ax.set_yticks(range(1, 行数 + 1))
    ax.tick_params(labelsize=6, length=0)
    ax.set_axisbelow(True)
    ax.grid(color="#B8B8B8", linewidth=0.45)

    for col in range(1, 列数 + 1):
        for row in range(1, 行数 + 1):
            if (col, row) not in owner:
                ax.add_patch(
                    Rectangle(
                        (col - 0.5, row - 0.5),
                        1,
                        1,
                        facecolor="#F4F4F4",
                        edgecolor="#C62828",
                        linewidth=1.1,
                        hatch="///",
                        zorder=1,
                    )
                )
                ax.plot(
                    [col - 0.32, col + 0.32],
                    [row - 0.32, row + 0.32],
                    color="#C62828",
                    linewidth=1.0,
                    zorder=3,
                )
                ax.plot(
                    [col - 0.32, col + 0.32],
                    [row + 0.32, row - 0.32],
                    color="#C62828",
                    linewidth=1.0,
                    zorder=3,
                )

    for placement in placements:
        pid = int(placement["id"])
        category = placement["category"]
        style = 类别样式[category]
        cells = {tuple(cell) for cell in placement["cells"]}

        for col, row in cells:
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1,
                    1,
                    facecolor=style["颜色"],
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=2,
                )
            )

            # 只在方块外轮廓画粗黑线，内部四格仍能看见淡白分隔线。
            sides = [
                ((col - 0.5, row - 0.5), (col - 0.5, row + 0.5), (-1, 0)),
                ((col + 0.5, row - 0.5), (col + 0.5, row + 0.5), (1, 0)),
                ((col - 0.5, row - 0.5), (col + 0.5, row - 0.5), (0, -1)),
                ((col - 0.5, row + 0.5), (col + 0.5, row + 0.5), (0, 1)),
            ]
            for start, end, (dc, dr) in sides:
                if (col + dc, row + dr) not in cells:
                    ax.plot(
                        [start[0], end[0]],
                        [start[1], end[1]],
                        color="#202020",
                        linewidth=1.15,
                        zorder=4,
                    )

        center_col = sum(col for col, _ in cells) / len(cells)
        center_row = sum(row for _, row in cells) / len(cells)
        if 显示方块编号:
            sequence = order_index.get(pid)
            label = str(sequence) if sequence is not None else style["简称"]
            ax.text(
                center_col,
                center_row,
                label,
                ha="center",
                va="center",
                fontsize=5.2,
                color="white" if category in ("L_blue", "z_blue", "T", "line") else "#202020",
                weight="bold",
                zorder=6,
                bbox={
                    "boxstyle": "circle,pad=0.12",
                    "facecolor": "#111111",
                    "edgecolor": "white",
                    "alpha": 0.68,
                    "linewidth": 0.35,
                },
            )
        if 显示求解器中心点:
            ax.plot(
                float(placement["col"]),
                float(placement["row"]),
                marker="+",
                markersize=4.0,
                markeredgewidth=0.8,
                color="#00FFFF",
                zorder=7,
            )

    passed = bool(check.get("通过", False))
    status = "通过" if passed else "异常"
    status_color = "#147D2A" if passed else "#C62828"
    schema = "+".join(check.get("记录字段", []))
    title = (
        f"{record['文件']}  第{record['行号']}条\n"
        f"缺口行={record['缺口行']}  缺少={record['缺少类别']}  "
        f"格式={schema}  {status}"
    )
    ax.set_title(title, fontsize=7.2, color=status_color, weight="bold", pad=4)

    if passed:
        note = "34块 / 136格 / 行覆盖 / 四色 / 支撑：通过"
        if "order" not in record["payload"]:
            note += "\n原结果无order；图中编号为诊断顺序"
    else:
        note = "；".join(check["错误"][:3])
    ax.text(
        0.5,
        -0.08,
        note,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.8,
        color=status_color,
    )


def 绘制总览(samples: Sequence[dict], output_dir: Path) -> List[str]:
    """将样本分页绘制为 PNG 总览图。"""
    output_files = []
    if not samples:
        return output_files

    columns = 3
    rows = max(1, (每页盘数 + columns - 1) // columns)
    legend_items = [
        Patch(
            facecolor=类别样式[category]["颜色"],
            edgecolor="#202020",
            label=类别样式[category]["名称"],
        )
        for category in 类别顺序
    ]
    legend_items.extend(
        [
            Patch(facecolor="#F4F4F4", edgecolor="#C62828", hatch="///", label="空格"),
            Line2D([0], [0], marker="+", color="none", markeredgecolor="#00A8A8", label="求解器中心点"),
        ]
    )

    for page_start in range(0, len(samples), 每页盘数):
        page_samples = samples[page_start : page_start + 每页盘数]
        page_number = page_start // 每页盘数 + 1
        figure, axes = plt.subplots(rows, columns, figsize=(13.5, 12.8))
        axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

        for ax, record in zip(axes_list, page_samples):
            绘制单盘(ax, record)
        for ax in axes_list[len(page_samples) :]:
            ax.axis("off")

        figure.suptitle(
            f"260分版型只读抽样检查（第 {page_number} 页）",
            fontsize=14,
            weight="bold",
            y=0.995,
        )
        figure.legend(
            handles=legend_items,
            loc="lower center",
            ncol=5,
            fontsize=7.2,
            frameon=True,
            bbox_to_anchor=(0.5, 0.005),
        )
        figure.text(
            0.99,
            0.01,
            "行1在底部；圆圈数字为支撑优先诊断顺序",
            ha="right",
            va="bottom",
            fontsize=7,
            color="#555555",
        )
        figure.tight_layout(rect=(0.015, 0.06, 0.985, 0.975), h_pad=2.1, w_pad=1.0)

        output_path = output_dir / f"版型总览_{page_number:02d}.png"
        figure.savefig(output_path, dpi=图片分辨率, bbox_inches="tight")
        plt.close(figure)
        output_files.append(str(output_path))

    return output_files


def main() -> None:
    配置中文字体()
    catalog_path = 结果目录 / "placement_catalog.json"
    if not catalog_path.is_file():
        raise SystemExit(f"找不到摆放目录：{catalog_path}")

    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog_document = json.load(handle)
    placements = catalog_document.get("placements")
    if not isinstance(placements, list) or not placements:
        raise SystemExit("placement_catalog.json 中没有有效 placements")
    placement_catalog = {int(item["id"]): item for item in placements}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = 可视化输出根目录 / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    samples, summary = 读取并检查结果(placement_catalog)
    image_files = 绘制总览(samples, output_dir)

    summary["图片文件"] = image_files
    summary["抽样记录"] = [
        {
            "文件": record["文件"],
            "行号": record["行号"],
            "缺口行": record["缺口行"],
            "缺少类别": record["缺少类别"],
            "通过": record["检查"].get("通过", False),
            "错误": record["检查"].get("错误", []),
            "警告": record["检查"].get("警告", []),
            "记录字段": record["检查"].get("记录字段", []),
        }
        for record in samples
    ]
    summary_path = output_dir / "检查汇总.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("===== 260分版型只读检查完成 =====")
    print(f"结果目录：{结果目录}")
    print(f"检查完整记录：{summary['检查完整JSON记录数']}")
    print(f"JSON解析错误：{summary['JSON解析错误数']}")
    print(f"硬约束异常记录：{summary['硬约束异常记录数']}")
    print(f"绘制版型：{summary['本次绘制盘数']}")
    print(f"图片页数：{len(image_files)}")
    print(f"输出目录：{output_dir}")
    print(f"检查汇总：{summary_path}")


if __name__ == "__main__":
    main()
