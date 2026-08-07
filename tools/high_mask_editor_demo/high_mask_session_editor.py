#!/usr/bin/env python3
"""比赛流程使用的高位 Mask 编辑子进程入口，不加载 YOLO 或 ROS。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parent
SRC_DIR = DEMO_DIR.parent.parent
IMAGE_PROCESS_DIR = SRC_DIR / "image_process"
for path in (DEMO_DIR, IMAGE_PROCESS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import high_mask_editor_demo as editor  # noqa: E402
from image_process_lib.high_mask_edit_session import (  # noqa: E402
    会话环境变量,
    提交高位Mask编辑结果,
    预览设备环境变量,
    编辑取消退出码,
    读取高位Mask编辑会话,
)


def _构建编辑项(manifest, image_bgr, masks, matcher):
    blocks = []
    total = len(masks)
    for offset, (record, mask) in enumerate(zip(manifest["blocks"], masks), start=1):
        crop_box = tuple(int(value) for value in record["crop_box"])
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        roi_bgr = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        category = str(record["category"])
        preview_match = record.get("preview_match")
        if preview_match is None:
            # 兼容没有预览数据的会话；新版比赛会话不会走这个慢路径。
            match = matcher.匹配(mask, category, crop_box)
        else:
            match = matcher.根据父进程结果构建预览(
                preview_match,
                category,
                crop_box,
            )
        blocks.append(editor.方块编辑项(
            index=int(record["index"]),
            category=category,
            detection_score=float(record.get("detection_score", 0.0)),
            detection_box=tuple(float(value) for value in record["detection_box"]),
            crop_box=crop_box,
            roi_bgr=roi_bgr,
            original_mask=mask.copy(),
            edited_mask=mask.copy(),
            original_match=match,
            current_match=match,
        ))
        if offset == total or offset % 10 == 0:
            print(f"已准备总览预览：{offset}/{total}", flush=True)
    return blocks


def main() -> int:
    manifest_value = os.environ.get(会话环境变量, "").strip()
    if not manifest_value:
        raise RuntimeError(f"缺少比赛编辑会话环境变量：{会话环境变量}")
    manifest_path = Path(manifest_value).resolve()
    print("高位 Mask 编辑子进程已启动，正在读取会话……", flush=True)
    manifest, image_bgr, masks = 读取高位Mask编辑会话(manifest_path)

    preview_device = os.environ.get(预览设备环境变量, "cuda").strip().lower()
    if preview_device not in ("cpu", "cuda"):
        raise RuntimeError(f"不支持的预览设备：{preview_device}")
    print(f"局部模板匹配设备：{preview_device}", flush=True)
    matcher = editor.高位模板匹配器(device=preview_device)
    text_drawer = editor.界面绘字器()
    print(f"正在准备 {len(masks)} 个方块的总览预览……", flush=True)
    blocks = _构建编辑项(manifest, image_bgr, masks, matcher)
    output_dir = manifest_path.parent / "preview"
    image_path = manifest_path.parent / str(manifest["image_path"])

    print("总览准备完成，正在打开窗口……", flush=True)
    committed = editor.运行交互总览(
        image_path,
        image_bgr,
        blocks,
        matcher,
        output_dir,
        text_drawer,
    )
    if not committed:
        print("已取消本轮高位 Mask 编辑。")
        return 编辑取消退出码

    提交高位Mask编辑结果(
        manifest_path,
        [block.edited_mask for block in blocks],
    )
    print("高位 Mask 编辑结果已提交给 ROS 父进程。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"\033[91m高位 Mask 编辑子进程失败：{exc}\033[0m")
        raise SystemExit(1)
