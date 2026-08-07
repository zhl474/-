"""高位 Mask 编辑 Demo 的纯逻辑和合成模板匹配测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


DEMO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = DEMO_DIR.parent.parent
sys.path.insert(0, str(DEMO_DIR))
sys.path.insert(0, str(SRC_DIR / "image_process"))

from mask_editor_core import (  # noqa: E402
    Mask编辑历史,
    保存二值mask,
    在mask上画圆,
    显示坐标转图像坐标,
    查找唯一输入图片,
    计算适配显示尺寸,
)
from image_process_lib.template_match.kernels_create import (  # noqa: E402
    create_rotation_kernels,
)
from image_process_lib.template_match.template_match import get_rect  # noqa: E402
from high_mask_editor_demo import 是总览提交键  # noqa: E402
from high_mask_session_editor import _构建编辑项  # noqa: E402


def test_find_single_input_image_accepts_one_supported_image(tmp_path):
    image_path = tmp_path / "高位照片.PNG"
    image_path.write_bytes(b"placeholder")
    (tmp_path / "说明.txt").write_text("不是图片", encoding="utf-8")

    assert 查找唯一输入图片(tmp_path) == image_path


def test_find_single_input_image_rejects_empty_and_multiple_directories(tmp_path):
    with pytest.raises(ValueError, match="没有图片"):
        查找唯一输入图片(tmp_path)

    (tmp_path / "a.jpg").write_bytes(b"a")
    (tmp_path / "b.png").write_bytes(b"b")
    with pytest.raises(ValueError, match="只能放一张图片"):
        查找唯一输入图片(tmp_path)


def test_draw_and_erase_brush_clip_at_mask_boundary():
    mask = np.zeros((20, 20), dtype=np.uint8)
    在mask上画圆(mask, (0, 0), radius=5, value=255)

    assert mask[0, 0] == 255
    assert mask[-1, -1] == 0
    assert set(np.unique(mask)) <= {0, 255}

    在mask上画圆(mask, (0, 0), radius=2, value=0)
    assert mask[0, 0] == 0
    assert cv2.countNonZero(mask) > 0


def test_mask_history_undo_and_restore_original():
    original = np.zeros((30, 30), dtype=np.uint8)
    original[10:20, 10:20] = 255
    history = Mask编辑历史(original, max_undo_count=3)

    history.开始一笔()
    在mask上画圆(history.current_mask, (2, 2), radius=2, value=255)
    changed = history.current_mask.copy()
    assert not np.array_equal(changed, original)

    assert history.撤销() is True
    assert np.array_equal(history.current_mask, original)
    assert history.撤销() is False

    history.开始一笔()
    在mask上画圆(history.current_mask, (15, 15), radius=3, value=0)
    assert history.恢复原始() is True
    assert np.array_equal(history.current_mask, original)
    assert history.撤销() is True
    assert not np.array_equal(history.current_mask, original)


def test_mask_history_redo_and_new_stroke_clear_redo_stack():
    original = np.zeros((30, 30), dtype=np.uint8)
    history = Mask编辑历史(original, max_undo_count=3)

    history.开始一笔()
    在mask上画圆(history.current_mask, (5, 5), radius=2, value=255)
    first_edit = history.current_mask.copy()
    history.开始一笔()
    在mask上画圆(history.current_mask, (20, 20), radius=2, value=255)
    second_edit = history.current_mask.copy()

    assert history.撤销() is True
    assert np.array_equal(history.current_mask, first_edit)
    assert history.重做() is True
    assert np.array_equal(history.current_mask, second_edit)

    assert history.撤销() is True
    history.开始一笔()
    在mask上画圆(history.current_mask, (15, 15), radius=2, value=255)
    assert history.重做() is False


def test_display_size_and_coordinate_mapping_are_consistent():
    display_w, display_h = 计算适配显示尺寸(
        (200, 400, 3),
        max_width=200,
        max_height=200,
        max_scale=1.0,
    )
    assert (display_w, display_h) == (200, 100)

    assert 显示坐标转图像坐标(
        (100, 50),
        image_shape=(200, 400),
        display_shape=(100, 200),
    ) == (200, 100)
    assert 显示坐标转图像坐标(
        (999, 999),
        image_shape=(200, 400),
        display_shape=(100, 200),
    ) == (399, 199)


def test_save_binary_mask_is_lossless_and_only_contains_zero_or_255(tmp_path):
    source = np.array([[0, 1, 127], [128, 254, 255]], dtype=np.uint8)
    output_path = 保存二值mask(tmp_path / "edited_mask.jpg", source)

    assert output_path.suffix == ".png"
    loaded = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
    assert loaded.ndim == 2
    assert set(np.unique(loaded)) == {0, 255}


def test_overview_enter_and_q_are_both_submit_keys():
    assert 是总览提交键(10) is True
    assert 是总览提交键(13) is True
    assert 是总览提交键(ord("q")) is True
    assert 是总览提交键(ord("Q")) is True
    assert 是总览提交键(27) is False


def test_competition_session_uses_parent_preview_without_initial_full_match():
    preview_result = object()

    class FakeMatcher:
        def 根据父进程结果构建预览(self, preview, category, crop_box):
            assert preview["rect_theta"] == -12.0
            assert category == "T"
            assert crop_box == (10, 20, 40, 50)
            return preview_result

        def 匹配(self, *_args, **_kwargs):
            raise AssertionError("有父进程预览时不应做 CPU 全角度初始匹配")

    manifest = {
        "blocks": [{
            "index": 1,
            "category": "T",
            "detection_score": 0.9,
            "detection_box": [12.0, 22.0, 38.0, 48.0],
            "crop_box": [10, 20, 40, 50],
            "preview_match": {
                "local_center": [15.0, 15.0],
                "rect_size": [30.0, 20.0],
                "rect_theta": -12.0,
                "px": 25.0,
                "py": 35.0,
                "theta": -12.0,
            },
        }],
    }
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:25, 5:25] = 255

    blocks = _构建编辑项(manifest, image, [mask], FakeMatcher())

    assert len(blocks) == 1
    assert blocks[0].original_match is preview_result
    assert blocks[0].current_match is preview_result


def _paste_kernel(mask, kernel, center_xy):
    """把一个合成模板放到指定中心，供干扰区域测试使用。"""
    kernel_mask = (kernel > 0.5).astype(np.uint8) * 255
    kernel_h, kernel_w = kernel_mask.shape
    center_x, center_y = center_xy
    x1 = int(center_x - kernel_w // 2)
    y1 = int(center_y - kernel_h // 2)
    target = mask[y1:y1 + kernel_h, x1:x1 + kernel_w]
    target[:] = np.maximum(target, kernel_mask)


def _match_synthetic_line(mask, prepared_templates):
    debug_output = {}
    rect = get_rect(
        mask,
        block_px=8,
        connector_px=2,
        category="line",
        img_bgr2=None,
        crop_x=0,
        crop_y=0,
        debug_output=debug_output,
        prepared_templates=prepared_templates,
    )
    return rect, debug_output


def test_erasing_synthetic_interference_restores_template_match():
    """验证错误前景能误导角度，擦掉后模板匹配可恢复。"""
    kernels, kernel_size, angles = create_rotation_kernels(
        block_px=8,
        connector_px=2,
        category="line",
        device="cpu",
        angle_values=[90.0, 0.0],
    )
    prepared = {
        "kernels": kernels,
        "kernel_size": kernel_size,
        "angles": angles,
    }
    mask = np.zeros((120, 220), dtype=np.uint8)

    # 正确主体是左侧水平 line，初始匹配应为 0°。
    _paste_kernel(mask, kernels[1, 0].numpy(), (50, 60))
    clean_rect, clean_debug = _match_synthetic_line(mask, prepared)
    assert clean_rect[2] == pytest.approx(0.0)

    # 右侧加入完整竖直干扰后，并列最高分会使结果跳到 90°干扰区域。
    _paste_kernel(mask, kernels[0, 0].numpy(), (160, 60))
    wrong_rect, wrong_debug = _match_synthetic_line(mask, prepared)
    assert wrong_rect[2] == pytest.approx(-90.0)
    assert wrong_rect[0][0] == pytest.approx(160.0)

    # 模拟用户擦掉影响判断的部分，模板重新回到正确主体。
    mask[:, 130:190] = 0
    restored_rect, restored_debug = _match_synthetic_line(mask, prepared)
    assert restored_rect[2] == pytest.approx(0.0)
    assert restored_rect[0][0] == pytest.approx(50.0)
    assert clean_debug["score"] == pytest.approx(restored_debug["score"])
    assert wrong_debug["score"] == pytest.approx(restored_debug["score"])
