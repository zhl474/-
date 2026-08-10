#!/home/zhl/fr3env/fr3env/bin/python
"""Gemini 335 实际流配置、厂商内参和显式去畸变探针。

运行前必须停止 camera_node，避免两个进程同时占用相机。本脚本不会移动机械臂。
"""

import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diagnostic_core import save_image, to_builtin, validate_image_size  # noqa: E402


# ==================== 运行参数（直接修改本文件）====================
CAMERA_CONFIG_PATH = SRC_ROOT / "camera" / "config" / "新相机参数.yaml"
OUTPUT_ROOT = Path.home() / "桌面" / "相机几何诊断" / "相机配置探针"
EXPECTED_IMAGE_SIZE = (1280, 720)  # (宽, 高)，不匹配时立即停止
WARMUP_FRAMES = 15
SAVE_FRAME_COUNT = 3


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "未安装或无包元数据"


def _describe_sdk_value(value):
    """尽量提取 SDK 对象中的有限标量和矩阵，同时保留类型与文本。"""
    description = {
        "python_type": f"{type(value).__module__}.{type(value).__name__}",
        "text": str(value),
    }
    try:
        array = np.asarray(value)
        if array.dtype != object and array.size and np.all(np.isfinite(array.astype(float))):
            description["array"] = array.astype(float).tolist()
    except (TypeError, ValueError):
        pass
    attributes = {}
    for name in (
        "width", "height", "fx", "fy", "cx", "cy",
        "k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2",
        "model", "intrinsic_matrix",
    ):
        if not hasattr(value, name):
            continue
        try:
            item = getattr(value, name)
            if callable(item):
                continue
            attributes[name] = to_builtin(np.asarray(item) if name == "intrinsic_matrix" else item)
        except Exception as exc:  # noqa: BLE001
            attributes[name] = f"读取失败: {exc}"
    if attributes:
        description["attributes"] = attributes
    return description


def _public_camera_attributes(camera):
    result = {}
    for name in (
        "img_width", "img_height", "fps", "depth_img_width", "depth_img_height",
        "depth_fps", "align_mode", "rgb_format", "depth_format", "min_distance",
        "max_distance", "distance_filter_enable", "ldp_enable", "buffer_size",
    ):
        if not hasattr(camera, name):
            continue
        try:
            result[name] = to_builtin(getattr(camera, name))
        except Exception as exc:  # noqa: BLE001
            result[name] = f"读取失败: {exc}"
    return result


def _difference_statistics(first, second):
    if first is None or second is None or first.shape != second.shape:
        return {"可比较": False}
    difference = np.abs(first.astype(np.float32) - second.astype(np.float32))
    changed = np.any(difference > 0.0, axis=2) if difference.ndim == 3 else difference > 0.0
    return {
        "可比较": True,
        "平均绝对差": float(np.mean(difference)),
        "P95绝对差": float(np.percentile(difference, 95)),
        "最大绝对差": float(np.max(difference)),
        "发生变化像素比例": float(np.mean(changed)),
        "说明": "存在差异只证明显式映射改变了图像，不能单独证明输入图是否已经去畸变",
    }


def main():
    """连接相机并输出一份不依赖猜测的实际配置快照。"""
    from akai_gemini335 import AkaiGemini335
    from pyorbbecsdk import Config

    run_time = datetime.now(timezone.utc).astimezone()
    output_dir = OUTPUT_ROOT / run_time.strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    config_data = yaml.safe_load(CAMERA_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    report = {
        "schema_version": 1,
        "探针时间": run_time.isoformat(timespec="seconds"),
        "相机型号": "Orbbec Gemini 335（由当前工程封装类确认）",
        "相机配置路径": str(CAMERA_CONFIG_PATH),
        "相机配置": config_data,
        "期望图像尺寸": list(EXPECTED_IMAGE_SIZE),
        "软件版本": {
            "python": sys.version,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "akai_gemini335": _package_version("akai_gemini335"),
            "pyorbbecsdk": _package_version("pyorbbecsdk"),
        },
        "软件对齐依据": (
            "AkaiGemini335.get_sw_align_config() 的公开说明明确为软件对齐；"
            "最终状态仍以实例 align_mode 和实际彩色/深度尺寸为准"
        ),
        "帧": [],
        "警告": [],
    }
    camera = None
    try:
        print("正在独占连接 Gemini335；若失败请先停止 camera_node。")
        camera = AkaiGemini335(yaml_path=str(CAMERA_CONFIG_PATH))
        camera.config = Config()  # 修复库 bug：release() 需要该属性
        report["封装实例公开属性"] = _public_camera_attributes(camera)
        try:
            report["厂商RGB内参"] = _describe_sdk_value(camera.get_intrinsic())
        except Exception as exc:  # noqa: BLE001
            report["厂商RGB内参"] = {"读取失败": str(exc)}
            report["警告"].append(f"get_intrinsic() 失败: {exc}")
        try:
            report["厂商RGB畸变"] = _describe_sdk_value(camera.get_distortion())
        except Exception as exc:  # noqa: BLE001
            report["厂商RGB畸变"] = {"读取失败": str(exc)}
            report["警告"].append(f"get_distortion() 失败: {exc}")

        for index in range(max(0, int(WARMUP_FRAMES))):
            color, depth = camera.read()
            if color is None or depth is None:
                print(f"预热帧 {index + 1}/{WARMUP_FRAMES} 不完整，继续等待。")

        for index in range(int(SAVE_FRAME_COUNT)):
            color, depth = camera.read()
            if color is None or depth is None:
                raise RuntimeError(f"第 {index + 1} 个保存帧不完整")
            validate_image_size(color, EXPECTED_IMAGE_SIZE)
            validate_image_size(depth, EXPECTED_IMAGE_SIZE)
            if color.ndim != 3 or color.shape[2] != 3:
                raise RuntimeError(f"彩色图格式异常: {color.shape}")
            if depth.ndim != 2:
                raise RuntimeError(f"深度图格式异常: {depth.shape}")

            frame_name = f"帧{index + 1:03d}"
            color_path = output_dir / f"{frame_name}_工程彩色图.png"
            depth_path = output_dir / f"{frame_name}_对齐深度.npy"
            save_image(color_path, color)
            np.save(depth_path, depth)
            corrected = None
            correction_error = ""
            try:
                corrected = camera.remove_distortion(color.copy())
                if corrected is None:
                    raise RuntimeError("remove_distortion() 返回 None")
                validate_image_size(corrected, EXPECTED_IMAGE_SIZE)
                save_image(output_dir / f"{frame_name}_显式去畸变.png", corrected)
            except Exception as exc:  # noqa: BLE001
                correction_error = str(exc)
                report["警告"].append(f"{frame_name} 显式去畸变失败: {exc}")
            report["帧"].append(
                {
                    "名称": frame_name,
                    "彩色图": str(color_path),
                    "深度图": str(depth_path),
                    "彩色shape": list(color.shape),
                    "彩色dtype": str(color.dtype),
                    "深度shape": list(depth.shape),
                    "深度dtype": str(depth.dtype),
                    "有效深度比例": float(np.mean(np.isfinite(depth) & (depth > 0))),
                    "显式去畸变错误": correction_error,
                    "显式去畸变差异": _difference_statistics(color, corrected),
                }
            )
            print(f"已保存 {frame_name}。")
        report["尺寸检查通过"] = True
    except Exception as exc:
        report["尺寸检查通过"] = False
        report["致命错误"] = str(exc)
        raise
    finally:
        if camera is not None:
            camera.release()
        report_path = output_dir / "相机配置探针.json"
        report_path.write_text(
            json.dumps(to_builtin(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"探针报告：{report_path}")


if __name__ == "__main__":
    main()

