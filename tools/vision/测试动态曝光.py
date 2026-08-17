#!/home/zhl/fr3env/fr3env/bin/python
"""动态调曝光可行性验证脚本（不动 ROS，直接 python 运行）。

验证目标：
  1. AkaiGemini335 实例暴露的 device / pipeline 属性能否拿到 ob.Device；
  2. 6 个曝光/增益属性（彩色 2000/2001/2002，深度 2016/2017/2018）是否支持、范围多少；
  3. 流开启状态下写曝光，画面亮度是否实时变化（流不重启、不断连）。

！！！运行前必读！！！
  相机是 USB 独占设备，运行本脚本前必须：
    1. 停掉 hardware.launch（或关掉面板里启动的硬件进程）；
    2. 关闭 OrbbecViewer。
  否则脚本会在打开设备时失败。

用法：
    /home/zhl/fr3env/fr3env/bin/python tools/vision/测试动态曝光.py
"""

import time

import cv2
import numpy as np
from akai_gemini335 import AkaiGemini335
from pyorbbecsdk import Config, OBPermissionType, OBPropertyID

# ===================== 传参区 =====================
CAMERA_CONFIG_YAML = "/home/zhl/SingleArmTetris/SingleArmTetris/src/camera/config/新相机参数.yaml"
# 取 device 途径: "device" -> cap.device；"pipeline" -> cap.pipeline.get_device()；"auto" 依次尝试
DEVICE_SOURCE = "auto"
# 曝光扫描：从 min 到 max 之间均匀取 N 档（彩色）
SWEEP_SAMPLES = 8
# 每档曝光写完后的等待秒数（等 AE 收敛画面）
SETTLE_SEC = 1.0
# 扫描结束后是否恢复彩色自动曝光
RESTORE_COLOR_AE = True
# ===================================================

COLOR_PROPERTIES = {
    "auto_exposure": OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
    "exposure": OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT,
    "gain": OBPropertyID.OB_PROP_COLOR_GAIN_INT,
}
DEPTH_PROPERTIES = {
    "auto_exposure": OBPropertyID.OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL,
    "exposure": OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT,
    "gain": OBPropertyID.OB_PROP_DEPTH_GAIN_INT,
}


def get_device(cap):
    sources = [DEVICE_SOURCE] if DEVICE_SOURCE != "auto" else ["device", "pipeline"]
    for source in sources:
        try:
            if source == "device":
                device = cap.device
            else:
                device = cap.pipeline.get_device()
            # 注意：pyorbbecsdk 2.0.13 没有 get_device_type()，探活用真实属性接口
            if device is None:
                print(f"[--] 途径 cap.{source} 返回空")
                continue
            device.get_int_property_range(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT)
            print(f"[OK] 通过 cap.{source} 拿到设备: {device}")
            return device
        except Exception as exc:
            print(f"[--] 途径 cap.{source} 失败: {exc}")
    return None


def report_property(device, name, prop_id):
    try:
        supported = device.is_property_supported(prop_id, OBPermissionType.READ_WRITE)
    except Exception as exc:
        print(f"  {name}: is_property_supported 异常: {exc}")
        return None
    if not supported:
        print(f"  {name}: 不支持或权限不足")
        return None
    try:
        rng = device.get_int_property_range(prop_id)
    except Exception as exc:
        print(f"  {name}: 范围查询失败: {exc}")
        return None
    current = None
    try:
        current = device.get_int_property(prop_id)
    except Exception as exc:
        print(f"  {name}: 当前值读取失败: {exc}")
    print(f"  {name}: 范围 [{rng.min}, {rng.max}] step={rng.step} default={rng.default_value} 当前={current}")
    return rng


def read_brightness(cap):
    color_image, _ = cap.read()
    if color_image is None:
        return None
    return float(np.mean(cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)))


def main():
    print("=== 1. 打开相机（若失败：检查 hardware.launch / OrbbecViewer 是否已关闭）===")
    cap = AkaiGemini335(yaml_path=CAMERA_CONFIG_YAML)
    cap.config = Config()  # 修复库 bug：release() 需要该属性
    try:
        print("=== 2. 确认取流正常 ===")
        time.sleep(1.0)
        base_brightness = read_brightness(cap)
        print(f"首帧亮度均值: {base_brightness}")
        if base_brightness is None:
            print("[FAIL] 读不到彩色帧，无法继续")
            return

        print("=== 3. 获取 ob.Device ===")
        device = get_device(cap)
        if device is None:
            print("[FAIL] 两种途径都拿不到 device，需要改方案")
            return

        print("=== 4. 属性支持与范围 ===")
        print("彩色:")
        color_ranges = {}
        for name, prop_id in COLOR_PROPERTIES.items():
            if name != "auto_exposure":
                rng = report_property(device, name, prop_id)
                if rng is not None:
                    color_ranges[name] = rng
        ae_state = device.get_bool_property(COLOR_PROPERTIES["auto_exposure"])
        print(f"  auto_exposure: 当前={ae_state}")
        print("深度:")
        for name, prop_id in DEPTH_PROPERTIES.items():
            if name == "auto_exposure":
                print(f"  auto_exposure: 当前={device.get_bool_property(prop_id)}")
            else:
                report_property(device, name, prop_id)

        if "exposure" not in color_ranges:
            print("[FAIL] 彩色曝光属性不可用")
            return
        rng = color_ranges["exposure"]

        print("=== 5. 关闭彩色自动曝光 ===")
        device.set_bool_property(COLOR_PROPERTIES["auto_exposure"], False)
        print(f"auto_exposure 当前={device.get_bool_property(COLOR_PROPERTIES['auto_exposure'])}")

        print(f"=== 6. 曝光扫描 {SWEEP_SAMPLES} 档（流保持运行）===")
        lo, hi = rng.min, rng.max
        for i in range(SWEEP_SAMPLES):
            exposure = int(lo + (hi - lo) * i / max(SWEEP_SAMPLES - 1, 1))
            device.set_int_property(COLOR_PROPERTIES["exposure"], exposure)
            time.sleep(SETTLE_SEC)
            brightness = read_brightness(cap)
            readback = device.get_int_property(COLOR_PROPERTIES["exposure"])
            print(f"  exposure={exposure:5d} 回读={readback:5d} 亮度={brightness:.1f}")

        if RESTORE_COLOR_AE:
            print("=== 7. 恢复彩色自动曝光 ===")
            device.set_bool_property(COLOR_PROPERTIES["auto_exposure"], True)
            print(f"auto_exposure 当前={device.get_bool_property(COLOR_PROPERTIES['auto_exposure'])}")

        print("=== 结论：流开启状态下动态写曝光可行 ===")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
