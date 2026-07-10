import os

import yaml


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PERCEPTION_CONFIG_PATH = os.path.join(
    SRC_DIR, "image_process", "config", "perception.yaml"
)


def test_perception_models_are_configured_and_exist():
    """正式使用的三个视觉模型必须由配置给出，且路径真实存在。"""
    with open(PERCEPTION_CONFIG_PATH, "r", encoding="utf-8") as config_file:
        model_config = (yaml.safe_load(config_file) or {})["models"]

    for model_key in ("detection", "board", "segmentation"):
        model_path = os.path.join(SRC_DIR, model_config[model_key])
        assert os.path.isfile(model_path), f"模型不存在: {model_path}"


def test_default_segmentation_model_uses_tensorrt_engine():
    """正式分割路径必须使用 TensorRT engine，避免意外退回较慢的 pt 推理。"""
    with open(PERCEPTION_CONFIG_PATH, "r", encoding="utf-8") as config_file:
        model_config = (yaml.safe_load(config_file) or {})["models"]

    assert os.path.splitext(model_config["segmentation"])[1].lower() == ".engine"
