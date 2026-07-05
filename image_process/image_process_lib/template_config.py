import os

import yaml


TEMPLATE_CONFIG_PATH = "/home/zhl/SingleArmTetris/SingleArmTetris/src/competition/config/template_config.yaml"


def load_template_config(config_path=TEMPLATE_CONFIG_PATH):
    """读取模板匹配配置。"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"模板配置不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "template_sizes" not in data:
        raise ValueError(f"模板配置格式错误: {config_path}")

    return data


def _read_positive_pixel(item, key, profile_name):
    value = item.get(key)
    if value is None:
        raise ValueError(f"模板几何配置 {profile_name} 缺少 {key}")

    try:
        pixel = int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"模板几何配置 {profile_name} 的 {key} 不是数字: {value}") from exc

    if pixel <= 0:
        raise ValueError(f"模板几何配置 {profile_name} 的 {key} 必须为正数")

    return pixel


def load_template_geometry(profile=None, config_path=TEMPLATE_CONFIG_PATH):
    """读取模板几何参数，返回 {"block_px": 子块像素, "connector_px": 连接处像素}。"""
    data = load_template_config(config_path)
    template_sizes = data["template_sizes"]
    if not isinstance(template_sizes, dict):
        raise ValueError("template_sizes 必须是字典")

    profiles = template_sizes.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("template_sizes 缺少 profiles 配置")

    profile_name = profile or template_sizes.get("active_profile")
    if not profile_name:
        raise ValueError("template_sizes 缺少 active_profile")
    if profile_name not in profiles:
        raise KeyError(f"模板几何配置中没有 profile: {profile_name}")

    item = profiles[profile_name]
    if not isinstance(item, dict):
        raise ValueError(f"模板几何配置 {profile_name} 必须是字典")

    return {
        "block_px": _read_positive_pixel(item, "block_px", profile_name),
        "connector_px": _read_positive_pixel(item, "connector_px", profile_name),
    }
