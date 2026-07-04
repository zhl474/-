import os

import yaml


TEMPLATE_CONFIG_PATH = "/home/zhl/SingleArmTetris/SingleArmTetris/src/competition/config/template_config.yaml"


def load_template_config(config_path=TEMPLATE_CONFIG_PATH):
    """读取模板匹配尺寸配置。"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"模板尺寸配置不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "template_sizes" not in data:
        raise ValueError(f"模板尺寸配置格式错误: {config_path}")

    return data


def load_template_sizes(config_path=TEMPLATE_CONFIG_PATH):
    """读取所有类别的模板尺寸，返回 {类别: (长边, 短边)}。"""
    data = load_template_config(config_path)
    template_sizes = {}
    for category, item in data["template_sizes"].items():
        if not isinstance(item, dict):
            raise ValueError(f"类别 {category} 的模板尺寸配置格式错误")

        long_side = item.get("long_side")
        short_side = item.get("short_side")
        if long_side is None or short_side is None:
            raise ValueError(f"类别 {category} 缺少 long_side 或 short_side")

        long_side = int(round(float(long_side)))
        short_side = int(round(float(short_side)))
        if long_side <= 0 or short_side <= 0:
            raise ValueError(f"类别 {category} 的模板尺寸必须为正数")

        template_sizes[category] = (long_side, short_side)

    return template_sizes


def get_template_size(category, template_sizes=None, config_path=TEMPLATE_CONFIG_PATH):
    """按类别获取模板尺寸，返回 get_rect 需要的 (w, h)。"""
    if template_sizes is None:
        template_sizes = load_template_sizes(config_path)

    if category not in template_sizes:
        raise KeyError(f"模板尺寸配置中没有类别: {category}")

    return template_sizes[category]
