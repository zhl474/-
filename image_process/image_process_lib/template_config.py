import os

import yaml

from image_process_lib.template_match.kernels_create import (
    TETRIS_BLOCKS,
    category_grid_counts,
    expand_ideal_runs,
    validate_runs,
)


PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, ".."))
TEMPLATE_CONFIG_PATH = os.path.join(SRC_DIR, "competition", "config", "template_config.yaml")


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


def _read_required_positive_int(item, key, context):
    if key not in item:
        raise ValueError(f"{context} 缺少 {key}")

    value = item[key]
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} 的 {key} 不是数字: {value}") from exc

    if number <= 0:
        raise ValueError(f"{context} 的 {key} 必须为正数")

    return int(round(number))


def _read_required_positive_number(item, key, context):
    if key not in item:
        raise ValueError(f"{context} 缺少 {key}")

    value = item[key]
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} 的 {key} 不是数字: {value}") from exc

    if number <= 0:
        raise ValueError(f"{context} 的 {key} 必须为正数")

    return number


def _load_profile_runs(profile_name, item, block_px, connector_px):
    """解析 profile 的按类别 overrides，与理想展开合并成全类别 runs 映射。

    overrides 里没写的类别或方向回落 block_px/connector_px 理想值；
    类别名、字段名、段数、正数性在这里硬校验，拼错直接报错。
    """
    overrides = item.get("overrides")
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError(f"模板几何配置 {profile_name} 的 overrides 必须是字典")
    unknown = sorted(str(name) for name in overrides if name not in TETRIS_BLOCKS)
    if unknown:
        raise ValueError(
            f"模板几何配置 {profile_name} 的 overrides 含未知类别: {unknown}，"
            f"合法类别: {sorted(TETRIS_BLOCKS)}"
        )
    runs = {}
    for category in TETRIS_BLOCKS:
        grid_w, grid_h = category_grid_counts(category)
        ideal_x = tuple(expand_ideal_runs(grid_w, block_px, connector_px))
        ideal_y = tuple(expand_ideal_runs(grid_h, block_px, connector_px))
        override = overrides.get(category)
        if override is None:
            runs[category] = {"x_runs": ideal_x, "y_runs": ideal_y}
            continue
        if not isinstance(override, dict):
            raise ValueError(
                f"模板几何配置 {profile_name} 的 overrides.{category} 必须是字典"
            )
        unexpected = sorted(str(key) for key in override if key not in ("x_runs", "y_runs"))
        if unexpected:
            raise ValueError(
                f"模板几何配置 {profile_name} 的 overrides.{category} 含未知字段: {unexpected}"
            )
        x_runs = override.get("x_runs")
        y_runs = override.get("y_runs")
        runs[category] = {
            "x_runs": validate_runs(category, "x", ideal_x if x_runs is None else x_runs),
            "y_runs": validate_runs(category, "y", ideal_y if y_runs is None else y_runs),
        }
    return runs


def load_template_geometry(profile=None, config_path=TEMPLATE_CONFIG_PATH):
    """读取模板几何参数。

    返回 {"block_px": 子块像素, "connector_px": 连接处像素,
    "runs": {类别: {"x_runs": (...), "y_runs": (...)}}}，runs 已把按类别
    overrides 与理想展开合并成全部 7 类；消费方可用
    kernels_create.resolve_category_runs 按类别取线段。
    """
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

    block_px = _read_positive_pixel(item, "block_px", profile_name)
    connector_px = _read_positive_pixel(item, "connector_px", profile_name)
    return {
        "block_px": block_px,
        "connector_px": connector_px,
        "runs": _load_profile_runs(profile_name, item, block_px, connector_px),
    }


def load_color_segmentation_config(category, config_path=TEMPLATE_CONFIG_PATH):
    """读取低位 RGB 局部颜色分割配置。"""
    data = load_template_config(config_path)
    if "color_segmentation" not in data:
        raise ValueError("模板配置缺少 color_segmentation")

    color_segmentation = data["color_segmentation"]
    if not isinstance(color_segmentation, dict):
        raise ValueError("color_segmentation 必须是字典")

    context = "color_segmentation"
    seed_search_half_size = _read_required_positive_int(
        color_segmentation,
        "seed_search_half_size",
        context,
    )
    seed_patch_size = _read_required_positive_int(
        color_segmentation,
        "seed_patch_size",
        context,
    )
    seed_stride = _read_required_positive_int(
        color_segmentation,
        "seed_stride",
        context,
    )
    local_dist_thresh = _read_required_positive_number(
        color_segmentation,
        "local_dist_thresh",
        context,
    )

    categories = color_segmentation.get("categories")
    if not isinstance(categories, dict):
        raise ValueError("color_segmentation 缺少 categories 配置")

    category_name = str(category or "").strip()
    if not category_name:
        raise ValueError("颜色分割配置缺少当前类别名")
    if category_name not in categories:
        raise KeyError(f"颜色分割配置中没有类别: {category_name}")

    category_config = categories[category_name]
    if not isinstance(category_config, dict):
        raise ValueError(f"颜色分割类别 {category_name} 必须是字典")
    if "rgb" not in category_config:
        raise ValueError(f"颜色分割类别 {category_name} 缺少 rgb")

    rgb = category_config["rgb"]
    if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
        raise ValueError(f"颜色分割类别 {category_name} 的 rgb 必须是三个数字")

    try:
        r_prior = float(rgb[0])
        g_prior = float(rgb[1])
        b_prior = float(rgb[2])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"颜色分割类别 {category_name} 的 rgb 不是数字: {rgb}") from exc

    return {
        "seed_search_half_size": seed_search_half_size,
        "seed_patch_size": seed_patch_size,
        "seed_stride": seed_stride,
        "local_dist_thresh": local_dist_thresh,
        "rgb": (r_prior, g_prior, b_prior),
    }
