"""方块类别名和数值编码的统一定义。"""


# 进阶任务动态库和旧代码里可能出现旧类别名，这里统一转成当前模型类别名。
CATEGORY_NAME_MAP = {
    "LR": "L_blue",
    "LL": "L_yellow",
    "ZL": "z_blue",
    "ZR": "z_green",
    "O": "square",
    "suqare": "square",
    "Line": "line",
}


# 类别码用于 GetTargetPos.float32[] 的兼容扩展，顺序必须保持稳定。
BLOCK_CATEGORY_NAMES = (
    "L_blue",
    "L_yellow",
    "z_blue",
    "z_green",
    "square",
    "T",
    "line",
)

CATEGORY_CODE_BY_NAME = {
    category: index
    for index, category in enumerate(BLOCK_CATEGORY_NAMES)
}


def normalize_category_name(category):
    """统一方块类别名，避免旧模型类别和新模型类别混用。"""
    return CATEGORY_NAME_MAP.get(category, category)


def category_to_code(category):
    """把方块类别名编码成 float 数组里可传递的整数码。"""
    normalized_category = normalize_category_name(str(category or "").strip())
    return CATEGORY_CODE_BY_NAME.get(normalized_category, -1)


def category_from_code(category_code):
    """把 GetTargetPos 返回的类别码解码成方块类别名。"""
    try:
        index = int(round(float(category_code)))
    except (TypeError, ValueError):
        return ""
    if 0 <= index < len(BLOCK_CATEGORY_NAMES):
        return BLOCK_CATEGORY_NAMES[index]
    return ""
