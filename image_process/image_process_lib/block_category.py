"""方块类别名称的统一定义。"""


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


# 顺序与进阶任务 IDBS 动态库的参数顺序保持一致。
BLOCK_CATEGORY_NAMES = (
    "L_blue",
    "L_yellow",
    "z_blue",
    "z_green",
    "square",
    "T",
    "line",
)

def normalize_category_name(category):
    """统一方块类别名，避免旧模型类别和新模型类别混用。"""
    return CATEGORY_NAME_MAP.get(category, category)
