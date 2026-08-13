"""单臂俄罗斯方块本地控制台后端。"""


def run_panel():
    """延迟导入 ROS 运行时，便于在无 ROS Master 时测试配置模块。"""
    from .runtime import run_panel as _run_panel

    return _run_panel()


__all__ = ["run_panel"]
