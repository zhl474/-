#!/usr/bin/env bash
# 桌面快捷方式调用本脚本，统一加载 ROS、工作空间和项目指定虚拟环境。
# catkin 的环境脚本会读取未定义变量，也可能执行用于探测环境的非零命令。
# 严格退出检查要等两份环境加载完成后再启用。
set -o pipefail

# 桌面启动时没有终端可见，因此从第一步开始记录日志，便于直接定位环境问题。
export XDG_STATE_HOME="${XDG_STATE_HOME:-/home/zhl/.local/state}"
PANEL_STATE_ROOT="${XDG_STATE_HOME}/single-arm-tetris"
mkdir -p "$PANEL_STATE_ROOT"
exec >> "$PANEL_STATE_ROOT/控制台启动.log" 2>&1

printf '\n[%s] 收到桌面启动请求。\n' "$(date '+%F %T')"
trap 'status=$?; printf "[%s] 启动脚本在第 %s 行退出，状态码：%s。\n" "$(date "+%F %T")" "$LINENO" "$status"' ERR

if ! source /opt/ros/noetic/setup.bash; then
  printf '[%s] ROS Noetic 环境加载失败。\n' "$(date '+%F %T')"
  exit 1
fi
if ! source /home/zhl/SingleArmTetris/SingleArmTetris/devel/setup.bash; then
  printf '[%s] 工作空间环境加载失败，请先执行 catkin_make。\n' "$(date '+%F %T')"
  exit 1
fi
set -e

printf '[%s] ROS 与工作空间环境加载完成，正在启动网页后端。\n' "$(date '+%F %T')"

# 即使工作空间尚未重新构建，也能从源码目录导入控制台包。
export PYTHONPATH="/home/zhl/SingleArmTetris/SingleArmTetris/src/operator_panel${PYTHONPATH:+:${PYTHONPATH}}"

exec /home/zhl/fr3env/fr3env/bin/python \
  /home/zhl/SingleArmTetris/SingleArmTetris/src/operator_panel/scripts/operator_panel.py
