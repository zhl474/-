# 单臂俄罗斯方块本地控制台

这是项目的本机中文“驾驶舱”。它只监听 `127.0.0.1:8765`，启动页面时不会自动启动相机、机械臂或感知节点。

## 日常使用

构建和安装完成后，双击桌面的“单臂俄罗斯方块控制台”。再次双击不会创建第二个后台实例，只会重新打开已有页面。

页面中的标准操作顺序为：

1. 启动硬件，等待控制服务和新相机帧都变为绿色。
2. 选择“正式”或“标定”感知模式，等待感知服务就绪。
3. 选择基础或进阶任务；进阶任务可拖动七类方块调整顺序。
4. 点击“开始识别”，查看相机与三类调试图。
5. 识别结束后在弹窗中确认、重新识别或结束本轮，再单独点击开始执行。
6. 用页面的进度、持块状态和 ROS 日志观察过程。

顶栏与相机卡片会明确显示 ROS Master、节点、图像话题和消息类型。“ROS 系统”页从当前 ROS Master 实时读取节点、话题、服务和相机话题频率，便于现场查看实际通信结构。

关闭浏览器不会停止后台或硬件。需要结束时使用左下角“显式退出控制台”；控制台只结束自己启动的进程，不会结束外部 ROS 节点。
若控制台自己启动了硬件，退出和“停止硬件”都会先确认机械臂已经停稳；有操作正在执行或运动状态未知时会拒绝退出。

## 人工选择

网页启动的感知节点不读取终端字符。V5 动态盘面失败时，页面会显示固定选项弹窗；可以停止本轮、回退固定盘面，速度模型不一致时还可以二次确认后继续动态盘面。弹窗 60 秒无人处理会自动停止本轮，关闭或断开浏览器也不会绕过超时保护。

识别成功后弹窗提供“确认结果、重新识别、结束本轮”，识别失败时提供“调整后重新识别、结束本轮”。结束本轮只清除识别数据，不移动机械臂、不改变吸盘，也不停止已经就绪的硬件和感知节点。

## 停止运动

红色“停止运动”按钮使用独立高优先级通道：

- 控制节点先锁存停止状态，拒绝后续机械臂和舵机命令；
- 通过独立 XML-RPC 连接调用厂商 `StopMotion()`；
- 中止当前任务并停止感知 launch，正在等待的 Mask 编辑器会随节点关闭；
- 保持吸盘当前状态，不自动喷气、关闭或复位；
- 只有人工检查现场并点击“解除停止锁”后才能重新开始。

如果停止锁生效时控制节点已经离线，页面仍允许单独点击“启动硬件”来恢复控制服务；恢复后会立即再次调用 `StopMotion()` 同步远端停止锁，感知、识别和手动动作仍保持禁用，直到人工解除锁。

如果无法确认停止成功，页面会持续显示全屏红色告警。网页软件停止不替代安全等级物理急停；运动没有立即停止时必须使用物理急停。

## 参数中心

参数中心只允许六个固定配置 ID，不接受任意文件路径：

- `competition/config/execution.yaml`
- `competition/config/visual_servo.yaml`
- `image_process/config/perception.yaml`
- `control/config/controller.yaml`
- `camera/config/新相机参数.yaml`
- `competition/config/template_config.yaml`

保存时使用 `ruamel.yaml` 保留中文注释和顺序，并执行以下保护：

- 不允许增加、删除 YAML 键或改变数组长度；
- 校验类型、有限值、已有物理范围和跨字段关系；
- 位姿、高度、安全范围、映射矩阵、模型路径等危险字段二次确认；
- 携带 revision，防止覆盖 VS Code 或其它程序刚做的修改；
- 同目录临时文件校验完成后原子替换；
- 修改前后全文、差异、Git 版本和时间写入 SQLite；
- 支持历史恢复和六份文件的命名预设事务恢复。

状态数据库默认位于：

```text
${XDG_STATE_HOME:-~/.local/state}/single-arm-tetris/panel.sqlite3
```

“仅保存”只落盘；若对应节点此刻正在运行，才把它记入待重启名单并在卡片上显示“待重启”。“保存并重启”在写盘后自动重启控制台拥有的对应节点；仅当对应节点正在运行且由控制台启动时才可点击，否则按钮置灰。节点停止运行即自动移出待重启名单，外部启动的节点显示“待重启·需手动重启”且控制台永远不会结束或重启它。

## 保留的兼容入口

原终端方式仍然可用：

```bash
roslaunch competition hardware.launch
roslaunch competition competition.launch
roslaunch competition calibration.launch
```

`competition.launch` 和 `calibration.launch` 都复用新的 `perception.launch`，默认使用 `interaction_mode:=terminal` 并继续接受原来的字符输入。网页控制台固定使用 `interaction_mode:=web`。独立 Mask 编辑窗口也保持不变。

## 开发和验证

依赖只能安装到项目指定虚拟环境：

```bash
/home/zhl/fr3env/fr3env/bin/python -m pip install -r operator_panel/requirements.txt
```

构建 ROS 服务和包：

```bash
cd /home/zhl/SingleArmTetris/SingleArmTetris
source /opt/ros/noetic/setup.bash
catkin_make
```

运行控制台测试：

```bash
cd /home/zhl/SingleArmTetris/SingleArmTetris/src
source /opt/ros/noetic/setup.bash
source ../devel/setup.bash
export PYTHONPATH="$PWD/operator_panel:${PYTHONPATH}"
/home/zhl/fr3env/fr3env/bin/python -m pytest -q operator_panel/tests
```

V1 提供手动控制里的 TCP 增量平移（dx/dy/dz），不提供网页 Mask 绘制、任意绝对 TCP 位姿运动、任务布局修改、标定矩阵逐项编辑、耗时/往复测试和局域网访问。
