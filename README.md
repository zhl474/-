# SingleArmTetris 粗定位与视觉伺服

正式流程只有一条：高位托盘/方块识别与粗定位，随后对每个任务执行方块低位视觉伺服抓取和托盘低位视觉伺服摆放。旧开环抓放不再进入运行代码。

## 代码结构

- `competition/competition_lib/`：任务状态机、硬件服务客户端、执行配置和通用视觉伺服闭环。
- `image_process/image_process_lib/`：高低位检测、深度粗定位、任务规划和图像 ROS 服务。
- `camera/`：彩色图像发布与像素转世界坐标服务。
- `control/`：机械臂、末端舵机和电子吸盘控制服务。
- `tools/`：视觉标定、偏心补偿和硬件单项测试，不会由正式 launch 启动。

## 正式 ROS 服务

- `/perception/prepare_task`：用一帧高位图像准备基础或进阶任务。
- `/perception/get_task_target`：返回一个完整抓放任务。
- `/perception/block_offset`：返回低位方块像素偏差。
- `/perception/board_offset`：返回低位托盘目标像素偏差。
- `/camera/pixel_to_world`：返回显式成功状态和世界坐标。
- `/control/move_arm`、`/control/rotate_tool`、`/control/set_suction`：硬件控制。

## 配置

- `competition/config/execution.yaml`：拍摄位姿、速度、抓放高度、伺服迭代限制和舵机边界。
- `competition/config/visual_servo.yaml`：像素到机械臂映射及相机到吸盘偏移。
- `image_process/config/perception.yaml`：模型、粗定位回退和低位检测参数。
- `image_process/config/task_layout.yaml`：基础任务唯一摆放表。

深度粗定位失败和分割失败回退均可在 `perception.yaml` 中关闭。回退启用时，日志会明确报告实际使用的定位或分割来源。

## 运行与测试

按项目要求使用指定虚拟环境：

```bash
/home/zhl/fr3env/fr3env/bin/python -m pytest -q
roslaunch competition competition.launch
```

标定与硬件测试脚本通过修改文件开头参数运行，例如：

```bash
/home/zhl/fr3env/fr3env/bin/python tools/hardware/测试舵机.py
```

硬件回归顺序固定为：仅启动节点、高位识别不运动、单块低速抓放、完整基础任务、完整进阶任务。视觉伺服失败后禁止继续下探；持块摆放失败时保持吸盘状态并停止自动运动。
