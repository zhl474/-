# SingleArmTetris 粗定位与视觉伺服

正式流程先执行高位托盘/方块识别与粗定位，再根据 YAML 开关选择低位视觉伺服闭环抓放或高位标定结果开环抓放。两种流程共用相机到吸盘偏移和运动安全校验。

## 代码结构

- `competition/competition_lib/`：任务状态机、硬件服务客户端、执行配置和通用视觉伺服闭环。
- `image_process/image_process_lib/`：高低位检测、像素到 TCP 标定定位、任务规划和图像 ROS 服务。
- `camera/`：彩色/深度图采集，以及标定模式使用的批量稳定世界 XYZ 服务。
- `control/`：机械臂、末端舵机和电子吸盘控制服务。
- `tools/`：视觉标定、偏心补偿和硬件单项测试，不会由正式 launch 启动。

## 正式 ROS 服务

- `/perception/prepare_task`：用一帧高位图像准备基础或进阶任务。
- `/perception/get_task_target`：返回一个完整抓放任务。
- `/perception/block_offset`：返回低位方块像素偏差。
- `/perception/board_offset`：返回低位托盘目标像素偏差。
- `/camera/stable_world_points`：用同一批深度帧和同一次相机位姿返回多个表面点的基坐标 XYZ，仅供标定模式使用。
- `/control/move_arm`、`/control/rotate_tool`、`/control/set_suction`：硬件控制。

## 配置

- `competition/config/execution.yaml`：拍摄位姿、速度、抓放高度、伺服迭代限制和舵机边界。
- `competition/config/visual_servo.yaml`：像素到机械臂映射及相机到吸盘偏移。
- `image_process/config/perception.yaml`：模型、高位 TCP 标定安全范围、高度模式和低位检测参数。
- `image_process/config/*_pixel_to_tcp_calibration.yaml`：方块和托盘各自的正式部署标定。
- `image_process/config/task_layout.yaml`：基础任务唯一摆放表。

`execution.yaml` 根节点的 `calibration_mode` 开关已移除，标定/正式模式由 launch 文件的
`calibration_mode` 参数决定：`roslaunch competition competition.launch` 传 `false`（正式抓放），
`roslaunch competition calibration.launch` 传 `true`（标定采集）。正式抓放只加载部署的
成对标定 YAML，不创建也不等待深度坐标客户端；标定采集时深度相机提供当前现场完整 XYZ
粗定位，低位视觉伺服只修正 XY，全程关闭吸气/吹气，并覆盖记录方块和托盘标定 CSV。

`execution.yaml` 的 `servo.enabled` 控制正式运行是否启用方块和托盘低位视觉伺服。
`true` 保持“粗观察位、视觉对准、偏置、抓放”的闭环流程；`false` 直接对方块或
托盘各自的高位标定结果应用吸盘偏置并开环执行。开环抓取仍严格执行“到达吸盘上方、
下探、吸取、抬回上方”，托盘摆放则保留托盘标定 Z 直接释放。配置在节点启动时读取，
修改后需要重启；标定采集模式会警告并强制开启视觉伺服。

正式模式的高位粗定位固定使用方块、托盘各自的 schema v1 或成对 schema v2 标定。
schema v2 将像素到 TCP XY 模型与 TCP Z 平面拆开；方块和托盘必须属于同一生成批次，
且托盘 Z 平面始终比方块观察 TCP 平面低 `7.0 mm`。标定样本凸包只用于分析采样覆盖，
不作为正式抓取范围。

标定模式不读取旧像素标定。方块深度点是上表面世界 XYZ，方块观察 TCP Z 为表面
Z 加 `192 mm`，最终抓取 TCP Z 为表面 Z 加 `162 mm`；托盘只使用深度世界 X/Y，
深度 Z 完全不参与控制高度。稳定深度默认缓存 15 帧、至少 10 帧有效，方块 MAD 和
方块观察平面 RMSE 均不得超过 `1.0 mm`，失败时禁止开始低位运动且不回退旧标定。
颜色分割回退只属于方块上表面识别，不参与高位 TCP 定位。

## 运行与测试

按项目要求使用指定虚拟环境：

```bash
PYTHONPATH=image_process:competition:. /home/zhl/fr3env/fr3env/bin/python -m pytest -q
roslaunch competition competition.launch
roslaunch competition calibration.launch
```

标定采集会识别画面内全部方块并逐个拾取；若同时识别到托盘，还会从 14×10 格点及其
半格位置随机抽取 34 个目标（整数点 9、左右中点 9、上下中点 8、四点中心 8）逐一抵达。
没有托盘时只采集方块，任意数量、任意类别组合均可成功；有托盘时至少需要 3 个不共线
方块提供观察 Z 平面。

高位拍摄会等待机械臂停稳，并只使用任务请求之后发布的新图像。标定文件缺失、主体错配、
预测非有限或 TCP 超出安全范围都会让本轮高位准备失败，由交互流程重新识别。

标定与硬件测试脚本通过修改文件开头参数运行，例如：

```bash
/home/zhl/fr3env/fr3env/bin/python tools/hardware/测试舵机.py
```

像素到 TCP 标定脚本一次分析方块和托盘 CSV：

```bash
/home/zhl/fr3env/fr3env/bin/python tools/vision/pixel_to_tcp_calibration_analysis.py
```

候选产物位于 `tools/vision/像素-tcp标定结果与数据分析/`。只有两份 CSV 的数据数量、
深度门禁、方块 Z 平面和独立 XY 映射全部通过时，才生成同批次的两份 schema v2 YAML；
确认报告并低速验证托盘中心和四角后，再同时手动复制到 `image_process/config/`。

硬件回归顺序固定为：仅启动节点、高位识别不运动、单块低速抓放、完整基础任务、完整进阶任务。视觉伺服失败后禁止继续下探；持块摆放失败时保持吸盘状态并停止自动运动。
