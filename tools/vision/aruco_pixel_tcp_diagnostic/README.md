# ArUco 高位像素—低位 TCP 独立诊断实验

## 目的

当前像素→TCP 标定数据总是不准。为了定位误差来源，本工具用一块 **ArUco 板**替换方块/托盘作为识别目标：ArUco 外框只负责识别 ID、方向和粗中心，最终像素取标记正中央黑白棋盘交叉点的灰度亚像素位置。高位、低位闭环和静止统计统一使用这一精中心，最后离线对比 affine / homography / poly2 / poly3 四种映射模型，判断原标定非线性主要来自**目标识别环节**还是**相机/几何链路**。

本工具完全独立，**不修改** competition / image_process / control / camera 的任何正式代码、服务或配置。

## 文件结构

```
tools/vision/aruco_pixel_tcp_diagnostic/
├── aruco_diagnostic_core.py    # 纯函数：ArUco 粗定位、中央角点精定位、绘图与统计（无 ROS）
├── run_aruco_experiment.py     # 实机实验入口（ROS）
├── aruco_frame_probe.py        # 相机画面探针：实时窗口 + 多字典诊断（ROS）
├── analyze_aruco_experiment.py # 离线模型对比
├── README.md
└── tests/
    └── test_aruco_diagnostic.py  # 无硬件单元测试
```

## 运行前提

- 已启动 `camera_node` 与 `controller`（相机、深度、控制服务在线）。
- 一张打印的 ArUco 板：字典 `DICT_6X6_50`，标记 ID=0（参数可在脚本开头修改）。
- ArUco 板要放在高、低位相机都能看到的区域，尽量平放。

## 实机实验

```bash
/home/zhl/fr3env/fr3env/bin/python tools/vision/aruco_pixel_tcp_diagnostic/run_aruco_experiment.py
```

流程（每次回车采集一个样本，直到输入 `q` 退出）：

1. 等待服务 → 强制吸盘 OFF → 校验参数 → 运动到高位拍摄位姿（`shooting_pose`）。
2. 提示移动 ArUco 板；**空回车开始采样，输入 `q` 正常结束**（仅在回到高位后退出）。
3. 高位连续采集 7 帧 ID=0 中央棋盘角点，逐轴取中位数；保存最接近中位数的一帧原图与调试图。
4. 对高位中心调用 `/camera/stable_world_points`（15 帧、至少 10 帧有效、2 秒超时），记录世界 XYZ、有效帧数、深度中位数与 MAD；MAD > 1.0 mm 判为深度失败。
5. 复用 `DepthRoughLocalizer.tcp_xy_from_world()` 得到粗 XY，组合固定 Z=220 mm 与固定 RPY（高位姿态的 RPY）。
6. 安全检查（6 维有限、Z 不低于 `competition/config/execution.yaml` 的
   `motion.minimum_tcp_z_mm`、XY 在 `perception.yaml` 安全范围内）通过后运动到低位。
7. 低位闭环：误差 = `中央精中心 - 图像中心`，复用正式 `run_offset_visual_servo_alignment()` 与 `pixel_to_robot_matrix`（阈值/步长/轮数/稳定帧/丢失帧全部对齐 `execution.yaml`）。每次修正位姿再次执行安全检查。
8. 对准成功后不再移动，额外采集 20 帧静止样本；有效比例 ≥80% 时生成零误差等效 TCP：
   `实测TCP_XY + pixel_to_robot_matrix @ 静止均值误差`（不应用限幅）。
9. 读取实测 TCP 与相机位姿；保存最接近均值的一帧低位原图与调试图。
10. CSV 每行写入后立即执行 flush + fsync，随后自动回高位等待下一个人工位置。

**低位首帧诊断**：运动到低位后、闭环开始前会先保存一帧（`样本NNN_低位首帧_原图/调试图.png`），并做多字典全量检测写入 CSV 的 `低位首帧检测ID` / `低位首帧最大标记像素尺寸` / `低位首帧rejected候选数` 列——即使样本失败，也能看出低位相机到底看到了什么、标记是否在画面内。

**低位伺服录像**：每个样本的低位闭环过程会录制 `样本NNN_低位伺服录像.avi`（每轮一帧，叠加轮次/误差/标记/中心参考，静止采样帧也在末尾）。配合 CSV 的 `低位伺服录像` 路径列，可直接回放机械臂修正过程，判断是测量噪声抖动还是运动震荡。

### 中央精定位原理

1. OpenCV ArUco 解码得到 ID 和四边形，对角线交点只作为亚像素优化初值。
2. 从字典生成当前 ID 的标准码图，确认中央 `2×2` 码格是黑白棋盘结构；ID=0 的中心为黑白对角交错。
3. 直接在原始灰度图使用 `cornerSubPix` 定位中央交叉点，不先二值化。
4. 精中心修正量不得超过 `0.12` 个码格，并且两个白格都必须比两个黑格至少亮 5 个灰度级；失败帧不会回退到粗中心。

调试图中绿色为 ArUco 粗四边形、青色为对角线粗中心、红色为中央精中心。CSV 同时记录粗中心与精中心统计，便于直接比较抖动。

### 中断后继续同一批次

默认创建新的时间戳目录：

```python
RESUME = False
RESUME_OUTPUT_DIR = None
```

需要从旧目录继续时，在 `run_aruco_experiment.py` 开头改为：

```python
RESUME = True
RESUME_OUTPUT_DIR = Path("/home/zhl/桌面/aruco诊断实验/20260808-220525")
```

续跑启动时会在机械臂运动前检查目录、CSV 表头、识别参数、运动配置和视觉伺服配置。任何条件与原批次不一致都会拒绝追加并打印差异。样本号取 CSV 和现有样本文件中的最大编号加一；硬中断留下的无 CSV 行文件会保留并自动跳号，绝不覆盖。元数据中的“运行历史”记录每次启动、结束和异常中断。

Ctrl+C 会立即停止发送运动命令，并在 `finally` 中关闭 CSV、保留已有数据。

## 相机画面探针（识别不出来的排查工具）

```bash
/home/zhl/fr3env/fr3env/bin/python tools/vision/aruco_pixel_tcp_diagnostic/aruco_frame_probe.py
```

只订阅相机话题，**不移动机械臂**。打开实时窗口（绿色四边形 = 检出的标记），操作：

- **回车**：保存当前帧（原图 + 调试图）到 `OUTPUT_DIR`，并打印该帧的多字典诊断：标准参数与放宽参数各一轮，列出每个字典检出的 ID 与标记像素尺寸，以及画面亮度统计。
- **q**：退出。

排查流程（例如"低位识别不到"）：

1. 把机械臂手动摆到低位观察位，运行探针看画面。
2. 若画面里**根本没有标记** → 深度/粗定位 XY 偏，相机没对准板子（这正是实验要查的相机/几何链路问题）。
3. 若标记在画面里但标准参数检不出 → 看放宽参数是否检出；若放宽也检不出，多半是反光/过曝/角度过大，调整板子摆放或光照。
4. 若用别的字典才检出 → 板子打印的字典不是 `DICT_6X6_50`，修改 `ARUCO_DICT_NAME` 后再实验。

### 参数

所有参数在 `run_aruco_experiment.py` 开头的“运行参数”区，直接改文件即可（不用命令行参数）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `ARUCO_DICT_NAME` / `ARUCO_MARKER_ID` | `DICT_6X6_50` / `0` | ArUco 字典与标记 ID |
| `CENTER_REFINE_WINDOW_RATIO` | `0.025` | 中央角点搜索半窗口占标记边长比例 |
| `CENTER_REFINE_MIN_WINDOW_PX` / `CENTER_REFINE_MAX_WINDOW_PX` | `3` / `15` | 亚像素搜索半窗口范围 |
| `CENTER_REFINE_MAX_SHIFT_CELL_RATIO` | `0.12` | 精中心相对粗中心最大修正码格比例 |
| `CENTER_REFINE_MIN_CONTRAST` | `5.0` | 中央白格与黑格最小灰度分离度 |
| `HIGH_SAMPLE_FRAMES` | `7` | 高位有效采样帧数 |
| `LOW_TCP_Z_MM` | `220.0` | 低位固定 TCP Z（mm） |
| `STATIC_SAMPLE_FRAMES` / `STATIC_MIN_VALID_RATIO` | `20` / `0.8` | 静止采样帧数与最小有效比例 |
| `DEPTH_FRAME_COUNT` / `DEPTH_MIN_VALID_FRAMES` / `DEPTH_CAPTURE_TIMEOUT_SEC` | `15` / `10` / `2.0` | 稳定深度参数 |
| `DEPTH_MAX_MAD_MM` | `1.0` | 深度 MAD 上限 |
| `OUTPUT_ROOT` | `~/桌面/aruco诊断实验` | 批次输出根目录 |
| `RESUME` | `False` | 是否从已有批次原目录继续 |
| `RESUME_OUTPUT_DIR` | `None` | `RESUME=True` 时必须指定的已有批次目录 |

## 输出

每个时间戳批次目录（`OUTPUT_ROOT/YYYYmmdd-HHMMSS/`）：

- `aruco实验数据.csv`：每次尝试一行（成功/失败均记录，utf-8-sig 可被 Excel 直接打开）
- `实验元数据.json`：参数、配置、运行历史、异常中断状态、累计与本次统计
- `样本NNN_高位/低位_原图/调试图.png`：每样本 4 张图，CSV 记录相对路径

CSV 关键列：高位精中心与粗中心、中央修正量和黑白分离度、高位/低位四角点、高位世界坐标 XYZ、深度统计、粗定位 TCP、最终命令 TCP、实测 TCP/相机六维位姿、最终精/粗像素误差、静止精/粗中心均值与标准差、静止采样完整标志、零误差等效 TCP XY、视觉伺服轮数、图片路径、失败原因。

## 离线分析

```bash
/home/zhl/fr3env/fr3env/bin/python tools/vision/aruco_pixel_tcp_diagnostic/analyze_aruco_experiment.py
```

输入 CSV 与输出目录在文件开头指定。复用正式标定分析的 `fit_xy_mapping()` / `predict_xy_mapping()` / `cross_validate_xy_mapping()` / `choose_xy_model()`：

- 主标签：`high_uv → 零误差等效TCP_XY`
- 辅助标签：`high_uv → 实测TCP_XY`
- 4 模型 × 5 折 × 固定种子 42；单个模型全量训练可行但某折无法训练时保留训练指标，CV 状态标为“数据不足”。

输出（中文文件）：

- `模型对比.csv`、`逐样本OOF误差.csv`、`分析报告.json`
- `模型CV误差对比.png`、`OOF误差空间分布.png`

**不生成、不覆盖任何正式 calibration YAML，也不做去畸变**；原始图片留给后续 raw/undistorted A/B 实验。

## 测试

```bash
/home/zhl/fr3env/fr3env/bin/python -m pytest tools/vision/aruco_pixel_tcp_diagnostic/tests
```

覆盖：合成图 ID=0 中央半像素角点；大尺寸低对比模糊图；透视变换；中央非棋盘 ID、低对比和过大修正量拒绝且不回退；错误 ID 与无标记；新建/连续续跑、表头只写一次、配置不一致拒绝、残留文件跳号和异常中断历史；位姿安全检查、零误差等效、离线模型分析与输出文件。

## 注意

- `image_process` 无 `__init__.py`，脚本已通过 `sys.path.insert` 让源码目录优先于 catkin `devel` 旧拷贝，不需要重新编译。
- 高位 `shooting_pose` Z=380 mm 与低位 Z=220 mm 都需看到板子，摆放位置由人工保证。
- 板子反光或角度过大时检测可能失败，会记入 CSV 失败行而不是中断实验。

## 检测尺度已知问题（已修复）

实机发现：**低位标记太大（约 447px，单格约 56px）时，OpenCV 默认自适应阈值窗口最大值 23px 会导致 `ids=None` 但 `rejected 非空`（找到外框但解码失败）**；高位约 227px 时默认参数刚好能过。修复：`create_aruco_detector` 默认自适应阈值窗口改为 **3/63/10**（Min=3, Max=63, Step=10），高位与低位两种尺度实测均可正常识别。

诊断区分方法（已内置）：

- `低位首帧rejected候选数` 列 / 探针打印的 `rejected 候选 N 个`：
  - `rejected=0` 且无 ID → 画面里没有标记（粗定位偏了，是相机/几何链路问题）。
  - `rejected≥1` 且无 ID → 有四边形但解码失败（尺度、光照、对比度问题）。
- 伺服失败时会额外保存 `低位失败帧_原图/调试图.png`，调试图中黄色四边形即 rejected 候选。
