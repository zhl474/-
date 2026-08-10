# Gemini 335 相机几何独立诊断

这组工具用于把三个问题拆开：

1. 工程收到的 1280×720 彩色图是否仍有残余镜头畸变；
2. D2C 对齐深度反投影后的相机坐标平面是否弯曲；
3. 修正彩色像素后，ArUco 高位像素到低位 TCP 的单应关系是否恢复。

工具不会修改 `camera_node`、机械臂流程或正式 pixel→TCP 配置。所有运行参数都在各脚本开头，直接修改后运行。

## 1. 相机配置探针

先停止占用 Gemini335 的 `camera_node`，然后运行：

```bash
/home/zhl/fr3env/fr3env/bin/python tools/vision/camera_geometry_diagnostic/camera_profile_probe.py
```

它严格检查彩色图和对齐深度都是 1280×720，保存工程彩色图、深度 `.npy`、显式 `remove_distortion()` 输出以及厂商 K/D、SDK版本和实例配置。探针图像差异本身不能证明输入是否已经去畸变，最终要结合圆点板留出误差判断。

## 2. RGB 非对称圆点板标定

编辑 `rgb_circle_calibration.py` 开头的 `MODE`：

- `board_check`：同时显示 `(4,7)` 和 `(7,4)` 的检测与点序；先用完整正视图确认。
- `capture`：订阅专用原始彩图 `/camera/image_raw`；正式 `camera_node` 只发布
  `/camera/image_rect`，需要重做 RGB 内参时应先停止正式节点并单独提供原始帧源。
- `calibrate`：重新检测保存的原图，求五参数模型、零畸变基线和5折留出误差。
- `verify`：加载已有 YAML，显示原图/去畸变图并打印指定像素修正量。

采集和离线求解固定使用同一个 `DETECTION_METHOD`，不会再逐帧在原始灰度、
CLAHE、反相和聚类之间切换。建议先在 `board_check` 中确定一种稳定方法：

```python
DETECTION_METHOD = "raw_white"  # 原始灰度白点，不使用聚类
```

其他可选值为 `raw_white_clustering`、`clahe_white`、
`clahe_white_clustering`、`inverted_black`、`inverted_black_clustering`。
方法一经开始采集就不要中途更换；若必须更换，应重新采一批图片。

当前 `BLOB_DETECTOR_OPTIONS` 是按现场保存原图收紧后的参数：真实白点中心灰度约
110以上，而旧参数会把灰度约37的黑底亮斑识别成第29个圆。导航面板现在显示
`候选圆点=n/28`；稳定画面应长期接近28/28。候选数不是28时先处理检测，不要保存。
任意时刻可按 `d` 把当前原图和完整导航界面保存到本批次的 `诊断帧/`，
该帧不会加入标定样本；空格/回车仍只保存检测成功且连续稳定的有效帧。

确认板型后必须设置：

```python
PATTERN_SIZE = (4, 7)       # 每行4点，共7行时
BOARD_CONFIRMED = True
GRID_STEP_MM = 你的实测值   # 相邻行纵向间距，也是同行水平圆心距的一半
GRID_STEP_IS_PHYSICAL = True
```

`capture` 右侧有定量导航面板：

- 连续5帧圆心抖动通过门禁后才允许保存；
- 板中心按九宫格统计中央、四边和四角；
- 尺寸按板长边占图像短边分为远（20%～34%）、中（34%～60%）、近（60%～82%）；
- 倾斜覆盖正视，以及左、右、上、下侧靠近相机四个方向；有效倾斜建议15°～30°；
- 面板持续显示当前数值、各类计数、总进度和下一步移动提示。

默认目标为至少20张、九个位置各至少1张、远/中/近分别至少4/8/4张、
正视至少4张、四个倾斜方向各至少3张。一张图片可同时满足位置、尺寸和倾斜三项。
倾角由约86°水平视场生成的近似内参估算，只用于采集导航，不进入最终标定计算。

运行：

```bash
/home/zhl/fr3env/fr3env/bin/python tools/vision/camera_geometry_diagnostic/rgb_circle_calibration.py
```

标定会保存 K/D、总体和逐图 RMS、最差图片、参数标准差、留出误差、旧项目 K/D 对照及 `quality_pass`。异常图只排序，不会自动删除。像素修正固定使用 `P=K`，不会混入裁剪或新内参造成的坐标变化。

## 3. 深度平面诊断

停止 `camera_node`，保持相机和圆点板静止，按距离分别修改：

```python
TEST_LABEL = "high_330mm"
APPROX_DISTANCE_MM = 330.0
```

再运行：

```bash
/home/zhl/fr3env/fr3env/bin/python tools/vision/camera_geometry_diagnostic/depth_plane_diagnostic.py
```

建议分别做 `high_330mm`、`near_170mm`、`control_500mm`。每次保存30帧原始深度、时间中位数、材料掩码、相机三维点 CSV、平面残差热力图和径向曲线。黑底有效率不足70%或白圆/黑底残差差超过1 mm 时，报告会明确标记材料混杂，不强行判断镜头曲率。二次曲面只用于留出诊断，不生成修正表。

## 4. ArUco 原始/去畸变 A/B

在 `tools/vision/aruco_pixel_tcp_diagnostic/analyze_aruco_experiment.py` 开头填写：

```python
CALIBRATION_YAML = Path("/圆点板标定结果/相机标定.yaml")
```

然后按原方式运行离线分析。新增输出包括：

- `畸变A_B模型对比.csv`
- `畸变A_B逐样本OOF误差.csv`
- `畸变A_B模型CV误差.png`
- `畸变A_B单应残差矢量.png`
- `分析报告.json` 中的 `畸变A_B`

只有去畸变后单应重复CV改善至少20%、距离最佳模型不超过5%、径向趋势同时减弱，才判定 RGB 畸变很可能是主要来源。分析不会写正式标定 YAML。

## 测试

```bash
/home/zhl/fr3env/fr3env/bin/python -m pytest tools/vision/camera_geometry_diagnostic/tests tools/vision/aruco_pixel_tcp_diagnostic/tests
```
