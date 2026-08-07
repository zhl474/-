# 高位 Mask 手动编辑 Demo

这是一个独立实验工具，只读取本地图片，不连接相机、ROS 或比赛流程，也不会修改输入原图。

同一目录还包含比赛流程使用的内部入口 `high_mask_session_editor.py`。它由图像
节点通过临时会话自动启动，不需要、也不应手工运行；该子进程不导入 ROS 和 YOLO，
比赛配置默认使用 CUDA 做局部模板匹配，也可通过
`high_mask_manual_editor.preview_device` 切换为 CPU。

## 准备图片

把一张高位原图放进本目录的 `input/`。支持 `.jpg`、`.jpeg` 和 `.png`，目录中必须正好只有一张图片。

## 运行

在项目 `src` 目录执行：

```bash
/home/zhl/fr3env/fr3env/bin/python tools/high_mask_editor_demo/high_mask_editor_demo.py
```

脚本顶部集中提供模型路径、置信度、ROI 外扩、画笔大小和窗口尺寸等参数，可以直接修改。

## 操作

总览窗口：

- `Ctrl + 鼠标左键`点击一个方块，进入对应 Mask 编辑窗口；普通左键保留给窗口拖动查看。
- 普通鼠标滚轮用于缩放总览。
- `T`：隐藏或显示总览中的全部绿色 Mask，不影响黄色模板轮廓。
- 编辑完成后会返回总览，可以继续点击其他方块。
- 按 `Enter`或 `Q` 提交并保存最终结果；按 `Esc`或直接关闭总览窗口取消整轮编辑。

在比赛流程中，`Enter`或 `Q` 会把编辑后的二值 Mask 提交给 ROS 父进程；总览
`Esc` 或关闭窗口会取消当前识别，不会沿用局部编辑结果。

编辑窗口：

- 普通鼠标左右键：保留给放大后的拖动查看，不修改 Mask。
- `Ctrl + 鼠标左键`：补画 Mask。
- `Ctrl + 鼠标右键`：擦除 Mask。
- 普通鼠标滚轮：缩放编辑视图。
- `Ctrl + 鼠标滚轮`或 `[`、`]`：调整画笔半径。
- `Z`：撤销上一笔。
- `Y`：重做最近一次被撤销的操作。
- `R`：恢复该方块最初的 Seg Mask。
- `T`：动态隐藏或显示黄色模板轮廓，仅影响局部编辑窗口显示。
- `Enter`：接受修改、保存该方块结果并返回总览。
- `Esc`：放弃本次修改并返回总览。

松开鼠标后会重新执行模板匹配，窗口会显示中心、角度和分数。Mask 为空或模板匹配失败时不能接受。

## 输出

输出写入 `output/<输入图片名>/`，包括：

- `overview_final.png`：最终总览图。
- `block_*_original_mask.png`：原始二值 Mask。
- `block_*_edited_mask.png`：编辑后的二值 Mask。
- `block_*_before.png` 和 `block_*_after.png`：修改前后模板匹配叠加图。
- `result.json`：检测框、编辑 ROI 和修改前后中心、角度、分数。
