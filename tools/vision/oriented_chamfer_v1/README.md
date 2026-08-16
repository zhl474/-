# V1 Oriented Chamfer 顶面配准离线测试

完全不接入正式识别流程。正式流程文件没有被修改。

## 文件

- `oriented_chamfer_v1.py`：可复用核心库。
  - 只读复用 `competition/model/best5.14.pt`
  - 只读复用 `competition/config/template_config.yaml`
  - 只读复用 `image_process_lib.template_match.kernels_create.create_base_shape`，
    保证模板形状和正式流程完全一致。
- `run_v1_test.py`：离线批量测试 / 手动单框测试入口。
- `output/`：运行输出，不提交。

## V1 思路

```text
YOLO 检测框 + 类别
        │
        ▼
按模板最大半径 + 搜索半径扩大 ROI
        │
        ▼
ROI Canny + distanceTransformWithLabels
        │
        ▼
已知类别的顶面外轮廓（正式 create_base_shape 的外轮廓）
        │
        ▼
粗搜 (x, y, θ)：
  θ step 2°，x/y step 1px，中心 ±14px
        │
        ▼
对 top-K 粗搜候选做精搜：
  θ step 0.2°，x/y step 0.5px
        │
        ▼
输出 px/py/pose_angle/theta/score 和调试图
```

打分（越低越好）：

```text
score = mean(min(chamfer_distance, 20))
      + 0.8 * (加权边缘方向误差 / 90)
      + 0.005 * ((x-cx)^2 + (y-cy)^2)
```

方向项不是“删边”，而是让正交轮廓（θ 与 θ+90°）更有吸引力；
中心项只是软先验，避免锁死 YOLO bbox 中心。

## 运行

```bash
cd /home/zhl/SingleArmTetris/SingleArmTetris/src

# 批量测试 image7 的全部 YOLO 检测
YOLO_CONFIG_DIR=/tmp/ultralytics python3 \
  tools/vision/oriented_chamfer_v1/run_v1_test.py \
  --image tools/vision/7.png \
  --output-dir /tmp/v1_img7

# 只测 T，并且不用 YOLO，手动给框
YOLO_CONFIG_DIR=/tmp/ultralytics python3 \
  tools/vision/oriented_chamfer_v1/run_v1_test.py \
  --image tools/vision/7.png \
  --category T --box 969 27 1101 134 \
  --output-dir /tmp/v1_t

# 关掉方向项，对比普通 Chamfer + 中心先验
... --orientation-weight 0
```

如果运行环境的 `~/.config` 只读，可以加 `YOLO_CONFIG_DIR=/tmp/ultralytics`
避免 ultralytics 写 settings 报错；比赛机上一般不需要。

## 输出

每个目标：

- `NN_category_px_py_roi_edges.png`：ROI Canny + 最终模板轮廓 + YOLO 中心到结果的位移
- `NN_category_px_py_full_crop.png`：原图 ROI 附近 + 检测框 + 最终轮廓
- `summary_full.png`：全图汇总
- `v1_results.json`：所有位姿、分数分量、覆盖率和耗时

## 角度说明

- `pose_angle_deg`：本库内部使用的模板生成角，和
  `kernels_create.rotate_image` / `cv2.getRotationMatrix2D` 正方向一致。
- `theta`：按正式流程 `get_rect` 的符号做了转换，范围 `[-180, 180)`，
  未来若接到正式链路，应和 `match_block_mask` 输出的 `theta` 对齐。

## 当前默认参数

```python
Canny:            50 / 150, 3x3 Gaussian
模板轮廓采样:      1.5 px
粗搜:             θ=2°, xy=1px, 中心 ±14px, topK=5
精搜:             θ=0.2°, xy=0.5px, θ窗口 ±2.5°, xy窗口 ±3px
chamfer 截断:      20 px
方向权重:          0.8, 只统计 4px 内边缘
中心先验:          0.005 * 偏移平方
```

## 已做离线验证（2026-08-16）

- `tools/vision/7.png` 35 个目标全部跑通，绝大多数 `score < 2`，
  `coverage_2px` 接近 1.0。
- 手动 T 框 `(969,27,1101,134)`：
  - V1：`px=1046.0, py=70.0, pose_angle=337.3°`
  - 与正式 best_seg.pt mask + 模板匹配结果一致（`px=1046, py=70`）。
- 左侧被裁切/边缘很弱的方块，`score` 会自然升高（例如 `>3`），
  这就是后续 V2 的 `edge confidence` 和“低置信回退正式 mask 流程”的触发依据。

## V2 预留

V2 只在这个库上增加 `EdgeConfidence`，不需要改正式流程：

1. 对每个 Canny 边缘点按两侧颜色分类；
2. 生成 0..1 权重场；
3. `_score_candidates` 的距离项改成加权 Chamfer：
   `mean(w_edge * min(distance, cap))`。

当前 `_score_candidates` 已经把距离、方向、中心项分离，V2 可以只替换
距离项内部实现。
