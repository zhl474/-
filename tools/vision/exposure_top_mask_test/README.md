# 曝光阈值分割实验（新路线，不接入正式识别流程）

硬件方案：调高曝光，让"方块侧面 + 白色背光板"都被推亮，上表面保持明显更暗。
若成立，上表面分割退化为 Lab-L 通道固定阈值 `top_mask = (L < T)`，
整个 LSD/边缘分类路线（见 `../edge_confidence_test/`）不再需要。

本实验回答一个问题：**固定阈值 T 是否存在足够宽的稳定区间**。

## 验收标准（先定后跑）

- 主判据：存在宽度 ≥20 灰度级的连续区间，区间内每个 T 满足
  `IoU(mask(T), mask(T+10)) ≥ 0.98`
- 辅判据（填了采样框才有）：`L_top_P99 < T < L_side_P1` 安全窗口宽度 ≥20
- 两项都过 → 曝光方案过关；只有孤立好阈值 → 不过关，先调硬件

## 运行

```bash
/home/zhl/fr3env/fr3env/bin/python threshold_sweep_test.py
```

参数全在脚本开头传参区：图片路径、ROI、扫描范围、形态学核、稳定性阈值、
手工采样框（TOP/SIDE/BACKGROUND 三组 `[x1,y1,x2,y2]`，留空跳过标定）。

## 输出（output/<图片名>/）

| 文件 | 内容 |
|---|---|
| `00_original.png` `01_L_channel.png` | 原图 / Lab-L 通道 |
| `02_threshold_raw_Txxx.png` | 每档 T 的原始阈值 mask |
| `03_threshold_close_Txxx.png` | 每档 close 3×3 后 mask |
| `04_overlay_Txxx.png` | 每档 mask 轮廓叠加原图 |
| `05_montage.png` | 全档拼图，绿标 = 该档 IoU 达标 |
| `06_stability_curves.png` | 面积曲线 + IoU 曲线 + 稳定区间高亮 |
| `99_report.txt` | 扫描表 / 稳定区间 / 采样统计 / 验收结论 |

## 注意事项

- 托盘网格是暗的，整图阈值会把网格划进 mask——实验聚焦方块区域即可，
  正式流程在 YOLO bbox 内做，天然排除；脚本留了 `ROI` 参数
- 黄色上表面最亮，单一 T 必须清得过黄色 top 的 P99；不行再按 YOLO 类别
  配多档 T（第二版）
- 背景若有暗影，第二版再加 `S > T_S` 饱和度条件；第一版只动 L

## 若验收通过，正式高位流程将简化为

```
高位固定曝光 → YOLO detect → bbox/class/center → Lab-L 固定阈值
→ top candidate mask → 3×3 轻修补 → 类别模板 + YOLO中心附近搜索 → px/py/theta
```
