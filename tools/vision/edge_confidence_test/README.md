# 边缘识别离线测试（不接入正式识别流程）

目的：验证能否用 **LSD + 白板 Mask + 侧面平行双边几何 + 相机方向验证**，
把有价值的上表面轮廓（高权重）与无用的侧面外边缘（低权重）区分开，
最终产出可用于轮廓模板匹配的 Edge Confidence Map。

## 文件说明

- `edge_confidence_map_test.py` — v1 简单版：LSD + 白板 mask 两级加权（0/85/255）
- `edge_pipeline_test.py` — v2 完整版：八阶段流水线（本轮主要测试）
- `output_v1/` — v1 的历史输出图
- `output/<图片名>/` — v2 每张测试图的输出（每图一个子目录，互不覆盖）

## v2 流水线阶段

| Stage | 内容 | 输出 |
|---|---|---|
| 1 | LSD 直线提取 + 短线过滤 | `01_lsd_raw.png` `02_lsd_length_filtered.png` |
| 2 | 共线线段合并 | `03_lsd_merged.png` |
| 3 | 白色背光板 Mask (HSV) | `04_white_mask.png` |
| 4 | 每条线两侧白板邻接比例 | `05_white_adjacency.png` |
| 5 | 侧面平行边对搜索（平行/法向距离/切向重叠） | — |
| 6 | 白板关系判定 A(非白-非白) / B(非白-白) 角色 | — |
| 7 | 相机方向验证（仅 pair 可信度，不单独删边） | — |
| 8 | 四类分类 + 权重 + 置信图 | `07` `08` `09` `10` |

分类颜色与权重：绿 TOP_BACKGROUND=3.0，蓝 TOP_SIDE=3.0，
红 SIDE_BACKGROUND=0.1（保留观察误判），灰 UNKNOWN=0.7。

`06_side_pairs.png`：同一侧面的两条线同色、中点连线，
标注 d=法向距离 a=角度差 ov=重叠比 cos=相机方向得分。

`99_debug_info.txt`：全部参数、每条线（L0..Ln）的长度/角度/两侧白比例/分类/配对，
以及所有几何候选边对和接受/拒绝结果。

## 运行

```bash
/home/zhl/fr3env/fr3env/bin/python edge_pipeline_test.py
```

参数全部在脚本开头"传参区"修改（图片路径、ROI、各阈值、权重），不走命令行。

## 验收关注点（对应需求文档第 17 节）

- 真实长边是否稳定检测出（01~03）
- TOP_SIDE（蓝）是否大多正确保留
- SIDE_BACKGROUND（红）是否大多明显降权
- 真正的 TOP_BACKGROUND（绿）不能因附近有平行边被误降权
- 两个方块靠近时是否出现错误 side pairing（06 重点检查）
