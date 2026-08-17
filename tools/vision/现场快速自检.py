# -*- coding: utf-8 -*-
"""比赛现场快速自检：用最新一份高位定位全景 JSON 判断当前部署标定是否还能用。

30秒内回答三个问题：
  1) 标定是否与新场地脱节（网格间距≠20mm → 线性误差场，今晚实测 stale 标定=1.4%尺度误差）
  2) 托盘是否被挪动/转动（网格中心/旋转 vs 基线）
  3) 台面高度是否变化（方块表面Z vs 基线 → fixed_tcp_z 常数是否要调）

运行：/home/zhl/fr3env/fr3env/bin/python 现场快速自检.py
前提：刚跑过一轮 prepare（面板会自动在桌面生成"高位定位全景_*.json"）
"""
import glob
import json
import os
import re
from datetime import datetime

import numpy as np

# ==== 参数（直接改这里）====
PANORAMA_GLOB = '/home/zhl/桌面/高位定位全景_*.json'
# 基线：2026-08-17 晚 18:32–19:04 稳定窗口实测（当晚新标定 ef77851c 部署后）。
# 若明早重标定并通过验证，可跑 UPDATE_BASELINE=True 打印新基线贴回来。
UPDATE_BASELINE = False
BASELINE = {
    'spacing_row_mm': 19.992,   # 行方向相邻格点TCP间距（真值=托盘节距，标称20mm）
    'spacing_col_mm': 20.018,   # 列方向
    'grid_rot_deg': 179.448,    # 网格在TCP系的朝向
    'center_x_mm': -276.85,     # 网格中心
    'center_y_mm': 21.18,
    'block_surface_z_mm': 6.46, # 方块表面Z读数中位（台面高度指纹）
}
TRUE_PITCH_MM = 20.0
# 判据（超限即报警）
TOL = {
    'scale_pct': 0.20,      # 间距偏差%——0.2%=每100mm偏0.2mm；今晚stale案例1.4%
    'rot_deg': 0.15,        # 板旋转
    'center_mm': 2.0,       # 板平移
    'surface_z_mm': 1.5,    # 台面高度变化
    'min_grid_points': 100,
    'min_blocks': 25,
}


def newest_panorama_with_grid():
    cands = []
    for path in glob.glob(PANORAMA_GLOB):
        m = re.search(r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.json$', os.path.basename(path))
        if m:
            cands.append((m.group(1), path))
    for _, path in sorted(cands, reverse=True):
        try:
            d = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        pts = [p for p in d.get('托盘格点', []) if p.get('TCP_X') is not None and p.get('错误') == '']
        if len(pts) >= TOL['min_grid_points']:
            return path, d, pts
    return None, None, None


def analyze(path, d, pts):
    rc = np.array([[p['行'], p['列']] for p in pts], float)
    xy = np.array([[p['TCP_X'], p['TCP_Y']] for p in pts], float)
    coef, *_ = np.linalg.lstsq(np.column_stack([rc, np.ones(len(rc))]), xy, rcond=None)
    A = coef[:2, :2]
    res = xy - np.column_stack([rc, np.ones(len(rc))]) @ coef
    blocks = [b for b in d.get('方块', []) if not b.get('错误')]
    zs = [b['表面Z毫米'] for b in blocks if b.get('表面Z毫米') is not None]
    cal = d.get('标定', {}).get('托盘', {})
    return {
        'spacing_row': float(np.hypot(*A[:, 0])),
        'spacing_col': float(np.hypot(*A[:, 1])),
        'rot': float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))),
        'center': xy.mean(0),
        'res_rms': float(np.sqrt(np.mean(res ** 2))),
        'n_blocks': len(blocks),
        'n_outside': sum(1 for b in blocks if not b.get('凸包内', True)),
        'surf_z': float(np.median(zs)) if zs else None,
        'cal_batch': cal.get('实验批次', '?'),
        'cal_sha': str(cal.get('文件sha256', ''))[:8],
    }


def main():
    path, d, pts = newest_panorama_with_grid()
    if path is None:
        print('[失败] 桌面没有含140格点的全景JSON——先在面板跑一轮 prepare 再自检')
        return
    r = analyze(path, d, pts)
    stamp = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.json$', os.path.basename(path)).group(1)
    print(f'检查文件: {os.path.basename(path)}')
    print(f"部署标定: 批次={r['cal_batch']} sha={r['cal_sha']}"
          f"{'（=今晚ef77851c，正常）' if r['cal_sha'] == 'ef77851c' else '（注意：不是今晚最后部署的批次！）'}")
    print(f"格点n={len(pts)} 残差RMS={r['res_rms']:.2f}mm 方块n={r['n_blocks']}(凸包外{r['n_outside']})")
    print()

    verdicts = []

    def check(name, ok, detail, action=''):
        tag = '[OK]  ' if ok else ('[失败]' if '重标定' in action or '失败' in action else '[警告]')
        verdicts.append(ok)
        print(f"{tag}{name}: {detail}")
        if action:
            print(f"      → {action}")

    # 1) 尺度（stale标定指纹）
    sc_row = (r['spacing_row'] / TRUE_PITCH_MM - 1) * 100
    sc_col = (r['spacing_col'] / TRUE_PITCH_MM - 1) * 100
    sc = max(abs(sc_row), abs(sc_col))
    check('网格间距(标定尺度)',
          abs(sc) <= TOL['scale_pct'],
          f"行{r['spacing_row']:.3f}mm({sc_row:+.2f}%) 列{r['spacing_col']:.3f}mm({sc_col:+.2f}%) "
          f"vs 真值{TRUE_PITCH_MM}mm基线{BASELINE['spacing_row_mm']:.2f}/{BASELINE['spacing_col_mm']:.2f} | "
          f"等效线性场≈{abs(sc):.2f}mm/100mm",
          '尺度偏差>0.2%说明部署标定与当前场地脱节（今晚stale案例1.4%）→ 重标定' if abs(sc) > TOL['scale_pct'] else '')

    # 2) 板姿态
    drot = r['rot'] - BASELINE['grid_rot_deg']
    drot = (drot + 180) % 360 - 180
    dc = r['center'] - np.array([BASELINE['center_x_mm'], BASELINE['center_y_mm']])
    check('托盘位置/旋转',
          abs(drot) <= TOL['rot_deg'] and np.hypot(*dc) <= TOL['center_mm'],
          f"旋转差{drot:+.3f}° 中心差({dc[0]:+.1f},{dc[1]:+.1f})mm",
          '板被挪动或格子识别条件变化：确认托盘没被动过；若确认没动，重标定' if abs(drot) > TOL['rot_deg'] or np.hypot(*dc) > TOL['center_mm'] else '')

    # 3) 台面高度
    if r['surf_z'] is not None:
        dz = r['surf_z'] - BASELINE['block_surface_z_mm']
        check('台面高度(方块表面Z)',
              abs(dz) <= TOL['surface_z_mm'],
              f"读数中位{r['surf_z']:.2f}mm vs 基线{BASELINE['block_surface_z_mm']:.2f} 差{dz:+.2f}mm",
              '台面高度变化：fixed_tcp_z两常数(173.46/182.46)可能整体偏移，先试抓一次再决定±调节量并重启' if abs(dz) > TOL['surface_z_mm'] else '')
    else:
        print('[警告]台面高度: 无方块表面Z数据（场上没方块？）')

    # 4) 识别链路
    check('识别链路',
          r['n_blocks'] >= TOL['min_blocks'] and len(pts) >= TOL['min_grid_points'],
          f"方块{r['n_blocks']}个 格点{len(pts)}个",
          '方块<25个：先查曝光/照明再重跑prepare（今晚z_blue漏检先例）' if r['n_blocks'] < TOL['min_blocks'] else '')

    if UPDATE_BASELINE:
        print()
        print('# 贴回文件顶部的新基线：')
        print(f"BASELINE = {{'spacing_row_mm': {r['spacing_row']:.3f}, 'spacing_col_mm': {r['spacing_col']:.3f},")
        print(f"    'grid_rot_deg': {r['rot']:.3f}, 'center_x_mm': {r['center'][0]:.2f}, 'center_y_mm': {r['center'][1]:.2f},")
        print(f"    'block_surface_z_mm': {r['surf_z']:.2f}}}")
        return
    print()
    if all(verdicts):
        print('★ 结论：当前标定有效，可以直接试抓（建议抓最远角位1-2次验证）')
    else:
        print('★ 结论：有项目报警——按上面每条的 → 行动；全部处理完再跑一次本脚本确认')


if __name__ == '__main__':
    main()
