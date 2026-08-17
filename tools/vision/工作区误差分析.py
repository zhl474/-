# -*- coding: utf-8 -*-
"""工作区位置相关误差挖掘：从伺服日志与定位全景数据中找"误差随工作空间位置变化"的证据。

核心思路：把数据里能观测到的量分成两类，分开统计——
  A. 机械臂侧观测量：实测TCP − 命令TCP（控制器跟踪误差，纯机械臂，与相机无关）
  B. 混合链路观测量：
     - 伺服修正量 Δ = 最终命令TCP − 粗定位TCP（部署标定的在位残差场）
     - 同批次重拟合 affine 的逐点残差 vs 位置（像素→TCP 映射的非线性部分）
     - 全景140格点对理想等距网格的仿射残差场（纯视觉+标定，机械臂不参与，作对照组）

直接运行：/home/zhl/fr3env/fr3env/bin/python 工作区误差分析.py
输出：控制台报告 + SAVE_PATH 处的 markdown 副本
"""
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

# ==== 参数（直接改这里）====
PANEL_ROOT = '/home/zhl/桌面/标定数据/实验日志'
PANORAMA_GLOB = '/home/zhl/桌面/高位定位全景_*.json'
ARUCO_DIRS = [
    '/home/zhl/桌面/aruco诊断实验/20260808-220525',
    '/home/zhl/桌面/aruco诊断实验/20260813-181610',
]
SAVE_PATH = '/home/zhl/桌面/工作区误差分析报告.md'
# 时代划分：panel 时间(HH) 前缀 → 标签；对应场地/标定的重大变更节点
ERA_RULES = [
    ('20260815', None, '0815'),
    ('20260816', range(0, 8), '0816凌晨'),
    ('20260816', None, '0816午后晚'),
    ('20260817', range(0, 8), '0817凌晨(实验室)'),
    ('20260817', None, '0817晚(场地重建后)'),
]

L_LINES = []


def log(s=''):
    print(s)
    L_LINES.append(s)


def era_of(panel_id):
    m = re.match(r'panel-(\d{8})-(\d{2})', panel_id or '')
    if not m:
        return '?'
    d, h = m.group(1), int(m.group(2))
    for date, hours, label in ERA_RULES:
        if d == date and (hours is None or h in hours):
            return label
    return d


def read_csv_rows(path):
    try:
        with open(path, encoding='utf-8-sig') as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


def fnum(row, key):
    v = row.get(key, '')
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fit_affine(src, dst):
    """最小二乘仿射 dst ≈ src @ A^T，返回 A 与残差 (dst - fit)。"""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    design = np.column_stack([src, np.ones(len(src))])
    coef, *_ = np.linalg.lstsq(design, dst, rcond=None)
    pred = design @ coef
    return coef, dst - pred


def pos_dependence(res, xy, name):
    """残差 res(N,) 对位置 xy(N,2) 的线性相关摘要：每100mm漂移 + R² + 半径相关。"""
    n = len(res)
    if n < 8:
        return f'  {name}: n={n} 样本不足'
    X, Y = xy[:, 0], xy[:, 1]
    design = np.column_stack([np.ones(n), X, Y])
    coef, *_ = np.linalg.lstsq(design, res, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((res - pred) ** 2))
    ss_tot = float(np.sum((res - res.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    r = np.hypot(X - X.mean(), Y - Y.mean())
    absd = np.abs(res)
    if r.std() > 1e-9 and absd.std() > 1e-9:
        rc = float(np.corrcoef(r, absd)[0, 1])
    else:
        rc = float('nan')
    rmse = float(np.sqrt(np.mean(res ** 2)))
    med = float(np.median(absd))
    p95 = float(np.percentile(absd, 95))
    return (f'  {name}: n={n} RMSE={rmse:.3f} 中位|e|={med:.3f} P95={p95:.3f} | '
            f'每100mm漂移 dX位置={coef[1]*100:+.2f} dY位置={coef[2]*100:+.2f} mm R²={r2:.2f} '
            f'| |e|与半径相关r={rc:+.2f}')


def load_servo_summaries():
    """所有panel的方块/托盘 每目标汇总CSV。"""
    out = []
    for panel_dir in sorted(glob.glob(os.path.join(PANEL_ROOT, 'panel-*'))):
        panel = os.path.basename(panel_dir)
        for kind in ('方块', '托盘'):
            rows = read_csv_rows(os.path.join(panel_dir, f'{kind}视觉伺服.csv'))
            for r in rows:
                ev = r.get('事件', '')
                if '伺服成功' not in ev and '成功' not in ev:
                    continue
                out.append({
                    'panel': panel, 'era': era_of(panel), 'kind': kind,
                    'cat': r.get('方块类别', ''),
                    'u': fnum(r, '高位检测像素X'), 'v': fnum(r, '高位检测像素Y'),
                    'tcp': (fnum(r, '实测TCP位置X'), fnum(r, '实测TCP位置Y'), fnum(r, '实测TCP位置Z')),
                    'cmd': (fnum(r, '最终命令TCP位置X'), fnum(r, '最终命令TCP位置Y'), fnum(r, '最终命令TCP位置Z')),
                    'coarse': (fnum(r, '粗定位TCP位置X'), fnum(r, '粗定位TCP位置Y'), fnum(r, '粗定位TCP位置Z')),
                    'n_corr': fnum(r, '执行修正次数'),
                })
    return out


def analyze_arm_tracking_rounds():
    """逐轮CSV: 本轮实测TCP − 末端命令（纯机械臂控制器跟踪误差）vs 命令位置。"""
    log('\n' + '=' * 78)
    log('A. 机械臂侧观测量：逐轮 实测TCP−末端命令 （控制器跟踪误差，与相机无关）')
    log('=' * 78)
    per_era = defaultdict(lambda: {'e': [], 'cmd': [], 'round_ms': []})
    for panel_dir in sorted(glob.glob(os.path.join(PANEL_ROOT, 'panel-*'))):
        panel = os.path.basename(panel_dir)
        for f in glob.glob(os.path.join(panel_dir, '*逐轮.csv')):
            for r in read_csv_rows(f):
                cx, cy, cz = fnum(r, '末端命令X'), fnum(r, '末端命令Y'), fnum(r, '末端命令Z')
                mx, my, mz = fnum(r, '本轮实测TCP位置X'), fnum(r, '本轮实测TCP位置Y'), fnum(r, '本轮实测TCP位置Z')
                if None in (cx, cy, cz, mx, my, mz):
                    continue
                e = era_of(panel)
                per_era[e]['e'].append((mx - cx, my - cy, mz - cz))
                per_era[e]['cmd'].append((cx, cy))
    for e in sorted(per_era):
        d = per_era[e]
        E = np.array(d['e'])
        C = np.array(d['cmd'])
        if len(E) < 10:
            log(f'\n[{e}] n={len(E)} 样本不足')
            continue
        log(f'\n[{e}] n={len(E)}轮（含高位/伺服各构型）')
        log(f'  跟踪误差分量统计(mm): ex μ={E[:,0].mean():+.4f} σ={E[:,0].std():.4f} | '
            f'ey μ={E[:,1].mean():+.4f} σ={E[:,1].std():.4f} | ez μ={E[:,2].mean():+.4f} σ={E[:,2].std():.4f}')
        log(f'  |e_xy| 中位={np.median(np.hypot(E[:,0],E[:,1])):.4f} P95={np.percentile(np.hypot(E[:,0],E[:,1]),95):.4f}')
        for i, nm in ((0, 'ex'), (1, 'ey'), (2, 'ez')):
            log(pos_dependence(E[:, i], C, nm))


def analyze_servo_field():
    """每目标汇总：重拟合残差场 + 伺服修正量场。"""
    log('\n' + '=' * 78)
    log('B. 混合链路：像素→实测TCP 重拟合残差 vs 位置（按时代/对象）')
    log('=' * 78)
    data = load_servo_summaries()
    groups = defaultdict(list)
    for s in data:
        if None in s['tcp'] or None in (s['u'], s['v']):
            continue
        groups[(s['era'], s['kind'])].append(s)
    for key in sorted(groups):
        rows = groups[key]
        if len(rows) < 10:
            log(f'\n[{key}] n={len(rows)} 样本不足')
            continue
        uv = [[s['u'], s['v']] for s in rows]
        xy = [[s['tcp'][0], s['tcp'][1]] for s in rows]
        coef, res = fit_affine(uv, xy)
        rmse = float(np.sqrt(np.mean(res ** 2)))
        log(f'\n[{key}] n={len(rows)} 重拟合affine RMSE={rmse:.3f} mm')
        xy = np.array(xy)
        log(pos_dependence(res[:, 0], xy, '残差dX'))
        log(pos_dependence(res[:, 1], xy, '残差dY'))
        # 四象限均值（以样本中心划分）
        xc, yc = xy[:, 0].mean(), xy[:, 1].mean()
        for nm, m in (('X<中位', xy[:, 0] < xc), ('X>中位', xy[:, 0] >= xc),
                      ('Y<中位', xy[:, 1] < yc), ('Y>中位', xy[:, 1] >= yc)):
            if m.sum() >= 3:
                log(f'    {nm}: n={m.sum()} dX均值={res[m,0].mean():+.2f} dY均值={res[m,1].mean():+.2f}')
        # 类别分组（识别环节贡献）
        bycat = defaultdict(list)
        for s, rr in zip(rows, res):
            bycat[s['cat']].append(rr)
        cats = sorted(bycat, key=lambda c: -len(bycat[c]))
        line = '  类别RMS(mm): ' + ', '.join(
            f"{c or '?'}={np.sqrt(np.mean(np.array(bycat[c])**2)):.2f}(n={len(bycat[c])})" for c in cats[:10])
        log(line)
    # 修正量场（部署标定在位残差）
    log('\n' + '=' * 78)
    log('C. 伺服修正量 Δ=最终命令TCP−粗定位TCP（部署标定的在位残差场，视觉+机械臂混合）')
    log('=' * 78)
    for key in sorted(groups):
        rows = [s for s in groups[key]
                if None not in s['coarse'] and None not in s['cmd'] and None not in s['tcp']]
        if len(rows) < 10:
            log(f'\n[{key}] n={len(rows)} 样本不足')
            continue
        coarse = np.array([[s['coarse'][0], s['coarse'][1]] for s in rows])
        cmd = np.array([[s['cmd'][0], s['cmd'][1]] for s in rows])
        d = cmd - coarse
        log(f'\n[{key}] n={len(rows)} |Δ|中位={np.median(np.hypot(*d.T)):.2f} P95={np.percentile(np.hypot(*d.T),95):.2f} mm '
            f'(μdX={d[:,0].mean():+.2f} μdY={d[:,1].mean():+.2f})')
        log(pos_dependence(d[:, 0], coarse, 'ΔX'))
        log(pos_dependence(d[:, 1], coarse, 'ΔY'))
    return data


def build_calibration_timeline():
    """从全景文件名时间戳 + 文件内记录的标定批次，构建部署标定时间线。"""
    tl = []
    for path in sorted(glob.glob(PANORAMA_GLOB)):
        m = re.search(r'_(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})\.json$', os.path.basename(path))
        if not m:
            continue
        ts = ''.join(m.groups()[:5])
        try:
            d = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        cal = d.get('标定', {})
        tl.append({
            'ts': ts,
            'tray_batch': cal.get('托盘', {}).get('实验批次', '?'),
            'block_batch': cal.get('方块', {}).get('实验批次', '?'),
        })
    tl.sort(key=lambda x: x['ts'])
    return tl


def panel_ts(panel_id):
    m = re.match(r'panel-(\d{8})-(\d{6})', panel_id or '')
    return m.group(1) + m.group(2) if m else ''


def analyze_servo_by_calibration():
    """按部署标定批次分组重新看伺服修正量场（消除'旧标定滞后'混淆）+ 逐panel时间漂移。"""
    log('\n' + '=' * 78)
    log("C2. 按'部署标定批次'分组的伺服修正量场（同批标定内的在位残差）+ 逐panel时间趋势")
    log('=' * 78)
    tl = build_calibration_timeline()
    if not tl:
        log('  无全景时间线，跳过')
        return
    data = load_servo_summaries()
    for kind, batch_key in (('托盘', 'tray_batch'), ('方块', 'block_batch')):
        groups = defaultdict(list)
        for s in data:
            if s['kind'] != kind:
                continue
            ts = panel_ts(s['panel'])
            dep = None
            for t in tl:  # 时间线有序，取最后一个早于panel的已知批次
                if t['ts'] <= ts and t[batch_key] != '?':
                    dep = t[batch_key]
            if dep:
                groups[dep].append(s)
        for dep in sorted(groups, key=lambda b: min(panel_ts(s['panel']) for s in groups[b])):
            rows = [s for s in groups[dep]
                    if None not in s['coarse'] and None not in s['cmd'] and None not in s['tcp']]
            if len(rows) < 12:
                continue
            coarse = np.array([[s['coarse'][0], s['coarse'][1]] for s in rows])
            cmd = np.array([[s['cmd'][0], s['cmd'][1]] for s in rows])
            d = cmd - coarse
            log(f'\n[{kind} 部署标定批次={dep}] n={len(rows)} '
                f'panels={sorted({s["panel"][-6:] for s in rows})}')
            log(f'  |Δ|中位={np.median(np.hypot(*d.T)):.2f} (μdX={d[:,0].mean():+.2f} μdY={d[:,1].mean():+.2f})')
            log(pos_dependence(d[:, 0], coarse, 'ΔX'))
            log(pos_dependence(d[:, 1], coarse, 'ΔY'))
            # 逐panel的均值漂移（热漂/机械松动会表现为随时间增长）
            by_panel = defaultdict(list)
            for s, dd in zip(rows, d):
                by_panel[s['panel']].append(dd)
            tr = []
            for p in sorted(by_panel, key=panel_ts):
                arr = np.array(by_panel[p])
                tr.append(f"{p[-6:]}:({arr[:,0].mean():+.1f},{arr[:,1].mean():+.1f})")
            log('  逐panel μΔ(X,Y)mm: ' + ' '.join(tr))
            # 二次项检验：Δ 是否含曲率
            n = len(rows)
            design2 = np.column_stack([np.ones(n), coarse, coarse ** 2, coarse[:, 0] * coarse[:, 1]])
            for i, nm in ((0, 'ΔX'), (1, 'ΔY')):
                coef, *_ = np.linalg.lstsq(design2, d[:, i], rcond=None)
                lin = np.column_stack([np.ones(n), coarse])
                cl, *_ = np.linalg.lstsq(lin, d[:, i], rcond=None)
                r2l = 1 - np.sum((d[:, i] - lin @ cl) ** 2) / np.sum((d[:, i] - d[:, i].mean()) ** 2)
                r2q = 1 - np.sum((d[:, i] - design2 @ coef) ** 2) / np.sum((d[:, i] - d[:, i].mean()) ** 2)
                log(f'    {nm}: 线性R²={r2l:.2f} 加入二次项R²={r2q:.2f}（差值大=有曲率/非线性成分）')


def analyze_refit_curvature():
    """B补充：affine vs 加二次项的像素→TCP拟合，看非线性是否呈畸变形态。"""
    log('\n' + '=' * 78)
    log('B2. 像素→TCP: affine vs +二次项 拟合对比（曲率=畸变/非线性证据）')
    log('=' * 78)
    data = load_servo_summaries()
    groups = defaultdict(list)
    for s in data:
        if None in s['tcp'] or None in (s['u'], s['v']):
            continue
        groups[(s['era'], s['kind'])].append(s)
    for key in sorted(groups):
        rows = groups[key]
        if len(rows) < 15:
            continue
        uv = np.array([[s['u'], s['v']] for s in rows], float)
        xy = np.array([[s['tcp'][0], s['tcp'][1]] for s in rows], float)
        n = len(rows)
        out = [f'[{key}] n={n}']
        for nm, design in (('affine', np.column_stack([uv, np.ones(n)])),
                           ('+quad', np.column_stack([uv, uv ** 2, uv[:, 0] * uv[:, 1], np.ones(n)]))):
            r2s = []
            for i in (0, 1):
                coef, *_ = np.linalg.lstsq(design, xy[:, i], rcond=None)
                r = xy[:, i] - design @ coef
                r2s.append(float(np.sqrt(np.mean(r ** 2))))
            out.append(f'{nm}: RMSE_x={r2s[0]:.3f} RMSE_y={r2s[1]:.3f} mm')
        log('  '.join(out))


def analyze_panorama():
    log('\n' + '=' * 78)
    log('D. 对照组：全景140格点对理想网格的仿射残差场（纯视觉+标定，机械臂静止不参与）')
    log('=' * 78)
    groups = defaultdict(list)
    for path in sorted(glob.glob(PANORAMA_GLOB)):
        try:
            d = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        pts = [p for p in d.get('托盘格点', [])
               if p.get('TCP_X') is not None and p.get('错误') == '']
        if len(pts) < 60:
            continue
        cal = d.get('标定', {}).get('托盘', {})
        gid = (d.get('模式', '?'), cal.get('实验批次', '?'), str(cal.get('文件sha256', ''))[:8])
        groups[gid].append((os.path.basename(path), pts))
    for gid in sorted(groups, key=lambda g: -len(groups[g])):
        files = groups[gid]
        n = len(files)
        cell_res = defaultdict(lambda: ([], []))
        rms_list = []
        for fname, pts in files:
            rc = np.array([[p['行'], p['列']] for p in pts], float)
            xy = np.array([[p['TCP_X'], p['TCP_Y']] for p in pts], float)
            _, res = fit_affine(rc, xy)
            rms_list.append(float(np.sqrt(np.mean(res ** 2))))
            for (r_, c_), rr in zip(rc.astype(int), res):
                cell_res[(r_, c_)][0].append(rr[0])
                cell_res[(r_, c_)][1].append(rr[1])
        # 跨文件平均场
        cells = sorted(cell_res)
        field = np.array([[np.mean(cell_res[k][0]), np.mean(cell_res[k][1])] for k in cells])
        noise = np.array([[np.std(cell_res[k][0]), np.std(cell_res[k][1])] for k in cells])
        field_rms = float(np.sqrt(np.mean(field ** 2)))
        noise_rms = float(np.sqrt(np.mean(noise ** 2)))
        rows_idx = sorted({k[0] for k in cells})
        cols_idx = sorted({k[1] for k in cells})
        log(f'\n[模式={gid[0]} 托盘标定批次={gid[1]} sha={gid[2]}] 文件数={n} 单文件RMS中位={np.median(rms_list):.3f} mm')
        log(f'  跨文件平均场RMS={field_rms:.3f} mm，跨文件噪声RMS={noise_rms:.3f} mm'
            f'（场/噪声比={field_rms/max(noise_rms,1e-6):.1f} → >1说明系统性空间场）')
        # 行/列均值剖面
        for nm, axis, idxs in (('按行', 0, rows_idx), ('按列', 1, cols_idx)):
            prof = []
            for it in idxs:
                sel = [i for i, k in enumerate(cells) if k[axis] == it]
                prof.append((it, field[sel, 0].mean(), field[sel, 1].mean()))
            line = f'  {nm}均值剖面 dX: ' + ' '.join(f'{it}:{dx:+.1f}' for it, dx, _ in prof)
            line2 = f'  {nm}均值剖面 dY: ' + ' '.join(f'{it}:{dy:+.1f}' for it, _, dy in prof)
            log(line)
            log(line2)


def analyze_aruco():
    log('\n' + '=' * 78)
    log('E. ArUco 像素→TCP 诊断（径向空间趋势）')
    log('=' * 78)
    for d in ARUCO_DIRS:
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith('分析报告.json') or f == '模型对比.csv':
                    p = os.path.join(root, f)
                    try:
                        if f.endswith('.json'):
                            j = json.load(open(p))
                            txt = json.dumps(j, ensure_ascii=False)
                            hits = re.findall(r'"([^"]*(?:径向|半径|相关)[^"]*)":\s*(-?\d+\.?\d*)', txt)
                            log(f'\n[{p}]')
                            for k, v in hits[:15]:
                                log(f'  {k} = {v}')
                        else:
                            log(f'\n[{p}]')
                            with open(p, encoding='utf-8-sig') as fh:
                                for i, line in enumerate(fh):
                                    if i < 8:
                                        log('  ' + line.rstrip()[:160])
                    except (OSError, json.JSONDecodeError) as ex:
                        log(f'  读取失败 {p}: {ex}')


def main():
    log('# 工作区位置相关误差分析报告')
    log(f'数据源: panel日志={PANORAMA_ROOT if (PANORAMA_ROOT := PANEL_ROOT) else ""} 全景={PANORAMA_GLOB}')
    analyze_arm_tracking_rounds()
    analyze_servo_field()
    analyze_refit_curvature()
    analyze_servo_by_calibration()
    analyze_panorama()
    analyze_aruco()
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L_LINES) + '\n')
    print(f'\n[已保存] {SAVE_PATH}')


if __name__ == '__main__':
    main()
