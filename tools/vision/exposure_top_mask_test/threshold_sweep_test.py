# -*- coding: utf-8 -*-
"""
曝光阈值分割实验（独立离线脚本，不接入正式识别流程）

硬件前提：调高曝光，让"方块侧面 + 白色背光板"都被推亮，上表面保持明显更暗。
若前提成立，上表面分割退化为 Lab-L 通道固定阈值：

    top_mask = (L < T)

本脚本用阈值扫描验证曝光方案是否真的稳定，而不是某一张图看起来能分开。

验收标准（先定后跑，避免事后挑阈值）：
  主判据：存在宽度 >= MIN_STABLE_WIDTH 灰度级的连续区间，
          区间内每个 T 满足 IoU(mask(T), mask(T+IOU_LAG)) >= IOU_STABLE_THRESH
  辅判据（填了采样框才有）：L_top_P99 < T < L_side_P1 的安全窗口宽度 >= MIN_STABLE_WIDTH
  两项都过 -> 曝光方案过关；只有孤立的好阈值 -> 不过关。

运行：改传参区后直接
/home/zhl/fr3env/fr3env/bin/python threshold_sweep_test.py
"""

import math
import os
from collections import OrderedDict

import cv2
import numpy as np

# ===================== 传参区（改这里） =====================
IMAGE_PATH = "/home/zhl/SingleArmTetris/SingleArmTetris/src/tools/vision/6.png"
ROI = None                # 如 (x, y, w, h)；None=整图。托盘网格是暗的会进 mask，可裁掉只看方块区

# 预处理
GAUSS_KSIZE = (3, 3)      # L 通道轻微模糊；None 则不模糊

# 阈值扫描
T_START, T_STOP, T_STEP = 120, 220, 5
CLOSE_KERNEL = 3          # close 核（填 1~2px 小洞）
DO_OPEN = False           # True 则再 open 一次去孤立噪点；第一版按方案只 close
OPEN_KERNEL = 3

# 稳定性判定
IOU_LAG = 10              # 比较闭区域mask(T)与mask(T+lag)的 IoU
IOU_STABLE_THRESH = 0.98
MIN_STABLE_WIDTH = 20     # 稳定区间最小宽度（灰度级）

# 手工采样框标定（像素坐标 [x1, y1, x2, y2]，ROI 裁剪后的坐标系；对应组留空则跳过）
# 建议每类取 2~4 个框：TOP 选方块上表面内部（避开边缘几 px），
# SIDE 选亮侧面带，BACKGROUND 选白板
SAMPLE_BOXES = {
    "TOP": [],
    "SIDE": [],
    "BACKGROUND": [],
}

# 可视化
SHOW_WINDOW = True
MONTAGE_SCALE = 0.30      # 拼图缩略比例
MONTAGE_COLS = 4

OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
# ==========================================================


# ---------------- 基础处理 ----------------
def build_close_mask(L, t):
    """L < T -> close（可选 open）-> uint8 0/255 mask"""
    mask = np.where(L < t, 255, 0).astype(np.uint8)
    kernel = np.ones((CLOSE_KERNEL, CLOSE_KERNEL), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if DO_OPEN:
        kernel_o = np.ones((OPEN_KERNEL, OPEN_KERNEL), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_o)
    return mask


def mask_iou(m1, m2):
    a = m1 > 0
    b = m2 > 0
    union = int(np.logical_or(a, b).sum())
    inter = int(np.logical_and(a, b).sum())
    return float(inter) / union if union > 0 else 1.0


def draw_overlay(bgr, mask):
    """原图 + mask 外轮廓（红色）"""
    vis = bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)
    return vis


# ---------------- 采样统计 ----------------
def stats_from_values(v):
    if v.size == 0:
        return None
    v = v.astype(np.float32)
    return OrderedDict([
        ("n", int(v.size)),
        ("min", float(v.min())),
        ("p1", float(np.percentile(v, 1))),
        ("p50", float(np.percentile(v, 50))),
        ("p99", float(np.percentile(v, 99))),
        ("max", float(v.max())),
    ])


def sample_stats(L, boxes):
    if not boxes:
        return None
    vals = []
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        vals.append(L[y1:y2, x1:x2].ravel())
    return stats_from_values(np.concatenate(vals))


def auto_sample_regions(bgr):
    """启发式自动采样（SAMPLE_BOXES 全空时使用，结果需结合 07 图人工核对）。

    原理：方块相关像素（上表面+被推亮的侧面）都保留较高饱和度（S>=90），
    而白板饱和度极低。彩色像素的 Lab-L 直方图若双峰：
        暗簇 = 上表面，亮簇 = 被曝光推亮的侧面，
    谷底即天然阈值，两侧簇的 P99/P1 给出安全窗口。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    colorful = S >= 90
    return {"COLORFUL": colorful}


def colorful_bimodal_stats(L, bgr):
    """彩色像素 L 双峰分析。返回 (stats_dict, regions_dict, valley) 或 (None, None, None)。"""
    regions = auto_sample_regions(bgr)
    colorful = regions["COLORFUL"]
    col = L[colorful]
    if col.size < 5000:
        return None, None, None
    # 在 [160, 230] 找直方图谷底（1 灰度级分辨率）
    lo, hi = 160, 230
    hist, _ = np.histogram(col, bins=np.arange(lo, hi + 2, 1))
    valley = lo + int(np.argmin(hist))
    dark = col[col < valley]
    bright = col[col >= valley]
    if dark.size < 1000 or bright.size < 1000:
        return None, None, None
    stats = {
        "TOP(暗簇)": stats_from_values(dark),
        "SIDE(亮簇)": stats_from_values(bright),
    }
    vis_regions = {
        "TOP": colorful & (L < valley),
        "SIDE": colorful & (L >= valley),
    }
    return stats, vis_regions, valley


def draw_sample_regions(bgr, regions):
    vis = bgr.copy()
    colors = {"TOP": (0, 255, 0), "SIDE": (0, 0, 255), "BACKGROUND": (255, 200, 0)}
    for name, mask in regions.items():
        if mask is None or not np.any(mask):
            continue
        vis[mask] = (0.5 * vis[mask] + 0.5 * np.array(colors.get(name, (255, 255, 255)))).astype(np.uint8)
    x0 = 10
    for i, (name, col) in enumerate(colors.items()):
        cv2.rectangle(vis, (x0, 10 + i * 26), (x0 + 18, 24 + i * 26), col, -1)
        cv2.putText(vis, name, (x0 + 26, 23 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# ---------------- 稳定区间 ----------------
def find_stable_ranges(ts, ious):
    """ious[i] 对应 ts[i]（值为 None 表示缺 T+lag 不可算）。
    稳定段 = 连续 iou 达标的 T 段，实际覆盖宽度延伸到段末 + IOU_LAG。"""
    ok = [i for i, v in enumerate(ious) if v is not None and v >= IOU_STABLE_THRESH]
    ranges = []
    if not ok:
        return ranges
    start = prev = ok[0]
    for i in ok[1:]:
        if i == prev + 1:
            prev = i
        else:
            ranges.append((ts[start], ts[prev] + IOU_LAG))
            start = prev = i
    ranges.append((ts[start], ts[prev] + IOU_LAG))
    return [(a, b) for a, b in ranges if b - a >= MIN_STABLE_WIDTH]


# ---------------- 拼图 / 曲线 ----------------
def build_montage(overlays, ts, ious):
    thumbs = []
    for img, t, io in zip(overlays, ts, ious):
        th = cv2.resize(img, None, fx=MONTAGE_SCALE, fy=MONTAGE_SCALE)
        bar = np.full((30, th.shape[1], 3), 30, dtype=np.uint8)
        label = "T=%d  IoU=%s" % (t, ("%.3f" % io) if io is not None else "--")
        color = (0, 255, 0) if (io is not None and io >= IOU_STABLE_THRESH) else (200, 200, 200)
        cv2.putText(bar, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 1, cv2.LINE_AA)
        thumbs.append(np.vstack([bar, th]))
    width = max(th.shape[1] for th in thumbs)
    thumbs = [np.pad(t_, ((0, 0), (0, width - t_.shape[1]), (0, 0)),
                     constant_values=20) for t_ in thumbs]
    rows = int(math.ceil(len(thumbs) / float(MONTAGE_COLS)))
    cols = min(MONTAGE_COLS, len(thumbs))
    grid = []
    for r in range(rows):
        row = thumbs[r * cols:(r + 1) * cols]
        while len(row) < cols:
            row.append(np.full_like(thumbs[0], 20))
        grid.append(np.hstack(row))
    return np.vstack(grid)


def save_stability_curves(ts, areas, ious, stable_ranges, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax1 = plt.subplots(figsize=(9, 4.5), dpi=120)
    ax1.plot(ts, areas, "o-", color="tab:blue", label="mask area (px)")
    ax1.set_xlabel("threshold T (Lab L)")
    ax1.set_ylabel("area", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(ts, [v if v is not None else float("nan") for v in ious],
             "s-", color="tab:red", label="IoU(T, T+%d)" % IOU_LAG)
    ax2.set_ylabel("IoU", color="tab:red")
    ax2.set_ylim(0.80, 1.01)
    ax2.axhline(IOU_STABLE_THRESH, ls="--", c="tab:red", alpha=0.5)
    for lo, hi in stable_ranges:
        ax1.axvspan(lo, hi, color="green", alpha=0.15)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)
    ax1.set_title("Threshold sweep stability (green = stable range)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


# ---------------- 主流程 ----------------
def main():
    bgr = cv2.imread(IMAGE_PATH)
    if bgr is None:
        raise FileNotFoundError("读不到图片: %s" % IMAGE_PATH)
    if ROI is not None:
        x, y, w, h = ROI
        bgr = bgr[y:y + h, x:x + w].copy()

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0].astype(np.float32)
    if GAUSS_KSIZE is not None:
        L = cv2.GaussianBlur(L, GAUSS_KSIZE, 0)

    stem = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    out_dir = os.path.join(OUTPUT_ROOT, stem)
    os.makedirs(out_dir, exist_ok=True)

    report = []
    report.append("image: %s  roi: %s  size: %dx%d" % (
        IMAGE_PATH, ROI, bgr.shape[1], bgr.shape[0]))
    report.append("params: T=%d..%d step %d  gauss=%s  close=%dx%d open=%s  "
                  "iou_lag=%d iou_thresh=%.2f min_width=%d" % (
                      T_START, T_STOP, T_STEP, GAUSS_KSIZE, CLOSE_KERNEL,
                      CLOSE_KERNEL, DO_OPEN, IOU_LAG, IOU_STABLE_THRESH,
                      MIN_STABLE_WIDTH))

    # ---------------- 1. 基础输出 ----------------
    cv2.imwrite(os.path.join(out_dir, "00_original.png"), bgr)
    cv2.imwrite(os.path.join(out_dir, "01_L_channel.png"),
                np.clip(L, 0, 255).astype(np.uint8))
    print("已保存 00_original.png / 01_L_channel.png")

    # ---------------- 2. 采样统计 ----------------
    # 优先用手工框；全空时退到彩色像素双峰分析（07 图可核对）
    manual = any(SAMPLE_BOXES.get(k) for k in ("TOP", "SIDE", "BACKGROUND"))
    stats = {}
    valley = None
    if manual:
        for name, boxes in SAMPLE_BOXES.items():
            st = sample_stats(L, boxes)
            if st is not None:
                stats[name] = st
        regions_note = "手工采样框"
    else:
        stats2, vis_regions, valley = colorful_bimodal_stats(L, bgr)
        if stats2 is not None:
            stats.update(stats2)
            cv2.imwrite(os.path.join(out_dir, "07_sample_regions.png"),
                        draw_sample_regions(bgr, vis_regions))
            print("已保存 07_sample_regions.png（绿=暗簇/上表面，红=亮簇/侧面，人工核对）")
            regions_note = "彩色像素双峰分析（谷底 T=%d）" % valley
        else:
            regions_note = "彩色像素过少，双峰分析不可用"

    report.append("")
    report.append("==== 采样统计（Lab-L，%s）====" % regions_note)
    window_lo, window_hi = None, None
    for name, st in stats.items():
        report.append(
            "%-12s n=%-8d min=%.0f p1=%.0f p50=%.0f p99=%.0f max=%.0f" % (
                name, st["n"], st["min"], st["p1"], st["p50"], st["p99"], st["max"]))
    if "TOP(暗簇)" in stats and "SIDE(亮簇)" in stats:
        window_lo, window_hi = stats["TOP(暗簇)"]["p99"], stats["SIDE(亮簇)"]["p1"]
        margin = window_hi - window_lo
        report.append("安全窗口: 暗簇p99=%.0f < T < 亮簇p1=%.0f  宽度=%.0f  %s" % (
            window_lo, window_hi, margin,
            "[达标>=20]" if margin >= MIN_STABLE_WIDTH else "[不足]"))
        report.append("推荐 T (窗口中点) = %.0f" % ((window_lo + window_hi) / 2.0))
    elif "TOP" in stats and "SIDE" in stats:
        window_lo, window_hi = stats["TOP"]["p99"], stats["SIDE"]["p1"]
        margin = window_hi - window_lo
        report.append("安全窗口: top_p99=%.0f < T < side_p1=%.0f  宽度=%.0f  %s" % (
            window_lo, window_hi, margin,
            "[达标>=20]" if margin >= MIN_STABLE_WIDTH else "[不足]"))
        report.append("推荐 T (窗口中点) = %.0f" % ((window_lo + window_hi) / 2.0))
    else:
        report.append("（缺 TOP/SIDE 采样，跳过安全窗口标定）")

    # ---------------- 3. 阈值扫描 ----------------
    ts = list(range(T_START, T_STOP + 1, T_STEP))
    masks = OrderedDict()
    overlays = []
    areas = []
    for t in ts:
        raw = np.where(L < t, 255, 0).astype(np.uint8)
        closed = build_close_mask(L, t)
        masks[t] = closed
        overlay = draw_overlay(bgr, closed)
        overlays.append(overlay)
        areas.append(int(np.count_nonzero(closed)))
        cv2.imwrite(os.path.join(out_dir, "02_threshold_raw_T%03d.png" % t), raw)
        cv2.imwrite(os.path.join(out_dir, "03_threshold_close_T%03d.png" % t), closed)
        cv2.imwrite(os.path.join(out_dir, "04_overlay_T%03d.png" % t), overlay)
    print("已保存 %d 档 02/03/04 输出" % len(ts))

    # ---------------- 4. 稳定性 ----------------
    ious = []
    for t in ts:
        t2 = t + IOU_LAG
        ious.append(mask_iou(masks[t], masks[t2]) if t2 in masks else None)
    stable_ranges = find_stable_ranges(ts, ious)

    report.append("")
    report.append("==== 扫描结果 ====")
    report.append("%-6s %-12s %-10s" % ("T", "area(px)", "IoU(T,T+%d)" % IOU_LAG))
    for t, a, io in zip(ts, areas, ious):
        report.append("%-6d %-12d %s" % (
            t, a, ("%.4f" % io) if io is not None else "--"))

    report.append("")
    report.append("==== 稳定区间（IoU>=%.2f，宽度>=%d）====" % (
        IOU_STABLE_THRESH, MIN_STABLE_WIDTH))
    if stable_ranges:
        for lo, hi in stable_ranges:
            report.append("  [%d, %d]  宽度 %d" % (lo, hi, hi - lo))
    else:
        report.append("  无")

    pass_main = bool(stable_ranges)
    pass_aux = (window_lo is not None and window_hi - window_lo >= MIN_STABLE_WIDTH)
    report.append("")
    report.append("==== 验收结论 ====")
    report.append("主判据(稳定区间): %s" % ("PASS" if pass_main else "FAIL"))
    if window_lo is not None:
        report.append("辅判据(安全窗口>=20): %s" % ("PASS" if pass_aux else "FAIL"))
    else:
        report.append("辅判据(安全窗口>=20): 未评（缺采样框）")
    if pass_main:
        lo, hi = max(stable_ranges, key=lambda r: r[1] - r[0])
        report.append("建议取稳定区间中点 T=%d 起步" % ((lo + hi) // 2))
    else:
        report.append("只有孤立好阈值 -> 曝光方案不稳定，先调硬件再谈软件")

    # ---------------- 5. 拼图 + 曲线 ----------------
    montage = build_montage(overlays, ts, ious)
    cv2.imwrite(os.path.join(out_dir, "05_montage.png"), montage)
    print("已保存 05_montage.png")
    if save_stability_curves(ts, areas, ious, stable_ranges,
                             os.path.join(out_dir, "06_stability_curves.png")):
        print("已保存 06_stability_curves.png")
    else:
        print("matplotlib 不可用，跳过 06 曲线图")

    report_path = os.path.join(out_dir, "99_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report) + "\n")
    print("已保存 99_report.txt")
    print("\n".join(report))

    if SHOW_WINDOW:
        cv2.imshow("montage (press any key to exit)",
                   cv2.resize(montage, None, fx=0.9, fy=0.9))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
