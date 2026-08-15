# -*- coding: utf-8 -*-
"""
边缘分类 + Edge Confidence Map 完整流水线（离线测试，独立脚本，不接入正式识别流程）

背景：
立体俄罗斯方块存在可见侧面，分割轮廓整体向侧面方向偏移，模板匹配被带偏。
本轮只验证：能否用 LSD + 白板 Mask + 平行双边几何 + 相机方向验证，
把有价值的上表面轮廓与无用的侧面外边缘区分开。不做模板匹配。

Stage 1  LSD 直线提取 + 短线过滤             -> 01_lsd_raw / 02_lsd_length_filtered
Stage 2  共线线段合并                         -> 03_lsd_merged
Stage 3  白色背光板 Mask(HSV)                 -> 04_white_mask
Stage 4  每条线两侧白板邻接比例               -> 05_white_adjacency
Stage 5  侧面平行边对搜索（平行/法向距离/重叠）
Stage 6  白板关系判定 A(非白-非白)/B(非白-白) 角色
Stage 7  相机方向验证（仅用于 pair 可信度，不单独删边）
Stage 8  四类分类 + 权重                      -> 07_edge_classification
         Edge Confidence Map（线周衰减扩散）   -> 08_edge_confidence_map
额外：09 分类叠加原图、10 四联对比图、99_debug_info.txt

分类与权重：
  TOP_BACKGROUND  上表面-白板    绿   3.0
  TOP_SIDE        上表面-侧面    蓝   3.0
  SIDE_BACKGROUND 侧面-白板      红   0.1（保留观察误判，不直接删除）
  UNKNOWN         无法判断       灰   0.7

运行：改下方参数后直接跑
/home/zhl/fr3env/fr3env/bin/python edge_pipeline_test.py
"""

import cv2
import math
import os
from collections import Counter

import numpy as np

# ===================== 传参区（改这里） =====================
IMAGE_PATH = "/home/zhl/SingleArmTetris/SingleArmTetris/src/tools/vision/3.png"
ROI = None                # 如 (x, y, w, h)；None = 整图

OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# --- Stage 1: LSD ---
GAUSS_KSIZE = (3, 3)      # LSD 前轻微模糊；None 则不模糊
MIN_LINE_LENGTH = 15.0    # 短线过滤阈值 px

# --- Stage 2: 共线合并 ---
MERGE_ANGLE_THRESH_DEG = 3.0    # 角度差小于此值才可能共线
MERGE_NORMAL_DISTANCE_PX = 2.5  # 所在直线法向距离
MERGE_TANGENT_GAP_PX = 8.0      # 切向投影间隔（重叠或间隔很小才合并）

# --- Stage 3: 白板 mask (HSV: V 高 + S 低) ---
WHITE_V_MIN = 180
WHITE_S_MAX = 60
MORPH_KSIZE = 3

# --- Stage 4: 两侧白板邻接 ---
SIDE_SAMPLE_GAP_PX = 1.0    # 离线段多少 px 后开始采样（避开边缘抗锯齿）
SIDE_SAMPLE_WIDTH_PX = 4.0  # 条带宽度 3~5px
WHITE_ADJACENT_THRESH = 0.75  # max(两侧比例) 超过 -> 白板邻接
SIDE_WHITE_RATIO = 0.7      # 单侧比例 >= 此值 -> 该侧视为白
SIDE_NONWHITE_RATIO = 0.3   # 单侧比例 <= 此值 -> 该侧视为非白

# --- Stage 5: 侧面平行边对 ---
PARALLEL_ANGLE_THRESH_DEG = 4.0  # 考虑 180° 等价
SIDE_MIN_WIDTH_PX = 3.0
SIDE_MAX_WIDTH_PX = 25.0
OVERLAP_RATIO_THRESH = 0.5

# --- Stage 7: 相机方向验证 ---
OBJECT_CENTER_UV = None            # 如 (640, 360)；None = 用线段中点均值估计
CAMERA_DIRECTION_COS_THRESH = 0.3  # 宽松阈值，仅作 pair 验证

# --- Stage 8: 分类权重 ---
W_TOP_BACKGROUND = 3.0
W_TOP_SIDE = 3.0
W_UNKNOWN = 0.7
W_SIDE_BACKGROUND = 0.1

# --- 可视化 ---
LINE_THICKNESS = 2
SHOW_WINDOW = True      # False 则只存图不弹窗
PANEL_SCALE = 0.35

CLASS_COLOR = {
    "TOP_BACKGROUND": (0, 255, 0),    # 绿
    "TOP_SIDE": (255, 0, 0),          # 蓝
    "SIDE_BACKGROUND": (0, 0, 255),   # 红
    "UNKNOWN": (160, 160, 160),       # 灰
}
CLASS_WEIGHT = {
    "TOP_BACKGROUND": W_TOP_BACKGROUND,
    "TOP_SIDE": W_TOP_SIDE,
    "SIDE_BACKGROUND": W_SIDE_BACKGROUND,
    "UNKNOWN": W_UNKNOWN,
}

# 置信图线周衰减：中心 1.0，±1px 0.8，±2px 0.5，±3px 0.2
CONF_PROFILE = [(0, 1.0), (1, 0.8), (-1, 0.8), (2, 0.5), (-2, 0.5), (3, 0.2), (-3, 0.2)]
# ==========================================================


# ---------------- 基础数据结构 ----------------
class Line(object):
    _next_id = 0

    def __init__(self, x1, y1, x2, y2):
        # 统一方向：dx>0，或 dx==0 且 dy>0，保证角度落在 [0,180)
        if (x2 < x1) or (x2 == x1 and y2 < y1):
            x1, y1, x2, y2 = x2, y2, x1, y1
        self.p1 = np.array([x1, y1], dtype=np.float64)
        self.p2 = np.array([x2, y2], dtype=np.float64)
        d = self.p2 - self.p1
        self.length = float(np.hypot(d[0], d[1]))
        if self.length < 1e-6:
            self.length = 1e-6
            d = np.array([1.0, 0.0])
        self.t = d / self.length              # 单位切向量
        self.n = np.array([-self.t[1], self.t[0]])  # 单位法向量
        self.angle = math.degrees(math.atan2(d[1], d[0])) % 180.0
        self.mid = (self.p1 + self.p2) / 2.0
        self.id = Line._next_id
        Line._next_id += 1
        # Stage 4 填充：法向 + / - 两侧的白板比例
        self.white_pos = 0.0
        self.white_neg = 0.0
        # Stage 8 填充
        self.cls = "UNKNOWN"
        self.weight = W_UNKNOWN
        self.paired_with = None

    @property
    def white_adjacent(self):
        return max(self.white_pos, self.white_neg) > WHITE_ADJACENT_THRESH


def angle_diff_deg(a, b):
    """角度差，考虑 180° 等价"""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def pt(p):
    return (int(round(p[0])), int(round(p[1])))


# ---------------- Stage 1: LSD ----------------
def lsd_detect(gray):
    blur = gray if GAUSS_KSIZE is None else cv2.GaussianBlur(gray, GAUSS_KSIZE, 0)
    lsd = cv2.createLineSegmentDetector()
    segs = lsd.detect(blur)[0]
    if segs is None:
        return []
    return [Line(*seg) for seg in segs[:, 0, :]]


# ---------------- Stage 2: 共线合并 ----------------
def try_merge(li, lj):
    if angle_diff_deg(li.angle, lj.angle) > MERGE_ANGLE_THRESH_DEG:
        return None
    if abs(float((lj.p1 - li.p1) @ li.n)) > MERGE_NORMAL_DISTANCE_PX:
        return None
    si = np.array([0.0, li.length])                       # i 在自身切向的投影区间
    sj = np.array([(lj.p1 - li.p1) @ li.t,
                   (lj.p2 - li.p1) @ li.t])               # j 投影到 i 的切向
    gap = max(si.min() - sj.max(), sj.min() - si.max(), 0.0)
    if gap > MERGE_TANGENT_GAP_PX:
        return None
    pts = np.stack([li.p1, li.p2, lj.p1, lj.p2])
    s = (pts - li.p1) @ li.t
    order = np.argsort(s)
    return Line(pts[order[0]][0], pts[order[0]][1],
                pts[order[-1]][0], pts[order[-1]][1])


def merge_collinear(lines):
    lines = list(lines)
    changed = True
    while changed:
        changed = False
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                m = try_merge(lines[i], lines[j])
                if m is not None:
                    lines[i] = m
                    lines.pop(j)
                    changed = True
                    break
            if changed:
                break
    return lines


# ---------------- Stage 3: 白板 mask ----------------
def build_white_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, WHITE_V_MIN])
    upper = np.array([179, WHITE_S_MAX, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((MORPH_KSIZE, MORPH_KSIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# ---------------- Stage 4: 两侧白板邻接 ----------------
def side_white_ratio(line, white_mask, sign):
    """sign=+1/-1 表示法向哪一侧；从 GAP 开始宽 WIDTH 的条带内白色像素比例"""
    offsets = np.arange(SIDE_SAMPLE_GAP_PX,
                        SIDE_SAMPLE_GAP_PX + SIDE_SAMPLE_WIDTH_PX + 1e-6, 1.0)
    n_t = max(int(line.length), 2)
    ts = np.linspace(0.0, 1.0, n_t)
    base = line.p1[None, :] + ts[:, None] * (line.p2 - line.p1)[None, :]
    h, w = white_mask.shape
    total, white = 0, 0
    for off in offsets:
        pts = base + (sign * off) * line.n[None, :]
        xi = np.round(pts[:, 0]).astype(int)
        yi = np.round(pts[:, 1]).astype(int)
        valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        if not np.any(valid):
            continue
        total += int(np.count_nonzero(valid))
        white += int(np.count_nonzero(white_mask[yi[valid], xi[valid]]))
    return white / max(total, 1)


def _side_label(r):
    if r >= SIDE_WHITE_RATIO:
        return "W"
    if r <= SIDE_NONWHITE_RATIO:
        return "N"
    return "?"


def side_pattern(line):
    """返回 (正法向侧标签, 负法向侧标签)：W=白 N=非白 ?=不确定"""
    return (_side_label(line.white_pos), _side_label(line.white_neg))


# ---------------- Stage 5+6: 平行边对搜索（几何 + 白板角色） ----------------
def find_side_pair_candidates(lines):
    """
    返回:
      geo_pairs: 所有通过几何约束的边对（含未通过角色检查，用于统计实际法向距离）
      cands:     进一步满足 A(非白-非白) / B(非白-白) 角色的边对
    """
    geo_pairs, cands = [], []
    n = len(lines)
    for i in range(n):
        for j in range(i + 1, n):
            li, lj = lines[i], lines[j]
            adiff = angle_diff_deg(li.angle, lj.angle)
            if adiff > PARALLEL_ANGLE_THRESH_DEG:
                continue
            d_normal = abs(float((lj.mid - li.p1) @ li.n))
            if not (SIDE_MIN_WIDTH_PX <= d_normal <= SIDE_MAX_WIDTH_PX):
                continue
            si = np.array([0.0, li.length])
            sj = np.array([(lj.p1 - li.p1) @ li.t,
                           (lj.p2 - li.p1) @ li.t])
            overlap = max(0.0, min(si.max(), sj.max()) - max(si.min(), sj.min()))
            ov_ratio = overlap / min(li.length, lj.length)
            if ov_ratio < OVERLAP_RATIO_THRESH:
                continue
            geo_pairs.append(dict(i=li, j=lj, angle_diff=adiff,
                                  d_normal=d_normal, overlap=ov_ratio))
            pi, pj = side_pattern(li), side_pattern(lj)
            a = b = None
            if pi == ("N", "N") and sorted(pj) == ["N", "W"]:
                a, b = li, lj
            elif pj == ("N", "N") and sorted(pi) == ["N", "W"]:
                a, b = lj, li
            if a is None:
                continue
            cands.append(dict(a=a, b=b, angle_diff=adiff,
                              d_normal=d_normal, overlap=ov_ratio))
    return geo_pairs, cands


# ---------------- Stage 7: 相机方向验证 ----------------
def camera_direction_check(cand, vdir):
    """B 相对 A 是否朝画面中心方向。只写入可信度，不在这里删任何边。"""
    delta = cand["b"].mid - cand["a"].mid
    norm = float(np.hypot(delta[0], delta[1]))
    cand["camera_cos"] = float(delta @ vdir) / norm if norm > 1e-6 else 0.0
    cand["camera_ok"] = cand["camera_cos"] >= CAMERA_DIRECTION_COS_THRESH


# ---------------- Stage 8: 边对接受 + 分类 ----------------
def assign_pairs(cands):
    """按相机方向/重叠/距离排序后贪心接受，保证每条线只进一个 pair"""
    ranked = sorted(cands,
                    key=lambda c: (c["camera_ok"], c["camera_cos"],
                                   c["overlap"], -c["d_normal"]),
                    reverse=True)
    accepted, used = [], set()
    for c in ranked:
        if not c["camera_ok"]:
            continue
        if id(c["a"]) in used or id(c["b"]) in used:
            continue
        used.add(id(c["a"]))
        used.add(id(c["b"]))
        c["a"].cls = "TOP_SIDE"          # 内侧边：上表面-侧面
        c["b"].cls = "SIDE_BACKGROUND"   # 外侧边：侧面-白板
        c["a"].paired_with = c["b"].id
        c["b"].paired_with = c["a"].id
        accepted.append(c)
    return accepted, ranked


def classify_rest(lines):
    for ln in lines:
        if ln.cls == "UNKNOWN" and ln.white_adjacent:
            ln.cls = "TOP_BACKGROUND"
        ln.weight = CLASS_WEIGHT[ln.cls]


# ---------------- Edge Confidence Map ----------------
def build_confidence_map(shape, lines):
    conf = np.zeros(shape, dtype=np.float32)
    for ln in lines:
        for k, f in CONF_PROFILE:
            off = k * ln.n
            tmp = np.zeros(shape, dtype=np.float32)
            cv2.line(tmp, pt(ln.p1 + off), pt(ln.p2 + off), 1.0, 1)
            np.maximum(conf, ln.weight * f * tmp, out=conf)
    return conf


# ---------------- 可视化 ----------------
def dim_bgr(bgr, factor=0.5):
    return (bgr * factor).astype(np.uint8)


def dim_gray_bgr(gray, factor=0.45):
    return cv2.cvtColor((gray * factor).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def draw_lines(base, lines, color_fn, thickness=LINE_THICKNESS):
    vis = base.copy()
    for ln in lines:
        cv2.line(vis, pt(ln.p1), pt(ln.p2), color_fn(ln), thickness, cv2.LINE_AA)
    return vis


def draw_legend(vis):
    x0 = 10
    y = 10
    for cls, col in CLASS_COLOR.items():
        cv2.rectangle(vis, (x0, y), (x0 + 18, y + 14), col, -1)
        cv2.putText(vis, cls, (x0 + 26, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return vis


def draw_side_pairs(base, accepted):
    vis = base.copy()
    rng = np.random.default_rng(0)
    for idx, c in enumerate(accepted):
        color = tuple(int(x) for x in rng.integers(70, 256, 3))
        a, b = c["a"], c["b"]
        cv2.line(vis, pt(a.p1), pt(a.p2), color, 3, cv2.LINE_AA)
        cv2.line(vis, pt(b.p1), pt(b.p2), color, 3, cv2.LINE_AA)
        cv2.line(vis, pt(a.mid), pt(b.mid), color, 1, cv2.LINE_AA)
        cv2.circle(vis, pt(a.mid), 4, color, -1)
        cv2.circle(vis, pt(b.mid), 4, color, -1)
        m = (a.mid + b.mid) / 2.0
        text = "P%d d=%.1f a=%.1f ov=%.2f cos=%.2f" % (
            idx, c["d_normal"], c["angle_diff"], c["overlap"], c["camera_cos"])
        cv2.putText(vis, text, (int(m[0]) + 6, int(m[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def to_bgr3(u8_gray):
    return cv2.cvtColor(u8_gray, cv2.COLOR_GRAY2BGR)


# ---------------- 调试文本 ----------------
def build_debug_text(extra, lines, geo_pairs, ranked, accepted):
    out = list(extra)
    out.append("")
    out.append("==== 线段明细 ====")
    for ln in lines:
        out.append(
            "L%d: length=%.1f angle=%.1fdeg white_pos=%.2f white_neg=%.2f "
            "class=%s weight=%.1f paired_with=%s" % (
                ln.id, ln.length, ln.angle, ln.white_pos, ln.white_neg,
                ln.cls, ln.weight,
                ("L%d" % ln.paired_with) if ln.paired_with is not None else "-"))
    out.append("")
    out.append("==== 几何候选边对（平行+距离+重叠，未做角色过滤）====")
    for g in geo_pairs:
        out.append("G: L%d-L%d d_normal=%.1fpx angle_diff=%.1fdeg overlap=%.2f" % (
            g["i"].id, g["j"].id, g["d_normal"], g["angle_diff"], g["overlap"]))
    out.append("")
    out.append("==== 角色匹配边对 ====")
    acc_ids = {(c["a"].id, c["b"].id) for c in accepted}
    for c in ranked:
        tag = "ACCEPTED" if (c["a"].id, c["b"].id) in acc_ids else "rejected"
        out.append(
            "pair(A=L%d, B=L%d): angle_diff=%.1fdeg normal_distance=%.1fpx "
            "overlap=%.2f camera_direction_cos=%.2f [%s]" % (
                c["a"].id, c["b"].id, c["angle_diff"], c["d_normal"],
                c["overlap"], c["camera_cos"], tag))
    return "\n".join(out)


# ---------------- 主流程 ----------------
def main():
    bgr = cv2.imread(IMAGE_PATH)
    if bgr is None:
        raise FileNotFoundError("读不到图片: %s" % IMAGE_PATH)
    if ROI is not None:
        x, y, w, h = ROI
        bgr = bgr[y:y + h, x:x + w].copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    # Stage 1: LSD + 短线过滤
    raw_lines = lsd_detect(gray)
    filtered = [ln for ln in raw_lines if ln.length >= MIN_LINE_LENGTH]
    print("[Stage1] LSD 原始 %d 条 -> 长度过滤后 %d 条" % (len(raw_lines), len(filtered)))

    # Stage 2: 共线合并，重新编号 L0..Ln
    merged = merge_collinear(filtered)
    for i, ln in enumerate(merged):
        ln.id = i
    print("[Stage2] 合并后 %d 条" % len(merged))

    # Stage 3: 白板 mask
    white_mask = build_white_mask(bgr)
    print("[Stage3] 白板 mask 白色像素占比 %.1f%%" %
          (100.0 * np.count_nonzero(white_mask) / white_mask.size))

    # Stage 4: 两侧白板邻接
    for ln in merged:
        ln.white_pos = side_white_ratio(ln, white_mask, +1)
        ln.white_neg = side_white_ratio(ln, white_mask, -1)
    n_adj = sum(1 for ln in merged if ln.white_adjacent)
    print("[Stage4] 白板邻接 %d / %d 条" % (n_adj, len(merged)))

    # 相机方向向量 v = (image_center - object_center) 归一化
    image_center = np.array([W / 2.0, H / 2.0])
    if OBJECT_CENTER_UV is not None:
        object_center = np.array(OBJECT_CENTER_UV, dtype=np.float64)
    elif merged:
        object_center = np.mean([ln.mid for ln in merged], axis=0)
    else:
        object_center = image_center
    vdir = image_center - object_center
    vnorm = float(np.hypot(vdir[0], vdir[1]))
    vdir = vdir / vnorm if vnorm > 1e-6 else np.zeros(2)
    print("[Stage7] image_center=(%.0f,%.0f) object_center=(%.0f,%.0f) v=(%.2f,%.2f)" % (
        image_center[0], image_center[1], object_center[0], object_center[1],
        vdir[0], vdir[1]))

    # Stage 5+6: 边对搜索
    geo_pairs, cands = find_side_pair_candidates(merged)
    print("[Stage5] 几何候选边对 %d 个，其中角色匹配 %d 个" % (len(geo_pairs), len(cands)))

    # Stage 7: 相机方向验证
    for c in cands:
        camera_direction_check(c, vdir)
    accepted, ranked = assign_pairs(cands)
    print("[Stage7] 相机方向验证通过并接受 %d 对" % len(accepted))

    # Stage 8: 分类 + 权重
    classify_rest(merged)
    counter = Counter(ln.cls for ln in merged)
    print("[Stage8] 分类统计: " +
          ", ".join("%s=%d" % (k, counter.get(k, 0))
                    for k in ["TOP_BACKGROUND", "TOP_SIDE", "SIDE_BACKGROUND", "UNKNOWN"]))

    # Edge Confidence Map
    conf = build_confidence_map(gray.shape, merged)

    # ---------------- 输出 ----------------
    stem = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    out_dir = os.path.join(OUTPUT_ROOT, stem)
    os.makedirs(out_dir, exist_ok=True)

    save = lambda name, img: (cv2.imwrite(os.path.join(out_dir, name), img),
                              print("已保存", name))[0]

    base_dim = dim_gray_bgr(gray)
    save("00_original.png", bgr)
    save("01_lsd_raw.png",
         draw_lines(base_dim, raw_lines, lambda ln: (0, 255, 255), 1))
    save("02_lsd_length_filtered.png",
         draw_lines(base_dim, filtered, lambda ln: (0, 255, 255)))
    save("03_lsd_merged.png",
         draw_lines(base_dim, merged, lambda ln: (0, 255, 255)))

    save("04_white_mask.png", white_mask)

    save("05_white_adjacency.png",
         draw_lines(dim_bgr(bgr, 0.5), merged,
                    lambda ln: (0, 255, 0) if ln.white_adjacent else (120, 120, 120)))

    save("06_side_pairs.png", draw_side_pairs(dim_bgr(bgr, 0.55), accepted))

    vis_class_dim = draw_legend(draw_lines(base_dim, merged,
                                           lambda ln: CLASS_COLOR[ln.cls]))
    save("07_edge_classification.png", vis_class_dim)

    conf_u8 = np.zeros_like(gray)
    if conf.max() > 0:
        conf_u8 = (conf / conf.max() * 255.0).astype(np.uint8)
    save("08_edge_confidence_map.png", conf_u8)

    vis_class_orig = draw_legend(draw_lines(bgr, merged,
                                            lambda ln: CLASS_COLOR[ln.cls]))
    save("09_classification_on_original.png", vis_class_orig)

    panel = np.hstack([bgr, vis_class_orig, vis_class_dim, to_bgr3(conf_u8)])
    save("10_panel.png", panel)

    # 调试文本
    extra = [
        "image: %s  size: %dx%d  roi: %s" % (IMAGE_PATH, W, H, ROI),
        "LSD raw=%d filtered=%d merged=%d white_adjacent=%d" % (
            len(raw_lines), len(filtered), len(merged), n_adj),
        "params: min_len=%.0f merge(ang=%.1fdeg,norm=%.1fpx,gap=%.1fpx) "
        "white(v>=%d,s<=%d) band(gap=%.0f,w=%.0f,adj>%.2f) "
        "pair(ang=%.1fdeg,d=%.0f~%.0fpx,ov>%.2f) cos>=%.2f" % (
            MIN_LINE_LENGTH, MERGE_ANGLE_THRESH_DEG, MERGE_NORMAL_DISTANCE_PX,
            MERGE_TANGENT_GAP_PX, WHITE_V_MIN, WHITE_S_MAX,
            SIDE_SAMPLE_GAP_PX, SIDE_SAMPLE_WIDTH_PX, WHITE_ADJACENT_THRESH,
            PARALLEL_ANGLE_THRESH_DEG, SIDE_MIN_WIDTH_PX, SIDE_MAX_WIDTH_PX,
            OVERLAP_RATIO_THRESH, CAMERA_DIRECTION_COS_THRESH),
        "image_center=(%.0f,%.0f) object_center=(%.0f,%.0f)" % (
            image_center[0], image_center[1], object_center[0], object_center[1]),
        "classification: " + ", ".join("%s=%d" % (k, counter.get(k, 0))
                                       for k in CLASS_COLOR),
        "accepted_pairs=%d geo_candidates=%d role_candidates=%d" % (
            len(accepted), len(geo_pairs), len(cands)),
    ]
    debug_txt = build_debug_text(extra, merged, geo_pairs, ranked, accepted)
    txt_path = os.path.join(out_dir, "99_debug_info.txt")
    with open(txt_path, "w") as f:
        f.write(debug_txt + "\n")
    print("已保存 99_debug_info.txt")
    print(debug_txt)

    if SHOW_WINDOW:
        cv2.imshow("panel (press any key to exit)",
                   cv2.resize(panel, None, fx=PANEL_SCALE, fy=PANEL_SCALE))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
