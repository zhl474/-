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
IMAGE_PATH = "/home/zhl/SingleArmTetris/SingleArmTetris/src/tools/vision/5.png"
ROI = None                # 如 (x, y, w, h)；None = 整图

OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# --- Stage 1: LSD ---
GAUSS_KSIZE = (3, 3)      # LSD 前轻微模糊；None 则不模糊
MIN_LINE_LENGTH = 15.0    # 短线过滤阈值 px

# --- Stage 1.5: 线段支撑率（仅诊断，默认不过滤）---
# 实测教训：LSD 的幻觉斜线骑在真实梯度上（有支撑），而弱对比真边反而支撑率低
# （3x3 模糊后 105px 真边只有 0.62，被错杀）——支撑率区分不了真假，净收益为负。
# 结构性工具是 Stage 1.6 主方向筛选；支撑率只保留在 debug 文本里供观察。
SUPPORT_FILTER_ENABLED = False
SUPPORT_WINDOW_PX = 2       # 垂直于线方向的检查半宽 px
SUPPORT_GRAD_THRESH = 60.0  # 梯度幅值阈值（0-255 尺度），低于此不算真实边缘支撑
MIN_SUPPORT_RATIO = 0.65    # 过滤开启时：至少这么比例的线长要有梯度支撑

# --- Stage 1.6: 主方向筛选 ---
# 上表面轮廓只有两组正交方向（θ0 与 θ0+90°）。LSD 会把"窄侧面+模糊角点"
# 解释成一条有梯度支撑的斜线，但它不属于这两族——用局部主方向族过滤。
# 注意：LSD 只是候选线提议器，不是正确轮廓生成器。
DIR_FILTER_ENABLED = True
DIR_NEIGHBOR_RADIUS_PX = 160.0  # 局部估计半径（离线全图按局部估计，兼容多方块不同朝向）
DIR_TOP_K_PEAKS = 3             # 邻域内保留的主方向族数（方块各一族+网格一族）
DIR_ANGLE_THRESH_DEG = 7.0      # 线角度折到 mod 90 后与主方向族的最大偏差
DIR_HIST_BIN_DEG = 1.0          # 直方图角分辨率
DIR_HIST_SIGMA_DEG = 3.0        # 高斯圆滑宽度
DIR_PEAK_MIN_SEP_DEG = 15.0     # 相邻两个峰的最小角距

# --- Stage 2: 共线合并 ---
MERGE_ANGLE_THRESH_DEG = 2.0    # 角度差小于此值才可能共线
MERGE_NORMAL_DISTANCE_PX = 2.5  # 所在直线法向距离（4 端点对称检查，防楔形拉直）
MERGE_TANGENT_GAP_PX = 3.0      # 切向投影间隔：真实检测断裂一般 1~3px，白缝不接
MERGE_CONNECT_NORMAL_RATIO = 0.5  # 硬规则：连接位移法向占比上限，只允许沿切向接、禁止斜跨三维转折

# --- Stage 3: 白板 mask (HSV: V 高 + S 低) ---
WHITE_V_MIN = 180
WHITE_S_MAX = 60
MORPH_KSIZE = 3

# --- Stage 4: 两侧白板邻接 ---
SIDE_SAMPLE_GAP_PX = 1.0    # 离线段多少 px 后开始采样（避开边缘抗锯齿）
SIDE_SAMPLE_WIDTH_PX = 4.0  # 条带宽度 3~5px；逐偏移线取中位数，抗稀释
WHITE_ADJACENT_THRESH = 0.7   # max(两侧比例) 超过 -> 白板邻接
SIDE_WHITE_RATIO = 0.65     # 单侧比例 >= 此值 -> 该侧视为白
SIDE_NONWHITE_RATIO = 0.3   # 单侧比例 <= 此值 -> 该侧视为非白

# --- Stage 5: 侧面平行边对（四边形模型）---
PARALLEL_ANGLE_THRESH_DEG = 4.0  # 考虑 180° 等价
SIDE_MIN_WIDTH_PX = 3.0
SIDE_MAX_WIDTH_PX = 25.0
OVERLAP_RATIO_THRESH = 0.5
# 侧面是四边形：除 A/B 平行边外，两端还有厚度方向连接边 P1-Q1、P2-Q2
PAIR_DELTA_ANGLE_MAX_DEG = 20.0    # 两端厚度向量 Δ1=Q1-P1 与 Δ2=Q2-P2 最大夹角（禁止交叉端点对应）
PAIR_DELTA_LEN_DIFF_PX = 6.0       # 两端厚度投影长度的最大差
CONNECTOR_SEARCH_RADIUS_PX = 12.0  # 端部连接边搜索半径
CONNECTOR_ANGLE_MAX_DEG = 25.0     # 连接边与 Δ 方向的最大夹角
CONNECTOR_MAX_LEN_PX = 45.0        # 厚度连接边只可能是短线

# --- Stage 7: 相机方向验证 ---
OBJECT_CENTER_UV = None            # 如 (640, 360)；None = 用线段中点均值估计
CAMERA_DIRECTION_COS_THRESH = 0.3  # 宽松阈值，仅作 pair 验证

# --- Stage 8: 分类权重 ---
W_TOP_BACKGROUND = 3.0
W_TOP_SIDE = 3.0
W_UNKNOWN = 0.7
W_SIDE_BACKGROUND = 0.1

# --- 可视化 ---
LINE_THICKNESS = 1      # 统一 1px，所见即所得；仅 06 边对图用 2px 突出配对关系
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
        # 像素证据层：合并只做逻辑归组，真实检测到的片段单独保留；
        # 分类/配对用合并线判断，绘制与置信图只画 fragments，绝不凭空补线
        self.fragments = [(self.p1.copy(), self.p2.copy())]
        # Stage 1.5 填充：线段支撑率（沿线的真实梯度覆盖比例）
        self.support_ratio = 1.0
        # Stage 1.6 填充：与邻域主方向族的最小偏差（deg），-1 表示未计算
        self.dir_dev = -1.0
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
def lsd_detect(blur):
    lsd = cv2.createLineSegmentDetector()
    segs = lsd.detect(blur)[0]
    if segs is None:
        return []
    return [Line(*seg) for seg in segs[:, 0, :]]


# ---------------- Stage 1.5: 线段支撑率验证 ----------------
def compute_support_ratios(lines, blur):
    """沿线段采样，检查垂直窗口内是否存在真实梯度。

    LSD 的直线检测结果偶尔回把跨角/亮暗渐变区域拟合成一条不存在的斜线，
    这种幻觉线只有部分长度落在真实边缘上；真实边的支撑率接近 1.0。
    """
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    h, w = blur.shape
    offsets = np.arange(-SUPPORT_WINDOW_PX, SUPPORT_WINDOW_PX + 1)
    for ln in lines:
        n_t = max(int(ln.length), 2)
        ts = np.linspace(0.0, 1.0, n_t)
        base = ln.p1[None, :] + ts[:, None] * (ln.p2 - ln.p1)[None, :]
        all_vals = np.zeros((len(offsets), n_t), dtype=np.float32)
        for idx, k in enumerate(offsets):
            pts = base + (float(k) * ln.n)[None, :]
            xi = np.round(pts[:, 0]).astype(int)
            yi = np.round(pts[:, 1]).astype(int)
            valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            if np.any(valid):
                all_vals[idx, valid] = mag[yi[valid], xi[valid]]
        ln.support_ratio = float(np.mean(all_vals.max(axis=0) >= SUPPORT_GRAD_THRESH))


# ---------------- Stage 1.6: 主方向筛选 ----------------
def _fold90(deg):
    return deg % 90.0


def _circ_dist90(a, b):
    d = abs(a - b) % 90.0
    return min(d, 90.0 - d)


def direction_filter(lines):
    """按邻域主方向族过滤。

    对每条线，统计半径 DIR_NEIGHBOR_RADIUS_PX 内所有线的
    (角度 mod 90) 长度加权圆直方图，取相互 separated 的 top-K 峰作为
    该邻域的主方向族；线与任一族偏差 <= DIR_ANGLE_THRESH_DEG 才保留。
    上表面是正交直线轮廓（θ0 / θ0+90° 同折一角），跨角假斜线不属于任何族。
    """
    if not lines:
        return []
    bins = int(round(90.0 / DIR_HIST_BIN_DEG))
    sigma = DIR_HIST_SIGMA_DEG / DIR_HIST_BIN_DEG
    spread = int(round(3.0 * sigma))
    mids = np.array([ln.mid for ln in lines])
    folds = np.array([_fold90(ln.angle) for ln in lines])
    lens = np.array([ln.length for ln in lines])
    gauss = np.exp(-0.5 * (np.arange(-spread, spread + 1) / sigma) ** 2)
    kept = []
    for idx, ln in enumerate(lines):
        dist = np.hypot(mids[:, 0] - ln.mid[0], mids[:, 1] - ln.mid[1])
        neighbor_ids = np.nonzero(dist <= DIR_NEIGHBOR_RADIUS_PX)[0]
        hist = np.zeros(bins)
        for j in neighbor_ids:
            center = int(round(folds[j] / DIR_HIST_BIN_DEG))
            for db in range(-spread, spread + 1):
                hist[(center + db) % bins] += lens[j] * gauss[db + spread]
        # 取 top-K 个互相 separated 的峰
        peaks = []
        for b in np.argsort(hist)[::-1]:
            if len(peaks) >= DIR_TOP_K_PEAKS:
                break
            ang = b * DIR_HIST_BIN_DEG
            if all(_circ_dist90(ang, p) >= DIR_PEAK_MIN_SEP_DEG for p in peaks):
                peaks.append(ang)
        dev = min((_circ_dist90(folds[idx], p) for p in peaks), default=90.0)
        ln.dir_dev = float(dev)
        if dev <= DIR_ANGLE_THRESH_DEG:
            kept.append(ln)
    return kept


# ---------------- Stage 2: 共线合并 ----------------
def try_merge(li, lj):
    if angle_diff_deg(li.angle, lj.angle) > MERGE_ANGLE_THRESH_DEG:
        return None
    # 对称法向检查：两条线共 4 个端点到对方所在直线的最大距离都要小，
    # 防止只查单端点时有夹角的"楔形"被拉直成一条线
    d = max(
        abs(float((lj.p1 - li.p1) @ li.n)),
        abs(float((lj.p2 - li.p1) @ li.n)),
        abs(float((li.p1 - lj.p1) @ lj.n)),
        abs(float((li.p2 - lj.p1) @ lj.n)),
    )
    if d > MERGE_NORMAL_DISTANCE_PX:
        return None
    si = np.array([0.0, li.length])                       # i 在自身切向的投影区间
    sj = np.array([(lj.p1 - li.p1) @ li.t,
                   (lj.p2 - li.p1) @ li.t])               # j 投影到 i 的切向
    gap = max(si.min() - sj.max(), sj.min() - si.max(), 0.0)
    if gap > MERGE_TANGENT_GAP_PX:
        return None
    # 硬规则：两段不重叠时，最近端点的连接位移必须以切向为主。
    # 需要斜着才能接上的两段（跨三维转折、跨角）绝对不合并。
    if gap > 0.0:
        best_g, best_len = None, None
        for pi_ in (li.p1, li.p2):
            for pj_ in (lj.p1, lj.p2):
                g = pj_ - pi_
                glen = float(np.hypot(g[0], g[1]))
                if best_len is None or glen < best_len:
                    best_g, best_len = g, glen
        if best_len is not None and best_len > 1e-6 and \
                abs(float(best_g @ li.n)) / best_len > MERGE_CONNECT_NORMAL_RATIO:
            return None
    pts = np.stack([li.p1, li.p2, lj.p1, lj.p2])
    s = (pts - li.p1) @ li.t
    order = np.argsort(s)
    merged = Line(pts[order[0]][0], pts[order[0]][1],
                  pts[order[-1]][0], pts[order[-1]][1])
    merged.fragments = li.fragments + lj.fragments
    return merged


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
    ratios = []
    for off in offsets:
        pts = base + (sign * off) * line.n[None, :]
        xi = np.round(pts[:, 0]).astype(int)
        yi = np.round(pts[:, 1]).astype(int)
        valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        if not np.any(valid):
            continue
        ratios.append(float(np.count_nonzero(white_mask[yi[valid], xi[valid]]))
                      / int(np.count_nonzero(valid)))
    # 逐偏移线取中位数：外圈偏移线蹭到别的结构只影响个别偏移线，不再稀释整体
    return float(np.median(ratios)) if ratios else 0.0


def _side_label(r):
    if r >= SIDE_WHITE_RATIO:
        return "W"
    if r <= SIDE_NONWHITE_RATIO:
        return "N"
    return "?"


def side_pattern(line):
    """返回 (正法向侧标签, 负法向侧标签)：W=白 N=非白 ?=不确定"""
    return (_side_label(line.white_pos), _side_label(line.white_neg))


# ---------------- Stage 5+6: 平行边对搜索（几何 + 白板角色 + 四边形拓扑） ----------------
def find_side_pair_candidates(lines, bonus_pool=None):
    """
    侧面按四边形建模：
        P1 ──── A ──── P2        A = 上表面-侧面棱线
        │             │          B = 侧面-白板边
        Q1 ──── B ──── Q2        P1-Q1 / P2-Q2 = 厚度方向连接边

    bonus_pool: 连接边搜索池（厚度短边不属于主方向族，从方向筛选前的线集里找）

    返回:
      geo_pairs: 所有通过几何约束的边对（含两侧模式、四边形与连接边结果，用于统计）
      cands:     通过模式 + 四边形约束的边对
      near_miss: 几何三关（平行/法向距离/切向重叠）只过了两关的边对，便于诊断
    """
    if bonus_pool is None:
        bonus_pool = lines
    geo_pairs, cands, near_miss = [], [], []
    n = len(lines)
    for i in range(n):
        for j in range(i + 1, n):
            li, lj = lines[i], lines[j]
            adiff = angle_diff_deg(li.angle, lj.angle)
            ang_ok = adiff <= PARALLEL_ANGLE_THRESH_DEG
            d_normal = abs(float((lj.mid - li.p1) @ li.n))
            dist_ok = SIDE_MIN_WIDTH_PX <= d_normal <= SIDE_MAX_WIDTH_PX
            si = np.array([0.0, li.length])
            sj = np.array([(lj.p1 - li.p1) @ li.t,
                           (lj.p2 - li.p1) @ li.t])
            overlap = max(0.0, min(si.max(), sj.max()) - max(si.min(), sj.min()))
            ov_ratio = overlap / min(li.length, lj.length)
            ov_ok = ov_ratio >= OVERLAP_RATIO_THRESH
            if not (ang_ok and dist_ok and ov_ok):
                # 近失误：三关过了两关才值得看
                if ang_ok + dist_ok + ov_ok == 2:
                    failed = "angle" if not ang_ok else (
                        "distance" if not dist_ok else "overlap")
                    near_miss.append(dict(i=li, j=lj, angle_diff=adiff,
                                          d_normal=d_normal, overlap=ov_ratio,
                                          failed=failed))
                continue
            pi, pj = side_pattern(li), side_pattern(lj)
            a = b = None
            if pi == ("N", "N") and sorted(pj) == ["N", "W"]:
                a, b = li, lj
            elif pj == ("N", "N") and sorted(pi) == ["N", "W"]:
                a, b = lj, li

            # 四边形约束：B 端点统一到与 A 同向后，Δ1=Q1-P1 与 Δ2=Q2-P2 应近似一致；
            # 交叉对应（P1↔Q2）会产生方向相反 / 长度悬殊的 Δ，直接淘汰
            quad_ok, delta_angle, delta_len_diff, d1, d2 = False, 180.0, 1e9, None, None
            if a is not None:
                bp1, bp2 = (b.p1, b.p2) if float(a.t @ b.t) > 0 else (b.p2, b.p1)
                d1 = bp1 - a.p1
                d2 = bp2 - a.p2
                n1 = float(np.hypot(d1[0], d1[1]))
                n2 = float(np.hypot(d2[0], d2[1]))
                delta_len_diff = abs(n1 - n2)
                if n1 > 1e-6 and n2 > 1e-6:
                    cos12 = float(np.clip(d1 @ d2 / (n1 * n2), -1.0, 1.0))
                    delta_angle = math.degrees(math.acos(cos12))
                    quad_ok = (delta_angle <= PAIR_DELTA_ANGLE_MAX_DEG and
                               delta_len_diff <= PAIR_DELTA_LEN_DIFF_PX)

            # 连接边 bonus：A 两端附近存在与 Δ 方向一致的短 LSD 线段
            # （即真实检测到的 P1-Q1 / P2-Q2 厚度边），只加分不否决
            connectors = 0
            if a is not None:
                for ep, dv in ((a.p1, d1), (a.p2, d2)):
                    if dv is None or float(np.hypot(dv[0], dv[1])) < 1e-6:
                        continue
                    dv_angle = math.degrees(math.atan2(dv[1], dv[0])) % 180.0
                    for ln2 in bonus_pool:
                        if ln2.length > CONNECTOR_MAX_LEN_PX:
                            continue
                        if float(np.hypot(ln2.mid[0] - ep[0],
                                          ln2.mid[1] - ep[1])) > CONNECTOR_SEARCH_RADIUS_PX:
                            continue
                        if angle_diff_deg(ln2.angle, dv_angle) <= CONNECTOR_ANGLE_MAX_DEG:
                            connectors += 1
                            break

            geo_pairs.append(dict(i=li, j=lj, angle_diff=adiff,
                                  d_normal=d_normal, overlap=ov_ratio,
                                  pat_i="".join(pi), pat_j="".join(pj),
                                  role_ok=a is not None, quad_ok=quad_ok,
                                  connectors=connectors))
            if a is None or not quad_ok:
                continue
            cands.append(dict(a=a, b=b, angle_diff=adiff,
                              d_normal=d_normal, overlap=ov_ratio,
                              delta_angle=delta_angle,
                              delta_len_diff=delta_len_diff,
                              connectors=connectors))
    return geo_pairs, cands, near_miss


# ---------------- Stage 7: 相机方向验证 ----------------
def camera_direction_check(cand, image_center, vdir=None):
    """B 相对 A 是否朝画面中心方向。只写入可信度，不在这里删任何边。

    vdir=None 时按每个候选对自身中心局部计算 v：多目标全图场景下，
    全局 object_center 估计会偏，局部化可避免误杀边对。"""
    if vdir is None:
        pair_center = (cand["a"].mid + cand["b"].mid) / 2.0
        v = image_center - pair_center
        vnorm = float(np.hypot(v[0], v[1]))
        v = v / vnorm if vnorm > 1e-6 else np.zeros(2)
    else:
        v = vdir
    delta = cand["b"].mid - cand["a"].mid
    norm = float(np.hypot(delta[0], delta[1]))
    cand["camera_cos"] = float(delta @ v) / norm if norm > 1e-6 else 0.0
    cand["camera_ok"] = cand["camera_cos"] >= CAMERA_DIRECTION_COS_THRESH


# ---------------- Stage 8: 边对接受 + 分类 ----------------
def assign_pairs(cands):
    """按 相机方向/连接边/重叠/距离 排序后贪心接受，保证每条线只进一个 pair"""
    ranked = sorted(cands,
                    key=lambda c: (c["camera_ok"], c["connectors"],
                                   c["camera_cos"], c["overlap"], -c["d_normal"]),
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
        if ln.cls == "UNKNOWN":
            # 上表面-白板边必须恰好一侧白、一侧非白。
            # 两侧全白 (W,W) 是白板内部的线（网格/幻觉斜线），绝不能给高权重
            if sorted(side_pattern(ln)) == ["N", "W"]:
                ln.cls = "TOP_BACKGROUND"
        ln.weight = CLASS_WEIGHT[ln.cls]


# ---------------- Edge Confidence Map ----------------
def build_confidence_map(shape, lines):
    """只画真实检测到的片段：合并组的逻辑 span 不产生任何像素证据"""
    conf = np.zeros(shape, dtype=np.float32)
    for ln in lines:
        for k, f in CONF_PROFILE:
            off = k * ln.n
            tmp = np.zeros(shape, dtype=np.float32)
            for f1, f2 in ln.fragments:
                cv2.line(tmp, pt(f1 + off), pt(f2 + off), 1.0, 1)
            np.maximum(conf, ln.weight * f * tmp, out=conf)
    return conf


# ---------------- 可视化 ----------------
def dim_bgr(bgr, factor=0.5):
    return (bgr * factor).astype(np.uint8)


def dim_gray_bgr(gray, factor=0.45):
    return cv2.cvtColor((gray * factor).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def draw_lines(base, lines, color_fn, thickness=LINE_THICKNESS, label=False):
    """只画真实检测片段（合并组的 fragments），不补画合并 span 中间的空隙。
    label=True 时在长度 >=25px 的线中点旁标 L 编号，方便和 debug 文本对照。"""
    vis = base.copy()
    for ln in lines:
        color = color_fn(ln)
        for f1, f2 in ln.fragments:
            cv2.line(vis, pt(f1), pt(f2), color, thickness, cv2.LINE_AA)
        if label and ln.length >= 25.0:
            lp = ln.mid + 6.0 * ln.n
            cv2.putText(vis, "L%d" % ln.id, (int(lp[0]), int(lp[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
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
        for f1, f2 in a.fragments:
            cv2.line(vis, pt(f1), pt(f2), color, 2, cv2.LINE_AA)
        for f1, f2 in b.fragments:
            cv2.line(vis, pt(f1), pt(f2), color, 2, cv2.LINE_AA)
        cv2.line(vis, pt(a.mid), pt(b.mid), color, 1, cv2.LINE_AA)
        cv2.circle(vis, pt(a.mid), 4, color, -1)
        cv2.circle(vis, pt(b.mid), 4, color, -1)
        m = (a.mid + b.mid) / 2.0
        text = "P%d d=%.1f ov=%.2f cos=%.2f conn=%d" % (
            idx, c["d_normal"], c["overlap"], c["camera_cos"], c["connectors"])
        cv2.putText(vis, text, (int(m[0]) + 6, int(m[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def to_bgr3(u8_gray):
    return cv2.cvtColor(u8_gray, cv2.COLOR_GRAY2BGR)


# ---------------- 调试文本 ----------------
def build_debug_text(extra, lines, geo_pairs, ranked, accepted, near_miss):
    out = list(extra)
    out.append("")
    out.append("==== 线段明细 ====")
    for ln in lines:
        out.append(
            "L%d: length=%.1f angle=%.1fdeg white_pos=%.2f white_neg=%.2f support=%.2f dirdev=%.0f frags=%d "
            "class=%s weight=%.1f paired_with=%s" % (
                ln.id, ln.length, ln.angle, ln.white_pos, ln.white_neg,
                ln.support_ratio, ln.dir_dev, len(ln.fragments), ln.cls, ln.weight,
                ("L%d" % ln.paired_with) if ln.paired_with is not None else "-"))
    out.append("")
    out.append("==== 几何候选边对（含两侧模式/四边形/连接边结果）====")
    for g in geo_pairs:
        out.append(
            "G: L%d-L%d d_normal=%.1fpx angle_diff=%.1fdeg overlap=%.2f "
            "pat=%s/%s role=%s quad=%s conn=%d" % (
                g["i"].id, g["j"].id, g["d_normal"], g["angle_diff"],
                g["overlap"], g["pat_i"], g["pat_j"],
                "ok" if g["role_ok"] else "no",
                "ok" if g["quad_ok"] else "no", g["connectors"]))
    out.append("")
    out.append("==== 近失误边对（几何三关只过两关）====")
    for g in near_miss:
        out.append(
            "N: L%d-L%d failed=%s angle_diff=%.1fdeg d_normal=%.1fpx "
            "overlap=%.2f" % (
                g["i"].id, g["j"].id, g["failed"], g["angle_diff"],
                g["d_normal"], g["overlap"]))
    out.append("")
    out.append("==== 角色匹配边对 ====")
    acc_ids = {(c["a"].id, c["b"].id) for c in accepted}
    for c in ranked:
        tag = "ACCEPTED" if (c["a"].id, c["b"].id) in acc_ids else "rejected"
        out.append(
            "pair(A=L%d, B=L%d): angle_diff=%.1fdeg normal_distance=%.1fpx "
            "overlap=%.2f delta_angle=%.1fdeg delta_len_diff=%.1fpx conn=%d "
            "camera_direction_cos=%.2f [%s]" % (
                c["a"].id, c["b"].id, c["angle_diff"], c["d_normal"],
                c["overlap"], c["delta_angle"], c["delta_len_diff"],
                c["connectors"], c["camera_cos"], tag))
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

    # Stage 1: LSD + 长度过滤（支撑率只做诊断，不再错杀弱真边）
    blur = gray if GAUSS_KSIZE is None else cv2.GaussianBlur(gray, GAUSS_KSIZE, 0)
    raw_lines = lsd_detect(blur)
    compute_support_ratios(raw_lines, blur)
    long_lines = [ln for ln in raw_lines if ln.length >= MIN_LINE_LENGTH]
    if SUPPORT_FILTER_ENABLED:
        filtered = [ln for ln in long_lines if ln.support_ratio >= MIN_SUPPORT_RATIO]
    else:
        filtered = list(long_lines)
    print("[Stage1] LSD 原始 %d 条 -> 长度过滤后 %d 条%s" % (
        len(raw_lines), len(filtered),
        "（支撑过滤又淘汰 %d 条）" % (len(long_lines) - len(filtered))
        if SUPPORT_FILTER_ENABLED else ""))

    # Stage 1.6: 主方向筛选（先方向筛选，再做合并/分类——伪线在早期就被删掉）
    bonus_pool = list(filtered)  # 连接边(厚度短边)不属于主方向族，bonus 从筛选前池子找
    if DIR_FILTER_ENABLED:
        n_pre = len(filtered)
        filtered = direction_filter(filtered)
        print("[Stage1.6] 主方向筛选: %d -> %d 条（淘汰 %d 条非主方向斜线）" % (
            n_pre, len(filtered), n_pre - len(filtered)))

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

    # 相机方向：OBJECT_CENTER_UV 手动指定时用全局 v；否则按每个候选对自身中心局部计算
    image_center = np.array([W / 2.0, H / 2.0])
    vdir = None
    if OBJECT_CENTER_UV is not None:
        oc = np.array(OBJECT_CENTER_UV, dtype=np.float64)
        vv = image_center - oc
        nv = float(np.hypot(vv[0], vv[1]))
        vdir = vv / nv if nv > 1e-6 else None
        print("[Stage7] image_center=(%.0f,%.0f) 手动 object_center=%s" % (
            image_center[0], image_center[1], OBJECT_CENTER_UV))
    else:
        print("[Stage7] image_center=(%.0f,%.0f) v 按每个候选对自身中心局部计算" % (
            image_center[0], image_center[1]))

    # Stage 5+6: 边对搜索
    geo_pairs, cands, near_miss = find_side_pair_candidates(merged, bonus_pool=bonus_pool)
    n_role = sum(1 for g in geo_pairs if g["role_ok"])
    print("[Stage5] 几何候选 %d 个，模式匹配 %d 个，四边形约束后 %d 个（淘汰 %d），近失误 %d 个" % (
        len(geo_pairs), n_role, len(cands), n_role - len(cands), len(near_miss)))

    # Stage 7: 相机方向验证
    for c in cands:
        camera_direction_check(c, image_center, vdir)
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
         draw_lines(base_dim, long_lines, lambda ln: (0, 255, 255)))
    if SUPPORT_FILTER_ENABLED:
        save("02b_lsd_support_filtered.png",
             draw_lines(base_dim, filtered, lambda ln: (0, 255, 255)))
    if DIR_FILTER_ENABLED:
        save("02c_direction_filtered.png",
             draw_lines(base_dim, filtered, lambda ln: (0, 255, 255), label=True))
    save("03_lsd_merged.png",
         draw_lines(base_dim, merged, lambda ln: (0, 255, 255), label=True))

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
                                            lambda ln: CLASS_COLOR[ln.cls], label=True))
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
        "image_center=(%.0f,%.0f) v_mode=%s" % (
            image_center[0], image_center[1],
            "manual" if vdir is not None else "per-pair-local"),
        "classification: " + ", ".join("%s=%d" % (k, counter.get(k, 0))
                                       for k in CLASS_COLOR),
        "accepted_pairs=%d geo_candidates=%d role_candidates=%d" % (
            len(accepted), len(geo_pairs), len(cands)),
    ]
    debug_txt = build_debug_text(extra, merged, geo_pairs, ranked, accepted,
                                 near_miss)
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
