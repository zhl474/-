"""ArUco 诊断实验纯函数核心：检测、中心计算、绘图与统计。

本模块不依赖 ROS，也不依赖任何正式代码，可离线单元测试。
高低位统一使用同一套检测函数，参考像素为每帧图像几何中心。
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

_EPSILON = 1e-9


@lru_cache(maxsize=8)
def _load_chinese_font(font_size: int):
    """缓存中文字体，避免伺服录像逐帧重复加载字体文件。"""
    from PIL import ImageFont

    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for font_path in candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, int(font_size))
    return ImageFont.load_default()


def put_chinese_text(image, text, org, color, font_size=20):
    """在 OpenCV 图像上绘制中文；Pillow 不可用时退回 OpenCV。"""
    try:
        from PIL import Image, ImageDraw

        font = _load_chinese_font(int(font_size))
        if image.ndim == 2:
            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)
            draw.text(org, str(text), font=font, fill=int(max(color)))
            image[:] = np.asarray(pil_image)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            draw = ImageDraw.Draw(pil_image)
            blue, green, red = color
            draw.text(org, str(text), font=font, fill=(red, green, blue))
            image[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    except Exception:  # noqa: BLE001
        cv2.putText(
            image, str(text), org, cv2.FONT_HERSHEY_SIMPLEX,
            0.6, color, 2, cv2.LINE_AA,
        )
    return image


@dataclass
class OffsetResult:
    """与正式视觉伺服响应协议对齐的检测结果。"""

    found: bool = False
    px: float = float("nan")
    py: float = float("nan")
    dx_px: float = float("nan")
    dy_px: float = float("nan")
    center_uv: Tuple[float, float] = (float("nan"), float("nan"))
    rough_center_uv: Tuple[float, float] = (float("nan"), float("nan"))
    refine_delta_uv: Tuple[float, float] = (float("nan"), float("nan"))
    refine_shift_px: float = float("nan")
    center_contrast: float = float("nan")
    corners: Optional[np.ndarray] = None
    rejected_corners: Optional[list] = None
    message: str = ""


@dataclass(frozen=True)
class CenterRefineConfig:
    """中央棋盘角点亚像素精定位参数。"""

    window_ratio: float = 0.025
    min_window_px: int = 3
    max_window_px: int = 15
    max_shift_cell_ratio: float = 0.12
    min_contrast: float = 5.0
    max_iterations: int = 50
    epsilon_px: float = 0.001
    sample_offset_cell_ratio: float = 0.5
    sample_radius_cell_ratio: float = 0.10


@dataclass
class CenterRefinementResult:
    """中央棋盘角点精定位的纯函数结果。"""

    found: bool = False
    center_uv: Tuple[float, float] = (float("nan"), float("nan"))
    delta_uv: Tuple[float, float] = (float("nan"), float("nan"))
    shift_px: float = float("nan")
    contrast: float = float("nan")
    window_px: int = 0
    message: str = ""


def image_center_uv(width: int, height: int) -> Tuple[float, float]:
    """正式低位逻辑的参考像素：每帧图像几何中心。"""
    return float(width) / 2.0, float(height) / 2.0


def create_aruco_detector(
    dict_name: str = "DICT_6X6_50",
    corner_refine: bool = True,
    min_perimeter_rate: Optional[float] = None,
    adaptive_thresh_max: int = 63,
):
    """创建 ArUco 检测器。

    corner_refine=True 时启用亚像素角点细化；
    min_perimeter_rate 可降低默认的标记最小周长比例（默认 0.03），
    用于探针对比诊断边缘或远景小标记。

    自适应阈值窗口默认 3/63/10：实测高位约 227px、低位约 447px 的标记，
    默认 OpenCV 最大值 23px 会导致大标记“找到外框但解码失败”
    （ids=None 且 rejected 非空），63px 时两种尺度均可正常识别。
    """
    if not hasattr(cv2.aruco, dict_name):
        raise ValueError(f"未知 ArUco 字典: {dict_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = (
        cv2.aruco.CORNER_REFINE_SUBPIX if corner_refine else cv2.aruco.CORNER_REFINE_NONE
    )
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = int(adaptive_thresh_max)
    params.adaptiveThreshWinSizeStep = 10
    if min_perimeter_rate is not None:
        params.minMarkerPerimeterRate = float(min_perimeter_rate)
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_all_markers(
    image: np.ndarray,
    dict_names: Sequence[str],
    corner_refine: bool = True,
    min_perimeter_rate: Optional[float] = None,
) -> dict:
    """多字典全量检测：返回每个字典检出的标记 ID、角点像素尺寸与角点。

    不做任何 ID 过滤，用于判断“标记是否在画面里、用哪个字典能识别”。
    rejected_count / rejected_corners 表示“找到了四边形但内部编码解码失败”
    的候选，用于区分“画面里没有标记”和“有标记但没解出来”。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    results = {}
    for dict_name in dict_names:
        detector = create_aruco_detector(
            dict_name, corner_refine=corner_refine, min_perimeter_rate=min_perimeter_rate
        )
        corners, ids, rejected = detector.detectMarkers(gray)
        ids_list = [] if ids is None else [int(value) for value in ids.ravel()]
        sizes = []
        corner_lists = []
        if corners is not None:
            for detected in corners:
                quad = np.asarray(detected, dtype=float).reshape(4, 2)
                sizes.append(
                    float(np.mean([
                        np.linalg.norm(quad[1] - quad[0]),
                        np.linalg.norm(quad[2] - quad[1]),
                    ]))
                )
                corner_lists.append(quad.round(3).tolist())
        rejected_lists = []
        if rejected is not None:
            for candidate in rejected:
                rejected_lists.append(
                    np.asarray(candidate, dtype=float).reshape(4, 2).round(3).tolist()
                )
        results[dict_name] = {
            "ids": ids_list,
            "sizes_px": sizes,
            "corners": corner_lists,
            "rejected_count": len(rejected_lists),
            "rejected_corners": rejected_lists,
        }
    return results


def diagonal_intersection(corners: np.ndarray) -> Tuple[float, float]:
    """OpenCV 顺序 [p0,p1,p2,p3] 下，对角线 p0-p2 与 p1-p3 的直线交点。

    拒绝近平行、退化或交点落在对角线段之外的四边形。
    """
    points = np.asarray(corners, dtype=float)
    if points.shape != (4, 2) or not np.all(np.isfinite(points)):
        raise ValueError("角点必须是 4 个有限二维点")
    p0, p1, p2, p3 = points
    d1 = p2 - p0
    d2 = p3 - p1
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < _EPSILON:
        raise ValueError("四边形对角线近平行，无法求交点")
    t = ((p1[0] - p0[0]) * d2[1] - (p1[1] - p0[1]) * d2[0]) / cross
    s = ((p1[0] - p0[0]) * d1[1] - (p1[1] - p0[1]) * d1[0]) / cross
    if t < -_EPSILON or t > 1.0 + _EPSILON or s < -_EPSILON or s > 1.0 + _EPSILON:
        raise ValueError("对角线交点越出四边形")
    center = p0 + t * d1
    if not np.all(np.isfinite(center)):
        raise ValueError("对角线交点非有限")
    return float(center[0]), float(center[1])


def point_in_convex_quad(point, corners: np.ndarray) -> bool:
    """用叉积符号一致判断点是否在凸四边形内部。"""
    p = np.asarray(point, dtype=float)
    quad = np.asarray(corners, dtype=float)
    if quad.shape != (4, 2):
        return False
    signs = []
    for index in range(4):
        edge = quad[(index + 1) % 4] - quad[index]
        to_point = p - quad[index]
        signs.append(edge[0] * to_point[1] - edge[1] * to_point[0])
    return all(value >= -_EPSILON for value in signs) or all(value <= _EPSILON for value in signs)


def _central_checkerboard_pattern(dictionary, marker_id: int) -> np.ndarray:
    """读取字典中标记中央 2x2 码格，并确认它构成棋盘角点。"""
    marker_size = int(dictionary.markerSize)
    total_cells = marker_size + 2
    if total_cells % 2 != 0:
        raise ValueError(f"标记总码格数 {total_cells} 不是偶数，没有唯一中央交叉点")
    try:
        canonical = cv2.aruco.generateImageMarker(
            dictionary, int(marker_id), total_cells, borderBits=1
        )
    except cv2.error as exc:
        raise ValueError(f"无法生成 ID={marker_id} 的标准码图: {exc}") from exc
    middle = total_cells // 2
    pattern = np.asarray(
        canonical[middle - 1:middle + 1, middle - 1:middle + 1], dtype=np.uint8
    )
    is_checkerboard = (
        pattern.shape == (2, 2)
        and pattern[0, 0] == pattern[1, 1]
        and pattern[0, 1] == pattern[1, 0]
        and pattern[0, 0] != pattern[0, 1]
    )
    if not is_checkerboard:
        raise ValueError(f"ID={marker_id} 的中央 2x2 码格不是黑白棋盘角点")
    return pattern > 127


def _circular_patch_mean(gray: np.ndarray, center: np.ndarray, radius: float) -> float:
    """计算浮点中心附近圆形小区域均值，越出图像时拒绝。"""
    height, width = gray.shape[:2]
    cx, cy = float(center[0]), float(center[1])
    x0 = int(np.floor(cx - radius))
    x1 = int(np.ceil(cx + radius))
    y0 = int(np.floor(cy - radius))
    y1 = int(np.ceil(cy + radius))
    if x0 < 0 or y0 < 0 or x1 >= width or y1 >= height:
        raise ValueError("中央码格采样区域越出图像")
    yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= float(radius) ** 2
    values = np.asarray(gray[y0:y1 + 1, x0:x1 + 1], dtype=float)[mask]
    if len(values) == 0:
        raise ValueError("中央码格采样区域为空")
    return float(values.mean())


def refine_marker_center(
    gray: np.ndarray,
    corners: np.ndarray,
    rough_center_uv: Sequence[float],
    dictionary,
    marker_id: int,
    config: Optional[CenterRefineConfig] = None,
) -> CenterRefinementResult:
    """用中央黑白棋盘交叉点把 ArUco 粗中心细化到亚像素位置。"""
    config = config or CenterRefineConfig()
    image = np.asarray(gray)
    if image.ndim != 2 or image.size == 0:
        return CenterRefinementResult(message="中央精定位需要非空灰度图")
    quad = np.asarray(corners, dtype=float)
    rough = np.asarray(rough_center_uv, dtype=float)
    if quad.shape != (4, 2) or rough.shape != (2,):
        return CenterRefinementResult(message="中央精定位输入尺寸无效")
    if not np.all(np.isfinite(quad)) or not np.all(np.isfinite(rough)):
        return CenterRefinementResult(message="中央精定位输入包含非有限数值")
    try:
        white_pattern = _central_checkerboard_pattern(dictionary, marker_id)
    except ValueError as exc:
        return CenterRefinementResult(message=str(exc))

    edge_lengths = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
    mean_side = float(edge_lengths.mean())
    axis_x = ((quad[1] - quad[0]) + (quad[2] - quad[3])) * 0.5
    axis_y = ((quad[3] - quad[0]) + (quad[2] - quad[1])) * 0.5
    axis_x_length = float(np.linalg.norm(axis_x))
    axis_y_length = float(np.linalg.norm(axis_y))
    total_cells = int(dictionary.markerSize) + 2
    if (
        not np.isfinite(mean_side)
        or mean_side <= 0.0
        or axis_x_length <= _EPSILON
        or axis_y_length <= _EPSILON
        or total_cells <= 0
    ):
        return CenterRefinementResult(message="ArUco 四边形尺度或方向无效")

    window_px = int(np.clip(
        round(mean_side * float(config.window_ratio)),
        int(config.min_window_px),
        int(config.max_window_px),
    ))
    height, width = image.shape[:2]
    if not (
        window_px + 1 <= rough[0] < width - window_px - 1
        and window_px + 1 <= rough[1] < height - window_px - 1
    ):
        return CenterRefinementResult(window_px=window_px, message="中央亚像素窗口越出图像")

    if image.dtype not in (np.uint8, np.float32):
        subpixel_image = image.astype(np.float32)
    else:
        subpixel_image = image
    point = rough.astype(np.float32).reshape(1, 1, 2)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        int(config.max_iterations),
        float(config.epsilon_px),
    )
    try:
        cv2.cornerSubPix(
            subpixel_image, point, (window_px, window_px), (-1, -1), criteria
        )
    except cv2.error as exc:
        return CenterRefinementResult(
            window_px=window_px, message=f"中央亚像素优化失败: {exc}"
        )
    refined = point.reshape(2).astype(float)
    delta = refined - rough
    shift_px = float(np.linalg.norm(delta))
    mean_cell_px = mean_side / total_cells
    max_shift_px = float(config.max_shift_cell_ratio) * mean_cell_px
    if not np.all(np.isfinite(refined)) or not np.isfinite(shift_px):
        return CenterRefinementResult(window_px=window_px, message="中央亚像素结果非有限")
    if shift_px > max_shift_px:
        return CenterRefinementResult(
            center_uv=(float(refined[0]), float(refined[1])),
            delta_uv=(float(delta[0]), float(delta[1])),
            shift_px=shift_px,
            window_px=window_px,
            message=(
                f"中央修正量 {shift_px:.3f}px 超过允许值 {max_shift_px:.3f}px"
            ),
        )
    if not point_in_convex_quad(refined, quad):
        return CenterRefinementResult(
            center_uv=(float(refined[0]), float(refined[1])),
            delta_uv=(float(delta[0]), float(delta[1])),
            shift_px=shift_px,
            window_px=window_px,
            message="中央亚像素结果越出 ArUco 四边形",
        )

    unit_x = axis_x / axis_x_length
    unit_y = axis_y / axis_y_length
    cell_x_px = axis_x_length / total_cells
    cell_y_px = axis_y_length / total_cells
    sample_radius = max(
        2.0,
        min(cell_x_px, cell_y_px) * float(config.sample_radius_cell_ratio),
    )
    sample_means = np.empty((2, 2), dtype=float)
    try:
        for row, sign_y in enumerate((-1.0, 1.0)):
            for column, sign_x in enumerate((-1.0, 1.0)):
                sample_center = (
                    refined
                    + sign_x * cell_x_px * float(config.sample_offset_cell_ratio) * unit_x
                    + sign_y * cell_y_px * float(config.sample_offset_cell_ratio) * unit_y
                )
                sample_means[row, column] = _circular_patch_mean(
                    image, sample_center, sample_radius
                )
    except ValueError as exc:
        return CenterRefinementResult(
            center_uv=(float(refined[0]), float(refined[1])),
            delta_uv=(float(delta[0]), float(delta[1])),
            shift_px=shift_px,
            window_px=window_px,
            message=str(exc),
        )

    white_values = sample_means[white_pattern]
    black_values = sample_means[~white_pattern]
    contrast = float(np.min(white_values) - np.max(black_values))
    if contrast < float(config.min_contrast):
        return CenterRefinementResult(
            center_uv=(float(refined[0]), float(refined[1])),
            delta_uv=(float(delta[0]), float(delta[1])),
            shift_px=shift_px,
            contrast=contrast,
            window_px=window_px,
            message=(
                f"中央黑白分离度 {contrast:.3f} 低于阈值 {config.min_contrast:.3f}"
            ),
        )
    return CenterRefinementResult(
        found=True,
        center_uv=(float(refined[0]), float(refined[1])),
        delta_uv=(float(delta[0]), float(delta[1])),
        shift_px=shift_px,
        contrast=contrast,
        window_px=window_px,
    )


def detect_aruco_center(
    image: np.ndarray,
    detector,
    marker_id: int = 0,
    ref_uv: Optional[Sequence[float]] = None,
    refine_config: Optional[CenterRefineConfig] = None,
) -> OffsetResult:
    """在高位和低位统一使用的 ArUco 中心检测函数。

    只接受唯一且 ID 匹配的标记；对角线交点仅作粗中心，最终中心取
    标记中央棋盘角点的灰度亚像素位置。误差符号为 refined_center - ref_uv
    （ref_uv 缺省为图像几何中心）。
    任何失败都以 found=False 返回，不抛出异常。
    """
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return OffsetResult(message="图像尺寸无效")
    if ref_uv is None:
        ref_uv = image_center_uv(width, height)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, ids, rejected = detector.detectMarkers(gray)
    rejected_lists = None
    if rejected is not None:
        rejected_lists = [
            np.asarray(candidate, dtype=float).reshape(4, 2).round(3).tolist()
            for candidate in rejected
        ]
    if ids is None or len(ids) == 0:
        return OffsetResult(
            message="未识别到任何 ArUco 标记", rejected_corners=rejected_lists
        )
    matches = np.flatnonzero(ids[:, 0] == int(marker_id))
    if len(matches) == 0:
        return OffsetResult(
            message=f"未找到 ID={marker_id} 的标记", rejected_corners=rejected_lists
        )
    if len(matches) > 1:
        return OffsetResult(
            message=f"检测到多个 ID={marker_id} 的标记，结果不唯一",
            rejected_corners=rejected_lists,
        )
    detected = np.asarray(corners[matches[0]], dtype=float).reshape(4, 2)
    if not np.all(np.isfinite(detected)):
        return OffsetResult(
            message="标记角点包含非有限数值", rejected_corners=rejected_lists
        )
    try:
        rough_center = diagonal_intersection(detected)
    except ValueError as exc:
        return OffsetResult(
            message=str(exc), corners=detected, rejected_corners=rejected_lists
        )
    if not point_in_convex_quad(rough_center, detected):
        return OffsetResult(
            rough_center_uv=rough_center,
            corners=detected,
            message="ArUco 粗中心越出四边形",
            rejected_corners=rejected_lists,
        )
    refinement = refine_marker_center(
        gray,
        detected,
        rough_center,
        detector.getDictionary(),
        marker_id,
        refine_config,
    )
    if not refinement.found:
        return OffsetResult(
            center_uv=refinement.center_uv,
            rough_center_uv=rough_center,
            refine_delta_uv=refinement.delta_uv,
            refine_shift_px=refinement.shift_px,
            center_contrast=refinement.contrast,
            corners=detected,
            rejected_corners=rejected_lists,
            message=refinement.message,
        )
    center = refinement.center_uv
    return OffsetResult(
        found=True,
        px=float(center[0]),
        py=float(center[1]),
        dx_px=float(center[0] - ref_uv[0]),
        dy_px=float(center[1] - ref_uv[1]),
        center_uv=(float(center[0]), float(center[1])),
        rough_center_uv=rough_center,
        refine_delta_uv=refinement.delta_uv,
        refine_shift_px=refinement.shift_px,
        center_contrast=refinement.contrast,
        corners=detected,
        rejected_corners=rejected_lists,
        message="",
    )


def median_center(centers: Sequence[Sequence[float]]) -> Tuple[float, float]:
    """逐轴取中位数。"""
    values = np.asarray(list(centers), dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        raise ValueError("中心序列必须是非空二维点集")
    return float(np.median(values[:, 0])), float(np.median(values[:, 1]))


def static_center_stats(
    centers: Sequence[Sequence[float]],
    requested_frames: int,
    min_valid_ratio: float = 0.8,
) -> dict:
    """静止采样统计：均值、标准差、有效帧数与完整度。

    centers 中每个元素为 (u, v) 或包含非有限值的无效帧。
    """
    valid = [
        (float(x), float(y))
        for x, y in centers
        if np.isfinite(float(x)) and np.isfinite(float(y))
    ]
    valid_count = len(valid)
    requested = max(0, int(requested_frames))
    complete = (
        requested > 0
        and valid_count >= 1
        and valid_count >= requested * float(min_valid_ratio)
    )
    if valid_count == 0:
        mean = (float("nan"), float("nan"))
        std = (float("nan"), float("nan"))
    else:
        array = np.asarray(valid, dtype=float)
        mean = (float(array[:, 0].mean()), float(array[:, 1].mean()))
        if valid_count >= 2:
            std = (
                float(array[:, 0].std(ddof=1)),
                float(array[:, 1].std(ddof=1)),
            )
        else:
            std = (0.0, 0.0)
    return {
        "均值": mean,
        "标准差": std,
        "有效帧数": valid_count,
        "请求帧数": requested,
        "完整": complete,
    }


def zero_error_tcp_xy(
    actual_tcp_xy: Sequence[float],
    static_mean_error: Sequence[float],
    matrix: Sequence[Sequence[float]],
    complete: bool,
) -> Optional[Tuple[float, float]]:
    """零误差等效 TCP：actual + matrix @ 静止均值误差。

    沿用正式控制矩阵的方向和符号，但不应用单步限幅或最小步长；
    静止采样不完整时不生成零误差标签，返回 None。
    """
    if not complete:
        return None
    actual = np.asarray(actual_tcp_xy, dtype=float)
    error = np.asarray(static_mean_error, dtype=float)
    transform = np.asarray(matrix, dtype=float)
    if actual.shape != (2,) or error.shape != (2,) or transform.shape != (2, 2):
        raise ValueError("零误差等效 TCP 需要 2 维实测点、2 维误差与 2x2 矩阵")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(error)) or not np.all(
        np.isfinite(transform)
    ):
        raise ValueError("零误差等效 TCP 输入包含非有限数值")
    equivalent = actual + transform @ error
    return float(equivalent[0]), float(equivalent[1])


def validate_motion_pose(
    pose: Sequence[float],
    minimum_z_mm: float,
    safe_x_range_mm: Sequence[float],
    safe_y_range_mm: Sequence[float],
) -> Tuple[bool, str]:
    """发送运动前的六维位姿安全检查，返回 (通过, 原因)。"""
    if len(pose) != 6:
        return False, f"位姿必须包含 6 个数值，实际 {len(pose)}"
    if not all(np.isfinite(float(value)) for value in pose):
        return False, "位姿包含 NaN 或无穷值"
    if pose[2] < float(minimum_z_mm):
        return False, f"TCP Z={pose[2]:.3f}mm 低于安全下限 {minimum_z_mm}mm"
    x_min, x_max = sorted(float(value) for value in safe_x_range_mm)
    y_min, y_max = sorted(float(value) for value in safe_y_range_mm)
    if not (x_min <= pose[0] <= x_max):
        return False, f"TCP X={pose[0]:.3f}mm 越出安全范围 [{x_min}, {x_max}]"
    if not (y_min <= pose[1] <= y_max):
        return False, f"TCP Y={pose[1]:.3f}mm 越出安全范围 [{y_min}, {y_max}]"
    return True, ""


def check_image_size_constant(
    width: int,
    height: int,
    expected_size: Optional[Tuple[int, int]],
) -> bool:
    """实验过程中图像尺寸必须保持不变。"""
    if expected_size is None:
        return True
    return (int(width), int(height)) == (int(expected_size[0]), int(expected_size[1]))


def draw_rejected_candidates(canvas: np.ndarray, rejected_corners) -> np.ndarray:
    """用黄色在画布上叠加 rejected 候选四边形（找到外框但解码失败）。"""
    if not rejected_corners:
        return canvas
    for corners in rejected_corners:
        quad = np.round(np.asarray(corners, dtype=float)).astype(int).reshape(4, 2)
        cv2.polylines(canvas, [quad], isClosed=True, color=(0, 255, 255), thickness=2)
    return canvas


def draw_servo_overlay(
    canvas: np.ndarray,
    *,
    round_no,
    error_xy=None,
    center_uv=None,
    rough_center_uv=None,
    refine_delta_uv=None,
    center_contrast=None,
    corners=None,
    rejected_corners=None,
) -> np.ndarray:
    """低位伺服叠加：绿色外框、青色粗中心、红色精中心与诊断数值。"""
    overlay = canvas.copy()
    height, width = overlay.shape[:2]
    ref = (int(width / 2), int(height / 2))
    cv2.drawMarker(
        overlay, ref, color=(255, 0, 255),
        markerType=cv2.MARKER_CROSS, markerSize=24, thickness=1,
    )
    if corners is not None:
        quad = np.round(np.asarray(corners, dtype=float)).astype(int).reshape(4, 2)
        cv2.polylines(overlay, [quad], isClosed=True, color=(0, 220, 0), thickness=2)
    rough = None
    if rough_center_uv is not None and np.all(np.isfinite(rough_center_uv)):
        rough = (int(round(rough_center_uv[0])), int(round(rough_center_uv[1])))
        cv2.drawMarker(
            overlay, rough, color=(255, 255, 0),
            markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2,
        )
    if center_uv is not None:
        center = (int(round(center_uv[0])), int(round(center_uv[1])))
        if rough is not None:
            cv2.line(overlay, rough, center, color=(255, 255, 0), thickness=1)
        cv2.drawMarker(
            overlay, center, color=(0, 0, 255),
            markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2,
        )
    overlay = draw_rejected_candidates(overlay, rejected_corners or [])
    lines = [f"轮次: {round_no}"]
    if error_xy is not None:
        lines.append(f"误差: ({error_xy[0]:+.2f}, {error_xy[1]:+.2f}) px")
    if refine_delta_uv is not None and np.all(np.isfinite(refine_delta_uv)):
        lines.append(
            f"中央修正: ({refine_delta_uv[0]:+.2f}, {refine_delta_uv[1]:+.2f}) px"
        )
    if center_contrast is not None and np.isfinite(float(center_contrast)):
        lines.append(f"中央黑白分离度: {float(center_contrast):.2f}")
    for index, line in enumerate(lines):
        put_chinese_text(
            overlay, line, (10, 4 + index * 25), (0, 200, 255), font_size=20,
        )
    return overlay


def draw_aruco_debug(
    image: np.ndarray,
    corners: Optional[np.ndarray],
    center_uv: Optional[Sequence[float]],
    ref_uv: Sequence[float],
    path: Path,
    *,
    rough_center_uv: Optional[Sequence[float]] = None,
    refine_delta_uv: Optional[Sequence[float]] = None,
    center_contrast: Optional[float] = None,
) -> bool:
    """绘制 ArUco 粗定位、中央精定位、参考像素并保存。"""
    canvas = image.copy()
    if corners is not None:
        quad = np.asarray(corners, dtype=float).reshape(4, 2)
        int_quad = np.round(quad).astype(int)
        cv2.polylines(canvas, [int_quad], isClosed=True, color=(0, 200, 0), thickness=2)
        for index, point in enumerate(int_quad):
            color = (0, 0, 255) if index == 0 else (0, 255, 255)
            cv2.circle(canvas, tuple(point), 4, color, -1)
        cv2.line(
            canvas,
            tuple(int_quad[0]),
            tuple(int_quad[2]),
            color=(255, 0, 0),
            thickness=1,
        )
        cv2.line(
            canvas,
            tuple(int_quad[1]),
            tuple(int_quad[3]),
            color=(255, 128, 0),
            thickness=1,
        )
    rough = None
    if rough_center_uv is not None and np.all(np.isfinite(rough_center_uv)):
        rough = (int(round(rough_center_uv[0])), int(round(rough_center_uv[1])))
        cv2.circle(canvas, rough, 5, (255, 255, 0), 1)
        cv2.drawMarker(
            canvas, rough, color=(255, 255, 0),
            markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2,
        )
    if center_uv is not None and np.all(np.isfinite(center_uv)):
        center = (int(round(center_uv[0])), int(round(center_uv[1])))
        if rough is not None:
            cv2.line(canvas, rough, center, color=(255, 255, 0), thickness=1)
        cv2.circle(canvas, center, 6, (0, 0, 255), 2)
        cv2.drawMarker(
            canvas, center, color=(0, 0, 255),
            markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2,
        )
    ref = (int(round(ref_uv[0])), int(round(ref_uv[1])))
    cv2.circle(canvas, ref, 5, (255, 0, 255), -1)
    lines = []
    if refine_delta_uv is not None and np.all(np.isfinite(refine_delta_uv)):
        lines.append(
            f"中央修正: ({refine_delta_uv[0]:+.2f}, {refine_delta_uv[1]:+.2f}) px"
        )
    if center_contrast is not None and np.isfinite(float(center_contrast)):
        lines.append(f"中央黑白分离度: {float(center_contrast):.2f}")
    for index, line in enumerate(lines):
        put_chinese_text(
            canvas, line, (10, 4 + index * 25), (0, 200, 255), font_size=20,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), canvas))
