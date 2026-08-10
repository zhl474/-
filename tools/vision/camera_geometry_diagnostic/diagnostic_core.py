"""相机几何诊断工具共用的纯计算函数。

本模块不导入 ROS，也不连接相机，便于离线分析和无硬件单元测试。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


PINHOLE_MODEL_NAME = "pinhole_radtan_5"
CIRCLE_DETECTION_METHODS = (
    "raw_white",
    "raw_white_clustering",
    "clahe_white",
    "clahe_white_clustering",
    "inverted_black",
    "inverted_black_clustering",
)


@dataclass(frozen=True)
class CircleDetection:
    """非对称圆点阵检测结果。"""

    found: bool
    centers: np.ndarray | None
    branch: str
    message: str
    keypoint_count: int = 0


@dataclass(frozen=True)
class PlaneFit:
    """三维点的正交平面拟合结果。"""

    centroid: np.ndarray
    normal: np.ndarray
    basis: np.ndarray
    residuals: np.ndarray
    rmse: float
    p95: float
    max_abs: float


class CircleStabilityTracker:
    """检查圆点检测是否在连续若干帧内真正静止。"""

    def __init__(
        self,
        required_frames: int = 5,
        max_rms_jitter_px: float = 0.65,
        max_point_jitter_px: float = 1.50,
    ):
        if int(required_frames) < 2:
            raise ValueError("required_frames 必须至少为 2")
        self.required_frames = int(required_frames)
        self.max_rms_jitter_px = float(max_rms_jitter_px)
        self.max_point_jitter_px = float(max_point_jitter_px)
        self._frames = []

    def reset(self) -> None:
        self._frames.clear()

    def update(self, centers: np.ndarray | None) -> dict:
        """加入一帧圆心；检测失败或移动过大时重新累计。"""
        if centers is None:
            self.reset()
            return {
                "count": 0,
                "required": self.required_frames,
                "stable": False,
                "rms_jitter_px": None,
                "max_jitter_px": None,
            }
        current = np.asarray(centers, dtype=float).reshape(-1, 2)
        if not np.all(np.isfinite(current)):
            self.reset()
            raise ValueError("圆心包含非有限数值")
        if self._frames and current.shape != self._frames[-1].shape:
            self.reset()
        self._frames.append(current.copy())
        self._frames = self._frames[-self.required_frames :]
        stack = np.stack(self._frames, axis=0)
        average = np.mean(stack, axis=0)
        jitter = np.linalg.norm(stack - average, axis=2)
        rms = float(np.sqrt(np.mean(jitter**2)))
        maximum = float(np.max(jitter))
        if (
            len(self._frames) > 1
            and (
                rms > self.max_rms_jitter_px
                or maximum > self.max_point_jitter_px
            )
        ):
            self._frames = [current.copy()]
            rms = 0.0
            maximum = 0.0
        count = len(self._frames)
        return {
            "count": count,
            "required": self.required_frames,
            "stable": count >= self.required_frames,
            "rms_jitter_px": rms,
            "max_jitter_px": maximum,
        }


def to_builtin(value):
    """递归转换 NumPy/Path 对象，供 JSON 和 YAML 安全序列化。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if hasattr(value, "name") and hasattr(value, "value"):
        return {"name": str(value.name), "value": to_builtin(value.value)}
    return value


def save_image(path: Path | str, image: np.ndarray) -> None:
    """保存图片并兼容中文路径。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extension = output_path.suffix or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise RuntimeError(f"图片编码失败: {output_path}")
    encoded.tofile(str(output_path))


def put_chinese_text(
    image_bgr: np.ndarray,
    text: str,
    origin: Sequence[int],
    color: Sequence[int],
    font_size: int = 22,
) -> np.ndarray:
    """用 Pillow 在 BGR 图像上绘制中文；无可用字体时回退 OpenCV。"""
    try:
        from PIL import Image, ImageDraw, ImageFont

        font_paths = (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
        font_path = next((path for path in font_paths if Path(path).is_file()), None)
        if font_path is None:
            raise RuntimeError("没有可用中文字体")
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.truetype(font_path, int(font_size))
        blue, green, red = (int(value) for value in color)
        draw.text(
            (int(origin[0]), int(origin[1])),
            str(text),
            font=font,
            fill=(red, green, blue),
        )
        image_bgr[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
        return image_bgr
    except Exception:  # noqa: BLE001
        cv2.putText(
            image_bgr,
            str(text),
            (int(origin[0]), int(origin[1]) + int(font_size)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            tuple(int(value) for value in color),
            1,
            cv2.LINE_AA,
        )
        return image_bgr


def read_image(path: Path | str) -> np.ndarray:
    """读取图片并兼容中文路径。"""
    image_path = Path(path)
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"图片读取失败: {image_path}")
    return image


def create_asymmetric_object_points(
    pattern_size: Sequence[int],
    grid_step: float,
) -> np.ndarray:
    """按 OpenCV 行优先顺序生成非对称圆阵物点。"""
    if len(pattern_size) != 2:
        raise ValueError("pattern_size 必须是 (每行点数, 行数)")
    columns, rows = (int(pattern_size[0]), int(pattern_size[1]))
    step = float(grid_step)
    if columns <= 1 or rows <= 1:
        raise ValueError("非对称圆阵行列数都必须大于 1")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("grid_step 必须是大于 0 的有限数值")
    points = [
        ((2 * column + row % 2) * step, row * step, 0.0)
        for row in range(rows)
        for column in range(columns)
    ]
    return np.asarray(points, dtype=np.float32)


def create_blob_detector(
    blob_color: int = 255,
    min_threshold: float = 50.0,
    max_threshold: float = 220.0,
    threshold_step: float = 5.0,
    min_area: float = 50.0,
    max_area: float = 10000.0,
    min_circularity: float = 0.50,
    min_convexity: float = 0.70,
    min_inertia_ratio: float = 0.35,
) -> cv2.SimpleBlobDetector:
    """创建圆点Blob检测器；默认阈值排除实测黑底上的低亮度伪圆。"""
    if int(blob_color) not in (0, 255):
        raise ValueError("blob_color 只能是 0 或 255")
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = float(min_threshold)
    params.maxThreshold = float(max_threshold)
    params.thresholdStep = float(threshold_step)
    params.minRepeatability = 2
    params.minDistBetweenBlobs = 8.0
    params.filterByColor = True
    params.blobColor = int(blob_color)
    params.filterByArea = True
    params.minArea = float(min_area)
    params.maxArea = float(max_area)
    params.filterByCircularity = True
    params.minCircularity = float(min_circularity)
    params.filterByConvexity = True
    params.minConvexity = float(min_convexity)
    params.filterByInertia = True
    params.minInertiaRatio = float(min_inertia_ratio)
    return cv2.SimpleBlobDetector_create(params)


def detect_asymmetric_circles(
    image_bgr: np.ndarray,
    pattern_size: Sequence[int],
    methods: Sequence[str] | str | None = None,
    blob_detector_options: dict | None = None,
) -> CircleDetection:
    """按明确指定的方法检测非对称圆阵；默认保留旧版全部尝试行为。"""
    if image_bgr is None or image_bgr.size == 0:
        return CircleDetection(False, None, "无", "输入图像为空", 0)
    columns, rows = (int(pattern_size[0]), int(pattern_size[1]))
    expected_count = columns * rows
    if expected_count <= 0:
        return CircleDetection(False, None, "无", "pattern_size 无效", 0)
    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    elif image_bgr.ndim == 2:
        gray = image_bgr.astype(np.uint8, copy=False)
    else:
        return CircleDetection(False, None, "无", "图像必须是灰度或 BGR", 0)

    if methods is None:
        selected_methods = CIRCLE_DETECTION_METHODS
    elif isinstance(methods, str):
        selected_methods = (methods,)
    else:
        selected_methods = tuple(str(item) for item in methods)
    unknown = [item for item in selected_methods if item not in CIRCLE_DETECTION_METHODS]
    if unknown:
        raise ValueError(
            f"未知圆点检测方法 {unknown}；可选值为 {CIRCLE_DETECTION_METHODS}"
        )
    if not selected_methods:
        raise ValueError("至少要指定一种圆点检测方法")

    clahe = None
    method_descriptions = {
        "raw_white": ("原始灰度-白点", False, False),
        "raw_white_clustering": ("原始灰度-白点-聚类", False, True),
        "clahe_white": ("局部增强-白点", True, False),
        "clahe_white_clustering": ("局部增强-白点-聚类", True, True),
        "inverted_black": ("反相灰度-黑点", False, False),
        "inverted_black_clustering": ("反相灰度-黑点-聚类", False, True),
    }
    last_keypoint_count = 0
    detector_options = dict(blob_detector_options or {})
    for method in selected_methods:
        branch_name, use_clahe, use_clustering = method_descriptions[method]
        if use_clahe:
            if clahe is None:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            branch_image = clahe
            blob_color = 255
        elif method.startswith("inverted_black"):
            branch_image = cv2.bitwise_not(gray)
            blob_color = 0
        else:
            branch_image = gray
            blob_color = 255
        detector = create_blob_detector(blob_color, **detector_options)
        last_keypoint_count = len(detector.detect(branch_image))
        flags = cv2.CALIB_CB_ASYMMETRIC_GRID
        if use_clustering:
            flags |= cv2.CALIB_CB_CLUSTERING
        found, centers = cv2.findCirclesGrid(
            branch_image,
            (columns, rows),
            flags=flags,
            blobDetector=detector,
        )
        if not found or centers is None:
            continue
        ordered = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
        if len(ordered) != expected_count or not np.all(np.isfinite(ordered)):
            continue
        return CircleDetection(
            True,
            ordered,
            branch_name,
            "检测成功",
            int(last_keypoint_count),
        )
    return CircleDetection(
        False,
        None,
        "/".join(selected_methods),
        f"指定方法未检测到完整的 {columns}×{rows} 非对称圆阵",
        int(last_keypoint_count),
    )


def draw_circle_order(
    image_bgr: np.ndarray,
    detection: CircleDetection,
    pattern_size: Sequence[int],
) -> np.ndarray:
    """绘制圆心顺序；0 为红色，末点为黄色，其余为绿色。"""
    canvas = image_bgr.copy()
    if not detection.found or detection.centers is None:
        put_chinese_text(
            canvas,
            "检测失败",
            (20, 15),
            (0, 0, 255),
            24,
        )
        return canvas
    centers = detection.centers
    columns, rows = int(pattern_size[0]), int(pattern_size[1])
    # 只连接同一物理行，并沿每行首点连接行方向；避免跨行折线被误认为错序。
    for row in range(rows):
        row_points = centers[row * columns : (row + 1) * columns]
        for first_point, second_point in zip(row_points[:-1], row_points[1:]):
            cv2.line(
                canvas,
                tuple(np.rint(first_point).astype(int)),
                tuple(np.rint(second_point).astype(int)),
                (0, 200, 0),
                2,
                cv2.LINE_AA,
            )
    row_starts = centers[::columns]
    for first_point, second_point in zip(row_starts[:-1], row_starts[1:]):
        cv2.line(
            canvas,
            tuple(np.rint(first_point).astype(int)),
            tuple(np.rint(second_point).astype(int)),
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )
    for index, point in enumerate(centers):
        center = tuple(np.rint(point).astype(int))
        color = (0, 0, 255) if index == 0 else (0, 255, 255) if index == len(centers) - 1 else (0, 220, 0)
        cv2.circle(canvas, center, 5, color, -1)
        cv2.putText(
            canvas,
            str(index),
            (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    first = tuple(np.rint(centers[0]).astype(int))
    second = tuple(np.rint(centers[1]).astype(int))
    cv2.arrowedLine(canvas, first, second, (255, 0, 255), 2, cv2.LINE_AA, tipLength=0.16)
    return canvas


def sample_signature(centers: np.ndarray, image_size: Sequence[int]) -> dict:
    """提取位置、尺度和方向，用于提示重复采样。"""
    points = np.asarray(centers, dtype=float).reshape(-1, 2)
    width, height = float(image_size[0]), float(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("图像尺寸无效")
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-9)
    direction = points[-1] - points[0]
    return {
        "中心归一化": (points.mean(axis=0) / np.array([width, height])).tolist(),
        "包围面积比例": float((span[0] * span[1]) / (width * height)),
        "首尾方向角度": float(np.degrees(np.arctan2(direction[1], direction[0]))),
    }


def estimate_capture_pose(
    centers: np.ndarray,
    pattern_size: Sequence[int],
    image_size: Sequence[int],
    approximate_hfov_deg: float = 86.0,
) -> dict:
    """估算采集位置、画面尺寸和板面倾角，只用于采集导航。"""
    points = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    width, height = float(image_size[0]), float(image_size[1])
    expected = int(pattern_size[0]) * int(pattern_size[1])
    if len(points) != expected or width <= 0.0 or height <= 0.0:
        raise ValueError("圆心数量或图像尺寸无效")
    center = np.mean(points, axis=0)
    center_normalized = center / np.array([width, height], dtype=float)

    def axis_zone(value):
        if value < 0.38:
            return -1
        if value > 0.62:
            return 1
        return 0

    horizontal = axis_zone(float(center_normalized[0]))
    vertical = axis_zone(float(center_normalized[1]))
    position_names = {
        (-1, -1): "左上", (0, -1): "上", (1, -1): "右上",
        (-1, 0): "左", (0, 0): "中央", (1, 0): "右",
        (-1, 1): "左下", (0, 1): "下", (1, 1): "右下",
    }
    position = position_names[(horizontal, vertical)]

    rectangle = cv2.minAreaRect(points.astype(np.float32))
    long_side = float(max(rectangle[1]))
    size_ratio = long_side / min(width, height)
    if size_ratio < 0.20:
        size_class = "过远"
    elif size_ratio < 0.34:
        size_class = "远"
    elif size_ratio < 0.60:
        size_class = "中"
    elif size_ratio <= 0.82:
        size_class = "近"
    else:
        size_class = "过近"

    horizontal_fov = np.radians(float(approximate_hfov_deg))
    focal = width / (2.0 * np.tan(horizontal_fov / 2.0))
    approximate_k = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    object_points = create_asymmetric_object_points(pattern_size, 1.0).astype(np.float64)
    success, rotation_vector, _translation_vector = cv2.solvePnP(
        object_points,
        points,
        approximate_k,
        np.zeros(5, dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("无法估算标定板采集姿态")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    normal = rotation[:, 2].astype(float)
    if normal[2] < 0.0:
        normal *= -1.0
    tilt_deg = float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))
    if tilt_deg < 10.0:
        tilt_direction = "正视"
    elif abs(normal[0]) >= abs(normal[1]):
        tilt_direction = "右侧靠近" if normal[0] > 0.0 else "左侧靠近"
    else:
        tilt_direction = "下侧靠近" if normal[1] > 0.0 else "上侧靠近"
    return {
        "center_normalized": center_normalized.tolist(),
        "position": position,
        "size_ratio": float(size_ratio),
        "size_class": size_class,
        "tilt_deg": tilt_deg,
        "tilt_direction": tilt_direction,
        "approximate_hfov_deg": float(approximate_hfov_deg),
        "navigation_only": True,
    }


def summarize_capture_coverage(
    poses: Sequence[dict],
    minimum_samples: int = 20,
) -> dict:
    """汇总定量采样进度，并给出下一项明确动作。"""
    position_order = ("中央", "左", "右", "上", "下", "左上", "右上", "左下", "右下")
    scale_order = ("远", "中", "近")
    tilt_order = ("正视", "左侧靠近", "右侧靠近", "上侧靠近", "下侧靠近")
    position_targets = {name: 1 for name in position_order}
    scale_targets = {"远": 4, "中": 8, "近": 4}
    tilt_targets = {"正视": 4, "左侧靠近": 3, "右侧靠近": 3, "上侧靠近": 3, "下侧靠近": 3}
    position_counts = {name: 0 for name in position_order}
    scale_counts = {name: 0 for name in scale_order}
    tilt_counts = {name: 0 for name in tilt_order}
    excessive_tilt_count = 0
    out_of_range_size_count = 0
    for pose in poses:
        if pose.get("position") in position_counts:
            position_counts[pose["position"]] += 1
        if pose.get("size_class") in scale_counts:
            scale_counts[pose["size_class"]] += 1
        tilt = float(pose.get("tilt_deg", np.inf))
        direction = pose.get("tilt_direction")
        if tilt <= 35.0 and direction in tilt_counts:
            tilt_counts[direction] += 1
        elif tilt > 35.0:
            excessive_tilt_count += 1
        if pose.get("size_class") in ("过远", "过近"):
            out_of_range_size_count += 1

    def capped_progress(counts, targets):
        complete = sum(min(counts[name], targets[name]) for name in targets)
        total = sum(targets.values())
        return complete, total

    position_progress = capped_progress(position_counts, position_targets)
    scale_progress = capped_progress(scale_counts, scale_targets)
    tilt_progress = capped_progress(tilt_counts, tilt_targets)
    sample_progress = (min(len(poses), int(minimum_samples)), int(minimum_samples))
    complete_units = sum(item[0] for item in (position_progress, scale_progress, tilt_progress, sample_progress))
    total_units = sum(item[1] for item in (position_progress, scale_progress, tilt_progress, sample_progress))

    missing_positions = [name for name in position_order if position_counts[name] < position_targets[name]]
    missing_scales = [name for name in scale_order if scale_counts[name] < scale_targets[name]]
    missing_tilts = [name for name in tilt_order if tilt_counts[name] < tilt_targets[name]]
    if not poses:
        instruction = "先把板放在图像中央，中等大小，基本正对相机"
    elif missing_positions:
        target = missing_positions[0]
        coordinate = {
            "中央": "u=50%, v=50%", "左": "u=25%, v=50%", "右": "u=75%, v=50%",
            "上": "u=50%, v=25%", "下": "u=50%, v=75%",
            "左上": "u=25%, v=25%", "右上": "u=75%, v=25%",
            "左下": "u=25%, v=75%", "右下": "u=75%, v=75%",
        }[target]
        instruction = f"把板中心移到{target}（约 {coordinate}），保持28点完整可见"
    elif missing_scales:
        target = missing_scales[0]
        action = {
            "远": "把板移远，使板长边约占图像短边20%~34%",
            "中": "调整到中距离，使板长边约占图像短边34%~60%",
            "近": "把板移近，使板长边约占图像短边60%~82%",
        }[target]
        instruction = action
    elif missing_tilts:
        target = missing_tilts[0]
        if target == "正视":
            instruction = "让板基本正对相机，估算倾角小于10°"
        else:
            instruction = f"让标定板{target}，把估算倾角保持在15°~30°"
    elif len(poses) < int(minimum_samples):
        instruction = "覆盖类型已齐，继续组合边缘位置和15°~30°倾斜补足样本数"
    else:
        instruction = "采样覆盖已达到默认要求，可以结束采集并离线求解"
    ready = (
        len(poses) >= int(minimum_samples)
        and not missing_positions
        and not missing_scales
        and not missing_tilts
    )
    return {
        "sample_count": int(len(poses)),
        "minimum_samples": int(minimum_samples),
        "position_counts": position_counts,
        "position_targets": position_targets,
        "scale_counts": scale_counts,
        "scale_targets": scale_targets,
        "tilt_counts": tilt_counts,
        "tilt_targets": tilt_targets,
        "excessive_tilt_count": int(excessive_tilt_count),
        "out_of_range_size_count": int(out_of_range_size_count),
        "progress_ratio": float(complete_units / max(total_units, 1)),
        "ready": bool(ready),
        "next_instruction": instruction,
    }


def find_similar_signature(
    signature: dict,
    existing: Iterable[dict],
    center_threshold: float = 0.08,
    area_relative_threshold: float = 0.15,
    angle_threshold_deg: float = 8.0,
) -> int | None:
    """返回最先命中的相似样本下标；只提示，不负责拒绝。"""
    center = np.asarray(signature["中心归一化"], dtype=float)
    area = float(signature["包围面积比例"])
    angle = float(signature["首尾方向角度"])
    for index, item in enumerate(existing):
        other_center = np.asarray(item["中心归一化"], dtype=float)
        other_area = float(item["包围面积比例"])
        other_angle = float(item["首尾方向角度"])
        center_delta = float(np.linalg.norm(center - other_center))
        area_delta = abs(area - other_area) / max(area, other_area, 1e-9)
        angle_delta = abs((angle - other_angle + 180.0) % 360.0 - 180.0)
        if (
            center_delta <= center_threshold
            and area_delta <= area_relative_threshold
            and angle_delta <= angle_threshold_deg
        ):
            return index
    return None


def validate_image_size(image: np.ndarray, expected_size: Sequence[int]) -> tuple[int, int]:
    """严格检查图像尺寸，不允许自动 resize。"""
    if image is None or image.ndim < 2:
        raise ValueError("图像为空或维数无效")
    actual = (int(image.shape[1]), int(image.shape[0]))
    expected = (int(expected_size[0]), int(expected_size[1]))
    if actual != expected:
        raise ValueError(f"图像尺寸不一致：期望 {expected}，实际 {actual}，禁止自动缩放")
    return actual


def _calibration_flags(zero_distortion: bool) -> int:
    if not zero_distortion:
        return 0
    flags = cv2.CALIB_ZERO_TANGENT_DIST
    for name in ("CALIB_FIX_K1", "CALIB_FIX_K2", "CALIB_FIX_K3", "CALIB_FIX_K4", "CALIB_FIX_K5", "CALIB_FIX_K6"):
        flags |= int(getattr(cv2, name))
    return flags


def calibrate_from_observations(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    image_size: Sequence[int],
    zero_distortion: bool = False,
) -> dict:
    """从已建立对应关系的点求相机参数和逐视图误差。"""
    if len(object_points) != len(image_points) or len(object_points) < 3:
        raise ValueError("相机标定至少需要 3 张且物点/像点数量一致")
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("标定图像尺寸无效")
    checked_object = [np.asarray(item, dtype=np.float32).reshape(-1, 3) for item in object_points]
    checked_image = [np.asarray(item, dtype=np.float32).reshape(-1, 1, 2) for item in image_points]
    for object_item, image_item in zip(checked_object, checked_image):
        if len(object_item) != len(image_item) or len(object_item) < 4:
            raise ValueError("单张图的物点/像点数量不一致或少于 4")
        if not np.all(np.isfinite(object_item)) or not np.all(np.isfinite(image_item)):
            raise ValueError("标定点包含非有限数值")
    initial_k = cv2.initCameraMatrix2D(checked_object, checked_image, (width, height), 0)
    initial_d = np.zeros((5, 1), dtype=np.float64)
    result = cv2.calibrateCameraExtended(
        checked_object,
        checked_image,
        (width, height),
        initial_k,
        initial_d,
        flags=_calibration_flags(zero_distortion),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 200, 1e-12),
    )
    (
        rms,
        camera_matrix,
        distortion,
        rotation_vectors,
        translation_vectors,
        intrinsic_std,
        extrinsic_std,
        per_view_errors,
    ) = result
    distortion = np.asarray(distortion, dtype=float).reshape(-1)[:5]
    if zero_distortion:
        distortion = np.zeros(5, dtype=float)
    per_view = np.asarray(per_view_errors, dtype=float).reshape(-1)
    return {
        "rms": float(rms),
        "K": np.asarray(camera_matrix, dtype=float),
        "D": distortion,
        "rvecs": [np.asarray(item, dtype=float).reshape(3) for item in rotation_vectors],
        "tvecs": [np.asarray(item, dtype=float).reshape(3) for item in translation_vectors],
        "intrinsic_std": np.asarray(intrinsic_std, dtype=float).reshape(-1),
        "extrinsic_std": np.asarray(extrinsic_std, dtype=float).reshape(-1),
        "per_view_errors": per_view,
    }


def heldout_reprojection_errors(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    image_size: Sequence[int],
    folds: int = 5,
    seed: int = 42,
    zero_distortion: bool = False,
) -> dict:
    """按视图交叉验证内参，并用每张留出图自己的外参计算重投影误差。"""
    sample_count = len(object_points)
    if sample_count < 6:
        raise ValueError("留出验证至少需要 6 张有效图片")
    fold_count = min(max(2, int(folds)), sample_count)
    rng = np.random.default_rng(int(seed))
    shuffled = np.arange(sample_count)
    rng.shuffle(shuffled)
    split_indices = [part for part in np.array_split(shuffled, fold_count) if len(part)]
    per_view = np.full(sample_count, np.nan, dtype=float)
    for test_indices in split_indices:
        test_set = set(int(index) for index in test_indices)
        train_indices = [index for index in range(sample_count) if index not in test_set]
        fitted = calibrate_from_observations(
            [object_points[index] for index in train_indices],
            [image_points[index] for index in train_indices],
            image_size,
            zero_distortion=zero_distortion,
        )
        for index in test_indices:
            object_item = np.asarray(object_points[int(index)], dtype=np.float32).reshape(-1, 3)
            image_item = np.asarray(image_points[int(index)], dtype=np.float32).reshape(-1, 2)
            success, rvec, tvec = cv2.solvePnP(
                object_item,
                image_item,
                fitted["K"],
                fitted["D"],
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                raise RuntimeError(f"第 {int(index)} 张留出图 solvePnP 失败")
            projected, _ = cv2.projectPoints(object_item, rvec, tvec, fitted["K"], fitted["D"])
            residual = projected.reshape(-1, 2) - image_item
            per_view[int(index)] = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    if not np.all(np.isfinite(per_view)):
        raise RuntimeError("留出验证没有覆盖全部视图")
    return {
        "folds": fold_count,
        "seed": int(seed),
        "per_view_errors": per_view,
        "rmse": float(np.sqrt(np.mean(per_view**2))),
        "mean": float(np.mean(per_view)),
        "max": float(np.max(per_view)),
    }


def undistort_pixel_coordinates(
    pixels: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    """在 P=K 的同一像素坐标系中修正像素点。"""
    points = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    d = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if len(d) < 4 or not np.all(np.isfinite(k)) or not np.all(np.isfinite(d)):
        raise ValueError("K/D 无效")
    corrected = cv2.undistortPoints(points, k, d, P=k)
    return corrected.reshape(-1, 2)


def fit_plane_svd(points_xyz: np.ndarray) -> PlaneFit:
    """用 SVD 正交拟合三维平面，并保留所有点的有符号残差。"""
    points = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    if len(points) < 3 or not np.all(np.isfinite(points)):
        raise ValueError("平面拟合至少需要 3 个有限三维点")
    centroid = points.mean(axis=0)
    _u, singular, vh = np.linalg.svd(points - centroid, full_matrices=False)
    if len(singular) < 3 or singular[1] <= 1e-12:
        raise ValueError("三维点共线或退化，无法拟合平面")
    normal = vh[-1].copy()
    if normal[2] < 0:
        normal *= -1.0
    basis = vh[:2].copy()
    residuals = (points - centroid) @ normal
    absolute = np.abs(residuals)
    return PlaneFit(
        centroid=centroid,
        normal=normal,
        basis=basis,
        residuals=residuals,
        rmse=float(np.sqrt(np.mean(residuals**2))),
        p95=float(np.percentile(absolute, 95)),
        max_abs=float(np.max(absolute)),
    )


def plane_quadratic_cross_validation(
    points_xyz: np.ndarray,
    folds: int = 5,
    seed: int = 42,
    max_points: int = 15000,
) -> dict:
    """比较平面和二次曲面留出误差，二次曲面只用于诊断。"""
    points = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    if len(points) < 30 or not np.all(np.isfinite(points)):
        raise ValueError("曲面留出验证至少需要 30 个有限三维点")
    rng = np.random.default_rng(int(seed))
    if len(points) > int(max_points):
        points = points[rng.choice(len(points), size=int(max_points), replace=False)]
    indices = np.arange(len(points))
    rng.shuffle(indices)
    parts = [part for part in np.array_split(indices, min(int(folds), len(points))) if len(part)]
    plane_squared = []
    quadratic_squared = []
    for test_indices in parts:
        test_set = set(int(index) for index in test_indices)
        train_indices = [index for index in range(len(points)) if index not in test_set]
        train = points[train_indices]
        test = points[test_indices]
        plane = fit_plane_svd(train)
        train_local = train - plane.centroid
        test_local = test - plane.centroid
        train_uv = train_local @ plane.basis.T
        test_uv = test_local @ plane.basis.T
        train_w = train_local @ plane.normal
        test_w = test_local @ plane.normal
        train_design = np.column_stack(
            [
                np.ones(len(train_uv)),
                train_uv[:, 0],
                train_uv[:, 1],
                train_uv[:, 0] ** 2,
                train_uv[:, 0] * train_uv[:, 1],
                train_uv[:, 1] ** 2,
            ]
        )
        coefficients, _residual_sum, rank, _singular = np.linalg.lstsq(
            train_design,
            train_w,
            rcond=None,
        )
        if rank < 6:
            raise ValueError("二次曲面设计矩阵秩不足")
        test_design = np.column_stack(
            [
                np.ones(len(test_uv)),
                test_uv[:, 0],
                test_uv[:, 1],
                test_uv[:, 0] ** 2,
                test_uv[:, 0] * test_uv[:, 1],
                test_uv[:, 1] ** 2,
            ]
        )
        plane_squared.extend((test_w**2).tolist())
        quadratic_squared.extend(((test_w - test_design @ coefficients) ** 2).tolist())
    plane_rmse = float(np.sqrt(np.mean(plane_squared)))
    quadratic_rmse = float(np.sqrt(np.mean(quadratic_squared)))
    improvement = 0.0 if plane_rmse <= 1e-12 else 1.0 - quadratic_rmse / plane_rmse
    return {
        "folds": len(parts),
        "seed": int(seed),
        "sample_count": int(len(points)),
        "plane_rmse_mm": plane_rmse,
        "quadratic_rmse_mm": quadratic_rmse,
        "quadratic_improvement_ratio": float(improvement),
    }


def make_board_material_masks(
    image_shape: Sequence[int],
    centers: np.ndarray,
    white_radius_ratio: float = 0.28,
    edge_margin_px: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """由圆心构造板内、白圆和黑底掩码。"""
    height, width = int(image_shape[0]), int(image_shape[1])
    points = np.asarray(centers, dtype=float).reshape(-1, 2)
    if len(points) < 4:
        raise ValueError("至少需要 4 个圆心构造板区域")
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    distances[distances <= 1e-9] = np.inf
    nearest = float(np.median(np.min(distances, axis=1)))
    radius = max(2, int(round(nearest * float(white_radius_ratio))))
    board_mask = np.zeros((height, width), dtype=np.uint8)
    hull = cv2.convexHull(np.rint(points).astype(np.int32))
    cv2.fillConvexPoly(board_mask, hull, 255)
    margin = max(0, int(edge_margin_px))
    if margin:
        kernel_size = 2 * margin + 1
        board_mask = cv2.erode(
            board_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        )
    white_mask = np.zeros_like(board_mask)
    for point in points:
        cv2.circle(white_mask, tuple(np.rint(point).astype(int)), radius, 255, -1)
    white_mask = cv2.bitwise_and(white_mask, board_mask)
    exclusion_radius = max(radius + 2, int(round(radius * 1.25)))
    white_exclusion = np.zeros_like(board_mask)
    for point in points:
        cv2.circle(white_exclusion, tuple(np.rint(point).astype(int)), exclusion_radius, 255, -1)
    black_mask = cv2.bitwise_and(board_mask, cv2.bitwise_not(white_exclusion))
    return board_mask.astype(bool), white_mask.astype(bool), black_mask.astype(bool), nearest
