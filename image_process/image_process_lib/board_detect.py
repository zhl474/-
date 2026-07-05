import cv2
import numpy as np
from ultralytics import YOLO


BOARD_MODEL_PATH = "/home/zhl/SingleArmTetris/SingleArmTetris/src/competition/model/best.pt"
BOARD_ROW_COUNT = 14
BOARD_COL_COUNT = 10
BOARD_POINT_COUNT = BOARD_ROW_COUNT * BOARD_COL_COUNT
DEFAULT_DEBUG_PATH = "/home/zhl/桌面/托盘格点识别.jpg"

_board_model = None


class BoardGridDetectionError(RuntimeError):
    """托盘格点识别失败。"""


def _get_board_model():
    """延迟加载托盘模型，避免每次服务调用都重新读取权重。"""
    global _board_model
    if _board_model is None:
        _board_model = YOLO(BOARD_MODEL_PATH)
    return _board_model


def _save_debug_image(image_path, image):
    """保存调试图像，兼容中文路径。"""
    if not image_path or image is None:
        return False
    try:
        dot_index = image_path.rfind(".")
        image_ext = image_path[dot_index:] if dot_index >= 0 else ".jpg"
        ok, encoded_image = cv2.imencode(image_ext, image)
        if not ok:
            return False
        encoded_image.tofile(image_path)
        return True
    except Exception:
        return False


def _red_error(message):
    """用红色输出现场必须处理的托盘识别错误。"""
    print(f"\033[91m{message}\033[0m")


def _make_blob_detector():
    """创建托盘格点 blob 检测器，用于检测白色圆点。"""
    params = cv2.SimpleBlobDetector_Params()

    # 二值图里格点颜色
    params.filterByColor = True
    params.blobColor = 0

    # 面积过滤
    params.filterByArea = True
    params.minArea = 25
    params.maxArea = 200

    # 圆度过滤
    params.filterByCircularity = True
    params.minCircularity = 0.3

    # 这两个默认可能会误杀一些不完美圆点，建议关掉
    params.filterByInertia = False
    params.filterByConvexity = False

    # 点间距明显大于 10，这个可以保留
    params.minDistBetweenBlobs = 8

    return cv2.SimpleBlobDetector_create(params)


def _get_board_crop(img):
    """用 YOLO OBB 找托盘，并透视裁剪到托盘局部图。"""
    model = _get_board_model()
    results = model(img)
    if not results:
        raise BoardGridDetectionError("未检测到托盘")

    result = results[0]
    if result.obb is None or len(result.obb) == 0:
        raise BoardGridDetectionError("未检测到托盘 OBB")

    obb_points = result.obb.xyxyxyxy.cpu().numpy()[0]
    pts = obb_points.reshape(4, 2).astype(np.float32)

    width = int(
        max(
            np.linalg.norm(pts[0] - pts[1]),
            np.linalg.norm(pts[2] - pts[3]),
        )
    )
    height = int(
        max(
            np.linalg.norm(pts[1] - pts[2]),
            np.linalg.norm(pts[3] - pts[0]),
        )
    )
    if width <= 1 or height <= 1:
        raise BoardGridDetectionError("托盘 OBB 尺寸异常")

    dst_pts = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(pts, dst_pts)
    crop_img = cv2.warpPerspective(img, matrix, (width, height))
    return crop_img, matrix


def _detect_grid_keypoints(crop_img):
    """在托盘裁剪图中检测 140 个圆形格点。"""
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    # cv2.imshow("1",blackhat)
    _, threshold_img = cv2.threshold(
        blackhat,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    # cv2.imshow("2",threshold_img)
    detector = _make_blob_detector()
    keypoints = detector.detect(gray)
    debug_image = cv2.drawKeypoints(
        crop_img,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return keypoints, debug_image, threshold_img


def _transform_points_to_origin(points, matrix):
    """把裁剪图格点坐标反变换回原始图像坐标。"""
    matrix_inv = np.linalg.inv(matrix)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    orig_points = cv2.perspectiveTransform(points, matrix_inv)
    return orig_points.reshape(-1, 2)


def _sort_points_to_grid(crop_points, orig_points):
    """把 140 个格点排序成 grid_points[row][col] 的 1-based 表。"""
    points = np.asarray(crop_points, dtype=np.float32)
    if len(points) != BOARD_POINT_COUNT:
        raise BoardGridDetectionError("托盘格点数量不是140个，无法排序")

    mean, eigenvectors = cv2.PCACompute(points, mean=None)
    center = mean[0]
    axis_a = eigenvectors[0]
    axis_b = eigenvectors[1]
    proj_a = np.dot(points - center, axis_a)
    proj_b = np.dot(points - center, axis_b)

    # 14 行方向通常是托盘长轴；用投影范围选择行轴。
    extent_a = float(np.max(proj_a) - np.min(proj_a))
    extent_b = float(np.max(proj_b) - np.min(proj_b))
    if extent_a >= extent_b:
        row_proj, col_proj = proj_a, proj_b
    else:
        row_proj, col_proj = proj_b, proj_a

    sorted_indices = np.argsort(row_proj)
    row_groups = [
        sorted_indices[i * BOARD_COL_COUNT:(i + 1) * BOARD_COL_COUNT]
        for i in range(BOARD_ROW_COUNT)
    ]

    # row=1 约定为画面中更靠下的一行，也就是左下角原点。
    first_row_y = float(np.mean(orig_points[row_groups[0], 1]))
    last_row_y = float(np.mean(orig_points[row_groups[-1], 1]))

    # 图像坐标系 y 越大越靠下。
    # 如果当前第一组在上面，就反转，让最下面那一行排到 row=1。
    if first_row_y < last_row_y:
        row_groups.reverse()

    grid_points = [[None for _ in range(BOARD_COL_COUNT + 1)] for _ in range(BOARD_ROW_COUNT + 1)]
    ordered_rows = []
    for row_indices in row_groups:
        row_indices = np.array(row_indices, dtype=np.int32)
        col_sorted = row_indices[np.argsort(col_proj[row_indices])]

        # col=1 约定为画面中更靠左的一列。
        if float(orig_points[col_sorted[0], 0]) > float(orig_points[col_sorted[-1], 0]):
            col_sorted = col_sorted[::-1]
        ordered_rows.append(col_sorted)

    for row_idx, col_sorted in enumerate(ordered_rows, start=1):
        for col_idx, point_index in enumerate(col_sorted, start=1):
            px, py = orig_points[point_index]
            grid_points[row_idx][col_idx] = np.array([float(px), float(py)], dtype=np.float32)

    return grid_points


def interpolate_grid_point(grid_points, row, col):
    """按 1-based 行列读取托盘点；小数行列用相邻真实格点双线性插值。"""
    row = float(row)
    col = float(col)
    if row < 1.0 or row > BOARD_ROW_COUNT or col < 1.0 or col > BOARD_COL_COUNT:
        raise ValueError(f"托盘目标超出范围: row={row}, col={col}")

    row0 = int(np.floor(row))
    row1 = int(np.ceil(row))
    col0 = int(np.floor(col))
    col1 = int(np.ceil(col))
    row_t = row - row0
    col_t = col - col0

    p00 = grid_points[row0][col0]
    p01 = grid_points[row0][col1]
    p10 = grid_points[row1][col0]
    p11 = grid_points[row1][col1]
    top = p00 * (1.0 - col_t) + p01 * col_t
    bottom = p10 * (1.0 - col_t) + p11 * col_t
    return top * (1.0 - row_t) + bottom * row_t


def draw_grid_debug(image, grid_points, target_point=None, center_point=None):
    """在原图上画出 140 个格点、目标点和相机中心，便于现场确认。"""
    debug_image = image.copy()
    for row in range(1, BOARD_ROW_COUNT + 1):
        for col in range(1, BOARD_COL_COUNT + 1):
            px, py = grid_points[row][col]
            cv2.circle(debug_image, (int(round(px)), int(round(py))), 3, (0, 255, 0), -1)
            if row in (1, BOARD_ROW_COUNT) and col in (1, BOARD_COL_COUNT):
                cv2.putText(
                    debug_image,
                    f"{row},{col}",
                    (int(round(px)) + 4, int(round(py)) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

    if center_point is not None:
        cx, cy = center_point
        cv2.drawMarker(
            debug_image,
            (int(round(cx)), int(round(cy))),
            (255, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )

    if target_point is not None:
        px, py = target_point
        cv2.drawMarker(
            debug_image,
            (int(round(px)), int(round(py))),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )
        if center_point is not None:
            cv2.line(
                debug_image,
                (int(round(center_point[0])), int(round(center_point[1]))),
                (int(round(px)), int(round(py))),
                (255, 0, 0),
                1,
            )
    return debug_image


def board_grid_detect(img, debug_path=DEFAULT_DEBUG_PATH):
    """识别完整托盘 140 个格点，返回 1-based 行列表。"""
    if img is None:
        return {
            "found": False,
            "grid_points": None,
            "debug_image": None,
            "message": "没有可用图像",
            "count": 0,
        }

    try:
        crop_img, matrix = _get_board_crop(img)
        keypoints, crop_debug_image, _ = _detect_grid_keypoints(crop_img)
        point_count = len(keypoints)
        _save_debug_image(debug_path, crop_debug_image)

        if point_count != BOARD_POINT_COUNT:
            message = f"托盘格点识别数量不是140个，请重新识别；当前识别到{point_count}个"
            _red_error(message)
            return {
                "found": False,
                "grid_points": None,
                "debug_image": crop_debug_image,
                "message": message,
                "count": point_count,
            }

        crop_points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        orig_points = _transform_points_to_origin(crop_points, matrix)
        grid_points = _sort_points_to_grid(crop_points, orig_points)
        debug_image = draw_grid_debug(img, grid_points)
        _save_debug_image(debug_path, debug_image)
        return {
            "found": True,
            "grid_points": grid_points,
            "debug_image": debug_image,
            "message": "托盘140个格点识别成功",
            "count": point_count,
        }
    except Exception as exc:
        message = f"托盘格点识别失败: {exc}"
        _red_error(message)
        _save_debug_image(debug_path, img)
        return {
            "found": False,
            "grid_points": None,
            "debug_image": img.copy() if img is not None else None,
            "message": message,
            "count": 0,
        }


def board_detect(img):
    """兼容旧接口：从完整 140 格点表中取四角返回。"""
    result = board_grid_detect(img)
    if not result["found"]:
        raise BoardGridDetectionError(result["message"])

    grid_points = result["grid_points"]

    # 新坐标系：
    # [1,1] 是左下角
    # [BOARD_ROW_COUNT,1] 是左上角
    # [1,BOARD_COL_COUNT] 是右下角
    # [BOARD_ROW_COUNT,BOARD_COL_COUNT] 是右上角
    left_bottom = grid_points[1][1]
    left_top = grid_points[BOARD_ROW_COUNT][1]
    right_bottom = grid_points[1][BOARD_COL_COUNT]
    right_top = grid_points[BOARD_ROW_COUNT][BOARD_COL_COUNT]

    return np.array([left_top, left_bottom, right_top, right_bottom], dtype=np.float32)


if __name__ == "__main__":
    image_path = "/home/zhl/图片/数据集/15_Color.png"
    image = cv2.imread(image_path)
    detect_result = board_grid_detect(image)
    print(detect_result["grid_points"])
