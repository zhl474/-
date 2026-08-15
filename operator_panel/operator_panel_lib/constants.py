"""控制台固定路径和白名单定义。"""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PACKAGE_DIR.parent
WORKSPACE_DIR = SRC_DIR.parent
PANEL_CONFIG_PATH = PACKAGE_DIR / "config" / "panel.yaml"

WRITABLE_CONFIG_FILES = {
    "execution": {
        "label": "比赛执行参数",
        "path": SRC_DIR / "competition" / "config" / "execution.yaml",
        "restart_scope": "perception",
        "description": "拍摄位姿、运动速度、抓放高度、视觉伺服和舵机边界",
    },
    "visual_servo": {
        "label": "视觉伺服映射",
        "path": SRC_DIR / "competition" / "config" / "visual_servo.yaml",
        "restart_scope": "perception",
        "description": "像素到机械臂映射矩阵和相机到吸盘偏移",
    },
    "perception": {
        "label": "图像识别参数",
        "path": SRC_DIR / "image_process" / "config" / "perception.yaml",
        "restart_scope": "perception",
        "description": "模型、规划、模板匹配、标定安全范围和低位识别",
    },
    "controller": {
        "label": "机械臂控制参数",
        "path": SRC_DIR / "control" / "config" / "controller.yaml",
        "restart_scope": "hardware",
        "description": "停稳判定、动作耗时日志和软件停止确认",
    },
    "camera": {
        "label": "相机参数",
        "path": SRC_DIR / "camera" / "config" / "新相机参数.yaml",
        "restart_scope": "hardware",
        "description": "彩色相机、深度相机、曝光和图像风格参数",
    },
    "template": {
        "label": "模板与颜色参数",
        "path": SRC_DIR / "competition" / "config" / "template_config.yaml",
        "restart_scope": "perception",
        "description": "高低位模板尺寸和颜色分割先验",
    },
}

READ_ONLY_CONFIG_FILES = {
    "task_layout": {
        "label": "基础任务盘面",
        "path": SRC_DIR / "image_process" / "config" / "task_layout.yaml",
        "kind": "layout",
    },
    "block_calibration": {
        "label": "方块像素标定",
        "path": SRC_DIR / "image_process" / "config" / "block_pixel_to_tcp_calibration.yaml",
        "kind": "calibration",
    },
    "tray_calibration": {
        "label": "托盘像素标定",
        "path": SRC_DIR / "image_process" / "config" / "tray_pixel_to_tcp_calibration.yaml",
        "kind": "calibration",
    },
    "legacy_camera_calibration": {
        "label": "旧相机标定矩阵",
        "path": SRC_DIR / "competition" / "config" / "calibration_matrix.yaml",
        "kind": "calibration",
    },
}

HAND_EYE_MATRIX_PATH = SRC_DIR / "camera" / "config" / "T_wrist2camera.npy"

ARUCO_ALIGN_SCRIPT_PATH = (
    SRC_DIR
    / "tools"
    / "vision"
    / "aruco_pixel_tcp_diagnostic"
    / "align_aruco_once.py"
)
ARUCO_ALIGN_DEFAULT_LOW_TCP_Z_MM = 220.0

DEBUG_IMAGE_FILES = {
    "block_mask": "方块上表面掩码.jpg",
    "template_match": "高位方块模板匹配结果.jpg",
    "board_grid": "托盘格点粗定位.jpg",
}

BLOCK_CATEGORY_NAMES = (
    "L_blue",
    "L_yellow",
    "z_blue",
    "z_green",
    "square",
    "T",
    "line",
)
