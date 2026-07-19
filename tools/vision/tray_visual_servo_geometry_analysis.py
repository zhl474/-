#!/home/zhl/fr3env/fr3env/bin/python
# -*- coding: utf-8 -*-
"""托盘高位检测像素到伺服成功 TCP XYZ 的标定与几何诊断入口。"""

from __future__ import annotations

from types import SimpleNamespace

try:
    # 以包方式导入时使用相对路径，便于测试代码直接导入。
    from .visual_servo_geometry_analysis import TRAY_ANALYSIS_SPEC, run_geometry_analysis
except ImportError:
    # 直接运行本文件时，Python 会把本目录加入模块搜索路径。
    from visual_servo_geometry_analysis import TRAY_ANALYSIS_SPEC, run_geometry_analysis


# ----------------------------- 直接运行配置 -----------------------------
# 修改本区参数后，直接运行本文件即可，不需要填写命令行参数。
INPUT_CSV = "/home/zhl/桌面/logs/托盘视觉伺服.csv"
OUTPUT_DIR = "tray_visual_servo_geometry_results"
EXPECTED_SUCCESS_COUNT = 34

PLANE_RMSE_TOL = None
PLANE_MAX_TOL = None
# 与方块分析一致：最小主轴/次小主轴不超过 1% 时，给出近似平面的通过判定。
PLANARITY_RATIO_TOL = 0.01
PARALLEL_ANGLE_TOL_DEG = 1.0
# 若高位世界坐标系与 TCP 坐标系轴方向不一致，在此填写 world->TCP 的 3×3 旋转矩阵。
WORLD_TO_TCP_ROTATION = None

UNIQUE_VALUE_TOL = 1e-9
CV_FOLDS = 5
RANDOM_SEED = 42
SIMPLE_MODEL_SLACK = 0.05
SKIP_PLOTS = False


def build_direct_args() -> SimpleNamespace:
    """将顶部的直接运行配置转换为共享分析核心需要的参数对象。"""
    return SimpleNamespace(
        input_csv=INPUT_CSV,
        output_dir=OUTPUT_DIR,
        expected_count=EXPECTED_SUCCESS_COUNT,
        plane_rmse_tol=PLANE_RMSE_TOL,
        plane_max_tol=PLANE_MAX_TOL,
        planarity_ratio_tol=PLANARITY_RATIO_TOL,
        parallel_angle_tol_deg=PARALLEL_ANGLE_TOL_DEG,
        world_to_tcp_rotation=WORLD_TO_TCP_ROTATION,
        unique_value_tol=UNIQUE_VALUE_TOL,
        cv_folds=CV_FOLDS,
        random_seed=RANDOM_SEED,
        simple_model_slack=SIMPLE_MODEL_SLACK,
        skip_plots=SKIP_PLOTS,
    )


def main() -> None:
    run_geometry_analysis(build_direct_args(), TRAY_ANALYSIS_SPEC)


if __name__ == "__main__":
    main()
