#!/home/zhl/fr3env/fr3env/bin/python
"""彩色图像发布、单点高度和批量稳定世界坐标查询节点。"""

import os
import threading
import time

import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from akai import DEG, MM, tf3d
from akai_fr import AkaiFr
from akai_gemini335 import AkaiGemini335
from camera.srv import (
    GetStableWorldPoints,
    GetStableWorldPointsResponse,
    GetSurfaceHeight,
    GetSurfaceHeightResponse,
)


PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CAMERA_CONFIG = os.path.join(PACKAGE_DIR, "config", "新相机参数.yaml")
DEFAULT_HAND_EYE_MATRIX = os.path.join(PACKAGE_DIR, "config", "T_wrist2camera.npy")


class CameraNode:
    def __init__(self):
        camera_config = rospy.get_param("~camera_config", DEFAULT_CAMERA_CONFIG)
        self.hand_eye_matrix_path = rospy.get_param("~hand_eye_matrix", DEFAULT_HAND_EYE_MATRIX)
        self.cap = AkaiGemini335(yaml_path=camera_config)
        self.world_bias_mm = np.asarray(rospy.get_param("~world_bias_mm", [0.0, 0.0, 0.0]), dtype=float)
        if self.world_bias_mm.shape != (3,) or not np.all(np.isfinite(self.world_bias_mm)):
            raise ValueError("world_bias_mm 必须包含 3 个有限数值")
        self.depth_max_age_sec = float(rospy.get_param("~depth_max_age_sec", 0.5))
        if not np.isfinite(self.depth_max_age_sec) or self.depth_max_age_sec <= 0.0:
            raise ValueError("depth_max_age_sec 必须是大于 0 的有限数值")

        self.image_pub = rospy.Publisher("/camera/image_raw", Image, queue_size=1)
        self.bridge = CvBridge()
        self.depth_lock = threading.Lock()
        self.depth_condition = threading.Condition(self.depth_lock)
        self.latest_depth_image = None
        self.latest_depth_monotonic = None
        # 批量请求只登记自己的采集器，不保存请求前的历史深度帧。
        self.depth_collectors = []

        # 机械臂连接和手眼矩阵只在首次有效高度查询时初始化。
        self.arm = None
        self.arm_init_lock = threading.Lock()
        self.service = rospy.Service(
            "/camera/surface_height",
            GetSurfaceHeight,
            self.get_surface_height,
        )
        self.stable_world_points_service = rospy.Service(
            "/camera/stable_world_points",
            GetStableWorldPoints,
            self.get_stable_world_points,
        )
        rospy.on_shutdown(self.close)
        rospy.loginfo("相机节点已启动")

    @staticmethod
    def _failure(message):
        rospy.logerr(message)
        return GetSurfaceHeightResponse(
            success=False,
            surface_z_mm=0.0,
            message=str(message),
        )

    def _ensure_arm_initialized(self):
        """线程安全地懒初始化机械臂及手眼标定。"""
        if self.arm is not None:
            return self.arm
        with self.arm_init_lock:
            if self.arm is not None:
                return self.arm
            wrist_to_camera = np.load(self.hand_eye_matrix_path)
            if wrist_to_camera.shape != (4, 4) or not np.all(np.isfinite(wrist_to_camera)):
                raise ValueError("手眼标定矩阵必须是有限的 4x4 矩阵")
            arm = AkaiFr()
            arm.set_tmat_wrist2camera(wrist_to_camera)
            # 全部配置成功后再公布实例，初始化失败时后续请求可重试。
            self.arm = arm
            return self.arm

    def _snapshot_fresh_depth(self):
        """复制一帧未过期的深度图，避免服务处理期间被发布线程替换。"""
        with self.depth_lock:
            if self.latest_depth_image is None:
                raise RuntimeError("没有可用深度图")
            if self.latest_depth_monotonic is None:
                raise RuntimeError("深度图缺少采集时间戳")
            depth_image = self.latest_depth_image.copy()
            depth_stamp = float(self.latest_depth_monotonic)

        depth_age_sec = time.monotonic() - depth_stamp
        if not np.isfinite(depth_age_sec) or depth_age_sec < 0.0:
            raise RuntimeError("深度图时间戳无效")
        if depth_age_sec > self.depth_max_age_sec:
            raise RuntimeError(
                f"深度图已过期：时龄 {depth_age_sec:.3f} 秒，"
                f"最大允许 {self.depth_max_age_sec:.3f} 秒"
            )
        return depth_image

    def _collect_new_depth_frames(self, frame_count, capture_timeout_sec):
        """等待并返回请求登记之后产生的指定数量深度帧。"""
        count = int(frame_count)
        timeout_sec = float(capture_timeout_sec)
        if count <= 0:
            raise ValueError("frame_count 必须是大于 0 的整数")
        if not np.isfinite(timeout_sec) or timeout_sec <= 0.0:
            raise ValueError("capture_timeout_sec 必须是大于 0 的有限数值")

        registered_monotonic = time.monotonic()
        collector = {
            "frame_count": count,
            "frames": [],
            "registered_monotonic": registered_monotonic,
        }
        deadline = registered_monotonic + timeout_sec
        with self.depth_condition:
            self.depth_collectors.append(collector)
            try:
                while len(collector["frames"]) < count:
                    remaining_sec = deadline - time.monotonic()
                    if remaining_sec <= 0.0:
                        actual_count = len(collector["frames"])
                        raise RuntimeError(
                            f"新深度帧不足：实际 {actual_count}/{count}"
                        )
                    self.depth_condition.wait(timeout=remaining_sec)
                return list(collector["frames"])
            finally:
                self.depth_collectors.remove(collector)

    def _cache_depth_frame(self, depth_image, captured_monotonic=None):
        """缓存最新帧，并把该新帧投递给当前已经登记的批量采集器。"""
        depth_copy = np.asarray(depth_image).copy()
        depth_stamp = (
            time.monotonic()
            if captured_monotonic is None
            else float(captured_monotonic)
        )
        if not np.isfinite(depth_stamp):
            raise ValueError("深度帧采集时间戳不是有限数值")
        with self.depth_condition:
            self.latest_depth_image = depth_copy
            self.latest_depth_monotonic = depth_stamp
            delivered = False
            for collector in self.depth_collectors:
                if (
                    depth_stamp > collector["registered_monotonic"]
                    and len(collector["frames"]) < collector["frame_count"]
                ):
                    collector["frames"].append(depth_copy)
                    delivered = True
            if delivered:
                self.depth_condition.notify_all()

    def _clear_depth_cache(self):
        """使旧单帧深度失效；已登记的批量采集器继续等待后续新帧。"""
        with self.depth_condition:
            self.latest_depth_image = None
            self.latest_depth_monotonic = None
            self.depth_condition.notify_all()

    @staticmethod
    def _median_depth_in_clipped_neighborhood(depth_image, x, y):
        """在裁剪到图像范围的 3x3 邻域中计算有效深度中位数。"""
        if not isinstance(depth_image, np.ndarray) or depth_image.ndim != 2:
            raise ValueError("深度图必须是二维数组")
        height, width = depth_image.shape
        if height <= 0 or width <= 0:
            raise ValueError("深度图尺寸无效")
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"像素坐标越界: ({x}, {y})，图像尺寸=({width}, {height})")

        x_start, x_end = max(0, x - 1), min(width, x + 2)
        y_start, y_end = max(0, y - 1), min(height, y + 2)
        neighborhood = np.asarray(
            depth_image[y_start:y_end, x_start:x_end],
            dtype=float,
        )
        valid_depths = neighborhood[np.isfinite(neighborhood) & (neighborhood > 0.0)]
        if valid_depths.size == 0:
            raise ValueError(f"像素 ({x}, {y}) 的 3x3 邻域没有有效深度值")
        return float(np.median(valid_depths))

    def get_surface_height(self, request):
        """用对齐深度返回目标点在机器人基坐标系下的绝对 Z。"""
        try:
            x, y = int(request.x), int(request.y)
            depth_image = self._snapshot_fresh_depth()
            depth_value = self._median_depth_in_clipped_neighborhood(depth_image, x, y)

            arm = self._ensure_arm_initialized()
            success, base_to_camera_pose = arm.get_camera_pose()
            try:
                pose = np.asarray(base_to_camera_pose, dtype=float)
            except (TypeError, ValueError):
                pose = np.asarray([], dtype=float)
            if not success or pose.shape != (6,) or not np.all(np.isfinite(pose)):
                raise RuntimeError("获取相机位姿失败")
            base_to_camera = tf3d.XYZRPY2TransformMatrix(
                pose,
                xyz_unit=MM,
                rpy_unit=DEG,
                T_unit=MM,
            )
            camera_point = np.asarray(
                self.cap.depth_pixel2cam_point3d(x, y, depth_value=depth_value),
                dtype=float,
            )
            if camera_point.shape != (3,) or not np.all(np.isfinite(camera_point)):
                raise RuntimeError("深度像素转相机坐标失败")
            world_point = np.asarray(tf3d.VectorTransform(base_to_camera, camera_point), dtype=float)
            world_point += self.world_bias_mm
            if world_point.shape != (3,) or not np.all(np.isfinite(world_point)):
                raise RuntimeError(f"转换后的世界坐标无效: {world_point}")
            return GetSurfaceHeightResponse(
                success=True,
                surface_z_mm=float(world_point[2]),
                message="表面绝对高度查询成功",
            )
        except Exception as exc:
            return self._failure(f"表面高度查询失败: {exc}")

    @staticmethod
    def _stable_failure(message):
        rospy.logerr(message)
        return GetStableWorldPointsResponse(
            success=False,
            point_valid=[],
            world_xyz=[],
            valid_frame_counts=[],
            depth_median_mm=[],
            depth_mad_mm=[],
            message=str(message),
        )

    def get_stable_world_points(self, request):
        """用同一批深度帧和同一次相机位姿返回多个表面点的基坐标 XYZ。"""
        try:
            xs = [int(value) for value in request.x]
            ys = [int(value) for value in request.y]
            if not xs or len(xs) != len(ys):
                raise ValueError("x 和 y 必须是长度相同的非空像素数组")
            frame_count = int(request.frame_count)
            min_valid_frames = int(request.min_valid_frames)
            capture_timeout_sec = float(request.capture_timeout_sec)
            if min_valid_frames <= 0 or min_valid_frames > frame_count:
                raise ValueError("min_valid_frames 必须在 1 到 frame_count 之间")
            frames = self._collect_new_depth_frames(
                frame_count,
                capture_timeout_sec,
            )

            arm = self._ensure_arm_initialized()
            success, base_to_camera_pose = arm.get_camera_pose()
            try:
                pose = np.asarray(base_to_camera_pose, dtype=float)
            except (TypeError, ValueError):
                pose = np.asarray([], dtype=float)
            if not success or pose.shape != (6,) or not np.all(np.isfinite(pose)):
                raise RuntimeError("获取相机位姿失败")
            base_to_camera = tf3d.XYZRPY2TransformMatrix(
                pose,
                xyz_unit=MM,
                rpy_unit=DEG,
                T_unit=MM,
            )

            point_valid = []
            world_xyz = []
            valid_counts = []
            depth_medians = []
            depth_mads = []
            for x, y in zip(xs, ys):
                depths = []
                for frame in frames:
                    try:
                        depths.append(
                            self._median_depth_in_clipped_neighborhood(frame, x, y)
                        )
                    except ValueError:
                        continue
                valid_count = len(depths)
                valid_counts.append(valid_count)
                if depths:
                    depth_array = np.asarray(depths, dtype=float)
                    depth_median = float(np.median(depth_array))
                    depth_mad = float(np.median(np.abs(depth_array - depth_median)))
                else:
                    depth_median = 0.0
                    depth_mad = 0.0
                depth_medians.append(depth_median)
                depth_mads.append(depth_mad)

                if valid_count < min_valid_frames:
                    point_valid.append(False)
                    world_xyz.extend([0.0, 0.0, 0.0])
                    continue

                camera_point = np.asarray(
                    self.cap.depth_pixel2cam_point3d(
                        x,
                        y,
                        depth_value=depth_median,
                    ),
                    dtype=float,
                )
                if camera_point.shape != (3,) or not np.all(np.isfinite(camera_point)):
                    point_valid.append(False)
                    world_xyz.extend([0.0, 0.0, 0.0])
                    continue
                world_point = np.asarray(
                    tf3d.VectorTransform(base_to_camera, camera_point),
                    dtype=float,
                )
                world_point += self.world_bias_mm
                if world_point.shape != (3,) or not np.all(np.isfinite(world_point)):
                    point_valid.append(False)
                    world_xyz.extend([0.0, 0.0, 0.0])
                    continue
                point_valid.append(True)
                world_xyz.extend(float(value) for value in world_point)

            return GetStableWorldPointsResponse(
                success=True,
                point_valid=point_valid,
                world_xyz=world_xyz,
                valid_frame_counts=valid_counts,
                depth_median_mm=depth_medians,
                depth_mad_mm=depth_mads,
                message="批量稳定世界坐标查询成功",
            )
        except Exception as exc:
            return self._stable_failure(f"批量稳定世界坐标查询失败: {exc}")

    def publish_images(self):
        rate = rospy.Rate(60)
        while not rospy.is_shutdown():
            try:
                color_image, depth_image = self.cap.read()
                captured_monotonic = time.monotonic()
            except Exception as exc:
                rospy.logerr("读取相机图像失败: %s", exc)
                rate.sleep()
                continue

            # 深度帧缺失或异常不影响同次彩色图像发布，但必须让旧深度立即失效。
            if depth_image is None:
                self._clear_depth_cache()
            else:
                try:
                    self._cache_depth_frame(depth_image, captured_monotonic)
                except Exception as exc:
                    self._clear_depth_cache()
                    rospy.logwarn("缓存深度图失败，本帧仍发布彩色图像: %s", exc)

            if color_image is None:
                rospy.logwarn("本次相机读取没有彩色图像")
            else:
                try:
                    message = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
                    message.header.stamp = rospy.Time.now()
                    message.header.frame_id = "camera_frame"
                    self.image_pub.publish(message)
                except Exception as exc:
                    rospy.logerr("发布彩色图像失败: %s", exc)
            rate.sleep()

    def close(self):
        if getattr(self, "cap", None) is not None:
            self.cap.release()

    def run(self):
        publish_thread = threading.Thread(target=self.publish_images, daemon=True)
        publish_thread.start()
        rospy.spin()


def main():
    rospy.init_node("camera_node")
    CameraNode().run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
