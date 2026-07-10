#!/home/zhl/fr3env/fr3env/bin/python
"""彩色/深度相机发布与像素转世界坐标节点。"""

import os
import threading

import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from akai import DEG, MM, tf3d
from akai_fr import AkaiFr
from akai_gemini335 import AkaiGemini335
from camera.srv import PixelToWorld, PixelToWorldResponse


PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CAMERA_CONFIG = os.path.join(PACKAGE_DIR, "config", "新相机参数.yaml")
DEFAULT_HAND_EYE_MATRIX = os.path.join(PACKAGE_DIR, "config", "T_wrist2camera.npy")


class CameraNode:
    def __init__(self):
        camera_config = rospy.get_param("~camera_config", DEFAULT_CAMERA_CONFIG)
        hand_eye_matrix = rospy.get_param("~hand_eye_matrix", DEFAULT_HAND_EYE_MATRIX)
        self.cap = AkaiGemini335(yaml_path=camera_config)
        self.arm = AkaiFr()
        wrist_to_camera = np.load(hand_eye_matrix)
        if wrist_to_camera.shape != (4, 4) or not np.all(np.isfinite(wrist_to_camera)):
            raise ValueError("手眼标定矩阵必须是有限的 4x4 矩阵")
        self.arm.set_tmat_wrist2camera(wrist_to_camera)
        self.world_bias_mm = np.asarray(rospy.get_param("~world_bias_mm", [0.0, 0.0, 0.0]), dtype=float)
        if self.world_bias_mm.shape != (3,) or not np.all(np.isfinite(self.world_bias_mm)):
            raise ValueError("world_bias_mm 必须包含 3 个有限数值")

        self.image_pub = rospy.Publisher("/camera/image_raw", Image, queue_size=10)
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_depth_image = None
        self.service = rospy.Service("/camera/pixel_to_world", PixelToWorld, self.pixel_to_world)
        rospy.on_shutdown(self.close)
        rospy.loginfo("相机节点已启动")

    @staticmethod
    def _failure(message):
        rospy.logerr(message)
        return PixelToWorldResponse(success=False, world_position=[0.0, 0.0, 0.0], message=str(message))

    def pixel_to_world(self, request):
        """把彩色图像像素及对齐深度转换到机器人基坐标系。"""
        with self.lock:
            if self.latest_depth_image is None:
                return self._failure("没有可用深度图")
            depth_image = self.latest_depth_image.copy()

        try:
            height, width = depth_image.shape[:2]
            x, y = int(request.x), int(request.y)
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"像素坐标越界: ({x}, {y})，图像尺寸=({width}, {height})")
            depth_value = float(depth_image[y, x])
            if not np.isfinite(depth_value) or depth_value <= 0:
                raise ValueError(f"像素深度无效: {depth_value}")

            success, base_to_camera_pose = self.arm.get_camera_pose()
            if not success or base_to_camera_pose is None or not np.all(np.isfinite(base_to_camera_pose)):
                raise RuntimeError("获取相机位姿失败")
            base_to_camera = tf3d.XYZRPY2TransformMatrix(
                base_to_camera_pose,
                xyz_unit=MM,
                rpy_unit=DEG,
                T_unit=MM,
            )
            camera_point = self.cap.depth_pixel2cam_point3d(x, y, depth_value=depth_value)
            world_point = np.asarray(tf3d.VectorTransform(base_to_camera, camera_point), dtype=float)
            world_point += self.world_bias_mm
            if world_point.shape != (3,) or not np.all(np.isfinite(world_point)):
                raise RuntimeError(f"转换后的世界坐标无效: {world_point}")
            return PixelToWorldResponse(
                success=True,
                world_position=world_point.tolist(),
                message="像素转世界坐标成功",
            )
        except Exception as exc:
            return self._failure(f"像素转世界坐标失败: {exc}")

    def publish_images(self):
        rate = rospy.Rate(60)
        while not rospy.is_shutdown():
            try:
                color_image, depth_image = self.cap.read()
                if color_image is None or depth_image is None:
                    rospy.logwarn("相机读取失败")
                    rate.sleep()
                    continue
                with self.lock:
                    self.latest_depth_image = depth_image.copy()
                message = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
                message.header.stamp = rospy.Time.now()
                message.header.frame_id = "camera_frame"
                self.image_pub.publish(message)
            except Exception as exc:
                rospy.logerr("发布相机图像失败: %s", exc)
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
