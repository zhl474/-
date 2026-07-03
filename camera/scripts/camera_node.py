#!/home/zhl/fr3env/fr3env/bin/python
import rospy
import cv2
import numpy as np
import threading
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from akai_gemini335 import AkaiGemini335
from akai_fr import AkaiFr
from akai import tf3d, MM, M, RAD, DEG
from camera.srv import pixel2world, pixel2worldResponse

class CameraNode:
    def __init__(self):
        rospy.init_node('camera_node')
        
        # 相机初始化
        self.cap = AkaiGemini335(yaml_path="/home/zhl/SingleArmTetris/SingleArmTetris/新相机参数.yaml")
        self.arm = AkaiFr()
        T_wrist2camera_mm = np.load('/home/zhl/SingleArmTetris/SingleArmTetris/测试/Log/T_wrist2camera.npy')
        self.arm.set_tmat_wrist2camera(T_wrist2camera_mm)
        self.tcf_biasx = -9.3
        self.tcf_biasy = -4
        self.tcf_biasz = 169
        # 创建发布者
        self.image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
        self.bridge = CvBridge()
        
        # 线程锁，保护共享数据
        self.lock = threading.Lock()
        self.latest_color_img = None
        self.latest_depth_img = None
        
        # 创建服务
        self.service = rospy.Service("get_world_pos", pixel2world, self.get_world_pos)
        
        rospy.loginfo("Camera node started: Publishing images and ready to provide world position service")
    
    def get_world_pos(self, req):
        """服务回调函数：像素坐标转世界坐标"""
        with self.lock:  # 加锁，确保读取图像时不会被修改
            if self.latest_depth_img is None:
                rospy.logerr("没有能用的深度图")
                return pixel2worldResponse(world_position=[0, 0, 0])
            
            depth_img = self.latest_depth_img.copy()
        
        # 使用深度图像计算世界坐标
        try:
            depth_value = depth_img[req.y, req.x]
            if depth_value == 0: 
                raise ValueError("无效深度0")
            
            # 获取相机位姿
            ret, pose_base2camera = self.arm.get_camera_pose()
            T_base2camera_mm = tf3d.XYZRPY2TransformMatrix(pose_base2camera, xyz_unit=MM, rpy_unit=DEG, T_unit=MM)
            
            # 像素坐标转世界坐标
            target = self.cap.depth_pixel2cam_point3d(req.x, req.y, depth_value=depth_value)
            base = tf3d.VectorTransform(T_base2camera_mm, target)
            base = [base[0]+self.tcf_biasx,base[1]+self.tcf_biasy,base[2]+self.tcf_biasz]
            return pixel2worldResponse(world_position=base)
            
        except Exception as e:
            rospy.logerr(f"Error in get_world_pos: {e}")
            return pixel2worldResponse(world_position=[0, 0, 0])
    
    def publish_images(self):
        """发布图像的主循环"""
        rate = rospy.Rate(5)  # 30Hz
        
        while not rospy.is_shutdown():
            try:
                # 读取图像
                color_img, depth_img = self.cap.read()
                
                if color_img is None or depth_img is None:
                    rospy.logwarn("获取图像失败")
                    continue
                
                # 更新共享图像（加锁保护）
                with self.lock:
                    self.latest_color_img = color_img.copy()
                    self.latest_depth_img = depth_img.copy()
                
                # 发布彩色图像
                ros_image = self.bridge.cv2_to_imgmsg(color_img, encoding="bgr8")
                ros_image.header.stamp = rospy.Time.now()
                ros_image.header.frame_id = "camera_frame"
                self.image_pub.publish(ros_image)
                
                # 可以同时发布深度图像（如果需要）
                # depth_msg = self.bridge.cv2_to_imgmsg(depth_img, encoding="32FC1")
                # depth_msg.header.stamp = ros_image.header.stamp
                # depth_msg.header.frame_id = "camera_frame"
                # depth_pub.publish(depth_msg)
                
            except Exception as e:
                rospy.logerr(f"Error publishing image: {e}")
            
            rate.sleep()
    
    def run(self):
        """运行节点"""
        # 启动图像发布线程
        pub_thread = threading.Thread(target=self.publish_images, daemon=True)
        pub_thread.start()
        
        # 主线程保持运行，处理服务请求
        rospy.spin()  # 这会阻塞，直到节点关闭
        
        # 清理资源
        self.cap.release()

if __name__ == '__main__':
    try:
        node = CameraNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node interrupted")
    except Exception as e:
        rospy.logerr(f"Node failed: {e}")