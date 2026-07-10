#!/home/zhl/fr3env/fr3env/bin/python
"""图像处理 ROS 节点启动入口。"""

import rospy

from image_process_lib.image_node import ImageProcessor


def main():
    rospy.init_node("image_processor")
    ImageProcessor()
    rospy.spin()


if __name__ == "__main__":
    main()
