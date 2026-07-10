#!/home/zhl/fr3env/fr3env/bin/python
"""比赛任务 ROS 启动入口。"""

import rospy

from competition_lib.task_runner import TaskRunner


def main():
    rospy.init_node("competition")
    TaskRunner().run_interactive()


if __name__ == "__main__":
    main()
