from akai_fr import AkaiFr

arm = AkaiFr()
arm.set_speed(50)
tcf = [0, 0, 0, 0, 0, 0.0]
arm.set_tcf(1, tcf)
# ret, tool_pose = arm.get_tool_pose()
# print(f"工具坐标系姿态 [XYZRPY]: {tool_pose}")
# shooting_angle = [-250.4151306152343 , 22.14801216125488, 460.3343505859375 , -180, 0, 90]
shooting_angle = [-117, 240.14801216125488, 200 , -180, 0, 90]
arm.set_tool_pose(shooting_angle)