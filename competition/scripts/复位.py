from akai_fr import AkaiFr

arm = AkaiFr()
arm.set_speed(50)
tcf = [0, 0, 0, 0, 0, 0.0]
arm.set_tcf(1, tcf)
# ret, tool_pose = arm.get_tool_pose()
# print(f"工具坐标系姿态 [XYZRPY]: {tool_pose}")
shooting_angle = [-9,-351.252,320.273, 180, 0, -170]
arm.set_tool_pose(shooting_angle)