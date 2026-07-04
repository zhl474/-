from akai_fr import AkaiFr

arm = AkaiFr()
arm.set_speed(50)
tcf = [0, 0, 0, 0, 0, 0.0]
arm.set_tcf(1, tcf)
shooting_angle = [-250.4151306152343 , 22.14801216125488, 380.3343505859375 , -180, 0, 90]
arm.set_tool_pose(shooting_angle)