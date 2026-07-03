from akai_fr import AkaiFr

arm = AkaiFr()
arm.set_speed(50)
tcf = [0, 0, 0, 0, 0, 0.0]
arm.set_tcf(1, tcf)
shooting_angle = [-280.59,25.669,400.595,-180,0,90]
# shooting_angle = [-238.59738165128698, 310.19724317891456, 167.79473989372167,-180,0,90]
arm.set_tool_pose(shooting_angle)