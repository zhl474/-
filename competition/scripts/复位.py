from akai_fr import AkaiFr

arm = AkaiFr()
arm.set_speed(50)
tcf = [0, 0, 0, 0, 0, 0.0]
arm.set_tcf(1, tcf)
shooting_angle = [-475.0334849461607,-156.55351754676806,180.83411546358315 , -180, 0, 90]
arm.set_tool_pose(shooting_angle)