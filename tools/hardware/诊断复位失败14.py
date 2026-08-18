#!/home/zhl/fr3env/fr3env/bin/python
# -*- coding: utf-8 -*-
"""只读诊断 MoveL 返回 14（ERROR_JOINT_CMD_POINT_ERROR 关节命令点错误）。

不会让机械臂运动，只读三类状态：
  1) 控制器告警（有未清除告警时任何运动指令都会被拒）
  2) 当前关节角 / TCP 位姿（当前构型是否接近关节限位）
  3) 复位目标点的逆解（GetInverseKin，MoveL 内部同款预检）

运行：/home/zhl/fr3env/fr3env/bin/python 诊断复位失败14.py
注意：可与 competition launch 并存（第二条 XML-RPC 只读连接），但别在机械臂
运动中跑。
"""

from akai_fr import AkaiFr

# ==== 参数（直接改这里）====
# 复位目标 = execution.yaml 的 shooting_pose
TARGET_POSE = [-250.4151306152343, 22.14801216125488, 380.3343505859375, -180, 0, 90]


def fmt(label, ret):
    if isinstance(ret, tuple):
        code, data = ret[0], list(ret[1:])
        name = AkaiFr.error_code_dict.get(code, "")
        print(f"{label}: 错误码 {code} {name}, 数据 {data}")
    else:
        name = AkaiFr.error_code_dict.get(ret, "")
        print(f"{label}: 错误码 {ret} {name}")


def main():
    arm = AkaiFr()
    rpc = arm.arm

    print("== 1. 控制器告警 ==")
    fmt("GetRobotErrorCode", rpc.GetRobotErrorCode())
    fmt("GetSafetyCode", rpc.GetSafetyCode())

    print("\n== 2. 当前位姿 ==")
    fmt("关节角(°)", rpc.GetActualJointPosDegree())
    fmt("TCP位姿", rpc.GetActualTCPPose())

    print("\n== 3. 目标点逆解（tool=0, config=-1 自动构型）==")
    fmt("GetInverseKin", rpc.GetInverseKin(0, list(TARGET_POSE), -1))

    print("\n== 4. 直线路径采样 IK（当前位姿→目标，每 10% 一点，参考当前构型）==")
    tcp_ret = rpc.GetActualTCPPose()
    joint_ret = rpc.GetActualJointPosDegree()
    if tcp_ret[0] == 0 and joint_ret[0] == 0:
        start, cur_joints = tcp_ret[1], joint_ret[1]
        for step in range(1, 10):  # 10%..90%，端点已验证
            sample = [start[i] + (TARGET_POSE[i] - start[i]) * step / 10.0 for i in range(6)]
            ret = rpc.GetInverseKinRef(0, sample, cur_joints)
            if isinstance(ret, tuple):
                sol = ret[1]
                # FR3 关节软限位（°），超出即 MoveL 会被拒
                limits = [(-175, 175), (-100, 120), (-200, 80), (-175, 175), (-175, 175), (-350, 350)]
                over = [f"j{i+1}={sol[i]:.1f}°" for i, (lo, hi) in enumerate(limits)
                        if not (lo <= sol[i] <= hi)]
                print(f"{step*10:>3}% 处 {['%.1f' % v for v in sample[:3]]}: "
                      f"{'关节超限! ' + ','.join(over) if over else 'OK'}")
            else:
                print(f"{step*10:>3}% 处 {['%.1f' % v for v in sample[:3]]}: IK 失败 错误码 {ret}")

    print("\n判读：")
    print("- 告警非 0 → 先在示教器/控制器清除告警并重新使能，再面板复位")
    print("- IK 非 0 → 目标点在当前工具坐标系下不可达/超关节限位")
    print("- 都正常但 MoveL 仍 14 → 直线路径中间点超限，先手动点动回中再复位")


if __name__ == "__main__":
    main()
