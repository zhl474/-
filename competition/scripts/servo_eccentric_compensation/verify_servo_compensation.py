#!/home/zhl/fr3env/fr3env/bin/python
import argparse
import os

from servo_eccentric_compensator import ServoEccentricCompensator


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "servo_eccentric_compensation.yaml")

    parser = argparse.ArgumentParser(description="验证吸盘舵机偏心补偿结果")
    parser.add_argument("--config", default=default_config, help="补偿 yaml 配置路径")
    parser.add_argument("--theta", type=float, required=True, help="舵机角度，单位 deg")
    parser.add_argument(
        "--pose",
        type=float,
        nargs="+",
        required=True,
        help="机械臂目标点，至少输入 x y z，可附加 rx ry rz",
    )
    args = parser.parse_args()

    compensator = ServoEccentricCompensator(args.config, debug=True)
    compensation = compensator.get_compensation(args.theta)
    p_cmd = compensator.apply(args.pose, args.theta)

    print(f"输入角度(deg): {args.theta}")
    print(f"补偿量(mm): {compensation.tolist()}")
    print(f"补偿前目标点: {args.pose}")
    print(f"补偿后目标点: {p_cmd.tolist()}")


if __name__ == "__main__":
    main()
