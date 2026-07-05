#!/home/zhl/fr3env/fr3env/bin/python
import argparse
import csv
import os

import yaml


def normalize_angle(theta_deg):
    theta = float(theta_deg) % 360.0
    if abs(theta - 360.0) < 1e-9:
        return 0.0
    return theta


def read_raw_points(raw_path):
    points = []
    with open(raw_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(row for row in f if row.strip() and not row.lstrip().startswith("#"))
        required_fields = {"theta_deg", "x_mm", "y_mm", "z_mm"}
        if not reader.fieldnames or not required_fields.issubset(set(reader.fieldnames)):
            raise ValueError("raw csv 必须包含字段: theta_deg,x_mm,y_mm,z_mm")

        for row in reader:
            points.append(
                {
                    "theta_deg": float(row["theta_deg"]),
                    "x_mm": float(row["x_mm"]),
                    "y_mm": float(row["y_mm"]),
                    "z_mm": float(row["z_mm"]),
                }
            )
    if not points:
        raise ValueError("raw csv 没有标定数据")
    return points


def build_config(points, theta0_deg, enabled):
    theta0_norm = normalize_angle(theta0_deg)
    p0 = None
    for point in points:
        if abs(normalize_angle(point["theta_deg"]) - theta0_norm) < 1e-9:
            p0 = point
            break
    if p0 is None:
        raise ValueError(f"raw csv 中找不到 theta0_deg={theta0_deg} 对应的基准点")

    table = []
    for point in points:
        table.append(
            {
                "theta_deg": normalize_angle(point["theta_deg"]),
                "dx_mm": point["x_mm"] - p0["x_mm"],
                "dy_mm": point["y_mm"] - p0["y_mm"],
                "dz_mm": point["z_mm"] - p0["z_mm"],
            }
        )
    table.sort(key=lambda item: item["theta_deg"])

    return {
        "enabled": bool(enabled),
        "unit": "mm",
        "angle_unit": "deg",
        "frame": "robot_base",
        "theta0_deg": float(theta0_deg),
        "periodic": True,
        "interpolation": "linear",
        "table": table,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="根据离线标定点生成吸盘舵机偏心补偿表")
    parser.add_argument(
        "--raw",
        default=os.path.join(script_dir, "raw_servo_eccentric_points.csv"),
        help="原始标定 csv 路径",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(script_dir, "servo_eccentric_compensation.yaml"),
        help="输出 yaml 配置路径",
    )
    parser.add_argument("--theta0", type=float, default=0.0, help="基准舵机角度，单位 deg")
    parser.add_argument("--enabled", action="store_true", help="输出配置中启用补偿")
    args = parser.parse_args()

    points = read_raw_points(args.raw)
    config = build_config(points, args.theta0, args.enabled)

    with open(args.output, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    print(f"已生成吸盘舵机偏心补偿配置: {args.output}")
    print(f"基准角度 theta0={args.theta0} deg，补偿点数量={len(config['table'])}")


if __name__ == "__main__":
    main()
