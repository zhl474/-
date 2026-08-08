#!/home/zhl/fr3env/fr3env/bin/python
"""独立标定采集 ROS 启动入口。"""

import rospy

from competition_lib.servo_csv_logger import DEFAULT_SERVO_CSV_OUTPUT_DIR
from competition_lib.task_runner import CalibrationTaskRunner


def main():
    rospy.init_node("calibration")
    output_dir = rospy.get_param(
        "~servo_csv_output_dir", str(DEFAULT_SERVO_CSV_OUTPUT_DIR)
    )
    session_id = rospy.get_param("~experiment_session_id", "")
    CalibrationTaskRunner(
        servo_csv_output_dir=output_dir,
        experiment_session_id=session_id,
    ).run_interactive()


if __name__ == "__main__":
    main()
