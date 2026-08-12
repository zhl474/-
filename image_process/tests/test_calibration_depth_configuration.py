import os

import yaml


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PERCEPTION_CONFIG_PATH = os.path.join(
    SRC_DIR,
    "image_process",
    "config",
    "perception.yaml",
)
COMPETITION_LAUNCH_PATH = os.path.join(
    SRC_DIR,
    "competition",
    "launch",
    "competition.launch",
)
CALIBRATION_LAUNCH_PATH = os.path.join(
    SRC_DIR,
    "competition",
    "launch",
    "calibration.launch",
)
HARDWARE_LAUNCH_PATH = os.path.join(
    SRC_DIR,
    "competition",
    "launch",
    "hardware.launch",
)


def test_calibration_depth_parameters_are_fixed_in_perception_yaml():
    with open(PERCEPTION_CONFIG_PATH, "r", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)

    assert config["calibration_depth"] == {
        "frame_count": 15,
        "min_valid_frames": 10,
        "capture_timeout_sec": 2.0,
        "block_max_mad_mm": 1.0,
        "block_plane_max_rmse_mm": 1.0,
        "tray_tcp_below_block_observation_mm": 7.0,
    }


def test_task_launches_do_not_own_hardware_nodes_and_share_session_id():
    with open(COMPETITION_LAUNCH_PATH, "r", encoding="utf-8") as file_handle:
        competition_launch_text = file_handle.read()
    with open(CALIBRATION_LAUNCH_PATH, "r", encoding="utf-8") as file_handle:
        calibration_launch_text = file_handle.read()

    for launch_text in (competition_launch_text, calibration_launch_text):
        assert 'pkg="camera"' not in launch_text
        assert 'pkg="control"' not in launch_text
        assert launch_text.count(
            '<param name="experiment_session_id" value="$(arg experiment_session_id)"/>'
        ) == 2

    assert "depth_max_age_sec" not in competition_launch_text
    assert "depth_batch_max_span_sec" not in competition_launch_text
    assert "depth_buffer_size" not in competition_launch_text


def test_hardware_launch_owns_camera_and_control_nodes():
    with open(HARDWARE_LAUNCH_PATH, "r", encoding="utf-8") as file_handle:
        launch_text = file_handle.read()

    assert '<node pkg="camera" type="camera_node.py" name="camera_node" output="screen"/>' in launch_text
    assert '<node pkg="control" type="controller.py" name="control_node" output="screen">' in launch_text
    assert 'pkg="image_process"' not in launch_text
    assert 'pkg="competition"' not in launch_text
