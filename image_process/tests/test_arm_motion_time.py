import json

import numpy as np
import pytest

from image_process_lib.arm_motion_time import (
    load_arm_motion_time_model,
    predict_arm_motion_time,
    predict_arm_motion_times,
)


def test_measured_points_and_zero_distance_are_exact():
    assert predict_arm_motion_time(0.0) == pytest.approx(0.0)
    assert predict_arm_motion_time(2.0) == pytest.approx(0.070727670999986)
    assert predict_arm_motion_time(450.0) == pytest.approx(0.9785447030008072)


def test_short_distance_uses_continuous_square_root_connection():
    expected = 0.070727670999986 * np.sqrt(1.0 / 2.0)
    assert predict_arm_motion_time(1.0) == pytest.approx(expected)
    left_of_boundary = predict_arm_motion_time(2.0 - 1e-9)
    at_boundary = predict_arm_motion_time(2.0)
    assert left_of_boundary == pytest.approx(at_boundary, abs=1e-9)


def test_calibrated_range_uses_linear_interpolation():
    time_at_300 = 0.7830294309987949
    time_at_350 = 0.8440214689999266
    expected = (time_at_300 + time_at_350) / 2.0
    assert predict_arm_motion_time(325.0) == pytest.approx(expected)


def test_long_distance_is_continuous_and_never_rejected_for_range():
    model = load_arm_motion_time_model()
    expected_500 = 0.9785447030008072 + model.long_distance_slope_s_per_mm * 50.0
    assert model.predict_seconds(500.0) == pytest.approx(expected_500)
    assert model.predict_seconds(1000.0) > model.predict_seconds(500.0)
    assert model.predict_seconds(450.0 + 1e-9) == pytest.approx(
        model.predict_seconds(450.0),
        abs=1e-9,
    )
    assert model.long_distance_slope_s_per_mm == pytest.approx(
        0.001293856900010724,
    )


def test_batch_prediction_preserves_matrix_shape_and_values():
    distances = np.array([[0.0, 2.0], [325.0, 500.0]])
    predicted = predict_arm_motion_times(distances)
    assert predicted.shape == distances.shape
    assert predicted[0, 0] == pytest.approx(0.0)
    assert predicted[0, 1] == pytest.approx(predict_arm_motion_time(2.0))
    assert predicted[1, 0] == pytest.approx(predict_arm_motion_time(325.0))
    assert predicted[1, 1] == pytest.approx(predict_arm_motion_time(500.0))


@pytest.mark.parametrize("distance", [-1.0, float("nan"), float("inf"), True])
def test_invalid_scalar_distance_is_rejected(distance):
    with pytest.raises(ValueError):
        predict_arm_motion_time(distance)


def test_invalid_calibration_rejects_non_monotonic_time(tmp_path):
    calibration = {
        "schema_version": 1,
        "motion_type": "MoveL",
        "move_speed_percent": 100.0,
        "acceleration_percent": 100.0,
        "time_metric": "move_call_median_s",
        "short_distance_method": "sqrt_to_origin",
        "short_distance_exponent": 0.5,
        "long_distance_method": "anchored_linear_regression",
        "long_distance_fit_start_mm": 2.0,
        "distance_time_table": [
            {"distance_mm": 2.0, "time_seconds": 0.2},
            {"distance_mm": 5.0, "time_seconds": 0.1},
        ],
    }
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")
    with pytest.raises(ValueError, match="单调不减"):
        load_arm_motion_time_model(path)
