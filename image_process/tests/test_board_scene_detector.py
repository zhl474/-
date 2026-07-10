import cv2
import numpy as np

from image_process_lib.board_scene_detector import _detect_grid_keypoints, _make_blob_detector


def test_high_board_blob_detector_is_available():
    detector = _make_blob_detector()
    assert detector is not None


def test_high_board_grid_detection_pipeline_can_run_on_synthetic_image():
    image = np.full((200, 240, 3), 230, dtype=np.uint8)
    for y in (50, 100, 150):
        for x in (50, 100, 150, 200):
            cv2.circle(image, (x, y), 5, (20, 20, 20), -1)

    keypoints, debug_image, threshold_image = _detect_grid_keypoints(image)

    assert keypoints is not None
    assert debug_image.shape == image.shape
    assert threshold_image.shape == image.shape[:2]
