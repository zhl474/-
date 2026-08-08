import os
import sys
import types

import numpy as np


def _load_debug_output(monkeypatch):
    rospy = types.ModuleType("rospy")
    rospy.logwarn = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "rospy", rospy)
    sys.modules.pop("image_process_lib.debug_output", None)
    from image_process_lib import debug_output

    return debug_output


def test_debug_video_archive_updates_latest_alias(monkeypatch, tmp_path):
    module = _load_debug_output(monkeypatch)
    writes = []

    class FakeWriter:
        def isOpened(self):
            return True

        def write(self, frame):
            writes.append(frame.copy())

        def release(self):
            return None

    monkeypatch.setattr(module.cv2, "VideoWriter", lambda *_args, **_kwargs: FakeWriter())
    archive_path = tmp_path / "实验日志" / "session" / "方块视觉伺服调试.avi"
    latest_path = tmp_path / "方块视觉伺服调试.avi"
    recorder = module.DebugVideoRecorder(
        str(archive_path),
        latest_alias_path=str(latest_path),
    )

    assert recorder.write(np.zeros((8, 12, 3), dtype=np.uint8)) is True
    assert len(writes) == 1
    assert latest_path.is_symlink()
    assert os.path.realpath(latest_path) == str(archive_path)
