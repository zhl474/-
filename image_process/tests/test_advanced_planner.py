import pytest

import image_process_lib.advanced_planner as advanced_module
from image_process_lib.advanced_planner import AdvancedPlanner


def test_advanced_planner_validates_seven_inputs_before_loading_library(monkeypatch):
    monkeypatch.setattr(advanced_module, "CDLL", lambda _path: (_ for _ in ()).throw(AssertionError("不应加载")))
    with pytest.raises(ValueError, match="7 类"):
        AdvancedPlanner("missing.so").build_layout([1], [1])


def test_advanced_planner_load_failure_only_affects_advanced_mode(monkeypatch):
    monkeypatch.setattr(advanced_module, "CDLL", lambda _path: (_ for _ in ()).throw(OSError("动态库不存在")))
    with pytest.raises(OSError, match="动态库不存在"):
        AdvancedPlanner("missing.so").build_layout([1] * 7, [1] * 7)
