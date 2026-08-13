import threading
import time

from operator_panel_lib.event_bus import EventBus


def test_event_bus支持按序号断线续传():
    bus = EventBus(capacity=10)
    first = bus.publish("state", {"value": 1})
    second = bus.publish("log", {"value": 2})

    assert [item["id"] for item in bus.events_after(first["id"])] == [second["id"]]
    assert bus.events_after(second["id"]) == []


def test_event_bus等待新事件且超时不伪造事件():
    bus = EventBus(capacity=10)

    def later():
        time.sleep(0.02)
        bus.publish("progress", {"current": 1})

    thread = threading.Thread(target=later)
    thread.start()
    events = bus.wait_after(0, timeout=0.5)
    thread.join()

    assert events[0]["type"] == "progress"
    assert bus.wait_after(events[0]["id"], timeout=0.01) == []
