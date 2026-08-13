"""网页 SSE 使用的线程安全事件总线。"""

from collections import deque
from copy import deepcopy
from datetime import datetime
import threading
import time


class EventBus:
    """保存有限数量事件，并支持按事件序号断线续传。"""

    def __init__(self, capacity=2000):
        self._events = deque(maxlen=max(10, int(capacity)))
        self._condition = threading.Condition()
        self._next_id = 1

    def publish(self, event_type, data):
        with self._condition:
            event = {
                "id": self._next_id,
                "type": str(event_type),
                "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "data": deepcopy(data),
            }
            self._next_id += 1
            self._events.append(event)
            self._condition.notify_all()
            return deepcopy(event)

    def events_after(self, last_id=0):
        with self._condition:
            return [deepcopy(item) for item in self._events if item["id"] > int(last_id)]

    def wait_after(self, last_id=0, timeout=15.0):
        """等待新事件；超时返回空列表，让 SSE 发送心跳。"""
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while True:
                events = [
                    deepcopy(item)
                    for item in self._events
                    if item["id"] > int(last_id)
                ]
                if events:
                    return events
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return []
                self._condition.wait(remaining)

    def snapshot(self, event_type=None, limit=200):
        with self._condition:
            selected = list(self._events)
            if event_type is not None:
                selected = [item for item in selected if item["type"] == event_type]
            return deepcopy(selected[-max(0, int(limit)):])
