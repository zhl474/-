"""动态盘面正式接入的终端失败选择与原子 JSON 工具。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import select
import sys
import tempfile
import threading
import time
from typing import Callable, Mapping, TextIO
import uuid


class OperatorPromptConflict(RuntimeError):
    """待处理提示不存在、已经回答或提示编号过期。"""


class OperatorPromptChoiceError(ValueError):
    """网页提交了当前提示不允许的选项。"""


class OperatorPromptBroker:
    """在 ROS 服务线程之间传递单个、限时且固定选项的人工选择。"""

    BASE_CHOICES = frozenset(("stop", "fixed_yaml"))
    CONTINUE_CHOICE = "continue_dynamic"

    def __init__(self, monotonic=time.monotonic, id_factory=None):
        self._monotonic = monotonic
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self._condition = threading.Condition(threading.RLock())
        self._pending = None
        self._closed = False

    @staticmethod
    def _timeout_seconds(value):
        try:
            timeout = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("人工选择超时必须是大于 0 的有限秒数") from exc
        if not timeout > 0.0 or not timeout < float("inf"):
            raise ValueError("人工选择超时必须是大于 0 的有限秒数")
        return timeout

    def request_dynamic_failure(self, reason, timeout_seconds, allow_continue=False):
        """发布动态盘面失败提示并等待网页回答；任何不确定情况均停止。"""
        timeout = self._timeout_seconds(timeout_seconds)
        allowed = set(self.BASE_CHOICES)
        if allow_continue:
            allowed.add(self.CONTINUE_CHOICE)
        with self._condition:
            if self._closed:
                return "stop"
            if self._pending is not None:
                raise OperatorPromptConflict("已有人工选择正在等待处理")
            prompt = {
                "prompt_id": str(self._id_factory()),
                "prompt_type": "dynamic_board_failure",
                "message": str(reason),
                "allow_fixed_yaml": True,
                "allow_continue_dynamic": bool(allow_continue),
                "allowed_choices": frozenset(allowed),
                "deadline": self._monotonic() + timeout,
                "choice": None,
            }
            self._pending = prompt
            self._condition.notify_all()
            while prompt["choice"] is None and not self._closed:
                remaining = prompt["deadline"] - self._monotonic()
                if remaining <= 0.0:
                    prompt["choice"] = "stop"
                    break
                self._condition.wait(remaining)
            choice = prompt["choice"] or "stop"
            if self._pending is prompt:
                self._pending = None
            self._condition.notify_all()
            return choice

    def snapshot(self):
        """返回适合 ROS 服务传输的当前提示快照。"""
        with self._condition:
            prompt = self._pending
            if prompt is None or prompt["choice"] is not None:
                return {
                    "pending": False,
                    "prompt_id": "",
                    "prompt_type": "",
                    "message": "",
                    "allow_fixed_yaml": False,
                    "allow_continue_dynamic": False,
                    "remaining_seconds": 0.0,
                }
            remaining = max(0.0, prompt["deadline"] - self._monotonic())
            return {
                "pending": True,
                "prompt_id": prompt["prompt_id"],
                "prompt_type": prompt["prompt_type"],
                "message": prompt["message"],
                "allow_fixed_yaml": prompt["allow_fixed_yaml"],
                "allow_continue_dynamic": prompt["allow_continue_dynamic"],
                "remaining_seconds": remaining,
            }

    def respond(self, prompt_id, choice):
        """接受当前提示的第一个合法回答。"""
        normalized_id = str(prompt_id or "")
        normalized_choice = str(choice or "")
        with self._condition:
            prompt = self._pending
            if prompt is None or prompt["choice"] is not None:
                raise OperatorPromptConflict("当前没有等待回答的人工选择")
            if normalized_id != prompt["prompt_id"]:
                raise OperatorPromptConflict("人工选择提示已过期，请刷新页面状态")
            if normalized_choice not in prompt["allowed_choices"]:
                raise OperatorPromptChoiceError("当前提示不允许此选项")
            prompt["choice"] = normalized_choice
            self._condition.notify_all()
            return normalized_choice

    def cancel(self):
        """安全取消当前提示，使等待方按停止本轮继续收尾。"""
        with self._condition:
            prompt = self._pending
            if prompt is None or prompt["choice"] is not None:
                return False
            prompt["choice"] = "stop"
            self._condition.notify_all()
            return True

    def close(self):
        """关闭代理并唤醒所有等待线程。"""
        with self._condition:
            self._closed = True
            if self._pending is not None and self._pending["choice"] is None:
                self._pending["choice"] = "stop"
            self._condition.notify_all()


def sha256_file(path) -> str:
    """流式计算文件 SHA256；路径为空时返回空字符串。"""
    if path is None or not str(path).strip():
        return ""
    resolved = Path(path).expanduser()
    digest = hashlib.sha256()
    with resolved.open("rb") as input_file:
        while True:
            chunk = input_file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path, document: Mapping) -> Path:
    """同目录临时文件落盘后原子替换 JSON。"""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(document, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    return output_path


def prompt_dynamic_selection_failure(
    reason: str,
    timeout_seconds: float = 60.0,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    wait_readable: Callable = select.select,
    monotonic: Callable[[], float] = time.monotonic,
    allow_continue: bool = False,
) -> str:
    """在真实 TTY 中询问失败处理，任何不确定情况均停止。"""
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stderr
    try:
        timeout = float(timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("failure_prompt_timeout_sec 必须是大于 0 的有限数") from exc
    if not timeout > 0.0 or not timeout < float("inf"):
        raise ValueError("failure_prompt_timeout_sec 必须是大于 0 的有限数")

    try:
        is_tty = bool(input_stream.isatty())
        input_stream.fileno()
    except (AttributeError, OSError, ValueError):
        is_tty = False
    if not is_tty:
        return "stop"

    print("\n动态盘面选择失败：" + str(reason), file=output_stream, flush=True)
    print("[s] 停止本轮", file=output_stream, flush=True)
    print("[f] 回退固定 task_layout.yaml", file=output_stream, flush=True)
    if allow_continue:
        print(
            "[c] 忽略时间标定速度不一致，继续动态盘面执行",
            file=output_stream,
            flush=True,
        )
        print(
            "    实际机械臂速度保持当前配置，仅盘面预测时间不再代表真实秒数。",
            file=output_stream,
            flush=True,
        )
    deadline = monotonic() + timeout
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0.0:
            return "stop"
        print("> ", end="", file=output_stream, flush=True)
        try:
            readable, _writable, _errors = wait_readable(
                [input_stream],
                [],
                [],
                remaining,
            )
        except (OSError, ValueError, TypeError):
            return "stop"
        if not readable:
            return "stop"
        try:
            line = input_stream.readline()
        except (OSError, ValueError):
            return "stop"
        if line == "":
            return "stop"
        choice = line.strip().lower()
        if choice in ("s", "stop"):
            return "stop"
        if choice in ("f", "fallback"):
            return "fixed_yaml"
        if allow_continue and choice in ("c", "continue"):
            return "continue_dynamic"
        choices_text = "s、f 或 c" if allow_continue else "s 或 f"
        print(f"输入无效，请输入 {choices_text}。", file=output_stream, flush=True)
