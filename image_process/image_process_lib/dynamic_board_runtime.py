"""动态盘面正式接入的终端失败选择与原子 JSON 工具。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import select
import sys
import tempfile
import time
from typing import Callable, Mapping, TextIO


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
