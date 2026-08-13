"""控制台 SQLite 状态、配置历史和预设存储。"""

from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import threading


class StateStore:
    """每次操作使用独立连接，避免 Flask 与 ROS 线程共享游标。"""

    def __init__(self, database_path):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self._initialize()

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(str(self.path), timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self):
        with self._write_lock, self.connection() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS config_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    git_revision TEXT NOT NULL DEFAULT '',
                    before_revision TEXT NOT NULL,
                    after_revision TEXT NOT NULL,
                    before_text TEXT NOT NULL,
                    after_text TEXT NOT NULL,
                    diff_text TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '网页保存'
                );
                CREATE INDEX IF NOT EXISTS idx_config_history_file
                    ON config_history(file_id, id DESC);
                CREATE TABLE IF NOT EXISTS presets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    git_revision TEXT NOT NULL DEFAULT '',
                    snapshot_json TEXT NOT NULL,
                    protected INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS run_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    task_count INTEGER NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT ''
                );
                """
            )
            connection.commit()

    @staticmethod
    def _now():
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def add_history(
        self,
        file_id,
        before_revision,
        after_revision,
        before_text,
        after_text,
        diff_text,
        git_revision="",
        reason="网页保存",
    ):
        return self.add_history_batch([{
            "file_id": file_id,
            "before_revision": before_revision,
            "after_revision": after_revision,
            "before_text": before_text,
            "after_text": after_text,
            "diff_text": diff_text,
            "git_revision": git_revision,
            "reason": reason,
        }])[0]

    def add_history_batch(self, entries):
        """在同一个 SQLite 事务中写入一批历史，供完整预设恢复使用。"""
        entries = list(entries)
        if not entries:
            return []
        history_ids = []
        with self._write_lock, self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                for entry in entries:
                    cursor = connection.execute(
                        """
                        INSERT INTO config_history(
                            file_id, created_at, git_revision, before_revision,
                            after_revision, before_text, after_text, diff_text, reason
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(entry["file_id"]), self._now(),
                            str(entry.get("git_revision") or ""),
                            str(entry["before_revision"]),
                            str(entry["after_revision"]),
                            str(entry["before_text"]),
                            str(entry["after_text"]),
                            str(entry["diff_text"]),
                            str(entry.get("reason", "网页保存")),
                        ),
                    )
                    history_ids.append(int(cursor.lastrowid))
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return history_ids

    def list_history(self, file_id=None, limit=100):
        query = (
            "SELECT id, file_id, created_at, git_revision, before_revision, "
            "after_revision, diff_text, reason FROM config_history"
        )
        parameters = []
        if file_id:
            query += " WHERE file_id = ?"
            parameters.append(str(file_id))
        query += " ORDER BY id DESC LIMIT ?"
        parameters.append(max(1, min(500, int(limit))))
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(query, parameters)]

    def get_history(self, history_id):
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM config_history WHERE id = ?", (int(history_id),)
            ).fetchone()
        return dict(row) if row is not None else None

    def save_preset(self, name, snapshot, git_revision="", protected=False, overwrite=False):
        now = self._now()
        payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
        with self._write_lock, self.connection() as connection:
            existing = connection.execute(
                "SELECT id, protected FROM presets WHERE name = ?", (str(name),)
            ).fetchone()
            if existing is not None:
                if not overwrite:
                    raise ValueError("预设名称已存在")
                if bool(existing["protected"]):
                    raise PermissionError("系统初始预设不允许覆盖")
                connection.execute(
                    """
                    UPDATE presets SET updated_at = ?, git_revision = ?,
                        snapshot_json = ?, protected = ? WHERE id = ?
                    """,
                    (now, str(git_revision or ""), payload, int(bool(protected)), existing["id"]),
                )
                preset_id = int(existing["id"])
            else:
                cursor = connection.execute(
                    """
                    INSERT INTO presets(name, created_at, updated_at, git_revision,
                                        snapshot_json, protected)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (str(name), now, now, str(git_revision or ""), payload, int(bool(protected))),
                )
                preset_id = int(cursor.lastrowid)
            connection.commit()
            return preset_id

    def list_presets(self):
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT id, name, created_at, updated_at, git_revision, protected
                FROM presets ORDER BY protected DESC, updated_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_preset(self, preset_id):
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM presets WHERE id = ?", (int(preset_id),)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["snapshot"] = json.loads(result.pop("snapshot_json"))
        return result

    def delete_preset(self, preset_id):
        with self._write_lock, self.connection() as connection:
            row = connection.execute(
                "SELECT protected FROM presets WHERE id = ?", (int(preset_id),)
            ).fetchone()
            if row is None:
                return False
            if bool(row["protected"]):
                raise PermissionError("系统初始预设不允许删除")
            connection.execute("DELETE FROM presets WHERE id = ?", (int(preset_id),))
            connection.commit()
            return True

    def add_run_summary(self, started_at, finished_at, mode, status, task_count, message=""):
        with self._write_lock, self.connection() as connection:
            connection.execute(
                """
                INSERT INTO run_summaries(
                    started_at, finished_at, mode, status, task_count, message
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (started_at, finished_at, mode, status, int(task_count), str(message)),
            )
            connection.commit()
