from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

GROUP_PREFIX = "group:"


def describe_subject(subject: str) -> tuple[str, str]:
    """把额度主体拆成 (会话类型, 群号或用户 ID)。"""
    if subject.startswith(GROUP_PREFIX):
        return "group", subject[len(GROUP_PREFIX) :]
    return "private", subject


class SessionRegistry:
    """记录使用过插件的会话，供 WebUI 会话管理页展示。

    键与限额主体一致：群聊为 group:<群号>，私聊为用户 ID。
    每次操作都重新读取文件再原子写回，中间没有 await，
    插件重载时新旧实例交替写入也不会互相覆盖。
    """

    MAX_SUBJECT_LENGTH = 256
    MAX_TEXT_LENGTH = 64

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / "sessions.json"

    @classmethod
    def normalize_subject(cls, subject: Any) -> str:
        subject = str(subject or "").strip()
        _, target_id = describe_subject(subject)
        if not target_id or len(subject) > cls.MAX_SUBJECT_LENGTH:
            raise ValueError("会话 ID 无效")
        return subject

    @classmethod
    def _clip(cls, value: Any) -> str:
        return str(value or "").strip()[: cls.MAX_TEXT_LENGTH]

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        sessions = payload.get("sessions") if isinstance(payload, dict) else None
        if not isinstance(sessions, dict):
            raise ValueError("会话记录文件格式无效")
        return {
            str(subject): record
            for subject, record in sessions.items()
            if isinstance(record, dict)
        }

    def _write(self, sessions: dict[str, dict[str, Any]]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}-", dir=self.data_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    {"version": 1, "sessions": sessions},
                    handle,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    @staticmethod
    def _new_record(subject: str, now: float) -> dict[str, Any]:
        kind, target_id = describe_subject(subject)
        return {
            "kind": kind,
            "target_id": target_id,
            "name": "",
            "remark": "",
            "platform": "",
            "last_user_id": "",
            "last_user_name": "",
            "total_success": 0,
            "first_seen": now,
            "last_seen": None,
            "last_success": None,
        }

    def list(self) -> dict[str, dict[str, Any]]:
        return self._read()

    def touch(
        self,
        subject: str,
        *,
        name: str = "",
        platform: str = "",
        user_id: str = "",
        user_name: str = "",
        now: float | None = None,
    ) -> None:
        """会话发起生图请求时更新会话信息。"""
        subject = self.normalize_subject(subject)
        now = time.time() if now is None else now
        sessions = self._read()
        record = sessions.setdefault(subject, self._new_record(subject, now))
        updates = {
            "name": self._clip(name),
            "platform": self._clip(platform),
            "last_user_id": str(user_id or "").strip()[: self.MAX_SUBJECT_LENGTH],
            "last_user_name": self._clip(user_name),
        }
        # 平台不一定每次都带群名或昵称，拿不到时保留上一次的值
        record.update({key: value for key, value in updates.items() if value})
        record["last_seen"] = now
        self._write(sessions)

    def record_success(self, subject: str, *, now: float | None = None) -> None:
        """累计成功发送的图片任务数，不受频率限制开关影响。"""
        subject = self.normalize_subject(subject)
        now = time.time() if now is None else now
        sessions = self._read()
        record = sessions.setdefault(subject, self._new_record(subject, now))
        record["total_success"] = int(record.get("total_success") or 0) + 1
        record["last_success"] = now
        self._write(sessions)

    def set_remark(
        self, subject: str, remark: str, *, now: float | None = None
    ) -> None:
        subject = self.normalize_subject(subject)
        now = time.time() if now is None else now
        sessions = self._read()
        record = sessions.setdefault(subject, self._new_record(subject, now))
        record["remark"] = self._clip(remark)
        self._write(sessions)

    def remove(self, subject: str) -> bool:
        subject = self.normalize_subject(subject)
        sessions = self._read()
        if sessions.pop(subject, None) is None:
            return False
        self._write(sessions)
        return True
