from __future__ import annotations

import asyncio
import json
import math
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class RateLimitStore:
    """Persistent, process-safe quota accounting for successful generations."""

    WINDOW_SECONDS = 86400
    PENDING_TTL_SECONDS = WINDOW_SECONDS

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.state_path = self.data_dir / "rate_limit_usage.json"
        self.lock_path = self.data_dir / "rate_limit_usage.lock"
        self._async_lock = asyncio.Lock()
        self._storage_error: str | None = None

        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with self._file_lock():
                if self.state_path.exists():
                    self._read_state()
                else:
                    self._write_state(self._empty_state())
        except Exception as exc:
            self._storage_error = str(exc)

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {"version": 1, "successful": {}, "pending": {}}

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)

            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _valid_timestamp(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        timestamp = float(value)
        if not math.isfinite(timestamp) or timestamp < 0:
            return None
        return timestamp

    def _read_state(self) -> dict[str, Any]:
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("限额记录文件格式无效")

        successful: dict[str, list[float]] = {}
        raw_successful = payload.get("successful", {})
        if not isinstance(raw_successful, dict):
            raise ValueError("限额成功记录格式无效")
        for user_id, values in raw_successful.items():
            if not isinstance(values, list):
                raise ValueError("限额成功记录格式无效")
            timestamps = [
                timestamp
                for value in values
                if (timestamp := self._valid_timestamp(value)) is not None
            ]
            if timestamps:
                successful[str(user_id)] = timestamps

        pending: dict[str, dict[str, float]] = {}
        raw_pending = payload.get("pending", {})
        if not isinstance(raw_pending, dict):
            raise ValueError("限额占位记录格式无效")
        for user_id, requests in raw_pending.items():
            if not isinstance(requests, dict):
                raise ValueError("限额占位记录格式无效")
            valid_requests = {
                str(request_id): timestamp
                for request_id, value in requests.items()
                if (timestamp := self._valid_timestamp(value)) is not None
            }
            if valid_requests:
                pending[str(user_id)] = valid_requests

        return {"version": 1, "successful": successful, "pending": pending}

    def _write_state(self, state: dict[str, Any]) -> None:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.state_path.name}-", dir=self.data_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(state, handle, ensure_ascii=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.state_path)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    def _prune(self, state: dict[str, Any], now: float) -> None:
        successful = state["successful"]
        for user_id in list(successful):
            timestamps = [
                timestamp
                for timestamp in successful[user_id]
                if now - timestamp < self.WINDOW_SECONDS
            ]
            if timestamps:
                successful[user_id] = timestamps
            else:
                del successful[user_id]

        pending = state["pending"]
        for user_id in list(pending):
            requests = {
                request_id: timestamp
                for request_id, timestamp in pending[user_id].items()
                if now - timestamp < self.PENDING_TTL_SECONDS
            }
            if requests:
                pending[user_id] = requests
            else:
                del pending[user_id]

    @staticmethod
    def _limit_message(period: str, limit: int) -> str:
        if period == "minute":
            return f"❌ 请求过于频繁，请稍后再试 (每分钟限 {limit} 次)"
        if period == "hour":
            return f"❌ 请求过于频繁，请稍后再试 (每小时限 {limit} 次)"
        return f"❌ 今日成功生图次数已达上限 (滚动24小时限 {limit} 次)"

    @staticmethod
    def _storage_message(error: str | None) -> str:
        detail = f": {error}" if error else ""
        return f"❌ 限额记录不可用，已停止生图以防止超额消费{detail}"

    async def reserve(
        self,
        user_id: str,
        request_id: str,
        *,
        enabled: bool,
        minute_limit: int,
        hour_limit: int,
        day_limit: int,
        now: float | None = None,
    ) -> tuple[bool, str]:
        if not enabled:
            return True, ""
        if self._storage_error:
            return False, self._storage_message(self._storage_error)

        now = time.time() if now is None else now
        user_id = str(user_id).strip()
        request_id = str(request_id).strip()

        async with self._async_lock:
            try:
                with self._file_lock():
                    state = self._read_state()
                    self._prune(state, now)

                    timestamps = list(state["successful"].get(user_id, []))
                    timestamps.extend(state["pending"].get(user_id, {}).values())
                    counts = {
                        "minute": sum(now - value < 60 for value in timestamps),
                        "hour": sum(now - value < 3600 for value in timestamps),
                        "day": len(timestamps),
                    }
                    limits = {
                        "minute": minute_limit,
                        "hour": hour_limit,
                        "day": day_limit,
                    }
                    for period in ("minute", "hour", "day"):
                        if counts[period] >= limits[period]:
                            self._write_state(state)
                            return False, self._limit_message(period, limits[period])

                    state["pending"].setdefault(user_id, {})[request_id] = now
                    self._write_state(state)
                    return True, ""
            except Exception as exc:
                self._storage_error = str(exc)
                return False, self._storage_message(self._storage_error)

    async def finish(
        self,
        user_id: str,
        request_id: str,
        *,
        successful: bool,
        now: float | None = None,
    ) -> bool:
        if self._storage_error:
            return False

        now = time.time() if now is None else now
        user_id = str(user_id).strip()
        request_id = str(request_id).strip()

        async with self._async_lock:
            try:
                with self._file_lock():
                    state = self._read_state()
                    self._prune(state, now)
                    requests = state["pending"].get(user_id, {})
                    if request_id not in requests:
                        self._write_state(state)
                        return False

                    del requests[request_id]
                    if not requests:
                        state["pending"].pop(user_id, None)
                    if successful:
                        state["successful"].setdefault(user_id, []).append(now)
                    self._write_state(state)
                    return True
            except Exception as exc:
                self._storage_error = str(exc)
                return False

    async def snapshot(
        self, user_id: str, *, now: float | None = None
    ) -> tuple[dict[str, int] | None, str]:
        if self._storage_error:
            return None, self._storage_message(self._storage_error)

        now = time.time() if now is None else now
        user_id = str(user_id).strip()
        async with self._async_lock:
            try:
                with self._file_lock():
                    state = self._read_state()
                    self._prune(state, now)
                    self._write_state(state)
                    successful = state["successful"].get(user_id, [])
                    pending = state["pending"].get(user_id, {})
                    return {
                        "minute": sum(now - value < 60 for value in successful),
                        "hour": sum(now - value < 3600 for value in successful),
                        "day": len(successful),
                        "pending": len(pending),
                    }, ""
            except Exception as exc:
                self._storage_error = str(exc)
                return None, self._storage_message(self._storage_error)
