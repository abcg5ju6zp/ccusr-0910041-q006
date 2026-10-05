"""持久化后台任务的状态机与存储。

运营作业会被请求、启动监听器和重试程序重复触发。仅靠内存中的
``asyncio.Task`` 无法回答两个问题：

1. 同一业务作业是否已经在跑（业务幂等键去重）；
2. 工作进程退出后，哪些任务可以被新进程安全接管。

本模块用落盘的持久状态回答它们。状态迁移为：

    ACCEPTED -> RUNNING -> COMPLETED
                        \\-> FAILED
                        \\-> CANCELLED
    持有租约的 ACCEPTED / RUNNING 在心跳停止后可被标记为
    LEASE_EXPIRED，随后允许被新的执行者接管
    （LEASE_EXPIRED -> RUNNING，attempts 累加）。

COMPLETED / FAILED / CANCELLED 为终态，落盘后不可再改变，
因此取消与接管竞争时至多有一个终态、至多有一个结果被采纳。

存储以 JSON 文件 + 操作系统文件锁实现（POSIX 用 ``fcntl``，
Windows 用 ``msvcrt``），所有读改写都持锁完成，并以临时文件 +
``os.replace`` 原子落盘，保证同机多进程并发安全。
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid

from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterator, Protocol


try:  # POSIX：跨进程咨询锁
    import fcntl

    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore
    _HAS_FCNTL = False

try:  # Windows：跨进程字节范围锁
    import msvcrt

    _HAS_MSVCRT = True
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore
    _HAS_MSVCRT = False

# 既无 fcntl 也无 msvcrt 的平台：退化为进程内锁，保证线程安全，
# 但不提供跨进程互斥（原子写仍保证文件不损坏）。
_fallback_lock = threading.RLock()


class TaskStatus(str, Enum):
    """持久任务的生命周期状态。"""

    ACCEPTED = "accepted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LEASE_EXPIRED = "lease_expired"

    @property
    def terminal(self) -> bool:
        return self in TERMINAL_STATUSES


TERMINAL_STATUSES = frozenset(
    {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }
)

# 可发起接管的非终态
EXECUTABLE_STATUSES = frozenset(
    {
        TaskStatus.ACCEPTED,
        TaskStatus.LEASE_EXPIRED,
    }
)

# 持有租约、执行者尚在心跳中的状态
LEASED_STATUSES = frozenset(
    {
        TaskStatus.ACCEPTED,
        TaskStatus.RUNNING,
    }
)

DEFAULT_LEASE_TIMEOUT = 30.0
DEFAULT_LEASE_REFRESH_INTERVAL = 10.0
DEFAULT_POLL_INTERVAL = 0.25


class PersistentTaskError(RuntimeError):
    """重建持久任务失败结果时抛出，携带持久化的错误信息。"""

    def __init__(self, key: str, error_type: str, message: str):
        self.key = key
        self.error_type = error_type
        self.message = message
        super().__init__(f"Persistent task {key!r} failed: "
                         f"{error_type}: {message}")


def worker_identity() -> str:
    """当前执行者标识。

    形如 ``pid-tid-uuid``：pid/tid 便于排障定位，随机后缀保证
    同一进程内多个管理器（如测试中模拟多进程接管）也被视为
    不同执行者。
    """
    return (
        f"{os.getpid()}-{threading.get_ident()}-"
        f"{uuid.uuid4().hex[:12]}"
    )


@dataclass
class TaskState:
    """单个持久任务的持久化视图。"""

    key: str
    status: TaskStatus
    created_at: float
    updated_at: float
    name: str | None = None
    handler: str | None = None
    payload: dict[str, Any] | None = None
    lease_owner: str | None = None
    lease_expires_at: float | None = None
    attempts: int = 0
    result: Any = None
    result_kind: str = "value"
    error: dict[str, str] | None = None
    terminal_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskState":
        record = dict(data)
        record["status"] = TaskStatus(record["status"])
        # 向前兼容：补齐后续版本新增字段
        for field_name in (
            "name",
            "handler",
            "payload",
            "lease_owner",
            "lease_expires_at",
            "attempts",
            "result",
            "result_kind",
            "error",
            "terminal_at",
        ):
            record.setdefault(field_name, None)
        record.setdefault("attempts", 0)
        record.setdefault("result_kind", "value")
        return cls(**record)

    def lease_is_active(self, now: float | None = None) -> bool:
        """当前时间戳下租约是否仍然有效。"""
        moment = time.time() if now is None else now
        if self.status not in LEASED_STATUSES:
            return False
        if self.lease_expires_at is None:
            return False
        return self.lease_expires_at > moment


def _serialize_error(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "module": type(exc).__module__,
        "message": str(exc),
    }


class ResultCodec(Protocol):
    """结果编解码协议：默认要求结果可被 JSON 序列化。"""

    def encode(self, value: Any) -> Any: ...
    def decode(self, raw: Any) -> Any: ...


class JSONResultCodec:
    """默认编解码：能被 JSON 编码的结果原样持久化。"""

    def encode(self, value: Any) -> Any:
        # 提前失败：不可 JSON 序列化的结果不应在复用时静默丢失
        json.dumps(value)
        return value

    def decode(self, raw: Any) -> Any:
        return raw


class TaskStore:
    """基于 JSON 文件的持久任务存储，多进程安全。

    所有判定与写回都在同一次持锁内完成，构成 compare-and-swap；
    终态记录拒绝任何后续迁移，保证竞争中只有一个终态。
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        codec: ResultCodec | None = None,
    ) -> None:
        self.path = os.fspath(path)
        self._codec = codec or JSONResultCodec()
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not os.path.exists(self.path):
            self._write_locked({})

    # ------------------------------------------------------------------ #
    # 底层持锁读写
    # ------------------------------------------------------------------ #

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """持有跨线程/跨进程排它锁的上下文。

        * POSIX：``fcntl.flock`` 锁独立的 ``.lock`` 文件；
        * Windows：``msvcrt.locking`` 对锁文件上字节范围锁；
        * 其它平台：进程内可重入锁（线程安全，不保证跨进程）。
        """
        lock_path = f"{self.path}.lock"
        if _HAS_FCNTL:
            with open(lock_path, "a+") as lock_fh:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        elif _HAS_MSVCRT:  # pragma: no cover - Windows
            with open(lock_path, "a+") as lock_fh:
                lock_fh.seek(0)
                msvcrt.locking(
                    lock_fh.fileno(), msvcrt.LK_LOCK, 1
                )
                try:
                    yield
                finally:
                    lock_fh.seek(0)
                    msvcrt.locking(
                        lock_fh.fileno(), msvcrt.LK_UNLCK, 1
                    )
        else:  # pragma: no cover
            with _fallback_lock:
                yield

    def _read_locked(self) -> dict[str, Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            # 极端情况下读到正在替换的文件不应发生（os.replace 原子），
            # 这里保守地视为空表，下一次写会修复文件。
            return {}

    def _write_locked(self, data: dict[str, Any]) -> None:
        directory = os.path.dirname(self.path) or "."
        fd, tmp = tempfile.mkstemp(
            prefix=".task_store.", suffix=".json", dir=directory
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            with suppress(OSError):
                os.unlink(tmp)
            raise

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #

    def get(self, key: str) -> TaskState | None:
        with self._locked():
            record = self._read_locked().get(key)
            return TaskState.from_dict(record) if record else None

    def load(self) -> dict[str, TaskState]:
        with self._locked():
            raw = self._read_locked()
        return {
            key: TaskState.from_dict(record)
            for key, record in raw.items()
        }

    # ------------------------------------------------------------------ #
    # 状态迁移（均在单次持锁内 CAS）
    # ------------------------------------------------------------------ #

    def claim(
        self,
        key: str,
        owner: str,
        *,
        lease_timeout: float = DEFAULT_LEASE_TIMEOUT,
        name: str | None = None,
        handler: str | None = None,
        payload: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> TaskState:
        """就任务发起受理或接管，返回最新状态。

        调用方根据返回状态判断分支：

        * ``lease_owner == owner``：受理（ACCEPTED）或接管
          （RUNNING）成功，应由本进程执行；
        * ``status.terminal``：任务已有终态，直接复用持久化结果，
          不得再次执行；
        * 其它情况：另一个执行者持有效租约，调用方应等待终态。
        """
        moment = now if now is not None else time.time()
        with self._locked():
            raw = self._read_locked()
            record = raw.get(key)

            if record is None:
                state = TaskState(
                    key=key,
                    name=name,
                    handler=handler,
                    payload=payload,
                    status=TaskStatus.ACCEPTED,
                    created_at=moment,
                    updated_at=moment,
                    lease_owner=owner,
                    lease_expires_at=moment + lease_timeout,
                    attempts=1,
                )
                raw[key] = state.to_dict()
                self._write_locked(raw)
                return state

            current = TaskState.from_dict(record)

            if current.status.terminal or current.lease_is_active(moment):
                # 已有终态：复用；他人租约有效：等待。均不改动记录。
                return current

            # 租约已失效：RUNNING/ACCEPTED 先收敛为 LEASE_EXPIRED，
            # 随后与显式 LEASE_EXPIRED 一样被本进程接管。
            current.status = TaskStatus.RUNNING
            current.lease_owner = owner
            current.lease_expires_at = moment + lease_timeout
            current.attempts += 1
            current.error = None
            current.updated_at = moment
            if name is not None:
                current.name = name
            if handler is not None:
                current.handler = handler
            if payload is not None:
                current.payload = payload
            raw[key] = current.to_dict()
            self._write_locked(raw)
            return current

    def mark_running(
        self,
        key: str,
        owner: str,
        *,
        now: float | None = None,
    ) -> bool:
        """受理 -> 执行。仅持有租约的执行者可迁移。"""
        moment = now if now is not None else time.time()
        with self._locked():
            raw = self._read_locked()
            record = raw.get(key)
            if record is None:
                return False
            current = TaskState.from_dict(record)
            if current.lease_owner != owner or current.status.terminal:
                return False
            if current.status == TaskStatus.ACCEPTED:
                current.status = TaskStatus.RUNNING
                current.updated_at = moment
                raw[key] = current.to_dict()
                self._write_locked(raw)
            return True

    def heartbeat(
        self,
        key: str,
        owner: str,
        *,
        lease_timeout: float = DEFAULT_LEASE_TIMEOUT,
        now: float | None = None,
    ) -> bool:
        """续约；仅当前租约持有者且任务未终结时可续。"""
        moment = now if now is not None else time.time()
        with self._locked():
            raw = self._read_locked()
            record = raw.get(key)
            if record is None:
                return False
            current = TaskState.from_dict(record)
            if current.lease_owner != owner or current.status.terminal:
                return False
            current.lease_expires_at = moment + lease_timeout
            current.updated_at = moment
            raw[key] = current.to_dict()
            self._write_locked(raw)
            return True

    def complete(
        self,
        key: str,
        owner: str,
        *,
        result: Any = None,
        now: float | None = None,
    ) -> bool:
        """标记成功。仅租约持有者可写，终态不可覆盖。"""
        moment = now if now is not None else time.time()
        with self._locked():
            raw = self._read_locked()
            record = raw.get(key)
            if record is None:
                return False
            current = TaskState.from_dict(record)
            if current.status.terminal or current.lease_owner != owner:
                return False
            current.status = TaskStatus.COMPLETED
            current.result = self._codec.encode(result)
            current.result_kind = "value"
            current.error = None
            current.lease_owner = None
            current.lease_expires_at = None
            current.terminal_at = moment
            current.updated_at = moment
            raw[key] = current.to_dict()
            self._write_locked(raw)
            return True

    def fail(
        self,
        key: str,
        owner: str,
        *,
        error: BaseException | dict[str, str] | None = None,
        now: float | None = None,
    ) -> bool:
        """标记失败。仅租约持有者可写，终态不可覆盖。"""
        moment = now if now is not None else time.time()
        with self._locked():
            raw = self._read_locked()
            record = raw.get(key)
            if record is None:
                return False
            current = TaskState.from_dict(record)
            if current.status.terminal or current.lease_owner != owner:
                return False
            current.status = TaskStatus.FAILED
            current.error = (
                _serialize_error(error)
                if isinstance(error, BaseException)
                else error
            )
            current.lease_owner = None
            current.lease_expires_at = None
            current.terminal_at = moment
            current.updated_at = moment
            raw[key] = current.to_dict()
            self._write_locked(raw)
            return True

    def cancel(
        self, key: str, *, now: float | None = None
    ) -> TaskState | None:
        """落入 CANCELLED 终态。

        终态（COMPLETED / FAILED / CANCELLED）原样返回、不再覆盖；
        非持久任务（无记录）返回 ``None``。这保证取消与完成/接管
        竞争时，先落盘的终态获胜且唯一。
        """
        moment = now if now is not None else time.time()
        with self._locked():
            raw = self._read_locked()
            record = raw.get(key)
            if record is None:
                return None
            current = TaskState.from_dict(record)
            if not current.status.terminal:
                current.status = TaskStatus.CANCELLED
                current.lease_owner = None
                current.lease_expires_at = None
                current.terminal_at = moment
                current.updated_at = moment
                raw[key] = current.to_dict()
                self._write_locked(raw)
            return current

    def revoke_lease(
        self, key: str, *, now: float | None = None
    ) -> TaskState | None:
        """强制撤销一个仍在有效期的租约（供强制接管使用）。"""
        moment = now if now is not None else time.time()
        with self._locked():
            raw = self._read_locked()
            record = raw.get(key)
            if record is None:
                return None
            current = TaskState.from_dict(record)
            if current.status.terminal:
                return current
            if current.status in LEASED_STATUSES:
                current.status = TaskStatus.LEASE_EXPIRED
                current.lease_owner = None
                current.updated_at = moment
                raw[key] = current.to_dict()
                self._write_locked(raw)
            return current

    def disclaim(
        self,
        key: str,
        owner: str,
        *,
        now: float | None = None,
    ) -> bool:
        """持有者主动放弃尚未开始执行的受理。

        例如发现本机无法重建作业处理器。不产生终态：记录回到
        LEASE_EXPIRED，租约清空，以便具备条件的进程立即接管。
        已被他人接管或已有终态时不做改动。
        """
        moment = now if now is not None else time.time()
        with self._locked():
            raw = self._read_locked()
            record = raw.get(key)
            if record is None:
                return False
            current = TaskState.from_dict(record)
            if current.status.terminal:
                return False
            if current.lease_owner != owner:
                return False
            current.status = TaskStatus.LEASE_EXPIRED
            current.lease_owner = None
            current.lease_expires_at = None
            current.updated_at = moment
            raw[key] = current.to_dict()
            self._write_locked(raw)
            return True

    def expire_stale(self, *, now: float | None = None) -> list[TaskState]:
        """将所有停跳的受理/执行中任务标记为租约过期。"""
        moment = now if now is not None else time.time()
        expired: list[TaskState] = []
        with self._locked():
            raw = self._read_locked()
            dirty = False
            for key, record in raw.items():
                current = TaskState.from_dict(record)
                if (
                    current.status in LEASED_STATUSES
                    and current.lease_expires_at is not None
                    and current.lease_expires_at <= moment
                ):
                    current.status = TaskStatus.LEASE_EXPIRED
                    current.lease_owner = None
                    current.updated_at = moment
                    raw[key] = current.to_dict()
                    dirty = True
                    expired.append(current)
            if dirty:
                self._write_locked(raw)
            return expired

    def set_name(self, key: str, name: str | None) -> None:
        with self._locked():
            raw = self._read_locked()
            record = raw.get(key)
            if record is not None:
                record["name"] = name
                record["updated_at"] = time.time()
                self._write_locked(raw)

    def delete(self, key: str) -> bool:
        with self._locked():
            raw = self._read_locked()
            if key not in raw:
                return False
            del raw[key]
            self._write_locked(raw)
            return True


_stores: dict[str, TaskStore] = {}


def get_or_create_store(path: str | os.PathLike[str]) -> TaskStore:
    """进程内缓存的存储工厂，避免为同一路径重复初始化。"""
    resolved = os.path.abspath(os.fspath(path))
    if resolved not in _stores:
        _stores[resolved] = TaskStore(resolved)
    return _stores[resolved]
