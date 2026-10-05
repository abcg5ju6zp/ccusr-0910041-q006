"""持久任务的运行时管理器。

``TaskStore`` 只描述状态与迁移，本模块负责把状态机接到 asyncio
事件循环上：受理去重、派发执行、租约心跳、终态落盘、终态等待与
进程启动后的接管恢复。

关键约定：

* 重复提交同一业务幂等键不会再次执行：已有有效租约时等待对方
  终态，已有终态时直接重放原结果（成功还原返回值，失败重建
  异常，取消抛出 :class:`asyncio.CancelledError`）；
* 取消与执行/接管竞争时，终态以先落盘者为准，所有等待方拿到的
  都是同一个终态；执行者在租约失效后完成的结果会被拒（fencing），
  不会产生第二个终态；
* 工作进程异常退出后心跳停止，租约过期，新进程通过
  :meth:`PersistentTaskManager.recover` 安全接管，``attempts``
  累加；
* 普通的 :meth:`sanic.Sanic.add_task` 不经过本管理器，保持
  轻量与完全向后兼容。
"""

from __future__ import annotations

import asyncio
import builtins
import importlib
import inspect

from collections.abc import Awaitable, Callable, Coroutine
from contextlib import suppress
from typing import Any

from sanic.task_store import (
    DEFAULT_LEASE_REFRESH_INTERVAL,
    DEFAULT_LEASE_TIMEOUT,
    DEFAULT_POLL_INTERVAL,
    PersistentTaskError,
    TaskState,
    TaskStatus,
    TaskStore,
    worker_identity,
)


TaskHandler = Callable[..., Awaitable[Any]]
# 可接受协程函数，也可直接接受协程对象（后者仅当次执行有效，
# 跨进程接管仍以持久化的函数引用重建）
TaskLike = TaskHandler | Coroutine[Any, Any, Any]


def _consume_exception(task: asyncio.Task[Any]) -> None:
    """回收后台任务的终态异常，避免事件循环未回收告警。"""
    if task.cancelled():
        return
    task.exception()


def _qualified_name(func: Any) -> str:
    if inspect.iscoroutine(func):
        code = func.cr_code
        module = func.cr_frame.f_globals.get("__name__", "")
        qualname = getattr(code, "co_qualname", code.co_name)
        return f"{module}:{qualname}" if module else qualname
    module = getattr(func, "__module__", "")
    qualname = (
        getattr(func, "__qualname__", "")
        or getattr(func, "__name__", "")
    )
    return f"{module}:{qualname}" if module else qualname


def _resolve_handler(qualified: str) -> TaskHandler:
    """按 ``module:qualname`` 重建可调用对象。

    跨进程接管时作业必须可从其定义模块重新导入；局部闭包等无法
    导入的对象不能跨进程重建，应在应用启动时显式注册 handler。
    """
    module_name, _, qualname = qualified.partition(":")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


class _TerminalWatcher:
    """桥接同进程内的终态通知；跨进程靠轮询存储兜底。"""

    def __init__(self, poll_interval: float) -> None:
        self._poll_interval = poll_interval
        self._events: dict[str, asyncio.Event] = {}

    def _event(self, key: str) -> asyncio.Event:
        event = self._events.get(key)
        if event is None:
            event = asyncio.Event()
            self._events[key] = event
        return event

    def notify(self, key: str) -> None:
        self._event(key).set()

    async def wait(
        self,
        key: str,
        state_getter: Callable[[], TaskState | None],
    ) -> TaskState:
        event = self._event(key)
        while True:
            state = state_getter()
            if state is not None and state.status.terminal:
                return state
            # 同进程终态由 event 即时唤醒；event 未触发时轮询存储，
            # 以感知其它进程落盘的终态。注意不能 shield：否则超时
            # 取消只作用于外壳，内部 Event.wait() 会泄漏为悬挂任务。
            try:
                await asyncio.wait_for(
                    event.wait(), timeout=self._poll_interval
                )
            except asyncio.TimeoutError:
                continue


class PersistentTaskManager:
    """绑定一个 Sanic 应用与一个 :class:`TaskStore`。"""

    def __init__(
        self,
        app: Any,
        store: TaskStore,
        *,
        lease_timeout: float = DEFAULT_LEASE_TIMEOUT,
        lease_refresh_interval: float = DEFAULT_LEASE_REFRESH_INTERVAL,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> None:
        self.app = app
        self.store = store
        self.lease_timeout = lease_timeout
        self.lease_refresh_interval = min(
            lease_refresh_interval, lease_timeout / 2
        )
        self.poll_interval = poll_interval
        self.owner = worker_identity()
        # 本进程内正在执行的任务：key -> asyncio.Task
        self._running: dict[str, asyncio.Task[Any]] = {}
        # 租约心跳任务：key -> asyncio.Task
        self._heartbeats: dict[str, asyncio.Task[Any]] = {}
        # 显式登记的作业，优先于持久化的 handler 字符串使用
        self._handlers: dict[str, TaskHandler] = {}
        self._watcher = _TerminalWatcher(poll_interval)

    # ------------------------------------------------------------------ #
    # 注册 / 查询
    # ------------------------------------------------------------------ #

    def register_handler(self, key: str, handler: TaskHandler) -> None:
        """登记幂等键对应的作业，供本进程执行与接管时使用。"""
        self._handlers[key] = handler

    def get_state(self, key: str) -> TaskState | None:
        return self.store.get(key)

    async def wait_for_terminal(self, key: str) -> TaskState:
        """等待任务进入终态并返回其持久状态。"""
        return await self._watcher.wait(key, lambda: self.store.get(key))

    # ------------------------------------------------------------------ #
    # 提交（业务幂等入口）
    # ------------------------------------------------------------------ #

    async def submit(
        self,
        key: str,
        handler: TaskLike | None = None,
        *,
        name: str | None = None,
        payload: dict[str, Any] | None = None,
        lease_timeout: float | None = None,
    ) -> Any:
        """按业务幂等键提交作业并返回其结果。

        重复提交同一幂等键：

        * 作业在本进程或其它进程执行中：等待终态，不重复执行；
        * 作业已完成：直接返回首次执行持久化的结果；
        * 作业已失败：抛出重建的 :class:`PersistentTaskError`；
        * 作业被取消：抛出 :class:`asyncio.CancelledError`。

        ``payload`` 必须可被 JSON 序列化，会随状态一起落盘，使
        接管进程能用相同参数重放作业。
        """
        timeout = (
            self.lease_timeout if lease_timeout is None else lease_timeout
        )
        if handler is not None and not inspect.iscoroutine(handler):
            # 协程对象只能 await 一次，不放入跨期复用的注册表
            self._handlers.setdefault(key, handler)

        # claim 与 _running 登记之间没有 await：单事件循环下原子，
        # 同进程并发提交不会双双进入执行。
        state = self.store.claim(
            key,
            self.owner,
            lease_timeout=timeout,
            name=name,
            handler=_qualified_name(handler) if handler is not None else None,
            payload=payload,
        )

        if state.status.terminal or (
            state.lease_owner != self.owner
            and state.lease_owner is not None
        ):
            # 终态重放或等待他人时，调用方传入的一次性协程不会执行，
            # 立即关闭以避免 "coroutine never awaited" 告警。
            if inspect.iscoroutine(handler):
                handler.close()

        if state.status.terminal:
            return self._replay(state)

        if state.lease_owner == self.owner:
            running = self._running.get(key)
            if running is not None:
                # 本进程已经在执行（例如 recover 后又收到请求）
                if inspect.iscoroutine(handler):
                    handler.close()
                return await self._guarded(key, running)
            return await self._start(key, timeout, handler=handler)

        # 他人持有有效租约：等待终态，绝不重复执行
        terminal = await self._watcher.wait(key, lambda: self.store.get(key))
        return self._replay(terminal)

    # ------------------------------------------------------------------ #
    # 执行、心跳、fencing
    # ------------------------------------------------------------------ #

    async def _start(
        self,
        key: str,
        lease_timeout: float,
        *,
        handler: Any = None,
    ) -> Any:
        state = self.store.get(key)
        if state is None:  # pragma: no cover - 理论不可达
            raise PersistentTaskError(key, "KeyError", "task record vanished")

        # 先解析处理器，再迁移到 RUNNING：否则处理器缺失会留下一个
        # 无人续约、也无人能接管的 RUNNING 租约。
        if handler is None:
            handler = self._handlers.get(key)
        if handler is None and state.handler:
            try:
                handler = _resolve_handler(state.handler)
            except (ImportError, AttributeError, ValueError) as exc:
                # 持久化的作业引用本机无法重建：放弃受理，保持过期，
                # 留给具备该定义模块/显式注册了 handler 的进程。
                self.store.disclaim(key, self.owner)
                raise PersistentTaskError(
                    key,
                    "HandlerNotFound",
                    f"cannot resolve persisted handler "
                    f"{state.handler!r}: {exc}",
                )
        if handler is None:
            # 全新任务却没有任何可执行作业：无法被任何节点重放，
            # 落 FAILED 终态让问题显式可见，而非悬挂在受理状态。
            self.store.fail(
                key,
                self.owner,
                error=PersistentTaskError(
                    key,
                    "HandlerNotFound",
                    "submit requires a handler on first submission, or a "
                    "pre-registered handler",
                ),
            )
            self._watcher.notify(key)
            raise PersistentTaskError(
                key,
                "HandlerNotFound",
                "no handler available; submit a resolvable handler or use "
                "register_handler() first",
            )
        if not inspect.iscoroutine(handler):
            self._handlers.setdefault(key, handler)

        self.store.mark_running(key, self.owner)

        async def runner() -> Any:
            try:
                result = await self._invoke(handler, state)  # type: ignore
            except asyncio.CancelledError:
                # 显式取消时取消方已落 CANCELLED 终态；若是进程关闭
                # 导致的取消则尚无终态——不落盘，让租约自然过期，
                # 以便新进程接管。
                latest = self.store.get(key)
                if latest is not None and latest.status.terminal:
                    self._watcher.notify(key)
                raise
            except BaseException as exc:
                if self.store.fail(key, self.owner, error=exc):
                    self._watcher.notify(key)
                    raise
                # fail 被拒：已被取消或被接管，终态以存储为准
                self._watcher.notify(key)
                terminal = await self._watcher.wait(
                    key, lambda: self.store.get(key)
                )
                return self._replay(terminal)
            else:
                if self.store.complete(key, self.owner, result=result):
                    self._watcher.notify(key)
                    return result
                # 租约期间被他人接管：自己的结果必须丢弃（fencing），
                # 等待接管者落唯一终态后重放。
                self._watcher.notify(key)
                terminal = await self._watcher.wait(
                    key, lambda: self.store.get(key)
                )
                return self._replay(terminal)
            finally:
                self._running.pop(key, None)
                await self._stop_heartbeat(self._heartbeats.pop(key, None))

        task = asyncio.get_running_loop().create_task(runner(), name=key)
        self._running[key] = task
        self._start_heartbeat(key, lease_timeout)
        return await self._guarded(key, task)

    async def _guarded(
        self, key: str, task: asyncio.Task[Any]
    ) -> Any:
        """等待本进程任务。

        等待期间被取消（如请求方断开、进程关闭）时：若任务已有
        终态（例如被 manager.cancel 取消）则重放该终态；否则向上
        抛出 CancelledError——后台执行任务本身并不因此取消，
        仍可继续跑完并落盘，供后续提交复用。
        """
        try:
            return await task
        except asyncio.CancelledError:
            state = self.store.get(key)
            if state is not None and state.status.terminal:
                return self._replay(state)
            raise

    async def _invoke(
        self, handler: Any, state: TaskState
    ) -> Any:
        # 直接提交的协程对象（与 add_task 习惯一致）：单期执行。
        # 跨进程接管时 handler 会由持久化的函数引用重新解析为可调用对象。
        if inspect.iscoroutine(handler):
            return await handler
        if inspect.isawaitable(handler):
            return await handler

        payload = state.payload or {}
        # 按形参数量适配：handler() / handler(app) / handler(app, payload)
        try:
            positional = len(
                [
                    p
                    for p in inspect.signature(handler).parameters.values()
                    if p.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                ]
            )
        except (TypeError, ValueError):
            positional = 1

        if positional >= 2:
            result = handler(self.app, payload)
        elif positional == 1:
            result = handler(self.app)
        else:
            result = handler()

        if inspect.isawaitable(result):
            result = await result
        return result

    def _start_heartbeat(self, key: str, lease_timeout: float) -> None:
        interval = min(self.lease_refresh_interval, lease_timeout / 2)

        async def beat() -> None:
            while True:
                await asyncio.sleep(interval)
                if not self.store.heartbeat(
                    key, self.owner, lease_timeout=lease_timeout
                ):
                    return

        self._heartbeats[key] = asyncio.get_running_loop().create_task(
            beat(), name=f"sanic.task.lease.{key}"
        )

    async def _stop_heartbeat(
        self, beat_task: asyncio.Task[Any] | None
    ) -> None:
        if beat_task is not None and not beat_task.done():
            beat_task.cancel()
            with suppress(BaseException):
                await beat_task

    # ------------------------------------------------------------------ #
    # 取消
    # ------------------------------------------------------------------ #

    async def cancel(self, key: str, msg: str | None = None) -> TaskState:
        """请求取消并等待终态。

        先尝试落 CANCELLED 终态，再取消本地执行任务。若执行者
        抢先完成/失败，则保留先落盘的终态。竞争双方与所有等待方
        最终只会看到同一个终态。
        """
        state = self.store.cancel(key)
        if state is None:
            raise KeyError(key)

        running = self._running.get(key)
        if running is not None and not running.done():
            running.cancel(msg)
            self._watcher.notify(key)
            try:
                await running
            except BaseException:
                pass

        return await self._watcher.wait(key, lambda: self.store.get(key))

    # ------------------------------------------------------------------ #
    # 接管 / 恢复
    # ------------------------------------------------------------------ #

    async def takeover(
        self,
        key: str,
        handler: TaskLike | None = None,
        *,
        force: bool = False,
        lease_timeout: float | None = None,
    ) -> Any:
        """接管单个任务并返回其结果。

        默认只接管受理滞留或租约已过期的任务；任务已有终态时
        直接重放原结果。``force=True`` 用于运维强制接管，会撤销
        仍然有效的租约。
        """
        if force:
            state = self.store.revoke_lease(key)
            if state is None:
                raise KeyError(key)
            if state.status.terminal:
                return self._replay(state)
        if handler is not None and not inspect.iscoroutine(handler):
            self._handlers.setdefault(key, handler)
        return await self.submit(key, handler, lease_timeout=lease_timeout)

    async def recover(
        self,
        *,
        lease_timeout: float | None = None,
    ) -> dict[str, asyncio.Task[Any]]:
        """进程启动后恢复可接管任务。

        * 租约停跳的 ACCEPTED / RUNNING：标记过期并接管执行；
        * 显式 LEASE_EXPIRED：直接接管；
        * 租约仍有效（同机另有活进程）或已终态：跳过。

        返回在后台恢复执行的 asyncio 任务映射；调用方可自行等待。
        """
        timeout = (
            self.lease_timeout if lease_timeout is None else lease_timeout
        )
        self.store.expire_stale()

        recovered: dict[str, asyncio.Task[Any]] = {}
        for key, state in self.store.load().items():
            if key in self._running or state.status.terminal:
                continue
            if state.lease_is_active():
                continue
            if state.status not in (
                TaskStatus.LEASE_EXPIRED,
                TaskStatus.ACCEPTED,
                TaskStatus.RUNNING,
            ):
                continue
            # 无法重建作业的 worker 不应受理：保持 LEASE_EXPIRED，
            # 留给显式注册了 handler 或能导入其定义模块的进程。
            if not self._has_handler(key, state):
                continue
            claimed = self.store.claim(key, self.owner, lease_timeout=timeout)
            if claimed.lease_owner != self.owner or claimed.status.terminal:
                continue
            task = asyncio.get_running_loop().create_task(
                self._start(key, timeout), name=key
            )
            self._running[key] = task
            # 后台接管任务无人 await：回收终态异常，避免事件循环告警。
            # 失败已经持久化，调用方可通过 get_state / submit 重放。
            task.add_done_callback(_consume_exception)
            recovered[key] = task
        return recovered

    def _has_handler(self, key: str, state: TaskState) -> bool:
        """本进程是否具备执行该任务所需的作业。"""
        if key in self._handlers:
            return True
        if not state.handler:
            return False
        try:
            self._handlers[key] = _resolve_handler(state.handler)
        except (ImportError, AttributeError, ValueError):
            return False
        return True

    # ------------------------------------------------------------------ #
    # 结果重放与关闭
    # ------------------------------------------------------------------ #

    def _replay(self, state: TaskState) -> Any:
        if state.status is TaskStatus.COMPLETED:
            return state.result
        if state.status is TaskStatus.CANCELLED:
            raise asyncio.CancelledError(
                f"Persistent task {state.key!r} was cancelled"
            )
        if state.status is TaskStatus.FAILED:
            raise self._build_error(state)
        raise RuntimeError(
            f"Task {state.key!r} is not terminal: {state.status.value}"
        )

    @staticmethod
    def _build_error(state: TaskState) -> BaseException:
        """优先重建原异常类型，失败时回退为 PersistentTaskError。"""
        error = state.error or {}
        error_type = error.get("type", "Exception")
        message = error.get("message", "")
        module = error.get("module")
        if module and module != "builtins":
            try:
                obj: Any = importlib.import_module(module)
                for part in error_type.split("."):
                    obj = getattr(obj, part)
                if isinstance(obj, type) and issubclass(obj, BaseException):
                    try:
                        return obj(message)
                    except TypeError:
                        return obj()
            except (ImportError, AttributeError, TypeError):
                pass
        elif error_type in {
            name
            for name in dir(builtins)
            if isinstance(getattr(builtins, name), type)
        }:
            cls = getattr(builtins, error_type)
            if isinstance(cls, type) and issubclass(cls, BaseException):
                try:
                    return cls(message)
                except TypeError:
                    return cls()
        return PersistentTaskError(state.key, error_type, message)

    async def aclose(self, drain_timeout: float = 5.0) -> None:
        """停止心跳并优雅排空本进程执行中的任务。

        取消执行中的任务并等待它们收尾，但不写终态：被中断的任务
        在存储中保持非终态（RUNNING），租约到期后可被新进程接管；
        已经完成或失败的任务其终态不受影响。

        ``drain_timeout`` 限制排空等待时长，避免拖住整个关闭流程。
        需要确定性取消终态时应在关闭前显式调用 :meth:`cancel`。
        """
        beat_tasks = list(self._heartbeats.values())
        self._heartbeats.clear()
        for beat_task in beat_tasks:
            if not beat_task.done():
                beat_task.cancel()

        running = [
            task
            for task in self._running.values()
            if not task.done()
        ]
        for task in running:
            task.cancel()

        if running or beat_tasks:
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    asyncio.gather(
                        *running,
                        *beat_tasks,
                        return_exceptions=True,
                    ),
                    timeout=drain_timeout,
                )

    def close(self) -> None:
        """同步关闭：尽力停止心跳（供无法 await 的关闭路径使用）。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        beat_tasks = list(self._heartbeats.values())
        self._heartbeats.clear()
        for beat_task in beat_tasks:
            if not beat_task.done():
                beat_task.cancel()
        if loop is not None and loop.is_running():
            # 交由事件循环回收，避免“从未获取结果”的取消告警
            asyncio.ensure_future(self._swallow(beat_tasks), loop=loop)

    @staticmethod
    async def _swallow(tasks: list[asyncio.Task[Any]]) -> None:
        for task in tasks:
            with suppress(BaseException):
                await task
