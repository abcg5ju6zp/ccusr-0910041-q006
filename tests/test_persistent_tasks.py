"""PersistentTaskManager 与 Sanic 持久任务 API 的集成测试。"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from sanic import Sanic
from sanic.application.state import (
    ApplicationServerInfo,
    ServerStage,
)
from sanic.exceptions import SanicException
from sanic.task_store import TaskStatus


pytestmark = pytest.mark.asyncio


def mark_serving(app: Sanic) -> None:
    """与 tests/test_tasks.py 一致：标记应用处于 SERVING。"""
    app.state.server_info.append(
        ApplicationServerInfo(
            stage=ServerStage.SERVING,
            settings={},
            server=object(),
        )
    )


def make_app(name, path, **kwargs):
    app = Sanic(name)
    app.enable_persistent_tasks(
        path,
        auto_recover=False,
        **kwargs,
    )
    return app


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path / "tasks.json")


async def _module_level_job(app, payload=None):
    """模块级作业：可被跨进程通过 module:qualname 重新导入。"""
    # 固定阻塞一小段，使首个执行者能在执行中被"杀掉"
    await asyncio.sleep(0.4)
    return (payload or {}).get("v", 0) * 2


async def test_submit_returns_result_and_persists(store_path):
    app = make_app("a", store_path, lease_timeout=5, poll_interval=0.01)

    async def job(app, payload):
        return {"v": payload["x"] + 1}

    result = await app.submit_task("k", job, payload={"x": 41})

    assert result == {"v": 42}
    state = app.get_task_state("k")
    assert state.status is TaskStatus.COMPLETED
    assert state.result == {"v": 42}
    assert state.attempts == 1
    assert state.terminal_at is not None


async def test_duplicate_submit_does_not_rerun(store_path):
    app = make_app("a", store_path, lease_timeout=5, poll_interval=0.01)
    runs = 0

    async def job(app, payload=None):
        nonlocal runs
        runs += 1
        await asyncio.sleep(0.02)
        return runs

    results = await asyncio.gather(
        app.submit_task("k", job, payload={}),
        app.submit_task("k", job, payload={}),
        app.submit_task("k", job, payload={}),
    )
    assert results == [1, 1, 1]
    assert runs == 1

    # 进程内"再次受理"（新事件循环视角）仍然只拿原结果
    assert await app.submit_task("k", job) == 1
    assert runs == 1


async def test_waiter_gets_original_result_without_executing(store_path):
    """第二个管理器（模拟另一进程）在租约有效期内等待，不执行。"""
    app_a = make_app(
        "a", store_path, lease_timeout=10, poll_interval=0.01
    )
    app_b = make_app(
        "b", store_path, lease_timeout=10, poll_interval=0.01
    )

    started = asyncio.Event()

    async def job_a(app, payload=None):
        started.set()
        await asyncio.sleep(0.15)
        return "from-a"

    async def job_b(app, payload=None):
        raise AssertionError("must not execute")

    task_a = asyncio.create_task(app_a.submit_task("k", job_a))
    await started.wait()
    # b 在 a 执行期间重复提交：等待并拿到 a 的结果
    result_b = await app_b.submit_task("k", job_b)
    assert result_b == "from-a"
    assert await task_a == "from-a"
    assert app_b.get_task_state("k").attempts == 1


async def test_failure_terminal_is_replayed_not_rerun(store_path):
    app = make_app("a", store_path, lease_timeout=5, poll_interval=0.01)
    runs = 0

    async def boom(app, payload=None):
        nonlocal runs
        runs += 1
        raise ValueError("kaboom")

    for _ in range(3):
        with pytest.raises(ValueError, match="kaboom"):
            await app.submit_task("fail", boom)

    state = app.get_task_state("fail")
    assert state.status is TaskStatus.FAILED
    assert state.error["type"] == "ValueError"
    assert state.error["module"] == "builtins"
    assert runs == 1


async def test_cancelled_terminal_is_replayed(store_path):
    app = make_app("a", store_path, lease_timeout=5, poll_interval=0.01)

    async def slow(app, payload=None):
        await asyncio.sleep(100)
        return "nope"

    task = asyncio.create_task(app.submit_task("k", slow))
    await asyncio.sleep(0.02)

    state = await app.cancel_persistent_task("k")
    assert state.status is TaskStatus.CANCELLED
    with pytest.raises(asyncio.CancelledError):
        await task

    # 再次提交与强制接管都只能重放取消终态
    with pytest.raises(asyncio.CancelledError):
        await app.submit_task("k", slow)
    with pytest.raises(asyncio.CancelledError):
        await app.takeover_task("k", slow, force=True)
    assert app.get_task_state("k").attempts == 1


async def test_cancel_loses_to_completion_single_terminal(store_path):
    """执行者抢先完成时，取消不能覆盖 COMPLETED 终态。"""
    app = make_app("a", store_path, lease_timeout=5, poll_interval=0.01)
    gate = asyncio.Event()

    async def job(app, payload=None):
        await gate.wait()
        return "done"

    task = asyncio.create_task(app.submit_task("k", job))
    await asyncio.sleep(0.02)  # 确保进入 RUNNING

    async def complete_then_cancel():
        gate.set()
        result = await task
        # 完成落盘后再取消：终态保持 COMPLETED
        state = await app.cancel_persistent_task("k")
        return result, state

    result, state = await complete_then_cancel()
    assert result == "done"
    assert state.status is TaskStatus.COMPLETED
    assert app.get_task_state("k").result == "done"


async def test_cancel_vs_competition_has_exactly_one_terminal(store_path):
    """并发取消与完成的压力测试：终态始终唯一且各方一致。"""
    app = make_app("a", store_path, lease_timeout=5, poll_interval=0.005)

    for i in range(30):
        key = f"race-{i}"
        release = asyncio.Event()

        async def job(app, payload=None, release=release):
            await release.wait()
            return f"ok-{key}"

        runner = asyncio.create_task(app.submit_task(key, job))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        async def cancel_soon():
            await asyncio.sleep(0.001 * (i % 3))
            return await app.cancel_persistent_task(key)

        canceller = asyncio.create_task(cancel_soon())
        release.set()

        state = await canceller
        assert state.status in (
            TaskStatus.COMPLETED,
            TaskStatus.CANCELLED,
        )

        # 提交方的结局必须与唯一终态一致：完成则拿结果，取消则抛 CE
        if state.status is TaskStatus.COMPLETED:
            assert await runner == f"ok-{key}"
        else:
            with pytest.raises(asyncio.CancelledError):
                await runner

        # 持久终态唯一且自洽
        persisted = app.get_task_state(key)
        assert persisted.status is state.status
        if state.status is TaskStatus.COMPLETED:
            assert persisted.result == f"ok-{key}"
        # 落盘后再取消/完成都不能改变终态
        again = await app.cancel_persistent_task(key)
        assert again.status is state.status


async def test_lease_expiry_then_takeover(store_path):
    app_a = make_app(
        "a",
        store_path,
        lease_timeout=0.2,
        lease_refresh_interval=0.03,
        poll_interval=0.01,
    )

    async def dying(app, payload=None):
        await asyncio.sleep(1000)

    runner = asyncio.create_task(
        app_a.submit_task("ops", dying, payload={"n": 7})
    )
    await asyncio.sleep(0.02)
    assert app_a.get_task_state("ops").status is TaskStatus.RUNNING

    # 模拟工作进程崩溃：执行任务与心跳全部消失，无终态落盘
    app_a.persistent_tasks._running["ops"].cancel()
    for beat in list(app_a.persistent_tasks._heartbeats.values()):
        beat.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner
    await asyncio.sleep(0.25)  # 等租约过期

    app_b = make_app(
        "b",
        store_path,
        lease_timeout=5,
        lease_refresh_interval=0.1,
        poll_interval=0.01,
    )
    seen_payload = {}

    async def finished(app, payload):
        seen_payload.update(payload)
        return "recovered"

    app_b.register_task_handler("ops", finished)
    recovered = await app_b.recover_persistent_tasks()
    assert set(recovered) == {"ops"}
    assert await recovered["ops"] == "recovered"

    state = app_b.get_task_state("ops")
    assert state.status is TaskStatus.COMPLETED
    assert state.attempts == 2
    assert seen_payload == {"n": 7}


async def test_recover_skips_active_and_terminal(store_path):
    app_a = make_app(
        "a", store_path, lease_timeout=10, poll_interval=0.01
    )
    app_b = make_app(
        "b", store_path, lease_timeout=10, poll_interval=0.01
    )
    running = asyncio.Event()

    async def active(app, payload=None):
        running.set()
        await asyncio.sleep(0.2)
        return 1

    async def done(app, payload=None):
        return 2

    t = asyncio.create_task(app_a.submit_task("active", active))
    await running.wait()
    assert await app_a.submit_task("done", done) == 2

    # a 持有效租约、done 已终态：b 没有可接管任务
    recovered = await app_b.recover_persistent_tasks()
    assert recovered == {}
    assert await t == 1


async def test_recover_skips_when_handler_unresolvable(store_path):
    app_a = make_app(
        "a",
        store_path,
        lease_timeout=0.2,
        lease_refresh_interval=0.03,
        poll_interval=0.01,
    )

    async def dying(app, payload=None):
        await asyncio.sleep(1000)

    runner = asyncio.create_task(app_a.submit_task("orphan", dying))
    await asyncio.sleep(0.02)
    app_a.persistent_tasks._running["orphan"].cancel()
    for beat in list(app_a.persistent_tasks._heartbeats.values()):
        beat.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner
    await asyncio.sleep(0.25)

    # 新进程没有也无法解析这个局部闭包：不得受理，保持过期
    app_b = make_app("b", store_path, poll_interval=0.01)
    recovered = await app_b.recover_persistent_tasks()
    assert recovered == {}
    state = app_b.get_task_state("orphan")
    assert state.status is TaskStatus.LEASE_EXPIRED

    # 显式登记作业后即可接管
    async def saved(app, payload=None):
        return "saved"

    app_b.register_task_handler("orphan", saved)
    recovered = await app_b.recover_persistent_tasks()
    assert await recovered["orphan"] == "saved"


async def test_module_level_handler_auto_resolved_on_takeover(store_path):
    """持久化的 handler 引用支持跨管理器自动重建。"""
    from tests.test_persistent_tasks import _module_level_job

    app_a = make_app(
        "a",
        store_path,
        lease_timeout=0.2,
        lease_refresh_interval=0.03,
        poll_interval=0.01,
    )
    runner = asyncio.create_task(
        app_a.submit_task("mlk", _module_level_job, payload={"v": 5})
    )
    await asyncio.sleep(0.02)
    app_a.persistent_tasks._running["mlk"].cancel()
    for beat in list(app_a.persistent_tasks._heartbeats.values()):
        beat.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner
    await asyncio.sleep(0.25)

    # 不显式注册 handler：recover 依据持久化的 module:qualname 重建
    app_b = make_app("b", store_path, poll_interval=0.01)
    recovered = await app_b.recover_persistent_tasks()
    assert await recovered["mlk"] == 10


async def test_aclose_drains_without_writing_terminal(store_path):
    """关闭时取消本地执行任务但不落终态：任务仍可被接管。"""
    app = make_app(
        "a",
        store_path,
        lease_timeout=100,
        lease_refresh_interval=1,
        poll_interval=0.01,
    )
    started = asyncio.Event()

    async def slow(app, payload=None):
        started.set()
        await asyncio.sleep(1000)

    runner = asyncio.create_task(app.submit_task("k", slow))
    await started.wait()

    await app.persistent_tasks.aclose(drain_timeout=2.0)
    with pytest.raises(asyncio.CancelledError):
        await runner

    state = app.get_task_state("k")
    # 没有终态：保留 RUNNING，租约到期后由新进程接管
    assert state.status is TaskStatus.RUNNING
    assert state.attempts == 1


async def test_force_takeover_revokes_healthy_lease(store_path):
    app_a = make_app(
        "a", store_path, lease_timeout=100, poll_interval=0.01
    )
    app_b = make_app(
        "b", store_path, lease_timeout=100, poll_interval=0.01
    )
    started = asyncio.Event()

    async def first(app, payload=None):
        started.set()
        await asyncio.sleep(1000)

    async def second(app, payload=None):
        return "forced"

    runner = asyncio.create_task(app_a.submit_task("k", first))
    await started.wait()

    result = await app_b.takeover_task("k", second, force=True)
    assert result == "forced"
    state = app_b.get_task_state("k")
    assert state.status is TaskStatus.COMPLETED
    assert state.attempts == 2
    # 旧执行者已失去租约：它的任何结果都会被 fencing 拒绝；
    # 中止它（取消后它只能看到 b 落的 COMPLETED 终态）。
    runner.cancel()
    try:
        old_result = await runner
    except asyncio.CancelledError:
        pass
    else:
        assert old_result == "forced"


async def test_handler_signature_variants(store_path):
    app = make_app("a", store_path, lease_timeout=5, poll_interval=0.01)

    async def no_args():
        return "z"

    async def app_only(app):
        return app.name

    async def app_payload(app, payload):
        return payload["x"]

    assert await app.submit_task("n0", no_args) == "z"
    assert await app.submit_task("n1", app_only) == "a"
    assert (
        await app.submit_task("n2", app_payload, payload={"x": 9})
        == 9
    )


async def test_coroutine_object_submission(store_path):
    """与 add_task 习惯一致，直接提交协程对象也可执行并持久化。"""
    app = make_app("a", store_path, lease_timeout=5, poll_interval=0.01)

    async def returns(x):
        return x

    result = await app.submit_task("co", returns(77))
    assert result == 77
    state = app.get_task_state("co")
    assert state.status is TaskStatus.COMPLETED
    # 重复提交仍只拿原结果
    assert await app.submit_task("co", returns(77)) == 77


async def test_lightweight_add_task_unaffected(store_path):
    """普通临时任务仍然轻量：无需持久管理器参与。"""
    app = Sanic("plain")
    mark_serving(app)
    assert app.get_task_state("anything") is None
    with pytest.raises(SanicException):
        _ = app.persistent_tasks

    async def quick():
        return 123

    # 未启用持久任务时 add_task 的既有行为不变
    task = app.add_task(quick())
    assert await task == 123
    assert app._persistent_tasks is None


async def test_enable_twice_returns_same_manager(store_path):
    app = Sanic("a")
    first = app.enable_persistent_tasks(store_path, auto_recover=False)
    second = app.enable_persistent_tasks(store_path, auto_recover=False)
    assert first is second


async def test_cancel_unknown_key_raises(store_path):
    app = make_app("a", store_path)
    with pytest.raises(KeyError):
        await app.cancel_persistent_task("missing")


async def test_submit_without_explicit_enable_uses_default_store(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    app = Sanic("auto")

    async def job(app, payload=None):
        return "auto"

    assert await app.submit_task("k", job) == "auto"
    assert app.get_task_state("k").status is TaskStatus.COMPLETED
    assert (tmp_path / ".sanic" / "task_store_auto.json").exists()


# --------------------------------------------------------------------- #
# 真实跨进程接管（文件锁 + JSON 持久化端到端）
# --------------------------------------------------------------------- #

WORKER_SCRIPT = '''
import asyncio
import json
import os
import sys

from sanic import Sanic


STORE = sys.argv[2]
RUNS = sys.argv[3]


def _touch(name):
    with open(RUNS, "a", encoding="utf-8") as fh:
        fh.write(name)


async def ops_job(app, payload=None):
    _touch("run")
    await asyncio.sleep(30)
    return "should-not-return"


async def finish_job(app, payload=None):
    _touch("takeover")
    return "finished-by-new-process"


async def phase_one():
    app = Sanic("proc-a")
    app.enable_persistent_tasks(
        STORE,
        lease_timeout=0.5,
        lease_refresh_interval=0.1,
        auto_recover=False,
    )
    task = asyncio.create_task(
        app.submit_task("ops/daily", ops_job)
    )
    # 等到状态确认为 RUNNING 后直接"猝死"：不写终态、无优雅退出
    while True:
        state = app.get_task_state("ops/daily")
        if state is not None and state.status.value == "running":
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    os._exit(0)


async def phase_two():
    app = Sanic("proc-b")
    mgr = app.enable_persistent_tasks(
        STORE,
        lease_timeout=5.0,
        lease_refresh_interval=0.5,
        poll_interval=0.02,
        auto_recover=False,
    )
    mgr.register_handler("ops/daily", finish_job)
    recovered = await mgr.recover()
    assert set(recovered) == {"ops/daily"}, list(recovered)
    result = await recovered["ops/daily"]
    assert result == "finished-by-new-process", result
    state = mgr.get_state("ops/daily")
    assert state.status.value == "completed"
    assert state.attempts == 2
    with open(sys.argv[4], "w", encoding="utf-8") as fh:
        json.dump(state.to_dict(), fh)


if __name__ == "__main__":
    phase = sys.argv[1]
    if phase == "one":
        asyncio.run(phase_one())
    else:
        asyncio.run(phase_two())
'''


async def test_real_process_takeover_after_crash(tmp_path):
    import subprocess
    import sys

    store = tmp_path / "tasks.json"
    runs = tmp_path / "runs.txt"
    state_out = tmp_path / "final_state.json"
    script = tmp_path / "worker.py"
    script.write_text(WORKER_SCRIPT)

    env = dict(os.environ)
    # 确保子进程使用本仓库的 sanic
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = (
        repo_root + os.pathsep + env.get("PYTHONPATH", "")
    )

    subprocess.run(
        [sys.executable, str(script), "one", str(store), str(runs)],
        check=True,
        env=env,
        timeout=30,
    )

    import time as _time

    _time.sleep(0.7)  # 等租约过期

    subprocess.run(
        [
            sys.executable,
            str(script),
            "two",
            str(store),
            str(runs),
            str(state_out),
        ],
        check=True,
        env=env,
        timeout=30,
    )

    final_state = json.loads(state_out.read_text())
    assert final_state["status"] == "completed"
    assert final_state["attempts"] == 2
    assert final_state["result"] == "finished-by-new-process"
    # 第一个进程崩溃前执行过一次，新进程接管后又执行一次
    assert runs.read_text() == "runtakeover"
