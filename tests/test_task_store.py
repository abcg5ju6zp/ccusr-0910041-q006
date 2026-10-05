"""TaskStore：持久状态机与文件锁存储的单元测试。"""

from __future__ import annotations

import json
import os

from concurrent.futures import ThreadPoolExecutor

import pytest

from sanic.task_store import (
    TaskStatus,
    TaskStore,
    worker_identity,
)


@pytest.fixture
def store(tmp_path):
    return TaskStore(tmp_path / "tasks.json")


def test_claim_creates_accepted_with_lease(store):
    state = store.claim("k", "owner-1", lease_timeout=1.0)

    assert state.key == "k"
    assert state.status is TaskStatus.ACCEPTED
    assert state.lease_owner == "owner-1"
    assert state.attempts == 1
    assert state.lease_expires_at > state.created_at
    assert state.lease_is_active()


def test_concurrent_claim_only_one_owner(store):
    owners = [f"owner-{i}" for i in range(20)]

    def claim(owner):
        return store.claim("race", owner, lease_timeout=5.0)

    with ThreadPoolExecutor(max_workers=8) as pool:
        states = list(pool.map(claim, owners))

    # 无论谁先抢到，所有观察者看到的都是同一个持有者，
    # 且有效租约期内没有发生接管（attempts 保持 1）。
    observed_owners = {s.lease_owner for s in states}
    assert observed_owners == {states[0].lease_owner}
    final = store.get("race")
    assert final.attempts == 1
    assert final.lease_owner in owners


def test_claim_after_lease_expiry_takes_over(store):
    t = 1000.0
    first = store.claim("k", "owner-1", lease_timeout=1.0, now=t)
    assert first.status is TaskStatus.ACCEPTED

    store.mark_running("k", "owner-1", now=t)
    running = store.get("k")
    assert running.status is TaskStatus.RUNNING

    # 租约未到期：接管被拒
    blocked = store.claim("k", "owner-2", lease_timeout=1.0, now=t + 0.5)
    assert blocked.lease_owner == "owner-1"

    # 租约到期：接管成功，attempts 累加
    second = store.claim("k", "owner-2", lease_timeout=1.0, now=t + 1.1)
    assert second.status is TaskStatus.RUNNING
    assert second.lease_owner == "owner-2"
    assert second.attempts == 2


def test_terminal_states_are_immutable(store):
    t = 1000.0
    store.claim("ok", "a", lease_timeout=1.0, now=t)
    store.mark_running("ok", "a", now=t)
    assert store.complete("ok", "a", result=1, now=t + 0.1)
    # 后来者无法覆盖终态
    assert not store.complete("ok", "b", result=2, now=t + 0.2)
    assert not store.fail("ok", "b", error=RuntimeError("x"), now=t + 0.2)
    cancelled = store.cancel("ok", now=t + 0.3)
    assert cancelled.status is TaskStatus.COMPLETED
    assert store.get("ok").result == 1

    store.claim("bad", "a", lease_timeout=1.0, now=t)
    store.mark_running("bad", "a", now=t)
    assert store.fail("bad", "a", error=ValueError("boom"), now=t + 0.1)
    assert not store.complete("bad", "a", result=9, now=t + 0.2)
    assert store.get("bad").status is TaskStatus.FAILED

    store.claim("gone", "a", lease_timeout=1.0, now=t)
    assert store.cancel("gone", now=t + 0.1).status is TaskStatus.CANCELLED
    # 取消终态同样不可改
    assert not store.complete("gone", "a", result=3, now=t + 0.2)
    assert store.cancel("gone", now=t + 0.3).status is TaskStatus.CANCELLED


def test_complete_requires_lease_owner_fencing(store):
    t = 1000.0
    store.claim("k", "a", lease_timeout=10.0, now=t)
    store.mark_running("k", "a", now=t)
    # 冒充者的结果必须被拒
    assert not store.complete("k", "intruder", result="x")
    assert not store.fail("k", "intruder", error=RuntimeError())
    assert store.get("k").status is TaskStatus.RUNNING
    assert store.complete("k", "a", result="y")


def test_heartbeat_only_owner_and_refresh(store):
    t = 1000.0
    store.claim("k", "a", lease_timeout=1.0, now=t)
    assert not store.heartbeat("k", "b", lease_timeout=1.0, now=t + 0.1)
    assert store.heartbeat("k", "a", lease_timeout=1.0, now=t + 0.9)
    # 续约后原到期时间点仍有效
    assert store.get("k").lease_is_active(now=t + 1.5)
    assert not store.get("k").lease_is_active(now=t + 2.0)


def test_heartbeat_stops_after_terminal(store):
    store.claim("k", "a", lease_timeout=10.0)
    store.mark_running("k", "a")
    store.cancel("k")
    assert not store.heartbeat("k", "a", lease_timeout=10.0)


def test_expire_stale_marks_dead_leases(store):
    t = 1000.0
    store.claim("dead", "a", lease_timeout=1.0, now=t)
    store.mark_running("dead", "a", now=t)
    store.claim("alive", "b", lease_timeout=10.0, now=t)
    store.mark_running("alive", "b", now=t)

    expired = store.expire_stale(now=t + 5)
    assert [s.key for s in expired] == ["dead"]
    assert store.get("dead").status is TaskStatus.LEASE_EXPIRED
    assert store.get("dead").lease_owner is None
    assert store.get("alive").status is TaskStatus.RUNNING


def test_revoke_active_lease_allows_force_takeover(store):
    t = 1000.0
    store.claim("k", "a", lease_timeout=100.0, now=t)
    state = store.revoke_lease("k", now=t + 1)
    assert state.status is TaskStatus.LEASE_EXPIRED
    second = store.claim("k", "b", lease_timeout=100.0, now=t + 1)
    assert second.lease_owner == "b"
    assert second.attempts == 2
    # 终态不受 revoke 影响
    store.complete("k", "b", result=1, now=t + 2)
    assert store.revoke_lease("k").status is TaskStatus.COMPLETED


def test_state_persists_across_store_instances(tmp_path):
    path = tmp_path / "tasks.json"
    first = TaskStore(path)
    first.claim("k", "a", lease_timeout=100.0, name="n",
                handler="mod:fn", payload={"x": 1})
    first.mark_running("k", "a")
    first.complete("k", "a", result={"v": [1, 2, 3]})

    reopened = TaskStore(path)
    state = reopened.get("k")
    assert state.status is TaskStatus.COMPLETED
    assert state.result == {"v": [1, 2, 3]}
    assert state.payload == {"x": 1}
    assert state.handler == "mod:fn"
    assert state.name == "n"

    with open(path, encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk["k"]["status"] == "completed"
    assert on_disk["k"]["terminal_at"] is not None


def test_atomic_write_leaves_no_temp_files(store, tmp_path):
    for i in range(5):
        store.claim(f"k{i}", "a", lease_timeout=100.0)
    leftovers = [
        p for p in os.listdir(tmp_path) if p.startswith(".task_store.")
    ]
    assert leftovers == []


def test_worker_identity_distinguishes_processes():
    # 同进程多次调用稳定（线程标识相同），pid 前缀保证跨进程不撞
    ident = worker_identity()
    assert ident.startswith(f"{os.getpid()}-")
    assert isinstance(ident, str) and ident


def test_cancel_unknown_key_returns_none(store):
    assert store.cancel("missing") is None
