"""持久任务在真实 HTTP 服务中的端到端幂等性测试。

本文件不使用 pytest-asyncio：``app.test_client`` 需要自行驱动事件
循环启动服务（与 tests/test_create_task.py 一致）。
"""

from __future__ import annotations

import asyncio

from sanic import Sanic
from sanic.response import json as json_response


def test_http_endpoint_idempotent_submit(tmp_path):
    """请求与重试重复触发同一运营作业，作业只执行一次。"""
    store_path = str(tmp_path / "tasks.json")
    # test_client 每次请求都会重启应用，计数放到文件中以跨重启保留
    runs_file = tmp_path / "runs"

    app = Sanic("http-idem")
    app.config.TASK_STORE_PATH = store_path

    app.enable_persistent_tasks(
        store_path,
        lease_timeout=30,
        lease_refresh_interval=1,
        poll_interval=0.01,
    )

    @app.route("/submit/<key:str>")
    async def submit(request, key):
        async def job(app, payload=None):
            with open(runs_file, "a", encoding="utf-8") as fh:
                fh.write("x")
            await asyncio.sleep(0.05)
            return {
                "runs_at_execution": len(runs_file.read_text()),
                "key": key,
            }

        # 同一请求内模拟多个触发器并发提交
        results = await asyncio.gather(
            *[app.submit_task(key, job) for _ in range(3)]
        )
        return json_response({"results": results})

    @app.route("/runs")
    async def runs(request):
        return json_response(
            {"runs": len(runs_file.read_text()) if runs_file.exists() else 0}
        )

    _, response = app.test_client.get("/submit/ops-1")
    assert response.json["results"] == [
        {"runs_at_execution": 1, "key": "ops-1"}
    ] * 3

    # 重试程序 / 启动监听器再次提交：直接拿原任务结果，不再执行
    _, response = app.test_client.get("/submit/ops-1")
    assert response.json["results"] == [
        {"runs_at_execution": 1, "key": "ops-1"}
    ] * 3
    _, response = app.test_client.get("/runs")
    assert response.json == {"runs": 1}

    # 不同幂等键是不同作业
    _, response = app.test_client.get("/submit/ops-2")
    assert response.json["results"][0] == {
        "runs_at_execution": 2,
        "key": "ops-2",
    }
