# Sanic 服务框架

本项目提供异步 HTTP 服务、路由、蓝图、中间件、信号和工作进程管理能力。生产源码位于 `sanic/`，核心回归测试位于 `tests/`。

## 安装

`python3 -m pip install -e . pytest sanic-testing pytest-asyncio`

## 测试

`python3 -m pytest -q tests/test_blueprints.py tests/test_blueprint_group.py`

## 构建

`python3 -m compileall -q sanic`

## 使用

应用通过 `Sanic` 创建服务，通过蓝图组合路由，并可使用测试客户端完成本地 HTTP 验收。

## 持久化后台任务

运营作业会被请求、启动监听器和重试程序重复触发。普通 `app.add_task()` 仍是轻量临时任务；需要跨触发幂等、跨进程接管时使用持久任务：

```python
app.enable_persistent_tasks("./data/tasks.json")  # 或 config.TASK_STORE_PATH

async def daily_report(app, payload):
    ...

# 业务幂等键：重复提交拿到的是首次执行的结果，作业不会再次执行
result = await app.submit_task("ops:daily:2026-10-05",
                               daily_report, payload={"day": "2026-10-05"})
```

持久状态包含受理（accepted）、执行（running）、完成（completed）、失败（failed）、取消（cancelled）与租约过期（lease_expired）。执行者按租约心跳续约；进程崩溃后心跳停止，任务在租约到期后由新进程启动时自动接管（`attempts` 累加）。完成/失败/取消为终态且不可覆盖，因此取消与接管竞争只有一个终态；`app.cancel_persistent_task(key)` 取消，`app.takeover_task(key, force=True)` 强制接管，`app.get_task_state(key)` 查询状态。
