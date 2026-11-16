# Server 六阶段交付核对

核对日期：2026-11-15。依据 [原方案](server_design.md) 的 R01–R28、Q01–Q12；用户已要求不做性能测试、不扩大设计和测试范围。此表整理实际代码和已有功能证据，不以测试数量替代验收，也不把后续增强建议变成新门槛。

本地环境为 macOS、Python 3.14、真实 nndeploy 原生扩展、SQLite/MySQL/PostgreSQL、Redis、MinIO。真实 COS/OSS/OBS 与 GPU 按用户安排稍后验收。MinIO 结果不代表其他云厂商，受控设备预算测试不代表真实 GPU。

本地交付环境已切换为 `.venv` 的源码可编辑安装，当前编译的原生库同步到 `python/nndeploy`，不再需要此前测试专用的临时 PYTHONPATH。切换前的安装包与原生库已备份，位置记录在 `logs/server-development/local-install-backup-path`。普通 `.venv/bin/python` 下的 API/CLI 回归 24 项通过（`local-install-smoke.log`）。本地编辑器在 `http://127.0.0.1:8000` 启动，使用当前前端构建；管理登录令牌位于仅当前用户可读的 `logs/server-development/local-admin-token`。启动参数、PID 和日志位置见 `local-server.json`，健康、管理接口与 OpenAPI 检查见 `local-server-checks.json`；这些运行状态以实际进程为准。

## 六阶段

| 阶段 | 当前实现与证据 | 验收边界 |
| --- | --- | --- |
| A：现有 Server 改造 | 公共 Server、API-only、GraphRunner、旧编辑器 Run/取消/日志/插件及进程监督；`test_server_api.py`、`test_editor_*` 和浏览器记录 | 本地已覆盖；不扩展为任意插件内部阻塞的硬取消保证 |
| B：工作流与代码接入 | JSON、直接 Graph、Graph 工厂、函数/对象；HTTP Deploy、绑定、预热、Start/Stop/Reload、独立队列；原生图片/视频/数组/音频用例 | 已有端口明确支持或拒绝；不要求为所有第三方模型另写适配器 |
| C：数据库 | SQLAlchemy、Alembic 至 `0033_editor_input_files`、三数据库事务/迁移/幂等/记录/恢复及备份迁移 | 本地真实三数据库已验证；特定远程数据库运维环境另验 |
| D：存储 | 统一 File/Storage、本地和 S3 内容校验/清理、云 SDK 实现、模型副本生命周期 | 本地及 MinIO 已验证；COS/OSS/OBS 真实账号读写仍待验收 |
| E：调度 | Redis Streams、按入口共享队列、Worker 心跳、任务归属、取消/期限、设备预算及事件通知 | 本地多进程/多实例有证据；真实 GPU 和跨主机运行环境未据此宣称通过 |
| F：扩展与公共能力 | SSE/WebSocket、专用算法路由、批量/动态合批、鉴权/配额/审计、日志与 Prometheus | 公共协议有功能测试；具体模型的 Token/GPU 指标随模型环境验收，无性能门槛 |

## 功能需求对应

下列测试文件均位于 `test/server/`，实现位于 `python/nndeploy/server/`，Graph 生命周期位于 `python/nndeploy/dag/graph_runner.py`。已有验证日志和环境说明见 [实施记录](server_implementation.md)。

| 需求 | 实现和可检查证据 | 当前结论 |
| --- | --- | --- |
| R01 编辑器启动 Server | `Server.start_workflow`、编辑器 Server 按钮；`test_editor_save_start_reload_stop_uses_existing_workflow`、真实 Chrome Start/Deploy/Stop | 本地流程已验证 |
| R02 JSON 独立启动 | `__main__.py`、`app.cli`；`test_api_only_does_not_import_editor_or_modelscope`、`test_cli.py` 真实 HTTP 进程 | 已验证 |
| R03 直接 Graph | `Server.add(graph=...)`、`Executor`、GraphRunner；`test_declared_resources.py`、`test_arrays.py` | 保留原对象与连线；已验证 |
| R04 Python 接入 | `deploy`/`handler_factory`、普通/异步生成器；`test_server_api.py`、`test_streaming.py`、`test_engine.py` | 已验证 |
| R05 同步 Deploy | `api/deploy.py`、Worker 执行线程/进程；`test_q01_q02_slow_entry_does_not_block_other_entry` | HTTP 事件循环不执行阻塞模型 |
| R06 异步任务 | `Server.submit`、Task 查询/取消/事件；`test_auth.py`、`test_streaming.py`、`test_process_worker.py` | 已验证 |
| R07 SSE / WebSocket | `streaming.py`、`api/stream.py`；`test_streaming.py`、`test_duplex.py`、`test_websocket_audit.py` | 有真实 HTTP/WebSocket 连接测试 |
| R08 绑定与校验 | `config.py`、`io.py`；`test_server_api.py`、`test_graph_media.py`、`test_arrays.py` | 默认值、类型/范围及已知节点发现有覆盖 |
| R09 常驻与预热 | `Executor.open`、`execution.warmup`；`test_worker_thread_owns_initialization_execution_and_close`、`test_warmup.py` | 初始化一次、预热后就绪；已验证 |
| R10 请求隔离 | GraphIO 绑定/恢复、输出复制、请求临时目录；`test_numpy_graph.py`、`test_native_media.py`、`test_server_api.py` | 默认值、复用缓冲和文件结果有覆盖 |
| R11 调度与背压 | 每入口 TaskQueue、总量及设备准入；`test_server_api.py`、`test_device_execution.py` | 独立排队、限额、等待公平性已验证 |
| R12 批量计算 | `batching.py`、批量 HTTP；`test_batch.py`、`test_dynamic_batching.py` | 仅声明批处理能力的算法启用；已验证 |
| R13 期限与取消 | 配置中的队列/执行/HTTP 等待期限；`test_wait_timeout.py`、`test_process_worker.py`、`test_engine.py` | 普通线程不承诺硬中断；进程硬期限已验证 |
| R14 启停重载恢复 | Server/SharedRuntime 控制与 Worker 退出确认；`test_process_worker.py`、`test_shared_reload.py`、`test_release_recovery.py` | 本地恢复已验证；未知远端执行不擅自重算 |
| R15 文件与媒体 | Storage、GraphIO、arrays/audio；`test_storage.py`、`test_graph_media.py`、`test_native_media.py`、`test_audio.py`、`test_tensor_formats.py` | 支持格式的内容与往返已验证；不支持的声明明确拒绝 |
| R16 三数据库 | `db.py`、migrations；`test_db.py`、`test_database_backup.py`、`test_database_transfer.py` | 本地真实三数据库已验证 |
| R17 多存储 | `storage/objects.py` 原生 SDK 分支；`test_storage.py` | 本地/S3 已验证；COS/OSS/OBS 仅离线 SDK 检查，真实验收未完成 |
| R18 Redis | RedisTaskQueue、RequestLimiter、Worker 心跳、RedisEvents；`test_redis_queue.py`、`test_request_rate.py`、`test_redis_events.py` | 真实 Redis 已验证；缓存/通知失效回源 DB，安全限流失败拒绝 |
| R19 算法扩展 | `registry.py`、`api/extensions.py`、`dependencies.py`；`demo/server/text/server/`、`test_extensions.py` | 算法协议复用公共执行与权限；已验证 |
| R20 安全与审计 | Authentication、身份/文件/任务配额、Audit、输入限制；`test_auth.py`、`test_subject_*.py`、`test_audit.py`、`test_editor_auth.py` | 公共操作有验证；反向代理沿用 Uvicorn 可信来源设置，见部署说明 |
| R21 可观测性 | health、request/log_policy、metrics；`test_metrics.py`、`test_request_logging.py`、`test_log_policy.py` | 公共指标已验证；真实 GPU/模型指标未冒充验收 |
| R22 兼容与易用性 | 旧 `/api`、英文 Server 面板、原工作流字段；编辑器测试及浏览器证据 | 生产构建通过；全项目 TypeScript 检查仍有已记录诊断 |
| R23 资源与重启 | resources、assets、已保存配置与 File 引用；`test_managed_workflow.py`、`test_model_resources.py`、`test_asset_cleanup.py` | 删除草稿后本地重启、Stop/Start、资源保护与回收已验证 |
| R24 重试一致性 | DB 条件提交、控制幂等、Redis 领取/确认；`test_db.py`、`test_control*.py`、`test_redis_queue.py` | 重复提交/投递与迟到结果有覆盖 |
| R25 多个独立 Server | 独立实例、工作区和端口；`test_q06_independent_cli_servers_do_not_share_tasks_or_shutdown` | 两个真实 CLI 进程已验证 |
| R26 一个 Server 多入口 | 独立 entries/Worker/TaskQueue；Q01/Q02/Q07 测试 | 慢入口/失败入口不阻塞其他入口；已验证 |
| R27 设备与隔离 | devices、CPU 配置、进程 Worker；`test_devices.py`、`test_device_execution.py`、`test_cpu_threads.py` | CPU 与声明预算已验证；真实 GPU 待环境 |
| R28 算法自身调度 | serial/concurrent/engine；`test_concurrent.py`、`test_engine.py`、`test_shared_server.py` | 共享引擎可接受多个在途请求；已验证 |

## 调度验收 Q01–Q12

| 门槛 | 证据与结论 |
| --- | --- |
| Q01 / Q02 | `test_q01_q02_slow_entry_does_not_block_other_entry`：同步屏障保持 A 运行及队列满额，B 实际返回结果 |
| Q03 | `test_native_graph_reused_parameters_reset_and_records_survive_restart` 与 Graph Worker 串行执行/隔离测试；不共享请求默认值或输出缓冲 |
| Q04 | `test_q04_two_workers_can_execute_same_entry_in_parallel`：现为两个独立原生 Graph/Node，以屏障验证重叠执行并检查各自输出 |
| Q05 | `test_q05_one_engine_accepts_second_request_while_first_is_running`、`test_q05_second_stream_finishes_while_first_stream_is_still_open` |
| Q06 | `test_cli.py` 启动两个真实 Server 进程，跨实例查不到对方任务，停止 A 后 B 继续 Deploy |
| Q07 | `test_q07_reload_failure_keeps_existing_graph_and_other_entry`、`test_q07_process_crash_does_not_stop_other_entry_and_restarts` |
| Q08 | `test_device_execution.py` 的独立设备、共享预算、保留名额和原子申请；`test_subject_tasks.py` 的总量约束 |
| Q09 | `test_shared_workers_route_by_entry_and_configuration`、`test_shared_capacity_and_idempotency_survive_concurrent_submitters` |
| Q10 | `test_expired_lease_never_releases_running_capacity_or_reexecutes`、`test_real_owner_death_preserves_budget_until_all_compute_exits` |
| Q11 / Q12 | CPU/声明预算的本地部分有证据；不同 GPU、共享 GPU、非受管 GPU 占用及真实模型加载失败仍待设备环境，未标记完成 |

## 交付物与本轮收尾

- 可运行入口：`python -m nndeploy.server`、`Server.run()`；Graph、工作流、算法示例及启动说明在 `demo/server/README.md`、`python/nndeploy/server/README.md`。
- 数据库迁移在 `python/nndeploy/server/migrations/`，可选依赖分组在 `python/setup.py`；不新增 Serving 包或发布对象。
- API 文档由 FastAPI `/docs`、`/openapi.json` 提供；文件、任务、控制、事件、身份与错误说明在 Server README。
- 编辑器资源上传/Run/Start/Deploy/Stop 的最终浏览器证据：`logs/server-development/editor-reconnect-cu6791eo/evidence.json`。
- 本轮 Redis 缓存/通知与 SSE/WebSocket 关联集成：`redis-events-integrations.log`，三数据库、真实 Redis/MinIO 环境，149 项通过、0 失败、0 跳过。
- Q04 改为真实 Graph 后，公共 API 和真实双进程 CLI 回归：`server-core-acceptance.log`，24 项通过。
- 资源回收：`asset-cleanup-final-integrations.log`，146 项通过；最终原生释放失败保护另有 `asset-cleanup-native-release.log`。

这些是不同时间的相关回归，不能累加为独立测试总数，也不是云/GPU 完成证明。跨主机恢复、特定第三方算法和其他操作系统的限制继续保留在 README 和实施记录中；新增适配或额外组合测试不自动成为本轮新增交付要求。
