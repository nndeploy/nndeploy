"""nndeploy 可视化编辑器与 Deploy API 启动入口。

功能与默认值：
- 默认启动可视化工作流编辑器、HTTP 服务和任务 Worker。
- 默认监听 0.0.0.0:8000；--api-only 模式默认监听 127.0.0.1:8000。
- --resources 默认 ./resources，编辑器日志 --log 默认 ./logs/nndeploy_server.log。
- app.py、server.py 和 python -m nndeploy.server 使用同一套参数。
- 启动时自动读取当前工作目录的 server.env；文件不存在也可启动，已有环境变量优先。
- 运行前需安装 nndeploy 及对应服务依赖；以下命令中的路径均为示例。

编辑器常用命令：
    python app.py
    python app.py --help
    python app.py --host 127.0.0.1 --port 9000
    python app.py --resources ./my_resources --log ./my_logs/server.log
    python app.py --plugin ./plugins/custom.py ./plugins/custom.so
    python app.py --front-end-version @latest
    python app.py --front-end-dir ./app/workflow/dist
    python app.py --json_file ./workflow_a.json ./workflow_b.json
    python app.py --json_file ./workflow_a.json,./workflow_b.json
    python app.py --debug
    python app.py --editor-history-retention-seconds 86400

仅启动 Deploy API：
    python app.py --api-only --workflow ./graph.json
    python app.py --api-only --workflow ./graph.json --name my_graph --port 9000
    python app.py --api-only --workflow ./graph.json --config ./graph.server.json

认证、共享运行和 MCP：
    python app.py --admin-token-env NNDEPLOY_ADMIN_TOKEN
    python app.py --auth-config ./auth.json --audit-config ./audit.json
    python app.py --database-url-env NNDEPLOY_DATABASE_URL --redis-config ./redis.json
    python app.py --storage-config ./storage.json --device-limits ./devices.json
    python app.py --cors-origin http://localhost:3000 --max-sync-connections 16
    python app.py --mcp --mcp-path /mcp
    python app.py --mcp --mcp-allowed-host localhost:8000 --mcp-allowed-origin http://localhost:3000

数据库维护（操作完成后退出，不启动服务）：
    python app.py --database-url-env NNDEPLOY_DATABASE_URL --database-backup ./backup.db
    python app.py --database-url-env NNDEPLOY_DATABASE_URL --database-restore ./backup.db
    python app.py --database-url-env NNDEPLOY_DATABASE_URL --database-export ./records.jsonl
    python app.py --database-url-env NNDEPLOY_DATABASE_URL --database-import ./records.jsonl

使用说明：
- --json_file 将工作流复制到资源目录供编辑器使用；API-only 模式使用 --workflow。
- --workflow、--config、--name 必须配合 --api-only；--api-only 必须指定 --workflow。
- --plugin 可指定多个插件；编辑器会复制到 resources/plugin，API-only 模式直接加载。
- --front-end-dir 使用本地前端构建；--front-end-version 用于选择下载的前端版本。
- 环境变量选项接收变量名称，运行前需在当前环境中设置对应的令牌或数据库 URL。
- --mcp 需要可选的 nndeploy[server-mcp] 依赖；允许主机、来源及 CORS 选项可重复传入。
- 编辑器历史默认保留；--editor-history-retention-seconds 至少为 60 秒。
- 数据库恢复/导入要求目标数据库为空；备份格式取决于数据库后端，示例 .db 用于 SQLite。
- 数据库维护选项互斥，且不能与 API-only 或工作流启动参数组合。
- 完整参数见 --help，配置文件结构见 python/nndeploy/server/README.md。
"""

from nndeploy.server.app import main


if __name__ == "__main__":
    raise SystemExit(main())
