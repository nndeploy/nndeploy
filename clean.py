"""nndeploy 本地服务目录清理入口。

功能与默认目录：
- logs：./logs；template：./resources/template；db：./resources/db。
- frontend：./frontend；plugin：./resources/plugin。
- server：./resources/server，包含 Deploy 状态和文件，仅显式 --server 时清理。
- 不传目录选择参数时，清理前五类目录，不清理 server。

使用示例：
    python clean.py --help
    python clean.py --dry-run                     # 预览默认清理范围，不删除文件
    python clean.py                               # 清理默认的五类目录
    python clean.py --logs                        # 仅清理日志
    python clean.py --template                    # 仅清理工作流模板缓存
    python clean.py --db                          # 仅清理编辑器数据库目录
    python clean.py --frontend                    # 仅清理前端缓存
    python clean.py --plugin                      # 仅清理插件
    python clean.py --logs --db                   # 同时清理所选目录
    python clean.py --server                      # 显式清理 Deploy 状态和文件
    python clean.py --keep logs db                # 保留日志和数据库，清理其余默认目录
    python clean.py --resources ./my_resources --db --dry-run
    python clean.py --resources ./my_resources --server
    python clean.py --logs --logs-dir ./my_logs
    python clean.py --frontend --frontend-dir ./my_frontend

使用说明：
- 清理前应停止使用这些目录的服务；--dry-run 可与目录选择参数组合。
- --resources 指定 template、db、plugin、server 的父目录。
- 编辑器模板缓存目前固定为 ./resources/template；使用自定义资源目录时需注意区别。
- --logs-dir 和 --frontend-dir 分别指定日志目录及前端目录。
- --keep 仅从默认五类目录中排除指定项，不能与 --logs、--db 等选择参数混用。
- 目录中的内容会被删除；若所选路径为符号链接，仅删除链接，不删除其目标。
- 成功或目录不存在时退出码为 0，清理失败为 1，参数错误为 2。
"""

from nndeploy.server.clean import main


if __name__ == "__main__":
    raise SystemExit(main())
