# Visual Workflow Quick Start

> [更多workflow模板](https://github.com/nndeploy/nndeploy-workflow)

## 环境要求

- Python 3.10+
- 支持的操作系统：Linux、Windows、macOS(待进一步测试)


## 启动可视化界面

nndeploy 提供了直观的 Web 界面用于模型部署：

```bash
# pip
pip install nndeploy

# 启动 Workflow 的 Web 服务
cd /path/nndeploy
python app.py --port 8000

# 或 使用简化命令 启动 Workflow 的 Web 服务
nndeploy-app --port 8000
```

> 注：Windows下命令行启动：nndeploy-app.exe --port 8000

在浏览器中访问 `http://localhost:8000` 开始使用。

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../../image/workflow.png">
    <img alt="nndeploy" src="../../image/workflow.png" width=100%>
  </picture>
</p>

## 启动参数说明

`app.py`启动脚本支持以下参数用于自定义Web服务行为：

| 参数名                    | 默认值                          | 说明                                                         |
| ---------------------- | ---------------------------- | --------------------------------------------------------         |
| `--host`               | `0.0.0.0`                    | 指定监听地址                                                       |
| `--port`               | `8888`                       | 指定监听端口                                                       |
| `--resources`          | `./resources`                | 指定资源文件目录路径                                                |
| `--log`                | `./logs/nndeploy_server.log` | 指定日志输出文件路径                                                |
| `--front-end-version`  | `!`                          | 指定前端版本，格式为 `owner/repo@tag`，如 `nndeploy/nndeploy-ui@v1.0.0` |
| `--debug / --no-debug` | `False`                      | 是否启用调试模式，启用后将禁用前端静态文件挂载                           |

## 常见问题

Q1: 浏览器打开 http://localhost:8000 显示404？

A1: 请确认你是否已经构建或下载前端资源，并且未使用--debug启动。

Q2: 启动时download前端资源文件一直失败怎么办？

A2: 从`https://github.com/nndeploy/nndeploy_frontend/releases/`下载对应的dist.zip，将zip解压到`frontend/owner_repo/tag/`目录下（通常下载失败后会自动建立该目录），重新启动服务。

Q3: 如何切换使用不同版本的前端?

A3: 使用 --front-end-version 参数指定版本，例如：
```
python app.py --front-end-version nndeploy/nndeploy-ui@v1.1.0
```

Q4: 前端资源下载完成了，还是无法打开前端界面？

A4: 检查服务端IP以及端口是否正确，如果`localhost`以及`127.0.0.1`都无法访问，替换成局域网IP（如`192.168.x.x`）重试。
