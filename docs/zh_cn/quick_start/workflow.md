# Visual Workflow Quick Start

nndeploy 提供了完整的 Python API，支持快速部署和推理各种深度学习模型。

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

在浏览器中访问 `http://localhost:8000` 开始使用。

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../../image/workflow.png">
    <img alt="nndeploy" src="../../image/workflow.png" width=100%>
  </picture>
</p>
