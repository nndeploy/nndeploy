# Visual Workflow Quick Start

> [更多 workflow 模板](https://github.com/nndeploy/nndeploy-workflow)

## 环境要求

- Python 3.10、Python 3.11、Python 3.12、Python 3.13
- 支持的操作系统：Linux、Windows、macOS

## 启动可视化界面

nndeploy 提供了直观的 Web 界面用于模型部署：

```bash
# pip
pip install --upgrade nndeploy

# 方式一：命令行
nndeploy-app --port 8000

# 方式二：代码启动
cd path/to/nndeploy
python app.py --port 8000

# 当更新了nndeploy时，建议清理过期前后端资源
nndeploy-clean
```

> 注：Windows 下命令行启动：nndeploy-app.exe --port 8000

在浏览器中访问 `http://localhost:8000` 开始使用。

<!-- <p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../../image/workflow.png">
    <img alt="nndeploy" src="../../image/workflow.png" width=100%>
  </picture>
</p> -->

![../../image/quick_start.gif](../../image/quick_start.gif)


### 目录结构说明

启动工作流后，会在当前工作目录下自动生成以下三个核心目录：

- **frontend/**：前端资源目录
  - 存放 Web 界面的静态资源文件
  - 包含 HTML、CSS、JavaScript 等前端组件
  
- **logs/**：日志输出目录
  - 记录工作流执行过程中的详细日志信息
  - 便于调试和问题排查
  
- **resources/**：资源管理目录，包含以下子目录：
  - **workflow/**：用户工作流存储目录，保存用户创建和编辑的工作流文件
  - **template/**：模板工作流目录，提供预置的工作流模板供用户参考使用
  - **models/**：模型文件存储目录，存放 AI 模型权重文件和配置文件
  - **images/**：图片资源目录，存放用于测试和处理的图像文件
  - **videos/**：视频资源目录，存放用于测试和处理的视频文件
  - **audios/**：音频资源目录，存放用于测试和处理的音频文件
  - **db/**：数据库文件目录，存放工作流相关的数据库文件
  - **plugin/**：自定义插件目录，存放用户开发的自定义节点和扩展功能
  
## 启动参数说明

`app.py`启动脚本支持以下参数用于自定义 Web 服务行为：

| 参数名                | 默认值                       | 说明                                                                               |
| --------------------- | ---------------------------- | ---------------------------------------------------------------------------------- | 
| `--host`              | `0.0.0.0`                    | 指定监听地址                                                                       |
| `--port`              | `8888`                       | 指定监听端口                                                                       |
| `--resources`         | `./resources`                | 指定资源文件目录路径                                                               |
| `--log`               | `./logs/nndeploy_server.log` | 指定日志输出文件路径                                                               |
| `--front-end-version` | `!`                          | 指定前端版本，格式为 `owner/repo@tag`，如 `nndeploy/nndeploy-ui@v1.0.0`            |
| `--plugin`            | `[]`                         | 支持传入多个 python 文件路径或者动态库路径，用于加载用户写好的自定义插件，默认为空 |

## 清理过期资源

随着nndeploy版本的迭代更新，frontend、logs、resources这三个核心目录中的资源文件也会相应地进行同步更新。为了保持系统的整洁性和避免磁盘空间的浪费，您可以使用以下命令行工具来快速清理过期和冗余的资源文件。

```bash

# 方式一：命令行清理所有资源
nndeploy-clean

# 方式二，执行python脚本
cd path/nndeploy
python clean.py
```

`clean.py`启动脚本支持以下参数用于自定义行为：

| 参数名           | 说明                                                |
| ---------------- | --------------------------------------------------- |
| `--logs`         | only clean logs directory                           |
| `--template`     | only clean template directory                       |
| `--resources`    | only clean database directory                       |
| `--db`           | only clean frontend directory                       |
| `--plugin`       | only clean plugin directory                         |
| `--keep logs db` | keep logs and db directory, clean other directories |

> 未指定参数时，清理上述所有资源

## 常见问题

Q1: 浏览器打开 http://localhost:8000 显示 404？

A1: 请确认你是否已经下载前端资源。

Q2: 启动时 download 前端资源文件一直失败怎么办？

A2: 从`https://github.com/nndeploy/nndeploy_frontend/releases/`下载对应的 dist.zip，将 zip 解压到`frontend/owner_repo/tag/`目录下（通常下载失败后会自动建立该目录），重新启动服务。

Q3: 如何切换使用不同版本的前端?

A3: 使用 --front-end-version 参数指定版本，例如：

```
python app.py --front-end-version nndeploy/nndeploy-ui@v1.1.0
```

Q4: 前端资源下载完成了，还是无法打开前端界面？

A4: 检查服务端 IP 以及端口是否正确，如果`localhost`以及`127.0.0.1`都无法访问，替换成局域网 IP（如`192.168.x.x`）重试。

## 案例展示

### YOLO 可视化调参与一键部署

可视化界面实时调整检测参数，无需修改代码即可观察效果变化，支持一键切换到 TensorRT 等推理引擎实现高性能部署。

<!-- <p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="yolo_edit_param.gif">
    <img alt="nndeploy" src="../../image/yolo_edit_deploy.gif" width=100%>
  </picture>
</p> -->

![../../image/yolo_edit_deploy.gif](../../image/yolo_edit_deploy.gif)

### 多模型工作流演示

可视化搭建检测+分割+分类工作流，支持多推理框架切换和并行模式，实现一次搭建、多端部署。

<!-- <p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="seg_detect_class.gif">
    <img alt="nndeploy" src="../../image/seg_detect_class.gif" width=100%>
  </picture>
</p> -->

![../../image/seg_detect_class.gif](../../image/seg_detect_class.gif)

### 零代码搭建换脸+分割工作流

通过拖拽操作组合人脸检测、换脸算法、人像分割等 AI 功能，无需编写代码，参数调整 1-2 秒看到效果。让**产品经理、设计师、非 AI 开发者**快速将创意变成原型。

<!-- <p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="face_swap_seg.gif">
    <img alt="nndeploy" src="../../image/face_swap_seg.gif" width=100%>
  </picture>
</p> -->

![../../image/face_swap_seg.gif](../../image/face_swap_seg.gif)
