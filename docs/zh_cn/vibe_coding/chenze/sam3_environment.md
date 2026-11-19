# SAM3 与 nndeploy 运行环境

日期：2026-11-19。目标：在既有 nndeploy Python 虚拟环境中接通原生 SAM3 推理，遵循仓库已有的依赖安装和插件使用方式。

## 1. 维护边界

- nndeploy 源码仓库：`/home/chenze/nndeploy`，分支 `chenze`，基线提交 `b7286d190c61362553ddca1d3628a3913e1f124d`。
- 本轮使用既有 `/home/chenze/nndeploy-runtime/develop/.venv`；该运行目录位于源码仓库外，属于用户授权可直接维护的非公共部分。
- 仓库内仅修改 SAM3 包依赖、专用测试及本目录文档。维护规则见 [开发与维护边界](sam3_development_rules.md)。
- nndeploy 原已通过 editable 安装指向当前源码。本轮只补模型依赖，不重新执行 `pip install -e .`、`setup.py` 或 C++ 编译。

## 2. 依赖安装

以下命令在远程 Linux 执行，路径按实际部署设置：

```bash
NNDEPLOY_ROOT=/home/chenze/nndeploy
PYTHON=/home/chenze/nndeploy-runtime/develop/.venv/bin/python

"$PYTHON" -m pip install \
  torch==2.10.0+cu128 torchvision==0.25.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
"$PYTHON" -m pip install \
  -r "$NNDEPLOY_ROOT/python/nndeploy/segment/sam3/requirements.txt" \
  --index-url https://pypi.org/simple
# Existing ONNX requires ml-dtypes>=0.5.4; resolve it together with SAM3's NumPy range.
"$PYTHON" -m pip install 'numpy>=1.26,<2' 'ml-dtypes>=0.5.4' \
  --index-url https://pypi.org/simple
"$PYTHON" -m pip check
```

本轮沿用 SAM3 requirements 中已有版本范围，由 pip 解析 NumPy/OpenCV 等兼容组合；新增 `pycocotools>=2.0` 用于 COCO RLE 输入输出。普通版 OpenCV 与 headless 不同时安装。

首次依赖检查发现原有 ml-dtypes 0.6.0 要求 NumPy 2，与 SAM3 的 NumPy 1.x 范围冲突。既有 ONNX 1.23.2 只要求 ml-dtypes>=0.5.4，因此将两项要求一起交给 pip 正常解析，选择了 ml-dtypes 0.5.4；未修改 ONNX 或仓库公共依赖声明。

初次安装使用的历史代理端口没有监听，连接在下载前失败。随后确认官方包索引可直连，仅在安装子进程中移除了失效代理变量；未修改 SSH 配置或系统代理。

## 3. 动态库与加载方式

运行时使用仓库外的 SAM3 环境包装脚本：

```bash
source /home/chenze/nndeploy-runtime/20261119-sam3-environment/sam3-env.sh
```

该脚本先加载原有 `20261116-develop/env.sh`，再把当前虚拟环境中 `nvidia/*/lib` 与 `torch/lib` 放到进程库搜索路径前面。源码路径、原生库和虚拟环境继续复用既有配置。

仅加载原脚本时，nndeploy 先导入会选中 CUDA 12.4 运行库，随后 PyTorch 报 `cudaGetDriverEntryPointByVersion` 未定义。调整进程级库优先级后，联合导入与 CUDA/cuDNN 检查通过。没有改动系统 CUDA、驱动、公共启动脚本或仓库二进制。

PyTorch 使用其 cu128 运行依赖；已有 nndeploy 原生组件仍使用原编译结果及相关 SDK。两者的兼容性以联合加载和实际推理验证为准，不因版本号不同自动升级系统 CUDA、cuDNN、TensorRT 或驱动。

代码入口为 `nndeploy.segment.sam3`。本轮不安装原始 `sam3` 包，不复制权重到仓库，不启用新编译/引擎优化。

## 4. 环境检查

```bash
source /home/chenze/nndeploy-runtime/20261119-sam3-environment/sam3-env.sh
export CUDA_VISIBLE_DEVICES=3  # 本轮验证设备；运行前按实际资源选择
python -B /home/chenze/nndeploy/test/sam3/check_environment.py \
  --cuda --server \
  --output /home/chenze/nndeploy-runtime/20261119-sam3-environment/environment-check.json
```

脚本检查：

- nndeploy、SAM3、Torch/TorchVision、OpenCV、RLE 与服务依赖联合导入，记录版本及加载路径。
- NumPy 与 Torch、nndeploy 原生 Tensor 转换；PNG 与 COCO RLE 往返。
- 小规模 CUDA 矩阵乘法和 cuDNN 卷积，不加载模型权重、不作为性能基准。
- 使用现有 Server 和一个 NumPy 复制 Graph，通过 FastAPI TestClient 发起两次进程内 HTTP 请求，并核对历史任务结果。

Server 检查使用临时工作目录，不注册新的公共算法，不修改既有服务或其配置。这个测试 Graph 不是计划中的三个 SAM3 业务 Graph。

## 5. 原生推理回归

复用 2026-11-16 的迁移验证脚本和固定样例，用新环境执行已迁移的 SAM3 包。随机种子为 42，`compile=False`，图片为 truck，视频为 bedroom 前 24 帧。

检查图片概念分割、实例提示与精修，以及视频概念、实例修正、传播和会话操作。迁移包验证时禁止导入顶层 `sam3` 包；先与历史基线比较，再在本轮相同虚拟环境下运行原始源码进行对照。两组比较分别记录，避免把环境变化直接归因于源码迁移。

该回归验证模型迁移及环境一致性，不代表 SAM3 业务 Graph、工作流或专用 HTTP 接口已实现。

## 6. 本轮实际结果

使用现有 nndeploy 虚拟环境完成依赖对齐，未创建第二套生产环境。

| 项目 | 实际结果 |
| --- | --- |
| Python / nndeploy | 3.12.15 / 3.0.10（editable） |
| PyTorch / TorchVision / Triton | 2.10.0+cu128 / 0.25.0+cu128 / 3.6.0 |
| NumPy / OpenCV / ml-dtypes | 1.26.4 / 4.11.0.86 / 0.5.4 |
| timm / pycocotools | 1.0.31 / 2.0.11 |
| FastAPI / Uvicorn | 0.162.4 / 0.54.0 |
| pip check | 通过 |
| 联合导入、数据/编码转换、CUDA/cuDNN、Server 简单 Graph | 四组均通过 |
| 迁移包真实推理 | 图片及视频共 44 组检查完成，生成 209 个输出数组；会话重置、移除对象、关闭与释放通过 |
| 当前同环境原始源码对照 | 原始与迁移包的 209 个数组键、shape、dtype、值完全一致 |
| 历史环境基线对照 | 未逐值一致，209 个数组中 85 个不同；shape/dtype 与对象 ID 未发现差异 |
| 已有 Web 服务 | 检查时 `/api/dag/info` 返回 HTTP 200；本轮未重启服务 |

历史对照中，发生变化的布尔 mask 数组共 41,126,400 个像素，其中 1,655 个不同；布尔 mask 对照记录中的最低 IoU 为 0.9549084。不同输出的浮点数误差也已记录在 `historical-differences.json`，因此不把历史基线检查标记为通过。当前环境中原始源码与迁移包完全一致，说明本轮未观察到迁移导致的输出差异；尚未逐项定位历史环境数值变化所对应的依赖或 GPU 内核因素。

原始 SAM3 对照启动时依赖旧 `pkg_resources`。仅在仓库外 `original-validation-compat/` 复制原环境的 setuptools 78.1.0 及其 pkg_resources，加入原始源码对照进程的 PYTHONPATH；未降级目标环境 setuptools，未修改原始模型代码，迁移包运行不需要此兼容目录。

结果分别保留，避免覆盖失败记录：

- `environment-check.json`：联合环境检查通过。
- `migrated/results.json`、`original/results.json`：本轮两套源码的真实推理记录。
- `current-environment-comparison.json`：本轮同环境 209 数组逐值一致。
- `comparison.json`、`historical-differences.json`：历史环境差异记录。
- `validation-result.json`：首轮包含严格历史比较的验证流程返回失败，保留原状；不能仅凭这个文件判断后续同环境对照。
- `final-validation-summary.json`：汇总本轮最终验证状态及限制。

本轮没有实现或部署三个 SAM3 业务 Graph，也没有完成 SAM3 工作流、专用 API、TensorRT 引擎或性能验收。现有 Web 进程未重新加载新环境，其 HTTP 可达性也不代表它已经运行 SAM3。接下来可在已确认的三个 Graph 与内部模块边界内开展实现。

## 7. 记录位置

本轮安装前后版本、安装日志、环境检查、推理结果和比较报告集中保存在：

`/home/chenze/nndeploy-runtime/20261119-sam3-environment/`

其中模型权重继续引用已有运行资产。日志、数值结果和运行缓存不进入源码提交。


