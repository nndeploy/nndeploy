# DAG 示例

## Python：State、执行顺序与嵌套控制流

[state_control.py](state_control.py) 不需要模型、GPU 或输入文件。先构建并安装包含 State 功能的 nndeploy Python 包，然后在仓库根目录运行：

```bash
python demo/dag/state_control.py
python demo/dag/state_control.py --parallel task
```

示例通过 `@dataclass` 定义 `DemoState`，节点在原有的 `run()` 中使用 `self.state.value`、`self.state.history` 读写状态。根图调用一次 `run_with_state(state)`，嵌套的 Graph、Condition、Loop 共享该请求的 State。

执行结构如下，所有节点均无数据 Edge：

```text
Graph: prepare -> repeat (Loop，3 轮)
                   每轮：iter_before -> choose (Condition) -> record -> iter_after
                                         ├─ step 为偶数：Graph(add_two -> double)
                                         └─ step 为奇数：add_ten
```

- 根图、循环体和偶数分支分别用 `set_execution_order([...])` 指定顺序；注册节点时故意使用相反顺序，便于观察差异。排序需要在 `init()` 前完成。
- Condition 的 `choose()` 返回分支的注册索引，每轮只执行一个分支。多步分支用 Graph 组织，候选分支之间不设置执行顺序。
- Loop 的 `loops()` 从 State 读取循环上限，`iter_before()`、`iter_after()` 记录过程并更新轮次。初始化钩子发生在请求 State 绑定前，不能在其中读取请求状态。
- 任务并行模式仍遵守显式依赖；本例各步按顺序修改同一字段。流水线带 State 执行暂不支持。

初始值为 1，三轮计算为 `(1 + 2) * 2 = 6`、`6 + 10 = 16`、`(16 + 2) * 2 = 36`。程序会校验结果，并输出：

```text
value=36, step=3, history=[6, 16, 36]
trace: prepare -> before(0) -> add(2) -> double -> record(6) -> after(0) -> before(1) -> add(10) -> record(16) -> after(1) -> before(2) -> add(2) -> double -> record(36) -> after(2)
```

执行完成后，结果仍保留在调用方的 `state` 中，节点的请求绑定已解除。处理另一个请求时，创建新的 `DemoState` 并再次调用 `run_with_state()`；`history` 和 `trace` 使用 `default_factory`，不会在请求间共享默认列表。

## Python：State 与 Edge 混用

[state_edge.py](state_edge.py) 展示现有数据流接口与 State、控制依赖共用一张图，无需模型或 GPU。在包含本次功能的 Python 环境中，从仓库根目录运行：

```bash
python demo/dag/state_edge.py
python demo/dag/state_edge.py --parallel task
```

```text
数据流：input Edge -> double -> doubled Edge -> apply_offset -> result Edge
控制流：prepare -------------------------------> apply_offset -> summarize
```

- `LegacyDouble.run()` 仅使用原有的 `get_input_data()` / `set_output_data()`，将列表逐项乘 2，无需添加 State 访问代码。
- `PrepareState` 没有数据边，将请求的 `state.bias` 写入 `state.offset`。
- `ApplyOffset` 同时读取 Edge 中的数据和 State 中的偏移量，将结果写回 Edge，并更新 State 中的数量与总和。
- `SummarizeState` 没有数据边，通过 State 读取统计结果并生成摘要。

`set_execution_order(["prepare", "apply_offset", "summarize"])` 只补充这三个节点的控制依赖。`double` 没有出现在列表中，但仍通过数据边参与执行。`apply_offset` 必须等待 `double` 与 `prepare` 都成功完成；任务并行模式允许这两个前驱并发执行，`summarize` 则必须等待状态统计完成。State 的字段读写本身不声明调度依赖，需要显式指定顺序。

示例初始化一次图，连续传入两组 Edge 输入和独立的 RequestState，校验 Edge 输出、State 统计、执行顺序及请求隔离。两种模式应得到相同结果：

```text
request 1: output=[12, 14, 16], count=3, total=42
trace: prepare -> apply_offset -> summarize
request 2: output=[7, 9], count=2, total=16
trace: prepare -> apply_offset -> summarize
```

输入通过 `source.set(values)` 提供，结果通过 `result.get_graph_output()` 获取；Edge 的更新与节点调度由现有执行器处理。图中的 Python Edge 对象保存在 `graph._edge_refs` 中，以确保构图使用的原生指针在执行期间有效。该示例不使用流水线模式。
