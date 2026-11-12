# DAG 共享状态与控制依赖统一设计

状态：第一版已实现，包含 C++ 与 Python 回归测试。更新日期：2026-11-12。

本方案让同一张 nndeploy Graph 同时支持通过 Edge 传递数据、通过 State 共享请求变量，以及手动指定节点执行顺序。最小实现保留 `Node::run()`、`Edge` 和 `Executor` 接口，将数据依赖与控制依赖合并到现有 `NodeWrapper` 的前驱、后继关系中。

Python 使用带类型声明的 `State`，通过 `state.latents` 等属性读写。一次执行中的节点共享同一状态，不同请求分别创建状态。第一版支持 Graph、Condition 和 Loop 的串行和任务并行 State 执行；带控制依赖或请求 State 的流水线执行需要后续扩展。

可运行示例：[state_control.py](../../../../demo/dag/state_control.py)，展示 State 属性读写、显式节点顺序，以及 Loop → Condition → 分支 Graph 的嵌套。运行方式与预期输出见 [DAG 示例说明](../../../../demo/dag/README.md)。

混合数据流示例：[state_edge.py](../../../../demo/dag/state_edge.py)，展示不访问 State 的原有数据流节点、同时使用 Edge 与 State 的节点，以及零边状态节点如何通过数据依赖和显式顺序共同调度，支持串行和任务并行验证。

## 1. 已确定的设计

| 项目 | 决定 |
| --- | --- |
| 请求状态名称 | C++ 为 `dag::State`，Python 为 `nndeploy.dag.State` |
| Python 访问方式 | dataclass 定义字段，节点通过 `self.state` 或 `self.get_state()` 获取状态，直接操作属性 |
| 节点执行接口 | 保留 `Node::run()`，不要求改写现有节点 |
| 统一调度依据 | `NodeWrapper::predecessors_` 和 `successors_` |
| 数据边 | 保持现有数据传递、类型、所有权、队列与背压机制 |
| 手动顺序 | 编译为相邻节点之间的控制依赖 |
| 执行器 | 复用顺序执行器和任务并行执行器 |
| 状态更新 | 直接修改，失败不自动回滚 |
| 生命周期 | State 属于一次请求；模型等长期资源继续使用图级资源接口 |

第一版不新增 `ScheduleType`、`ExplicitExecutor` 或独立的 `ExecutionPlan` 类。现有调度结构已经可以承载无环的显式执行关系，后续扩展不需要改变用户侧的状态访问方式。

## 2. 统一模型

```text
Graph
  ├─ Nodes：已有节点及 run() 实现
  ├─ Edges：数据连接与传递
  ├─ Control dependencies：用户指定的先后关系
  └─ 当前绑定的 State：本次请求共享变量

数据边产生的依赖 + 用户指定的控制依赖
                    ↓
       NodeWrapper 前驱与后继关系
                    ↓
             现有 Executor
```

例如 A 通过 Edge 向 C 传入 embeddings，B 将 latents 写入 State：

```text
A ──数据连接──→ C
B ──控制依赖──→ C

C.predecessors_ = {A, B}
```

C 在 A、B 成功完成后运行，同时读取输入 Edge 和 State。控制关系不需要分配占位数据，也不创建 `dag::Edge`。

共享变量的读写不会自动生成依赖。构图者必须显式规定有关读写的先后关系；数据依赖与控制依赖冲突时，初始化失败。

## 3. 用户侧接口

以下示例使用第一版接口；需要重新编译框架和 Python 扩展。示例中的业务模型和节点实例由调用方提供。

### 3.1 定义状态

```python
from dataclasses import dataclass, field

import torch
from nndeploy.dag import State


@dataclass
class DiffusionState(State):
    prompt: str = ""
    num_steps: int = 20
    step_index: int = 0
    prompt_embeds: torch.Tensor | None = None
    latents: torch.Tensor | None = None
    logs: list[str] = field(default_factory=list)
```

类型声明提供 IDE 补全和静态检查依据；第一版不自动完成所有 Python 类型的运行时校验。可变默认值使用 `default_factory`，不同 State 实例不共享默认 list、dict。

### 3.2 节点直接读写属性

```python
import nndeploy.base
from nndeploy.dag import Node


class DenoiseNode(Node):
    def run(self):
        state: DiffusionState = self.state
        state.latents = self.model(
            state.latents,
            state.prompt_embeds,
        )
        state.step_index += 1
        state.logs.append("denoise finished")
        return nndeploy.base.Status.ok()
```

`self.model` 由节点初始化或已有图级资源机制提供。`self.state` 是只读访问入口，不能通过赋值替换整个绑定，但状态字段可以修改。未绑定 State 时访问该属性，抛出明确的运行时异常。

### 3.3 指定顺序并执行

```python
# encode、prepare、denoise、decode 为已创建的节点。
graph.add_node(encode)
graph.add_node(prepare)
graph.add_node(denoise)
graph.add_node(decode)

graph.set_execution_order([
    "encode", "prepare", "denoise", "decode",
])

state = DiffusionState(prompt="一只猫", num_steps=30)

# 调用方按现有 Status 约定检查各步骤结果。
graph.init()
try:
    status = graph.run_with_state(state)
finally:
    graph.deinit()

result = state.latents
```

也可以只添加局部控制约束，并与已有数据边混用：

```python
graph.add_control_dependency("prepare", "denoise")
graph.remove_control_dependency("prepare", "denoise")
```

`run_with_state()` 返回执行 Status，调用方持有的 State 保存结果。Graph 不需要为每次请求重新初始化；初始化成功后，可依次用不同 State 执行多次，最后统一释放。

## 4. State 的存储与生命周期

### 4.1 数据归属

| 数据 | 位置 | 生命周期 |
| --- | --- | --- |
| prompt、latents、timesteps、step_index | State | 请求级；调用方可以在执行后继续持有 |
| 模型、tokenizer、固定配置 | 现有图级资源 | 图级，由资源自身决定共享安全性 |
| 节点自身参数 | 现有 Param | 节点级 |
| 端口数据 | 现有 Edge | 延续现有 Edge 生命周期 |

State 不放入进程级单例，也不使用唯一的 `current_request` 全局变量。Graph 仅在执行期间持有当前 State 的绑定。

### 4.2 C++ 接口

`framework/include/nndeploy/dag/state.h` 及对应实现提供小型存储接口：

```cpp
class State {
 public:
  base::Status set(const std::string& key, const base::Any& value);
  base::Status get(const std::string& key, base::Any& value) const;
  bool contains(const std::string& key) const;
  bool erase(const std::string& key);

 private:
  mutable std::mutex mutex_;
  std::map<std::string, base::Any> values_;
};
```

`get()` 在锁内取值，返回副本或共享对象句柄，不返回容器内部的可变引用。缺失字段返回明确的错误 Status。模板便利接口可基于它添加，但必须明确类型错误和缺失字段的行为。

大对象通过共享句柄保存，避免复制 Tensor 数据。存储锁只保证字段访问安全，不保证对象内部修改或一组读写操作的原子性。

### 4.3 Python 属性适配

Python `State` 是底层 C++ State 的适配层，支持 dataclass 子类：

```text
state.latents / state.step_index
              ↓
Python State 属性适配与类型转换
              ↓
同一个 C++ dag::State 存储
              ↑
C++ 节点通过 get/set 访问
```

实现要求：

1. 在 dataclass 生成的初始化方法写入字段前，建立底层存储。
2. 对已声明字段统一处理读写，包括继承字段、默认值和 `default_factory`。
3. 不能仅依靠 `__getattr__`：类上的默认值可能遮蔽底层字段。使用字段描述器或能够优先读取已声明字段的属性访问实现。
4. 字段值只保存在底层 State，Python 对象只保存句柄、schema 等内部元数据。
5. `get_state()` 返回相同 schema 的共享视图；Python 发起的执行期间保留原始包装对象，保证节点访问与调用方访问一致。
6. 第一版拒绝写入未声明的业务字段，避免拼写错误静默创建变量；字段名不能覆盖 State 的内部保留成员。
7. 不通过 `asdict()` 将整个 dataclass 复制进 C++ map；避免深复制 Tensor 和形成两份状态。

仅添加 `py::dynamic_attr()` 不足以满足上述要求，因为属性可能只存入 Python 的 `__dict__`。

绑定层支持 bool、int、float、str、Param 及以共享句柄表示的 nndeploy Tensor、Buffer。Python int 在 int 范围内保存为 C++ int，更大的值保存为 int64_t，超出 int64_t 范围报错。纯 Python 对象保存引用，普通 C++ 节点不能据此直接读取为任意 C++ 类型，例如 `torch.Tensor` 不自动转为 `device::Tensor`。

`py::object` 的引用计数操作、替换与释放必须正确持有 GIL。不要在 State 存储锁内调用 Python 用户代码或触发 Python 对象的最终析构；替换、删除时将旧值移出临界区后再释放，避免 GIL 与存储锁顺序造成死锁。

### 4.4 绑定规则

Node 新增 `getState()`。默认沿已有 `graph_`、`composite_node_` 查找，Graph 返回当前绑定；未自行绑定的嵌套图继承父图 State。

Graph 新增 `runWithState(std::shared_ptr<State>)`。它校验执行方式、取得执行独占权、用 RAII 绑定 State，调用现有执行路径，在所有本次任务结束后恢复绑定。错误、异常和中断路径也必须恢复。

第一版禁止同一图实例重入执行。`run()` 与 `runWithState()` 共用执行互斥机制。带状态执行在发布 State 前预留参与执行的子图；执行器通过内部调用作用域标识正常调度的子图，独立调用无法进入该次执行。嵌套图继承外层 State。Graph、Condition、Loop 的嵌套调度已覆盖测试；Condition 会预留全部候选分支中的子图，Loop 会预留整个循环体中的子图。choose、loops 和迭代钩子内递归执行当前图同样被拒绝；自定义 CompositeNode 若直接调用内嵌 Graph 的 run，需要改用内部调度辅助函数或将该 Graph 提升为外层图的节点。不同图实例并发时，调用方必须使用独立 State 和可安全共享的资源。

State 在 `runWithState()` 期间有效，不要求在 `init()`、`deinit()` 时存在。节点初始化从 Param 或图级资源读取模型配置。执行结束解除绑定不会清空调用方持有的 State，复用同一 State 再次执行会保留已有字段值。

## 5. 控制依赖与手动顺序

### 5.1 Graph 新增接口

```cpp
base::Status addControlDependency(
    const std::string& predecessor,
    const std::string& successor);

base::Status removeControlDependency(
    const std::string& predecessor,
    const std::string& successor);

base::Status setExecutionOrder(
    const std::vector<std::string>& node_names);

base::Status runWithState(std::shared_ptr<State> state);
```

保存声明所需的最小成员：

```cpp
std::set<std::pair<std::string, std::string>> control_dependencies_;
std::vector<std::string> execution_order_;
std::shared_ptr<State> active_state_;
```

另增加执行互斥及必要的私有辅助方法，不暴露可任意设置当前 State 的公共 setter。

### 5.2 setExecutionOrder 的语义

`setExecutionOrder({"A", "B", "C"})` 产生 A → B、B → C 的控制约束。

- 替换上一次通过该接口设置的顺序；空列表清除此部分约束。
- 与现有数据依赖和单独声明的控制依赖合并。
- 不作为执行节点白名单，不删除列表外原本参与执行的节点。
- 单节点列表显式启用该节点，即使它没有数据边。
- 节点必须存在于当前图的直接子节点中，列表不允许重复节点。
- 不直接引用嵌套子图内部节点；将子图作为外层节点调度。
- 重复控制声明幂等；自依赖及合并后形成的环报错。
- 初始化后禁止修改结构，需先 `deinit()`。

不支持通过列表重复节点来表达循环。循环使用现有 Loop 结构；循环体可声明控制依赖和执行顺序，各轮共享同一请求 State。

### 5.3 构建步骤

在 Graph 中提取内部 `rebuildDependencies()`：

1. 清空所有 `NodeWrapper` 的 `predecessors_`、`successors_`。
2. 根据当前 EdgeWrapper 的生产者、消费者重建数据依赖。
3. 合并 `control_dependencies_`。
4. 合并 `execution_order_` 中相邻节点的依赖。
5. 去重，校验端点和环。
6. 生成运行节点列表。

运行节点的保留条件为：关联数据边，或关联显式控制依赖，或出现在执行顺序中。这样既支持零边节点，又保留旧图对未连接、未显式启用节点的处理。

结构变更统一使派生调度信息失效，在执行器初始化前重建。删除节点要清理对应控制声明；断开数据边不能移除仍由控制声明产生的依赖。用于构图和序列化的节点名称应在注册后保持稳定；如提供改名入口，必须同步更新声明。

图优化发生结构变化后，也必须重新生成依赖和运行列表。第一版对含新控制声明或顺序列表的图跳过尚未适配的结构优化，并给出日志。对于仅通过 State 产生副作用的节点，也不能套用未经审计的数据流删除或融合优化；第一版 State 用法要求显式声明执行顺序或控制关系，避免绕过该保护。

## 6. 执行器兼容策略

| 类 | 最小实现策略 |
| --- | --- |
| Node | 保留 `run()` 和输入输出接口，新增 `getState()` |
| NodeWrapper | 结构不变，前驱后继包含两种来源的依赖 |
| Edge、EdgeWrapper | 接口和存储行为不变 |
| Executor | 保留 `init(edges, nodes)`、`run()` 等接口 |
| SequentialExecutor | 继续拓扑排序并逐个执行 |
| ParallelTaskExecutor | 继续等待全部前驱成功完成后提交后继 |
| ParallelPipelineExecutor | 旧数据流水线行为保留；新增控制依赖、显式顺序和请求 State 组合暂不支持 |
| Condition | choose() 与所选分支共享 State；保留按注册索引选择单个分支的语义 |
| Loop | loops()、iterBefore()、iterAfter() 和循环体共享 State；循环体支持控制依赖与显式顺序 |
| CompositeNode | 子节点沿父级查找 State；自定义执行嵌套 Graph 时使用内部调度辅助函数 |

当前 `Node::updateInput()` 在无输入时已经返回 Complete，零边节点无需新的输入检查分支。顺序和任务并行执行器继续调用原来的 `run()`。

当前流水线执行器通过 Edge 等待数据，其每个节点的运行循环不会根据控制前驱逐请求等待。仅向 NodeWrapper 增加依赖不足以保证顺序。第一版在初始化时拒绝流水线图上的新控制声明和显式顺序，在 `runWithState()` 时拒绝未支持的流水线执行；检查必须覆盖参与执行的嵌套结构。

Condition 的候选节点是互斥分支，仍不接受分支之间的 `set_execution_order()` 或 `add_control_dependency()`。需要分支内多步执行时，将 Graph 或 Loop 作为该分支节点。`choose()` 可以读取 `self.state.branch` 并返回候选节点的注册索引；只有选中的分支执行。

Loop 每轮重新读取 `loops()` 返回的上限，按 `iter_before()` → 循环体 → `iter_after()` 执行。可以在这些方法中直接读写 `self.state`，并根据状态动态缩短或延长循环；零次或负数上限不执行循环体。Python 还可覆盖 `init_start()`、`init_end()`，这两个初始化钩子发生在请求 State 绑定前，不应访问请求状态。循环体保留原有全部注册节点参与执行的语义，控制依赖只补充排序约束。

Graph、Condition、Loop 复用执行互斥、运行标志清理和异常处理。分支、循环体或迭代钩子失败后停止本次执行，保留已有状态修改并解除请求绑定。中断后不再启动下一轮。FixedLoop 的 JSON 保留节点类型、循环次数、循环体以及控制顺序。

## 7. 更新、失败与并发语义

Python 属性赋值立即更新共享字段；list、dict、Tensor 的原地操作直接修改共享对象。节点失败时停止后续调度，已经发生的修改保留，供调用方诊断或丢弃。第一版不提供事务、自动重试和回滚。

串行依赖保证后继在前驱成功返回后访问更新后的主机状态。任务并行中，不同字段可以分别更新；对同一字段或其内部对象的冲突访问必须通过控制依赖排序。第一版不自动检测所有读写冲突，也不引入 reducer。

`state.step_index += 1` 是读、改、写多个步骤，不是原子操作。容器锁不能保护取出的 Tensor，也不能保证模型或 scheduler 内部状态线程安全。

控制完成默认指 `run()` 返回，不等于 GPU 计算完成。跨 stream 的依赖需要节点或数据适配层建立事件等待或必要同步，不能仅凭控制关系读取尚未完成的设备结果。

## 8. JSON 与 Python 绑定

### 8.1 JSON 只保存构图声明

新增可选字段，示例如下：

```json
{
  "control_dependencies": [
    {"from": "prepare", "to": "denoise"}
  ],
  "execution_order": ["encode", "prepare", "denoise", "decode"]
}
```

旧 JSON 缺少字段时默认为空。反序列化先创建节点，再解析声明并校验引用。嵌套图保存自己的声明；不序列化原始指针或派生的 NodeWrapper 关系。

第一版不把运行中的 State、Tensor、模型和 Python dataclass 类型自动写入工作流 JSON。加载工作流后，由调用方构造 State 并传给 `run_with_state()`。因此，现有仅调用 `run()` 的 JSON runner 不会自动获得请求状态；需要集成时由 runner 显式接收 State，不能从 JSON 自动导入任意 Python 类。

### 8.2 Python 导出

- 新增 `nndeploy.dag.State` 与 dataclass 属性适配。
- Node 增加 `get_state()` 和只读属性 `state`。
- Graph 增加 `run_with_state()`、`add_control_dependency()`、`remove_control_dependency()`、`set_execution_order()`。
- 使用同一套绑定转换逻辑处理 State，不复制当前 Node 与 Graph 资源绑定的不一致路径。
- 不修改现有 `run()` 的 Python trampoline 签名；新状态访问方法的 trampoline 和生命周期保持一致。

## 9. 文件级修改清单

以下路径相对于仓库根目录。

| 文件或目录 | 修改内容 |
| --- | --- |
| `framework/include/nndeploy/dag/state.h`，`framework/source/nndeploy/dag/state.cc` | 新增 State 存储 |
| [node.h](../../../../framework/include/nndeploy/dag/node.h)、[node.cc](../../../../framework/source/nndeploy/dag/node.cc) | 新增 `getState()`，沿父级查找状态 |
| [graph.h](../../../../framework/include/nndeploy/dag/graph.h)、[graph.cc](../../../../framework/source/nndeploy/dag/graph.cc) | 控制声明、依赖重建、运行节点筛选、State 绑定、执行互斥、JSON 与能力校验 |
| [condition.cc](../../../../framework/source/nndeploy/dag/condition.cc)、[loop.cc](../../../../framework/source/nndeploy/dag/loop.cc) | 共享执行保护、分支与循环 State、FixedLoop JSON |
| [util.h](../../../../framework/include/nndeploy/dag/util.h)、[util.cc](../../../../framework/source/nndeploy/dag/util.cc) | 复用 NodeWrapper 与拓扑检查；按需补控制依赖的调试输出 |
| [optimizer.cc](../../../../framework/source/nndeploy/dag/optimizer.cc) 及 Graph 优化入口 | 防止未适配的优化改变控制关系和状态副作用 |
| [executor 目录](../../../../framework/source/nndeploy/dag/executor) | 保留现有接口；验证顺序、任务调度，明确流水线限制 |
| [python/src/dag/node.cc](../../../../python/src/dag/node.cc)、[dag.h](../../../../python/src/dag/dag.h) | Node、Graph 新接口绑定与 trampoline |
| `python/src/dag/state.cc`、`python/nndeploy/dag/state.py` | 新增 State 绑定、schema 和属性适配 |
| [python/nndeploy/dag/node.py](../../../../python/nndeploy/dag/node.py)、[graph.py](../../../../python/nndeploy/dag/graph.py)、[__init__.py](../../../../python/nndeploy/dag/__init__.py) | Python 入口与导出 |
| [test/dag](../../../../test/dag) | C++ 调度回归与 Python 状态测试 |
| 对应 CMake 和扩展构建清单 | 按现有源码收集方式接入新增文件 |

Node::run、Edge、Executor 的调用签名保持源码兼容。新增虚函数或成员会影响 C++ ABI，需要重新编译框架、相关插件和 Python 扩展。使用新字段的工作流应在支持该功能的版本中运行，不能假定旧版本会理解或拒绝这些字段。

## 10. 实施顺序与验收

### 第一阶段：控制依赖统一

实现 Graph 控制声明、依赖重建、零边节点筛选和 JSON 支持，复用现有顺序与任务并行执行器。加入初始化后不可改图、环检测和未支持执行方式检查。

### 第二阶段：State 与 Python 属性

实现 C++ State、Graph 绑定、Node 状态查找、执行互斥和 Python dataclass 适配。完成相同底层状态的跨节点访问及异常路径清理。

### 第三阶段：集成验收

将纯 State 节点和已有 Edge 节点放入同一张图验证，检查旧数据流执行行为，并完成以下验收：

| 场景 | 预期 |
| --- | --- |
| 零边节点按 A、B、C 排序 | 严格按顺序执行，后继看到前驱写入 |
| 只有一个显式排序节点 | 节点正常运行，不被筛掉 |
| 数据依赖与控制依赖混用 | 任务并行等待全部所需前驱 |
| 同一对节点存在两种依赖 | 去重；移除一种来源不丢失另一种 |
| 缺失节点、自依赖、环、重复顺序节点 | 明确报错，不能静默忽略 |
| 反复初始化、删除节点、JSON 往返 | 无残留依赖、悬空引用或重复关系 |
| 两次运行传入不同 State | 字段与默认容器不串请求 |
| 同一次运行的多个 Python 节点 | 属性读写访问同一底层状态，schema 保留 |
| C++ 与 Python 读取受支持的同一字段 | 值和类型转换符合约定 |
| dataclass 默认值、继承字段和 default_factory | 不遮蔽底层更新，不共享可变默认值 |
| 未绑定 State、字段拼写错误 | 明确异常 |
| 异常、中断和失败 | 等待本次任务结束并解除绑定，保留已有状态修改 |
| 同图重入和混用 run/run_with_state | 被统一拒绝或串行化，不覆盖当前绑定 |
| 嵌套 Graph、CompositeNode | 状态查找正确；特殊调度未支持配置明确拒绝 |
| Condition / Loop 嵌套、零边分支、动态循环次数与 Python 钩子 | 选择与顺序正确，同一请求共享状态，失败后解绑并可重新执行 |
| 新功能与流水线组合 | 提前报错，不出现伪顺序保证 |
| 旧 DAG、旧 JSON、纯数据流水线 | 原有回归测试通过 |

已新增 `state_control_test`（26 项）、`test_state_control.py`（10 项）和 `test_state_control_flow.py`（8 项，含参数子用例）。2026-11-12 本地验证结果：5 个 CTest 目标共 74 项 C++ 测试通过；State、Condition/Loop、DAG 边界、Edge 所有权及 runner 的 36 项 Python 测试通过。以下命令可用于重新验证；Python 应加载本次构建的扩展和源码包，而非环境中旧版本的已安装包。

```bash
cmake -S . -B build
cmake --build build --target state_control_test pynndeploy dag_regression_test executor_regression_test edge_test graph_test -j 6
ctest --test-dir build -R '^(state_control_test|dag_regression_test|executor_regression_test|edge_test|graph_test)$' --output-on-failure
python -m pytest -q test/dag/test_state_control.py test/dag/test_state_control_flow.py test/dag/test_dag_boundaries.py test/dag/test_edge_ownership.py test/dag/test_graph_runner_pipeline.py
```

Graph、Condition、Loop 的串行和任务并行 State 执行已覆盖。参与执行结构中含流水线时，`run_with_state()` 返回不支持；流水线控制声明也仍在初始化时拒绝。

## 11. 后续扩展

流水线支持需要引入逐请求 State、请求与轮次标识、控制完成信号、跳帧传播、取消清理和背压协调。不能仅复用一个图级 `active_state_`，也不能只给 map 加锁。

基于更新合并的条件路由、并行更新合并、检查点恢复可在现有设计上逐项扩展。它们需要额外的执行语义，不作为第一版默认能力。

## 12. 设计依据

- 当前 NodeWrapper 已有前驱、后继列表；SequentialExecutor 与 ParallelTaskExecutor 可复用这一调度结构。
- 当前 Graph::construct 按数据边筛选运行节点，需要补入显式控制节点。
- 当前 Node::updateInput 对零输入节点返回 Complete。
- 当前图级 resource 接口保留为长期资源入口，与请求 State 区分。
- SGLang 本地 `python/sglang/multimodal_gen/runtime/pipelines_core/schedule_batch.py` 中的 Req 为请求状态参考，latents、timesteps 等随请求传递和更新。
- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) 支持 TypedDict、dataclass 等状态定义，通常通过节点返回更新再合并。本方案借鉴 State 命名与类型化定义，第一版明确采用直接属性修改，不承诺 LangGraph 的更新合并或恢复语义。
