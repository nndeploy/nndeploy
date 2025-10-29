
# 生产环境部署指南

nndeploy提供了从拖拉拽开发到生产环境部署的完整解决方案，确保算法模型能够高效、稳定地在生产环境中运行。

- 开发阶段：使用可视化界面设计和调试工作流。**必要时开发自定义节点**
- 部署阶段：导出JSON配置文件并通过API加载运行

无论是通过可视化界面还是API调用，所有工作流都在统一的高性能C++计算引擎中执行，确保开发和部署环境的一致性。

## 部署阶段

在可视化界面中完成工作流搭建后，可保存为 JSON 文件，然后通过 Python/C++ API 加载执行。

### 自定义输入输出模式

**使用场景**：在代码中动态设置输入数据和自定义输出结果处理

**Python大语言模型示例**：

```python
import nndeploy

# 创建并加载工作流
graph = nndeploy.dag.Graph("")
graph.remove_in_out_node()  # 移除输入输出节点以实现自定义数据流
graph.load_file("path/to/llm_workflow.json")
graph.init()

# 设置输入数据
input_edge = graph.get_input(0)
text = nndeploy.tokenizer.TokenizerText()
text.texts_ = ["<|im_start|>user\n请介绍NBA超级巨星迈克尔·乔丹<|im_end|>\n<|im_start|>assistant\n"]
input_edge.set(text)

# 执行推理
status = graph.run()

# 获取输出结果
output_edge = graph.get_output(0)
result = output_edge.get_graph_output()

# 清理资源
graph.deinit()
```

**C++大语言模型示例**：

```cpp
#include "nndeploy/dag/graph.h"

// 创建并加载工作流
std::shared_ptr<dag::Graph> graph = std::make_shared<dag::Graph>("");
base::Status status = graph->loadFile("path/to/llm_workflow.json");
graph->removeInOutNode();  // 移除输入输出节点
status = graph->init();

// 设置输入数据
dag::Edge* input = graph->getInput(0);
tokenizer::TokenizerText* text = new tokenizer::TokenizerText();
text->texts_ = {
    "<|im_start|>user\n请介绍NBA超级巨星迈克尔·乔丹<|im_end|>\n<|im_start|>assistant\n"
};
input->set(text, false);

// 执行推理
status = graph->run();

// 获取输出结果
dag::Edge* output = graph->getOutput(0);
tokenizer::TokenizerText* result = output->getGraphOutput<tokenizer::TokenizerText>();

// 清理资源
status = graph->deinit();
```

### 完整工作流模式

**使用场景**：工作流包含完整的输入输出处理逻辑，无需额外的数据设置

**Python示例**：

```python
import nndeploy

# 加载并执行完整工作流
graph = nndeploy.dag.Graph("")
graph.load_file("path/to/llm_workflow.json")
graph.init()
status = graph.run()  # 工作流内部处理所有输入输出
graph.deinit()
```

**C++示例**：

```cpp
// 加载并执行完整工作流
std::shared_ptr<dag::Graph> graph = std::make_shared<dag::Graph>("");
base::Status status = graph->loadFile("path/to/llm_workflow.json");
status = graph->init();
status = graph->run();  // 工作流内部处理所有输入输出
status = graph->deinit();
```

---

### 更多示例代码

| 算法类型 | Python示例 | C++示例 |
| -------- | ---------- | ------- |
| **大语言模型** | [Python LLM](https://github.com/nndeploy/nndeploy/blob/main/demo/llm/demo.py) | [C++ LLM](https://github.com/nndeploy/nndeploy/blob/main/demo/llm/demo.cc) |
| **目标检测** | [Python Detection](https://github.com/nndeploy/nndeploy/blob/main/demo/detect/demo.py) | [C++ Detection](https://github.com/nndeploy/nndeploy/blob/main/demo/detect/demo.cc) |


## 调试工作流

完成工作流搭建后，保存为 JSON 文件并通过命令行执行，用于调试工作流

```bash
# Python CLI
nndeploy-run-json --json_file path/to/workflow.json
# C++ CLI
cd path/to/nndeploy
./build/nndeploy_demo_run_json --json_file path/to/workflow.json
```

## 文档

- [Python++ API](https://nndeploy-zh.readthedocs.io/zh-cn/latest/python_api/index.html)
- [Python 自定义节点开发手册](docs/zh_cn/quick_start/plugin_python.md)
- [C++ API](https://nndeploy-zh.readthedocs.io/zh-cn/latest/cpp_api/doxygen.html)
- [C++自定义节点开发手册](docs/zh_cn/quick_start/plugin.md)



